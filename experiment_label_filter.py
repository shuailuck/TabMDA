"""Experiment: filter contexts by label-invariance (TabPFN argmax == y_i).

Motivation
----------
TabMDA adds ``num_contexts`` label-stratified contexts per training sample and
uses them all. These contexts differ in how reliably the frozen TabPFN encoder
recovers the true label ``y_i`` from them. If the embedding is only
*approximately* label-invariant, then "bad" contexts (those where TabPFN itself
fails to predict ``y_i``) inject noise.

This experiment takes ONE fixed cell ``(context_size=cs, num_contexts=K)``
(the "best fixed cell") and compares, under an identical frozen-TabPFN +
LogisticRegression downstream protocol, two arms that BOTH have exactly ``K``
contexts per training sample:

  * **unfiltered**  - the first ``K`` contexts TabMDA would use (no constraint).
  * **filtered**    - keep *sampling* contexts (leak-free, label-stratified,
                      deterministic) until exactly ``K`` contexts satisfy the
                      label-invariance constraint ``argmax P(y | x_i, C) == y_i``,
                      then use those ``K``.

Because the filtered arm keeps sampling until it has ``K`` valid contexts, it is
guaranteed ``num_contexts == K`` for every query it retains — the same per-query
count as the unfiltered arm (and as TabMDA). A query that still fails to reach
``K`` valid contexts within ``max_attempts`` is dropped (reported), so the two
arms never differ in context count, only in *which* contexts are kept.

Note on TabMDA consistency
--------------------------
The real TabMDA pipeline shares ONE stratified context per context *index*
(reused across queries) and may place a query in its own context. That structure
cannot be combined with a per-query correctness filter (the filter depends on
``x_i``). This script therefore builds per-query contexts for BOTH arms so the
comparison is controlled; the real shared-context TabMDA is run separately by
``run_label_filter.sh`` via ``train.py`` as the external anchor.

Examples
--------
    python experiment_label_filter.py --dataset vehicle \\
        --num_real_samples 100 --repeat_id 0 --context_size 0.5 --num_contexts 5

    # quick smoke test
    python experiment_label_filter.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from dataset.datasets import get_tabular_dataset, ManualStopError
from tabmda_contrastive.context_sampler import _stratified_select, resolve_context_size
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


TABPFN_HIDDEN_DIM = 512


# ---------------------------------------------------------------- data utils
def relabel_shared(y_train, y_test):
    """Map labels to contiguous [0, C) using the train classes."""
    classes = np.unique(np.asarray(y_train))
    mapping = {int(c): i for i, c in enumerate(classes)}
    y_tr = np.asarray([mapping[int(c)] for c in y_train], dtype=np.int64)
    try:
        y_te = np.asarray([mapping[int(c)] for c in y_test], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"Test label {e} not present in the training set.") from e
    return y_tr, y_te, len(classes)


# ---------------------------------------------------------------- context gen
def make_context_generator(X_train, y_train, q, context_size, seed):
    """Return a function ``gen(k)`` yielding the k-th leak-free context's row
    indices for query ``q``.

    Reproduces :func:`tabmda_contrastive.context_sampler.sample_contexts` in
    ``self_context="exclude"`` mode exactly, but lazily (one attempt at a time),
    so the filtered arm can keep sampling until ``K`` valid contexts are found.
    """
    n = X_train.shape[0]
    all_indices = np.arange(n)
    pool = all_indices[all_indices != q]          # exclude the query row itself
    pool_y = y_train[pool]
    pool_nclasses = len(np.unique(pool_y))
    target = resolve_context_size(context_size, n - 1)

    def gen(k: int) -> np.ndarray:
        ctx_seed = seed * 1000003 + q * 7919 + k * 131071
        train_size = min(target, len(pool) - pool_nclasses)
        if train_size <= 0:
            train_size = target
        sel = _stratified_select(len(pool), pool_y, train_size, ctx_seed)
        return pool[sel]

    return gen


# ---------------------------------------------------------------- backbone
def make_backbone(base_path: str):
    """A frozen TabPFN v1 encoder with the decoder intact (both encode + predict work)."""
    from TabPFN.tabpfn.scripts.transformer_prediction_interface import TabPFNEncoder

    enc = TabPFNEncoder(
        device="cpu",
        base_path=base_path,
        N_ensemble_configurations=1,   # deterministic
        only_inference=True,
        no_grad=True,
        no_preprocess_mode=True,
    )
    for p in enc.model[2].parameters():
        p.requires_grad = False
    return enc


def encode_query(enc, model, X_q, X_ctx, y_ctx):
    """Encode a single query row X_q ([1,F]) against context [B,F] -> [512]."""
    with torch.no_grad():
        e = enc.encode(X_to_encode=X_q, X_context=X_ctx, y_context=y_ctx, model=model)
    return e.squeeze(1)[0].detach().cpu().numpy().astype(np.float32)


def predict_label(enc, X_ctx, y_ctx, X_q):
    """Argmax label TabPFN predicts for X_q given context (X_ctx, y_ctx).

    Returns the predicted class, or None if the context holds fewer than two
    classes (TabPFN cannot be fit).
    """
    C_y = np.asarray(y_ctx, dtype=np.int64)
    if len(np.unique(C_y)) < 2:
        return None
    enc.fit(np.asarray(X_ctx, dtype=np.float32), C_y)
    preds = enc.predict(np.asarray(X_q, dtype=np.float32))
    return int(preds[0])


def compute_test_embeddings(enc, model, X_train, y_train, X_test, dim):
    """Encode each test row against the FULL training set (TabMDA test protocol)."""
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    emb = np.zeros((X_test.shape[0], dim), dtype=np.float32)
    for j in range(X_test.shape[0]):
        emb[j] = encode_query(enc, model, X_te[j:j + 1], X_tr, y_tr)
    return emb


def fit_and_eval(X_train_sel, y_train_sel, X_test_emb, y_test):
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X_train_sel, y_train_sel)
    return float(balanced_accuracy_score(y_test, clf.predict(X_test_emb)))


def compute_query(enc, model, i, X_tr, y_tr, y_i, K, dim, context_size, seed, max_attempts):
    """Build both arms for query ``i``, each with exactly ``K`` contexts.

    Returns:
        unf_emb  [K, dim]  first K contexts (no constraint)
        filt_emb [K, dim]  first K contexts satisfying TabPFN(x_i|C)==y_i
                           (zero-filled if not enough found)
        ok       bool      True iff ``K`` valid contexts were found
        n_attempts int     how many contexts were sampled for the filter
    """
    gen = make_context_generator(X_tr.numpy(), y_tr.numpy(), i, context_size, seed)
    x_q = X_tr[i:i + 1]
    x_q_np = X_tr[i:i + 1].numpy().astype(np.float32)
    X_np = X_tr.numpy().astype(np.float32)
    y_np = y_tr.numpy().astype(np.int64)

    # unfiltered selection: attempts 0 .. K-1
    unf_sel = list(range(K))

    # filtered selection: keep sampling until K valid, up to max_attempts
    filt_sel = []
    attempts = 0
    while len(filt_sel) < K and attempts < max_attempts:
        idx = gen(attempts)
        pred = predict_label(enc, X_np[idx], y_np[idx], x_q_np)
        if pred == y_i:
            filt_sel.append(attempts)
        attempts += 1
    ok = (len(filt_sel) == K)

    # encode only the union of the two selections
    need = sorted(set(unf_sel) | set(filt_sel))
    emb_map = {}
    for k in need:
        idx = gen(k)
        emb_map[k] = encode_query(enc, model, x_q, X_tr[idx], y_tr[idx])

    unf_emb = np.stack([emb_map[k] for k in unf_sel]).astype(np.float32)
    filt_emb = np.zeros((K, dim), dtype=np.float32)
    if ok:
        filt_emb = np.stack([emb_map[k] for k in filt_sel]).astype(np.float32)

    return unf_emb, filt_emb, ok, attempts


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="vehicle")
    p.add_argument("--num_real_samples", type=int, default=100)
    p.add_argument("--repeat_id", type=int, default=0)
    p.add_argument("--context_size", type=float, default=0.5, help="fixed cell cs (proportion)")
    p.add_argument("--num_contexts", type=int, default=5, help="target K contexts per query")
    p.add_argument("--max_attempts", type=int, default=100,
                   help="hard cap on contexts sampled per query while filtering")
    p.add_argument("--self_context", type=str, default="exclude",
                   choices=["exclude", "random"],
                   help="'exclude' = leak-free per-query contexts (required for per-query filter)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None, help="cap #train samples to encode (debug)")
    p.add_argument("--max_test", type=int, default=None, help="cap #test rows to encode (debug)")
    p.add_argument("--smoke", action="store_true", help="tiny deterministic smoke config")
    p.add_argument("--cache_dir", default="results_label_filter/cache")
    p.add_argument("--out_dir", default="results_label_filter")
    args = p.parse_args()

    if args.self_context != "exclude":
        raise ValueError("the per-query label filter requires self_context='exclude'.")

    if args.smoke:
        args.num_real_samples = 20
        args.context_size = 0.5
        args.num_contexts = 5
        args.max_samples = 8
        args.max_test = 2

    K = args.num_contexts

    repo_root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(repo_root, args.out_dir)
    base_cache_dir = os.path.join(repo_root, args.cache_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(base_cache_dir, exist_ok=True)

    tag = (f"sc_{args.self_context}_seed{args.seed}_d{args.dataset}_n{args.num_real_samples}"
           f"_r{args.repeat_id}_cs{str(args.context_size).replace('.', 'p')}"
           f"_nc{K}_maxa{args.max_attempts}")
    cache_dir = os.path.join(base_cache_dir, tag)
    os.makedirs(cache_dir, exist_ok=True)

    # ---------- data ----------
    try:
        dataset = get_tabular_dataset(
            dataset=args.dataset,
            num_real_samples=args.num_real_samples,
            repeat_id=args.repeat_id,
        )
    except ManualStopError as e:
        print("[data] manual stop:", e)
        return

    X_train = np.asarray(dataset["X_train"], dtype=np.float32)
    y_train = np.asarray(dataset["y_train"], dtype=np.int64)
    X_test = np.asarray(dataset["X_test"], dtype=np.float32)
    y_test = np.asarray(dataset["y_test"], dtype=np.int64)

    y_train, y_test, num_classes = relabel_shared(y_train, y_test)
    N = X_train.shape[0]

    print(f"[setup] dataset={args.dataset} train={N} test={X_test.shape[0]} classes={num_classes}")
    print(f"[setup] cell=(cs={args.context_size}, nc={K}) filter until {K} valid/query "
          f"max_attempts={args.max_attempts}")

    # ---------- backbone ----------
    tabpfn_base = os.path.join(repo_root, "TabPFN", "tabpfn")
    enc = make_backbone(tabpfn_base)
    model = enc.model[2]

    # ---------- test embeddings (full train context), cached ----------
    n_test_use = X_test.shape[0] if args.max_test is None else min(args.max_test, X_test.shape[0])
    test_emb_path = os.path.join(cache_dir, f"test_emb_{n_test_use}.npy")
    if os.path.exists(test_emb_path):
        test_emb = np.load(test_emb_path)
        test_emb = None if test_emb.shape[0] != n_test_use else test_emb
    else:
        test_emb = None
    if test_emb is None:
        print(f"[test-emb] encoding {n_test_use} test rows against full train context ...")
        test_emb = compute_test_embeddings(enc, model, X_train, y_train,
                                           X_test[:n_test_use], TABPFN_HIDDEN_DIM)
        np.save(test_emb_path, test_emb)
    y_test_use = y_test[:n_test_use]

    # ---------- phase 1: per-query arms ----------
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)

    cap = args.max_samples if args.max_samples else N
    cap = min(cap, N)
    if cap < N:
        from sklearn.model_selection import train_test_split
        sample_ids, _ = train_test_split(
            np.arange(N), train_size=cap, random_state=args.seed, stratify=y_train,
        )
        sample_ids = np.sort(np.asarray(sample_ids, dtype=np.int64))
    else:
        sample_ids = np.arange(N, dtype=np.int64)

    y_sel = y_train[sample_ids]

    print(f"[phase1] filtering {cap} queries, K={K} ...")
    for pos, i in enumerate(tqdm(sample_ids.tolist(), desc="queries")):
        qp = os.path.join(cache_dir, f"q_{i}.npz")
        if os.path.exists(qp):
            z = np.load(qp)
            unf_emb, filt_emb = z["unf_emb"], z["filt_emb"]
            ok, n_attempts = bool(z["ok"]), int(z["n_attempts"])
        else:
            unf_emb, filt_emb, ok, n_attempts = compute_query(
                enc, model, i, X_tr, y_tr, int(y_train[i]), K, TABPFN_HIDDEN_DIM,
                args.context_size, args.seed, args.max_attempts,
            )
            np.savez(qp, unf_emb=unf_emb, filt_emb=filt_emb, ok=ok, n_attempts=n_attempts)

    if cap < N:
        print(f"[warn] stratified subset {cap}/{N} samples used. classes={np.unique(y_sel).tolist()}")
    y_train = y_sel
    N = cap

    # ---------- phase 2: downstream evaluation ----------
    unf_rows, unf_yrows = [], []
    filt_rows, filt_yrows = [], []
    n_dropped = 0
    attempts_sum = 0
    n_kept = 0

    for pos, i in enumerate(sample_ids.tolist()):
        z = np.load(os.path.join(cache_dir, f"q_{i}.npz"))
        unf_emb, filt_emb = z["unf_emb"], z["filt_emb"]
        ok, n_attempts = bool(z["ok"]), int(z["n_attempts"])
        y_i = int(y_train[pos])
        for k in range(K):
            unf_rows.append(unf_emb[k])
            unf_yrows.append(y_i)
        if ok:
            for k in range(K):
                filt_rows.append(filt_emb[k])
                filt_yrows.append(y_i)
            attempts_sum += n_attempts
            n_kept += 1
        else:
            n_dropped += 1

    unf_bacc = fit_and_eval(np.asarray(unf_rows, dtype=np.float32),
                            np.asarray(unf_yrows, dtype=np.int64), test_emb, y_test_use)

    filt_bacc = None
    if filt_rows:
        filt_bacc = fit_and_eval(np.asarray(filt_rows, dtype=np.float32),
                                 np.asarray(filt_yrows, dtype=np.int64), test_emb, y_test_use)

    mean_attempts = (attempts_sum / n_kept) if n_kept else float("nan")
    # acceptance ~= K valid kept per (attempts) sampled
    total_kept = n_kept * K
    accept = (total_kept / attempts_sum) if attempts_sum else float("nan")

    results = {
        "config": {
            "dataset": args.dataset,
            "num_real_samples": args.num_real_samples,
            "repeat_id": args.repeat_id,
            "seed": args.seed,
            "self_context": args.self_context,
            "context_size": args.context_size,
            "num_contexts": K,
            "max_attempts": args.max_attempts,
            "num_classes": int(num_classes),
            "train_size": int(N),
            "test_size": int(X_test.shape[0]),
        },
        "arms": {
            "unfiltered": {
                "balanced_accuracy": unf_bacc,
                "n_train_emb": int(len(unf_rows)),
                "num_contexts": K,
                "n_queries": int(N),
            },
            "filtered": {
                "balanced_accuracy": float(filt_bacc) if filt_bacc is not None else float("nan"),
                "n_train_emb": int(len(filt_rows)),
                "num_contexts": K,
                "n_queries_kept": int(n_kept),
                "n_queries_dropped": int(n_dropped),
                "mean_attempts_per_query": float(mean_attempts),
                "acceptance_rate": float(accept),
            },
        },
    }

    cs_str = str(args.context_size).replace('.', 'p')
    out_path = os.path.join(out_dir, f"label_filter_{args.dataset}_r{args.repeat_id}_cs{cs_str}_nc{K}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---------- report ----------
    u = results["arms"]["unfiltered"]
    fl = results["arms"]["filtered"]
    print("\n==== label-filter experiment (both arms have nc=K per query) ====")
    print(f"  unfiltered  bacc={u['balanced_accuracy']:.4f}  (n_emb={u['n_train_emb']}, nc={u['num_contexts']})")
    print(f"  filtered    bacc={fl['balanced_accuracy']:.4f}  (n_emb={fl['n_train_emb']}, nc={fl['num_contexts']}, "
          f"kept={fl['n_queries_kept']}/{N}, dropped={fl['n_queries_dropped']}, "
          f"mean_attempts={fl['mean_attempts_per_query']:.2f}, accept={fl['acceptance_rate']:.3f})")
    print(f"\n[out] wrote {out_path}")


if __name__ == "__main__":
    main()