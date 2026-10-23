"""Experiment: is the TabPFN encoder embedding only *approximately* label-invariant?

Hypothesis
----------
Under TabMDA, the frozen TabPFN encoder embedding of a sample ``x_i`` should be
(roughly) invariant to the labeled context ``C`` the sample is conditioned on.
If the invariance held exactly, ``num_context`` and ``context_size`` would not
matter. In practice they do, which suggests different contexts inject different
amounts of noise.

This experiment ranks contexts by how confidently TabPFN itself predicts the
true label of ``x_i`` from that context, i.e.

    P( TabPFN(x_i | C) == y_i )

and asks: does the downstream classifier do better on the Top-K contexts (by this
probability) than on the contexts TabMDA would use by default (same
``num_contexts``, ``context_size=0.5``)?

Protocol
--------
1. For every sample ``x_i``, generate a pool of contexts over the grid
   ``num_context in {5,20,50} x context_size in {0.5,0.7,0.9}``, i.e.
   (5+20+50) * 3 = 225 contexts per sample (leak-free, ``x_i`` excluded).
2. For each context ``C``: compute the embedding ``encode(x_i | C)`` and the
   scalar probability ``P(TabPFN(x_i | C) == y_i)`` (0 if ``y_i`` never appears
   in ``C``). Both come from a single frozen TabPFN v1 backbone with an intact
   decoder (`TabPFNEncoder`), so `encode` and `fit`+`predict_proba` are both valid.
3. Sort each sample's probabilities descending; take Top 5/20/50 embeddings and
   train a downstream `LogisticRegression` on them. Compare against the per-cell
   configs (in particular ``num_contexts=k, context_size=0.5``, which is TabMDA's
   reference configuration).

All embeddings are cached on disk (per sample) so a long run can be resumed.

Examples
--------
    python experiment_label_invariant.py --dataset vehicle \\
        --num_real_samples 100 --repeat_id 0

    # quick smoke test
    python experiment_label_invariant.py --dataset vehicle \\
        --num_real_samples 20 --repeat_id 0 \\
        --grid_num_contexts 5 --grid_context_size 0.5 --top_k 5 --max_samples 2
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from dataset.datasets import get_tabular_dataset, ManualStopError
from tabmda_contrastive.context_sampler import sample_contexts
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


def parse_int_list(s: str):
    return [int(x) for x in s.split(",")]


def parse_float_list(s: str):
    return [float(x) for x in s.split(",")]


def build_grid(num_contexts, context_sizes):
    """Return ordered list of (num_contexts, context_size) cells.

    The user's grid is (5+20+50) x 3 contexts per sample; the total number of
    contexts is ``sum(num_contexts) * len(context_sizes)``.
    """
    cells = []
    for cs in context_sizes:
        for nc in num_contexts:
            cells.append((nc, cs))
    return cells


# ---------------------------------------------------------------- backbone
def make_backbone(base_path: str):
    """A frozen TabPFN v1 encoder with the decoder intact (both encode + proba work)."""
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


def prob_true(enc, X_ctx, y_ctx, X_q, y_i: int) -> float:
    """P(TabPFN(x_i | context) == y_i), 0 if y_i is absent from the context."""
    C_y = np.asarray(y_ctx, dtype=np.int64)
    if len(np.unique(C_y)) < 2:
        return 0.0
    enc.fit(np.asarray(X_ctx, dtype=np.float32), C_y)
    probs = enc.predict_proba(np.asarray(X_q, dtype=np.float32))
    classes = enc.classes_
    if y_i not in classes:
        return 0.0
    return float(probs[0, np.where(classes == y_i)[0][0]])


def compute_sample(enc, model, i, ctx_indices, X_tr, y_tr, y_i, dim):
    """Embedding + P(true label) for all pooled contexts of sample ``i``."""
    total = len(ctx_indices)
    emb = np.zeros((total, dim), dtype=np.float32)
    prob = np.zeros(total, dtype=np.float32)
    x_q = X_tr[i:i + 1]
    x_q_np = X_tr[i:i + 1].numpy().astype(np.float32)
    for c in range(total):
        idx = ctx_indices[c].astype(np.int64)
        X_sub = X_tr[idx]
        y_sub = y_tr[idx]
        emb[c] = encode_query(enc, model, x_q, X_sub, y_sub)
        prob[c] = prob_true(enc, X_sub.numpy().astype(np.float32),
                            y_sub.numpy().astype(np.int64), x_q_np, y_i)
    return emb, prob


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


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="vehicle")
    p.add_argument("--num_real_samples", type=int, default=100)
    p.add_argument("--repeat_id", type=int, default=0)
    p.add_argument("--grid_num_contexts", type=parse_int_list, default=[5, 20, 50])
    p.add_argument("--grid_context_size", type=parse_float_list, default=[0.5, 0.7, 0.9])
    p.add_argument("--top_k", type=parse_int_list, default=[5, 20, 50])
    p.add_argument("--tabmda_context_size", type=float, default=0.5)
    p.add_argument("--self_context", type=str, default="random", choices=["exclude", "random"],
                   help="'random' matches TabMDA default (shared stratified context, "
                        "query row may appear); 'exclude' is per-query leave-one-out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None, help="cap #train samples to encode (debug)")
    p.add_argument("--max_test", type=int, default=None, help="cap #test rows to encode (debug)")
    p.add_argument("--smoke", action="store_true", help="tiny deterministic smoke config")
    p.add_argument("--cache_dir", default="results_label_invariant/cache")
    p.add_argument("--out_dir", default="results_label_invariant")
    args = p.parse_args()
    if args.smoke:
        args.num_real_samples = 20
        args.grid_num_contexts = [5]
        args.grid_context_size = [0.5]
        args.top_k = [5]
        args.max_samples = 8
        args.max_test = 2

    repo_root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(repo_root, args.out_dir)
    base_cache_dir = os.path.join(repo_root, args.cache_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(base_cache_dir, exist_ok=True)

    # Tag the cache + output by the full sampling config so different runs
    # (self_context mode, grid, seed, sample size) never share stale embeddings.
    tag = (
        f"sc_{args.self_context}_seed{args.seed}_n{args.num_real_samples}_r{args.repeat_id}"
        f"_nc{'-'.join(map(str, args.grid_num_contexts))}"
        f"_cs{'-'.join(str(c).replace('.', 'p') for c in args.grid_context_size)}"
    )
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
    F = X_train.shape[1]

    cells = build_grid(args.grid_num_contexts, args.grid_context_size)
    total_ctx = sum(nc for nc, _ in cells)
    print(f"[setup] dataset={args.dataset} train={N} test={X_test.shape[0]} "
          f"classes={num_classes} features={F}")
    print(f"[setup] grid cells={cells} total contexts/sample={total_ctx}")

    # ---------- context index structure ----------
    ctx_indices_all = None
    pos_meta = []  # (cell_idx, num_contexts, context_size, k)
    for ci, (nc, cs) in enumerate(cells):
        ctx = sample_contexts(X_train, y_train, num_contexts=nc, context_size=cs,
                              seed=args.seed, self_context=args.self_context)
        if ctx_indices_all is None:
            n_queries = len(ctx.indices)
            ctx_indices_all = [[] for _ in range(n_queries)]
        for k in range(nc):
            pos_meta.append((ci, nc, cs, k))
            for i in range(n_queries):
                ctx_indices_all[i].append(ctx.indices[i][k].astype(np.int64))
    # positions belonging to each cell
    cell_pos = [[] for _ in cells]
    for p, (ci, _, _, _) in enumerate(pos_meta):
        cell_pos[ci].append(p)

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

    # ---------- phase 1: per-sample embeddings + probabilities ----------
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    emb_all = np.zeros((N, total_ctx, TABPFN_HIDDEN_DIM), dtype=np.float32)
    prob_all = np.zeros((N, total_ctx), dtype=np.float32)

    cap = args.max_samples if args.max_samples else N
    cap = min(cap, N)

    # `--max_samples` must not simply take the first `cap` rows: the dataset is
    # class-contiguous, so a small cap can collapse to a single class and break
    # the downstream LogisticRegression. Select a label-stratified subset instead
    # (indices are original row ids, keeping the per-sample cache valid).
    if cap < N:
        from sklearn.model_selection import train_test_split
        sample_ids, _ = train_test_split(
            np.arange(N), train_size=cap,
            random_state=args.seed, stratify=y_train,
        )
        sample_ids = np.sort(np.asarray(sample_ids, dtype=np.int64))
    else:
        sample_ids = np.arange(N, dtype=np.int64)

    y_sel = y_train[sample_ids]
    for pos, i in enumerate(tqdm(sample_ids.tolist(), desc="samples")):
        sp = os.path.join(cache_dir, f"sample_{i}.npz")
        if os.path.exists(sp):
            z = np.load(sp)
            emb, prob = z["emb"], z["prob"]
        else:
            emb, prob = compute_sample(
                enc, model, i, ctx_indices_all[i], X_tr, y_tr,
                int(y_train[i]), TABPFN_HIDDEN_DIM,
            )
            np.savez(sp, emb=emb, prob=prob)
        emb_all[pos] = emb
        prob_all[pos] = prob
    if cap < N:
        print(f"[warn] truncated to {cap}/{N} samples (--max_samples, stratified). "
              f"Selected classes: {np.unique(y_sel).tolist()}. Downstream will use {cap} samples.")
        emb_all = emb_all[:cap]
        prob_all = prob_all[:cap]
        y_train = y_sel
        N = cap

    # ---------- phase 2: downstream evaluation ----------
    results = {}

    # (a) per-cell: train LogReg on EVERY context of a single (nc, cs) config.
    per_cell = {}
    for ci, (nc, cs) in enumerate(cells):
        pos = cell_pos[ci]
        Xsel = emb_all[:, pos, :].reshape(N * nc, TABPFN_HIDDEN_DIM)
        ysel = np.repeat(y_train, nc)
        bacc = fit_and_eval(Xsel, ysel, test_emb, y_test_use)
        mean_prob = float(prob_all[:, pos].mean())
        per_cell[f"nc{nc}_cs{cs}"] = {
            "num_contexts": nc,
            "context_size": cs,
            "balanced_accuracy": bacc,
            "mean_prob_true": mean_prob,
            "n_train_emb": int(N * nc),
        }
    results["per_cell"] = per_cell

    # (b) top-k by probability.
    top_k_results = {}
    for k in args.top_k:
        assert k <= total_ctx, f"top_k={k} exceeds pooled contexts {total_ctx}"
        top_idx = np.argsort(-prob_all, axis=1)[:, :k]
        Xsel = emb_all[np.arange(N)[:, None], top_idx, :].reshape(N * k, TABPFN_HIDDEN_DIM)
        ysel = np.repeat(y_train, k)
        bacc = fit_and_eval(Xsel, ysel, test_emb, y_test_use)
        mean_prob = float(prob_all[np.arange(N)[:, None], top_idx].mean())
        top_k_results[str(k)] = {
            "k": k,
            "balanced_accuracy": bacc,
            "mean_prob_true": mean_prob,
            "n_train_emb": int(N * k),
        }
    results["top_k_by_probability"] = top_k_results

    # (c) explicit TabMDA reference cells for the comparison.
    tabmda_ref = {}
    for k in args.top_k:
        key = f"nc{k}_cs{args.tabmda_context_size}"
        if key in per_cell:
            tabmda_ref[str(k)] = per_cell[key]
    results["tabmda_reference"] = tabmda_ref

    results["meta"] = {
        "dataset": args.dataset,
        "num_real_samples": args.num_real_samples,
        "repeat_id": args.repeat_id,
        "seed": args.seed,
        "self_context": args.self_context,
        "num_classes": int(num_classes),
        "train_size": int(N),
        "test_size": int(X_test.shape[0]),
        "grid": [list(c) for c in cells],
    }

    out_path = os.path.join(out_dir,
                            f"label_invariant_{args.dataset}_r{args.repeat_id}_sc_{args.self_context}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---------- report ----------
    print("\n==== per-cell downstream (LogReg) ====")
    for k, v in per_cell.items():
        print(f"  {k:12s}  bacc={v['balanced_accuracy']:.4f}  mean_prob_true={v['mean_prob_true']:.4f}")
    print("\n==== top-k by probability ====")
    for k, v in top_k_results.items():
        print(f"  top-{k:3s}  bacc={v['balanced_accuracy']:.4f}  mean_prob_true={v['mean_prob_true']:.4f}")
    print(f"\n[out] wrote {out_path}")


if __name__ == "__main__":
    main()