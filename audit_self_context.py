#!/usr/bin/env python
"""
Audit whether TabMDA train-time embeddings leak the sample's own label through
self-context inclusion.

Encoding path traced from TabMDA/model.py + tabpfn/scripts/transformer_prediction_interface.py:
  - train call uses x = X_train, x_context = [X_train], y_context = [y_train]
  - context_subsetting() samples context indices from the FULL X_train, never
    excluding the query row
  - TabPFNEncoder.encode() concats [X_context; X_to_encode] and feeds the
    context labels (only) into the encoder; the query attends to all context
    tokens (efficient_eval_masking)

So if a train sample x_i lands in its own context, its embedding is conditioned
on (x_i, y_i) — a labelled duplicate of itself. Test samples can never be
self-included. This measures how much that mismatch matters.

Arms:
  original   : replicate train.py (context may include the query's own (x_i, y_i))
  remove_self: SAME shared-context sampling as original (deterministic per-context
               seed), but when x_j is in its own context, drop x_j from it. Isolates
               the pure self-inclusion effect without changing the sampling.
  loo        : for each train sample x_j, sample the context from X_train \\ {x_j}
               (confounded: removes self AND re-samples per query independently)

plus a direct, controlled measure of self-inclusion shift: for a fixed context C
(not containing x_j), compare embeddings of x_j with C vs C U {x_j}.

Usage (run inside the `tabmda` conda env):
  python audit_self_context.py --dataset texture --num_real_samples 50 \
      --repeat_id 0 --context_size 0.5 --num_contexts 20
"""
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.linear_model import LogisticRegression

from TabMDA.constructor import construct_tabmda_model
from dataset.datasets import get_tabular_dataset
from utils import GLOBAL_SEED, set_seed, get_available_device, to_numpy

TABPFN_HIDDEN_DIM = 512


def encode_val_test(model, X, y, X_train, y_train, device):
    """Encode val/test with full train context (num_contexts=1) — shared by both arms."""
    X_enc, y_enc = model.encode_batch(
        batch={"x": X, "y": y, "x_context": [X_train], "y_context": [y_train]},
        context_subsetting_params={"num_contexts": 1, "context_size": 1},
        smote_params=None,
    )
    return to_numpy(X_enc), to_numpy(y_enc)


def encode_train_original(model, X, y, num_contexts, context_size, device):
    """Exact replica of train.py's encoding (self-inclusion possible)."""
    X_enc, y_enc = model.encode_batch(
        batch={"x": X, "y": y, "x_context": [X], "y_context": [y]},
        context_subsetting_params={"num_contexts": num_contexts, "context_size": context_size},
        smote_params=None,
    )
    return to_numpy(X_enc), to_numpy(y_enc)


def encode_train_loo(model, X, y, num_contexts, context_size, device, seed=42):
    """Encode each train sample x_j with contexts sampled from X_train \\ {x_j}."""
    N = X.shape[0]
    n_classes = len(torch.unique(y))
    X_np = to_numpy(X)
    y_np = to_numpy(y)

    out = torch.zeros(N, num_contexts, TABPFN_HIDDEN_DIM, device=device)
    for j in range(N):
        keep = np.arange(N) != j
        X_pool = X_np[keep]
        y_pool = y_np[keep]
        pool_idx = np.arange(len(X_pool))
        for i in range(num_contexts):
            ctx_int = int(context_size * N)  # same cardinality as the original arm
            ctx_int = min(ctx_int, len(X_pool))
            if ctx_int >= len(X_pool):
                idx_tr = pool_idx
            else:
                idx_tr = train_test_split(
                    pool_idx, pool_idx,
                    train_size=min(ctx_int, len(X_pool) - n_classes),
                    random_state=seed + i,
                    stratify=y_pool,
                )[0]
            X_sub = torch.tensor(X_pool[idx_tr], dtype=torch.float32, device=device)
            y_sub = torch.tensor(y_pool[idx_tr], dtype=torch.long, device=device)
            enc = model.encode(X_to_encode=X[j:j + 1], X_context=X_sub, y_context=y_sub)
            out[j, i, :] = enc[:, 0, :]

    y_rep = torch.repeat_interleave(y, num_contexts, dim=0)
    return to_numpy(out.reshape(-1, TABPFN_HIDDEN_DIM)), to_numpy(y_rep)


def encode_train_remove_self(model, X, y, num_contexts, context_size, device, seed=42):
    """Same shared-context sampling as `original`, but drop x_j from its own context.

    For each context_idx i, sample the shared context index set EXACTLY as
    train.py's context_subsetting() does (stratified train_test_split with
    random_state=seed+i). Then encode each query x_j with that SAME context,
    minus {j} if j happens to be in it. The only difference from `original` is
    the removal of self-inclusion: every non-self query keeps the identical
    context, so per-query resampling (the loo confound) is NOT introduced.

    Per-query encoding from a shared context is equivalent to batched encoding
    here because under efficient_eval_masking queries attend only to context
    tokens (never to other queries), and preprocessing (normalization, feature
    selection) uses only the context positions.
    """
    N = X.shape[0]
    n_classes = len(torch.unique(y))
    X_np = to_numpy(X)
    y_np = to_numpy(y)

    ctx_int = int(context_size * N)

    indices = np.arange(N)
    shared_contexts = []
    for i in range(num_contexts):
        if ctx_int == N:
            idx = indices.copy()
        else:
            idx = train_test_split(
                indices, indices,
                train_size=min(ctx_int, N - n_classes),
                random_state=seed + i,
                stratify=y_np,
            )[0]
        shared_contexts.append(idx)

    out = torch.zeros(N, num_contexts, TABPFN_HIDDEN_DIM, device=device)
    for j in range(N):
        x_j = X[j:j + 1]
        for i in range(num_contexts):
            idx = shared_contexts[i]
            keep = idx[idx != j]
            X_sub = torch.tensor(X_np[keep], dtype=torch.float32, device=device)
            y_sub = torch.tensor(y_np[keep], dtype=torch.long, device=device)
            enc = model.encode(X_to_encode=x_j, X_context=X_sub, y_context=y_sub)
            out[j, i, :] = enc[:, 0, :]

    y_rep = torch.repeat_interleave(y, num_contexts, dim=0)
    return to_numpy(out.reshape(-1, TABPFN_HIDDEN_DIM)), to_numpy(y_rep)


def measure_self_inclusion_shift(model, X, y, device, n_samples=None, seed=42):
    """For a fixed context C (x_j not in C), compare embed(x_j | C) vs embed(x_j | C U {x_j})."""
    N = X.shape[0]
    n_classes = len(torch.unique(y))
    X_np = to_numpy(X)
    y_np = to_numpy(y)
    if n_samples is None:
        n_samples = N
    sample_js = np.arange(N)[:n_samples]

    l2_diffs, cos_sims, rel_diffs = [], [], []
    for j in sample_js:
        pool_idx = np.arange(N)[np.arange(N) != j]
        ctx_int = min(int(0.5 * N), len(pool_idx) - n_classes)
        idx_tr = train_test_split(
            pool_idx, pool_idx, train_size=ctx_int,
            random_state=seed, stratify=y_np[pool_idx],
        )[0]
        C = torch.tensor(X_np[idx_tr], dtype=torch.float32, device=device)
        y_C = torch.tensor(y_np[idx_tr], dtype=torch.long, device=device)
        x_j = X[j:j + 1]
        y_j = y[j:j + 1]

        z_excl = model.encode(X_to_encode=x_j, X_context=C, y_context=y_C)[:, 0, :]       # (1,512)
        z_incl = model.encode(X_to_encode=x_j,
                              X_context=torch.cat([C, x_j], dim=0),
                              y_context=torch.cat([y_C, y_j], dim=0))[:, 0, :]            # (1,512)

        d = torch.norm(z_incl - z_excl, dim=-1)
        denom = torch.norm(z_excl, dim=-1)
        l2_diffs.append(d.item())
        rel_diffs.append((d / denom).item())
        cos_sims.append(torch.nn.functional.cosine_similarity(z_incl, z_excl, dim=-1).item())

    return {
        "l2_mean": float(np.mean(l2_diffs)),
        "rel_mean": float(np.mean(rel_diffs)),
        "cos_mean": float(np.mean(cos_sims)),
    }


def eval_classifier(X_train, y_train, X_val, y_val, X_test, y_test):
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X_train, y_train)
    return {
        "train_bal": balanced_accuracy_score(y_train, clf.predict(X_train)),
        "train_acc": accuracy_score(y_train, clf.predict(X_train)),
        "val_bal": balanced_accuracy_score(y_val, clf.predict(X_val)),
        "test_bal": balanced_accuracy_score(y_test, clf.predict(X_test)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="texture")
    ap.add_argument("--num_real_samples", type=int, default=50)
    ap.add_argument("--repeat_id", type=int, default=0)
    ap.add_argument("--context_size", type=float, default=0.5)
    ap.add_argument("--num_contexts", type=int, default=20)
    ap.add_argument("--shift_samples", type=int, default=None)
    args = ap.parse_args()

    set_seed(GLOBAL_SEED)
    device = get_available_device()

    ds = get_tabular_dataset(args.dataset, args.num_real_samples, args.repeat_id)
    X_train = torch.tensor(ds["X_train"], dtype=torch.float32).to(device)
    y_train = torch.tensor(ds["y_train"], dtype=torch.long).to(device)
    X_val = torch.tensor(ds["X_val"], dtype=torch.float32).to(device)
    y_val = torch.tensor(ds["y_val"], dtype=torch.long).to(device)
    X_test = torch.tensor(ds["X_test"], dtype=torch.float32).to(device)
    y_test = torch.tensor(ds["y_test"], dtype=torch.long).to(device)

    print(f"\n[audit] {args.dataset} n={args.num_real_samples} repeat={args.repeat_id} "
          f"cs={args.context_size} nc={args.num_contexts}")
    print(f"[audit] N_train={X_train.shape[0]} N_val={X_val.shape[0]} N_test={X_test.shape[0]} "
          f"n_classes={len(torch.unique(y_train))} device={device}")

    model = construct_tabmda_model(classifier=None, freeze_encoder=True, device=device)

    X_val_enc, y_val_enc = encode_val_test(model, X_val, y_val, X_train, y_train, device)
    X_test_enc, y_test_enc = encode_val_test(model, X_test, y_test, X_train, y_train, device)

    print("\n=== encoding train (original) ===")
    X_orig, y_orig = encode_train_original(model, X_train, y_train, args.num_contexts, args.context_size, device)
    print(f"[audit] original train encoded shape: {X_orig.shape}")

    print("=== encoding train (remove-self) ===")
    X_rs, y_rs = encode_train_remove_self(model, X_train, y_train, args.num_contexts, args.context_size, device)
    print(f"[audit] remove_self train encoded shape: {X_rs.shape}")

    print("=== encoding train (leave-one-out) ===")
    X_loo, y_loo = encode_train_loo(model, X_train, y_train, args.num_contexts, args.context_size, device)
    print(f"[audit] loo train encoded shape: {X_loo.shape}")

    print("\n=== direct self-inclusion shift (fixed context C, +- x_j) ===")
    shift = measure_self_inclusion_shift(model, X_train, y_train, device, n_samples=args.shift_samples)
    print(f"[audit] self-inclusion shift: L2={shift['l2_mean']:.4f}  "
          f"rel={shift['rel_mean']:.4f}  cos={shift['cos_mean']:.4f}")

    print("\n=== downstream LogReg (balanced accuracy) ===")
    res_orig = eval_classifier(X_orig, y_orig, X_val_enc, y_val_enc, X_test_enc, y_test_enc)
    res_rs = eval_classifier(X_rs, y_rs, X_val_enc, y_val_enc, X_test_enc, y_test_enc)
    res_loo = eval_classifier(X_loo, y_loo, X_val_enc, y_val_enc, X_test_enc, y_test_enc)

    print(f"{'arm':<12} {'train_bal':>10} {'train_acc':>10} {'val_bal':>10} {'test_bal':>10}")
    for name, r in [("original", res_orig), ("remove_self", res_rs), ("loo", res_loo)]:
        print(f"{name:<12} {r['train_bal']:>10.4f} {r['train_acc']:>10.4f} "
              f"{r['val_bal']:>10.4f} {r['test_bal']:>10.4f}")

    print(f"\n[audit] delta (original - remove_self) test_bal = {res_orig['test_bal'] - res_rs['test_bal']:+.4f}")
    print(f"[audit] delta (original - loo)         test_bal = {res_orig['test_bal'] - res_loo['test_bal']:+.4f}")
    print(f"[audit] delta (remove_self - loo)      test_bal = {res_rs['test_bal'] - res_loo['test_bal']:+.4f}")


if __name__ == "__main__":
    main()