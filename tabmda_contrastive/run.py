"""Command-line entry points for the TabMDA contrastive extension.

Subcommands
-----------
``smoke``     Run the full train/eval pipeline on synthetic data with the
              ``identity`` backend (no TabPFN). Fast CPU sanity check.
``train``     Real TabPFN single/multi-seed experiment on a supported dataset.
``predict``   Predict held-out (unlabeled) rows from a saved adapter checkpoint.

Examples
--------
    python -m tabmda_contrastive.run smoke --epochs 5
    python -m tabmda_contrastive.run train --dataset vehicle \
        --num_real_samples 100 --repeat_id 0 --epochs 30
    python -m tabmda_contrastive.run predict --checkpoint adapter_checkpoint.pt \
        --dataset vehicle --num_real_samples 100 --repeat_id 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from tabmda_contrastive.config import ContrastiveConfig
from tabmda_contrastive.context_sampler import sample_contexts
from tabmda_contrastive.evaluate import (
    prepare_eval_embeddings,
    evaluate_fixed_protocol,
    make_backend,
)
from tabmda_contrastive.train import train


# ---------------------------------------------------------------- data utils
def make_synthetic(n: int = 200, n_features: int = 8, n_classes: int = 3, seed: int = 0):
    """Generate a small 2D-Gaussian-mixture classification problem."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n).astype(np.int64)
    centers = (rng.standard_normal((n_classes, n_features)).astype(np.float32) * 3.0)
    X += centers[y]
    return X, y


def relabel_shared(y_train, y_test):
    """Map labels to contiguous [0, C) using the train classes; returns tuples."""
    classes = np.unique(np.asarray(y_train))
    mapping = {int(c): i for i, c in enumerate(classes)}
    y_tr = np.asarray([mapping[int(c)] for c in y_train], dtype=np.int64)
    try:
        y_te = np.asarray([mapping[int(c)] for c in y_test], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"Test label {e} not present in the training set; cannot evaluate.") from e
    return y_tr, y_te, len(classes)


def contiguous(y):
    return np.asarray(y, dtype=np.int64)


# ---------------------------------------------------------------- config parse
def config_from_args(args) -> ContrastiveConfig:
    cfg = ContrastiveConfig()
    for key in vars(args):
        val = getattr(args, key)
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)
    cfg.device = "cpu"
    return cfg


def add_common_train_args(p):
    p.add_argument("--K", type=int, default=None, help="contexts per original sample (default 2)")
    p.add_argument("--context_size", type=float, default=None,
                   help="int=abs size, float in (0,1]=proportion, None=whole pool")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adapter_hid_dim", type=int, default=None)
    p.add_argument("--adapter_out_dim", type=int, default=None)
    p.add_argument("--use_projection", type=lambda b: b.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--lambda_con", type=float, default=None)
    p.add_argument("--lambda_ctx", type=float, default=None)
    p.add_argument("--contrastive_mode", type=str, default=None, choices=["grouped", "uniform"])
    p.add_argument("--require_negatives", type=lambda b: b.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--optimizer", type=str, default=None, choices=["adamw", "adam", "sgd"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--n_samples_per_batch", type=int, default=None)
    p.add_argument("--K_eval", type=int, default=None)
    p.add_argument("--aggregation", type=str, default=None, choices=["probability_mean", "embedding_mean"])
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="results_contrastive",
                   help="directory for checkpoints and metrics")


# ---------------------------------------------------------------- subcommands
def cmd_smoke(args):
    X, y = make_synthetic(args.n, args.n_features, args.n_classes, args.seed)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    y_train, y_test, num_classes = relabel_shared(y_train, y_test)

    cfg = config_from_args(args)
    cfg.backend = "identity"
    cfg.cache_dir = None

    backend = make_backend(cfg)
    ctx = sample_contexts(X_train, y_train, num_contexts=cfg.K, context_size=cfg.context_size, seed=cfg.seed)
    embeddings, fp = backend.embed(X_train, y_train, ctx)
    print(f"[smoke] train embeddings {tuple(embeddings.shape)} fingerprint={fp}")

    result = train(embeddings, y_train, num_classes, cfg)
    model = result["model"]

    eval_emb = prepare_eval_embeddings(backend, X_train, y_train, X_test, cfg)
    metrics = evaluate_fixed_protocol(model, eval_emb, y_test, cfg)

    _write_metrics(args.out_dir, "smoke", metrics)
    print("[smoke] test metrics:", json.dumps(metrics))
    return metrics


def cmd_train(args):
    from dataset.datasets import get_tabular_dataset, ManualStopError

    try:
        dataset = get_tabular_dataset(
            dataset=args.dataset,
            num_real_samples=args.num_real_samples,
            repeat_id=args.repeat_id,
        )
    except ManualStopError as e:
        print("[train] manual stop:", e)
        return

    X_train = np.asarray(dataset["X_train"], dtype=np.float32)
    y_train = np.asarray(dataset["y_train"])
    X_test = np.asarray(dataset["X_test"], dtype=np.float32)
    y_test = np.asarray(dataset["y_test"])

    y_train, y_test, num_classes = relabel_shared(y_train, y_test)

    cfg = config_from_args(args)
    cfg.backend = "frozen_tabpfn"

    backend = make_backend(cfg)
    ctx = sample_contexts(X_train, y_train, num_contexts=cfg.K, context_size=cfg.context_size, seed=cfg.seed)
    embeddings, fp = backend.embed(X_train, y_train, ctx)
    print(f"[train] train embeddings {tuple(embeddings.shape)} fingerprint={fp}")

    result = train(embeddings, y_train, num_classes, cfg)
    model = result["model"]

    eval_emb = prepare_eval_embeddings(backend, X_train, y_train, X_test, cfg)
    metrics = evaluate_fixed_protocol(model, eval_emb, y_test, cfg)

    _write_metrics(args.out_dir, f"train_{args.dataset}_r{args.repeat_id}", metrics)
    print("[train] test metrics:", json.dumps(metrics))
    return metrics


def cmd_predict(args):
    from dataset.datasets import get_tabular_dataset, ManualStopError
    from tabmda_contrastive.model import ContrastiveModel

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    num_classes = ckpt.get("num_classes")
    if num_classes is None:
        raise ValueError("Checkpoint does not store num_classes.")

    state = ckpt.get("model")
    if state is None:
        raise ValueError("Checkpoint does not store a 'model' state dict.")

    # Prefer the stored config so the adapter architecture is reproduced exactly.
    saved = ckpt.get("config", {})
    adapter_hid_dim = args.adapter_hid_dim or saved.get("adapter_hid_dim", 256)
    adapter_out_dim = args.adapter_out_dim or saved.get("adapter_out_dim", 128)
    use_projection = saved.get("use_projection", False)
    proj_dim = saved.get("proj_dim", 128)

    in_dim = state["adapter.net.0.weight"].shape[1]
    model = ContrastiveModel(
        in_dim=in_dim, num_classes=num_classes,
        adapter_hid_dim=adapter_hid_dim,
        adapter_out_dim=adapter_out_dim,
        use_projection=use_projection,
        proj_dim=proj_dim,
    )
    model.load_state_dict(state)
    model.eval()

    try:
        dataset = get_tabular_dataset(
            dataset=args.dataset,
            num_real_samples=args.num_real_samples,
            repeat_id=args.repeat_id,
        )
    except ManualStopError as e:
        print("[predict] manual stop:", e)
        return

    X_train = np.asarray(dataset["X_train"], dtype=np.float32)
    y_train = np.asarray(dataset["y_train"])
    X_test = np.asarray(dataset["X_test"], dtype=np.float32)

    cfg = config_from_args(args)
    cfg.backend = args.backend or "frozen_tabpfn"
    cfg.adapter_out_dim = adapter_out_dim

    backend = make_backend(cfg)
    eval_emb = prepare_eval_embeddings(backend, X_train, y_train, X_test, cfg)
    probs = _predict_probs(model, eval_emb, cfg)
    preds = probs.argmax(dim=1).numpy()

    out = os.path.join(args.out_dir, "predictions.npz")
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(out, probs=probs.numpy(), preds=preds)
    print(f"[predict] wrote predictions to {out}")


def _predict_probs(model, embeddings, cfg):
    from tabmda_contrastive.evaluate import predict_model
    return predict_model(model, embeddings, cfg)


def _write_metrics(out_dir, name, metrics):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics] wrote {path}")


# ---------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tabmda-contrastive", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("smoke", help="synthetic-data CPU smoke test (identity backend)")
    ps.add_argument("--n", type=int, default=200)
    ps.add_argument("--n_features", type=int, default=8)
    ps.add_argument("--n_classes", type=int, default=3)
    add_common_train_args(ps)

    pt = sub.add_parser("train", help="real TabPFN experiment")
    pt.add_argument("--dataset", type=str, required=True)
    pt.add_argument("--num_real_samples", type=int, default=-1)
    pt.add_argument("--repeat_id", type=int, default=0)
    pt.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds for multi-seed ablation")
    add_common_train_args(pt)

    pp = sub.add_parser("predict", help="predict unlabeled rows from a checkpoint")
    pp.add_argument("--checkpoint", type=str, required=True)
    pp.add_argument("--dataset", type=str, required=True)
    pp.add_argument("--num_real_samples", type=int, default=-1)
    pp.add_argument("--repeat_id", type=int, default=0)
    pp.add_argument("--backend", type=str, default=None)
    pp.add_argument("--adapter_hid_dim", type=int, default=None)
    pp.add_argument("--adapter_out_dim", type=int, default=None)
    pp.add_argument("--K_eval", type=int, default=None)
    pp.add_argument("--context_size", type=float, default=None)
    pp.add_argument("--seed", type=int, default=42)
    pp.add_argument("--out_dir", type=str, default="results_contrastive")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        cmd_smoke(args)
    elif args.command == "train":
        if args.seeds:
            all_metrics = {}
            for s in [int(x) for x in args.seeds.split(",")]:
                args.seed = s
                m = cmd_train(args)
                all_metrics[str(s)] = m
            print("[train] multi-seed summary:", json.dumps(all_metrics, indent=2))
        else:
            cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        build_parser().print_help()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())