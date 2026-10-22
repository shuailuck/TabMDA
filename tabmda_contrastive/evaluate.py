"""Fixed-protocol evaluation for the TabMDA contrastive extension.

Inference always uses ``K_eval`` labeled contexts. Each context is classified
individually and probabilities are averaged (``probability_mean``) — the default
fixed protocol. ``embedding_mean`` (average embeddings before classification) is
also available but is not the default.

Evaluation never requires ground-truth labels for the query rows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from tabmda_contrastive.config import ContrastiveConfig
from tabmda_contrastive.context_sampler import sample_eval_contexts
from tabmda_contrastive.embedding_backend import FrozenTabPFNBackend, IdentityBackend


def predict_model(
    model,
    embeddings: torch.Tensor,        # [N_eval, K, D]
    cfg: ContrastiveConfig,
) -> torch.Tensor:
    """Average over the K eval contexts to produce ``[N_eval, C]`` probabilities."""
    model.eval()
    device = cfg.device
    n, K, d = embeddings.shape
    if cfg.aggregation == "embedding_mean":
        z_mean = F.normalize(embeddings.float().mean(dim=1), p=2, dim=-1, eps=1e-8)
        with torch.no_grad():
            logits = model.head(model.adapter(z_mean.to(device)))
        return F.softmax(logits, dim=-1).cpu()
    # probability_mean (default)
    with torch.no_grad():
        probs = []
        for k in range(K):
            out = model(embeddings[:, k : k + 1].to(device))
            probs.append(F.softmax(out["logits"], dim=-1).squeeze(1))
        return torch.stack(probs, dim=1).mean(dim=1).cpu()


def evaluate_fixed_protocol(
    model,
    embeddings: torch.Tensor,
    targets: np.ndarray,
    cfg: ContrastiveConfig,
) -> dict:
    """Return metrics (accuracy, balanced accuracy, macro-F1, log loss) on fixed K_eval contexts."""
    targets = np.asarray(targets)
    probs = predict_model(model, embeddings, cfg)
    preds = torch.argmax(probs, dim=1).numpy()
    probs_np = probs.numpy()

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
    )

    acc = accuracy_score(targets, preds)
    bacc = balanced_accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    ll = log_loss(targets, probs_np, labels=np.arange(probs_np.shape[1])) if probs_np.shape[1] > 1 else float("nan")
    return {"accuracy": float(acc), "balanced_accuracy": float(bacc), "macro_f1": float(macro_f1), "log_loss": float(ll)}


def prepare_eval_embeddings(
    backend,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    cfg: ContrastiveConfig,
    query_ids: Optional[np.ndarray] = None,
    preprocess_fingerprint: str = "raw",
) -> torch.Tensor:
    """Build ``[N_eval, K_eval, D]`` embeddings for held-out rows using train-only contexts."""
    n_eval = X_eval.shape[0]
    ctx = sample_eval_contexts(
        X_train, y_train,
        n_eval=n_eval,
        num_contexts=cfg.K_eval,
        context_size=cfg.context_size,
        seed=cfg.seed,
    )
    # Split the sampled contexts per query but the backend embeds with a query position.
    # For held-out rows the query is NOT in the training pool, so embed each eval query
    # by concatenating it to the sampled context (query index = train size).
    emb, _ = backend.embed_eval(X_train, y_train, X_eval, ctx, preprocess_fingerprint=preprocess_fingerprint)
    return emb


def make_backend(cfg: ContrastiveConfig):
    if cfg.backend == "frozen_tabpfn":
        return FrozenTabPFNBackend(
            device=cfg.device,
            cache_dir=cfg.cache_dir,
            model_fingerprint=cfg.model_fingerprint or "tabpfn-v1:prior_diff_real_checkpoint_n_0_epoch_42.cpkt|TransformerModel.encode",
        )
    if cfg.backend == "identity":
        return IdentityBackend(embed_dim=cfg.adapter_out_dim, seed=cfg.seed)
    raise ValueError(f"Unknown backend {cfg.backend}")