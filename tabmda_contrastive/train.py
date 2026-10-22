"""Training loop for the TabMDA contrastive extension.

Trains the small adapter + classification head over frozen TabPFN embeddings
using the combined objective ``L = CE + lambda_con * L_con + lambda_ctx * L_ctx``.

The minibatch is built from ``n_samples_per_batch`` distinct original samples,
each expanded to K views. A simple sampler prefers batches with at least two
classes and (when possible) at least two samples per class so that the
contrastive loss has valid anchors; when data is too small it falls back to
plain classification training without fabricating positives.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from tabmda_contrastive.config import ContrastiveConfig
from tabmda_contrastive.losses import JointLoss, LossConfig
from tabmda_contrastive.model import ContrastiveModel
from tabmda_contrastive.evaluate import evaluate_fixed_protocol


class MinibatchSampler:
    """Draws minibatches of *original samples* (each with K views)."""

    def __init__(self, y: np.ndarray, classes: np.ndarray, cfg: ContrastiveConfig):
        self.y = np.asarray(y)
        self.classes = classes
        self.cfg = cfg
        self.n_samples = cfg.batch_size if cfg.n_samples_per_batch is None else cfg.n_samples_per_batch
        if self.n_samples < 2:
            raise ValueError("n_samples_per_batch (or batch_size) must be >= 2 for contrastive training.")
        self.by_class = {int(c): np.where(self.y == c)[0] for c in classes}
        self.n_classes = len(classes)

    def __len__(self) -> int:
        return max(1, int(math.ceil(len(self.y) / self.n_samples)))

    def __iter__(self):
        return self._gen()

    def _gen(self):
        rng = np.random.default_rng(self.cfg.seed)
        classes = list(self.classes)
        for _ in range(len(self)):
            # Prefer 2+ classes; fallback to any single class if only one is available.
            if self.n_classes >= 2:
                n_cls = min(self.n_classes, max(2, rng.integers(2, self.n_classes + 1)))
                chosen = rng.choice(classes, size=n_cls, replace=False)
            else:
                chosen = classes
            per = self.n_samples // len(chosen)
            picks = []
            for c in chosen:
                pool = self.by_class[int(c)]
                k = min(len(pool), per)
                picks.append(rng.choice(pool, size=k, replace=False))
            if len(picks):
                picks = np.concatenate(picks)
            else:
                picks = np.array([], dtype=np.int64)
            if picks.size < self.n_samples:
                rest = self.n_samples - picks.size
                extra_pool = np.setdiff1d(np.arange(len(self.y)), picks)
                if len(extra_pool):
                    picks = np.concatenate([picks, rng.choice(extra_pool, size=min(rest, len(extra_pool)), replace=False)])
            rng.shuffle(picks)
            yield picks


def con_weight_schedule(step: int, total_steps: int, target: float, warmup_ratio: float) -> float:
    """Linear warmup of the contrastive weight over the first ``warmup_ratio`` fraction."""
    if warmup_ratio <= 0 or target <= 0:
        return target
    wu_steps = max(1, int(total_steps * warmup_ratio))
    if step >= wu_steps:
        return target
    return target * (step + 1) / wu_steps


def train(
    embeddings: torch.Tensor,        # [N, K, D]
    labels: np.ndarray,              # [N]
    num_classes: int,
    cfg: ContrastiveConfig,
    val_loader: Optional[Callable] = None,
    logger: Optional[Callable[[dict], None]] = None,
    progress: bool = False,
) -> dict:
    """Train the adapter + head. Returns a state dict and final training metrics."""
    device = cfg.device
    emb_dim = embeddings.shape[-1]
    n_train = embeddings.shape[0]
    labels = np.asarray(labels, dtype=np.int64)

    model = ContrastiveModel(
        in_dim=emb_dim,
        num_classes=num_classes,
        adapter_hid_dim=cfg.adapter_hid_dim,
        adapter_out_dim=cfg.adapter_out_dim,
        dropout=cfg.dropout,
        use_projection=cfg.use_projection,
        proj_dim=cfg.proj_dim,
    ).to(device)

    loss_cfg = LossConfig(
        temperature=cfg.temperature,
        alpha=cfg.alpha,
        lambda_con=cfg.lambda_con,
        lambda_ctx=cfg.lambda_ctx,
        mode=cfg.contrastive_mode,
        require_negatives=cfg.require_negatives,
    )
    joint = JointLoss(loss_cfg)

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {cfg.optimizer}")

    # Flatten views once; index original-sample -> views.
    emb_flat = embeddings.reshape(-1, emb_dim).to(device)
    K = embeddings.shape[1]
    sample_ids_flat = torch.repeat_interleave(
        torch.arange(n_train, device=device), K
    )
    labels_flat = torch.repeat_interleave(
        torch.tensor(labels, device=device), K
    )

    sampler = MinibatchSampler(labels, np.arange(num_classes), cfg)
    steps_per_epoch = max(1, len(sampler))
    total_steps = steps_per_epoch * cfg.epochs

    history = []
    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses = []
        for picks in sampler:
            idx = torch.tensor(picks, device=device, dtype=torch.long)
            # Map original-sample picks -> flat view indices.
            view_idx = (idx.unsqueeze(1) * K + torch.arange(K, device=device).unsqueeze(0)).reshape(-1)
            e = emb_flat[view_idx]
            out = model(e)  # logits [B*K, C], z [B*K, D]
            y = labels_flat[view_idx]
            sid = sample_ids_flat[view_idx]
            sids_local = sid // K  # original sample id (== picks)

            w = con_weight_schedule(global_step, total_steps, cfg.lambda_con, cfg.con_warmup_ratio)
            loss, log = joint(out["logits"], out["z"], labels=y, sample_ids=sid,
                              con_weight=w)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_losses.append(log)
            global_step += 1

            if cfg.log_interval > 0 and global_step % cfg.log_interval == 0:
                log["epoch"] = epoch
                log["step"] = global_step
                if logger:
                    logger(log)

        # End-of-epoch summary (skip empty epochs, e.g. zero-length fallback).
        if not epoch_losses:
            continue
        agg = {k: float(np.nanmean([d[k] for d in epoch_losses])) for k in
               ["loss_ce", "loss_con", "loss_ctx", "total_loss", "prop_valid_anchors"]}
        agg["epoch"] = epoch
        summary = {"epoch": epoch, **agg}
        history.append(summary)
        if val_loader is not None and (epoch + 1) % cfg.eval_interval == 0:
            val_metrics = val_loader(model, epoch)
            summary.update({f"val_{k}": v for k, v in val_metrics.items()})
        if logger:
            logger(summary)

    torch.save(
        {"model": model.state_dict(), "config": cfg.to_dict(), "num_classes": num_classes},
        "adapter_checkpoint.pt",
    )
    return {"model": model, "state_dict": model.state_dict(), "history": history}