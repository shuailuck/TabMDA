"""Contrastive learning losses for the TabMDA extension.

Implements the grouped weighted SupCon loss specified in
``TabMDA_Contrastive_Loss_Design_and_Codex_Task.docx`` together with the
classification (CE) and optional cross-context consistency terms.

The core idea: a frozen TabPFN encoder produces K embeddings per original
sample (one per labeled context). We flatten the ``[N, K, D]`` views into
``M = N*K`` rows and, for each anchor view ``a``, build three mutually
exclusive masks over the other views ``b``:

    not_self[a,b]      a != b
    instance_positive   not_self && sample_id[a] == sample_id[b]   (same record)
    class_positive      not_self && sample_id differs && label equal
    negative            not_self && label differs

The target distribution ``q`` splits total probability mass ``alpha`` across
the instance-positive group and ``1 - alpha`` across the class-positive group
(when both are nonempty), each evenly within its group. ``q`` is zero on
negatives and on the diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossConfig(nn.Module):
    """Validated hyper-parameters for the combined training objective."""

    def __init__(
        self,
        temperature: float = 0.2,
        alpha: float = 0.5,
        lambda_con: float = 0.1,
        lambda_ctx: float = 0.0,
        mode: str = "grouped",
        require_negatives: bool = True,
    ):
        super().__init__()
        if not temperature > 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if mode == "grouped":
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"alpha must be in (0, 1) for grouped mode, got {alpha}")
        elif mode == "uniform":
            # For uniform SupCon alpha is ignored; still require it to be set sanely.
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"alpha must be in [0, 1] for uniform mode, got {alpha}")
        else:
            raise ValueError(f"Unknown contrastive mode {mode!r} (expected 'grouped' or 'uniform')")

        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.lambda_con = float(lambda_con)
        self.lambda_ctx = float(lambda_ctx)
        self.mode = mode
        self.require_negatives = bool(require_negatives)


class ContrastiveLoss(nn.Module):
    """Vectorized grouped/uniform SupCon loss over flattened multi-context views.

    Args:
        temperature: softmax temperature (must be > 0).
        alpha: probability mass assigned to the instance-positive group.
        mode: ``"grouped"`` (the proposed extension) or ``"uniform"`` (ablation).
        require_negatives: if True, anchors with no negative view are skipped.
    """

    def __init__(self, temperature: float = 0.2, alpha: float = 0.5,
                 mode: str = "grouped", require_negatives: bool = True):
        super().__init__()
        self.cfg = LossConfig(
            temperature=temperature, alpha=alpha, mode=mode, require_negatives=require_negatives
        )

    # ------------------------------------------------------------------ masks
    @staticmethod
    def build_masks(sample_ids: torch.Tensor, labels: torch.Tensor):
        """Build the three boolean ``[M, M]`` masks over flattened views.

        Returns ``(instance_positive, class_positive, negative)``. ``not_self``
        is implied by the three masks being mutually exclusive (diagonal is
        excluded from all of them).
        """
        m = sample_ids.numel()
        a = torch.arange(m, device=sample_ids.device)
        b = a[None, :]
        not_self = (a[:, None] != b)

        same_sample = sample_ids[:, None] == sample_ids[None, :]
        same_label = labels[:, None] == labels[None, :]

        instance_pos = not_self & same_sample & same_label
        class_pos = not_self & (~same_sample) & same_label
        negative = not_self & (~same_label)

        return instance_pos, class_pos, negative

    # ------------------------------------------------------------- q weights
    def target_probs(
        self,
        instance_pos: torch.Tensor,
        class_pos: torch.Tensor,
        sample_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Target distribution q[a,b] in (0, 1], with q diagonal/negatives = 0.

        Rows with no positive are zero (invalid anchor). The optional
        ``negative_mask`` precondition is applied by the caller.
        """
        m = sample_ids.numel()
        q = torch.zeros((m, m), device=sample_ids.device, dtype=torch.float32)

        if self.cfg.mode == "grouped":
            n_inst = instance_pos.sum(dim=1)
            n_class = class_pos.sum(dim=1)

            both = (n_inst > 0) & (n_class > 0)
            only_inst = (n_inst > 0) & (n_class == 0)
            only_class = (n_inst == 0) & (n_class > 0)

            if both.any():
                q[both] += (self.cfg.alpha / n_inst[both].float()).unsqueeze(1) * instance_pos[both].float()
                q[both] += ((1.0 - self.cfg.alpha) / n_class[both].float()).unsqueeze(1) * class_pos[both].float()
            if only_inst.any():
                q[only_inst] += (1.0 / n_inst[only_inst].float()).unsqueeze(1) * instance_pos[only_inst].float()
            if only_class.any():
                q[only_class] += (1.0 / n_class[only_class].float()).unsqueeze(1) * class_pos[only_class].float()
        else:  # uniform
            pos = instance_pos | class_pos
            n_pos = pos.sum(dim=1)
            has = n_pos > 0
            if has.any():
                q[has] = (1.0 / n_pos[has].float()).unsqueeze(1) * pos[has].float()

        return q

    # ------------------------------------------------------------- forward
    def forward(
        self,
        z: torch.Tensor,
        sample_ids: torch.Tensor,
        labels: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor:
        """Compute the (grouped) SupCon loss over flattened views.

        Args:
            z: ``[M, D]`` already L2-normalized representations (or raw adapter
               outputs; they are normalized here again for safety).
            sample_ids: ``[M]`` original-record ids.
            labels: ``[M]`` class labels.

        Returns:
            Scalar loss, or ``(loss, stats_dict)`` if ``return_stats``.
        """
        temperature = self.cfg.temperature
        z = F.normalize(z, p=2, dim=-1, eps=1e-8)

        instance_pos, class_pos, negative = self.build_masks(sample_ids, labels)
        q = self.target_probs(instance_pos, class_pos, sample_ids)

        # Determine valid anchors before masking negatives.
        anchor_has_any_pos = q.sum(dim=1) > 0
        if not anchor_has_any_pos.any():
            dummy = z.sum() * 0.0
            if return_stats:
                return dummy, self._empty_stats()
            return dummy

        # Mask negatives from the denominator by setting q to 0 at those
        # positions (they never contribute, avoiding 0 * -inf).
        if self.cfg.require_negatives:
            has_neg = negative.sum(dim=1) > 0
            valid = anchor_has_any_pos & has_neg
            q_allowed = q * valid.unsqueeze(1).float()
        else:
            valid = anchor_has_any_pos
            q_allowed = q

        # Denominator: logsumexp over all non-self candidates (diagonal -> -inf).
        logits = z @ z.t()  # [M, M]
        logits = logits / temperature
        logits = logits.masked_fill(torch.eye(z.shape[0], device=z.device, dtype=torch.bool), float("-inf"))

        with torch.no_grad():
            row_has_candidate = torch.any(torch.isfinite(logits), dim=1)
        valid = valid & row_has_candidate

        if not valid.any():
            dummy = z.sum() * 0.0
            if return_stats:
                return dummy, self._empty_stats()
            return dummy

        log_denom = torch.logsumexp(logits, dim=1, keepdim=True)  # [M, 1]
        log_pi = logits - log_denom  # -inf on diagonal

        # Accumulate only over valid anchors and positive positions. The
        # diagonal has log_pi = -inf and q = 0, so 0 * (-inf) would be NaN;
        # zero out non-finite log_pi (those positions carry q = 0 anyway).
        log_pi = log_pi.masked_fill(~torch.isfinite(log_pi), 0.0)
        pos_contrib = q_allowed * log_pi
        per_anchor = -pos_contrib.sum(dim=1)  # [M], 0 for rows w/o positive
        loss = per_anchor[valid].mean()

        if return_stats:
            stats = {
                "loss_con": loss.item(),
                "n_anchors": int(valid.sum().item()),
                "n_views": int(z.shape[0]),
                "prop_valid_anchors": valid.float().mean().item(),
                "mean_n_instance_pos": instance_pos.sum(dim=1).float().mean().item(),
                "mean_n_class_pos": class_pos.sum(dim=1).float().mean().item(),
                "mean_n_negatives": negative.sum(dim=1).float().mean().item(),
                "mean_sim_instance_pos": self._masked_sim(z, instance_pos),
                "mean_sim_class_pos": self._masked_sim(z, class_pos),
                "mean_sim_negative": self._masked_sim(z, negative),
            }
            return loss, stats

        return loss

    @staticmethod
    def _masked_sim(z: torch.Tensor, mask: torch.Tensor) -> float:
        sim = z @ z.t()
        vals = sim[mask]
        if vals.numel() == 0:
            return float("nan")
        return vals.mean().item()

    def _empty_stats(self) -> dict:
        return {
            "loss_con": 0.0,
            "n_anchors": 0,
            "n_views": 0,
            "prop_valid_anchors": 0.0,
            "mean_n_instance_pos": float("nan"),
            "mean_n_class_pos": float("nan"),
            "mean_n_negatives": float("nan"),
            "mean_sim_instance_pos": float("nan"),
            "mean_sim_class_pos": float("nan"),
            "mean_sim_negative": float("nan"),
        }


class ContextConsistencyLoss(nn.Module):
    """Optional absolute cross-context consistency: mean 1 - cos(z[i,k], z[i,l])."""

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        # z: [M, D]; sample_ids: [M]
        z = F.normalize(z, p=2, dim=-1, eps=1e-8)
        if sample_ids.numel() == 0:
            return z.sum() * 0.0
        n_per_sample = torch.bincount(sample_ids, minlength=int(sample_ids.max().item()) + 1)
        if (n_per_sample < 2).all():
            return z.sum() * 0.0

        sim = z @ z.t()
        same_sample = sample_ids[:, None] == sample_ids[None, :]
        eye = torch.eye(sample_ids.numel(), device=sample_ids.device, dtype=torch.bool)
        pairs = same_sample & ~eye
        if not pairs.any():
            return z.sum() * 0.0
        dists = 1.0 - sim[pairs]
        return dists.mean()


class JointLoss(nn.Module):
    """Combined CE + contrastive + optional consistency objective."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.contrastive = ContrastiveLoss(
            temperature=cfg.temperature, alpha=cfg.alpha,
            mode=cfg.mode, require_negatives=cfg.require_negatives,
        )
        self.consistency = ContextConsistencyLoss()

        if cfg.lambda_ctx != 0.0 and cfg.lambda_con == 0.0:
            raise ValueError(
                "lambda_ctx != 0 but lambda_con == 0: do not train with the "
                "consistency term alone (it can collapse)."
            )

    def forward(
        self,
        logits: torch.Tensor,          # [M, C]
        z: torch.Tensor,               # [M, D]
        labels: torch.Tensor,          # [M]
        sample_ids: torch.Tensor,      # [M]
        con_weight: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Returns ``(loss, log_dict)``. ``log_dict`` holds scalar floats."""
        loss_ce = F.cross_entropy(logits, labels)

        con_weight = self.cfg.lambda_con if con_weight is None else float(con_weight)
        if con_weight > 0:
            loss_con, stats = self.contrastive(
                z, sample_ids=sample_ids, labels=labels, return_stats=True
            )
        else:
            loss_con = z.sum() * 0.0
            stats = self.contrastive._empty_stats()

        if self.cfg.lambda_ctx != 0.0:
            loss_ctx = self.consistency(z, sample_ids)
        else:
            loss_ctx = z.sum() * 0.0

        total = loss_ce + con_weight * loss_con + self.cfg.lambda_ctx * loss_ctx

        log = {
            "loss_ce": float(loss_ce.detach().item()),
            "loss_con": float(loss_con.detach().item()),
            "loss_ctx": float(loss_ctx.detach().item()),
            "total_loss": float(total.detach().item()),
            "lambda_con": con_weight,
            "prop_valid_anchors": stats.get("prop_valid_anchors", 0.0),
            "mean_n_instance_pos": stats.get("mean_n_instance_pos", float("nan")),
            "mean_n_class_pos": stats.get("mean_n_class_pos", float("nan")),
            "mean_n_negatives": stats.get("mean_n_negatives", float("nan")),
            "mean_sim_instance_pos": stats.get("mean_sim_instance_pos", float("nan")),
            "mean_sim_class_pos": stats.get("mean_sim_class_pos", float("nan")),
            "mean_sim_negative": stats.get("mean_sim_negative", float("nan")),
        }
        return total, log