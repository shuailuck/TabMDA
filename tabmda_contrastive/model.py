"""Trainable adapter + classification head on top of frozen TabPFN embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    """Small MLP adapter: ``Linear(D, hid) -> GELU -> Linear(hid, out)``.

    ``out`` defaults to ``out_dim``; set ``out = 2 * out_dim`` when a projection
    head is requested (see :class:`ContrastiveModel`). The adapter output is
    never detached so contrastive gradients flow back into it.
    """

    def __init__(self, in_dim: int, hid_dim: int = 256, out_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveModel(nn.Module):
    """Adapter + classification head over frozen TabPFN embeddings.

    Args:
        in_dim: TabPFN embedding dimension (512 for the vendored v1 encoder).
        num_classes: number of classes for the supervised classification head.
        adapter_hid_dim: adapter hidden width.
        adapter_out_dim: adapter output dimension (default 128). This is also
            the z/r representation dimension unless ``use_projection`` is set.
        dropout: adapter dropout.
        use_projection: if True, a dedicated projection head ``adapter_out_dim ->
            proj_dim`` feeds z while ``r`` still feeds the classifier; gradients
            still reach the adapter.
        proj_dim: projection head output dimension (only used if ``use_projection``).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        adapter_hid_dim: int = 256,
        adapter_out_dim: int = 128,
        dropout: float = 0.0,
        use_projection: bool = False,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.use_projection = use_projection
        self.adapter = Adapter(
            in_dim=in_dim,
            hid_dim=adapter_hid_dim,
            out_dim=adapter_out_dim * 2 if use_projection else adapter_out_dim,
            dropout=dropout,
        )
        self.head = nn.Linear(adapter_out_dim, num_classes)
        self.proj = nn.Sequential(nn.Linear(adapter_out_dim, proj_dim), nn.ReLU()) if use_projection else None

    def forward(self, e: torch.Tensor) -> dict:
        """Given embeddings ``e`` of shape ``[..., in_dim]``, return unified
        representations ``r`` and L2-normalized ``z``.

        Returns a dict with keys ``"r"``, ``"z"``, ``"logits"``.
        """
        flat = e.reshape(-1, e.shape[-1])
        out = self.adapter(flat)
        if self.use_projection:
            r = out[:, : out.shape[1] // 2]
            z = self.proj(r).reshape(*e.shape[:-1], -1)
        else:
            r = out
            z = r.reshape(*e.shape[:-1], -1)
        r = r.reshape(*e.shape[:-1], -1)
        z = z.reshape(*e.shape[:-1], -1)

        r_flat = r.reshape(-1, r.shape[-1])
        logits = self.head(r_flat)
        logits = logits.reshape(*e.shape[:-1], -1)
        z = F.normalize(z, p=2, dim=-1, eps=1e-8)
        return {"r": r, "z": z, "logits": logits}

    def classify(self, e: torch.Tensor) -> torch.Tensor:
        """Class probabilities (softmax) from embeddings ``e``."""
        return F.softmax(self(e)["logits"], dim=-1)