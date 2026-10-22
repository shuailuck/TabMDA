"""Frozen TabPFN embedding backend for the contrastive extension.

Produces, for each original training sample, ``K`` labeled-context embeddings of
shape ``[N, K, D]`` using the vendored TabPFN v1 encoder (frozen, ``eval`` mode,
``requires_grad=False``). A content-addressed cache keyed by a SHA-256
fingerprint avoids re-encoding identical inputs.

The backend encodes each query *independently* against its own context (the
per-query ``exclude`` semantics). Queries are not batched through the encoder in
a way that makes them attend to each other, so the caching key does not depend
on query-batch composition or order.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from tabmda_contrastive.context_sampler import ContextInfo


ENCODER_VERSION = "tabpfn-v1:prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
EMBED_LOCATION = "TransformerModel.encode (post style/pos, pre-decoder)"
DEFAULT_MODEL_FINGERPRINT = ENCODER_VERSION + "|" + EMBED_LOCATION


def hash_bytes(*arrays) -> str:
    """SHA-256 of one or more numpy arrays (dtype + shape + contents)."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(a.dtype.str.encode())
        h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_fingerprint(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ctx: ContextInfo,
    model_fingerprint: str = DEFAULT_MODEL_FINGERPRINT,
    preprocess_fingerprint: str = "raw",
) -> str:
    """Fingerprint covering data content, contexts (order-sensitive), labels,
    preprocessor state and model/checkpoint/embed-location version."""
    # Contexts: serialize all per-query/per-context index arrays in order.
    ctx_parts: List[np.ndarray] = []
    for q_idx_list in ctx.indices:
        for ctx_arr in q_idx_list:
            ctx_parts.append(np.asarray(ctx_arr, dtype=np.int64))
    parts = [np.asarray(X_train), np.asarray(y_train)] + ctx_parts
    h = hash_bytes(*parts)
    h = hashlib.sha256(h.encode()).digest()
    tail = f"{ctx.context_sizes}|{model_fingerprint}|{preprocess_fingerprint}"
    return hashlib.sha256(h + tail.encode("utf-8")).hexdigest()


class FrozenTabPFNBackend:
    """Wraps a frozen TabPFN encoder to emit ``[N, K, D]`` embeddings.

    Args:
        device: torch device for the encoder. Default ``"cpu"``.
        cache_dir: directory for on-disk embedding cache (None disables disk cache).
        model_fingerprint: string identifying checkpoint + extraction location.
    """

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        model_fingerprint: str = DEFAULT_MODEL_FINGERPRINT,
    ):
        self.device = device
        self.cache_dir = cache_dir
        self.model_fingerprint = model_fingerprint

        from TabMDA.constructor import construct_tabmda_model

        self.tabmda = construct_tabmda_model(
            encoder_name="TabPFN", classifier=None, freeze_encoder=True, device=device
        )
        # Ensure frozen even if the constructor's freeze path is bypassed.
        for p in self.tabmda.parameters():
            p.requires_grad = False

    def embed(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        ctx: ContextInfo,
        preprocess_fingerprint: str = "raw",
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, str]:
        """Return embeddings ``[N, K, D]`` (float32, detached) and the fingerprint.

        Raises ``ValueError`` if ``ctx`` metadata is inconsistent with the data.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        n = X_train.shape[0]
        if len(ctx.indices) != n:
            raise ValueError(
                f"Context has {len(ctx.indices)} queries but X_train has {n} rows."
            )

        fp = build_fingerprint(
            X_train, y_train, ctx,
            model_fingerprint=self.model_fingerprint,
            preprocess_fingerprint=preprocess_fingerprint,
        )

        cached = self._load_cache(fp) if (use_cache and self.cache_dir) else None
        if cached is not None:
            return cached, fp

        K = len(ctx.indices[0]) if n else 0
        D = self._embedding_dim()
        embeddings = torch.empty((n, K, D), dtype=torch.float32, device=self.device)

        X_tr = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        for q in range(n):
            for k in range(K):
                idx = ctx.indices[q][k].astype(np.int64)
                X_sub = X_tr[idx]
                y_sub = torch.tensor(y_train[idx], dtype=torch.long, device=self.device)
                X_q = X_tr[q : q + 1]
                with torch.no_grad():
                    enc = self.tabmda.encode(X_to_encode=X_q, X_context=X_sub, y_context=y_sub)
                embeddings[q, k] = enc.squeeze(1)[0].detach()

        if use_cache and self.cache_dir:
            self._save_cache(fp, embeddings)

        return embeddings, fp

    def embed_eval(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        ctx: ContextInfo,
        preprocess_fingerprint: str = "raw",
    ) -> Tuple[torch.Tensor, str]:
        """Embed held-out queries ``X_eval`` against train-only contexts (from ``ctx``).

        ``ctx`` must be a :class:`ContextInfo` with one entry per eval query
        (as produced by ``sample_eval_contexts``). The query is NOT in the
        training pool, so it is concatenated after the context rows.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        X_eval = np.asarray(X_eval, dtype=np.float32)
        n_eval = X_eval.shape[0]
        if len(ctx.indices) != n_eval:
            raise ValueError(
                f"Context has {len(ctx.indices)} queries but X_eval has {n_eval} rows."
            )

        fp = build_fingerprint(
            X_train, y_train, ctx,
            model_fingerprint=self.model_fingerprint,
            preprocess_fingerprint=preprocess_fingerprint,
        ) + ":eval:" + hash_bytes(X_eval)

        K = len(ctx.indices[0]) if n_eval else 0
        D = self._embedding_dim()
        embeddings = torch.empty((n_eval, K, D), dtype=torch.float32, device=self.device)

        X_tr = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        X_ev = torch.tensor(X_eval, dtype=torch.float32, device=self.device)
        for q in range(n_eval):
            for k in range(K):
                idx = ctx.indices[q][k].astype(np.int64)
                X_sub = X_tr[idx]
                y_sub = torch.tensor(y_train[idx], dtype=torch.long, device=self.device)
                X_q = X_ev[q : q + 1]
                with torch.no_grad():
                    enc = self.tabmda.encode(X_to_encode=X_q, X_context=X_sub, y_context=y_sub)
                embeddings[q, k] = enc.squeeze(1)[0].detach()

        return embeddings, fp

    def _embedding_dim(self) -> int:
        from TabMDA.model import TABPFN_HIDDEN_DIM

        return TABPFN_HIDDEN_DIM

    # ------------------------------------------------------------- cache
    def _cache_path(self, fp: str) -> str:
        return os.path.join(self.cache_dir, f"{fp}.pt")

    def _load_cache(self, fp: str) -> Optional[torch.Tensor]:
        path = self._cache_path(fp)
        if not os.path.isfile(path):
            return None
        try:
            emb = torch.load(path, map_location=self.device)
            if isinstance(emb, torch.Tensor):
                return emb
        except Exception:
            return None
        return None

    def _save_cache(self, fp: str, embeddings: torch.Tensor) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(embeddings.detach().cpu(), self._cache_path(fp))


class IdentityBackend:
    """Deterministic pseudo-'embedding' for CPU smoke tests only.

    Produces ``[N, K, D]`` features derived from the raw inputs via a fixed
    random projection. This backend does NOT touch TabPFN and must never be
    used to report TabMDA performance; it exists so the full training/eval
    pipeline can be exercised without loading the encoder.
    """

    def __init__(self, embed_dim: int = 128, seed: int = 0, **_: object):
        self.embed_dim = embed_dim
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((1, embed_dim)).astype(np.float32)

    def embed(self, X_train, y_train, ctx, preprocess_fingerprint="raw", use_cache=True):
        X_train = np.asarray(X_train, dtype=np.float32)
        n = X_train.shape[0]
        K = len(ctx.indices[0]) if n else 1
        # Deterministic pseudo-embedding: row-sum signature expanded via hash.
        emb = np.zeros((n, K, self.embed_dim), dtype=np.float32)
        for q in range(n):
            for k in range(K):
                seed = int(hash_bytes(np.asarray([q], dtype=np.int64), np.asarray([k], dtype=np.int64))[:8], 16)
                rng = np.random.default_rng(seed)
                emb[q, k] = (rng.standard_normal(self.embed_dim).astype(np.float32) + X_train[q].sum())
        return torch.tensor(emb, dtype=torch.float32), "identity"

    def embed_eval(self, X_train, y_train, X_eval, ctx, preprocess_fingerprint="raw"):
        X_eval = np.asarray(X_eval, dtype=np.float32)
        n_eval = X_eval.shape[0]
        K = len(ctx.indices[0]) if n_eval else 1
        emb = np.zeros((n_eval, K, self.embed_dim), dtype=np.float32)
        for q in range(n_eval):
            for k in range(K):
                seed = int(hash_bytes(
                    np.asarray([q], dtype=np.int64), np.asarray([k], dtype=np.int64)
                )[:8], 16)
                rng = np.random.default_rng(seed)
                emb[q, k] = (rng.standard_normal(self.embed_dim).astype(np.float32) + X_eval[q].sum())
        return torch.tensor(emb, dtype=torch.float32), "identity:eval"