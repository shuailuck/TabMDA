"""Leak-free context sampling for the TabMDA contrastive extension.

Each original training sample ``q`` (identified by its row index) receives
``num_contexts`` labeled contexts sampled from the training pool with query ``q``
itself excluded. Sampling is stratified by label and deterministic given a seed,
so a fixed seed reproduces the same contexts (and hence embeddings).

The sampler only returns *indices* (and metadata) — embedding is delegated to
:mod:`tabmda_contrastive.embedding_backend`, which is responsible for fingerprint
keys and caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class ContextInfo:
    """Contexts for a full training pool (all queries at once).

    Attributes:
        indices: list over queries ``q`` of list over contexts ``k`` of ``np.ndarray``
            of row indices (into ``X_train``) forming that context.
        labels: list over queries of list over contexts of ``np.ndarray`` label arrays.
        context_sizes: int, actual (maximum) context size used.
    """

    indices: List[List[np.ndarray]]
    labels: List[List[np.ndarray]]
    context_sizes: int


def resolve_context_size(context_size: Optional[float], pool_size: int) -> int:
    """Map ``context_size`` (None/int/float) to an integer context size."""
    if context_size is None:
        return pool_size
    if context_size > 1:  # int
        size = int(context_size)
    else:  # proportion in (0, 1]
        size = int(round(context_size * pool_size))
    return max(1, min(size, pool_size))


def _stratified_select(n_pool: int, pool_y: np.ndarray, train_size: int, seed: int) -> np.ndarray:
    """Select ``train_size`` indices stratified by ``pool_y`` when possible.

    Falls back to a deterministic non-stratified sample when stratification
    cannot hold (e.g. a class has too few members for the requested size).
    """
    if train_size >= n_pool:
        return np.arange(n_pool)
    try:
        sel, _ = train_test_split(
            np.arange(n_pool),
            train_size=train_size,
            random_state=int(seed % (2 ** 31 - 1)),
            stratify=pool_y,
        )
        return sel
    except ValueError:
        rng = np.random.default_rng(seed)
        return rng.choice(n_pool, size=train_size, replace=False)


def sample_contexts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_contexts: int = 2,
    context_size: Optional[float] = None,
    seed: int = 42,
) -> ContextInfo:
    """Sample per-query, per-context labeled contexts excluding the query itself.

    Args:
        X_train: ``[N, F]`` training features.
        y_train: ``[N]`` integer labels.
        num_contexts: number of contexts per original sample (K).
        context_size: None (whole pool), int (absolute size), or float in (0, 1]
            (proportion of the non-self pool).
        seed: base seed; per-(query, context) seeds are derived deterministically.

    Returns:
        :class:`ContextInfo`.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")
    if y_train.ndim != 1 or y_train.shape[0] != X_train.shape[0]:
        raise ValueError(f"y_train must be 1D with the same length as X_train, got {y_train.shape}")

    n = X_train.shape[0]
    if n < 2:
        raise ValueError("At least two training samples are required to build a non-self context.")

    all_indices = np.arange(n)
    context_size_int = resolve_context_size(context_size, n - 1)

    indices: List[List[np.ndarray]] = []
    labels: List[List[np.ndarray]] = []

    for q in range(n):
        pool = all_indices[all_indices != q]  # exclude the query row itself
        pool_y = y_train[pool]
        pool_nclasses = len(np.unique(pool_y))
        target = context_size_int  # already excludes self implicitly (pool has n-1 rows)

        q_indices: List[np.ndarray] = []
        q_labels: List[np.ndarray] = []
        for k in range(num_contexts):
            # Deterministic per-(query, context) seed -> sample-independent contexts.
            ctx_seed = seed * 1000003 + q * 7919 + k * 131071
            train_size = min(target, len(pool) - pool_nclasses)
            if train_size <= 0:
                train_size = target
            sel = _stratified_select(len(pool), pool_y, train_size, ctx_seed)
            q_indices.append(pool[sel])
            q_labels.append(y_train[pool[sel]])

        indices.append(q_indices)
        labels.append(q_labels)

    return ContextInfo(indices=indices, labels=labels, context_sizes=context_size_int)


def context_info_to_fingerprint_material(ctx: ContextInfo, y_train: np.ndarray) -> dict:
    """Summarize context structure for embedding cache keys."""
    return {
        "num_queries": len(ctx.indices),
        "num_contexts": len(ctx.indices[0]) if ctx.indices else 0,
        "context_sizes": ctx.context_sizes,
        "indices_hash": None,  # replaced by backend with a content hash
    }


def sample_eval_contexts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_eval: int,
    num_contexts: int,
    context_size: Optional[float] = None,
    seed: int = 42,
) -> ContextInfo:
    """Sample contexts for ``n_eval`` (held-out) queries from the training pool.

    Held-out queries never appear in the training pool, so no self-exclusion is
    needed; contexts are sampled stratified from ``X_train``/``y_train`` only.
    The returned :class:`ContextInfo` uses the same layout as
    :func:`sample_contexts` (indices/labels over queries x contexts).
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    n = X_train.shape[0]
    if n < 2:
        raise ValueError("At least two training samples are required to build an eval context.")
    context_size_int = resolve_context_size(context_size, n)
    all_indices = np.arange(n)
    pool_nclasses = len(np.unique(y_train))

    indices: List[List[np.ndarray]] = []
    labels: List[List[np.ndarray]] = []
    for q in range(n_eval):
        q_indices: List[np.ndarray] = []
        q_labels: List[np.ndarray] = []
        for k in range(num_contexts):
            ctx_seed = seed * 1000003 + q * 7919 + k * 131071
            train_size = min(context_size_int, n - pool_nclasses)
            if train_size <= 0:
                train_size = context_size_int
            sel = _stratified_select(n, y_train, train_size, ctx_seed)
            q_indices.append(all_indices[sel])
            q_labels.append(y_train[all_indices[sel]])
        indices.append(q_indices)
        labels.append(q_labels)

    return ContextInfo(indices=indices, labels=labels, context_sizes=context_size_int)