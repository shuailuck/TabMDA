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


def _sample_contexts_random(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_contexts: int,
    context_size: Optional[float],
    seed: int,
) -> ContextInfo:
    """TabMDA ``self_context="random"``: one shared stratified context per index.

    Reproduces :meth:`TabMDA.model.TabMDA.context_subsetting` with
    ``self_context="random"`` (TabMDA's default): given ``context_size``, sample a
    label-stratified context from the FULL pool (the query row may appear in its own
    context) and reuse it for every query. The context size is computed off
    ``len(X_train)`` (not ``n-1``) and the per-index seed is ``seed + k``, both
    matching TabMDA exactly.
    """
    n = X_train.shape[0]
    num_classes = len(np.unique(y_train))

    if context_size is None:
        size = n
    elif context_size > 1:  # int
        size = int(context_size)
    else:  # proportion in (0, 1]
        size = int(context_size * n)
    size = max(1, min(size, n))

    all_indices = np.arange(n)
    seeds_for_contexts = [seed + k for k in range(num_contexts)]

    shared_indices: List[np.ndarray] = []
    shared_labels: List[np.ndarray] = []
    for k in range(num_contexts):
        if size >= n:
            sel = all_indices
        else:
            train_size = min(size, n - num_classes)
            if train_size <= 0:
                train_size = size
            try:
                sel, _, _, _ = train_test_split(
                    all_indices, all_indices,
                    train_size=train_size,
                    random_state=seeds_for_contexts[k],
                    stratify=np.asarray(y_train),
                )
            except ValueError:
                rng = np.random.default_rng(seeds_for_contexts[k])
                sel = rng.choice(n, size=train_size, replace=False)
        shared_indices.append(sel)
        shared_labels.append(np.asarray(y_train)[sel])

    # The same context is shared across all queries (TabMDA "random" behaviour),
    # so the per-query layout is just a repeating copy.
    indices = [list(shared_indices) for _ in range(n)]
    labels = [list(shared_labels) for _ in range(n)]
    return ContextInfo(indices=indices, labels=labels, context_sizes=size)


def sample_contexts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_contexts: int = 2,
    context_size: Optional[float] = None,
    seed: int = 42,
    self_context: str = "exclude",
) -> ContextInfo:
    """Sample per-query labeled contexts for each training sample.

    Args:
        X_train: ``[N, F]`` training features.
        y_train: ``[N]`` integer labels.
        num_contexts: number of contexts per original sample (K).
        context_size: None (whole pool), int (absolute size), or float in (0, 1]
            (proportion of the pool).
        seed: base seed; per-(query, context) seeds are derived deterministically.
        self_context: how to treat a query row's own ``(x_i, y_i)`` when sampling
            its context. ``"exclude"`` (default): per-query leave-one-out, the
            query row is never in its own context. ``"random"``: reproduces
            TabMDA's default — one shared label-stratified context per context
            index, sampled from the full pool (so the query row *may* appear in
            its own context).

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
        raise ValueError("At least two training samples are required to build a context.")

    if self_context == "random":
        return _sample_contexts_random(X_train, y_train, num_contexts, context_size, seed)
    if self_context != "exclude":
        raise ValueError(f"self_context must be 'exclude' or 'random', got {self_context!r}")

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