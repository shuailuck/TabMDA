"""Acceptance tests for the context sampler (leak-free, deterministic)."""

import numpy as np

from tabmda_contrastive.context_sampler import (
    ContextInfo,
    resolve_context_size,
    sample_contexts,
    sample_eval_contexts,
)


def _data(n=40, n_class=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5)).astype(np.float32)
    y = rng.integers(0, n_class, size=n).astype(np.int64)
    return X, y


def test_resolve_context_size():
    assert resolve_context_size(None, 100) == 100
    assert resolve_context_size(30, 100) == 30
    assert resolve_context_size(0.5, 100) == 50
    assert resolve_context_size(0, 100) == 1  # clamped to >= 1


def test_sample_contexts_excludes_self():
    X, y = _data()
    ctx = sample_contexts(X, y, num_contexts=2, context_size=10, seed=42)
    n = X.shape[0]
    assert len(ctx.indices) == n
    for q in range(n):
        for k in range(ctx.num_contexts if hasattr(ctx, "num_contexts") else 2):
            assert q not in ctx.indices[q][k]


def test_sample_contexts_shapes_and_labels():
    X, y = _data()
    ctx = sample_contexts(X, y, num_contexts=3, context_size=8, seed=1)
    for q in range(X.shape[0]):
        assert len(ctx.indices[q]) == 3
        for k in range(3):
            idx = ctx.indices[q][k]
            assert idx.shape[0] == min(8, X.shape[0] - 1) or idx.shape[0] <= 8
            # labels correspond to the sampled indices
            assert np.array_equal(ctx.labels[q][k], y[idx])


def test_sample_contexts_deterministic():
    X, y = _data()
    a = sample_contexts(X, y, num_contexts=2, context_size=10, seed=7)
    b = sample_contexts(X, y, num_contexts=2, context_size=10, seed=7)
    for q in range(X.shape[0]):
        for k in range(2):
            assert np.array_equal(a.indices[q][k], b.indices[q][k])


def test_sample_eval_contexts_no_exclusion_needed():
    X, y = _data(n=40)
    n_eval = 12
    ctx = sample_eval_contexts(X, y, n_eval=n_eval, num_contexts=2, context_size=10, seed=3)
    assert len(ctx.indices) == n_eval
    for q in range(n_eval):
        for k in range(2):
            idx = ctx.indices[q][k]
            assert idx.max() < X.shape[0]
            assert np.array_equal(ctx.labels[q][k], y[idx])


def test_eval_contexts_deterministic():
    X, y = _data()
    a = sample_eval_contexts(X, y, n_eval=8, num_contexts=2, context_size=6, seed=11)
    b = sample_eval_contexts(X, y, n_eval=8, num_contexts=2, context_size=6, seed=11)
    for q in range(8):
        for k in range(2):
            assert np.array_equal(a.indices[q][k], b.indices[q][k])