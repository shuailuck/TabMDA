"""End-to-end smoke test on the identity backend (no TabPFN).

Exercises the full embed -> train -> fixed-protocol eval pipeline and checks
the fixed-protocol probability/embedding aggregation paths and training return.
"""

import numpy as np
import torch

from tabmda_contrastive.config import ContrastiveConfig
from tabmda_contrastive.context_sampler import sample_contexts
from tabmda_contrastive.embedding_backend import IdentityBackend
from tabmda_contrastive.evaluate import (
    evaluate_fixed_protocol,
    predict_model,
    prepare_eval_embeddings,
)
from tabmda_contrastive.model import ContrastiveModel
from tabmda_contrastive.train import train


def _data(n=60, n_class=3):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 6)).astype(np.float32)
    y = rng.integers(0, n_class, size=n).astype(np.int64)
    return X, y


def test_full_pipeline_smoke():
    X, y = _data()
    split = int(0.8 * len(X))
    X_train, y_train, X_eval, y_eval = X[:split], y[:split], X[split:], y[split:]

    cfg = ContrastiveConfig(epochs=2, backend="identity", adapter_out_dim=64, batch_size=16)
    backend = IdentityBackend(embed_dim=cfg.adapter_out_dim, seed=cfg.seed)

    ctx = sample_contexts(X_train, y_train, num_contexts=cfg.K, context_size=None, seed=cfg.seed)
    embeddings, fp = backend.embed(X_train, y_train, ctx)
    assert embeddings.shape == (len(X_train), cfg.K, cfg.adapter_out_dim)

    result = train(embeddings, y_train, int(y_train.max()) + 1, cfg)
    assert result["model"] is not None
    assert len(result["history"]) == cfg.epochs

    eval_emb = prepare_eval_embeddings(backend, X_train, y_train, X_eval, cfg)
    assert eval_emb.shape == (len(X_eval), cfg.K_eval, cfg.adapter_out_dim)

    metrics = evaluate_fixed_protocol(result["model"], eval_emb, y_eval, cfg)
    for k in ["accuracy", "balanced_accuracy", "macro_f1", "log_loss"]:
        assert k in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_predict_model_aggregations_match():
    torch.manual_seed(0)
    model = ContrastiveModel(in_dim=64, num_classes=3, adapter_out_dim=64)
    emb = torch.randn(5, 4, 64)

    cfg_pm = ContrastiveConfig(aggregation="probability_mean")
    cfg_em = ContrastiveConfig(aggregation="embedding_mean")
    probs_pm = predict_model(model, emb, cfg_pm)
    probs_em = predict_model(model, emb, cfg_em)
    assert probs_pm.shape == (5, 3)
    assert probs_em.shape == (5, 3)
    # rows sum to 1 (probabilities) for both paths.
    assert torch.allclose(probs_pm.sum(1), torch.ones(5), atol=1e-5)
    assert torch.allclose(probs_em.sum(1), torch.ones(5), atol=1e-5)