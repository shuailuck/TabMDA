"""Acceptance tests: adapter model, config, and hash/fingerprint utilities."""

import numpy as np
import torch

from tabmda_contrastive.config import ContrastiveConfig
from tabmda_contrastive.embedding_backend import build_fingerprint, hash_bytes
from tabmda_contrastive.model import Adapter, ContrastiveModel
from tabmda_contrastive.context_sampler import sample_contexts


def test_adapter_output_shape():
    a = Adapter(in_dim=512, hid_dim=256, out_dim=128)
    x = torch.randn(4, 512)
    assert a(x).shape == (4, 128)


def test_model_forward_shapes():
    m = ContrastiveModel(in_dim=512, num_classes=5, adapter_hid_dim=256, adapter_out_dim=128)
    e = torch.randn(3, 2, 512)
    out = m(e)
    assert out["logits"].shape == (3, 2, 5)
    assert out["z"].shape == (3, 2, 128)
    assert out["r"].shape == (3, 2, 128)
    # z is L2 normalized along the last dim.
    n = torch.norm(out["z"], p=2, dim=-1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-5)


def test_projection_mode_shapes():
    m = ContrastiveModel(in_dim=512, num_classes=3, use_projection=True, proj_dim=64)
    e = torch.randn(2, 2, 512)
    out = m(e)
    assert out["z"].shape == (2, 2, 64)
    assert out["logits"].shape == (2, 2, 3)


def test_config_defaults_and_fingerprint():
    cfg = ContrastiveConfig()
    assert cfg.K == 2
    assert cfg.adapter_out_dim == 128
    assert cfg.temperature == 0.2
    assert cfg.alpha == 0.5
    assert cfg.lambda_con == 0.1
    d = cfg.to_dict()
    assert d["temperature"] == 0.2
    assert isinstance(cfg.fingerprint(), str) and len(cfg.fingerprint()) == 16


def test_hash_bytes_deterministic_and_sensitive():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    assert hash_bytes(a) == hash_bytes(a)
    assert hash_bytes(a) != hash_bytes(b)
    assert hash_bytes(a, a) == hash_bytes(a, a)


def test_build_fingerprint_includes_contexts():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    ctx1 = sample_contexts(X, y, num_contexts=1, context_size=2, seed=0)
    fp1 = build_fingerprint(X, y, ctx1, model_fingerprint="test:model")
    fp1_again = build_fingerprint(X, y, ctx1, model_fingerprint="test:model")
    assert fp1 == fp1_again

    # Different contexts (different size) -> different fingerprint.
    ctx2 = sample_contexts(X, y, num_contexts=1, context_size=3, seed=0)
    fp2 = build_fingerprint(X, y, ctx2, model_fingerprint="test:model")
    assert fp1 != fp2