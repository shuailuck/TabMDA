"""Acceptance tests for the grouped weighted SupCon loss.

Covers masks, q-weights/fallbacks, no-positive anchors, no-negatives handling,
numeric stability and the gradient path.
"""

import numpy as np
import torch

from tabmda_contrastive.losses import (
    ContrastiveLoss,
    ContextConsistencyLoss,
    JointLoss,
    LossConfig,
)


def _mk(z, sample_ids, labels, **kw):
    """Small helper to run the contrastive loss with return_stats."""
    return ContrastiveLoss(**kw)(z, sample_ids=sample_ids, labels=labels, return_stats=True)


def test_build_masks_mutually_exclusive():
    sample_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    labels = torch.tensor([0, 0, 0, 0, 1, 1])
    cl = ContrastiveLoss()
    instance_pos, class_pos, negative = cl.build_masks(sample_ids, labels)

    # Diagonal excluded everywhere.
    assert not instance_pos.diagonal().any()
    assert not class_pos.diagonal().any()
    assert not negative.diagonal().any()

    # Mutually exclusive (for a valid label config there is no overlap).
    assert not (instance_pos & class_pos).any()
    assert not (instance_pos & negative).any()
    assert not (class_pos & negative).any()

    # sample 0 (label 0): instance positive is view 1; class positives = rows 2,3.
    assert instance_pos[0, 1].item()
    assert class_pos[0, 2].item() and class_pos[0, 3].item()
    assert negative[0, 4].item() and negative[0, 5].item()


def test_target_probs_grouped_mass_split():
    sample_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    labels = torch.tensor([0, 0, 0, 0, 1, 1])
    cl = ContrastiveLoss(alpha=0.5)
    instance_pos, class_pos, negative = cl.build_masks(sample_ids, labels)
    q = cl.target_probs(instance_pos, class_pos, sample_ids)

    # Row 0: 1 instance positive (α=0.5) + 2 class positives (0.25 each).
    assert np.isclose(q[0, 1].item(), 0.5)
    assert np.isclose(q[0, 2].item(), 0.25)
    assert np.isclose(q[0, 3].item(), 0.25)
    assert np.isclose(q[0].sum().item(), 1.0)

    # Sample 4 (label 1): no class positive (only one distinct sample in its
    # class), so the instance-positive group absorbs the full mass.
    assert np.isclose(q[4, 5].item(), 1.0)
    assert np.isclose(q[4].sum().item(), 1.0)


def test_target_probs_only_class_positive_fallback():
    # K=1 per sample (no instance positives) but multiple samples per class.
    sample_ids = torch.tensor([0, 1, 2, 3])
    labels = torch.tensor([0, 0, 1, 1])
    cl = ContrastiveLoss(alpha=0.5)
    instance_pos, class_pos, negative = cl.build_masks(sample_ids, labels)
    q = cl.target_probs(instance_pos, class_pos, sample_ids)
    # No instance positives at all.
    assert instance_pos.sum().item() == 0
    # Row 0: one class positive (row 1) gets full mass.
    assert np.isclose(q[0, 1].item(), 1.0)
    assert np.isclose(q[0].sum().item(), 1.0)


def test_no_positive_anchor_skipped():
    # A singleton with no same-sample and no same-label view has no positives.
    sample_ids = torch.tensor([0, 1, 2])
    labels = torch.tensor([0, 1, 2])
    z = torch.randn(3, 8)
    loss, stats = ContrastiveLoss()(z, sample_ids=sample_ids, labels=labels, return_stats=True)
    assert stats["n_anchors"] == 0
    assert stats["loss_con"] == 0.0


def test_no_negatives_handling():
    # All same-sample same-label: no negatives available.
    sample_ids = torch.tensor([0, 0, 1, 1])
    labels = torch.tensor([0, 0, 0, 0])
    z = torch.randn(4, 8)

    # require_negatives=True -> anchors skipped -> loss 0, finite.
    loss_req, _ = ContrastiveLoss(require_negatives=True)(
        z, sample_ids=sample_ids, labels=labels, return_stats=True
    )
    assert torch.isfinite(loss_req)

    # require_negatives=False -> still finite (denominator over positives only).
    loss_no_req, _ = ContrastiveLoss(require_negatives=False)(
        z, sample_ids=sample_ids, labels=labels, return_stats=True
    )
    assert torch.isfinite(loss_no_req)


def test_numeric_stability_random():
    torch.manual_seed(0)
    n, K = 50, 3
    sample_ids = torch.repeat_interleave(torch.arange(n), K)
    labels = torch.repeat_interleave(torch.arange(n) % 5, K)
    z = torch.randn(n * K, 16)
    loss = ContrastiveLoss()(z, sample_ids=sample_ids, labels=labels)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_gradient_path():
    z = torch.randn(6, 4, requires_grad=True)
    sample_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    loss = ContrastiveLoss()(z, sample_ids=sample_ids, labels=labels)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_consistency_loss_range():
    z = torch.randn(4, 8)
    sample_ids = torch.tensor([0, 0, 1, 1])
    closs = ContextConsistencyLoss()(z, sample_ids)
    # 1 - cos in [0, 2].
    assert 0.0 <= closs.item() <= 2.0


def test_joint_loss_shape_and_log():
    logits = torch.randn(6, 3)
    z = torch.randn(6, 8)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    sample_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    cfg = LossConfig()
    loss, log = JointLoss(cfg)(logits, z, labels=labels, sample_ids=sample_ids)
    assert torch.isfinite(loss)
    for k in ["loss_ce", "loss_con", "loss_ctx", "total_loss", "prop_valid_anchors"]:
        assert k in log


def test_permutation_invariance():
    """The contrastive loss must not depend on the order of the flattened views."""
    n, K = 8, 3
    sample_ids = torch.repeat_interleave(torch.arange(n), K)
    labels = torch.repeat_interleave(torch.arange(n) % 3, K)
    z = torch.randn(n * K, 12)

    perm = torch.randperm(n * K)
    loss1 = ContrastiveLoss()(z, sample_ids=sample_ids, labels=labels)
    loss2 = ContrastiveLoss()(z[perm], sample_ids=sample_ids[perm], labels=labels[perm])
    assert torch.allclose(loss1, loss2, atol=1e-6)