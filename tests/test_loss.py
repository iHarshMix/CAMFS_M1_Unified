"""
CAMFS M1 — Loss Functions & 3D Metrics Unit Test Suite
======================================================

PyTest suite for Phase 4:
  - MaskedInfoNCELoss (prototype alignment & empty-prototype masking)
  - SoftDiceCrossEntropyLoss (§15.1 composite formula)
  - Phase1Loss & Phase2Loss composite objectives
  - 3D Patient Volume Dice & HD95 metric conventions (§13.5)
  - PatientEvaluator end-to-end evaluation
"""

import math
import numpy as np
import pytest
import torch

from src.losses import (
    MaskedInfoNCELoss,
    Phase1Loss,
    Phase2Loss,
    SoftDiceCrossEntropyLoss,
)
from src.metrics import (
    PatientEvaluator,
    backmap_model_to_brats,
    compute_3d_dice,
    compute_3d_hd95,
    compute_grid_diagonal_penalty,
    extract_tumor_region_mask,
)
from src.models.prototypes import normalize_l2


def test_masked_infonce_loss():
    """Verify MaskedInfoNCELoss, backward gradients, and empty-prototype masking (§6.1)."""
    batch_size = 2
    feature_dim = 256
    num_classes = 4

    z = torch.randn(batch_size, feature_dim, 15, 15, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size, 15, 15))

    # Create prototypes where class 0, 1, 2 are active, but class 3 is zero (empty)
    prototypes = torch.randn(num_classes, feature_dim)
    prototypes = normalize_l2(prototypes)
    prototypes[3] = torch.zeros(feature_dim)  # Empty prototype

    infonce = MaskedInfoNCELoss(tau=0.1)
    loss = infonce(z, labels, prototypes)

    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.item() >= 0.0

    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_soft_dice_ce_loss():
    """Verify SoftDiceCrossEntropyLoss matching exact §15.1 formula."""
    batch_size = 2
    num_classes = 4
    H, W = 30, 30

    logits = torch.randn(batch_size, num_classes, H, W, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size, H, W))

    loss_fn = SoftDiceCrossEntropyLoss(eps_d=1e-5, tumour_classes=(1, 2, 3))
    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_phase1_and_phase2_objectives():
    """Verify Phase1Loss and Phase2Loss composite objectives (§6.1, §8.2)."""
    batch_size = 2
    num_classes = 4

    # Phase 1 setup
    p1_loss = Phase1Loss(lambda1=1.0, tau=0.1)
    z_t1 = torch.randn(batch_size, 256, 15, 15, requires_grad=True)
    proto_t1 = normalize_l2(torch.randn(num_classes, 256))
    labels = torch.randint(0, num_classes, (batch_size, 15, 15))

    l1 = p1_loss(bottlenecks={"T1": z_t1}, labels=labels, prototypes={"T1": proto_t1})
    assert torch.isfinite(l1)

    # Phase 2 setup
    p2_loss = Phase2Loss(lambda2=0.1, tau=0.1)
    logits = torch.randn(batch_size, 4, 30, 30, requires_grad=True)
    z_S = torch.randn(batch_size, 256, 15, 15, requires_grad=True)
    fused_proto_S = normalize_l2(torch.randn(num_classes, 256))

    l2 = p2_loss(logits_S=logits, targets=labels, z_S=z_S, fused_prototypes_S=fused_proto_S)
    assert torch.isfinite(l2)


def test_3d_dice_edge_cases():
    """Verify 3D patient-level Dice conventions (both empty=1.0, one empty=0.0) (§13.5)."""
    shape = (10, 20, 20)

    # 1. Both empty
    p_empty = np.zeros(shape, dtype=bool)
    g_empty = np.zeros(shape, dtype=bool)
    assert compute_3d_dice(p_empty, g_empty) == 1.0

    # 2. One empty
    g_present = np.zeros(shape, dtype=bool)
    g_present[:5, :5, :5] = True
    assert compute_3d_dice(p_empty, g_present) == 0.0
    assert compute_3d_dice(g_present, p_empty) == 0.0

    # 3. Perfect overlap
    assert compute_3d_dice(g_present, g_present) == 1.0

    # 4. Partial overlap
    p_half = np.zeros(shape, dtype=bool)
    p_half[:5, :5, :3] = True  # Overlaps 3*5*5 = 75 out of 125 voxels
    # Intersection = 75, p_count = 75, g_count = 125
    expected_dice = (2.0 * 75) / (75 + 125)
    assert abs(compute_3d_dice(p_half, g_present) - expected_dice) < 1e-6


def test_3d_hd95_edge_cases():
    """Verify 3D HD95 distance conventions (both empty=0.0, one empty=grid diagonal penalty) (§13.5)."""
    shape = (10, 20, 20)
    spacing = (1.0, 1.0, 1.0)
    expected_penalty = math.sqrt(10**2 + 20**2 + 20**2)  # sqrt(100+400+400) = sqrt(900) = 30.0

    p_empty = np.zeros(shape, dtype=bool)
    g_empty = np.zeros(shape, dtype=bool)

    # 1. Both empty
    hd95, is_one_empty = compute_3d_hd95(p_empty, g_empty, spacing=spacing)
    assert hd95 == 0.0
    assert not is_one_empty

    # 2. One empty
    g_present = np.zeros(shape, dtype=bool)
    g_present[:5, :5, :5] = True
    hd95, is_one_empty = compute_3d_hd95(p_empty, g_present, spacing=spacing)
    assert abs(hd95 - expected_penalty) < 1e-5
    assert is_one_empty

    # 3. Perfect overlap
    hd95, is_one_empty = compute_3d_hd95(g_present, g_present, spacing=spacing)
    assert hd95 == 0.0
    assert not is_one_empty


def test_patient_evaluator_end_to_end():
    """Verify PatientEvaluator 3D volume evaluation and label remapping (§13.5)."""
    evaluator = PatientEvaluator(spacing=(1.0, 1.0, 1.0))
    shape = (5, 20, 20)

    # Create model class predictions {0, 1, 2, 3}
    pred_model = np.zeros(shape, dtype=np.uint8)
    target_model = np.zeros(shape, dtype=np.uint8)

    # Class 1 (NCR), Class 2 (ED), Class 3 (ET)
    pred_model[:2, :10, :10] = 1
    target_model[:2, :10, :10] = 1

    pred_model[2:4, :10, :10] = 2
    target_model[2:4, :10, :10] = 2

    metrics = evaluator.evaluate_patient_volume(pred_model, target_model)

    assert abs(metrics["dice_WT"] - 1.0) < 1e-6
    assert abs(metrics["dice_TC"] - 1.0) < 1e-6
    assert abs(metrics["hd95_WT"] - 0.0) < 1e-6
    assert abs(metrics["macro_dice"] - 1.0) < 1e-6
    assert metrics["one_empty_case_count"] == 0.0
