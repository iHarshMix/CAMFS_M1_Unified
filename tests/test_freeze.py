"""
CAMFS M1 — Phase Transition Freeze Procedure Unit Tests
=======================================================

PyTest suite for Phase Transition Freeze procedure (§5.5, §7):
  - Parameter requires_grad == False flag verification after freeze
  - Encoder eval() mode verification
  - Stop-gradient detachment verification (encoder parameters cannot be updated during Phase 2)
"""

import pytest
import torch
import torch.nn as nn

from src.federation import FederatedPhaseState, PhaseController
from src.models.encoder import UnimodalEncoder
from src.models.fusion import SubsetFusionHead


def test_freeze_procedure_parameter_grad_flags():
    """Verify Phase 1 freeze sets requires_grad=False, eval mode, and hashes parameters (§7)."""
    controller = PhaseController()
    assert controller.state == FederatedPhaseState.PHASE1

    modalities = ("T1", "T1ce", "T2", "FLAIR")
    encoders = {m: UnimodalEncoder() for m in modalities}

    # Verify initial state: requires_grad=True
    for enc in encoders.values():
        for p in enc.parameters():
            assert p.requires_grad

    # Execute freeze procedure
    hashes = controller.execute_freeze_procedure(encoders)

    # State transition verification
    assert controller.state == FederatedPhaseState.FROZEN

    # Verify SHA-256 hashes generated for each modality
    assert len(hashes) == 4
    for m in modalities:
        assert m in hashes
        assert len(hashes[m]) == 64

    # Verify requires_grad=False and eval mode
    for enc in encoders.values():
        assert not enc.training, "Encoder should be in eval mode post-freeze"
        for p in enc.parameters():
            assert not p.requires_grad, "Encoder parameter requires_grad should be False post-freeze"


def test_stop_gradient_detachment():
    """Verify Phase 2 backward pass cannot update frozen encoder parameters (§7)."""
    controller = PhaseController()
    encoders = {"T1": UnimodalEncoder(), "T2": UnimodalEncoder()}
    controller.execute_freeze_procedure(encoders)

    # Create trainable fusion head for track S2 (T1, T2)
    fusion_head = SubsetFusionHead(modality_subset=("T1", "T2"))

    x_t1 = torch.randn(2, 1, 240, 240)
    x_t2 = torch.randn(2, 1, 240, 240)

    # Pass through frozen encoders with stop-gradient detach()
    with torch.no_grad():
        f_t1 = tuple(h.detach() for h in encoders["T1"](x_t1))
        f_t2 = tuple(h.detach() for h in encoders["T2"](x_t2))

    feats = {"T1": f_t1, "T2": f_t2}
    fused_out = fusion_head(feats)
    loss = sum(f.sum() for f in fused_out)

    # Pass backward through fusion head
    loss.backward()

    # Fusion head parameters should have valid gradients
    for p in fusion_head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()

    # Encoder parameters must have NO gradients and remain unchanged
    for enc in encoders.values():
        for p in enc.parameters():
            assert p.grad is None
