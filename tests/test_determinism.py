"""
CAMFS M1 — Determinism & Replay Reproducibility Unit Tests
==========================================================

PyTest suite for determinism rules (§4, §15.4):
  - Deterministic algorithm configuration via set_all_seeds()
  - Replay reproducibility: two runs with identical seed produce bit-exact loss and state hashes
"""

import hashlib
import io
import pytest
import torch

from src.models.encoder import UnimodalEncoder
from src.seed import set_deterministic


def get_model_hash(model: torch.nn.Module) -> str:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    return hashlib.sha256(buf.read()).hexdigest()


def test_determinism_replay_reproducibility():
    """Verify same seed produces identical model weights, loss values, and state hashes (§15.4)."""
    seed = 17

    # Run 1
    set_deterministic(seed)
    enc1 = UnimodalEncoder()
    hash1 = get_model_hash(enc1)
    x1 = torch.randn(2, 1, 240, 240)
    _, _, _, _, z1 = enc1(x1)
    loss1 = z1.sum().item()

    # Run 2
    set_deterministic(seed)
    enc2 = UnimodalEncoder()
    hash2 = get_model_hash(enc2)
    x2 = torch.randn(2, 1, 240, 240)
    _, _, _, _, z2 = enc2(x2)
    loss2 = z2.sum().item()

    # Determinism checks
    assert hash1 == hash2, "Model weight hashes differ between identical seed runs"
    assert loss1 == loss2, f"Loss values differ between identical seed runs: {loss1} vs {loss2}"
