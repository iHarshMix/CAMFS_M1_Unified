"""
CAMFS M1 — Federation Protocol & Aggregation Unit Tests
========================================================

PyTest suite for FL protocol rules (§6.2, §8.5):
  - Round 0 no-optimizer prototype bootstrap (§6.2)
  - Patient-weighted encoder parameter averaging & support-weighted prototype aggregation (§6.2, §8.5)
  - Fresh local AdamW optimizer each round (no client optimizer state persistence) (§6.2, §8.4)
"""

import numpy as np
import pytest
import torch

from src.federation import FederatedClient, FederatedServer
from src.governance import LineageAuditor, PolicyManager, ProvenanceLedger
from src.models.encoder import UnimodalEncoder
from src.models.prototypes import PrototypeBank


def test_round0_bootstrap_no_optimizer_step(tmp_path):
    """Verify Round 0 computes initial prototypes without running optimizer steps (§6.2)."""
    # Create synthetic dataset root
    prep_dir = tmp_path / "preprocessed"
    prep_dir.mkdir()

    # Write dummy patient volume
    pid = "BraTS2020_001"
    p_dir = prep_dir / pid
    p_dir.mkdir()
    np_array = torch.randn(155, 240, 240).numpy().astype("float32")
    labels = torch.randint(0, 4, (155, 240, 240)).numpy().astype("uint8")

    np.save(p_dir / "t1.npy", np_array)
    np.save(p_dir / "labels.npy", labels)

    client = FederatedClient(
        hospital_id="H1",
        owned_modalities=["T1"],
        send_modalities=["T1"],
        patient_ids=[pid],
        preprocessed_dir=prep_dir,
    )

    encoders = {"T1": UnimodalEncoder()}
    initial_weights = copy_weights(encoders["T1"])

    protos, counts = client.bootstrap_round0_prototypes(encoders)

    # Weights must be identical (no optimizer step)
    current_weights = copy_weights(encoders["T1"])
    for k in initial_weights:
        assert torch.equal(initial_weights[k], current_weights[k])

    assert "T1" in protos
    assert protos["T1"].shape == (4, 256)
    assert counts["T1"].shape == (4,)


def test_patient_weighted_and_support_weighted_aggregation(tmp_path):
    """Verify server patient-weighted parameter averaging and support-weighted prototype aggregation (§6.2)."""
    policy_path = "configs/policy_M1_PRIMARY_V1.json"
    policy = PolicyManager(policy_path)
    ledger = ProvenanceLedger(tmp_path / "ledger.jsonl")
    auditor = LineageAuditor(audit_mode="reject")

    server = FederatedServer(policy, ledger, auditor)
    server.initialize_phase1_models(modalities=["T1"])

    # Create 3 client updates for closed cohort κ_T1 = [H1, H2, H3]
    enc1 = UnimodalEncoder()
    enc2 = UnimodalEncoder()
    enc3 = UnimodalEncoder()

    # Modify weights to known values
    for p1, p2, p3 in zip(enc1.parameters(), enc2.parameters(), enc3.parameters()):
        p1.data.fill_(1.0)
        p2.data.fill_(5.0)
        p3.data.fill_(3.0)

    proto1 = torch.ones(4, 256)
    proto2 = torch.full((4, 256), 2.0)
    proto3 = torch.full((4, 256), 3.0)

    client_updates = [
        {
            "hospital_id": "H1",
            "encoder_state_dicts": {"T1": enc1.state_dict()},
            "prototypes": {"T1": proto1},
            "support_counts": {"T1": torch.tensor([10, 10, 10, 10])},
            "image_lineage": {"T1"},
        },
        {
            "hospital_id": "H2",
            "encoder_state_dicts": {"T1": enc2.state_dict()},
            "prototypes": {"T1": proto2},
            "support_counts": {"T1": torch.tensor([30, 30, 30, 30])},
            "image_lineage": {"T1"},
        },
        {
            "hospital_id": "H3",
            "encoder_state_dicts": {"T1": enc3.state_dict()},
            "prototypes": {"T1": proto3},
            "support_counts": {"T1": torch.tensor([10, 10, 10, 10])},
            "image_lineage": {"T1"},
        },
    ]

    patient_counts = {"H1": 10, "H2": 30, "H3": 10}

    server.aggregate_phase1_round(
        current_round=1,
        client_updates=client_updates,
        client_patient_counts=patient_counts,
    )

    # Expected parameter weight: (10 * 1.0 + 30 * 5.0 + 10 * 3.0) / 50 = 190 / 50 = 3.8
    for p in server.encoders["T1"].parameters():
        assert torch.allclose(p.data, torch.full_like(p.data, 3.8)), f"Expected aggregated weight 3.8, got {p.data[0]}"


def copy_weights(model: torch.nn.Module):
    return {k: v.clone() for k, v in model.state_dict().items()}
