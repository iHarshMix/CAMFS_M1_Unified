"""
CAMFS M2 — Core Test Suite
==========================

Unit tests, architectural bounds, Day-0 identities, and P0-P6 policy tests
for CAMFS M2 CDRD knowledge transfer.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.m2.adapter import CDRDAdapter
from src.m2.adapter_trainer import CDRDLoss
from src.m2.base_registry import (
    compute_canonical_base_hash,
    compute_rights_digest,
    verify_base_integrity,
)
from src.m2.calibration import RecipientGate
from src.m2.frozen_base import load_frozen_base
from src.m2.grant import (
    create_primary_grant,
    allow_build,
    allow_release,
    run_policy_suite,
)
from src.m2.provenance import ProvenanceLedger
from src.m2.splits import compute_m2_splits
from src.m2.teacher import (
    build_restricted_teacher,
    verify_teacher_day0_identity,
)

P1_CKPT = "outputs/checkpoints/camfs_primary__part1103__seed17/phase1_frozen.pt"
S3_CKPT = "outputs/checkpoints/camfs_primary__part1103__seed17/best_track_S3.pt"
PART_FILE = "outputs/partitions/partition_1103.json"
TEST_FILE = "outputs/partitions/h3_test_patients.json"


class TestM2Core(unittest.TestCase):

    def test_01_patient_splits_and_sacred_quarantine(self):
        """Verifies deterministic patient counts and strict test quarantine."""
        splits = compute_m2_splits(PART_FILE, TEST_FILE, split_seed=42)

        # H1 checks
        self.assertEqual(len(splits.h1.train_ids), 102)
        self.assertEqual(len(splits.h1.val_ids), 13)
        self.assertEqual(len(splits.h1.release_audit_ids), 12)
        self.assertEqual(splits.h1.total_count, 127)

        # Disjointness H1
        s_train = set(splits.h1.train_ids)
        s_val = set(splits.h1.val_ids)
        s_audit = set(splits.h1.release_audit_ids)
        self.assertEqual(len(s_train & s_val), 0)
        self.assertEqual(len(s_train & s_audit), 0)
        self.assertEqual(len(s_val & s_audit), 0)

        # H3 checks
        self.assertEqual(len(splits.h3.m1_train_ids), 52)
        self.assertEqual(len(splits.h3.m2_cal_ids), 6)
        self.assertEqual(len(splits.h3.m2_accept_val_ids), 6)
        self.assertEqual(len(splits.h3.sacred_test_ids), 50)
        self.assertEqual(splits.h3.total_count, 114)

        # Disjointness H3
        s3_train = set(splits.h3.m1_train_ids)
        s3_cal = set(splits.h3.m2_cal_ids)
        s3_accept = set(splits.h3.m2_accept_val_ids)
        s3_test = set(splits.h3.sacred_test_ids)

        self.assertEqual(len(s3_train & s3_cal), 0)
        self.assertEqual(len(s3_train & s3_accept), 0)
        self.assertEqual(len(s3_cal & s3_accept), 0)

        # SACRED TEST SET QUARANTINE: Must not overlap with any training or validation sets!
        self.assertEqual(len((s3_train | s3_cal | s3_accept) & s3_test), 0)

    def test_02_canonical_base_hash(self):
        """Verifies deterministic computation of base digest b and rights digest."""
        res = compute_canonical_base_hash(P1_CKPT, S3_CKPT)
        b = res["canonical_base_hash"]
        self.assertEqual(len(b), 64)
        rho_b = compute_rights_digest(b)
        self.assertEqual(len(rho_b), 64)
        # Integrity verification should pass
        self.assertTrue(verify_base_integrity(P1_CKPT, S3_CKPT, expected_base_hash=b))

    def test_03_frozen_base_interface(self):
        """Verifies frozen base eval mode, stop-gradient, and output shapes."""
        base = load_frozen_base(P1_CKPT, S3_CKPT, device="cpu")
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))

        x = {
            "T1": torch.randn(2, 1, 240, 240),
            "FLAIR": torch.randn(2, 1, 240, 240),
        }
        features, logits_b = base(x)
        self.assertEqual(len(features), 5)
        expected_channels = [32, 64, 128, 256, 256]
        expected_shapes = [(240, 240), (120, 120), (60, 60), (30, 30), (15, 15)]

        for i, f in enumerate(features):
            self.assertEqual(f.shape[1], expected_channels[i])
            self.assertEqual(f.shape[2:], expected_shapes[i])
            self.assertFalse(f.requires_grad)

        self.assertEqual(logits_b.shape, (2, 4, 240, 240))
        self.assertFalse(logits_b.requires_grad)

    def test_04_teacher_day0_identity(self):
        """Verifies that Net2Net widened teacher matches base on Day 0 within 1e-4."""
        base = load_frozen_base(P1_CKPT, S3_CKPT, device="cpu")
        teacher = build_restricted_teacher(P1_CKPT, S3_CKPT, device="cpu")
        diff = verify_teacher_day0_identity(teacher, base, device="cpu", tol=1e-4)
        self.assertLess(diff, 1e-4)

    def test_05_adapter_parameters_and_day0_identity(self):
        """Verifies adapter has exactly 16,344 parameters and produces zero delta initially."""
        adapter = CDRDAdapter()
        params = adapter.count_trainable_parameters()
        self.assertEqual(params, 16344, f"Expected 16,344 params, got {params}")

        features = (
            torch.randn(2, 32, 240, 240),
            torch.randn(2, 64, 120, 120),
            torch.randn(2, 128, 60, 60),
            torch.randn(2, 256, 30, 30),
            torch.randn(2, 256, 15, 15),
        )
        logits_b = torch.randn(2, 4, 240, 240)
        delta_ell, cand = adapter(features, logits_b)

        # Initialized to zero
        self.assertTrue(torch.allclose(delta_ell, torch.zeros_like(delta_ell), atol=1e-7))
        self.assertTrue(torch.allclose(cand, logits_b, atol=1e-7))

        # Output centering check with non-zero weights
        adapter.final_conv.weight.data.normal_()
        adapter.final_conv.bias.data.normal_()
        delta_rand, _ = adapter(features, logits_b)
        class_sum = torch.sum(delta_rand, dim=1)
        self.assertTrue(torch.allclose(class_sum, torch.zeros_like(class_sum), atol=1e-5))

    def test_06_cdrd_loss_and_gradients(self):
        """Verifies 4-term loss computation and adapter gradient backpropagation."""
        adapter = CDRDAdapter()
        loss_fn = CDRDLoss()

        features = (
            torch.randn(2, 32, 240, 240),
            torch.randn(2, 64, 120, 120),
            torch.randn(2, 128, 60, 60),
            torch.randn(2, 256, 30, 30),
            torch.randn(2, 256, 15, 15),
        )
        logits_b = torch.randn(2, 4, 240, 240)
        logits_t_cal = torch.randn(2, 4, 240, 240)
        targets = torch.randint(0, 4, (2, 240, 240))

        delta_ell, cand = adapter(features, logits_b)
        loss, loss_dict = loss_fn(delta_ell, cand, logits_t_cal, logits_b, targets)

        self.assertGreater(loss.item(), 0.0)
        for key in ["loss_total", "loss_seg", "loss_kd", "loss_delta", "loss_tv"]:
            self.assertIn(key, loss_dict)

        loss.backward()
        total_grad = sum(p.grad.norm().item() for p in adapter.parameters() if p.grad is not None)
        self.assertGreater(total_grad, 0.0)

    def test_07_recipient_gate_and_fallback(self):
        """Verifies RecipientGate initialization (alpha ~ 0.018) and fallback (d_g = 0)."""
        gate = RecipientGate(init_val=-4.0)
        self.assertTrue(torch.allclose(gate.alphas, torch.tensor([0.0179862] * 4), atol=1e-5))

        lb = torch.randn(2, 4, 60, 60)
        delta = torch.randn(2, 4, 60, 60)

        # Fallback test: d_g = 0 must produce bit-identical output to lb
        fallback = gate(lb, delta, d_g=0.0)
        self.assertTrue(torch.allclose(fallback, lb, atol=1e-7))

        # Augmented: d_g = 1 must modify output
        aug = gate(lb, delta, d_g=1.0)
        self.assertFalse(torch.allclose(aug, lb))

    def test_08_governance_policy_suite_p0_to_p6(self):
        """Executes full formal policy test suite P0-P6 (§12.1)."""
        res = compute_canonical_base_hash(P1_CKPT, S3_CKPT)
        base_hash = res["canonical_base_hash"]
        rights_digest = compute_rights_digest(base_hash)

        grant = create_primary_grant(base_hash, rights_digest)
        adapter = CDRDAdapter()
        manifest = {
            "grant_id": grant.grant_id,
            "donor_hospital": "H1",
            "recipient_hospital": "H3",
            "canonical_base_hash": base_hash,
            "rights_digest": rights_digest,
            "parameter_count": adapter.count_trainable_parameters(),
            "q_T": 1,
            "teacher_temperature": 1.0,
        }

        policy_results = run_policy_suite(grant, adapter.state_dict(), manifest)
        for test_name, passed in policy_results.items():
            self.assertTrue(passed, f"Policy test {test_name} failed!")

    def test_09_provenance_ledger_cryptographic_chain(self):
        """Verifies append-only SHA-256 chain verification in ProvenanceLedger."""
        with tempfile.TemporaryDirectory() as td:
            lpath = Path(td) / "test_ledger.jsonl"
            ledger = ProvenanceLedger(lpath)

            ledger.record_event("EV1", "H1", "G1", {"step": 1})
            ledger.record_event("EV2", "H1", "G1", {"step": 2})
            ledger.record_event("EV3", "H3", "G1", {"step": 3})

            chain_ok, err = ledger.verify_integrity()
            self.assertTrue(chain_ok, f"Integrity check failed: {err}")


if __name__ == "__main__":
    unittest.main()
