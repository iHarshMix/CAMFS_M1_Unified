"""
CAMFS M1 — Policy Engine & Governance Unit Tests
=================================================

PyTest suite for PolicyManager (§5.3, §9):
  - Manifest loading & SHA-256 digest computation
  - Closed consent cohort verification (Closed(m, κ))
  - Track Receive Safety Rule (S ⊆ O(i) and listed authorized_receivers)
  - Send-Gated Track Routing (Send(i) == S' vs R_contribute(i, S') opt-in masking)
"""

import pytest
from pathlib import Path

from src.governance import PolicyManager

POLICY_PATH = Path("configs/policy_M1_PRIMARY_V1.json")


def test_policy_manifest_loading_and_digest():
    """Verify policy manifest loading and SHA-256 digest generation (§5.3)."""
    policy = PolicyManager(POLICY_PATH)

    assert policy.version == "M1_PRIMARY_V1"
    digest = policy.get_digest()
    assert len(digest) == 64, f"Digest length mismatch: {len(digest)}"
    assert all(c in "0123456789abcdef" for c in digest)


def test_cohort_closure():
    """Verify Closed(m, κ) rule (§5.3)."""
    policy = PolicyManager(POLICY_PATH)

    # Valid closed cohorts
    assert policy.verify_cohort_closure("T1", ["H1", "H2", "H3"])
    assert policy.verify_cohort_closure("T1ce", ["H1"])
    assert policy.verify_cohort_closure("T2", ["H1", "H2"])
    assert policy.verify_cohort_closure("FLAIR", ["H1", "H3"])

    # Invalid proposed cohorts (policy violations)
    assert not policy.verify_cohort_closure("T1ce", ["H1", "H2"])  # H2 not allowed for T1ce
    assert not policy.verify_cohort_closure("FLAIR", ["H1", "H2", "H3"])  # H2 does not own FLAIR


def test_track_receive_safety_rule():
    """Verify Track Receive Safety Rule (§5.3): S ⊆ O(i) and authorized receiver listing."""
    policy = PolicyManager(POLICY_PATH)

    # Authorized track loads
    assert policy.verify_track_receive_safety("H1", "S1")  # H1 loads full 4-modality S1
    assert policy.verify_track_receive_safety("H2", "S2")  # H2 loads dual T1+T2 S2
    assert policy.verify_track_receive_safety("H3", "S3")  # H3 loads dual T1+FLAIR S3
    assert policy.verify_track_receive_safety("H4", "S4")  # H4 loads triple T1+T1ce+T2 S4

    # Unauthorized track loads (policy violations)
    assert not policy.verify_track_receive_safety("H3", "S1")  # H3 does not own T1ce or T2
    assert not policy.verify_track_receive_safety("H2", "S1")  # H2 does not own FLAIR
    assert not policy.verify_track_receive_safety("H3", "S2")  # H3 does not own T2
    assert not policy.verify_track_receive_safety("H1", "S2")  # H1 not authorized receiver for S2 cohort


def test_send_gated_routing():
    """Verify Send-Gated Track Routing rule (§9)."""
    policy = PolicyManager(POLICY_PATH)

    # Native track matches (Send(i) == S')
    assert policy.verify_send_gated_routing("H1", "S1")
    assert policy.verify_send_gated_routing("H2", "S2")
    assert policy.verify_send_gated_routing("H3", "S3")
    assert policy.verify_send_gated_routing("H4", "S4")

    # Explicit opt-in multi-track contribution (R_contribute(H1, S3) == 1 with masking)
    assert policy.verify_send_gated_routing("H1", "S3")

    # Rejected non-native contribution routes
    assert not policy.verify_send_gated_routing("H2", "S1")  # H2 cannot contribute to S1
    assert not policy.verify_send_gated_routing("H3", "S1")  # H3 cannot contribute to S1
