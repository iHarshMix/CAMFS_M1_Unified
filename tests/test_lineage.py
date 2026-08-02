"""
CAMFS M1 — Cryptographic Ledger & Lineage Audit Unit Tests
==========================================================

PyTest suite for Phase 5 provenance & audit:
  - ProvenanceLedger SHA-256 chain construction & integrity verification (§5.6)
  - Tamper detection in provenance chain
  - LineageAuditor Encoder Lineage Rule: ImageLineage(θ_{E_m}) ⊆ {m} (§14.4)
  - LineageAuditor Track Lineage Rule: ImageLineage(θ_{F_S} ∪ θ_{D_S}) ⊆ S (§14.4)
  - Primary rejection mode ("reject") vs DisentAFL shadow mode ("shadow")
"""

import json
import pytest
from pathlib import Path

from src.governance import LineageAuditor, ProvenanceLedger


def test_provenance_ledger_chaining_and_verification(tmp_path):
    """Verify SHA-256 chained JSONL provenance log creation and verification (§5.6)."""
    ledger_path = tmp_path / "test_run_ledger.jsonl"
    ledger = ProvenanceLedger(ledger_path)

    # Record sequence of lifecycle events
    e1 = ledger.record_event("POLICY_MANIFEST", {"policy_version": "M1_PRIMARY_V1", "digest": "abcd1234"})
    e2 = ledger.record_event("PHASE1_AGGREGATION", {"round": 1, "cohort": "T1", "contributors": ["H1", "H2", "H3"]})
    e3 = ledger.record_event("PHASE_TRANSITION", {"round": 25, "drift": 0.008, "state": "FROZEN"})
    e4 = ledger.record_event("TRACK_CREATION", {"track_id": "S1", "modalities": ["T1", "T1ce", "T2", "FLAIR"]})

    # Verify ledger integrity
    valid, msg, last_hash = ProvenanceLedger.verify_chain_file(ledger_path)
    assert valid, f"Ledger verification failed: {msg}"
    assert last_hash == e4["record_hash"]


def test_provenance_ledger_tamper_detection(tmp_path):
    """Verify SHA-256 chain detects tampered records in ledger file (§5.6)."""
    ledger_path = tmp_path / "tampered_ledger.jsonl"
    ledger = ProvenanceLedger(ledger_path)

    ledger.record_event("POLICY_MANIFEST", {"digest": "1111"})
    ledger.record_event("PHASE1_AGGREGATION", {"round": 1})
    ledger.record_event("PHASE1_AGGREGATION", {"round": 2})

    # Read lines and tamper with round 1 payload
    lines = ledger_path.read_text().splitlines()
    rec1 = json.loads(lines[1])
    rec1["payload"]["round"] = 999  # Tampered!
    lines[1] = json.dumps(rec1, sort_keys=True)
    ledger_path.write_text("\n".join(lines) + "\n")

    valid, msg, _ = ProvenanceLedger.verify_chain_file(ledger_path)
    assert not valid, "Tampered ledger passed verification unexpectedly"
    assert "mismatch" in msg.lower() or "line 2" in msg.lower()


def test_lineage_auditor_primary_mode():
    """Verify LineageAuditor packet rejection in primary CAMFS mode ('reject') (§14.4)."""
    auditor = LineageAuditor(audit_mode="reject")

    # 1. Valid Encoder Packet: Lineage matches target modality
    approved, err = auditor.audit_encoder_packet(
        packet_id="pkt_001",
        source_hospital="H1",
        target_modality="T1",
        image_lineage_set={"T1"},
    )
    assert approved
    assert err is None

    # 2. Invalid Encoder Packet: ImageLineage contains forbidden modality
    approved, err = auditor.audit_encoder_packet(
        packet_id="pkt_002",
        source_hospital="H2",
        target_modality="T1",
        image_lineage_set={"T1", "T1ce"},  # Leak!
    )
    assert not approved
    assert err is not None
    assert "T1ce" in err

    # 3. Valid Track Packet: Lineage ⊆ Track modalities
    approved, err = auditor.audit_track_packet(
        packet_id="pkt_003",
        source_hospital="H2",
        track_id="S2",
        track_modalities={"T1", "T2"},
        image_lineage_set={"T1", "T2"},
    )
    assert approved

    # 4. Invalid Track Packet: Lineage contains unapproved modality
    approved, err = auditor.audit_track_packet(
        packet_id="pkt_004",
        source_hospital="H2",
        track_id="S2",
        track_modalities={"T1", "T2"},
        image_lineage_set={"T1", "T2", "FLAIR"},  # Leak!
    )
    assert not approved
    assert "FLAIR" in err
    assert auditor.rejected_packets_count == 2


def test_lineage_auditor_shadow_mode():
    """Verify LineageAuditor shadow logging in DisentAFL B3 mode ('shadow') (§14.4)."""
    auditor = LineageAuditor(audit_mode="shadow")

    # Audit invalid track packet in shadow mode
    approved, err = auditor.audit_track_packet(
        packet_id="pkt_shadow_01",
        source_hospital="H3",
        track_id="S3",
        track_modalities={"T1", "FLAIR"},
        image_lineage_set={"T1", "FLAIR", "T1ce"},  # Non-compliant leakage!
    )

    # In shadow mode, packet is approved (returns True), but recorded in shadow_violations list
    assert approved
    assert err is not None
    assert len(auditor.shadow_violations) == 1
    assert auditor.shadow_violations[0]["packet_id"] == "pkt_shadow_01"
    assert auditor.rejected_packets_count == 0
