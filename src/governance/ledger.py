"""
CAMFS M1 — SHA-256 Chained Provenance Ledger
==============================================

Implements §5.6 of the CAMFS M1 Specification.
Provides an append-only, SHA-256 chained JSONL provenance log (`ledger.jsonl`).
Ensures complete cryptographic lineage tracking across Phase 1 contrastive training,
the phase transition freeze, Phase 2 track-isolated training, and checkpoint selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class ProvenanceLedger:
    """
    Cryptographic provenance ledger manager (§5.6).

    Parameters:
        ledger_path: Path to output JSONL file, e.g. 'outputs/ledgers/run_ledger.jsonl'.
    """

    GENESIS_PREV_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(self, ledger_path: Union[str, Path]) -> None:
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_hash = self.GENESIS_PREV_HASH

        # Resume last_hash if ledger file exists and is non-empty
        if self.ledger_path.exists() and self.ledger_path.stat().st_size > 0:
            valid, msg, last_h = self.verify_chain_file(self.ledger_path)
            if not valid:
                raise ValueError(f"Existing ledger file at {self.ledger_path} is corrupted: {msg}")
            self.last_hash = last_h

    @staticmethod
    def canonical_json(data: Dict[str, Any]) -> str:
        """
        Produce deterministic canonical JSON string (§5.6).
        Sorted keys, compact separators (',', ':'), UTF-8.
        """
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def compute_record_hash(prev_hash: str, record_body: Dict[str, Any]) -> str:
        """
        Compute record_hash = SHA256(prev_hash + CanonicalJSON(record_body)).
        """
        canonical_str = ProvenanceLedger.canonical_json(record_body)
        payload = f"{prev_hash}:{canonical_str}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def record_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Append a new signed record to the provenance log.

        Args:
            event_type: One of ('POLICY_MANIFEST', 'PHASE1_AGGREGATION', 'PHASE_TRANSITION',
                                'TRACK_CREATION', 'CHECKPOINT_SELECTED').
            payload: Dict of event data.

        Returns:
            The complete recorded ledger entry dict including record_hash and prev_hash.
        """
        body = {
            "event_type": event_type,
            "payload": payload,
            "prev_hash": self.last_hash,
        }

        record_hash = self.compute_record_hash(self.last_hash, body)
        entry = dict(body)
        entry["record_hash"] = record_hash

        # Append to JSONL file
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(self.canonical_json(entry) + "\n")

        self.last_hash = record_hash
        return entry

    @classmethod
    def verify_chain_file(cls, ledger_file: Union[str, Path]) -> Tuple[bool, str, str]:
        """
        Verify cryptographic integrity of a ledger JSONL file (§5.6).

        Returns:
            Tuple of (is_valid: bool, message: str, last_record_hash: str)
        """
        path = Path(ledger_file)
        if not path.exists():
            return False, f"File does not exist: {path}", ""

        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return True, "Empty ledger file", cls.GENESIS_PREV_HASH

        expected_prev_hash = cls.GENESIS_PREV_HASH

        for idx, line in enumerate(lines):
            try:
                record = json.loads(line)
            except Exception as e:
                return False, f"Line {idx+1} is not valid JSON: {e}", ""

            if "record_hash" not in record:
                return False, f"Line {idx+1} missing 'record_hash' field", ""
            if "prev_hash" not in record:
                return False, f"Line {idx+1} missing 'prev_hash' field", ""

            actual_record_hash = record["record_hash"]
            actual_prev_hash = record["prev_hash"]

            if actual_prev_hash != expected_prev_hash:
                return (
                    False,
                    f"Line {idx+1} prev_hash mismatch: expected {expected_prev_hash}, got {actual_prev_hash}",
                    "",
                )

            # Recompute record body hash
            body = {k: v for k, v in record.items() if k != "record_hash"}
            expected_record_hash = cls.compute_record_hash(expected_prev_hash, body)

            if actual_record_hash != expected_record_hash:
                return (
                    False,
                    f"Line {idx+1} record_hash mismatch: expected {expected_record_hash}, got {actual_record_hash}",
                    "",
                )

            expected_prev_hash = actual_record_hash

        return True, f"Verified {len(lines)} records cleanly", expected_prev_hash
