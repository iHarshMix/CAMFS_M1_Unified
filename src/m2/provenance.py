"""
CAMFS M2 — Cryptographic Provenance Ledger
==========================================

Implements §5.6, §10.7 of the CAMFS M2 Specification:
Append-only JSONL provenance ledger tracking the complete lineage of knowledge transfer
from Hospital H1 to Hospital H3.
Maintains a cryptographic SHA-256 hash chain over all life-cycle events.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class ProvenanceLedger:
    """
    Append-only SHA-256 chained audit ledger.
    """

    def __init__(self, ledger_path: Union[str, Path]) -> None:
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        """Reads the hash of the last record in the ledger, or returns genesis hash."""
        if not self.ledger_path.exists() or self.ledger_path.stat().st_size == 0:
            return "0000000000000000000000000000000000000000000000000000000000000000"

        last_line = ""
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()

        if not last_line:
            return "0000000000000000000000000000000000000000000000000000000000000000"

        try:
            record = json.loads(last_line)
            return record.get("record_hash", "")
        except Exception:
            return "0000000000000000000000000000000000000000000000000000000000000000"

    def record_event(
        self,
        event_type: str,
        actor_hospital: str,
        grant_id: str,
        payload: Dict[str, Any],
    ) -> str:
        """
        Appends a new signed event to the ledger with cryptographic linkage.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        payload_str = json.dumps(payload, sort_keys=True)
        payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        # Chain linkage: SHA-256(prev_hash || timestamp || event_type || payload_hash)
        chain_input = f"{self._last_hash}:{timestamp}:{event_type}:{actor_hospital}:{payload_hash}"
        record_hash = hashlib.sha256(chain_input.encode("utf-8")).hexdigest()

        record = {
            "record_hash": record_hash,
            "prev_hash": self._last_hash,
            "timestamp": timestamp,
            "event_type": event_type,
            "actor_hospital": actor_hospital,
            "grant_id": grant_id,
            "payload_hash": payload_hash,
            "payload": payload,
        }

        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self._last_hash = record_hash
        return record_hash

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verifies the cryptographic integrity of the entire ledger chain from genesis.
        """
        if not self.ledger_path.exists():
            return True, None

        prev_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                record = json.loads(line.strip())

                if record.get("prev_hash") != prev_hash:
                    return False, f"Broken chain at line {line_idx+1}: prev_hash mismatch"

                payload_str = json.dumps(record.get("payload", {}), sort_keys=True)
                payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
                if payload_hash != record.get("payload_hash"):
                    return False, f"Tampered payload at line {line_idx+1}"

                chain_input = f"{prev_hash}:{record['timestamp']}:{record['event_type']}:{record['actor_hospital']}:{payload_hash}"
                expected_hash = hashlib.sha256(chain_input.encode("utf-8")).hexdigest()
                if expected_hash != record.get("record_hash"):
                    return False, f"Tampered record hash at line {line_idx+1}"

                prev_hash = record.get("record_hash")

        return True, None
