"""
CAMFS M1 — Governance Policy Engine
====================================

Implements §5.3 and §9 of the CAMFS M1 Specification.
Manages versioned, exogenous, non-learned policy manifests (`M1_PRIMARY_V1`),
computes SHA-256 manifest digests, compiles closed consent cohorts, and enforces
the Track Receive Safety Rule and Send-Gated Track Routing rules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


class PolicyManager:
    """
    Governance policy manager and cohort compiler (§5.3, §9).

    Parameters:
        policy_path: Path to policy JSON file, e.g., 'configs/policy_M1_PRIMARY_V1.json'.
    """

    def __init__(self, policy_path: Union[str, Path]) -> None:
        self.policy_path = Path(policy_path)
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Policy manifest file not found at: {self.policy_path}")

        raw_bytes = self.policy_path.read_bytes()
        self.digest_hex = hashlib.sha256(raw_bytes).hexdigest()

        with open(self.policy_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.version: str = self.manifest["policy_version"]
        self.modalities: List[str] = self.manifest["modalities"]
        self.hospitals: List[str] = self.manifest["hospitals"]
        self.ownership: Dict[str, List[str]] = self.manifest["ownership"]
        self.send_subsets: Dict[str, List[str]] = self.manifest["send_subsets"]
        self.encoder_cohorts: Dict[str, List[str]] = self.manifest["encoder_cohorts"]
        self.track_cohorts: Dict[str, Dict] = self.manifest["track_cohorts"]

    def get_digest(self) -> str:
        """Return the SHA-256 digest of the loaded policy manifest."""
        return self.digest_hex

    def get_encoder_cohort(self, modality: str) -> List[str]:
        """Return allowed contributing hospital IDs for unimodal encoder m."""
        if modality not in self.encoder_cohorts:
            raise KeyError(f"Modality '{modality}' not found in policy encoder cohorts.")
        return list(self.encoder_cohorts[modality])

    def get_track_info(self, track_id: str) -> Dict:
        """Return track information dictionary (modalities, contributors, authorized_receivers)."""
        if track_id not in self.track_cohorts:
            raise KeyError(f"Track '{track_id}' not found in policy track cohorts.")
        return self.track_cohorts[track_id]

    def verify_cohort_closure(self, modality: str, proposed_contributors: List[str]) -> bool:
        """
        Verify Closed(m, κ) rule (§5.3):
        Returns True iff proposed_contributors exactly matches the manifest's closed cohort.
        """
        allowed = set(self.get_encoder_cohort(modality))
        proposed = set(proposed_contributors)
        return proposed == allowed

    def verify_track_receive_safety(self, recipient_id: str, track_id: str) -> bool:
        """
        Enforce Track Receive Safety Rule (§5.3):
        Recipient i may load track S iff:
          1. S ⊆ O(i)  (recipient physically owns all modalities in track S)
          2. recipient_id is in track S's authorized_receivers list.

        Returns True if safe, False otherwise.
        """
        if recipient_id not in self.ownership:
            return False

        track_info = self.get_track_info(track_id)
        track_modalities = set(track_info["modalities"])
        owned_modalities = set(self.ownership[recipient_id])

        # Condition 1: S ⊆ O(i)
        if not track_modalities.issubset(owned_modalities):
            return False

        # Condition 2: Explicitly listed as authorized receiver
        authorized_receivers = set(track_info["authorized_receivers"])
        if recipient_id not in authorized_receivers:
            return False

        return True

    def verify_send_gated_routing(self, contributor_id: str, track_id: str) -> bool:
        """
        Enforce Send-Gated Track Routing (§9):
        Client i may contribute to track S' aggregate iff:
          1. Send(i) == S' (native track match), OR
          2. R_contribute(i, S') == 1 (explicit opt-in contribution with masking).

        Returns True if allowed, False otherwise.
        """
        if contributor_id not in self.send_subsets:
            return False

        track_info = self.get_track_info(track_id)
        track_modalities = set(track_info["modalities"])
        send_modalities = set(self.send_subsets[contributor_id])

        # Check explicit contributors list in track cohort
        allowed_contributors = set(track_info["contributors"])
        if contributor_id not in allowed_contributors:
            return False

        # Rule 1: Native track match (Send(i) == S')
        if send_modalities == track_modalities:
            return True

        # Rule 2: Explicit opt-in R_contribute(i, S') == 1
        multi_track = self.manifest.get("multi_track_contribution", {}).get("R_contribute", {})
        client_opts = multi_track.get(contributor_id, {})
        if client_opts.get(track_id, 0) == 1:
            owned_modalities = set(self.ownership[contributor_id])
            if track_modalities.issubset(owned_modalities):
                return True

        return False
