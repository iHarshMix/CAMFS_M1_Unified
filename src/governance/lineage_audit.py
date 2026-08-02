"""
CAMFS M1 — Lineage Auditor & Policy Compliance Probe
=====================================================

Implements §14.4 of the CAMFS M1 Specification.
Provides a packet-level executable lineage auditor that tracks `image_lineage` sets
for every update packet.
Enforces:
  - Encoder Lineage Rule: ImageLineage(θ_{E_m}) ⊆ {m}
  - Track Lineage Rule:   ImageLineage(θ_{F_S} ∪ θ_{D_S}) ⊆ S
Supports both CAMFS primary mode ("reject") and DisentAFL shadow mode ("shadow").
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class PolicyViolationError(Exception):
    """Exception raised when a non-compliant update packet is rejected under audit_mode='reject'."""

    pass


class LineageAuditor:
    """
    Executable lineage auditor (§14.4).

    Parameters:
        audit_mode: Operating mode ('reject' for CAMFS primary, 'shadow' for DisentAFL B3).
    """

    def __init__(self, audit_mode: str = "reject") -> None:
        if audit_mode not in ("reject", "shadow"):
            raise ValueError(f"Invalid audit_mode '{audit_mode}'. Must be 'reject' or 'shadow'.")

        self.audit_mode = audit_mode
        self.accepted_packets_count = 0
        self.rejected_packets_count = 0
        self.shadow_violations: List[Dict] = []

    def audit_encoder_packet(
        self,
        packet_id: str,
        source_hospital: str,
        target_modality: str,
        image_lineage_set: Set[str],
    ) -> Tuple[bool, Optional[str]]:
        """
        Audit Phase 1 unimodal encoder update packet.

        Rule (§14.4): ImageLineage(θ_{E_m}) ⊆ {m}

        Args:
            packet_id: Unique identifier for update packet.
            source_hospital: Hospital sending update, e.g. 'H1'.
            target_modality: Intended unimodal encoder modality, e.g. 'T1'.
            image_lineage_set: Set of modalities whose image tensors shaped this update.

        Returns:
            Tuple of (is_approved, violation_reason)
        """
        allowed = {target_modality}
        unauthorized = image_lineage_set - allowed

        if unauthorized:
            msg = (
                f"Encoder Lineage Violation in packet '{packet_id}' from {source_hospital}: "
                f"Target modality is '{target_modality}', but lineage contains unauthorized modalities {unauthorized}."
            )

            violation_entry = {
                "packet_id": packet_id,
                "source_hospital": source_hospital,
                "target_type": "encoder",
                "target_key": target_modality,
                "lineage": list(image_lineage_set),
                "unauthorized": list(unauthorized),
                "reason": msg,
            }

            if self.audit_mode == "reject":
                self.rejected_packets_count += 1
                logger.error(msg)
                return False, msg
            else:
                # Shadow mode: record violation but do not reject
                self.accepted_packets_count += 1
                self.shadow_violations.append(violation_entry)
                logger.warning(f"[SHADOW AUDIT] {msg}")
                return True, msg

        self.accepted_packets_count += 1
        return True, None

    def audit_track_packet(
        self,
        packet_id: str,
        source_hospital: str,
        track_id: str,
        track_modalities: Set[str],
        image_lineage_set: Set[str],
    ) -> Tuple[bool, Optional[str]]:
        """
        Audit Phase 2 track fusion/decoder update packet.

        Rule (§14.4): ImageLineage(θ_{F_S} ∪ θ_{D_S}) ⊆ S

        Args:
            packet_id: Unique identifier for update packet.
            source_hospital: Hospital sending update, e.g. 'H2'.
            track_id: Track subset identifier, e.g. 'S2'.
            track_modalities: Set of modalities in track S, e.g. {'T1', 'T2'}.
            image_lineage_set: Set of modalities whose image tensors shaped this update.

        Returns:
            Tuple of (is_approved, violation_reason)
        """
        unauthorized = image_lineage_set - track_modalities

        if unauthorized:
            msg = (
                f"Track Lineage Violation in packet '{packet_id}' from {source_hospital} for track '{track_id}': "
                f"Track subset is {track_modalities}, but lineage contains unauthorized modalities {unauthorized}."
            )

            violation_entry = {
                "packet_id": packet_id,
                "source_hospital": source_hospital,
                "target_type": "track",
                "target_key": track_id,
                "lineage": list(image_lineage_set),
                "unauthorized": list(unauthorized),
                "reason": msg,
            }

            if self.audit_mode == "reject":
                self.rejected_packets_count += 1
                logger.error(msg)
                return False, msg
            else:
                self.accepted_packets_count += 1
                self.shadow_violations.append(violation_entry)
                logger.warning(f"[SHADOW AUDIT] {msg}")
                return True, msg

        self.accepted_packets_count += 1
        return True, None
