"""
CAMFS M1 — Federated Server Aggregation Module
===============================================

Implements §6.2 and §8.5 of the CAMFS M1 Specification.
Manages global federated aggregation:
  - Integrated with PolicyManager, ProvenanceLedger, and LineageAuditor
  - Phase 1 cohort encoder aggregation (patient-weighted parameter averaging)
  - Phase 1 support-weighted prototype aggregation
  - Phase 2 track fusion head & decoder aggregation (send-keyed patient-weighted parameter averaging)
  - Zero-support prototype retention
  - No server optimizer
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn

from src.governance.ledger import ProvenanceLedger
from src.governance.lineage_audit import LineageAuditor, PolicyViolationError
from src.governance.policy import PolicyManager
from src.models.decoder import UNetDecoder
from src.models.encoder import UnimodalEncoder
from src.models.fusion import SubsetFusionHead
from src.models.prototypes import PrototypeBank


class FederatedServer:
    """
    Federated Learning Server Orchestrator (§6.2, §8.5).

    Parameters:
        policy_manager: PolicyManager instance.
        ledger: ProvenanceLedger instance.
        auditor: LineageAuditor instance.
        device: Torch device (default: 'cpu').
    """

    def __init__(
        self,
        policy_manager: PolicyManager,
        ledger: ProvenanceLedger,
        auditor: LineageAuditor,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.policy = policy_manager
        self.ledger = ledger
        self.auditor = auditor
        self.device = torch.device(device)

        # Phase 1 global state
        self.encoders: Dict[str, UnimodalEncoder] = {}
        self.prototypes: Dict[str, torch.Tensor] = {}

        # Phase 2 global state
        self.fusion_heads: Dict[str, SubsetFusionHead] = {}
        self.decoders: Dict[str, UNetDecoder] = {}
        self.fused_prototypes: Dict[str, torch.Tensor] = {}

    def initialize_phase1_models(
        self,
        modalities: Sequence[str] = ("T1", "T1ce", "T2", "FLAIR"),
    ) -> None:
        """Initialize global Phase 1 unimodal encoders and zero prototypes."""
        for m in modalities:
            enc = UnimodalEncoder(in_channels=1).to(self.device)
            self.encoders[m] = enc
            self.prototypes[m] = torch.zeros(4, 256, device=self.device)

    def initialize_phase2_track(
        self,
        track_id: str,
        track_modalities: Sequence[str],
    ) -> None:
        """Initialize global Phase 2 fusion head, decoder, and zero fused prototypes for track S."""
        fusion = SubsetFusionHead(modality_subset=track_modalities).to(self.device)
        decoder = UNetDecoder(num_classes=4).to(self.device)

        self.fusion_heads[track_id] = fusion
        self.decoders[track_id] = decoder
        self.fused_prototypes[track_id] = torch.zeros(4, 256, device=self.device)

    def aggregate_phase1_round(
        self,
        current_round: int,
        client_updates: List[Dict],
        client_patient_counts: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate Phase 1 unimodal encoders and prototypes across consent cohorts (§6.2).

        Args:
            current_round: Current Phase 1 round number.
            client_updates: List of client update dicts from FederatedClient.train_phase1_round.
            client_patient_counts: Dict mapping hospital_id ('H1') -> total patient count N_i.

        Returns:
            Dict mapping modality 'T1' -> aggregated prototypes tensor (4, 256).
        """
        # Group updates by modality
        modality_updates: Dict[str, List[Dict]] = {m: [] for m in self.policy.modalities}

        for update in client_updates:
            hid = update["hospital_id"]
            lineage = update["image_lineage"]

            for m, state_dict in update["encoder_state_dicts"].items():
                lineage_set = lineage[m] if isinstance(lineage, dict) and m in lineage else (lineage if isinstance(lineage, set) else {m})
                # 1. Audit Encoder Lineage Rule (§14.4)
                approved, err = self.auditor.audit_encoder_packet(
                    packet_id=f"p1_r{current_round}_{hid}_{m}",
                    source_hospital=hid,
                    target_modality=m,
                    image_lineage_set=lineage_set,
                )

                if not approved:
                    raise PolicyViolationError(f"Phase 1 update rejected: {err}")

                modality_updates[m].append({
                    "hospital_id": hid,
                    "state_dict": state_dict,
                    "proto": update["prototypes"].get(m),
                    "support_count": update["support_counts"].get(m),
                    "n_patients": client_patient_counts.get(hid, 1),
                })

        # Perform parameter and prototype aggregation per modality cohort
        for m, updates in modality_updates.items():
            if not updates:
                continue

            # Verify Closed(m, κ) rule (§5.3)
            contributing_hospitals = [u["hospital_id"] for u in updates]
            if not self.policy.verify_cohort_closure(m, contributing_hospitals):
                raise PolicyViolationError(
                    f"Cohort closure violation for modality '{m}': "
                    f"Proposed contributors {contributing_hospitals} do not match manifest closed cohort {self.policy.get_encoder_cohort(m)}."
                )

            # 2. Patient-weighted encoder parameter averaging (§6.2)
            total_cohort_patients = sum(u["n_patients"] for u in updates)
            if total_cohort_patients > 0:
                avg_state_dict = {}
                first_sd = updates[0]["state_dict"]

                for key in first_sd.keys():
                    weighted_sum = sum(
                        (u["n_patients"] / float(total_cohort_patients)) * u["state_dict"][key].float()
                        for u in updates
                    )
                    avg_state_dict[key] = weighted_sum

                self.encoders[m].load_state_dict(avg_state_dict)

            # 3. Support-weighted prototype aggregation (§6.2)
            valid_protos = [u["proto"] for u in updates if u["proto"] is not None]
            valid_counts = [u["support_count"] for u in updates if u["support_count"] is not None]

            if valid_protos and valid_counts:
                prev_proto = self.prototypes.get(m)
                agg_proto = PrototypeBank.aggregate_prototypes(
                    client_prototypes=valid_protos,
                    client_support_counts=valid_counts,
                    previous_prototypes=prev_proto,
                )
                self.prototypes[m] = agg_proto

        # Record event in provenance ledger
        self.ledger.record_event(
            "PHASE1_AGGREGATION",
            {
                "round": current_round,
                "contributing_hospitals": {m: [u["hospital_id"] for u in modality_updates[m]] for m in modality_updates},
            },
        )

        return self.prototypes

    def aggregate_phase2_round(
        self,
        current_round: int,
        track_id: str,
        client_updates: List[Dict],
        client_patient_counts: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate Phase 2 track-isolated fusion head and decoder parameters (§8.5).

        Args:
            current_round: Current Phase 2 round number.
            track_id: Track subset identifier ('S1', 'S2').
            client_updates: List of client update dicts from FederatedClient.train_phase2_round.
            client_patient_counts: Dict mapping hospital_id -> total patient count N_i.

        Returns:
            Dict containing aggregated fused_prototypes for track_id.
        """
        if track_id not in self.fusion_heads:
            raise KeyError(f"Track '{track_id}' not initialized on server.")

        track_info = self.policy.get_track_info(track_id)
        track_mods = set(track_info["modalities"])

        valid_updates = []
        valid_protos = []
        valid_counts = []

        for update in client_updates:
            hid = update["hospital_id"]
            lineage = update["image_lineage"]

            # 1. Audit Track Lineage Rule (§14.4)
            approved, err = self.auditor.audit_track_packet(
                packet_id=f"p2_r{current_round}_{hid}_{track_id}",
                source_hospital=hid,
                track_id=track_id,
                track_modalities=track_mods,
                image_lineage_set=lineage,
            )

            if not approved:
                raise PolicyViolationError(f"Phase 2 track update rejected: {err}")

            # 2. Check Send-Gated Track Routing (§9)
            if not self.policy.verify_send_gated_routing(hid, track_id):
                raise PolicyViolationError(f"Send-gated routing violation: Hospital '{hid}' not permitted to contribute to track '{track_id}'.")

            valid_updates.append({
                "hospital_id": hid,
                "fusion_sd": update["fusion_state_dict"],
                "decoder_sd": update["decoder_state_dict"],
                "n_patients": client_patient_counts.get(hid, 1),
            })

            valid_protos.append(update["fused_prototypes"])
            valid_counts.append(update["support_counts"])

        # 3. Patient-weighted fusion head and decoder parameter averaging (§8.5)
        total_track_patients = sum(u["n_patients"] for u in valid_updates)
        if total_track_patients > 0:
            # Average fusion head
            avg_fusion_sd = {}
            first_fusion_sd = valid_updates[0]["fusion_sd"]
            for key in first_fusion_sd.keys():
                w_sum = sum(
                    (u["n_patients"] / float(total_track_patients)) * u["fusion_sd"][key].float()
                    for u in valid_updates
                )
                avg_fusion_sd[key] = w_sum
            self.fusion_heads[track_id].load_state_dict(avg_fusion_sd)

            # Average decoder
            avg_decoder_sd = {}
            first_decoder_sd = valid_updates[0]["decoder_sd"]
            for key in first_decoder_sd.keys():
                w_sum = sum(
                    (u["n_patients"] / float(total_track_patients)) * u["decoder_sd"][key].float()
                    for u in valid_updates
                )
                avg_decoder_sd[key] = w_sum
            self.decoders[track_id].load_state_dict(avg_decoder_sd)

        # 4. Support-weighted fused prototype aggregation (§8.3)
        if valid_protos and valid_counts:
            prev_fused = self.fused_prototypes.get(track_id)
            agg_fused = PrototypeBank.aggregate_prototypes(
                client_prototypes=valid_protos,
                client_support_counts=valid_counts,
                previous_prototypes=prev_fused,
            )
            self.fused_prototypes[track_id] = agg_fused

        # Record event in provenance ledger
        self.ledger.record_event(
            "TRACK_AGGREGATION",
            {
                "round": current_round,
                "track_id": track_id,
                "contributors": [u["hospital_id"] for u in valid_updates],
            },
        )

        return self.fused_prototypes
