"""
CAMFS M1 — Federation Phase Controller & State Machine
======================================================

Implements §5.5, §6.3, §7, and §8.6 of the CAMFS M1 Specification.
Manages the global FL lifecycle state machine:
  PHASE1 -> FROZEN -> PHASE2 -> RELEASED

Handles:
  - Phase 1 contrastive drift tracking & convergence check (ε = 0.01, patience K = 5 after round 20)
  - Phase transition freeze procedure: eval mode, requires_grad=False, encoder parameter hashing
  - Phase 2 validation tracking, early stopping (patience 10, threshold 1e-4), and model selection
"""

from __future__ import annotations

import hashlib
import io
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from src.models.encoder import UnimodalEncoder


class FederatedPhaseState:
    PHASE1 = "PHASE1"
    FROZEN = "FROZEN"
    PHASE2 = "PHASE2"
    RELEASED = "RELEASED"


class PhaseController:
    """
    FL Lifecycle Phase Controller & State Machine (§5.5, §7).

    Parameters:
        min_p1_rounds: Minimum Phase 1 rounds before convergence check (default: 20).
        max_p1_rounds: Maximum Phase 1 rounds hard cap (default: 100).
        drift_epsilon: Prototype drift convergence threshold (default: 0.01).
        drift_patience: Consecutive rounds under drift threshold required (default: 5).
        min_p2_rounds: Minimum Phase 2 rounds (default: 20).
        max_p2_rounds: Maximum Phase 2 rounds hard cap (default: 100).
        p2_patience: Phase 2 early stopping patience (default: 10).
        p2_improvement_threshold: Minimum macro-Dice improvement threshold (default: 1e-4).
    """

    def __init__(
        self,
        min_p1_rounds: int = 20,
        max_p1_rounds: int = 100,
        drift_epsilon: float = 0.01,
        drift_patience: int = 5,
        min_p2_rounds: int = 20,
        max_p2_rounds: int = 100,
        p2_patience: int = 10,
        p2_improvement_threshold: float = 1e-4,
    ) -> None:
        self.state = FederatedPhaseState.PHASE1
        self.min_p1_rounds = min_p1_rounds
        self.max_p1_rounds = max_p1_rounds
        self.drift_epsilon = drift_epsilon
        self.drift_patience = drift_patience

        self.min_p2_rounds = min_p2_rounds
        self.max_p2_rounds = max_p2_rounds
        self.p2_patience = p2_patience
        self.p2_improvement_threshold = p2_improvement_threshold

        # Phase 1 tracking
        self.p1_drift_history: List[float] = []
        self.p1_consecutive_low_drift_count = 0

        # Phase 2 tracking per track
        self.p2_best_val_dice: Dict[str, float] = {}
        self.p2_best_round: Dict[str, int] = {}
        self.p2_no_improvement_count: Dict[str, int] = {}

    def compute_prototype_drift(
        self,
        current_prototypes: Dict[str, torch.Tensor],
        previous_prototypes: Dict[str, torch.Tensor],
    ) -> float:
        """
        Compute mean L2 prototype drift across modalities between rounds (§6.3).

        Formula:
            MeanDrift = 1 / |M| sum_{m} || Proto_m^(t) - Proto_m^(t-1) ||_2
        """
        if not previous_prototypes:
            return 1.0

        drifts = []
        for m, curr_p in current_prototypes.items():
            if m in previous_prototypes:
                prev_p = previous_prototypes[m]
                diff_norm = torch.linalg.vector_norm(curr_p - prev_p, ord=2).item()
                drifts.append(diff_norm)

        if not drifts:
            return 1.0

        mean_drift = float(sum(drifts) / len(drifts))
        return mean_drift

    def check_phase1_convergence(
        self,
        current_round: int,
        current_prototypes: Dict[str, torch.Tensor],
        previous_prototypes: Dict[str, torch.Tensor],
    ) -> Tuple[bool, float]:
        """
        Check if Phase 1 contrastive training has converged (§6.3).

        Returns:
            Tuple of (should_stop_phase1: bool, mean_drift: float)
        """
        if self.state != FederatedPhaseState.PHASE1:
            raise RuntimeError(f"Cannot check Phase 1 convergence in state '{self.state}'.")

        mean_drift = self.compute_prototype_drift(current_prototypes, previous_prototypes)
        self.p1_drift_history.append(mean_drift)

        if mean_drift < self.drift_epsilon:
            self.p1_consecutive_low_drift_count += 1
        else:
            self.p1_consecutive_low_drift_count = 0

        # Check stopping criteria after min_p1_rounds
        if current_round >= self.min_p1_rounds:
            if self.p1_consecutive_low_drift_count >= self.drift_patience:
                return True, mean_drift

        if current_round >= self.max_p1_rounds:
            return True, mean_drift

        return False, mean_drift

    def execute_freeze_procedure(
        self,
        encoders: Dict[str, UnimodalEncoder],
    ) -> Dict[str, str]:
        """
        Execute Phase Transition Freeze Procedure (§7).
        Sets encoders to eval mode, sets requires_grad=False, and computes parameter SHA-256 digests.

        Returns:
            Dict mapping modality 'T1' -> encoder_state_sha256_hex
        """
        if self.state != FederatedPhaseState.PHASE1:
            raise RuntimeError(f"Cannot execute freeze procedure from state '{self.state}'. Must be in PHASE1.")

        state_hashes = {}

        for m, encoder in encoders.items():
            # 1. Set to eval mode
            encoder.eval()

            # 2. Disable gradients for all parameters
            for param in encoder.parameters():
                param.requires_grad = False

            # 3. Serialize parameters & compute SHA-256 digest
            buf = io.BytesIO()
            torch.save(encoder.state_dict(), buf)
            buf.seek(0)
            hash_hex = hashlib.sha256(buf.read()).hexdigest()
            state_hashes[m] = hash_hex

        self.state = FederatedPhaseState.FROZEN
        return state_hashes

    def start_phase2(self) -> None:
        """Transition state from FROZEN to PHASE2."""
        if self.state != FederatedPhaseState.FROZEN:
            raise RuntimeError(f"Cannot start Phase 2 from state '{self.state}'. Must be FROZEN.")
        self.state = FederatedPhaseState.PHASE2

    def check_phase2_stopping(
        self,
        current_round: int,
        track_val_dices: Dict[str, float],
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Check Phase 2 validation improvement and early stopping (§8.6).

        Args:
            current_round: Current Phase 2 round number.
            track_val_dices: Dict mapping track_id ('S1', 'S2') -> validation macro-Dice.

        Returns:
            Tuple of (all_tracks_stopped: bool, per_track_improved: Dict[str, bool])
        """
        if self.state != FederatedPhaseState.PHASE2:
            raise RuntimeError(f"Cannot check Phase 2 stopping in state '{self.state}'.")

        per_track_improved = {}
        all_tracks_stopped = True

        for track_id, val_dice in track_val_dices.items():
            if track_id not in self.p2_best_val_dice:
                self.p2_best_val_dice[track_id] = val_dice
                self.p2_best_round[track_id] = current_round
                self.p2_no_improvement_count[track_id] = 0
                per_track_improved[track_id] = True
            else:
                if track_id not in self.p2_no_improvement_count:
                    self.p2_no_improvement_count[track_id] = 0
                best_dice = self.p2_best_val_dice[track_id]
                # Check improvement threshold (> 1e-4)
                if val_dice > best_dice + self.p2_improvement_threshold:
                    self.p2_best_val_dice[track_id] = val_dice
                    self.p2_best_round[track_id] = current_round
                    self.p2_no_improvement_count[track_id] = 0
                    per_track_improved[track_id] = True
                else:
                    self.p2_no_improvement_count[track_id] += 1
                    per_track_improved[track_id] = False

            # Check if this track should stop
            track_stopped = False
            if current_round >= self.min_p2_rounds:
                if self.p2_no_improvement_count.get(track_id, 0) >= self.p2_patience:
                    track_stopped = True

            if current_round >= self.max_p2_rounds:
                track_stopped = True

            if not track_stopped:
                all_tracks_stopped = False

        if all_tracks_stopped:
            self.state = FederatedPhaseState.RELEASED

        return all_tracks_stopped, per_track_improved
