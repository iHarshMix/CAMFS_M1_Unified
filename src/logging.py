"""
CAMFS M1 — Metrics Logging & Checkpoint Management
===================================================

Implements Code Standards §6 and §15.3 of the CAMFS M1 Specification.
Provides:
  - Phase1CSVLogger: per-round unimodal contrastive metrics logging
  - Phase2CSVLogger: per-round track fusion training and 3D validation metrics logging
  - EvaluationCSVLogger: patient-level 3D test evaluation results logging
  - CheckpointManager: atomic checkpoint saving & loading with RNG state restoration
"""

from __future__ import annotations

import csv
import datetime
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class Phase1CSVLogger:
    """
    CSV logger for Phase 1 unimodal contrastive training rounds (Code Standards §6).

    Header:
        round,timestamp,hospital_id,modality,num_patients,info_nce_loss,prototype_drift_l2
    """

    HEADER = [
        "round",
        "timestamp",
        "hospital_id",
        "modality",
        "num_patients",
        "info_nce_loss",
        "prototype_drift_l2",
    ]

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        if not self.filepath.exists():
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADER)

    def log_round(
        self,
        round_num: int,
        hospital_id: str,
        modality: str,
        num_patients: int,
        info_nce_loss: float,
        prototype_drift_l2: float,
        timestamp: Optional[str] = None,
    ) -> None:
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        row = [
            round_num,
            timestamp,
            hospital_id,
            modality,
            num_patients,
            f"{info_nce_loss:.6f}",
            f"{prototype_drift_l2:.6f}",
        ]

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)


class Phase2CSVLogger:
    """
    CSV logger for Phase 2 track fusion training & 3D validation metrics (Code Standards §6).

    Header:
        round,timestamp,track_id,hospital_id,loss_dice_ce,loss_fused_align,total_loss,
        val_dice_ET,val_dice_TC,val_dice_WT,val_dice_macro,
        val_hd95_ET,val_hd95_TC,val_hd95_WT,val_hd95_macro
    """

    HEADER = [
        "round",
        "timestamp",
        "track_id",
        "hospital_id",
        "loss_dice_ce",
        "loss_fused_align",
        "total_loss",
        "val_dice_ET",
        "val_dice_TC",
        "val_dice_WT",
        "val_dice_macro",
        "val_hd95_ET",
        "val_hd95_TC",
        "val_hd95_WT",
        "val_hd95_macro",
    ]

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        if not self.filepath.exists():
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADER)

    def log_round(
        self,
        round_num: int,
        track_id: str,
        hospital_id: str,
        loss_dice_ce: float,
        loss_fused_align: float,
        total_loss: float,
        val_dice_dict: Dict[str, float],
        val_hd95_dict: Dict[str, float],
        timestamp: Optional[str] = None,
    ) -> None:
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        row = [
            round_num,
            timestamp,
            track_id,
            hospital_id,
            f"{loss_dice_ce:.6f}",
            f"{loss_fused_align:.6f}",
            f"{total_loss:.6f}",
            f"{val_dice_dict.get('ET', 0.0):.4f}",
            f"{val_dice_dict.get('TC', 0.0):.4f}",
            f"{val_dice_dict.get('WT', 0.0):.4f}",
            f"{val_dice_dict.get('macro', 0.0):.4f}",
            f"{val_hd95_dict.get('ET', 0.0):.4f}",
            f"{val_hd95_dict.get('TC', 0.0):.4f}",
            f"{val_hd95_dict.get('WT', 0.0):.4f}",
            f"{val_hd95_dict.get('macro', 0.0):.4f}",
        ]

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)


class EvaluationCSVLogger:
    """
    CSV logger for patient-level 3D test evaluation metrics (Code Standards §6).

    Header:
        patient_id,track_id,hospital_id,
        dice_ET,dice_TC,dice_WT,dice_macro,
        hd95_ET,hd95_TC,hd95_WT,hd95_macro
    """

    HEADER = [
        "patient_id",
        "track_id",
        "hospital_id",
        "dice_ET",
        "dice_TC",
        "dice_WT",
        "dice_macro",
        "hd95_ET",
        "hd95_TC",
        "hd95_WT",
        "hd95_macro",
    ]

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        if not self.filepath.exists():
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADER)

    def log_patient_metrics(
        self,
        patient_id: str,
        track_id: str,
        hospital_id: str,
        dice_dict: Dict[str, float],
        hd95_dict: Dict[str, float],
    ) -> None:
        row = [
            patient_id,
            track_id,
            hospital_id,
            f"{dice_dict.get('ET', 0.0):.4f}",
            f"{dice_dict.get('TC', 0.0):.4f}",
            f"{dice_dict.get('WT', 0.0):.4f}",
            f"{dice_dict.get('macro', 0.0):.4f}",
            f"{hd95_dict.get('ET', 0.0):.4f}",
            f"{hd95_dict.get('TC', 0.0):.4f}",
            f"{hd95_dict.get('WT', 0.0):.4f}",
            f"{hd95_dict.get('macro', 0.0):.4f}",
        ]

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)


class CheckpointManager:
    """
    Atomic Checkpoint Management (§15.3).
    Handles saving and restoring model states, prototypes, training round/epoch, and RNG states.
    """

    @staticmethod
    def save_checkpoint(
        filepath: Union[str, Path],
        state_dict: Dict,
        prototypes: Dict,
        current_round: int,
        current_epoch: int = 0,
        val_macro_dice: float = 0.0,
        extra_metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Atomically save model checkpoint with RNG state serialization (§15.3).
        Writes to `.tmp` path first, then replaces target file atomically.
        """
        final_path = Path(filepath)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.with_suffix(".tmp")

        rng_states = {
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "py_rng": random.getstate(),
        }

        if torch.cuda.is_available():
            rng_states["cuda_rng"] = torch.cuda.get_rng_state_all()

        checkpoint = {
            "state_dict": state_dict,
            "prototypes": prototypes,
            "round": current_round,
            "epoch": current_epoch,
            "val_macro_dice": val_macro_dice,
            "rng_states": rng_states,
            "metadata": extra_metadata or {},
        }

        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, final_path)
        return final_path

    @staticmethod
    def load_checkpoint(
        filepath: Union[str, Path],
        restore_rng: bool = True,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict:
        """
        Load checkpoint file and optionally restore exact RNG states (§15.3).
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        target_device = device if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=target_device)

        if restore_rng and "rng_states" in checkpoint:
            rng = checkpoint["rng_states"]
            if "torch_rng" in rng:
                torch_rng = rng["torch_rng"]
                if isinstance(torch_rng, torch.Tensor):
                    torch_rng = torch_rng.cpu()
                torch.set_rng_state(torch_rng)
            if "numpy_rng" in rng:
                np.random.set_state(rng["numpy_rng"])
            if "py_rng" in rng:
                random.setstate(rng["py_rng"])
            if torch.cuda.is_available() and "cuda_rng" in rng:
                cuda_rng = rng["cuda_rng"]
                if isinstance(cuda_rng, (list, tuple)):
                    cuda_rng = [t.cpu() if isinstance(t, torch.Tensor) else t for t in cuda_rng]
                elif isinstance(cuda_rng, torch.Tensor):
                    cuda_rng = cuda_rng.cpu()
                try:
                    torch.cuda.set_rng_state_all(cuda_rng)
                except Exception:
                    pass

        return checkpoint
