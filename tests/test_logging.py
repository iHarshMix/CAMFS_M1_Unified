"""
CAMFS M1 — CSV Logging & Checkpoint Management Unit Tests
===========================================================

PyTest suite for Phase 7 logging & checkpoints (Code Standards §6, §15.3):
  - Phase1CSVLogger exact header & row format validation
  - Phase2CSVLogger exact header & row format validation
  - EvaluationCSVLogger exact header & row format validation
  - CheckpointManager atomic file saving (.tmp -> .pt atomic rename)
  - Checkpoint loading & RNG state restoration determinism
"""

import csv
import random
import numpy as np
import pytest
import torch

from src.logging import CheckpointManager, EvaluationCSVLogger, Phase1CSVLogger, Phase2CSVLogger
from src.models.encoder import UnimodalEncoder


def test_phase1_csv_logger_headers_and_rows(tmp_path):
    """Verify Phase1CSVLogger creates correct CSV header and appends formatted rows (Code Standards §6)."""
    csv_file = tmp_path / "phase1_test.csv"
    logger = Phase1CSVLogger(csv_file)

    logger.log_round(1, "H1", "T1", 10, 0.456789, 0.012345)
    logger.log_round(2, "H2", "T2", 30, 0.345678, 0.009876)

    assert csv_file.exists()
    lines = csv_file.read_text().splitlines()

    assert len(lines) == 3  # Header + 2 rows
    assert lines[0] == "round,timestamp,hospital_id,modality,num_patients,info_nce_loss,prototype_drift_l2"

    row1 = lines[1].split(",")
    assert row1[0] == "1"
    assert row1[2] == "H1"
    assert row1[3] == "T1"
    assert row1[4] == "10"
    assert row1[5] == "0.456789"
    assert row1[6] == "0.012345"


def test_phase2_csv_logger_headers_and_rows(tmp_path):
    """Verify Phase2CSVLogger creates correct header and formats validation Dice/HD95 columns."""
    csv_file = tmp_path / "phase2_test.csv"
    logger = Phase2CSVLogger(csv_file)

    val_dice = {"ET": 0.8234, "TC": 0.8567, "WT": 0.9123, "macro": 0.8641}
    val_hd95 = {"ET": 4.123, "TC": 3.456, "WT": 2.789, "macro": 3.456}

    logger.log_round(1, "S1", "H1", 0.123, 0.045, 0.168, val_dice, val_hd95)

    lines = csv_file.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0] == (
        "round,timestamp,track_id,hospital_id,loss_dice_ce,loss_fused_align,total_loss,"
        "val_dice_ET,val_dice_TC,val_dice_WT,val_dice_macro,"
        "val_hd95_ET,val_hd95_TC,val_hd95_WT,val_hd95_macro"
    )

    row = lines[1].split(",")
    assert row[0] == "1"
    assert row[2] == "S1"
    assert row[3] == "H1"
    assert row[4] == "0.123000"
    assert row[7] == "0.8234"
    assert row[10] == "0.8641"
    assert row[14] == "3.4560"


def test_evaluation_csv_logger_headers_and_rows(tmp_path):
    """Verify EvaluationCSVLogger logs patient-level test 3D Dice and HD95 metrics."""
    csv_file = tmp_path / "eval_test.csv"
    logger = EvaluationCSVLogger(csv_file)

    dice = {"ET": 0.85, "TC": 0.88, "WT": 0.92, "macro": 0.8833}
    hd95 = {"ET": 3.5, "TC": 2.8, "WT": 2.1, "macro": 2.80}

    logger.log_patient_metrics("BraTS2020_001", "S1", "H1", dice, hd95)

    lines = csv_file.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0] == (
        "patient_id,track_id,hospital_id,"
        "dice_ET,dice_TC,dice_WT,dice_macro,"
        "hd95_ET,hd95_TC,hd95_WT,hd95_macro"
    )

    row = lines[1].split(",")
    assert row[0] == "BraTS2020_001"
    assert row[1] == "S1"
    assert row[2] == "H1"
    assert row[3] == "0.8500"
    assert row[6] == "0.8833"


def test_checkpoint_manager_atomic_saving_and_restoration(tmp_path):
    """Verify CheckpointManager performs atomic saving (.tmp -> .pt) and restores RNG state (§15.3)."""
    chkpt_file = tmp_path / "test_model.pt"
    tmp_file = tmp_path / "test_model.tmp"

    model = UnimodalEncoder()
    protos = {"T1": torch.randn(4, 256)}

    # Advance Python and NumPy RNG state
    random.seed(12345)
    np.random.seed(12345)

    final_path = CheckpointManager.save_checkpoint(
        filepath=chkpt_file,
        state_dict=model.state_dict(),
        prototypes=protos,
        current_round=15,
        current_epoch=1,
        val_macro_dice=0.8765,
    )

    assert final_path.exists()
    assert not tmp_file.exists(), "Temporary checkpoint file should be atomically renamed"

    # Mutate RNG state
    random.seed(99999)
    np.random.seed(99999)
    mutated_rand = random.random()

    # Reset seed to 12345 and get expected value
    random.seed(12345)
    expected_rand = random.random()

    # Load checkpoint and restore RNG state
    chkpt_data = CheckpointManager.load_checkpoint(final_path, restore_rng=True)

    assert chkpt_data["round"] == 15
    assert chkpt_data["val_macro_dice"] == 0.8765
    assert torch.equal(chkpt_data["prototypes"]["T1"], protos["T1"])

    # Verify RNG state restored matching seed 12345
    restored_rand = random.random()
    assert restored_rand == expected_rand, "RNG state not correctly restored from checkpoint"
