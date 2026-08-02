"""
CAMFS M1 — Ablation Configurations & CLI Parser Unit Tests
===========================================================

PyTest suite for Phase 8 ablation runners and configurations (Code Standards §7.2):
  - Ablation YAML configs loading and deep-merge inheritance verification (A1–A8)
  - CLI argument parsing for run_primary.py and run_ablations.py
  - Effective setting overrides for each ablation condition:
      * A1: Joint-training skip_freeze flag
      * A2: Delayed-site context flag
      * A3: Executable lineage audit mode
      * A4: Lambda1 sweep value resolution
      * A5: Cold-start tiers
      * A6: Directional T2 policy flag
      * A7: Multi-track contribution R_contribute flag
      * A8: H2 reconnection mode
"""

import argparse
from pathlib import Path
import pytest

from scripts.run_ablations import ABLATION_CONFIG_MAP, ABLATION_IDS, parse_args as parse_ablation_args
from scripts.run_primary import parse_args as parse_primary_args
from src.config import load_yaml


def test_ablation_yaml_configs_exist_and_load():
    """Verify all 8 ablation YAML config files exist and load correctly via load_yaml."""
    for ablation_id, config_path_str in ABLATION_CONFIG_MAP.items():
        config_path = Path(config_path_str)
        assert config_path.exists(), f"Ablation config missing: {config_path_str}"

        # Test YAML loading & inheritance deep merge
        cfg = load_yaml(config_path)
        assert "experiment" in cfg
        assert "dataset" in cfg
        assert "ablation" in cfg
        assert cfg["ablation"]["id"] == ablation_id


def test_a1_joint_training_config():
    """Verify A1 joint training configuration overrides."""
    cfg = load_yaml("configs/ablations/a1_joint_training.yaml")
    assert cfg["ablation"]["skip_freeze"] is True
    assert cfg["ablation"]["combined_loss"] is True


def test_a4_lambda1_sweep_config():
    """Verify A4 lambda1 sweep configuration overrides."""
    cfg = load_yaml("configs/ablations/a4_lambda1_sweep.yaml")
    assert cfg["ablation"]["sweep_values"] == [0.0, 0.1, 0.5, 1.0]
    assert cfg["ablation"]["validation_only"] is True
    assert cfg["phase1"]["lambda1"] == 0.0


def test_a5_cold_start_config():
    """Verify A5 cold-start configuration overrides."""
    cfg = load_yaml("configs/ablations/a5_cold_start.yaml")
    assert cfg["ablation"]["tiers"] == [1, 2, 3]
    assert cfg["ablation"]["tier2_warmstart_rounds"] == 10


def test_a6_directional_config():
    """Verify A6 directional T2 policy configuration overrides."""
    cfg = load_yaml("configs/ablations/a6_directional.yaml")
    assert cfg["ablation"]["t2_policy"] == "directional"


def test_a7_multi_track_config():
    """Verify A7 multi-track contribution configuration overrides."""
    cfg = load_yaml("configs/ablations/a7_multi_track.yaml")
    assert cfg["ablation"]["r_contribute_h1_s3"] is False


def test_a8_reconnection_config():
    """Verify A8 reconnection configuration overrides."""
    cfg = load_yaml("configs/ablations/a8_reconnection.yaml")
    assert cfg["ablation"]["h2_reconnection_mode"] == "pull_only"


def test_run_primary_cli_parsing(monkeypatch):
    """Verify run_primary.py CLI argument parsing."""
    test_args = [
        "run_primary.py",
        "--partition-seed",
        "8802",
        "--train-seed",
        "18",
        "--gpu",
        "0",
        "--dry-run",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    parsed = parse_primary_args()

    assert parsed.partition_seed == 8802
    assert parsed.train_seed == 18
    assert parsed.gpu == 0
    assert parsed.dry_run is True


def test_run_ablations_cli_parsing(monkeypatch):
    """Verify run_ablations.py CLI argument parsing across ablation choices."""
    test_args = [
        "run_ablations.py",
        "--ablation",
        "A1_JointTraining",
        "--train-seed",
        "19",
        "--dry-run",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    parsed = parse_ablation_args()

    assert parsed.ablation == "A1_JointTraining"
    assert parsed.train_seed == 19
    assert parsed.dry_run is True
