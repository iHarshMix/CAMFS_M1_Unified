"""
CAMFS M1 — Experiment Runner CLI & Manifest Unit Tests
======================================================

PyTest suite for Phase 8 experiment runners (Code Standards §7.2):
  - CLI argument parsing verification for run_primary.py, run_baselines.py, and run_ablations.py
  - Baseline manifests existence & valid JSON schema verification (disentafl.json, fedamm.json)
"""

import json
from pathlib import Path

import pytest

from scripts.run_ablations import parse_args as parse_ablation_args
from scripts.run_baselines import parse_args as parse_baseline_args
from scripts.run_primary import parse_args as parse_primary_args


def test_baseline_manifests_exist_and_valid():
    """Verify disentafl.json and fedamm.json baseline manifests exist and contain valid schemas (§3.4)."""
    disentafl_file = Path("configs/baseline_manifests/disentafl.json")
    fedamm_file = Path("configs/baseline_manifests/fedamm.json")

    assert disentafl_file.exists(), "configs/baseline_manifests/disentafl.json missing"
    assert fedamm_file.exists(), "configs/baseline_manifests/fedamm.json missing"

    with open(disentafl_file, "r") as f:
        d_data = json.load(f)
    assert d_data["baseline_id"] == "B3_DisentAFL"
    assert d_data["audit_mode"] == "shadow"
    assert "source_commit" in d_data

    with open(fedamm_file, "r") as f:
        f_data = json.load(f)
    assert f_data["baseline_id"] == "B5_FedAMM"
    assert f_data["prototype_aggregation_scheme"] == "per_combination"


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


def test_run_baselines_cli_parsing(monkeypatch):
    """Verify run_baselines.py CLI argument parsing."""
    test_args = [
        "run_baselines.py",
        "--baseline",
        "B3_DisentAFL",
        "--partition-seed",
        "8801",
        "--dry-run",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    parsed = parse_baseline_args()

    assert parsed.baseline == "B3_DisentAFL"
    assert parsed.partition_seed == 8801
    assert parsed.dry_run is True


def test_run_ablations_cli_parsing(monkeypatch):
    """Verify run_ablations.py CLI argument parsing."""
    test_args = [
        "run_ablations.py",
        "--ablation",
        "A1_NoPhase1Align",
        "--train-seed",
        "19",
        "--dry-run",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    parsed = parse_ablation_args()

    assert parsed.ablation == "A1_NoPhase1Align"
    assert parsed.train_seed == 19
    assert parsed.dry_run is True
