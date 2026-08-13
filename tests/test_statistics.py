"""
CAMFS M1 — Statistical Analysis Unit Tests
===========================================

PyTest suite for Phase 9 statistical methods (§14.5):
  - Percentile-bootstrap confidence intervals (10,000 resamples, seed 8803)
  - Paired patient-level contrast computation d_p
  - Hierarchical run-aware sensitivity intervals (seed 8804)
  - Per-hospital non-aggregation statistics
"""

import numpy as np
import pandas as pd
import pytest

from src.statistics import (
    bootstrap_confidence_intervals,
    compute_paired_patient_contrasts,
    hierarchical_sensitivity_intervals,
    per_hospital_summary,
)


def test_bootstrap_confidence_intervals():
    """Verify bootstrap CI mean and bounds on synthetic patient scores."""
    scores = np.array([0.85, 0.88, 0.82, 0.89, 0.86, 0.84, 0.87, 0.83, 0.90, 0.81])
    mean_val, lower, upper = bootstrap_confidence_intervals(scores, num_resamples=1000, seed=8803)

    assert isinstance(mean_val, float)
    assert isinstance(lower, float)
    assert isinstance(upper, float)

    # Check mean value
    assert pytest.approx(mean_val, abs=1e-4) == float(np.mean(scores))

    # CI lower must be <= mean <= CI upper
    assert lower <= mean_val <= upper
    assert lower >= 0.75 and upper <= 0.95


def test_compute_paired_patient_contrasts():
    """Verify paired patient contrasts d_p computation across partitions and seeds."""
    patient_data = []
    patients = [f"patient_{i:03d}" for i in range(10)]
    partitions = [1103, 2207, 3301]
    seeds = [17, 29, 43]

    for p in patients:
        for part in partitions:
            for s in seeds:
                patient_data.append({
                    "patient_id": p,
                    "partition_seed": part,
                    "train_seed": s,
                    "condition": "camfs_primary",
                    "dice_wt": 0.88,
                })
                patient_data.append({
                    "patient_id": p,
                    "partition_seed": part,
                    "train_seed": s,
                    "condition": "ablation_a1_joint_training",
                    "dice_wt": 0.85,
                })

    df = pd.DataFrame(patient_data)
    dp_dict = compute_paired_patient_contrasts(
        df,
        condition_a="camfs_primary",
        condition_b="ablation_a1_joint_training",
        metric_col="dice_wt",
    )

    assert len(dp_dict) == 10
    for p in patients:
        assert p in dp_dict
        # 0.88 - 0.85 = 0.03 contrast
        assert pytest.approx(dp_dict[p], abs=1e-4) == 0.03


def test_hierarchical_sensitivity_intervals():
    """Verify hierarchical sensitivity interval computation."""
    patient_data = []
    patients = [f"patient_{i:03d}" for i in range(10)]
    partitions = [1103, 2207, 3301]
    seeds = [17, 29, 43]

    for p in patients:
        for part in partitions:
            for s in seeds:
                patient_data.append({
                    "patient_id": p,
                    "partition_seed": part,
                    "train_seed": s,
                    "dice_wt": 0.85 + np.random.normal(0, 0.02),
                })

    df = pd.DataFrame(patient_data)
    mean_val, lower, upper = hierarchical_sensitivity_intervals(df, metric_col="dice_wt", num_resamples=500, seed=8804)

    assert isinstance(mean_val, float)
    assert lower <= mean_val <= upper


def test_per_hospital_summary():
    """Verify per-hospital summary table calculation."""
    eval_data = []
    hospitals = ["H1", "H2", "H3", "H4"]

    for hid in hospitals:
        for i in range(10):
            eval_data.append({
                "patient_id": f"{hid}_patient_{i:02d}",
                "hospital": hid,
                "dice_wt": 0.88 if hid != "H3" else 0.82,
                "dice_tc": 0.85 if hid != "H3" else 0.79,
                "dice_et": 0.82 if hid != "H3" else 0.75,
            })

    df = pd.DataFrame(eval_data)
    summary_df = per_hospital_summary(df)

    assert len(summary_df) == 4
    assert "hospital" in summary_df.columns
    assert "dice_wt_mean" in summary_df.columns
    assert "dice_wt_ci_lower" in summary_df.columns
    assert "dice_wt_ci_upper" in summary_df.columns
