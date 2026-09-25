"""
CAMFS M2 — Patient Partition Splitting
======================================

Deterministic patient partition splitting for M2 CDRD knowledge transfer.
Subdivides existing M1 validation patients into M2-specific roles without
requiring M1 retraining (Option A):

- H1 (127 patients):
    - 102 Train: Teacher & Adapter training
    - 13 Val: Model selection (highest validation Macro Dice)
    - 12 Release-Audit: Teacher viability gate (q_T evaluation)
- H3 (114 patients: 64 non-test + 50 sacred test):
    - 52 M1 Train: Frozen base track S3
    - 6 M2 Calibration: 4 gate logits optimization (a_c)
    - 6 M2 Acceptance-Val: Augmented vs fallback go/no-go routing
    - 50 Sacred Test: Quarantined, evaluated strictly at the end
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class H1M2Splits:
    """Patient ID splits for Donor Hospital H1."""
    train_ids: List[str]          # 102 patients
    val_ids: List[str]            # 13 patients (checkpoint selection)
    release_audit_ids: List[str]  # 12 patients (teacher viability gate)

    @property
    def total_count(self) -> int:
        return len(self.train_ids) + len(self.val_ids) + len(self.release_audit_ids)


@dataclass(frozen=True)
class H3M2Splits:
    """Patient ID splits for Recipient Hospital H3."""
    m1_train_ids: List[str]       # 52 patients (frozen base)
    m2_cal_ids: List[str]         # 6 patients (4 gate parameters)
    m2_accept_val_ids: List[str]  # 6 patients (acceptance go/no-go)
    sacred_test_ids: List[str]    # 50 patients (quarantined test set)

    @property
    def non_test_count(self) -> int:
        return len(self.m1_train_ids) + len(self.m2_cal_ids) + len(self.m2_accept_val_ids)

    @property
    def total_count(self) -> int:
        return self.non_test_count + len(self.sacred_test_ids)


@dataclass(frozen=True)
class M2Partition:
    """Container for all M2 patient splits across H1 and H3."""
    partition_id: int
    split_seed: int
    h1: H1M2Splits
    h3: H3M2Splits


def get_m1_hospital_train_val_split(
    patient_ids: List[str], val_ratio: float = 0.2
) -> Tuple[List[str], List[str]]:
    """
    Replicates exact M1 train/val split logic from FederatedClient.__init__:
    n_val = max(1, int(len(patient_ids) * val_ratio))
    train_ids = patient_ids[:-n_val]
    val_ids = patient_ids[-n_val:]
    """
    n_val = max(1, int(len(patient_ids) * val_ratio))
    train_ids = list(patient_ids[:-n_val])
    val_ids = list(patient_ids[-n_val:])
    return train_ids, val_ids


def compute_m2_splits(
    partition_path: Union[str, Path] = "outputs/partitions/partition_1103.json",
    test_patients_path: Optional[Union[str, Path]] = "outputs/partitions/h3_test_patients.json",
    split_seed: int = 42,
) -> M2Partition:
    """
    Deterministic M2 patient split computation from M1 partition file.

    Parameters:
        partition_path: Path to partition_1103.json
        test_patients_path: Path to h3_test_patients.json (optional, can read from partition if present)
        split_seed: Deterministic seed for PCG64 shuffling of validation subsets

    Returns:
        M2Partition object containing validated, non-overlapping splits
    """
    partition_path = Path(partition_path)
    if not partition_path.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_path}")

    with open(partition_path, "r") as f:
        pdata = json.load(f)

    part_id = pdata.get("partition_seed", 1103)
    hosp_data = pdata.get("hospitals", {})

    if "H1" not in hosp_data or "H3" not in hosp_data:
        raise KeyError("Partition file must contain 'H1' and 'H3' hospital definitions")

    h1_all = hosp_data["H1"]["patient_ids"]
    h3_non_test = hosp_data["H3"]["patient_ids"]

    # Sacred test set for H3
    if test_patients_path and Path(test_patients_path).exists():
        with open(test_patients_path, "r") as f:
            h3_test = json.load(f)
            if isinstance(h3_test, dict) and "patient_ids" in h3_test:
                h3_test = h3_test["patient_ids"]
    elif "fixed_test_patient_ids" in hosp_data["H3"]:
        h3_test = hosp_data["H3"]["fixed_test_patient_ids"]
    else:
        raise ValueError("Could not locate 50 sacred H3 test patients")

    # Replicate M1 base split
    h1_train, h1_val_all = get_m1_hospital_train_val_split(h1_all)
    h3_train, h3_val_all = get_m1_hospital_train_val_split(h3_non_test)

    # Deterministic permutation of validation sets using PCG64
    rng = np.random.Generator(np.random.PCG64(split_seed))

    # H1: 25 val -> 13 val + 12 release-audit
    h1_val_shuffled = list(rng.permutation(h1_val_all))
    h1_val = sorted(h1_val_shuffled[:13])
    h1_audit = sorted(h1_val_shuffled[13:])

    # H3: 12 val -> 6 cal + 6 accept-val
    h3_val_shuffled = list(rng.permutation(h3_val_all))
    h3_cal = sorted(h3_val_shuffled[:6])
    h3_accept = sorted(h3_val_shuffled[6:])

    # Assertions for mathematical integrity and strict quarantine
    assert len(set(h1_train) & set(h1_val)) == 0, "H1 train/val overlap"
    assert len(set(h1_train) & set(h1_audit)) == 0, "H1 train/audit overlap"
    assert len(set(h1_val) & set(h1_audit)) == 0, "H1 val/audit overlap"
    assert len(h1_train) + len(h1_val) + len(h1_audit) == len(h1_all)

    assert len(set(h3_train) & set(h3_cal)) == 0, "H3 train/cal overlap"
    assert len(set(h3_train) & set(h3_accept)) == 0, "H3 train/accept overlap"
    assert len(set(h3_cal) & set(h3_accept)) == 0, "H3 cal/accept overlap"
    assert len(h3_train) + len(h3_cal) + len(h3_accept) == len(h3_non_test)

    # SACRED TEST SET QUARANTINE ASSERTION
    h3_non_test_set = set(h3_non_test)
    test_set = set(h3_test)
    overlap = h3_non_test_set & test_set
    assert len(overlap) == 0, f"FATAL: Sacred test set overlaps with training/val set: {overlap}"
    assert len(h3_test) == 50, f"Expected exactly 50 test patients, found {len(h3_test)}"

    h1_splits = H1M2Splits(train_ids=h1_train, val_ids=h1_val, release_audit_ids=h1_audit)
    h3_splits = H3M2Splits(
        m1_train_ids=h3_train,
        m2_cal_ids=h3_cal,
        m2_accept_val_ids=h3_accept,
        sacred_test_ids=sorted(list(h3_test)),
    )

    return M2Partition(
        partition_id=part_id,
        split_seed=split_seed,
        h1=h1_splits,
        h3=h3_splits,
    )


def save_m2_splits_json(
    splits: M2Partition,
    output_path: Union[str, Path] = "outputs/partitions/m2_splits_1103.json",
) -> None:
    """Saves M2 partition splits to JSON for reproducible inspection."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(asdict(splits), f, indent=2)
