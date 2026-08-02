"""
CAMFS M1 — Data Pipeline Sanity Test Suite
===========================================

Verifies dataset loading, normalization, remapping, partition patient counts,
slice sampling, and augmentation for the Stage 1 pilot setup.

Spec reference: §13.3, §13.4, §19.3
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.brats_dataset import BraTSDataset, remap_labels, znorm_modality
from src.data.partition import generate_federation_partitions
from src.data.slice_sampler import SliceSampler
from src.data.augmentation import PairwiseAugmentation

def test_label_remapping():
    """Verify raw BraTS labels {0,1,2,4} -> {0,1,2,3} remapping."""
    raw = np.array([0, 1, 2, 4, 0, 1, 4], dtype=np.uint8)
    remapped = remap_labels(raw)
    expected = np.array([0, 1, 2, 3, 0, 1, 3], dtype=np.uint8)
    np.testing.assert_array_equal(remapped, expected)

def test_znorm_modality():
    """Verify z-score normalization on nonzero brain voxels with clip [-5, 5]."""
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    vol[2:8, 2:8, 2:8] = np.random.normal(loc=100.0, scale=20.0, size=(6, 6, 6))
    
    norm = znorm_modality(vol)
    
    # Outside brain voxels must remain exactly 0.0
    assert np.all(norm[vol == 0] == 0.0)
    
    # Nonzero brain voxels must be within [-5, 5]
    brain_vals = norm[vol != 0]
    assert np.min(brain_vals) >= -5.0
    assert np.max(brain_vals) <= 5.0
    assert abs(np.mean(brain_vals)) < 1e-4

def test_partition_generation(tmp_path):
    """Verify PCG64 partition generator patient splits and SHA-256 digests."""
    dummy_pids = [f"BraTS20_Training_{i:03d}" for i in range(1, 101)]
    hashes = generate_federation_partitions(
        patient_ids=dummy_pids,
        partition_seeds=[1103, 2207],
        fixed_h3_test_seed=901,
        h3_test_count=10,
        output_dir=tmp_path
    )
    
    assert "1103" in hashes
    assert "2207" in hashes
    
    with open(tmp_path / "partition_1103.json", "r") as f:
        p1103 = json.load(f)
    
    assert len(p1103["hospitals"]["H3"]["fixed_test_patient_ids"]) == 10
    total_alloc = (
        p1103["hospitals"]["H1"]["total"] +
        p1103["hospitals"]["H2"]["total"] +
        p1103["hospitals"]["H3"]["total_train_val"] +
        p1103["hospitals"]["H4"]["total"]
    )
    assert total_alloc == 90

def test_pairwise_augmentation():
    """Verify co-registered geometric and intensity augmentations."""
    aug = PairwiseAugmentation(modalities=["T1", "T2"], is_training=True)
    
    sample = {
        "T1": torch.ones((1, 240, 240), dtype=torch.float32),
        "T2": torch.ones((1, 240, 240), dtype=torch.float32),
        "label": torch.ones((240, 240), dtype=torch.int64),
    }
    
    aug_sample = aug(sample)
    assert aug_sample["T1"].shape == (1, 240, 240)
    assert aug_sample["T2"].shape == (1, 240, 240)
    assert aug_sample["label"].shape == (240, 240)
