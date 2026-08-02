"""
CAMFS M1 — 1:1 Tumor/Non-Tumor 2D Axial Slice Sampler
======================================================

Spec reference: §13.4
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import torch

class SliceSampler:
    """
    Constructs 1:1 Tumor vs Non-Tumor 2D axial slice index lists for training epochs,
    and full 155-slice sequential index lists for validation/testing.
    """
    def __init__(self, cache_root: Union[str, Path], patient_ids: List[str]):
        self.cache_root = Path(cache_root)
        self.patient_ids = patient_ids
        
        self.tumor_slices: List[Tuple[str, int]] = []
        self.nontumor_brain_slices: List[Tuple[str, int]] = []
        self.all_slices: List[Tuple[str, int]] = []

        self._index_slices()

    def _index_slices(self) -> None:
        """Scan patient label arrays to classify slices into tumor vs non-tumor brain slices."""
        for pid in self.patient_ids:
            lbl_path = self.cache_root / pid / "labels.npy"
            if not lbl_path.exists():
                raise FileNotFoundError(f"Label cache missing for patient {pid} at {lbl_path}")
            
            lbl_vol = np.load(lbl_path, mmap_mode="r")  # [155, 240, 240]
            
            for s_idx in range(155):
                slice_lbl = lbl_vol[s_idx]
                self.all_slices.append((pid, s_idx))

                # Check if slice has tumor pixels (classes 1, 2, or 3)
                has_tumor = np.any(slice_lbl > 0)
                has_brain = np.any(slice_lbl >= 0)  # non-empty brain slice

                if has_tumor:
                    self.tumor_slices.append((pid, s_idx))
                elif has_brain:
                    self.nontumor_brain_slices.append((pid, s_idx))

    def get_epoch_samples(self, seed: int, is_training: bool = True) -> List[Tuple[str, int]]:
        """
        Get sample list for an epoch.
        If training: returns all tumor slices + equal count 1:1 sample of non-tumor brain slices.
        If validation/testing: returns all 155 slices per patient sequentially.
        """
        if not is_training:
            return list(self.all_slices)

        num_tumor = len(self.tumor_slices)
        if num_tumor == 0:
            return list(self.all_slices)

        rng = random.Random(seed)
        num_nontumor = len(self.nontumor_brain_slices)

        if num_nontumor >= num_tumor:
            sampled_nontumor = rng.sample(self.nontumor_brain_slices, num_tumor)
        else:
            # Sample with replacement if non-tumor pool is smaller
            sampled_nontumor = [rng.choice(self.nontumor_brain_slices) for _ in range(num_tumor)]

        epoch_samples = self.tumor_slices + sampled_nontumor
        rng.shuffle(epoch_samples)
        return epoch_samples
