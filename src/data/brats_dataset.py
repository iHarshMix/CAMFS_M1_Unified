"""
CAMFS M1 — BraTS Dataset Loader & Normalizer
============================================

Implements per-patient, per-modality z-score normalization on nonzero brain voxels,
clipping to [-5, 5], label remapping {0,1,2,4} -> {0,1,2,3}, and lazy memory-mapped
loading from uncompressed canonical per-patient .npy caches.

Spec reference: §13.4, §5.2
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}
MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]
MODALITY_FILE_SUFFIXES = {
    "T1": ["_t1.nii", "_t1.nii.gz"],
    "T1ce": ["_t1ce.nii", "_t1ce.nii.gz"],
    "T2": ["_t2.nii", "_t2.nii.gz"],
    "FLAIR": ["_flair.nii", "_flair.nii.gz"],
    "SEG": ["_seg.nii", "_seg.nii.gz"],
}

def remap_labels(seg_array: np.ndarray) -> np.ndarray:
    """Remap raw BraTS labels {0,1,2,4} -> {0,1,2,3}."""
    remapped = np.zeros_like(seg_array, dtype=np.uint8)
    remapped[seg_array == 1] = 1
    remapped[seg_array == 2] = 2
    remapped[seg_array == 4] = 3
    return remapped

def znorm_modality(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute z-score normalization over nonzero brain voxels, clip to [-5, 5],
    and preserve outside-brain voxels as zero. Do not crop.
    Spec reference: §13.4
    """
    brain_mask = volume != 0
    if not np.any(brain_mask):
        return np.zeros_like(volume, dtype=np.float32)

    brain_voxels = volume[brain_mask]
    mean = np.mean(brain_voxels)
    std = np.std(brain_voxels)
    
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[brain_mask] = (brain_voxels - mean) / (std + eps)
    normalized[brain_mask] = np.clip(normalized[brain_mask], -5.0, 5.0)
    return normalized

def load_patient_raw_nifti(
    patient_dir: Path
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Load raw NIfTI files for a patient folder and return volumes on [155, 240, 240] grid.
    NIfTI spatial shape is typically (240, 240, 155), so transpose to (155, 240, 240).
    """
    patient_id = patient_dir.name
    modality_volumes = {}

    for mod in MODALITIES:
        suffixes = MODALITY_FILE_SUFFIXES[mod]
        nii_files = []
        for suf in suffixes:
            nii_files.extend(list(patient_dir.glob(f"*{suf}")) + list(patient_dir.glob(f"*{suf.upper()}")))
        
        if not nii_files:
            raise FileNotFoundError(f"Missing {mod} NIfTI file in {patient_dir}")
        
        nii_path = nii_files[0]
        img = nib.load(str(nii_path))
        data = img.get_fdata(dtype=np.float32)
        
        # Transpose from (240, 240, 155) to (155, 240, 240) if needed
        if data.shape == (240, 240, 155):
            data = np.transpose(data, (2, 0, 1))
        modality_volumes[mod] = data

    # Load segmentation mask
    seg_suffixes = MODALITY_FILE_SUFFIXES["SEG"]
    seg_files = []
    for suf in seg_suffixes:
        seg_files.extend(list(patient_dir.glob(f"*{suf}")) + list(patient_dir.glob(f"*{suf.upper()}")))
    
    if not seg_files:
        raise FileNotFoundError(f"Missing SEG NIfTI file in {patient_dir}")
    
    seg_img = nib.load(str(seg_files[0]))
    seg_data = seg_img.get_fdata()
    if seg_data.shape == (240, 240, 155):
        seg_data = np.transpose(seg_data, (2, 0, 1))
    
    seg_remapped = remap_labels(seg_data)
    return modality_volumes, seg_remapped

def normalize_and_cache_patient(
    patient_dir: Path,
    output_cache_dir: Path
) -> Path:
    """
    Load raw patient NIfTI, compute canonical z-norm, and write uncompressed .npy files.
    """
    patient_id = patient_dir.name
    target_dir = output_cache_dir / patient_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if already fully cached
    expected_files = [target_dir / "labels.npy"] + [target_dir / f"{m.lower()}.npy" for m in MODALITIES]
    if all(f.exists() for f in expected_files):
        return target_dir

    modality_vols, seg = load_patient_raw_nifti(patient_dir)

    for mod, vol in modality_vols.items():
        norm_vol = znorm_modality(vol)
        np.save(target_dir / f"{mod.lower()}.npy", norm_vol)

    np.save(target_dir / "labels.npy", seg)
    return target_dir

class BraTSDataset(Dataset):
    """
    Lazy, memory-mapped PyTorch Dataset reading canonical .npy preprocessed caches.
    """
    def __init__(
        self,
        patient_ids: List[str],
        cache_root: Union[str, Path],
        modalities: Optional[List[str]] = None,
        transform = None
    ):
        self.cache_root = Path(cache_root)
        self.patient_ids = patient_ids
        self.modalities = modalities or MODALITIES
        self.transform = transform
        
        # Build slice index: list of (patient_id, slice_idx)
        self.samples: List[Tuple[str, int]] = []
        for pid in self.patient_ids:
            for s_idx in range(155):
                self.samples.append((pid, s_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        pid, s_idx = self.samples[idx]
        patient_dir = self.cache_root / pid

        sample_dict = {"patient_id": pid, "slice_idx": s_idx}
        
        # Load memory-mapped 2D slice for each requested modality
        for mod in self.modalities:
            npy_path = patient_dir / f"{mod.lower()}.npy"
            vol = np.load(npy_path, mmap_mode="r")
            slice_data = np.array(vol[s_idx], dtype=np.float32)
            sample_dict[mod] = torch.from_numpy(slice_data).unsqueeze(0)  # [1, 240, 240]

        # Load label slice
        lbl_path = patient_dir / "labels.npy"
        lbl_vol = np.load(lbl_path, mmap_mode="r")
        lbl_slice = np.array(lbl_vol[s_idx], dtype=np.int64)
        sample_dict["label"] = torch.from_numpy(lbl_slice)  # [240, 240]

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)

        return sample_dict

    def load_patient_volume(self, pid: str) -> Dict[str, Union[Dict[str, np.ndarray], np.ndarray]]:
        """Load 3D volume arrays (155, 240, 240) using memory mapping for minimal RAM footprint."""
        patient_dir = self.cache_root / pid
        res = {"modalities": {}, "labels": None}
        for mod in self.modalities:
            npy_path = patient_dir / f"{mod.lower()}.npy"
            if npy_path.exists():
                res["modalities"][mod] = np.load(npy_path, mmap_mode="r")
        lbl_path = patient_dir / "labels.npy"
        if lbl_path.exists():
            res["labels"] = np.load(lbl_path, mmap_mode="r")
        return res
