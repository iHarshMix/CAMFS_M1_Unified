"""
CAMFS M1 — Dataset Preprocessing Runner
======================================

Reads raw BraTS NIfTI patient volumes from dataset/raw, computes per-modality
z-score normalization on nonzero brain voxels, clips to [-5, 5], remaps labels
{0,1,2,4} -> {0,1,2,3}, and writes canonical unaugmented per-patient .npy caches
to outputs/preprocessed/patient_<ID>/.

Spec reference: §13.4, §5.3
"""

import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

from src.config import load_yaml
from src.data.brats_dataset import normalize_and_cache_patient

def find_patient_directories(raw_root: Path) -> list[Path]:
    """Recursively search raw_root for patient directories containing NIfTI files."""
    patient_dirs = []
    for root, dirs, files in os.walk(raw_root):
        root_path = Path(root)
        # Check if this folder contains t1 nifti file
        has_t1 = any("_t1.nii" in f.lower() for f in files)
        has_seg = any("_seg.nii" in f.lower() for f in files)
        if has_t1 and has_seg:
            patient_dirs.append(root_path)
    return sorted(patient_dirs, key=lambda p: p.name)

def main():
    config = load_yaml("configs/default.yaml")
    raw_root = Path(config["dataset"]["root"]).resolve()
    cache_root = Path(config["dataset"]["cache_root"]).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"Scanning raw BraTS dataset in: {raw_root}")
    patient_dirs = find_patient_directories(raw_root)
    print(f"Found {len(patient_dirs)} patient directories.")

    if not patient_dirs:
        print("Error: No patient directories found in dataset/raw!")
        sys.exit(1)

    start_time = time.time()
    successful = 0

    print(f"Starting canonical z-normalization & .npy caching to: {cache_root}")
    for p_dir in tqdm(patient_dirs, desc="Preprocessing patients"):
        try:
            normalize_and_cache_patient(p_dir, cache_root)
            successful += 1
        except Exception as e:
            print(f"\nError processing {p_dir.name}: {e}")

    elapsed = time.time() - start_time
    print(f"\nSuccessfully cached {successful}/{len(patient_dirs)} patients in {elapsed:.2f}s.")

if __name__ == "__main__":
    main()
