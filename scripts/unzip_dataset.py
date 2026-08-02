"""
Unzip BraTS 2020 dataset zip file into dataset/raw/ directory.
"""

import os
import sys
import zipfile
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parent.parent
    zip_path = root_dir / "dataset" / "BraTS2020_TrainingData.zip"
    extract_to = root_dir / "dataset" / "raw"

    if not zip_path.exists():
        print(f"Error: Zip file not found at {zip_path}")
        sys.exit(1)

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} to {extract_to} ...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print("Extraction complete!")

if __name__ == "__main__":
    main()
