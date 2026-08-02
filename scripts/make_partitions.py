"""
CAMFS M1 — Federation Partition Generator Runner
===============================================

Reads patient directories from outputs/preprocessed/, sorts them lexicographically,
and generates JSON partition manifests in outputs/partitions/ for partition seeds
{1103, 2207, 3301} using PCG64 seed 901 for fixed H3 test cohort.

Spec reference: §13.3
"""

import sys
from pathlib import Path
from src.config import load_yaml
from src.data.partition import generate_federation_partitions

def main():
    config = load_yaml("configs/default.yaml")
    cache_root = Path(config["dataset"]["cache_root"]).resolve()
    output_dir = Path("outputs/partitions").resolve()

    if not cache_root.exists():
        print(f"Error: Preprocessed cache directory not found at {cache_root}")
        print("Please run 'python scripts/preprocess.py' first.")
        sys.exit(1)

    patient_dirs = [p for p in cache_root.iterdir() if p.is_dir()]
    patient_ids = sorted([p.name for p in patient_dirs])

    print(f"Found {len(patient_ids)} preprocessed patients in cache.")
    if not patient_ids:
        print("Error: No patients found in cache directory!")
        sys.exit(1)

    partition_seeds = config["experiment"]["partition_seeds"]
    test_seed = config["partition"]["fixed_h3_final_test_seed"]
    test_count = config["partition"]["fixed_h3_final_test_patients"]

    print(f"Generating federation partitions for seeds {partition_seeds}...")
    hashes = generate_federation_partitions(
        patient_ids=patient_ids,
        partition_seeds=partition_seeds,
        fixed_h3_test_seed=test_seed,
        h3_test_count=test_count,
        output_dir=output_dir
    )

    print("\nPartition Manifests successfully generated!")
    for seed, h_val in hashes.items():
        print(f"  Partition {seed}: SHA-256 = {h_val}")

if __name__ == "__main__":
    main()
