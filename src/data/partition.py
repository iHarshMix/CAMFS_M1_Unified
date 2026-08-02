"""
CAMFS M1 — Federation Patient Partitioner
=========================================

Sorts official patient identifiers lexicographically, selects one fixed H3
final-test cohort using PCG64 seed 901, and splits remaining patients across
H1, H2, H3 (non-test), and H4 for partition seeds {1103, 2207, 3301}.

Spec reference: §13.3
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def compute_json_sha256(data: dict) -> str:
    """Compute SHA-256 hash digest of a canonical JSON object."""
    canonical_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()

def generate_federation_partitions(
    patient_ids: List[str],
    partition_seeds: List[int] = [1103, 2207, 3301],
    fixed_h3_test_seed: int = 901,
    h3_test_count: int = 50,
    output_dir: Path = Path("outputs/partitions")
) -> Dict[str, str]:
    """
    Generate deterministic patient partitions and write JSON manifests.
    Returns mapping of partition seed to SHA-256 hash.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_pids = sorted(patient_ids)
    total_patients = len(sorted_pids)

    if total_patients < h3_test_count + 10:
        raise ValueError(f"Too few patients ({total_patients}) to construct partition")

    # Step 1: Select fixed 50 H3 final-test patients using PCG64 seed 901
    pcg_901 = np.random.Generator(np.random.PCG64(fixed_h3_test_seed))
    shuffled_all = list(sorted_pids)
    pcg_901.shuffle(shuffled_all)
    
    h3_test_pids = sorted(shuffled_all[:h3_test_count])
    remaining_pids = sorted(shuffled_all[h3_test_count:])

    # Save fixed H3 test set
    h3_test_manifest = {
        "fixed_h3_test_seed": fixed_h3_test_seed,
        "count": len(h3_test_pids),
        "patient_ids": h3_test_pids,
    }
    h3_test_hash = compute_json_sha256(h3_test_manifest)
    h3_test_manifest["sha256"] = h3_test_hash
    with open(output_dir / "h3_test_patients.json", "w", encoding="utf-8") as f:
        json.dump(h3_test_manifest, f, indent=2)

    # Step 2: Compute hospital patient counts for remaining patients
    rem_count = len(remaining_pids)
    # H1 40%, H2 25%, H3 20%, H4 15%
    h1_count = int(round(rem_count * 0.40))
    h2_count = int(round(rem_count * 0.25))
    h3_non_test_count = int(round(rem_count * 0.20))
    h4_count = rem_count - (h1_count + h2_count + h3_non_test_count)

    partition_hashes = {}

    for seed in partition_seeds:
        pcg_seed = np.random.Generator(np.random.PCG64(seed))
        permuted_rem = list(remaining_pids)
        pcg_seed.shuffle(permuted_rem)

        h1_pids = sorted(permuted_rem[:h1_count])
        h2_pids = sorted(permuted_rem[h1_count : h1_count + h2_count])
        h3_train_val_pids = sorted(permuted_rem[h1_count + h2_count : h1_count + h2_count + h3_non_test_count])
        h4_pids = sorted(permuted_rem[h1_count + h2_count + h3_non_test_count:])

        partition_data = {
            "partition_seed": seed,
            "total_patients": total_patients,
            "hospitals": {
                "H1": {"total": len(h1_pids), "patient_ids": h1_pids},
                "H2": {"total": len(h2_pids), "patient_ids": h2_pids},
                "H3": {
                    "total_train_val": len(h3_train_val_pids),
                    "patient_ids": h3_train_val_pids,
                    "fixed_test_patient_ids": h3_test_pids
                },
                "H4": {"total": len(h4_pids), "patient_ids": h4_pids},
            }
        }
        
        part_hash = compute_json_sha256(partition_data)
        partition_data["sha256"] = part_hash
        partition_hashes[str(seed)] = part_hash

        with open(output_dir / f"partition_{seed}.json", "w", encoding="utf-8") as f:
            json.dump(partition_data, f, indent=2)

    # Save manifest hash dictionary
    manifest_summary = {
        "fixed_h3_test_hash": h3_test_hash,
        "partition_hashes": partition_hashes
    }
    with open(output_dir / "manifests_sha256.json", "w", encoding="utf-8") as f:
        json.dump(manifest_summary, f, indent=2)

    return partition_hashes

def load_partition(partition_seed: int, partitions_dir: Path = Path("outputs/partitions")) -> dict:
    """Load partition JSON for a given partition seed."""
    part_file = partitions_dir / f"partition_{partition_seed}.json"
    if not part_file.exists():
        raise FileNotFoundError(f"Partition file not found: {part_file}")
    with open(part_file, "r", encoding="utf-8") as f:
        return json.load(f)
