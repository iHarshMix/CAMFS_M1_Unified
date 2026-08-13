"""
CAMFS M1 — Pure 50-Patient Held-Out Test Set Evaluator
======================================================

Evaluates ALL four hospital models (H1, H2, H3, H4) on the EXACT SAME 50 pure held-out
test patients (fixed_h3_test_patients, seed 901) that were 100% excluded from all training,
validation, and model selection across all hospitals.

Models Evaluated:
  - Hospital 1 (H1): Track S1 = {T1, T1ce, T2, FLAIR} -> best_track_S1.pt
  - Hospital 2 (H2): Track_local,2 = {T1, T1ce, T2}   -> best_H2_local_head.pt
  - Hospital 3 (H3): Track S3 = {T1, FLAIR}           -> best_track_S3.pt
  - Hospital 4 (H4): Track S4 = {T1, T1ce, T2}         -> best_track_S4.pt

Output:
  - Per-patient 3D metrics logged to results/pure_50_test_patient_metrics.csv
  - Hospital summary table (Mean ± Std & Single Exact Averages) printed to console
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.config import load_yaml
from src.data.brats_dataset import BraTSDataset
from src.logging import CheckpointManager, EvaluationCSVLogger
from src.metrics import PatientEvaluator
from src.models import SubsetFusionHead, UNetDecoder, UnimodalEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="CAMFS M1 Pure 50-Patient Test Set Evaluator")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--partition-seed", type=int, default=1103, help="Partition seed (1103, 2207, 3301)")
    parser.add_argument("--train-seed", type=int, default=17, help="Training seed (17, 29, 43)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--preprocessed-dir", type=str, default="outputs/preprocessed", help="Path to preprocessed data")
    parser.add_argument("--partitions-dir", type=str, default="outputs/partitions", help="Path to partition manifests")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Root output directory")
    return parser.parse_args()


def evaluate_pure_50_test_set():
    args = parse_args()
    config = load_yaml(args.config)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    print(f"=== Execution Device: {device} (CUDA Available: {torch.cuda.is_available()}) ===", flush=True)

    experiment_id = f"camfs_primary__part{args.partition_seed}__seed{args.train_seed}"
    chkpt_dir = Path(args.output_dir) / "checkpoints" / experiment_id
    results_dir = Path(args.output_dir) / "results" / experiment_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load partition manifest & extract the 50 pure held-out test patients
    partition_file = Path(args.partitions_dir) / f"partition_{args.partition_seed}.json"
    if not partition_file.exists():
        raise FileNotFoundError(f"Partition manifest not found: {partition_file}")

    with open(partition_file, "r") as f:
        partition_data = json.load(f)

    h3_info = partition_data["hospitals"]["H3"]
    pure_50_test_pids = h3_info.get("fixed_test_patient_ids", [])
    if not pure_50_test_pids:
        # Fallback to loading h3_test_patients.json
        h3_test_file = Path(args.partitions_dir) / "h3_test_patients.json"
        if h3_test_file.exists():
            with open(h3_test_file, "r") as f:
                pure_50_test_pids = json.load(f)["patient_ids"]

    print(f"=== Loaded PURE 50-Patient Universal Held-Out Test Set (N = {len(pure_50_test_pids)}) ===", flush=True)

    # 2. Load frozen Phase 1 encoders
    phase1_frozen_chkpt = chkpt_dir / "phase1_frozen.pt"
    if not phase1_frozen_chkpt.exists():
        raise FileNotFoundError(f"Frozen Phase 1 checkpoint not found: {phase1_frozen_chkpt}")

    p1_data = CheckpointManager.load_checkpoint(phase1_frozen_chkpt, device=device)
    encoders: Dict[str, UnimodalEncoder] = {}
    for m, enc_state in p1_data["state_dict"].items():
        enc = UnimodalEncoder(in_channels=1).to(device)
        enc.load_state_dict(enc_state)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad = False
        encoders[m] = enc

    print(f"=== Loaded Frozen Encoders from {phase1_frozen_chkpt.name} ({list(encoders.keys())}) ===", flush=True)

    # 3. Map hospital evaluation models & checkpoints
    eval_specs = {
        "H1": {
            "track_id": "S1",
            "modalities": ["T1", "T1ce", "T2", "FLAIR"],
            "chkpt_name": "best_track_S1.pt",
        },
        "H2": {
            "track_id": "H2_local",
            "modalities": ["T1", "T1ce", "T2"],
            "chkpt_name": "best_H2_local_head.pt",
        },
        "H3": {
            "track_id": "S3",
            "modalities": ["T1", "FLAIR"],
            "chkpt_name": "best_track_S3.pt",
        },
        "H4": {
            "track_id": "S4",
            "modalities": ["T1", "T1ce", "T2"],
            "chkpt_name": "best_track_S4.pt",
        },
    }

    # 4. Initialize CSV Logger & Evaluator
    eval_csv_path = results_dir / "pure_50_test_patient_metrics.csv"
    eval_logger = EvaluationCSVLogger(eval_csv_path)
    evaluator = PatientEvaluator()

    hospital_summary = {}

    # 5. Evaluate ALL 4 hospital models on the EXACT SAME 50 pure test patients
    print(f"\n===================================================================================", flush=True)
    print(f"=== STARTING STANDARDIZED EVALUATION: ALL 4 HOSPITALS ON SAME 50 PURE TEST PATIENTS ===", flush=True)
    print(f"===================================================================================\n", flush=True)

    for hid, h_spec in eval_specs.items():
        chkpt_path = chkpt_dir / h_spec["chkpt_name"]
        if not chkpt_path.exists():
            print(f"WARNING: Checkpoint {chkpt_path} not found for {hid}. Skipping.", flush=True)
            continue

        chkpt_data = CheckpointManager.load_checkpoint(chkpt_path, device=device, restore_rng=False)
        state = chkpt_data["state_dict"]

        # Load fusion head & decoder
        fusion_head = SubsetFusionHead(modality_subset=h_spec["modalities"]).to(device)
        fusion_head.load_state_dict(state["fusion"])
        fusion_head.eval()

        decoder = UNetDecoder(num_classes=4).to(device)
        decoder.load_state_dict(state["decoder"])
        decoder.eval()

        val_dice = chkpt_data.get("val_macro_dice", 0.0)
        round_num = chkpt_data.get("round", 0)
        print(f"--- Evaluator [{hid}] Track {h_spec['track_id']} ({len(pure_50_test_pids)} Pure Test Patients) ---", flush=True)
        print(f"    Loaded {chkpt_path.name} (Round/Epoch {round_num}, Peak Val Dice: {val_dice*100:.2f}%)", flush=True)

        # Setup dataset for pure test patients
        dataset = BraTSDataset(
            patient_ids=pure_50_test_pids,
            cache_root=args.preprocessed_dir,
            modalities=h_spec["modalities"],
        )

        h_patient_metrics = []

        with torch.no_grad():
            for pid_idx, pid in enumerate(pure_50_test_pids):
                vol = dataset.load_patient_volume(pid)
                lab_vol = vol["labels"]
                num_slices = lab_vol.shape[0]
                batch_size = 16

                pred_logits_list = []
                for start in range(0, num_slices, batch_size):
                    end = min(start + batch_size, num_slices)

                    mod_feats_b = {}
                    for m in h_spec["modalities"]:
                        if m in vol["modalities"]:
                            x_m = torch.from_numpy(vol["modalities"][m][start:end]).float().unsqueeze(1).to(device)
                            h1, h2, h3, h4, z = encoders[m](x_m)
                            mod_feats_b[m] = (h1, h2, h3, h4, z)

                    f1, f2, f3, f4, z_S_b = fusion_head(mod_feats_b)
                    logits_b = decoder(z_S=z_S_b, f_S_4=f4, f_S_3=f3, f_S_2=f2, f_S_1=f1)
                    pred_logits_list.append(logits_b.cpu())

                pred_logits_3d = torch.cat(pred_logits_list, dim=0).permute(1, 0, 2, 3).numpy()  # (4, D, H, W)
                p_metrics = evaluator.evaluate_patient_volume(pred_logits_3d, lab_vol)
                h_patient_metrics.append(p_metrics)

                # Log to CSV
                dice_dict = {
                    "ET": p_metrics["dice_ET"],
                    "TC": p_metrics["dice_TC"],
                    "WT": p_metrics["dice_WT"],
                    "macro": p_metrics["macro_dice"],
                }
                hd95_dict = {
                    "ET": p_metrics["hd95_ET"],
                    "TC": p_metrics["hd95_TC"],
                    "WT": p_metrics["hd95_WT"],
                    "macro": p_metrics["macro_hd95"],
                }
                eval_logger.log_patient_metrics(
                    patient_id=pid,
                    track_id=h_spec["track_id"],
                    hospital_id=hid,
                    dice_dict=dice_dict,
                    hd95_dict=hd95_dict,
                )

                if (pid_idx + 1) % 10 == 0 or (pid_idx + 1) == len(pure_50_test_pids):
                    print(f"    Processed {pid_idx+1}/{len(pure_50_test_pids)} test patients ({pid}) - "
                          f"Macro Dice: {p_metrics['macro_dice']*100:.2f}%", flush=True)

        # Compute hospital averages
        avg_ET = float(np.mean([m["dice_ET"] for m in h_patient_metrics])) * 100
        std_ET = float(np.std([m["dice_ET"] for m in h_patient_metrics])) * 100
        avg_TC = float(np.mean([m["dice_TC"] for m in h_patient_metrics])) * 100
        std_TC = float(np.std([m["dice_TC"] for m in h_patient_metrics])) * 100
        avg_WT = float(np.mean([m["dice_WT"] for m in h_patient_metrics])) * 100
        std_WT = float(np.std([m["dice_WT"] for m in h_patient_metrics])) * 100
        avg_macro = float(np.mean([m["macro_dice"] for m in h_patient_metrics])) * 100
        std_macro = float(np.std([m["macro_dice"] for m in h_patient_metrics])) * 100

        med_hd_macro = float(np.median([m["macro_hd95"] for m in h_patient_metrics]))
        avg_hd_macro = float(np.mean([m["macro_hd95"] for m in h_patient_metrics]))

        hospital_summary[hid] = {
            "track": h_spec["track_id"],
            "modalities": "+".join(h_spec["modalities"]),
            "n_test": len(pure_50_test_pids),
            "ET": avg_ET,
            "TC": avg_TC,
            "WT": avg_WT,
            "Macro": avg_macro,
            "Std_Macro": std_macro,
            "HD95_Mean": avg_hd_macro,
            "HD95_Median": med_hd_macro,
        }

        print(f"    [{hid}] Pure 50 Test Macro Dice: {avg_macro:.2f}% (Median HD95: {med_hd_macro:.2f} mm)\n", flush=True)

    # 6. Print publication-ready summary table
    print("\n================================================================================================================", flush=True)
    print("=== STANDARDIZED PURE 50-PATIENT HELD-OUT TEST EVALUATION SUMMARY TABLE ===", flush=True)
    print("================================================================================================================", flush=True)
    print(f"{'Hospital':<10} {'Track':<12} {'Modalities':<22} {'WT (%)':<10} {'TC (%)':<10} {'ET (%)':<10} {'Avg (Macro %)':<16} {'HD95 Med (mm)':<14}", flush=True)
    print("-" * 115, flush=True)

    macro_dices = []
    for hid, s in hospital_summary.items():
        print(f"{hid:<10} {s['track']:<12} {s['modalities']:<22} {s['WT']:<10.2f} {s['TC']:<10.2f} {s['ET']:<10.2f} {s['Macro']:<16.2f} {s['HD95_Median']:<14.2f}", flush=True)
        macro_dices.append(s["Macro"])

    if macro_dices:
        overall_mean = float(np.mean(macro_dices))
        print("-" * 115, flush=True)
        print(f"{'MEAN':<10} {'---':<12} {'---':<22} {float(np.mean([s['WT'] for s in hospital_summary.values()])):<10.2f} {float(np.mean([s['TC'] for s in hospital_summary.values()])):<10.2f} {float(np.mean([s['ET'] for s in hospital_summary.values()])):<10.2f} {overall_mean:<16.2f}", flush=True)
        print("================================================================================================================", flush=True)

    print(f"\nDetailed per-patient pure 3D metrics saved to: {eval_csv_path}\n", flush=True)


if __name__ == "__main__":
    evaluate_pure_50_test_set()
