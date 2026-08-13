"""
CAMFS M1 — Held-Out 3D Test Set Evaluator
===========================================

Executes end-to-end 3D patient-level evaluation on the held-out test set
across all four hospital cohorts:
  - Hospital 1 (H1): Track S1 = {T1, T1ce, T2, FLAIR} -> best_track_S1.pt
  - Hospital 2 (H2): Track_local,2 = {T1, T1ce, T2}   -> best_H2_local_head.pt
  - Hospital 3 (H3): Track S3 = {T1, FLAIR}           -> best_track_S3.pt
  - Hospital 4 (H4): Track S4 = {T1, T1ce, T2}         -> best_track_S4.pt

Output:
  - Per-patient 3D metrics logged to results/test_patient_metrics.csv
  - Hospital summary table (Mean ± Std for Dice & HD95) printed to console
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
    parser = argparse.ArgumentParser(description="CAMFS M1 Held-Out 3D Test Set Evaluator")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--partition-seed", type=int, default=1103, help="Partition seed (1103, 2207, 3301)")
    parser.add_argument("--train-seed", type=int, default=17, help="Training seed (17, 29, 43)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--preprocessed-dir", type=str, default="outputs/preprocessed", help="Path to preprocessed data")
    parser.add_argument("--partitions-dir", type=str, default="outputs/partitions", help="Path to partition manifests")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Root output directory")
    return parser.parse_args()


def get_val_patient_ids(all_patient_ids: List[str]) -> List[str]:
    """
    Reproduce the exact 80/20 train/val split used by FederatedClient.
    The last 20% of patient_ids are the validation (held-out) patients.
    """
    n_val = max(1, int(len(all_patient_ids) * 0.2)) if len(all_patient_ids) > 1 else 0
    return all_patient_ids[-n_val:] if n_val > 0 else all_patient_ids


def evaluate_test_set():
    args = parse_args()
    config = load_yaml(args.config)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    print(f"=== Execution Device: {device} (CUDA Available: {torch.cuda.is_available()}) ===", flush=True)

    experiment_id = f"camfs_primary__part{args.partition_seed}__seed{args.train_seed}"
    chkpt_dir = Path(args.output_dir) / "checkpoints" / experiment_id
    results_dir = Path(args.output_dir) / "results" / experiment_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load partition manifest
    partition_file = Path(args.partitions_dir) / f"partition_{args.partition_seed}.json"
    if not partition_file.exists():
        raise FileNotFoundError(f"Partition manifest not found: {partition_file}")

    with open(partition_file, "r") as f:
        partition_data = json.load(f)

    hospitals_config = partition_data["hospitals"]

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

    # 3. Determine evaluation patient IDs per hospital
    # For H1, H2, H4: use the validation split (last 20% of patient_ids, same as FederatedClient)
    # For H3: use fixed_test_patient_ids if available (50 held-out patients from §partition config)
    eval_patient_ids = {}
    for hid in ["H1", "H2", "H3", "H4"]:
        h_info = hospitals_config[hid]
        if "fixed_test_patient_ids" in h_info and h_info["fixed_test_patient_ids"]:
            eval_patient_ids[hid] = h_info["fixed_test_patient_ids"]
        else:
            all_pids = h_info["patient_ids"]
            eval_patient_ids[hid] = get_val_patient_ids(all_pids)

    print(f"\n=== Evaluation Patient Counts ===", flush=True)
    for hid, pids in eval_patient_ids.items():
        print(f"  {hid}: {len(pids)} test/validation patients", flush=True)

    # 4. Map hospital evaluation models & checkpoints
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

    # 5. Initialize CSV Logger & Evaluator
    eval_csv_path = results_dir / "test_patient_metrics.csv"
    eval_logger = EvaluationCSVLogger(eval_csv_path)
    evaluator = PatientEvaluator()

    hospital_summary = {}

    # 6. Evaluate each hospital cohort on its held-out test split
    print(f"\n=======================================================================", flush=True)
    print(f"=== STARTING 3D HELD-OUT TEST EVALUATION ACROSS ALL 4 HOSPITALS ===", flush=True)
    print(f"=======================================================================\n", flush=True)

    for hid, h_spec in eval_specs.items():
        test_pids = eval_patient_ids[hid]

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
        print(f"--- Evaluator [{hid}] Track {h_spec['track_id']} ({len(test_pids)} Test Patients) ---", flush=True)
        print(f"    Loaded {chkpt_path.name} (Round/Epoch {round_num}, Peak Val Dice: {val_dice*100:.2f}%)", flush=True)

        # Setup dataset for this hospital's test patients
        dataset = BraTSDataset(
            patient_ids=test_pids,
            cache_root=args.preprocessed_dir,
            modalities=h_spec["modalities"],
        )

        h_patient_metrics = []

        with torch.no_grad():
            for pid_idx, pid in enumerate(test_pids):
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

                if (pid_idx + 1) % 5 == 0 or (pid_idx + 1) == len(test_pids):
                    print(f"    Processed {pid_idx+1}/{len(test_pids)} test patients ({pid}) - "
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

        avg_hd_macro = float(np.mean([m["macro_hd95"] for m in h_patient_metrics]))
        std_hd_macro = float(np.std([m["macro_hd95"] for m in h_patient_metrics]))

        hospital_summary[hid] = {
            "track": h_spec["track_id"],
            "modalities": "+".join(h_spec["modalities"]),
            "n_test": len(test_pids),
            "ET": (avg_ET, std_ET),
            "TC": (avg_TC, std_TC),
            "WT": (avg_WT, std_WT),
            "Macro": (avg_macro, std_macro),
            "HD95": (avg_hd_macro, std_hd_macro),
        }

        print(f"    [{hid}] Mean Macro Dice: {avg_macro:.2f}% ± {std_macro:.2f}%\n", flush=True)

    # 7. Print publication-ready summary table
    print("\n==========================================================================================================", flush=True)
    print("=== CAMFS M1 HELD-OUT 3D TEST EVALUATION SUMMARY TABLE ===", flush=True)
    print("==========================================================================================================", flush=True)
    print(f"{'Hospital':<10} {'Track':<12} {'Modalities':<22} {'N':<5} {'ET Dice (%)':<18} {'TC Dice (%)':<18} {'WT Dice (%)':<18} {'Macro Dice (%)':<18} {'HD95 (mm)':<16}", flush=True)
    print("-" * 137, flush=True)

    macro_dices = []
    for hid, s in hospital_summary.items():
        et_str = f"{s['ET'][0]:.2f} ± {s['ET'][1]:.2f}"
        tc_str = f"{s['TC'][0]:.2f} ± {s['TC'][1]:.2f}"
        wt_str = f"{s['WT'][0]:.2f} ± {s['WT'][1]:.2f}"
        macro_str = f"{s['Macro'][0]:.2f} ± {s['Macro'][1]:.2f}"
        hd_str = f"{s['HD95'][0]:.2f} ± {s['HD95'][1]:.2f}"
        print(f"{hid:<10} {s['track']:<12} {s['modalities']:<22} {s['n_test']:<5} {et_str:<18} {tc_str:<18} {wt_str:<18} {macro_str:<18} {hd_str:<16}", flush=True)
        macro_dices.append(s['Macro'][0])

    if macro_dices:
        overall_mean = float(np.mean(macro_dices))
        print("-" * 137, flush=True)
        print(f"{'MEAN':<10} {'---':<12} {'---':<22} {sum(s['n_test'] for s in hospital_summary.values()):<5} {'---':<18} {'---':<18} {'---':<18} {overall_mean:.2f}%", flush=True)
        print("==========================================================================================================", flush=True)

    print(f"\nDetailed per-patient 3D metrics saved to: {eval_csv_path}\n", flush=True)


if __name__ == "__main__":
    evaluate_test_set()
