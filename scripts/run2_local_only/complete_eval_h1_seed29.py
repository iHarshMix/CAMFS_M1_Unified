#!/usr/bin/env python3
"""
Complete remaining test evaluation for Local-Only H1 Seed 29.
Evaluates any test patient missing from outputs/results/local_only_H1__part1103__seed29/pure_50_test_patient_metrics.csv.
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.brats_dataset import BraTSDataset
from src.logging import CheckpointManager, EvaluationCSVLogger
from src.metrics import PatientEvaluator
from src.models import SubsetFusionHead, UNetDecoder, UnimodalEncoder

HOSPITAL_ID = "H1"
MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]
PARTITION_SEED = 1103
TRAIN_SEED = 29
RUN_ID = f"local_only_{HOSPITAL_ID}__part{PARTITION_SEED}__seed{TRAIN_SEED}"

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Completing Test Evaluation for {RUN_ID} on {device} ===")

    chkpt_path = Path("outputs/checkpoints") / RUN_ID / "best_model.pt"
    results_dir = Path("outputs/results") / RUN_ID
    csv_path = results_dir / "pure_50_test_patient_metrics.csv"
    preprocessed_path = Path("outputs/preprocessed")

    if not chkpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {chkpt_path}")

    # Build models
    encoders = {m: UnimodalEncoder(in_channels=1).to(device) for m in MODALITIES}
    fusion_head = SubsetFusionHead(modality_subset=MODALITIES).to(device)
    decoder = UNetDecoder(num_classes=4).to(device)

    # Load checkpoint
    chkpt_data = CheckpointManager.load_checkpoint(chkpt_path, device=device)
    state = chkpt_data["state_dict"]
    for m, enc_s in state.get("encoders", {}).items():
        if m in encoders:
            encoders[m].load_state_dict(enc_s)
    if "fusion" in state:
        fusion_head.load_state_dict(state["fusion"])
    if "decoder" in state:
        decoder.load_state_dict(state["decoder"])

    for enc in encoders.values():
        enc.eval()
    fusion_head.eval()
    decoder.eval()
    print(f"Loaded best checkpoint (saved at epoch {chkpt_data.get('metadata', {}).get('best_epoch', 'unknown')})")

    # Determine missing patients
    test_manifest_path = Path("outputs/partitions/h3_test_patients.json")
    with open(test_manifest_path, "r") as f:
        manifest = json.load(f)
    all_test_pids = manifest["patient_ids"] if isinstance(manifest, dict) and "patient_ids" in manifest else manifest

    existing_pids = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_pids.add(r["patient_id"])

    missing_pids = [p for p in all_test_pids if p not in existing_pids]
    print(f"Total test patients: {len(all_test_pids)} | Already evaluated: {len(existing_pids)} | Missing: {len(missing_pids)}")

    if not missing_pids:
        print("All 50 patients are already evaluated!")
    else:
        print(f"Evaluating {len(missing_pids)} missing patients: {missing_pids}")
        test_dataset = BraTSDataset(patient_ids=missing_pids, cache_root=preprocessed_path, modalities=MODALITIES)
        evaluator = PatientEvaluator()
        eval_logger = EvaluationCSVLogger(csv_path)

        with torch.no_grad():
            for pid in missing_pids:
                print(f"  Evaluating {pid}...", flush=True)
                vol = test_dataset.load_patient_volume(pid)
                lab_vol = vol["labels"]
                num_slices = lab_vol.shape[0]

                pred_logits_list = []
                batch_size = 4
                for start in range(0, num_slices, batch_size):
                    end = min(start + batch_size, num_slices)
                    mod_feats_b = {}
                    for m in MODALITIES:
                        x_m = torch.from_numpy(vol["modalities"][m][start:end]).float().unsqueeze(1).to(device)
                        h1, h2, h3, h4, z = encoders[m](x_m)
                        mod_feats_b[m] = (h1, h2, h3, h4, z)
                    f1, f2, f3, f4, z_S_b = fusion_head(mod_feats_b)
                    logits_b = decoder(z_S=z_S_b, f_S_4=f4, f_S_3=f3, f_S_2=f2, f_S_1=f1)
                    pred_logits_list.append(logits_b.cpu())

                pred_logits_3d = torch.cat(pred_logits_list, dim=0).permute(1, 0, 2, 3).numpy()
                p_metrics = evaluator.evaluate_patient_volume(pred_logits_3d, lab_vol)

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
                    track_id=f"{HOSPITAL_ID}_local",
                    hospital_id=HOSPITAL_ID,
                    dice_dict=dice_dict,
                    hd95_dict=hd95_dict,
                )
                print(f"    Done {pid}: Macro Dice = {p_metrics['macro_dice']*100:.2f}%", flush=True)

    # Compute final full 50-patient statistics
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "macro": float(r["dice_macro"]),
                "WT": float(r["dice_WT"]),
                "TC": float(r["dice_TC"]),
                "ET": float(r["dice_ET"]),
                "hd_macro": float(r["hd95_macro"]),
            })

    macro_dice = np.mean([r["macro"] for r in rows])
    wt_dice = np.mean([r["WT"] for r in rows])
    tc_dice = np.mean([r["TC"] for r in rows])
    et_dice = np.mean([r["ET"] for r in rows])
    med_hd95 = np.median([r["hd_macro"] for r in rows])

    print(f"\n=== FINAL TEST RESULTS FOR {RUN_ID} (N={len(rows)}) ===")
    print(f"  Macro Dice:  {macro_dice*100:.2f}%")
    print(f"  WT Dice:     {wt_dice*100:.2f}%")
    print(f"  TC Dice:     {tc_dice*100:.2f}%")
    print(f"  ET Dice:     {et_dice*100:.2f}%")
    print(f"  Median HD95: {med_hd95:.2f} mm")

if __name__ == "__main__":
    main()
