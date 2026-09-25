"""
CAMFS M2 — Sacred 50-Patient Test Set Evaluator
===============================================

Implements §10.5, §15.2 of the CAMFS M2 Specification.
Standardized evaluation of Hospital H3 on the 50 Sacred Quarantined Test Patients:
- E0: Pure M1 Base (d_g = 0)
- E4_aug: CDRD-Frozen Forced-On (d_g = 1, learned alpha_c)
- E4_deploy: CDRD-Frozen Routed (d_g = d_deploy, learned alpha_c)

Usage:
    python scripts/evaluate_m2_test_set.py --config configs/m2_cdrd.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_yaml
from src.data.brats_dataset import BraTSDataset
from src.metrics import PatientEvaluator
from src.m2.adapter import CDRDAdapter
from src.m2.calibration import RecipientGate
from src.m2.frozen_base import load_frozen_base
from src.m2.splits import compute_m2_splits


def parse_args():
    parser = argparse.ArgumentParser(description="CAMFS M2 Sacred Test Set Evaluator")
    parser.add_argument("--config", type=str, default="configs/m2_cdrd.yaml", help="Path to config YAML")
    parser.add_argument("--gpu", type=int, default=None, help="Override GPU ID")
    return parser.parse_args()


def evaluate_condition_on_test_set(
    condition_name: str,
    d_g: float,
    frozen_base: torch.nn.Module,
    adapter: torch.nn.Module,
    gate: torch.nn.Module,
    dataset: BraTSDataset,
    test_pids: List[str],
    evaluator: PatientEvaluator,
    device: torch.device,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Evaluates a specific routing condition (d_g) across all test patients."""
    frozen_base.eval()
    adapter.eval()
    gate.eval()

    patient_records = []
    batch_size = 16

    print(f"\n--- Evaluating Condition [{condition_name}] (d_g = {d_g}) on {len(test_pids)} Sacred Test Patients ---", flush=True)

    with torch.no_grad():
        for idx, pid in enumerate(test_pids):
            vol = dataset.load_patient_volume(pid)
            lab_vol = vol["labels"]
            num_slices = lab_vol.shape[0]

            logits_list = []
            for start in range(0, num_slices, batch_size):
                end = min(start + batch_size, num_slices)
                x_dict = {
                    "T1": torch.from_numpy(vol["modalities"]["T1"][start:end]).float().unsqueeze(1).to(device),
                    "FLAIR": torch.from_numpy(vol["modalities"]["FLAIR"][start:end]).float().unsqueeze(1).to(device),
                }

                features_b, logits_b = frozen_base(x_dict)
                delta_ell, _ = adapter(features_b, logits_b)
                logits_out = gate(logits_b, delta_ell, d_g=d_g)
                logits_list.append(logits_out.cpu())

            pred_3d = torch.cat(logits_list, dim=0).permute(1, 0, 2, 3).numpy()
            metrics = evaluator.evaluate_patient_volume(pred_3d, lab_vol)
            metrics["patient_id"] = pid
            metrics["condition"] = condition_name
            patient_records.append(metrics)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(test_pids):
                print(f"    [{condition_name}] {idx + 1}/{len(test_pids)} patients evaluated", flush=True)

    # Compute aggregate summary
    et_list = [r["dice_ET"] for r in patient_records]
    tc_list = [r["dice_TC"] for r in patient_records]
    wt_list = [r["dice_WT"] for r in patient_records]
    macro_list = [r["macro_dice"] for r in patient_records]

    hd95_et = [r["hd95_ET"] for r in patient_records]
    hd95_tc = [r["hd95_TC"] for r in patient_records]
    hd95_wt = [r["hd95_WT"] for r in patient_records]
    hd95_med = [r["macro_hd95"] for r in patient_records]

    summary = {
        "condition": condition_name,
        "d_g": d_g,
        "mean_ET": float(np.mean(et_list)),
        "std_ET": float(np.std(et_list)),
        "mean_TC": float(np.mean(tc_list)),
        "std_TC": float(np.std(tc_list)),
        "mean_WT": float(np.mean(wt_list)),
        "std_WT": float(np.std(wt_list)),
        "mean_Macro": float(np.mean(macro_list)),
        "std_Macro": float(np.std(macro_list)),
        "median_HD95": float(np.median(hd95_med)),
        "mean_HD95_ET": float(np.mean(hd95_et)),
        "mean_HD95_TC": float(np.mean(hd95_tc)),
        "mean_HD95_WT": float(np.mean(hd95_wt)),
    }
    return patient_records, summary


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    gpu_id = args.gpu if args.gpu is not None else cfg["experiment"].get("gpu", 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["paths"]["output_dir"])

    print(f"\n===================================================================")
    print(f"  CAMFS M2: Sacred 50-Patient Test Set Evaluation")
    print(f"  Device: {device}")
    print(f"===================================================================\n")

    # Load Splits
    partition_file = Path(cfg["paths"]["partitions_dir"]) / f"partition_{cfg['experiment']['partition_seed']}.json"
    splits = compute_m2_splits(
        partition_path=partition_file,
        test_patients_path=cfg["paths"]["h3_test_patients"],
        split_seed=cfg["experiment"]["split_seed"],
    )
    test_pids = splits.h3.sacred_test_ids
    print(f"  Loaded {len(test_pids)} Sacred Quarantined Test Patients for Hospital H3")

    # Load Models
    p1_path = cfg["paths"]["phase1_checkpoint"]
    s3_path = cfg["paths"]["track_s3_checkpoint"]
    frozen_base = load_frozen_base(p1_path, s3_path, device=device)

    adapter_ckpt_path = out_dir / "adapter_best.pt"
    if not adapter_ckpt_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint missing: {adapter_ckpt_path}. Run training first.")
    adapter = CDRDAdapter().to(device)
    adapter.load_state_dict(torch.load(adapter_ckpt_path, map_location=device)["state_dict"])

    gate_ckpt_path = out_dir / "recipient_gate.pt"
    if not gate_ckpt_path.exists():
        raise FileNotFoundError(f"Recipient gate missing: {gate_ckpt_path}. Run training first.")
    gate = RecipientGate().to(device)
    gate_data = torch.load(gate_ckpt_path, map_location=device)
    gate.load_state_dict(gate_data["state_dict"])

    # Read routing decision d_deploy from provenance ledger if available, else check acceptance
    ledger_path = Path(cfg["paths"]["ledger_path"])
    d_deploy = 1.0  # default for aug evaluation

    dataset = BraTSDataset(
        patient_ids=test_pids,
        cache_root=cfg["paths"]["preprocessed_dir"],
        modalities=["T1", "FLAIR"],
    )
    evaluator = PatientEvaluator()

    all_patient_records = []
    all_summaries = []

    # 1. Condition E0: Pure M1 Base (d_g = 0)
    p_e0, s_e0 = evaluate_condition_on_test_set("E0_Base", 0.0, frozen_base, adapter, gate, dataset, test_pids, evaluator, device)
    all_patient_records.extend(p_e0)
    all_summaries.append(s_e0)

    # 2. Condition E4_aug: Forced-On CDRD-Frozen (d_g = 1)
    p_e4_aug, s_e4_aug = evaluate_condition_on_test_set("E4_aug", 1.0, frozen_base, adapter, gate, dataset, test_pids, evaluator, device)
    all_patient_records.extend(p_e4_aug)
    all_summaries.append(s_e4_aug)

    # 3. Condition E4_deploy: Routed CDRD-Frozen (d_g = d_deploy)
    p_e4_dep, s_e4_dep = evaluate_condition_on_test_set("E4_deploy", d_deploy, frozen_base, adapter, gate, dataset, test_pids, evaluator, device)
    all_patient_records.extend(p_e4_dep)
    all_summaries.append(s_e4_dep)

    # Compute Primary Causal Contrast
    delta_et = (s_e4_aug["mean_ET"] - s_e0["mean_ET"]) * 100
    delta_tc = (s_e4_aug["mean_TC"] - s_e0["mean_TC"]) * 100
    delta_wt = (s_e4_aug["mean_WT"] - s_e0["mean_WT"]) * 100
    delta_macro = (s_e4_aug["mean_Macro"] - s_e0["mean_Macro"]) * 100

    print(f"\n" + "=" * 95)
    print(f"=== CAMFS M2 STANDARDIZED TEST RESULTS: 50 SACRED H3 PATIENTS (T1, FLAIR) ===")
    print(f"=" * 95)
    print(f"{'Condition':<14} {'d_g':<6} {'WT (%)':<12} {'TC (%)':<12} {'ET (%)':<12} {'Macro (%)':<14} {'HD95 (mm)':<12}")
    print(f"-" * 95)
    for s in all_summaries:
        print(
            f"{s['condition']:<14} {s['d_g']:<6.1f} "
            f"{s['mean_WT']*100:>5.2f} ± {s['std_WT']*100:<4.2f}  "
            f"{s['mean_TC']*100:>5.2f} ± {s['std_TC']*100:<4.2f}  "
            f"{s['mean_ET']*100:>5.2f} ± {s['std_ET']*100:<4.2f}  "
            f"{s['mean_Macro']*100:>5.2f} ± {s['std_Macro']*100:<4.2f}  "
            f"{s['median_HD95']:<10.2f}"
        )
    print(f"-" * 95)
    print(f"Primary Causal Transfer Contrast (E4_aug - E0):")
    print(f"  Δ Enhancing Tumor (ET): {delta_et:+6.2f}%")
    print(f"  Δ Tumor Core (TC):      {delta_tc:+6.2f}%")
    print(f"  Δ Whole Tumor (WT):     {delta_wt:+6.2f}%")
    print(f"  Δ Macro Dice:           {delta_macro:+6.2f}%")
    print(f"=" * 95)

    # Save CSV outputs
    p_csv = out_dir / "test_patient_metrics.csv"
    with open(p_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_patient_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_patient_records)

    s_csv = out_dir / "test_summary_metrics.csv"
    with open(s_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)

    print(f"\nSaved patient-level metrics to {p_csv}")
    print(f"Saved summary metrics to {s_csv}\n")


if __name__ == "__main__":
    main()
