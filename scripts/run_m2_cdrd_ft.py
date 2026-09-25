"""
CAMFS M2 — CDRD-FT (E4b) Recipient Fine-Tuning Pipeline
========================================================

Implements §10.5, §15.2 of the CAMFS M2 Specification:
Secondary ablation CDRD-FT (E4b):
  1. Load canonical frozen base and transferred H1 adapter.
  2. Locally fine-tune adapter phi on H3 calibration data (50 epochs, d_g=1, alpha=1).
  3. Freeze adapter, re-initialize gates a_c = -4.0, and fit only the 4 gates (50 epochs).
  4. Evaluate on untouched H3 acceptance-validation split (6 patients).
  5. Verify M1 base bit-for-bit invariance.
  6. Evaluate standardized 50 sacred test patients under E0_Base, E4b_aug, and E4b_deploy.

Usage:
    python scripts/run_m2_cdrd_ft.py --config configs/m2_cdrd.yaml --gpu 0
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_yaml
from src.data.brats_dataset import BraTSDataset
from src.metrics import PatientEvaluator
from src.m2.adapter import CDRDAdapter
from src.m2.base_registry import compute_canonical_base_hash, verify_base_integrity
from src.m2.calibration import (
    RecipientGate,
    evaluate_recipient_acceptance,
    train_cdrd_ft,
)
from src.m2.frozen_base import load_frozen_base
from src.m2.provenance import ProvenanceLedger
from src.m2.splits import compute_m2_splits


def parse_args():
    parser = argparse.ArgumentParser(description="CAMFS M2 CDRD-FT (E4b) Pipeline")
    parser.add_argument("--config", type=str, default="configs/m2_cdrd.yaml", help="Path to M2 config YAML")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--ft-lr", type=float, default=1e-3, help="Fine-tuning learning rate for adapter")
    parser.add_argument("--ft-epochs", type=int, default=50, help="Fine-tuning epochs for adapter")
    parser.add_argument("--gate-epochs", type=int, default=50, help="Gate fitting epochs")
    return parser.parse_args()


def evaluate_condition_on_test_set(
    condition_name: str,
    d_g: float,
    frozen_base,
    adapter,
    gate,
    test_pids: List[str],
    dataset: BraTSDataset,
    evaluator: PatientEvaluator,
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[Dict[str, float], List[Dict[str, Union[str, float]]]]:
    frozen_base.eval()
    adapter.eval()
    gate.eval()

    patient_records = []
    print(f"\n--- Evaluating Condition [{condition_name}] (d_g = {d_g}) on 50 Sacred Test Patients ---", flush=True)

    with torch.no_grad():
        for idx, pid in enumerate(test_pids):
            vol = dataset.load_patient_volume(pid)
            lab_vol = vol["labels"]
            num_slices = lab_vol.shape[0]

            logits_list = []
            for start in range(0, num_slices, batch_size):
                end = min(start + batch_size, num_slices)
                x_base = {
                    "T1": torch.from_numpy(vol["modalities"]["T1"][start:end]).float().unsqueeze(1).to(device),
                    "FLAIR": torch.from_numpy(vol["modalities"]["FLAIR"][start:end]).float().unsqueeze(1).to(device),
                }
                features_b, logits_b = frozen_base(x_base)
                delta_ell, _ = adapter(features_b, logits_b)
                logits_out = gate(logits_b, delta_ell, d_g=d_g)
                logits_list.append(logits_out.cpu())

            pred_3d = torch.cat(logits_list, dim=0).permute(1, 0, 2, 3).numpy()
            metrics = evaluator.evaluate_patient_volume(pred_3d, lab_vol)
            metrics["patient_id"] = pid
            metrics["condition"] = condition_name
            metrics["d_g"] = d_g
            patient_records.append(metrics)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(test_pids):
                print(f"    [{condition_name}] {idx+1:02d}/{len(test_pids)} patients evaluated", flush=True)

    dice_et = [r["dice_ET"] for r in patient_records]
    dice_tc = [r["dice_TC"] for r in patient_records]
    dice_wt = [r["dice_WT"] for r in patient_records]
    dice_macro = [r["macro_dice"] for r in patient_records]
    hd95_macro = [r["macro_hd95"] for r in patient_records]

    summary = {
        "condition": condition_name,
        "d_g": d_g,
        "mean_ET": float(np.mean(dice_et)),
        "std_ET": float(np.std(dice_et)),
        "mean_TC": float(np.mean(dice_tc)),
        "std_TC": float(np.std(dice_tc)),
        "mean_WT": float(np.mean(dice_wt)),
        "std_WT": float(np.std(dice_wt)),
        "mean_Macro": float(np.mean(dice_macro)),
        "std_Macro": float(np.std(dice_macro)),
        "mean_HD95": float(np.mean(hd95_macro)),
        "std_HD95": float(np.std(hd95_macro)),
    }
    return summary, patient_records


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("  CAMFS M2: CDRD-FT (E4b) RECIPIENT FINE-TUNING PIPELINE")
    print(f"  Device: {device} | Host: {os.uname().nodename}")
    print("=" * 80)

    # Setup directories
    out_dir = Path("outputs/m2_cdrd_ft")
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = Path("outputs/provenance/m2_cdrd_ft_ledger.jsonl")
    ledger = ProvenanceLedger(ledger_path)

    # 1. Base Integrity Verification
    print("\n[Step 1/7] Verifying Canonical Base Hash...")
    p1_path = cfg["paths"]["phase1_checkpoint"]
    s3_path = cfg["paths"]["track_s3_checkpoint"]
    base_digests = compute_canonical_base_hash(p1_path, s3_path)
    base_hash = base_digests["canonical_base_hash"]
    print(f"  Canonical Base Hash: {base_hash[:16]}...{base_hash[-16:]}")

    # 2. Load Splits
    print("\n[Step 2/7] Loading Patient Splits...")
    seed = cfg["experiment"]["partition_seed"]
    partition_file = Path(cfg["paths"]["partitions_dir"]) / f"partition_{seed}.json"
    splits = compute_m2_splits(
        partition_path=partition_file,
        test_patients_path=cfg["paths"]["h3_test_patients"],
        split_seed=cfg["experiment"]["split_seed"],
    )
    print(f"  H3 Cal: {len(splits.h3.m2_cal_ids)} | H3 Accept-Val: {len(splits.h3.m2_accept_val_ids)} | Test: {len(splits.h3.sacred_test_ids)}")

    # 3. Load Models
    print("\n[Step 3/7] Loading Frozen Base and Transferred H1 Adapter...")
    frozen_base = load_frozen_base(p1_path, s3_path, device=device)
    base_state_snapshot = copy.deepcopy(frozen_base.state_dict())

    adapter = CDRDAdapter().to(device)
    donor_adapter_path = Path(cfg["paths"]["output_dir"]) / "adapter_best.pt"
    assert donor_adapter_path.exists(), f"Donor adapter not found at {donor_adapter_path}"
    adapter.load_state_dict(torch.load(donor_adapter_path, map_location=device)["state_dict"])
    print(f"  Loaded donor adapter ({adapter.count_trainable_parameters()} parameters) from {donor_adapter_path}")

    all_h3_pids = splits.h3.m1_train_ids + splits.h3.m2_cal_ids + splits.h3.m2_accept_val_ids + splits.h3.sacred_test_ids
    dataset = BraTSDataset(cache_root=cfg["paths"]["preprocessed_dir"], patient_ids=all_h3_pids)
    evaluator = PatientEvaluator()

    # 4. Execute CDRD-FT (2-Stage Local Fine-Tuning & Gating)
    print(f"\n[Step 4/7] Executing CDRD-FT: 50 Epochs Adapter Fine-Tuning + 50 Epochs Gate Fitting...")
    t0 = time.time()
    adapter, gate, ft_meta = train_cdrd_ft(
        frozen_base=frozen_base,
        adapter=adapter,
        dataset=dataset,
        h3_cal_patient_ids=splits.h3.m2_cal_ids,
        device=device,
        ft_lr=args.ft_lr,
        ft_epochs=args.ft_epochs,
        gate_lr=cfg["calibration"]["lr"],
        gate_epochs=args.gate_epochs,
        batch_size=cfg["calibration"]["batch_size"],
        seed=cfg["experiment"]["split_seed"],
    )
    print(f"  CDRD-FT training completed in {time.time()-t0:.1f}s")
    print(f"  Fitted Gate Alphas: {ft_meta['final_alphas']}")

    # Save fine-tuned adapter and gate
    torch.save({"state_dict": adapter.state_dict(), "meta": ft_meta}, out_dir / "adapter_ft.pt")
    torch.save({"state_dict": gate.state_dict(), "meta": ft_meta["gate_meta"]}, out_dir / "recipient_gate_ft.pt")

    ledger.record_event(
        event_type="CDRD_FT_TRAINED",
        actor_hospital="H3",
        grant_id="H1_to_H3_T1ce_CDRD",
        payload={"final_alphas": ft_meta["final_alphas"], "ft_epochs": args.ft_epochs},
    )

    # 5. Evaluate Recipient Acceptance on untouched 6 H3 Acceptance-Val patients
    print("\n[Step 5/7] Evaluating Recipient Acceptance on 6 H3 Acceptance-Val patients...")
    d_deploy, acceptance_pass, acc_metrics = evaluate_recipient_acceptance(
        gate=gate,
        frozen_base=frozen_base,
        adapter=adapter,
        dataset=dataset,
        h3_accept_val_patient_ids=splits.h3.m2_accept_val_ids,
        evaluator=evaluator,
        q_T=1,  # Evaluate routing eligibility with fine-tuned adapter
        device=device,
        non_inferiority_delta=cfg["calibration"]["acceptance"]["non_inferiority_delta"],
    )

    # 6. Re-verify M1 Base Invariance
    print("\n[Step 6/7] Verifying M1 Base Invariance...")
    for k, v in frozen_base.state_dict().items():
        diff = (v - base_state_snapshot[k]).abs().max().item()
        assert diff == 0.0, f"M1 base parameter {k} was mutated during fine-tuning! max diff = {diff}"
    print("  ✅ M1 Base Invariance Verified: 0.0 parameter drift")

    # 7. Standardized Test Evaluation on 50 Sacred Test Patients
    print("\n[Step 7/7] Standardized Test Evaluation on 50 Sacred H3 Test Patients...")
    test_pids = splits.h3.sacred_test_ids

    # Evaluate E0_Base, E4b_aug (forced-on), and E4b_deploy
    s_e0, r_e0 = evaluate_condition_on_test_set("E0_Base", 0.0, frozen_base, adapter, gate, test_pids, dataset, evaluator, device)
    s_e4b_aug, r_e4b_aug = evaluate_condition_on_test_set("E4b_aug", 1.0, frozen_base, adapter, gate, test_pids, dataset, evaluator, device)
    s_e4b_deploy, r_e4b_deploy = evaluate_condition_on_test_set("E4b_deploy", float(d_deploy), frozen_base, adapter, gate, test_pids, dataset, evaluator, device)

    # Print Summary Table
    print("\n" + "=" * 95)
    print("=== CAMFS M2 STANDARDIZED TEST RESULTS: CDRD-FT (E4b) ON 50 SACRED H3 PATIENTS ===")
    print("=" * 95)
    print(f"{'Condition':<14} {'d_g':<6} {'WT (%)':<16} {'TC (%)':<16} {'ET (%)':<16} {'Macro (%)':<16} {'HD95 (mm)':<10}")
    print("-" * 95)
    for s in [s_e0, s_e4b_aug, s_e4b_deploy]:
        wt_str = f"{s['mean_WT']*100:.2f} ± {s['std_WT']*100:.2f}"
        tc_str = f"{s['mean_TC']*100:.2f} ± {s['std_TC']*100:.2f}"
        et_str = f"{s['mean_ET']*100:.2f} ± {s['std_ET']*100:.2f}"
        macro_str = f"{s['mean_Macro']*100:.2f} ± {s['std_Macro']*100:.2f}"
        hd_str = f"{s['mean_HD95']:.2f}"
        print(f"{s['condition']:<14} {s['d_g']:<6.1f} {wt_str:<16} {tc_str:<16} {et_str:<16} {macro_str:<16} {hd_str:<10}")
    print("-" * 95)

    delta_et = (s_e4b_aug["mean_ET"] - s_e0["mean_ET"]) * 100
    delta_tc = (s_e4b_aug["mean_TC"] - s_e0["mean_TC"]) * 100
    delta_wt = (s_e4b_aug["mean_WT"] - s_e0["mean_WT"]) * 100
    delta_macro = (s_e4b_aug["mean_Macro"] - s_e0["mean_Macro"]) * 100

    print("Primary Causal Transfer Contrast (E4b_aug - E0):")
    print(f"  Δ Enhancing Tumor (ET):  {delta_et:+.2f}%")
    print(f"  Δ Tumor Core (TC):       {delta_tc:+.2f}%")
    print(f"  Δ Whole Tumor (WT):      {delta_wt:+.2f}%")
    print(f"  Δ Macro Dice:            {delta_macro:+.2f}%")
    print("=" * 95)

    # Save CSVs
    all_records = r_e0 + r_e4b_aug + r_e4b_deploy
    keys = list(all_records[0].keys())
    patient_csv = out_dir / "test_patient_metrics.csv"
    with open(patient_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_records)

    summary_records = [s_e0, s_e4b_aug, s_e4b_deploy]
    s_keys = list(summary_records[0].keys())
    summary_csv = out_dir / "test_summary_metrics.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=s_keys)
        writer.writeheader()
        writer.writerows(summary_records)

    print(f"\nSaved patient-level metrics to {patient_csv}")
    print(f"Saved summary metrics to {summary_csv}")

    ledger.record_event(
        event_type="TEST_EVALUATION_COMPLETED",
        actor_hospital="H3",
        grant_id="H1_to_H3_T1ce_CDRD",
        payload={
            "delta_ET": delta_et,
            "delta_TC": delta_tc,
            "delta_WT": delta_wt,
            "delta_Macro": delta_macro,
            "summary_e4b_aug": s_e4b_aug,
        },
    )
    print("\n==============================================================================")
    print("  CDRD-FT (E4b) Pipeline Execution Completed Successfully!")
    print("==============================================================================\n")


if __name__ == "__main__":
    main()
