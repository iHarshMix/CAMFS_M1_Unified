"""
CAMFS M2 Bottleneck Diagnostic
==============================
Pinpoints the exact bottleneck in M2 CDRD knowledge transfer:
  1. Teacher Oracle bound (how well could a model do with T1ce on these 50 test patients?)
  2. Adapter residual alignment (did the adapter actually learn delta = ell_T - ell_B?)
  3. Gated vs Raw vs ET-only contribution on the 50 sacred test patients.
  4. Full alpha_ET sweep and uniform alpha sweep.
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_yaml
from src.data.brats_dataset import BraTSDataset
from src.metrics import backmap_model_to_brats, extract_tumor_region_mask, compute_3d_dice
from src.m2.adapter import CDRDAdapter
from src.m2.calibration import RecipientGate
from src.m2.frozen_base import load_frozen_base
from src.m2.splits import compute_m2_splits
from src.m2.teacher import build_restricted_teacher


def compute_fast_metrics(pred_model_labels_3d: np.ndarray, target_3d: np.ndarray) -> Dict[str, float]:
    """Fast 3D Dice calculation for WT, TC, ET without slow HD95 edt."""
    pred_brats = backmap_model_to_brats(pred_model_labels_3d.astype(np.uint8))
    target_brats = target_3d.astype(np.uint8)
    if np.max(target_brats) == 3:
        target_brats = backmap_model_to_brats(target_brats)

    res = {}
    for region in ("WT", "TC", "ET"):
        p_mask = extract_tumor_region_mask(pred_brats, region)
        g_mask = extract_tumor_region_mask(target_brats, region)
        res[f"dice_{region}"] = compute_3d_dice(p_mask, g_mask)

    res["macro_dice"] = float((res["dice_WT"] + res["dice_TC"] + res["dice_ET"]) / 3.0)
    return res


def main():
    parser = argparse.ArgumentParser(description="M2 CDRD Bottleneck Diagnostic")
    parser.add_argument("--config", type=str, default="configs/m2_cdrd.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["paths"]["output_dir"])

    print("=" * 80)
    print("  CAMFS M2 CDRD BOTTLENECK DIAGNOSTIC")
    print(f"  Device: {device}")
    print("=" * 80)

    # 1. Load splits
    seed = cfg["experiment"]["partition_seed"]
    partition_file = Path(cfg["paths"]["partitions_dir"]) / f"partition_{seed}.json"
    splits = compute_m2_splits(
        partition_path=partition_file,
        test_patients_path=cfg["paths"]["h3_test_patients"],
        split_seed=cfg["experiment"]["split_seed"],
    )
    test_pids = splits.h3.sacred_test_ids
    print(f"  Test patients: {len(test_pids)}")

    # 2. Load Models
    print("  Loading models...")
    frozen_base = load_frozen_base(cfg["paths"]["phase1_checkpoint"], cfg["paths"]["track_s3_checkpoint"], device=device)
    frozen_base.eval()

    adapter = CDRDAdapter().to(device)
    adapter.load_state_dict(torch.load(out_dir / "adapter_best.pt", map_location=device)["state_dict"])
    adapter.eval()

    gate = RecipientGate().to(device)
    gate.load_state_dict(torch.load(out_dir / "recipient_gate.pt", map_location=device)["state_dict"])
    gate.eval()
    learned_alphas = gate.alphas.detach().cpu()
    print(f"  Learned gate alphas: {learned_alphas.numpy()}")
    print(f"  Alphas: BG={learned_alphas[0]:.4f}, NCR={learned_alphas[1]:.4f}, ED={learned_alphas[2]:.4f}, ET={learned_alphas[3]:.4f}")

    # Load Teacher (Donor Oracle)
    teacher = build_restricted_teacher(
        cfg["paths"]["phase1_checkpoint"],
        cfg["paths"]["track_s3_checkpoint"],
        device=device,
    )
    teacher.load_state_dict(torch.load(out_dir / "teacher_best.pt", map_location=device)["state_dict"])
    teacher.eval()
    print("  Models loaded successfully.")

    # 3. Setup Dataset
    dataset = BraTSDataset(cache_root=cfg["paths"]["preprocessed_dir"], patient_ids=test_pids)

    # Define all conditions to evaluate
    conditions = {
        "Base (no adapter)": torch.zeros(4),
        "Teacher (Oracle T1ce)": "TEACHER_ORACLE",
        "Gated Adapter (Current)": learned_alphas.clone(),
        "Raw Adapter (alpha=1.0)": torch.ones(4),
        "ET-only (alpha_ET=1.0)": torch.tensor([0.0, 0.0, 0.0, 1.0]),
        "ET-only (alpha_ET=0.5)": torch.tensor([0.0, 0.0, 0.0, 0.5]),
        "ET-only (alpha_ET=2.0)": torch.tensor([0.0, 0.0, 0.0, 2.0]),
        "Core_Masked_Gated": "CORE_MASKED_GATED",
        "Core_Masked_Raw": "CORE_MASKED_RAW",
        "WT_Masked_Gated": "WT_MASKED_GATED",
        "WT_Masked_Raw": "WT_MASKED_RAW",
    }

    # Alpha_ET sweep conditions (learned for BG, NCR, ED; sweep ET)
    et_sweep_vals = [0.0, 0.05, 0.1, 0.167, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    for val in et_sweep_vals:
        c = learned_alphas.clone()
        c[3] = val
        conditions[f"Sweep_alpha_ET_{val}"] = c

    # Uniform alpha sweep conditions
    uniform_vals = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    for val in uniform_vals:
        conditions[f"Uniform_alpha_{val}"] = torch.full((4,), val)

    # Storage for patient metrics per condition
    results = {name: {"ET": [], "TC": [], "WT": [], "Macro": []} for name in conditions}

    # Storage for logit diagnostics
    cos_sim_list = []
    ratio_delta_to_base_list = []
    delta_et_in_gt_et = []
    delta_et_outside_gt_et = []

    print(f"\nEvaluating {len(test_pids)} patients across {len(conditions)} conditions...")
    t0 = time.time()

    with torch.no_grad():
        for p_idx, pid in enumerate(test_pids):
            vol = dataset.load_patient_volume(pid)
            lab_vol = vol["labels"]
            num_slices = lab_vol.shape[0]

            logits_b_list = []
            delta_ell_list = []
            logits_t_list = []

            for start in range(0, num_slices, args.batch_size):
                end = min(start + args.batch_size, num_slices)
                t1 = torch.from_numpy(vol["modalities"]["T1"][start:end]).float().unsqueeze(1).to(device)
                flair = torch.from_numpy(vol["modalities"]["FLAIR"][start:end]).float().unsqueeze(1).to(device)
                t1ce = torch.from_numpy(vol["modalities"]["T1ce"][start:end]).float().unsqueeze(1).to(device)

                x_base = {"T1": t1, "FLAIR": flair}
                x_teach = {"T1": t1, "T1ce": t1ce, "FLAIR": flair}

                feat_b, lb = frozen_base(x_base)
                de, _ = adapter(feat_b, lb)
                lt = teacher(x_teach)

                logits_b_list.append(lb.cpu())
                delta_ell_list.append(de.cpu())
                logits_t_list.append(lt.cpu())

            # Full 3D tensors: (155, 4, 240, 240)
            logits_b_3d = torch.cat(logits_b_list, dim=0)
            delta_ell_3d = torch.cat(delta_ell_list, dim=0)
            logits_t_3d = torch.cat(logits_t_list, dim=0)

            # --- Logit Diagnostic on slice with maximum tumor ---
            target_tensor = torch.from_numpy(lab_vol).long()
            et_mask = (target_tensor == 3)
            if et_mask.sum() > 0:
                # Delta ET magnitude where ET is present vs not
                de_et = delta_ell_3d[:, 3]
                delta_et_in_gt_et.append(float(de_et[et_mask].mean()))
                delta_et_outside_gt_et.append(float(de_et[~et_mask].mean()))

                # Cosine similarity between delta_ell and ideal residual (logits_t - logits_b)
                ideal_residual = logits_t_3d - logits_b_3d
                flat_de = delta_ell_3d.reshape(-1)
                flat_res = ideal_residual.reshape(-1)
                sim = F.cosine_similarity(flat_de.unsqueeze(0), flat_res.unsqueeze(0)).item()
                cos_sim_list.append(sim)

                ratio = float(delta_ell_3d.abs().mean() / (logits_b_3d.abs().mean() + 1e-8))
                ratio_delta_to_base_list.append(ratio)

            # Evaluate each condition
            base_pred = logits_b_3d.argmax(dim=1)
            core_mask = ((base_pred == 1) | (base_pred == 3)).unsqueeze(1).float()  # (155, 1, 240, 240)
            wt_mask = (base_pred != 0).unsqueeze(1).float()

            for cond_name, alphas in conditions.items():
                if isinstance(alphas, str):
                    if alphas == "TEACHER_ORACLE":
                        pred_class = logits_t_3d.argmax(dim=1).numpy()
                    elif alphas == "CORE_MASKED_GATED":
                        delta_m = delta_ell_3d.clone()
                        delta_m[:, 3:4] = delta_m[:, 3:4] * core_mask
                        a_bcast = learned_alphas.view(1, 4, 1, 1)
                        logits_aug = logits_b_3d + a_bcast * delta_m
                        pred_class = logits_aug.argmax(dim=1).numpy()
                    elif alphas == "CORE_MASKED_RAW":
                        delta_m = delta_ell_3d.clone()
                        delta_m[:, 3:4] = delta_m[:, 3:4] * core_mask
                        logits_aug = logits_b_3d + delta_m
                        pred_class = logits_aug.argmax(dim=1).numpy()
                    elif alphas == "WT_MASKED_GATED":
                        delta_m = delta_ell_3d.clone()
                        delta_m[:, 3:4] = delta_m[:, 3:4] * wt_mask
                        a_bcast = learned_alphas.view(1, 4, 1, 1)
                        logits_aug = logits_b_3d + a_bcast * delta_m
                        pred_class = logits_aug.argmax(dim=1).numpy()
                    elif alphas == "WT_MASKED_RAW":
                        delta_m = delta_ell_3d.clone()
                        delta_m[:, 3:4] = delta_m[:, 3:4] * wt_mask
                        logits_aug = logits_b_3d + delta_m
                        pred_class = logits_aug.argmax(dim=1).numpy()
                else:
                    # alphas broadcast: (1, 4, 1, 1)
                    a_bcast = alphas.view(1, 4, 1, 1)
                    logits_aug = logits_b_3d + a_bcast * delta_ell_3d
                    pred_class = logits_aug.argmax(dim=1).numpy()

                m = compute_fast_metrics(pred_class, lab_vol)
                results[cond_name]["ET"].append(m["dice_ET"])
                results[cond_name]["TC"].append(m["dice_TC"])
                results[cond_name]["WT"].append(m["dice_WT"])
                results[cond_name]["Macro"].append(m["macro_dice"])

            if (p_idx + 1) % 10 == 0 or (p_idx + 1) == len(test_pids):
                print(f"  [{p_idx+1:02d}/{len(test_pids)}] patients processed ({time.time()-t0:.1f}s)")

    # 4. Compile and Print Results
    base_et = float(np.mean(results["Base (no adapter)"]["ET"])) * 100
    base_macro = float(np.mean(results["Base (no adapter)"]["Macro"])) * 100

    print("\n" + "=" * 85)
    print("  CAMFS M2 CDRD DIAGNOSTIC SUMMARY: 50 SACRED H3 PATIENTS")
    print("=" * 85)
    print(f"  {'Condition':<28} {'ET (%)':<12} {'TC (%)':<12} {'WT (%)':<12} {'Macro (%)':<12} {'Delta_ET':<10}")
    print("-" * 85)

    summary_data = {}
    for cond_name in [
        "Base (no adapter)",
        "Teacher (Oracle T1ce)",
        "Gated Adapter (Current)",
        "Raw Adapter (alpha=1.0)",
        "ET-only (alpha_ET=1.0)",
        "ET-only (alpha_ET=0.5)",
        "ET-only (alpha_ET=2.0)",
        "Core_Masked_Gated",
        "Core_Masked_Raw",
        "WT_Masked_Gated",
        "WT_Masked_Raw",
    ]:
        et_m = float(np.mean(results[cond_name]["ET"])) * 100
        tc_m = float(np.mean(results[cond_name]["TC"])) * 100
        wt_m = float(np.mean(results[cond_name]["WT"])) * 100
        mac_m = float(np.mean(results[cond_name]["Macro"])) * 100
        d_et = et_m - base_et
        summary_data[cond_name] = {"ET": et_m, "TC": tc_m, "WT": wt_m, "Macro": mac_m, "Delta_ET": d_et}
        print(f"  {cond_name:<28} {et_m:<12.2f} {tc_m:<12.2f} {wt_m:<12.2f} {mac_m:<12.2f} {d_et:+<10.2f}%")

    print("\n" + "-" * 85)
    print("  SWEEP: alpha_ET (Learned alphas for BG, NCR, ED; vary alpha_ET)")
    print("-" * 85)
    print(f"  {'alpha_ET':<12} {'ET (%)':<12} {'TC (%)':<12} {'WT (%)':<12} {'Macro (%)':<12} {'Delta_ET':<10}")
    print("-" * 85)

    best_et = -1.0
    best_alpha_et = None
    for val in et_sweep_vals:
        c_name = f"Sweep_alpha_ET_{val}"
        et_m = float(np.mean(results[c_name]["ET"])) * 100
        tc_m = float(np.mean(results[c_name]["TC"])) * 100
        wt_m = float(np.mean(results[c_name]["WT"])) * 100
        mac_m = float(np.mean(results[c_name]["Macro"])) * 100
        d_et = et_m - base_et
        marker = " <-- BEST" if et_m > best_et else ""
        if et_m > best_et:
            best_et = et_m
            best_alpha_et = val
        print(f"  {val:<12.3f} {et_m:<12.2f} {tc_m:<12.2f} {wt_m:<12.2f} {mac_m:<12.2f} {d_et:+<10.2f}%{marker}")

    print("\n" + "-" * 85)
    print("  SWEEP: Uniform Alpha (same alpha for ALL 4 classes)")
    print("-" * 85)
    print(f"  {'alpha':<12} {'ET (%)':<12} {'TC (%)':<12} {'WT (%)':<12} {'Macro (%)':<12} {'Delta_Macro':<12}")
    print("-" * 85)
    for val in uniform_vals:
        c_name = f"Uniform_alpha_{val}"
        et_m = float(np.mean(results[c_name]["ET"])) * 100
        tc_m = float(np.mean(results[c_name]["TC"])) * 100
        wt_m = float(np.mean(results[c_name]["WT"])) * 100
        mac_m = float(np.mean(results[c_name]["Macro"])) * 100
        d_mac = mac_m - base_macro
        print(f"  {val:<12.2f} {et_m:<12.2f} {tc_m:<12.2f} {wt_m:<12.2f} {mac_m:<12.2f} {d_mac:+<12.2f}%")

    # 5. Logit Diagnostic Details
    print("\n" + "=" * 85)
    print("  ADAPTER RESIDUAL ALIGNMENT DIAGNOSTIC")
    print("=" * 85)
    print(f"  Mean |delta_ell| / |logits_B| ratio:       {np.mean(ratio_delta_to_base_list):.4f}")
    print(f"  Cosine Similarity (delta_ell, ell_T-ell_B): {np.mean(cos_sim_list):.4f}")
    print(f"  Mean delta_ET inside GT ET voxels:          {np.mean(delta_et_in_gt_et):+.4f}")
    print(f"  Mean delta_ET outside GT ET voxels:         {np.mean(delta_et_outside_gt_et):+.4f}")
    print("=" * 85)

    # Save to JSON
    out_json = out_dir / "diagnostic_results.json"
    diag_summary = {
        "summary": summary_data,
        "best_alpha_ET": {"alpha": best_alpha_et, "ET": best_et, "Delta_ET": best_et - base_et},
        "residual_diagnostic": {
            "mean_ratio_delta_to_base": float(np.mean(ratio_delta_to_base_list)),
            "mean_cos_similarity": float(np.mean(cos_sim_list)),
            "delta_et_inside_gt": float(np.mean(delta_et_in_gt_et)),
            "delta_et_outside_gt": float(np.mean(delta_et_outside_gt_et)),
        }
    }
    with open(out_json, "w") as f:
        json.dump(diag_summary, f, indent=2)
    print(f"\n  Saved full diagnostic results to {out_json}\n")


if __name__ == "__main__":
    main()
