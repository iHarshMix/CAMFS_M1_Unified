"""
CAMFS M1 — Local-Only Baseline Runner Script
============================================

Implements RUN-2 of the CAMFS M1 Execution Plan.
Each hospital trains completely alone on its own local data partition:
  - No federation, no shared server, no global models.
  - H1: T1, T1ce, T2, FLAIR (102 train / 25 val)
  - H2: T1, T1ce, T2 (64 train / 16 val) — local T1ce included!
  - H3: T1, FLAIR (52 train / 12 val)
  - H4: T1, T1ce, T2 (38 train / 9 val)

Architecture:
  - Same UnimodalEncoder per owned modality.
  - Same SubsetFusionHead for the owned modality subset.
  - Same UNetDecoder (4 classes).
  - Trained end-to-end with AdamW (1e-3 LR) and SoftDiceCrossEntropyLoss.

Evaluation:
  - Evaluated on the universal 50 held-out test patients (pure_50_test_pids).
  - Evaluated strictly using PatientEvaluator (155-slice stacked 3D volume Dice & HD95).
  - Output: pure_50_test_patient_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce deterministic environment variables before torch initialization
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from src.config import load_yaml
from src.data.augmentation import PairwiseAugmentation
from src.data.brats_dataset import BraTSDataset
from src.data.slice_sampler import SliceSampler
from src.logging import CheckpointManager, EvaluationCSVLogger
from src.losses import SoftDiceCrossEntropyLoss
from src.metrics import PatientEvaluator, compute_3d_dice_tensor
from src.models import SubsetFusionHead, UNetDecoder, UnimodalEncoder
from src.seed import set_deterministic

# Canonical hospital modality ownership (§B.2)
HOSPITAL_MODALITIES = {
    "H1": ["T1", "T1ce", "T2", "FLAIR"],
    "H2": ["T1", "T1ce", "T2"],
    "H3": ["T1", "FLAIR"],
    "H4": ["T1", "T1ce", "T2"],
}


def dump_environment_info(output_path: Path) -> None:
    """Dump system environment, PyTorch, CUDA, and library versions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = [
        f"Python Version: {sys.version}",
        f"Platform: {platform.platform()}",
        f"PyTorch Version: {torch.__version__}",
        f"CUDA Available: {torch.cuda.is_available()}",
        f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}",
        f"Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"NumPy Version: {np.__version__}",
        f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}",
        f"CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}",
    ]
    output_path.write_text("\n".join(info) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="CAMFS M1 Local-Only Baseline Runner")
    parser.add_argument("--hospital", type=str, required=True, choices=["H1", "H2", "H3", "H4"], help="Hospital ID")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--policy-config", type=str, default="configs/policy_M1_PRIMARY_V1.json", help="Path to policy JSON")
    parser.add_argument("--partition-seed", type=int, default=1103, help="Partition seed (1103, 2207, 3301)")
    parser.add_argument("--train-seed", type=int, default=17, help="Training seed (17, 29, 43)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--max-gpu-memory-gb", type=float, default=75.0, help="Max GPU VRAM memory limit in GB")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max local training epochs")
    parser.add_argument("--min-epochs", type=int, default=20, help="Minimum epochs before early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for 2D slices")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (2 epochs, 2 patients)")
    parser.add_argument("--preprocessed-dir", type=str, default="outputs/preprocessed", help="Path to preprocessed data")
    parser.add_argument("--partitions-dir", type=str, default="outputs/partitions", help="Path to partition manifests")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Root output directory")
    return parser.parse_args()


def run_local_only():
    args = parse_args()
    hid = args.hospital
    modalities = HOSPITAL_MODALITIES[hid]

    # Determinism
    set_deterministic(args.train_seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    print(f"=== [RUN-2 Local-Only {hid}] Device: {device} | Modalities: {modalities} | Train Seed: {args.train_seed} ===", flush=True)

    # Memory management
    if torch.cuda.is_available() and args.gpu >= 0 and args.max_gpu_memory_gb is not None:
        device_id = args.gpu
        total_mem = torch.cuda.get_device_properties(device_id).total_memory
        max_bytes = int(args.max_gpu_memory_gb * 1024 * 1024 * 1024)
        if total_mem > max_bytes:
            fraction = max_bytes / total_mem
            torch.cuda.set_per_process_memory_fraction(fraction, device_id)
            print(f"GPU VRAM limit set to {args.max_gpu_memory_gb:.1f} GB (fraction: {fraction:.4f})", flush=True)

    experiment_id = f"local_only_{hid}__part{args.partition_seed}__seed{args.train_seed}"
    log_dir = Path(args.output_dir) / "logs" / experiment_id
    chkpt_dir = Path(args.output_dir) / "checkpoints" / experiment_id
    results_dir = Path(args.output_dir) / "results" / experiment_id

    log_dir.mkdir(parents=True, exist_ok=True)
    chkpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    dump_environment_info(log_dir / "environment_version_dump.txt")

    # Load partition
    partition_file = Path(args.partitions_dir) / f"partition_{args.partition_seed}.json"
    if not partition_file.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_file}")

    with open(partition_file, "r") as f:
        partitions_data = json.load(f)

    all_hospital_pids = partitions_data["hospitals"][hid]["patient_ids"]

    # 80/20 train/val split per hospital
    n_val = max(1, int(len(all_hospital_pids) * 0.2)) if len(all_hospital_pids) > 1 else 0
    train_pids = all_hospital_pids[:-n_val] if n_val > 0 else all_hospital_pids
    val_pids = all_hospital_pids[-n_val:] if n_val > 0 else all_hospital_pids

    if args.dry_run:
        train_pids = train_pids[:2]
        val_pids = val_pids[:2]
        args.max_epochs = 2
        args.min_epochs = 1
        args.patience = 2

    print(f"Hospital {hid}: {len(all_hospital_pids)} total -> {len(train_pids)} train, {len(val_pids)} val", flush=True)

    # Initialize neural network modules
    encoders: Dict[str, UnimodalEncoder] = {m: UnimodalEncoder(in_channels=1).to(device) for m in modalities}
    fusion_head = SubsetFusionHead(modality_subset=modalities).to(device)
    decoder = UNetDecoder(num_classes=4).to(device)

    # Optimizer: all parameters trained end-to-end
    all_params = []
    for enc in encoders.values():
        all_params.extend(enc.parameters())
    all_params.extend(fusion_head.parameters())
    all_params.extend(decoder.parameters())

    optimizer = torch.optim.AdamW(all_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    loss_fn = SoftDiceCrossEntropyLoss(eps_d=1e-5)
    augmenter = PairwiseAugmentation(modalities=modalities, is_training=True)

    # Data loaders
    preprocessed_path = Path(args.preprocessed_dir)
    train_dataset = BraTSDataset(patient_ids=train_pids, cache_root=preprocessed_path, modalities=modalities)
    val_dataset = BraTSDataset(patient_ids=val_pids, cache_root=preprocessed_path, modalities=modalities)
    slice_sampler = SliceSampler(preprocessed_path, train_pids)

    # Checkpoint restoration (auto-resume)
    latest_chkpt_path = chkpt_dir / "latest_model.pt"
    best_chkpt_path = chkpt_dir / "best_model.pt"
    start_epoch = 1
    best_val_dice = -1.0
    best_epoch = 0
    no_improvement_count = 0

    if latest_chkpt_path.exists():
        print(f"=== [Auto-Resume] Found existing checkpoint ({latest_chkpt_path.name}). Resuming... ===", flush=True)
        chkpt_data = CheckpointManager.load_checkpoint(latest_chkpt_path, device=device)
        state = chkpt_data["state_dict"]
        for m, enc_s in state.get("encoders", {}).items():
            if m in encoders:
                encoders[m].load_state_dict(enc_s)
        if "fusion" in state:
            fusion_head.load_state_dict(state["fusion"])
        if "decoder" in state:
            decoder.load_state_dict(state["decoder"])
        start_epoch = chkpt_data.get("epoch", 0) + 1
        best_val_dice = chkpt_data.get("metadata", {}).get("best_val_dice", -1.0)
        best_epoch = chkpt_data.get("metadata", {}).get("best_epoch", 0)
        no_improvement_count = chkpt_data.get("metadata", {}).get("no_improvement_count", 0)
        print(f"Resumed at Epoch {start_epoch} (Previous Best Val Dice: {best_val_dice*100:.2f}% at Epoch {best_epoch})", flush=True)

    # Training Metrics CSV
    metrics_csv_path = log_dir / "training_metrics.csv"
    if not metrics_csv_path.exists():
        with open(metrics_csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_dice_ET,val_dice_TC,val_dice_WT,val_dice_macro\n")

    vol_cache: Dict[str, Dict] = {}

    print(f"\n--- Starting Local-Only Training: Epochs {start_epoch} to {args.max_epochs} ---", flush=True)
    start_time = time.time()

    for epoch in range(start_epoch, args.max_epochs + 1):
        for enc in encoders.values():
            enc.train()
        fusion_head.train()
        decoder.train()

        samples = slice_sampler.get_epoch_samples(seed=args.train_seed + epoch, is_training=True)
        total_batches = (len(samples) + args.batch_size - 1) // args.batch_size
        running_loss = 0.0

        for b_idx in range(total_batches):
            batch_samples = samples[b_idx * args.batch_size : (b_idx + 1) * args.batch_size]

            mod_slices = {m: [] for m in modalities}
            lbl_slices = []

            for pid, s_idx in batch_samples:
                if pid not in vol_cache:
                    vol_cache[pid] = train_dataset.load_patient_volume(pid)
                vol = vol_cache[pid]

                lbl_slices.append(torch.from_numpy(vol["labels"][s_idx]).long())
                for m in modalities:
                    mod_slices[m].append(torch.from_numpy(vol["modalities"][m][s_idx]).unsqueeze(0).float())

            if not lbl_slices:
                continue

            y_b = torch.stack(lbl_slices).to(device)
            x_b = {m: torch.stack(mod_slices[m]).to(device) for m in modalities}

            # Augmentation
            x_b, y_b = augmenter.augment_batch(x_b, y_b)

            # Forward pass
            mod_feats = {}
            for m in modalities:
                h1, h2, h3, h4, z = encoders[m](x_b[m])
                mod_feats[m] = (h1, h2, h3, h4, z)

            f1, f2, f3, f4, z_S = fusion_head(mod_feats)
            logits = decoder(z_S=z_S, f_S_4=f4, f_S_3=f3, f_S_2=f2, f_S_1=f1)

            optimizer.zero_grad()
            loss = loss_fn(logits, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / max(1, total_batches)

        # Validation on local val set
        for enc in encoders.values():
            enc.eval()
        fusion_head.eval()
        decoder.eval()

        val_dices = []
        with torch.no_grad():
            for pid in val_pids:
                if pid not in vol_cache:
                    vol_cache[pid] = val_dataset.load_patient_volume(pid)
                vol = vol_cache[pid]
                lab_vol = vol["labels"]
                num_slices = lab_vol.shape[0]

                pred_logits_list = []
                for start in range(0, num_slices, args.batch_size):
                    end = min(start + args.batch_size, num_slices)

                    mod_feats_b = {}
                    for m in modalities:
                        x_m = torch.from_numpy(vol["modalities"][m][start:end]).float().unsqueeze(1).to(device)
                        h1, h2, h3, h4, z = encoders[m](x_m)
                        mod_feats_b[m] = (h1, h2, h3, h4, z)

                    f1, f2, f3, f4, z_S_b = fusion_head(mod_feats_b)
                    logits_b = decoder(z_S=z_S_b, f_S_4=f4, f_S_3=f3, f_S_2=f2, f_S_1=f1)
                    pred_logits_list.append(logits_b)

                pred_logits_3d = torch.cat(pred_logits_list, dim=0).permute(1, 0, 2, 3)
                p_dice = compute_3d_dice_tensor(pred_logits_3d, lab_vol)
                val_dices.append(p_dice)

        v_et = np.mean([d["ET"] for d in val_dices])
        v_tc = np.mean([d["TC"] for d in val_dices])
        v_wt = np.mean([d["WT"] for d in val_dices])
        v_macro = np.mean([d["macro"] for d in val_dices])

        print(f"[{hid}] Epoch {epoch:03d}/{args.max_epochs:03d} | Train Loss: {epoch_loss:.4f} | Val Dice: Macro={v_macro*100:.2f}%, WT={v_wt*100:.2f}%, TC={v_tc*100:.2f}%, ET={v_et*100:.2f}%", flush=True)

        with open(metrics_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{epoch_loss:.6f},{v_et:.4f},{v_tc:.4f},{v_wt:.4f},{v_macro:.4f}\n")

        # Save latest checkpoint for auto-resumption
        current_state = {
            "encoders": {m: enc.cpu().state_dict() for m, enc in encoders.items()},
            "fusion": fusion_head.cpu().state_dict(),
            "decoder": decoder.cpu().state_dict(),
        }
        for enc in encoders.values():
            enc.to(device)
        fusion_head.to(device)
        decoder.to(device)

        CheckpointManager.save_checkpoint(
            filepath=latest_chkpt_path,
            state_dict=current_state,
            prototypes={},
            current_round=epoch,
            current_epoch=epoch,
            val_macro_dice=float(v_macro),
            extra_metadata={
                "hospital": hid,
                "best_val_dice": float(best_val_dice),
                "best_epoch": best_epoch,
                "no_improvement_count": no_improvement_count,
            },
        )

        # Check for best model
        if v_macro > best_val_dice:
            best_val_dice = float(v_macro)
            best_epoch = epoch
            no_improvement_count = 0
            CheckpointManager.save_checkpoint(
                filepath=best_chkpt_path,
                state_dict=current_state,
                prototypes={},
                current_round=epoch,
                current_epoch=epoch,
                val_macro_dice=best_val_dice,
                extra_metadata={"hospital": hid, "best_epoch": best_epoch},
            )
            print(f"  --> Saved new best model for {hid} (Macro Dice: {best_val_dice*100:.2f}%)", flush=True)
        else:
            no_improvement_count += 1

        if epoch >= args.min_epochs and no_improvement_count >= args.patience:
            print(f"  Early stopping triggered at epoch {epoch} (best: {best_val_dice*100:.2f}% at epoch {best_epoch})", flush=True)
            break

    elapsed = time.time() - start_time
    print(f"\n=== Training Complete for {hid} in {elapsed/60:.1f} minutes. Best Val Dice: {best_val_dice*100:.2f}% ===", flush=True)

    # =========================================================================
    # Final Evaluation on Universal 50 Pure Held-Out Test Patients
    # =========================================================================
    print(f"\n=== Evaluating Best Model on Universal 50 Pure Held-Out Test Patients ===", flush=True)

    # Load best checkpoint
    best_data = CheckpointManager.load_checkpoint(best_chkpt_path, device=device)
    b_state = best_data["state_dict"]
    for m, enc_s in b_state["encoders"].items():
        encoders[m].load_state_dict(enc_s)
        encoders[m].eval()
    fusion_head.load_state_dict(b_state["fusion"])
    fusion_head.eval()
    decoder.load_state_dict(b_state["decoder"])
    decoder.eval()

    # Load universal 50 test set manifest
    test_manifest_path = Path(args.partitions_dir) / "h3_test_patients.json"
    if not test_manifest_path.exists():
        raise FileNotFoundError(f"Universal test set manifest not found: {test_manifest_path}")

    with open(test_manifest_path, "r") as f:
        test_manifest = json.load(f)

    pure_50_test_pids = test_manifest["patient_ids"] if isinstance(test_manifest, dict) and "patient_ids" in test_manifest else test_manifest

    if args.dry_run:
        pure_50_test_pids = pure_50_test_pids[:3]

    test_dataset = BraTSDataset(
        patient_ids=pure_50_test_pids,
        cache_root=preprocessed_path,
        modalities=modalities,
    )

    evaluator = PatientEvaluator()
    eval_csv_path = results_dir / "pure_50_test_patient_metrics.csv"
    eval_logger = EvaluationCSVLogger(eval_csv_path)

    patient_test_metrics = []

    with torch.no_grad():
        for pid in pure_50_test_pids:
            vol = test_dataset.load_patient_volume(pid)
            lab_vol = vol["labels"]
            num_slices = lab_vol.shape[0]

            pred_logits_list = []
            for start in range(0, num_slices, args.batch_size):
                end = min(start + args.batch_size, num_slices)

                mod_feats_b = {}
                for m in modalities:
                    x_m = torch.from_numpy(vol["modalities"][m][start:end]).float().unsqueeze(1).to(device)
                    h1, h2, h3, h4, z = encoders[m](x_m)
                    mod_feats_b[m] = (h1, h2, h3, h4, z)

                f1, f2, f3, f4, z_S_b = fusion_head(mod_feats_b)
                logits_b = decoder(z_S=z_S_b, f_S_4=f4, f_S_3=f3, f_S_2=f2, f_S_1=f1)
                pred_logits_list.append(logits_b.cpu())

            pred_logits_3d = torch.cat(pred_logits_list, dim=0).permute(1, 0, 2, 3).numpy()
            p_metrics = evaluator.evaluate_patient_volume(pred_logits_3d, lab_vol)
            patient_test_metrics.append(p_metrics)

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
                track_id=f"{hid}_local",
                hospital_id=hid,
                dice_dict=dice_dict,
                hd95_dict=hd95_dict,
            )

    mean_wt = np.mean([p["dice_WT"] for p in patient_test_metrics])
    mean_tc = np.mean([p["dice_TC"] for p in patient_test_metrics])
    mean_et = np.mean([p["dice_ET"] for p in patient_test_metrics])
    mean_macro = np.mean([p["macro_dice"] for p in patient_test_metrics])

    mean_hd_wt = np.mean([p["hd95_WT"] for p in patient_test_metrics])
    mean_hd_tc = np.mean([p["hd95_TC"] for p in patient_test_metrics])
    mean_hd_et = np.mean([p["hd95_ET"] for p in patient_test_metrics])
    mean_hd_macro = np.mean([p["macro_hd95"] for p in patient_test_metrics])

    print("\n" + "=" * 70, flush=True)
    print(f"  [RUN-2 LOCAL-ONLY RESULTS] Hospital {hid} (Modalities: {modalities})", flush=True)
    print(f"  Partition: {args.partition_seed} | Train Seed: {args.train_seed} | N={len(pure_50_test_pids)} Test Patients", flush=True)
    print("=" * 70, flush=True)
    print(f"  WT Dice:    {mean_wt*100:6.2f}%  |  WT HD95:    {mean_hd_wt:6.2f} mm", flush=True)
    print(f"  TC Dice:    {mean_tc*100:6.2f}%  |  TC HD95:    {mean_hd_tc:6.2f} mm", flush=True)
    print(f"  ET Dice:    {mean_et*100:6.2f}%  |  ET HD95:    {mean_hd_et:6.2f} mm", flush=True)
    print(f"  Macro Dice: {mean_macro*100:6.2f}%  |  Macro HD95: {mean_hd_macro:6.2f} mm", flush=True)
    print("=" * 70, flush=True)
    print(f"Results written to: {eval_csv_path}", flush=True)


if __name__ == "__main__":
    run_local_only()
