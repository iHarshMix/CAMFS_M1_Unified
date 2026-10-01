"""
CAMFS M1 — RUN-5: Centralized Ceiling (Oracle Upper Bound) Baseline
===================================================================

Upper-bound non-federated ceiling where all privacy and consent boundaries
are dissolved. All 256 training patients across all 4 hospitals are pooled
onto a single centralized server to train one model per clinical modality configuration:
  - full4:   T1, T1ce, T2, FLAIR (Ceiling for Hospital H1)
  - 3mod_a:  T1, T1ce, T2        (Ceiling for Hospitals H2 and H4)
  - 2mod:    T1, FLAIR           (Ceiling for Hospital H3)

All configurations are strictly evaluated on the universal 50 pure held-out
BraTS 2020 test patients using src/metrics.py::PatientEvaluator.

Parity Specifications:
  - Optimizer: AdamW (lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-4)
  - Scheduler: CosineAnnealingLR (3e-4 -> 1e-6)
  - Gradient norm clipping: max_norm=1.0
  - In-memory dataset caching: vol_cache exploiting large system RAM
  - Fast validation: 32-slice inference batches
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce deterministic environment variables
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from src.data.augmentation import PairwiseAugmentation
from src.data.brats_dataset import BraTSDataset
from src.data.slice_sampler import SliceSampler
from src.logging import CheckpointManager, EvaluationCSVLogger
from src.losses import SoftDiceCrossEntropyLoss
from src.metrics import PatientEvaluator, compute_3d_dice_tensor
from src.models import SubsetFusionHead, UNetDecoder, UnimodalEncoder
from src.seed import set_deterministic

CONFIG_MODALITIES = {
    "full4": ["T1", "T1ce", "T2", "FLAIR"],
    "3mod_a": ["T1", "T1ce", "T2"],
    "2mod": ["T1", "FLAIR"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RUN-5: Centralized Ceiling (Oracle Upper Bound) Baseline")
    parser.add_argument(
        "--modality-config",
        type=str,
        choices=["full4", "3mod_a", "2mod"],
        default="full4",
        help="Target clinical modality subset (full4, 3mod_a, 2mod). Default: full4",
    )
    parser.add_argument("--partition-seed", type=int, default=1103, help="Data partition seed (default: 1103)")
    parser.add_argument("--train-seed", type=int, default=17, help="Training randomness seed (default: 17)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training slice batch size (default: 16)")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Validation/test slice batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for end-to-end training (default: 3e-4)")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum training epochs (default: 100)")
    parser.add_argument("--min-epochs", type=int, default=20, help="Minimum epochs before early stopping (default: 20)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience epochs (default: 20)")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU device index (default: 0)")
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default="outputs/preprocessed",
        help="Path to preprocessed BraTS data (default: outputs/preprocessed)",
    )
    parser.add_argument("--partitions-dir", type=str, default="outputs/partitions", help="Path to partitions directory")
    parser.add_argument("--dry-run", action="store_true", help="Fast smoke test (2 patients, 1 epoch)")
    return parser.parse_args()


def run_centralized(
    modality_config: Optional[str] = None,
    partition_seed: Optional[int] = None,
    train_seed: Optional[int] = None,
    gpu: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    args = parse_args()
    if modality_config is not None:
        args.modality_config = modality_config
    if partition_seed is not None:
        args.partition_seed = partition_seed
    if train_seed is not None:
        args.train_seed = train_seed
    if gpu is not None:
        args.gpu = gpu
    if dry_run:
        args.dry_run = True

    modalities = CONFIG_MODALITIES[args.modality_config]

    # Enforce deterministic training
    set_deterministic(args.train_seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 78, flush=True)
    print(f"=== [RUN-5 Centralized Oracle: {args.modality_config.upper()}] ===", flush=True)
    print(f"    Modalities: {modalities}", flush=True)
    print(f"    Device: {device} | Partition Seed: {args.partition_seed} | Train Seed: {args.train_seed}", flush=True)
    print(f"    Batch Size: {args.batch_size} (train) / {args.eval_batch_size} (eval) | LR: {args.lr}", flush=True)
    print("=" * 78 + "\n", flush=True)

    # Isolated directory paths
    run_id = (
        f"dry_run_oracle_{args.modality_config}"
        if args.dry_run
        else f"centralized_{args.modality_config}__part{args.partition_seed}__seed{args.train_seed}"
    )
    log_dir = Path("outputs/logs") / run_id
    chkpt_dir = Path("outputs/checkpoints") / run_id
    results_dir = Path("outputs/results") / run_id

    log_dir.mkdir(parents=True, exist_ok=True)
    chkpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load partition manifest
    partition_file = Path(args.partitions_dir) / f"partition_{args.partition_seed}.json"
    if not partition_file.exists():
        raise FileNotFoundError(f"Partition file not found: {partition_file}")

    with open(partition_file, "r") as f:
        partitions_data = json.load(f)

    # Strict zero-leakage pooling: combine training splits and validation splits from all hospitals
    train_pids: List[str] = []
    val_pids: List[str] = []

    for h_id in ("H1", "H2", "H3", "H4"):
        h_pids = partitions_data["hospitals"][h_id]["patient_ids"]
        n_val = max(1, int(len(h_pids) * 0.2)) if len(h_pids) > 1 else 0
        h_train = h_pids[:-n_val] if n_val > 0 else h_pids
        h_val = h_pids[-n_val:] if n_val > 0 else []
        train_pids.extend(h_train)
        val_pids.extend(h_val)

    train_pids = sorted(train_pids)
    val_pids = sorted(val_pids)

    if args.dry_run:
        train_pids = train_pids[:2]
        val_pids = val_pids[:2]
        args.max_epochs = 1
        args.min_epochs = 1
        args.patience = 1

    print(
        f"Pooled Dataset: {len(train_pids)} train patients | {len(val_pids)} validation patients",
        flush=True,
    )

    # Initialize neural network architecture
    encoders: Dict[str, UnimodalEncoder] = {m: UnimodalEncoder(in_channels=1).to(device) for m in modalities}
    fusion_head = SubsetFusionHead(modality_subset=modalities).to(device)
    decoder = UNetDecoder(num_classes=4).to(device)

    # Collect parameters for end-to-end training
    all_params = []
    for enc in encoders.values():
        all_params.extend(enc.parameters())
    all_params.extend(fusion_head.parameters())
    all_params.extend(decoder.parameters())

    total_trainable = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"Total Trainable Parameters: {total_trainable:,}", flush=True)

    optimizer = torch.optim.AdamW(all_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    loss_fn = SoftDiceCrossEntropyLoss(eps_d=1e-5)
    augmenter = PairwiseAugmentation(modalities=modalities, is_training=True)

    # Dataset and sampler
    preprocessed_path = Path(args.preprocessed_dir)
    train_dataset = BraTSDataset(patient_ids=train_pids, cache_root=preprocessed_path, modalities=modalities)
    val_dataset = BraTSDataset(patient_ids=val_pids, cache_root=preprocessed_path, modalities=modalities)
    slice_sampler = SliceSampler(preprocessed_path, train_pids)

    # Checkpoint auto-resumption protocol
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
        print(
            f"Resumed at Epoch {start_epoch} (Previous Best Val Dice: {best_val_dice*100:.2f}% at Epoch {best_epoch})",
            flush=True,
        )

    # Training Metrics CSV
    metrics_csv_path = log_dir / "training_metrics.csv"
    if not metrics_csv_path.exists():
        with open(metrics_csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_dice_ET,val_dice_TC,val_dice_WT,val_dice_macro\n")

    # Persistent in-memory volume cache (exploiting 314 GB RAM on V100)
    vol_cache: Dict[str, Dict] = {}

    print(f"\n--- Starting Centralized Training: Epochs {start_epoch} to {args.max_epochs} ---", flush=True)
    start_time = time.time()

    for epoch in range(start_epoch, args.max_epochs + 1):
        epoch_t0 = time.time()
        for enc in encoders.values():
            enc.train()
        fusion_head.train()
        decoder.train()

        # 1:1 Tumor vs Non-tumor slice sampling
        samples = slice_sampler.get_epoch_samples(seed=args.train_seed + epoch, is_training=True)
        if args.dry_run:
            samples = samples[: args.batch_size]

        total_batches = (len(samples) + args.batch_size - 1) // args.batch_size
        running_loss = 0.0
        valid_batches = 0

        for b_idx in range(total_batches):
            batch_samples = samples[b_idx * args.batch_size : (b_idx + 1) * args.batch_size]

            mod_slices = {m: [] for m in modalities}
            lbl_slices = []

            for pid, s_idx in batch_samples:
                if pid not in vol_cache:
                    vol_cache[pid] = train_dataset.load_patient_volume(pid)
                vol = vol_cache[pid]

                # Filter out degenerate empty slices outside the skull (0 brain voxels)
                nonzero_voxels = sum(np.count_nonzero(vol["modalities"][m][s_idx]) for m in modalities if m in vol["modalities"])
                if nonzero_voxels == 0:
                    continue

                lbl_slices.append(torch.from_numpy(vol["labels"][s_idx]).long())
                for m in modalities:
                    mod_slices[m].append(torch.from_numpy(vol["modalities"][m][s_idx]).unsqueeze(0).float())

            if not lbl_slices:
                continue

            y_b = torch.stack(lbl_slices).to(device)
            x_b = {m: torch.stack(mod_slices[m]).to(device) for m in modalities}

            # Spatial & intensity augmentations directly on GPU
            x_b, y_b = augmenter.augment_batch(x_b, y_b)

            # Forward pass
            mod_feats = {}
            for m in modalities:
                h1, h2, h3, h4, z = encoders[m](x_b[m])
                mod_feats[m] = (h1, h2, h3, h4, z)

            f1, f2, f3, f4, z_S = fusion_head(mod_feats)
            logits = decoder(z_S=z_S, f_S_4=f4, f_S_3=f3, f_S_2=f2, f_S_1=f1)

            loss = loss_fn(logits, y_b)

            # Numerical stability guard
            if not torch.isfinite(loss):
                print(f"[Epoch {epoch:03d} | Batch {b_idx}] Warning: Non-finite loss encountered. Skipping step.", flush=True)
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()

            # Gradient health check
            has_nan_grad = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in all_params)
            if has_nan_grad:
                print(f"[Epoch {epoch:03d} | Batch {b_idx}] Warning: Non-finite gradients detected. Skipping step.", flush=True)
                optimizer.zero_grad()
                continue

            # Gradient clipping (max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            valid_batches += 1

        scheduler.step()
        epoch_loss = running_loss / max(1, valid_batches)

        # =====================================================================
        # Fast Validation on Pooled Val Set (eval_batch_size=32)
        # =====================================================================
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
                for start in range(0, num_slices, args.eval_batch_size):
                    end = min(start + args.eval_batch_size, num_slices)

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

        epoch_sec = time.time() - epoch_t0
        print(
            f"Epoch {epoch:03d}/{args.max_epochs:03d} ({epoch_sec:4.1f}s) | Train Loss: {epoch_loss:.4f} | "
            f"Val Dice: Macro={v_macro*100:.2f}%, WT={v_wt*100:.2f}%, TC={v_tc*100:.2f}%, ET={v_et*100:.2f}%",
            flush=True,
        )

        with open(metrics_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{epoch_loss:.6f},{v_et:.4f},{v_tc:.4f},{v_wt:.4f},{v_macro:.4f}\n")

        # Atomic checkpointing for auto-resumption
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
                "modality_config": args.modality_config,
                "best_val_dice": float(best_val_dice),
                "best_epoch": best_epoch,
                "no_improvement_count": no_improvement_count,
            },
        )

        # Check for new best model
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
                val_macro_dice=float(v_macro),
                extra_metadata={
                    "modality_config": args.modality_config,
                    "best_val_dice": float(best_val_dice),
                    "best_epoch": best_epoch,
                    "no_improvement_count": 0,
                },
            )
            print(f"  --> Saved new best Oracle model ({args.modality_config}): {best_val_dice*100:.2f}% at Epoch {best_epoch}", flush=True)
        else:
            no_improvement_count += 1

        # Early stopping check
        if epoch >= args.min_epochs and no_improvement_count >= args.patience:
            print(
                f"\nEarly stopping triggered at Epoch {epoch} (Best: {best_val_dice*100:.2f}% at Epoch {best_epoch})",
                flush=True,
            )
            break

    elapsed = (time.time() - start_time) / 60.0
    print(
        f"\n=== Training Complete for Oracle {args.modality_config} in {elapsed:.1f} minutes. Best Val Dice: {best_val_dice*100:.2f}% ===",
        flush=True,
    )

    # =========================================================================
    # Evaluation on Universal 50 Pure Held-Out BraTS Test Patients
    # =========================================================================
    print(f"\n=== Evaluating Best Model on Universal 50 Pure Held-Out Test Patients ===", flush=True)

    if best_chkpt_path.exists():
        best_data = CheckpointManager.load_checkpoint(best_chkpt_path, device=device)
        b_state = best_data["state_dict"]
        for m, enc_s in b_state.get("encoders", {}).items():
            if m in encoders:
                encoders[m].load_state_dict(enc_s)
        if "fusion" in b_state:
            fusion_head.load_state_dict(b_state["fusion"])
        if "decoder" in b_state:
            decoder.load_state_dict(b_state["decoder"])
        print(f"Loaded best checkpoint from Epoch {best_data.get('metadata', {}).get('best_epoch', 'unknown')}", flush=True)

    for enc in encoders.values():
        enc.eval()
    fusion_head.eval()
    decoder.eval()

    # Load universal 50 test set manifest
    test_manifest_path = Path(args.partitions_dir) / "h3_test_patients.json"
    if not test_manifest_path.exists():
        raise FileNotFoundError(f"Universal test set manifest not found: {test_manifest_path}")

    with open(test_manifest_path, "r") as f:
        test_manifest = json.load(f)

    pure_50_test_pids = test_manifest["patient_ids"] if isinstance(test_manifest, dict) and "patient_ids" in test_manifest else test_manifest

    if args.dry_run:
        pure_50_test_pids = pure_50_test_pids[:2]

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
            for start in range(0, num_slices, args.eval_batch_size):
                end = min(start + args.eval_batch_size, num_slices)

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
                track_id=f"oracle_{args.modality_config}",
                hospital_id="centralized_oracle",
                dice_dict=dice_dict,
                hd95_dict=hd95_dict,
            )

    mean_wt = float(np.mean([p["dice_WT"] for p in patient_test_metrics]))
    mean_tc = float(np.mean([p["dice_TC"] for p in patient_test_metrics]))
    mean_et = float(np.mean([p["dice_ET"] for p in patient_test_metrics]))
    mean_macro = float(np.mean([p["macro_dice"] for p in patient_test_metrics]))

    mean_hd_wt = float(np.mean([p["hd95_WT"] for p in patient_test_metrics]))
    mean_hd_tc = float(np.mean([p["hd95_TC"] for p in patient_test_metrics]))
    mean_hd_et = float(np.mean([p["hd95_ET"] for p in patient_test_metrics]))
    mean_hd_macro = float(np.mean([p["macro_hd95"] for p in patient_test_metrics]))

    summary_metrics = {
        "modality_config": args.modality_config,
        "modalities": modalities,
        "partition_seed": args.partition_seed,
        "train_seed": args.train_seed,
        "num_test_patients": len(pure_50_test_pids),
        "best_epoch": best_epoch,
        "val_macro_dice": best_val_dice,
        "test_dice_macro": mean_macro,
        "test_dice_WT": mean_wt,
        "test_dice_TC": mean_tc,
        "test_dice_ET": mean_et,
        "test_hd95_macro": mean_hd_macro,
        "test_hd95_WT": mean_hd_wt,
        "test_hd95_TC": mean_hd_tc,
        "test_hd95_ET": mean_hd_et,
    }

    summary_path = results_dir / "summary_test_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print("\n" + "=" * 78, flush=True)
    print(f"  [RUN-5 CENTRALIZED ORACLE RESULTS] Config: {args.modality_config.upper()} (Modalities: {modalities})", flush=True)
    print(f"  Partition: {args.partition_seed} | Train Seed: {args.train_seed} | N={len(pure_50_test_pids)} Test Patients", flush=True)
    print("=" * 78, flush=True)
    print(f"  WT Dice:    {mean_wt*100:6.2f}%  |  WT HD95:    {mean_hd_wt:6.2f} mm", flush=True)
    print(f"  TC Dice:    {mean_tc*100:6.2f}%  |  TC HD95:    {mean_hd_tc:6.2f} mm", flush=True)
    print(f"  ET Dice:    {mean_et*100:6.2f}%  |  ET HD95:    {mean_hd_et:6.2f} mm", flush=True)
    print(f"  Macro Dice: {mean_macro*100:6.2f}%  |  Macro HD95: {mean_hd_macro:6.2f} mm", flush=True)
    print("=" * 78, flush=True)
    print(f"Per-patient metrics written to: {eval_csv_path}", flush=True)
    print(f"Summary JSON written to:        {summary_path}", flush=True)


if __name__ == "__main__":
    run_centralized()
