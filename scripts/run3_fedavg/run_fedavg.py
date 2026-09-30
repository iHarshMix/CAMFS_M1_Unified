"""
CAMFS M1 — RUN-3: Compliant FedAvg Baseline Runner
===================================================

Implements §RUN-3 of the Master Experiment Execution Plan:
  - Architecture Parity: 4 Unimodal Encoders + 4-modality SubsetFusionHead + UNetDecoder
  - Single Global Model: Shared parameters theta_global = {encoders, fusion_head, decoder}
  - Input Handling: 4 channels fed; missing modalities are strictly ZERO-FILLED
  - Consent Compliance:
      * H1 (102 train): T1, T1ce, T2, FLAIR (all 4 real)
      * H2 (64 train): T1, T2 (T1ce ZERO-FILLED during federated training for compliance; FLAIR zero-filled)
      * H3 (52 train): T1, FLAIR (T1ce, T2 zero-filled)
      * H4 (38 train): T1, T1ce, T2 (FLAIR zero-filled)
  - Aggregation: Patient-weighted parameter averaging (w_k = N_k / N)
  - Optimizer: AdamW (lr=3e-4, weight_decay=1e-4) + CosineAnnealingLR + clip_grad_norm_(1.0)
  - Checkpointing: Atomic auto-resumption per round (outputs/checkpoints/fedavg__part{part}__seed{seed}/)
  - Evaluation: Universal 50 pure held-out test patients per hospital (outputs/results/fedavg__part{part}__seed{seed}/)
"""

from __future__ import annotations

import argparse
import copy
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
from src.metrics import PatientEvaluator
from src.models import SubsetFusionHead, UNetDecoder, UnimodalEncoder
from src.seed import set_deterministic

ALL_MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]

# Hospital modality permissions during FEDERATED TRAINING (Consent Compliant)
TRAIN_MODALITIES = {
    "H1": ["T1", "T1ce", "T2", "FLAIR"],
    "H2": ["T1", "T2"],                   # T1ce is zero-filled during training for consent compliance!
    "H3": ["T1", "FLAIR"],
    "H4": ["T1", "T1ce", "T2"],
}

# Hospital modality permissions during LOCAL VALIDATION & TEST EVALUATION
EVAL_MODALITIES = {
    "H1": ["T1", "T1ce", "T2", "FLAIR"],
    "H2": ["T1", "T1ce", "T2"],           # Local use of owned T1ce is allowed for hospital diagnostic eval
    "H3": ["T1", "FLAIR"],
    "H4": ["T1", "T1ce", "T2"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RUN-3: Compliant FedAvg Baseline")
    parser.add_argument("--partition-seed", type=int, default=1103, help="Data partition seed (default: 1103)")
    parser.add_argument("--train-seed", type=int, default=17, help="Training randomness seed (default: 17)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training and slice inference")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for local AdamW optimizer (default: 3e-4)")
    parser.add_argument("--max-rounds", type=int, default=100, help="Maximum federated communication rounds (default: 100)")
    parser.add_argument("--min-rounds", type=int, default=20, help="Minimum rounds before early stopping (default: 20)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience rounds (default: 20)")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round (default: 1)")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU device index")
    parser.add_argument("--preprocessed-dir", type=str, default="outputs/preprocessed", help="Path to preprocessed data")
    parser.add_argument("--partitions-dir", type=str, default="outputs/partitions", help="Path to partitions")
    parser.add_argument("--dry-run", action="store_true", help="Fast smoke test (2 patients, 2 rounds)")
    return parser.parse_args()


class FedAvgModel(nn.Module):
    """
    Unified FedAvg model wrapping 4 unimodal encoders, a 4-modality fusion head, and a UNet decoder.
    Missing modalities are zero-filled before entering their respective encoders.
    """

    def __init__(self, modalities: List[str] = ALL_MODALITIES, num_classes: int = 4) -> None:
        super().__init__()
        self.modalities = modalities
        self.encoders = nn.ModuleDict({m: UnimodalEncoder(in_channels=1) for m in modalities})
        self.fusion_head = SubsetFusionHead(modality_subset=modalities)
        self.decoder = UNetDecoder(num_classes=num_classes)

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        input_dict: dict of modality -> tensor (B, 1, 240, 240).
        Missing modalities in input_dict must already be zero-filled or will be zero-filled.
        """
        first_tensor = next(iter(input_dict.values()))
        batch_size = first_tensor.shape[0]
        device = first_tensor.device

        mod_features = {}
        for m in self.modalities:
            if m in input_dict:
                x = input_dict[m]
            else:
                x = torch.zeros((batch_size, 1, 240, 240), dtype=torch.float32, device=device)
            h1, h2, h3, h4, z = self.encoders[m](x)
            mod_features[m] = (h1, h2, h3, h4, z)

        f1, f2, f3, f4, z_S = self.fusion_head(mod_features)
        logits = self.decoder(z_S=z_S, f_S_4=f4, f_S_3=f3, f_S_2=f2, f_S_1=f1)
        return logits


def train_client_round(
    client_id: str,
    global_model: FedAvgModel,
    dataset: BraTSDataset,
    slice_sampler: SliceSampler,
    patient_ids: List[str],
    allowed_modalities: List[str],
    device: torch.device,
    batch_size: int,
    lr: float,
    seed: int,
    round_idx: int,
    max_rounds: int,
    local_epochs: int = 1,
    dry_run: bool = False,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Execute 1 round of local training at a hospital client.
    Returns (local_state_dict, avg_loss).
    """
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    # Cosine annealing LR schedule across communication rounds
    eta_min = 1e-6
    round_lr = eta_min + 0.5 * (lr - eta_min) * (1.0 + math.cos(math.pi * (round_idx - 1) / max(1, max_rounds)))

    optimizer = torch.optim.AdamW(
        local_model.parameters(),
        lr=round_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    loss_fn = SoftDiceCrossEntropyLoss(eps_d=1e-5)
    augmenter = PairwiseAugmentation(modalities=allowed_modalities, is_training=True)

    samples = slice_sampler.get_epoch_samples(seed=seed + round_idx, is_training=True)
    if dry_run:
        samples = samples[:batch_size]

    total_batches = (len(samples) + batch_size - 1) // batch_size
    vol_cache: Dict[str, Dict] = {}
    total_loss = 0.0
    valid_batches = 0

    for ep in range(local_epochs):
        for b_idx in range(total_batches):
            batch_samples = samples[b_idx * batch_size : (b_idx + 1) * batch_size]

            mod_slices = {m: [] for m in allowed_modalities}
            lbl_slices = []

            for pid, s_idx in batch_samples:
                if pid not in vol_cache:
                    vol_cache[pid] = dataset.load_patient_volume(pid)
                vol = vol_cache[pid]

                # Filter out degenerate empty slices outside skull (0 brain voxels)
                nonzero_voxels = sum(np.count_nonzero(vol["modalities"][m][s_idx]) for m in allowed_modalities if m in vol["modalities"])
                if nonzero_voxels == 0:
                    continue

                lbl_slices.append(torch.from_numpy(vol["labels"][s_idx]).long())
                for m in allowed_modalities:
                    mod_slices[m].append(torch.from_numpy(vol["modalities"][m][s_idx]).unsqueeze(0).float())

            if not lbl_slices:
                continue

            y_b = torch.stack(lbl_slices).to(device)
            x_avail = {m: torch.stack(mod_slices[m]).to(device) for m in allowed_modalities}

            # Batch-native GPU spatial and intensity augmentation
            x_avail, y_b = augmenter.augment_batch(x_avail, y_b)

            # Assemble full batch_inputs with zero-filling for unowned/unshared modalities
            ref_tensor = x_avail[allowed_modalities[0]]
            B = ref_tensor.shape[0]
            batch_inputs = {}
            for m in ALL_MODALITIES:
                if m in allowed_modalities:
                    batch_inputs[m] = x_avail[m]
                else:
                    batch_inputs[m] = torch.zeros((B, 1, 240, 240), dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits = local_model(batch_inputs)
            loss = loss_fn(logits, y_b)

            # Numerical stability guard
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss detected at client {client_id}, round {round_idx}, batch {b_idx}. Skipping.", flush=True)
                optimizer.zero_grad()
                continue

            loss.backward()

            # Gradient health guard: verify finite gradients before optimizer step
            has_nan_grad = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in local_model.parameters())
            if has_nan_grad:
                print(f"Warning: Non-finite gradients at client {client_id}, round {round_idx}, batch {b_idx}. Skipping.", flush=True)
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

    avg_loss = total_loss / max(valid_batches, 1)

    # Return state_dict on CPU to save GPU VRAM
    local_state = {k: v.cpu().clone() for k, v in local_model.state_dict().items()}
    return local_state, avg_loss


def evaluate_client_validation(
    client_id: str,
    model: FedAvgModel,
    dataset: BraTSDataset,
    val_pids: List[str],
    eval_modalities: List[str],
    device: torch.device,
    batch_size: int,
    evaluator: PatientEvaluator,
) -> Dict[str, float]:
    """
    Evaluate global model on a hospital's validation cohort.
    Uses hospital's diagnostic modalities (H2 includes T1ce; unpossessed modalities zero-filled).
    """
    model.eval()
    val_dice_records = []

    with torch.no_grad():
        for pid in val_pids:
            vol = dataset.load_patient_volume(pid)
            lab_vol = vol["labels"]
            num_slices = lab_vol.shape[0]

            pred_logits_list = []
            for start in range(0, num_slices, batch_size):
                end = min(start + batch_size, num_slices)
                B = end - start

                batch_inputs = {}
                for m in ALL_MODALITIES:
                    if m in eval_modalities:
                        batch_inputs[m] = torch.from_numpy(vol["modalities"][m][start:end]).float().unsqueeze(1).to(device)
                    else:
                        batch_inputs[m] = torch.zeros((B, 1, 240, 240), dtype=torch.float32, device=device)

                logits_b = model(batch_inputs)
                pred_logits_list.append(logits_b.cpu())

            pred_logits_3d = torch.cat(pred_logits_list, dim=0).permute(1, 0, 2, 3).numpy()
            p_metrics = evaluator.evaluate_patient_volume(pred_logits_3d, lab_vol)
            val_dice_records.append(p_metrics)

    mean_macro = float(np.mean([p["macro_dice"] for p in val_dice_records]))
    mean_wt = float(np.mean([p["dice_WT"] for p in val_dice_records]))
    mean_tc = float(np.mean([p["dice_TC"] for p in val_dice_records]))
    mean_et = float(np.mean([p["dice_ET"] for p in val_dice_records]))

    return {"macro": mean_macro, "WT": mean_wt, "TC": mean_tc, "ET": mean_et}


def aggregate_fedavg(
    client_states: Dict[str, Dict[str, torch.Tensor]],
    client_weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """
    Patient-weighted parameter averaging: theta_global = sum_k (w_k * theta_k).
    """
    for c_id, state in client_states.items():
        for k, v in state.items():
            if torch.is_floating_point(v) and not torch.isfinite(v).all():
                raise RuntimeError(f"Client {c_id} produced non-finite weights in parameter {k}!")

    first_client = next(iter(client_states.keys()))
    aggregated_state = {}

    for param_key in client_states[first_client].keys():
        first_tensor = client_states[first_client][param_key]
        if first_tensor.dtype in [torch.int64, torch.int32, torch.bool]:
            # Non-trainable integer buffers (e.g. num_batches_tracked) copy from first client
            aggregated_state[param_key] = first_tensor.clone()
        else:
            accum = torch.zeros_like(first_tensor, dtype=torch.float32)
            for c_id, state in client_states.items():
                accum += client_weights[c_id] * state[param_key].float()
            aggregated_state[param_key] = accum.to(first_tensor.dtype)

    return aggregated_state


def run_fedavg() -> None:
    args = parse_args()

    # Enforce deterministic environment
    set_deterministic(args.train_seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"=== [RUN-3 Compliant FedAvg Baseline] Device: {device} | Partition Seed: {args.partition_seed} | Train Seed: {args.train_seed} ===", flush=True)

    # Directories
    run_id = f"dry_run_fedavg" if args.dry_run else f"fedavg__part{args.partition_seed}__seed{args.train_seed}"
    log_dir = Path("outputs/logs") / run_id
    chkpt_dir = Path("outputs/checkpoints") / run_id
    results_dir = Path("outputs/results") / run_id

    log_dir.mkdir(parents=True, exist_ok=True)
    chkpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load partition manifests
    partition_file = Path(args.partitions_dir) / f"partition_{args.partition_seed}.json"
    if not partition_file.exists():
        raise FileNotFoundError(f"Partition manifest not found at: {partition_file}")

    with open(partition_file, "r") as f:
        partition_data = json.load(f)

    preprocessed_path = Path(args.preprocessed_dir)
    hospitals = ["H1", "H2", "H3", "H4"]

    client_train_pids: Dict[str, List[str]] = {}
    client_val_pids: Dict[str, List[str]] = {}
    client_datasets: Dict[str, BraTSDataset] = {}
    client_samplers: Dict[str, SliceSampler] = {}

    for hid in hospitals:
        all_hospital_pids = partition_data["hospitals"][hid]["patient_ids"]
        n_val = max(1, int(len(all_hospital_pids) * 0.2)) if len(all_hospital_pids) > 1 else 0
        tr_pids = all_hospital_pids[:-n_val] if n_val > 0 else all_hospital_pids
        val_pids = all_hospital_pids[-n_val:] if n_val > 0 else all_hospital_pids

        if args.dry_run:
            tr_pids = tr_pids[:1]
            val_pids = val_pids[:1]

        client_train_pids[hid] = tr_pids
        client_val_pids[hid] = val_pids
        client_datasets[hid] = BraTSDataset(patient_ids=tr_pids + val_pids, cache_root=preprocessed_path, modalities=ALL_MODALITIES)
        client_samplers[hid] = SliceSampler(preprocessed_path, tr_pids)
        print(f"Hospital {hid}: {len(tr_pids)} train, {len(val_pids)} val | Train Modalities: {TRAIN_MODALITIES[hid]}", flush=True)

    # Calculate FedAvg aggregation weights based on training patient counts
    total_train_patients = sum(len(client_train_pids[h]) for h in hospitals)
    client_weights = {h: len(client_train_pids[h]) / total_train_patients for h in hospitals}
    print(f"FedAvg Aggregation Weights: {client_weights} (Total: {total_train_patients} patients)", flush=True)

    # Initialize global model
    global_model = FedAvgModel(modalities=ALL_MODALITIES, num_classes=4).to(device)

    # Checkpoint restoration (auto-resume)
    latest_chkpt_path = chkpt_dir / "latest_model.pt"
    best_chkpt_path = chkpt_dir / "best_model.pt"
    start_round = 1
    best_val_macro = -1.0
    best_round = 0
    no_improvement_count = 0

    if latest_chkpt_path.exists():
        print(f"=== [Auto-Resume] Found existing checkpoint ({latest_chkpt_path.name}). Resuming... ===", flush=True)
        checkpoint = torch.load(latest_chkpt_path, map_location=device)
        global_model.load_state_dict(checkpoint["model_state_dict"])
        start_round = checkpoint["round"] + 1
        best_val_macro = checkpoint["best_val_macro"]
        best_round = checkpoint["best_round"]
        no_improvement_count = checkpoint["no_improvement_count"]
        print(f"Resumed at Round {start_round} (Previous Best Global Val Macro: {best_val_macro*100:.2f}% at Round {best_round})", flush=True)

    max_rounds = 2 if args.dry_run else args.max_rounds
    evaluator = PatientEvaluator()
    metrics_csv_path = log_dir / "training_metrics.csv"

    if not metrics_csv_path.exists():
        with open(metrics_csv_path, "w") as f:
            f.write("round,avg_loss,val_H1_macro,val_H2_macro,val_H3_macro,val_H4_macro,global_val_macro\n")

    # Federated Training Loop
    t_start = time.time()
    print(f"\n--- Starting Compliant FedAvg Training: Rounds {start_round} to {max_rounds} ---", flush=True)

    for r_idx in range(start_round, max_rounds + 1):
        round_t0 = time.time()
        client_updates = {}
        client_losses = {}

        # 1. Local Training across all hospitals
        for hid in hospitals:
            local_state, c_loss = train_client_round(
                client_id=hid,
                global_model=global_model,
                dataset=client_datasets[hid],
                slice_sampler=client_samplers[hid],
                patient_ids=client_train_pids[hid],
                allowed_modalities=TRAIN_MODALITIES[hid],
                device=device,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.train_seed,
                round_idx=r_idx,
                max_rounds=max_rounds,
                local_epochs=args.local_epochs,
                dry_run=args.dry_run,
            )
            client_updates[hid] = local_state
            client_losses[hid] = c_loss

        # 2. FedAvg Parameter Aggregation
        aggregated_state = aggregate_fedavg(client_updates, client_weights)
        global_model.load_state_dict(aggregated_state)

        # 3. Global Validation across all 4 hospitals
        hosp_val_metrics = {}
        for hid in hospitals:
            v_res = evaluate_client_validation(
                client_id=hid,
                model=global_model,
                dataset=client_datasets[hid],
                val_pids=client_val_pids[hid],
                eval_modalities=EVAL_MODALITIES[hid],
                device=device,
                batch_size=args.batch_size,
                evaluator=evaluator,
            )
            hosp_val_metrics[hid] = v_res

        # Weighted global validation score
        global_val_macro = sum(client_weights[h] * hosp_val_metrics[h]["macro"] for h in hospitals)
        avg_train_loss = sum(client_weights[h] * client_losses[h] for h in hospitals)

        # Log metrics
        with open(metrics_csv_path, "a") as f:
            f.write(f"{r_idx},{avg_train_loss:.6f},{hosp_val_metrics['H1']['macro']:.4f},{hosp_val_metrics['H2']['macro']:.4f},{hosp_val_metrics['H3']['macro']:.4f},{hosp_val_metrics['H4']['macro']:.4f},{global_val_macro:.4f}\n")

        round_elapsed = (time.time() - round_t0) / 60.0
        print(f"[FedAvg] Round {r_idx:03d}/{max_rounds:03d} ({round_elapsed:.1f}m) | Train Loss: {avg_train_loss:.4f} | Global Val Macro: {global_val_macro*100:.2f}% (H1={hosp_val_metrics['H1']['macro']*100:.1f}%, H2={hosp_val_metrics['H2']['macro']*100:.1f}%, H3={hosp_val_metrics['H3']['macro']*100:.1f}%, H4={hosp_val_metrics['H4']['macro']*100:.1f}%)", flush=True)

        # Check for improvement
        is_best = False
        if global_val_macro > best_val_macro:
            best_val_macro = global_val_macro
            best_round = r_idx
            no_improvement_count = 0
            is_best = True
            print(f"  --> Saved new best FedAvg model (Global Val Macro: {best_val_macro*100:.2f}%)", flush=True)
        else:
            no_improvement_count += 1

        # Atomic checkpoint save
        chkpt_data = {
            "round": r_idx,
            "model_state_dict": global_model.state_dict(),
            "best_val_macro": best_val_macro,
            "best_round": best_round,
            "no_improvement_count": no_improvement_count,
        }
        tmp_latest = chkpt_dir / "latest_model.pt.tmp"
        torch.save(chkpt_data, tmp_latest)
        tmp_latest.replace(latest_chkpt_path)

        if is_best:
            tmp_best = chkpt_dir / "best_model.pt.tmp"
            torch.save(chkpt_data, tmp_best)
            tmp_best.replace(best_chkpt_path)

        # Early stopping check
        if r_idx >= args.min_rounds and no_improvement_count >= args.patience:
            print(f"  Early stopping triggered at round {r_idx} (best: {best_val_macro*100:.2f}% at round {best_round})", flush=True)
            break

    total_time = (time.time() - t_start) / 60.0
    print(f"\n=== Training Complete for FedAvg in {total_time:.1f} minutes. Best Global Val Macro: {best_val_macro*100:.2f}% ===", flush=True)

    # 4. Universal 3D Test Set Evaluation (All 50 Pure Held-Out Patients for each Hospital)
    print("\n=== Evaluating Best FedAvg Model on Universal 50 Pure Held-Out Test Patients ===", flush=True)
    if best_chkpt_path.exists():
        best_chk = torch.load(best_chkpt_path, map_location=device)
        global_model.load_state_dict(best_chk["model_state_dict"])
        print(f"Loaded best checkpoint from Round {best_chk['best_round']}", flush=True)
    else:
        print("Warning: No best checkpoint found, evaluating with current model weights", flush=True)

    global_model.eval()

    test_manifest_path = Path(args.partitions_dir) / "h3_test_patients.json"
    if not test_manifest_path.exists():
        raise FileNotFoundError(f"Universal test set manifest not found: {test_manifest_path}")

    with open(test_manifest_path, "r") as f:
        test_manifest = json.load(f)

    pure_50_test_pids = test_manifest["patient_ids"] if isinstance(test_manifest, dict) and "patient_ids" in test_manifest else test_manifest
    if args.dry_run:
        pure_50_test_pids = pure_50_test_pids[:1]

    test_dataset = BraTSDataset(patient_ids=pure_50_test_pids, cache_root=preprocessed_path, modalities=ALL_MODALITIES)
    eval_csv_path = results_dir / "pure_50_test_patient_metrics.csv"
    eval_logger = EvaluationCSVLogger(eval_csv_path)

    all_hospital_results = {}

    with torch.no_grad():
        for hid in hospitals:
            hosp_mods = EVAL_MODALITIES[hid]
            patient_metrics = []

            for pid in pure_50_test_pids:
                vol = test_dataset.load_patient_volume(pid)
                lab_vol = vol["labels"]
                num_slices = lab_vol.shape[0]

                pred_logits_list = []
                for start in range(0, num_slices, args.batch_size):
                    end = min(start + args.batch_size, num_slices)
                    B = end - start

                    batch_inputs = {}
                    for m in ALL_MODALITIES:
                        if m in hosp_mods:
                            batch_inputs[m] = torch.from_numpy(vol["modalities"][m][start:end]).float().unsqueeze(1).to(device)
                        else:
                            # Zero-filled input for unpossessed test modalities
                            batch_inputs[m] = torch.zeros((B, 1, 240, 240), dtype=torch.float32, device=device)

                    logits_b = global_model(batch_inputs)
                    pred_logits_list.append(logits_b.cpu())

                pred_logits_3d = torch.cat(pred_logits_list, dim=0).permute(1, 0, 2, 3).numpy()
                p_metrics = evaluator.evaluate_patient_volume(pred_logits_3d, lab_vol)
                patient_metrics.append(p_metrics)

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
                    track_id=f"{hid}_fedavg",
                    hospital_id=hid,
                    dice_dict=dice_dict,
                    hd95_dict=hd95_dict,
                )

            mean_wt = float(np.mean([p["dice_WT"] for p in patient_metrics]))
            mean_tc = float(np.mean([p["dice_TC"] for p in patient_metrics]))
            mean_et = float(np.mean([p["dice_ET"] for p in patient_metrics]))
            mean_macro = float(np.mean([p["macro_dice"] for p in patient_metrics]))

            mean_hd_wt = float(np.mean([p["hd95_WT"] for p in patient_metrics]))
            mean_hd_tc = float(np.mean([p["hd95_TC"] for p in patient_metrics]))
            mean_hd_et = float(np.mean([p["hd95_ET"] for p in patient_metrics]))
            mean_hd_macro = float(np.mean([p["macro_hd95"] for p in patient_metrics]))

            all_hospital_results[hid] = {
                "WT": mean_wt, "TC": mean_tc, "ET": mean_et, "macro": mean_macro,
                "hd_WT": mean_hd_wt, "hd_TC": mean_hd_tc, "hd_ET": mean_hd_et, "hd_macro": mean_hd_macro,
            }

    print("\n" + "=" * 70, flush=True)
    print(f"  [RUN-3 FEDAVG RESULTS] Partition: {args.partition_seed} | Train Seed: {args.train_seed}", flush=True)
    print("=" * 70, flush=True)
    for hid in hospitals:
        res = all_hospital_results[hid]
        print(f"  {hid} (Mods: {EVAL_MODALITIES[hid]}): Macro={res['macro']*100:5.2f}% | WT={res['WT']*100:5.2f}% | TC={res['TC']*100:5.2f}% | ET={res['ET']*100:5.2f}% | Med HD95={res['hd_macro']:5.2f} mm", flush=True)
    print("=" * 70, flush=True)
    print(f"Results written to: {eval_csv_path}", flush=True)


if __name__ == "__main__":
    run_fedavg()
