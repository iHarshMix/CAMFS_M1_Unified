"""
CAMFS M2 — CDRD Adapter Loss & Training Module
===============================================

Implements §10.4, §16.5 of the CAMFS M2 Specification:
- 4-Term CDRD Objective:
    L_A = lambda_seg * L_DiceCE(y, cand)
        + lambda_KD * T^2 * KL(softmax(ell_T^cal / T) || softmax(cand / T))
        + lambda_delta * SmoothL1(Delta_ell, Delta_ell_star)
        + lambda_TV * TV(Delta_ell)
- Exact target coordinate:
    Delta_ell_star = stop_gradient(center(ell_T^cal - ell_B))
- H1 Adapter Training Loop with early stopping on H1 validation Macro Dice.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.brats_dataset import BraTSDataset
from src.data.slice_sampler import SliceSampler
from src.losses import SoftDiceCrossEntropyLoss
from src.metrics import PatientEvaluator
from src.m2.adapter import CDRDAdapter
from src.m2.frozen_base import FrozenBase
from src.m2.teacher import RestrictedTeacher


def compute_total_variation(delta_ell: torch.Tensor) -> torch.Tensor:
    """
    Computes separately normalized horizontal and vertical forward difference TV (§10.4).
    Shape: [N, 4, H, W]
    """
    N, C, H, W = delta_ell.shape

    # Horizontal forward difference
    diff_h = torch.abs(delta_ell[:, :, :, 1:] - delta_ell[:, :, :, :-1])
    norm_h = 4.0 * N * H * (W - 1)
    tv_h = torch.sum(diff_h) / norm_h

    # Vertical forward difference
    diff_v = torch.abs(delta_ell[:, :, 1:, :] - delta_ell[:, :, :-1, :])
    norm_v = 4.0 * N * (H - 1) * W
    tv_v = torch.sum(diff_v) / norm_v

    return tv_h + tv_v


def compute_kl_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Computes class-sum inside spatial mean KL divergence (§10.4):
    KL(P || Q) = (1 / |Omega|) * sum_{q in Omega} sum_{c=1}^4 P_c(q) * [log P_c(q) - log Q_c(q)]
    using FP32 log_softmax for numerical stability.
    """
    # Scale logits by KD temperature T
    scaled_p = p_logits / temperature
    scaled_q = q_logits / temperature

    log_p = F.log_softmax(scaled_p, dim=1)
    p = F.softmax(scaled_p, dim=1)
    log_q = F.log_softmax(scaled_q, dim=1)

    # Pixelwise KL: sum over classes (dim=1)
    kl_per_pixel = torch.sum(p * (log_p - log_q), dim=1)  # [B, H, W]

    # Mean over batch and spatial locations
    return torch.mean(kl_per_pixel)


class CDRDLoss(nn.Module):
    """
    CDRD 4-Term Composite Loss Function (§10.4).
    """

    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_kd: float = 1.0,
        lambda_delta: float = 0.1,
        lambda_tv: float = 1e-5,
        temperature: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_kd = lambda_kd
        self.lambda_delta = lambda_delta
        self.lambda_tv = lambda_tv
        self.temperature = temperature

        self.seg_loss_fn = SoftDiceCrossEntropyLoss()

    def forward(
        self,
        delta_ell: torch.Tensor,
        logits_cand: torch.Tensor,
        logits_t_cal: torch.Tensor,
        logits_b: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes composite adapter loss.
        """
        # 1. Segmentation Loss: Dice + CE on candidates
        loss_seg = self.seg_loss_fn(logits_cand, targets)

        # 2. Knowledge Distillation Loss: T^2 * KL
        loss_kd_raw = compute_kl_divergence(
            p_logits=logits_t_cal,
            q_logits=logits_cand,
            temperature=self.temperature,
        )
        loss_kd = (self.temperature ** 2) * loss_kd_raw

        # 3. Residual Distillation Target: delta_star = sg(center(ell_T^cal - ell_B))
        raw_diff = (logits_t_cal - logits_b).detach()
        mean_diff = torch.mean(raw_diff, dim=1, keepdim=True)
        delta_star = raw_diff - mean_diff

        # SmoothL1 with beta=1.0
        loss_delta = F.smooth_l1_loss(delta_ell, delta_star, beta=1.0, reduction="mean")

        # 4. Total Variation Regularization
        loss_tv = compute_total_variation(delta_ell)

        # Composite total
        total_loss = (
            self.lambda_seg * loss_seg
            + self.lambda_kd * loss_kd
            + self.lambda_delta * loss_delta
            + self.lambda_tv * loss_tv
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_seg": loss_seg.item(),
            "loss_kd": loss_kd.item(),
            "loss_delta": loss_delta.item(),
            "loss_tv": loss_tv.item(),
        }
        return total_loss, loss_dict


def train_cdrd_adapter(
    adapter: CDRDAdapter,
    frozen_base: FrozenBase,
    restricted_teacher: RestrictedTeacher,
    teacher_temperature: float,
    dataset: BraTSDataset,
    h1_train_patient_ids: List[str],
    h1_val_patient_ids: List[str],
    evaluator: PatientEvaluator,
    device: Union[str, torch.device] = "cpu",
    lr: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 7,
    batch_size: int = 16,
    seed: int = 17,
    lambda_seg: float = 1.0,
    lambda_kd: float = 1.0,
    lambda_delta: float = 0.1,
    lambda_tv: float = 1e-5,
    kd_temperature: float = 2.0,
) -> Tuple[CDRDAdapter, Dict[str, Any]]:
    """
    Trains CDRD Adapter on H1 paired slices (§16.5).
    """
    device = torch.device(device)
    adapter = adapter.to(device)
    frozen_base = frozen_base.to(device)
    restricted_teacher = restricted_teacher.to(device)

    frozen_base.eval()
    restricted_teacher.eval()

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = CDRDLoss(
        lambda_seg=lambda_seg,
        lambda_kd=lambda_kd,
        lambda_delta=lambda_delta,
        lambda_tv=lambda_tv,
        temperature=kd_temperature,
    )

    sampler = SliceSampler(cache_root=dataset.cache_root, patient_ids=h1_train_patient_ids)

    best_val_macro = -1.0
    best_weights = copy.deepcopy(adapter.state_dict())
    patience_counter = 0
    history = []

    print(f"\n[CDRD Training] Starting Adapter Optimization on {len(h1_train_patient_ids)} H1 patients (max {max_epochs} epochs)...", flush=True)

    for epoch in range(max_epochs):
        adapter.train()
        epoch_samples = sampler.get_epoch_samples(seed=seed + epoch, is_training=True)
        total_slices = len(epoch_samples)

        running_losses = {"loss_total": 0.0, "loss_seg": 0.0, "loss_kd": 0.0, "loss_delta": 0.0, "loss_tv": 0.0}
        num_batches = 0

        # Memory optimization: process batch by batch
        for b_start in range(0, total_slices, batch_size):
            b_samples = epoch_samples[b_start : b_start + batch_size]
            b_size = len(b_samples)

            # Assemble batch tensors
            t1_list, t1ce_list, flair_list, y_list = [], [], [], []
            for pid, s_idx in b_samples:
                pdir = dataset.cache_root / pid
                t1_vol = np.load(pdir / "t1.npy", mmap_mode="r")
                t1ce_vol = np.load(pdir / "t1ce.npy", mmap_mode="r")
                flair_vol = np.load(pdir / "flair.npy", mmap_mode="r")
                lbl_vol = np.load(pdir / "labels.npy", mmap_mode="r")

                t1_list.append(torch.from_numpy(np.array(t1_vol[s_idx], dtype=np.float32)).unsqueeze(0))
                t1ce_list.append(torch.from_numpy(np.array(t1ce_vol[s_idx], dtype=np.float32)).unsqueeze(0))
                flair_list.append(torch.from_numpy(np.array(flair_vol[s_idx], dtype=np.float32)).unsqueeze(0))
                y_list.append(torch.from_numpy(np.array(lbl_vol[s_idx], dtype=np.int64)))

            x_base = {
                "T1": torch.stack(t1_list, dim=0).to(device),
                "FLAIR": torch.stack(flair_list, dim=0).to(device),
            }
            x_teacher = {
                "T1": x_base["T1"],
                "T1ce": torch.stack(t1ce_list, dim=0).to(device),
                "FLAIR": x_base["FLAIR"],
            }
            y_batch = torch.stack(y_list, dim=0).to(device)

            # Frozen base and teacher forward passes with stop-gradient
            with torch.no_grad():
                features_b, logits_b = frozen_base(x_base)
                logits_t = restricted_teacher(x_teacher)
                logits_t_cal = logits_t / teacher_temperature

            # Adapter forward pass
            delta_ell, logits_cand = adapter(features_b, logits_b)

            # Compute loss
            loss, loss_dict = loss_fn(
                delta_ell=delta_ell,
                logits_cand=logits_cand,
                logits_t_cal=logits_t_cal,
                logits_b=logits_b,
                targets=y_batch,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in running_losses:
                running_losses[k] += loss_dict[k]
            num_batches += 1

        avg_train_loss = {k: v / max(1, num_batches) for k, v in running_losses.items()}

        # -------------------------------------------------------------------
        # Validation Evaluation on 13 H1 Validation Patients
        # -------------------------------------------------------------------
        adapter.eval()
        val_macro_dices = []
        val_et_dices = []

        with torch.no_grad():
            for pid in h1_val_patient_ids:
                vol = dataset.load_patient_volume(pid)
                lab_vol = vol["labels"]
                num_slices = lab_vol.shape[0]

                pred_cand_list = []
                for s in range(0, num_slices, batch_size):
                    e = min(s + batch_size, num_slices)
                    x_b = {
                        "T1": torch.from_numpy(vol["modalities"]["T1"][s:e]).float().unsqueeze(1).to(device),
                        "FLAIR": torch.from_numpy(vol["modalities"]["FLAIR"][s:e]).float().unsqueeze(1).to(device),
                    }
                    feats_b, lb = frozen_base(x_b)
                    _, l_cand = adapter(feats_b, lb)
                    pred_cand_list.append(l_cand.cpu())

                pred_cand = torch.cat(pred_cand_list, dim=0).permute(1, 0, 2, 3).numpy()
                m = evaluator.evaluate_patient_volume(pred_cand, lab_vol)
                val_macro_dices.append(m["macro_dice"])
                val_et_dices.append(m["dice_ET"])

        val_macro = float(np.mean(val_macro_dices))
        val_et = float(np.mean(val_et_dices))

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss["loss_total"],
            "val_macro_dice": val_macro,
            "val_et_dice": val_et,
        }
        history.append(epoch_record)

        print(
            f"  [Epoch {epoch+1:02d}/{max_epochs:02d}] "
            f"Loss: {avg_train_loss['loss_total']:.4f} "
            f"(Seg: {avg_train_loss['loss_seg']:.4f}, KD: {avg_train_loss['loss_kd']:.4f}, Delta: {avg_train_loss['loss_delta']:.4f}) | "
            f"Val Macro: {val_macro*100:.2f}% | Val ET: {val_et*100:.2f}%",
            flush=True,
        )

        # Model selection: improvement threshold 10^-4 (§16.5)
        if val_macro > best_val_macro + 1e-4:
            best_val_macro = val_macro
            best_weights = copy.deepcopy(adapter.state_dict())
            patience_counter = 0
            print(f"    ⭐ Best adapter checkpoint updated: Val Macro = {val_macro*100:.2f}%", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch+1} (patience={patience})", flush=True)
                break

    adapter.load_state_dict(best_weights)
    meta = {
        "best_val_macro_dice": best_val_macro,
        "epochs_trained": epoch + 1,
        "history": history,
    }
    return adapter, meta
