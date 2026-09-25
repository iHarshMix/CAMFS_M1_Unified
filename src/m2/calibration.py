"""
CAMFS M2 — Recipient Gate Calibration & Acceptance Gating
=========================================================

Implements §10.5, §16.6 of the CAMFS M2 Specification:
- RecipientGate: 4 learned scalar parameters a_c initialized to -4.0 (alpha_c ~ 0.018).
  ell_{aug, c} = ell_{B, c} + d_g * alpha_c * Delta_ell_c
- Gate calibration: Trained on H3 M2-calibration split (6 patients) for exactly 50 epochs
  with Adam (lr=1e-2, weight_decay=0).
- Acceptance evaluation: Evaluated on untouched H3 M2-acceptance-validation split (6 patients).
  Requires:
    1. Non-inferiority on TC (drop <= 0.02)
    2. Non-inferiority on WT (drop <= 0.02)
    3. Causal ET improvement (> 0)
- Deployment routing decision:
    d_deploy = 1 iff (q_T == 1 and acceptance_pass) else 0.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.data.brats_dataset import BraTSDataset
from src.data.slice_sampler import SliceSampler
from src.losses import SoftDiceCrossEntropyLoss
from src.metrics import PatientEvaluator
from src.m2.adapter import CDRDAdapter
from src.m2.frozen_base import FrozenBase


class RecipientGate(nn.Module):
    """
    Per-class learned gating sidecar (§10.5).
    Contains exactly 4 scalar parameters.
    """

    def __init__(self, init_val: float = -4.0, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes
        # Gate logits a_c initialized to -4.0 (so sigma(-4) ~ 0.017986)
        self.a = nn.Parameter(torch.full((num_classes,), fill_value=init_val, dtype=torch.float32))

    @property
    def alphas(self) -> torch.Tensor:
        """Returns alpha_c = sigmoid(a_c)."""
        return torch.sigmoid(self.a)

    def forward(
        self,
        logits_B: torch.Tensor,
        delta_ell: torch.Tensor,
        d_g: float = 1.0,
    ) -> torch.Tensor:
        """
        Computes augmented logits:
            ell_{aug, c} = ell_{B, c} + d_g * alpha_c * Delta_ell_c
        Args:
            logits_B: [B, 4, H, W]
            delta_ell: [B, 4, H, W]
            d_g: scalar routing gate (1.0 or 0.0)
        """
        # Reshape alpha for broadcasting: [1, 4, 1, 1]
        alpha = self.alphas.view(1, self.num_classes, 1, 1)
        return logits_B + d_g * alpha * delta_ell


def train_recipient_gates(
    gate: RecipientGate,
    frozen_base: FrozenBase,
    adapter: CDRDAdapter,
    dataset: BraTSDataset,
    h3_cal_patient_ids: List[str],
    device: Union[str, torch.device] = "cpu",
    lr: float = 1e-2,
    epochs: int = 50,
    batch_size: int = 16,
    seed: int = 42,
) -> Tuple[RecipientGate, Dict[str, Any]]:
    """
    Fits 4 gate logits on H3 calibration data (§16.6).
    Base and Adapter are strictly frozen. Exactly 50 epochs, keeps final checkpoint.
    """
    device = torch.device(device)
    gate = gate.to(device)
    frozen_base = frozen_base.to(device)
    adapter = adapter.to(device)

    frozen_base.eval()
    adapter.eval()
    gate.train()

    optimizer = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=0.0)
    criterion = SoftDiceCrossEntropyLoss()
    sampler = SliceSampler(cache_root=dataset.cache_root, patient_ids=h3_cal_patient_ids)

    print(f"\n[Recipient Calibration] Fitting 4 gate parameters on {len(h3_cal_patient_ids)} H3 calibration patients ({epochs} epochs)...", flush=True)

    history = []

    for epoch in range(epochs):
        epoch_samples = sampler.get_epoch_samples(seed=seed + epoch, is_training=True)
        total_slices = len(epoch_samples)
        running_loss = 0.0
        num_batches = 0

        for b_start in range(0, total_slices, batch_size):
            b_samples = epoch_samples[b_start : b_start + batch_size]

            t1_list, flair_list, y_list = [], [], []
            for pid, s_idx in b_samples:
                pdir = dataset.cache_root / pid
                t1_vol = np.load(pdir / "t1.npy", mmap_mode="r")
                flair_vol = np.load(pdir / "flair.npy", mmap_mode="r")
                lbl_vol = np.load(pdir / "labels.npy", mmap_mode="r")

                t1_list.append(torch.from_numpy(np.array(t1_vol[s_idx], dtype=np.float32)).unsqueeze(0))
                flair_list.append(torch.from_numpy(np.array(flair_vol[s_idx], dtype=np.float32)).unsqueeze(0))
                y_list.append(torch.from_numpy(np.array(lbl_vol[s_idx], dtype=np.int64)))

            x_base = {
                "T1": torch.stack(t1_list, dim=0).to(device),
                "FLAIR": torch.stack(flair_list, dim=0).to(device),
            }
            y_batch = torch.stack(y_list, dim=0).to(device)

            with torch.no_grad():
                features_b, logits_b = frozen_base(x_base)
                delta_ell, _ = adapter(features_b, logits_b)

            logits_aug = gate(logits_b, delta_ell, d_g=1.0)
            loss = criterion(logits_aug, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        current_alphas = gate.alphas.detach().cpu().numpy().tolist()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            alpha_str = ", ".join([f"c{i}:{a:.4f}" for i, a in enumerate(current_alphas)])
            print(f"  [Gate Epoch {epoch+1:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Alphas: [{alpha_str}]", flush=True)

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "alphas": current_alphas,
        })

    # §16.6: Keep final checkpoint
    meta = {
        "final_alphas": gate.alphas.detach().cpu().numpy().tolist(),
        "final_a": gate.a.detach().cpu().numpy().tolist(),
        "history": history,
    }
    return gate, meta


def evaluate_recipient_acceptance(
    gate: RecipientGate,
    frozen_base: FrozenBase,
    adapter: CDRDAdapter,
    dataset: BraTSDataset,
    h3_accept_val_patient_ids: List[str],
    evaluator: PatientEvaluator,
    q_T: int,
    device: Union[str, torch.device] = "cpu",
    non_inferiority_delta: float = 0.02,
) -> Tuple[int, bool, Dict[str, float]]:
    """
    Evaluates Augmented vs Fallback acceptance rule on untouched H3 acceptance-val split (§10.5).

    Returns:
        d_deploy: 1 if all criteria pass, else 0
        acceptance_pass: True/False
        summary_metrics: Dict of comparative metrics
    """
    device = torch.device(device)
    gate.eval()
    frozen_base.eval()
    adapter.eval()

    base_dice_et, base_dice_tc, base_dice_wt, base_dice_macro = [], [], [], []
    aug_dice_et, aug_dice_tc, aug_dice_wt, aug_dice_macro = [], [], [], []

    with torch.no_grad():
        for pid in h3_accept_val_patient_ids:
            vol = dataset.load_patient_volume(pid)
            lab_vol = vol["labels"]
            num_slices = lab_vol.shape[0]

            logits_b_list, logits_aug_list = [], []

            for s in range(0, num_slices, 16):
                e = min(s + 16, num_slices)
                x_b = {
                    "T1": torch.from_numpy(vol["modalities"]["T1"][s:e]).float().unsqueeze(1).to(device),
                    "FLAIR": torch.from_numpy(vol["modalities"]["FLAIR"][s:e]).float().unsqueeze(1).to(device),
                }
                feats_b, lb = frozen_base(x_b)
                delta_ell, _ = adapter(feats_b, lb)
                l_aug = gate(lb, delta_ell, d_g=1.0)

                logits_b_list.append(lb.cpu())
                logits_aug_list.append(l_aug.cpu())

            pred_b = torch.cat(logits_b_list, dim=0).permute(1, 0, 2, 3).numpy()
            pred_aug = torch.cat(logits_aug_list, dim=0).permute(1, 0, 2, 3).numpy()

            mb = evaluator.evaluate_patient_volume(pred_b, lab_vol)
            ma = evaluator.evaluate_patient_volume(pred_aug, lab_vol)

            base_dice_et.append(mb["dice_ET"])
            base_dice_tc.append(mb["dice_TC"])
            base_dice_wt.append(mb["dice_WT"])
            base_dice_macro.append(mb["macro_dice"])

            aug_dice_et.append(ma["dice_ET"])
            aug_dice_tc.append(ma["dice_TC"])
            aug_dice_wt.append(ma["dice_WT"])
            aug_dice_macro.append(ma["macro_dice"])

    mean_b_et = float(np.mean(base_dice_et))
    mean_a_et = float(np.mean(aug_dice_et))
    delta_et = mean_a_et - mean_b_et

    mean_b_tc = float(np.mean(base_dice_tc))
    mean_a_tc = float(np.mean(aug_dice_tc))
    delta_tc = mean_a_tc - mean_b_tc

    mean_b_wt = float(np.mean(base_dice_wt))
    mean_a_wt = float(np.mean(aug_dice_wt))
    delta_wt = mean_a_wt - mean_b_wt

    mean_b_macro = float(np.mean(base_dice_macro))
    mean_a_macro = float(np.mean(aug_dice_macro))
    delta_macro = mean_a_macro - mean_b_macro

    # Acceptance criteria (§10.5):
    # 1. Non-inferiority on TC (drop <= 0.02)
    # 2. Non-inferiority on WT (drop <= 0.02)
    # 3. Causal improvement on ET (> 0)
    tc_ok = delta_tc >= -non_inferiority_delta
    wt_ok = delta_wt >= -non_inferiority_delta
    et_ok = delta_et > 0.0

    acceptance_pass = tc_ok and wt_ok and et_ok
    d_deploy = 1 if (q_T == 1 and acceptance_pass) else 0

    metrics = {
        "acceptance_pass": acceptance_pass,
        "d_deploy": d_deploy,
        "q_T": q_T,
        "base_macro": mean_b_macro,
        "aug_macro": mean_a_macro,
        "delta_macro": delta_macro,
        "base_ET": mean_b_et,
        "aug_ET": mean_a_et,
        "delta_ET": delta_et,
        "delta_TC": delta_tc,
        "delta_WT": delta_wt,
    }

    print("\n[Recipient Acceptance Evaluation Results]:")
    print(f"  Base Macro: {mean_b_macro*100:.2f}% | Aug Macro: {mean_a_macro*100:.2f}% (Delta: {delta_macro*100:+.2f}%)")
    print(f"  Base ET:    {mean_b_et*100:.2f}% | Aug ET:    {mean_a_et*100:.2f}% (Delta: {delta_et*100:+.2f}%)")
    print(f"  Delta TC:   {delta_tc*100:+.2f}% (non-inferior threshold: -{non_inferiority_delta*100:.1f}%)")
    print(f"  Delta WT:   {delta_wt*100:+.2f}% (non-inferior threshold: -{non_inferiority_delta*100:.1f}%)")
    print(f"  Teacher Viability q_T: {q_T}")
    print(f"  => Acceptance Pass: {acceptance_pass} | Final Deployment Gate d_deploy: {d_deploy}")

    return d_deploy, acceptance_pass, metrics


def train_cdrd_ft(
    frozen_base: FrozenBase,
    adapter: CDRDAdapter,
    dataset: BraTSDataset,
    h3_cal_patient_ids: List[str],
    device: Union[str, torch.device] = "cpu",
    ft_lr: float = 1e-3,
    ft_epochs: int = 50,
    gate_lr: float = 1e-2,
    gate_epochs: int = 50,
    batch_size: int = 16,
    seed: int = 42,
) -> Tuple[CDRDAdapter, RecipientGate, Dict[str, Any]]:
    """
    Implements CDRD-FT (E4b) secondary ablation (§10.5, §15.2):
    Stage 1: Locally fine-tune adapter phi on H3 calibration data for ft_epochs with
             d_g=1.0, alpha_c=1.0, and Dice + CrossEntropy loss.
    Stage 2: Freeze adapter phi, reinitialize gate a_c = -4.0, and fit only the 4 gate
             logits for gate_epochs using standard recipient calibration.
    """
    device = torch.device(device)
    frozen_base = frozen_base.to(device)
    adapter = adapter.to(device)

    frozen_base.eval()
    for p in frozen_base.parameters():
        p.requires_grad = False

    # Stage 1: Fine-tune adapter
    adapter.train()
    for p in adapter.parameters():
        p.requires_grad = True

    optimizer_ft = torch.optim.AdamW(adapter.parameters(), lr=ft_lr, weight_decay=1e-4)
    criterion = SoftDiceCrossEntropyLoss()
    sampler = SliceSampler(cache_root=dataset.cache_root, patient_ids=h3_cal_patient_ids)

    print(f"\n[CDRD-FT Stage 1] Fine-tuning transferred adapter on {len(h3_cal_patient_ids)} H3 calibration patients ({ft_epochs} epochs)...", flush=True)
    ft_history = []

    for epoch in range(ft_epochs):
        epoch_samples = sampler.get_epoch_samples(seed=seed + epoch, is_training=True)
        total_slices = len(epoch_samples)
        running_loss = 0.0
        num_batches = 0

        for b_start in range(0, total_slices, batch_size):
            b_samples = epoch_samples[b_start : b_start + batch_size]

            t1_list, flair_list, y_list = [], [], []
            for pid, s_idx in b_samples:
                pdir = dataset.cache_root / pid
                t1_vol = np.load(pdir / "t1.npy", mmap_mode="r")
                flair_vol = np.load(pdir / "flair.npy", mmap_mode="r")
                lbl_vol = np.load(pdir / "labels.npy", mmap_mode="r")

                t1_list.append(torch.from_numpy(np.array(t1_vol[s_idx], dtype=np.float32)).unsqueeze(0))
                flair_list.append(torch.from_numpy(np.array(flair_vol[s_idx], dtype=np.float32)).unsqueeze(0))
                y_list.append(torch.from_numpy(np.array(lbl_vol[s_idx], dtype=np.int64)))

            x_base = {
                "T1": torch.stack(t1_list, dim=0).to(device),
                "FLAIR": torch.stack(flair_list, dim=0).to(device),
            }
            y_batch = torch.stack(y_list, dim=0).to(device)

            with torch.no_grad():
                features_b, logits_b = frozen_base(x_base)

            delta_ell, logits_cand = adapter(features_b, logits_b)
            loss = criterion(logits_cand, y_batch)

            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == ft_epochs:
            print(f"  [CDRD-FT Epoch {epoch+1:02d}/{ft_epochs:02d}] Seg Loss: {avg_loss:.4f}", flush=True)

        ft_history.append({"epoch": epoch + 1, "loss": avg_loss})

    # Stage 2: Freeze adapter and fit recipient gate
    print("\n[CDRD-FT Stage 2] Freezing fine-tuned adapter and calibrating recipient gate (§10.5)...", flush=True)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad = False

    gate = RecipientGate(init_val=-4.0).to(device)
    gate, gate_meta = train_recipient_gates(
        gate=gate,
        frozen_base=frozen_base,
        adapter=adapter,
        dataset=dataset,
        h3_cal_patient_ids=h3_cal_patient_ids,
        device=device,
        lr=gate_lr,
        epochs=gate_epochs,
        batch_size=batch_size,
        seed=seed,
    )

    meta = {
        "ft_history": ft_history,
        "gate_meta": gate_meta,
        "final_alphas": gate_meta["final_alphas"],
    }
    return adapter, gate, meta

