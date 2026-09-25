"""
CAMFS M2 — Restricted Donor Teacher & Viability Gate
====================================================

Implements §10.2, §16.4 of the CAMFS M2 Specification:
1. RestrictedTeacher: Net2Net widened from Track S3 (T1, FLAIR) to S3+T1ce (T1, T1ce, FLAIR).
   - Day-0 property: Initialized with zero weights on new T1ce column -> exactly equals Base.
   - Encoders strictly frozen from Phase 1.
   - Trains widened fusion head + decoder on H1 (102 patients).
2. Temperature Calibration: Fits single scalar T_T^{cal} on H1-val by NLL minimization.
3. Teacher Viability Gate: Bootstrap percentile test on 12 H1 release-audit patients.
   - Requires non-inferiority on TC & WT (drop <= 0.02) and causal ET surge (P(Delta > 0) >= 0.90).
"""

from __future__ import annotations

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
from src.models.decoder import UNetDecoder
from src.models.encoder import UnimodalEncoder
from src.models.fusion import SubsetFusionHead, net2net_widen_fusion_head
from src.m2.frozen_base import FrozenBase, _extract_component_state_dict


class RestrictedTeacher(nn.Module):
    """
    Donor Restricted Teacher T_{(S3 + T1ce)} (§10.2).
    Modalities: ('T1', 'T1ce', 'FLAIR') in canonical order.
    """

    def __init__(
        self,
        encoder_t1: UnimodalEncoder,
        encoder_t1ce: UnimodalEncoder,
        encoder_flair: UnimodalEncoder,
        fusion_head: SubsetFusionHead,
        decoder: UNetDecoder,
    ) -> None:
        super().__init__()
        self.encoder_t1 = encoder_t1
        self.encoder_t1ce = encoder_t1ce
        self.encoder_flair = encoder_flair
        self.fusion_head = fusion_head
        self.decoder = decoder

        # Encoders are strictly frozen
        for p in self.encoder_t1.parameters():
            p.requires_grad = False
        for p in self.encoder_t1ce.parameters():
            p.requires_grad = False
        for p in self.encoder_flair.parameters():
            p.requires_grad = False

        self.encoder_t1.eval()
        self.encoder_t1ce.eval()
        self.encoder_flair.eval()

        # Fusion head and decoder are trainable
        for p in self.fusion_head.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x_dict: Dict with keys 'T1', 'T1ce', 'FLAIR' ([B, 1, 240, 240]).
        Returns:
            Logits tensor [B, 4, 240, 240].
        """
        # Ensure encoders stay in eval mode
        self.encoder_t1.eval()
        self.encoder_t1ce.eval()
        self.encoder_flair.eval()

        with torch.no_grad():
            feat_t1 = self.encoder_t1(x_dict["T1"])
            feat_t1ce = self.encoder_t1ce(x_dict["T1ce"])
            feat_flair = self.encoder_flair(x_dict["FLAIR"])

        mod_features = {
            "T1": feat_t1,
            "T1ce": feat_t1ce,
            "FLAIR": feat_flair,
        }

        f1, f2, f3, f4, z_s = self.fusion_head(mod_features)
        logits = self.decoder(z_s, f4, f3, f2, f1)
        return logits


def build_restricted_teacher(
    phase1_checkpoint_path: Union[str, Path],
    track_s3_checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> RestrictedTeacher:
    """
    Constructs RestrictedTeacher initialized via Net2Net widening from Track S3.
    """
    device = torch.device(device)

    # 1. Encoders
    encoder_t1 = UnimodalEncoder().to(device)
    encoder_t1ce = UnimodalEncoder().to(device)
    encoder_flair = UnimodalEncoder().to(device)

    p1_ckpt = torch.load(phase1_checkpoint_path, map_location=device)
    p1_sd = p1_ckpt.get("state_dict", p1_ckpt)

    encoder_t1.load_state_dict(_extract_component_state_dict(p1_sd, "T1", "T1."))
    encoder_t1ce.load_state_dict(_extract_component_state_dict(p1_sd, "T1ce", "T1ce."))
    encoder_flair.load_state_dict(_extract_component_state_dict(p1_sd, "FLAIR", "FLAIR."))

    # 2. Source Track S3 fusion head & decoder
    s3_ckpt = torch.load(track_s3_checkpoint_path, map_location=device)
    s3_sd = s3_ckpt.get("state_dict", s3_ckpt)

    s3_fusion = SubsetFusionHead(modality_subset=("T1", "FLAIR")).to(device)
    s3_fusion.load_state_dict(_extract_component_state_dict(s3_sd, "fusion", "fusion_head."))

    # 3. Net2Net widening: ('T1', 'FLAIR') -> ('T1', 'T1ce', 'FLAIR')
    widened_fusion = net2net_widen_fusion_head(
        source_head=s3_fusion,
        source_modalities=("T1", "FLAIR"),
        target_modalities=("T1", "T1ce", "FLAIR"),
    ).to(device)

    # 4. Decoder byte-copy
    decoder = UNetDecoder(num_classes=4).to(device)
    decoder.load_state_dict(_extract_component_state_dict(s3_sd, "decoder", "decoder."))

    teacher = RestrictedTeacher(
        encoder_t1=encoder_t1,
        encoder_t1ce=encoder_t1ce,
        encoder_flair=encoder_flair,
        fusion_head=widened_fusion,
        decoder=decoder,
    ).to(device)

    return teacher


def verify_teacher_day0_identity(
    teacher: RestrictedTeacher,
    frozen_base: FrozenBase,
    device: Union[str, torch.device] = "cpu",
    tol: float = 1e-4,
) -> float:
    """
    Day-0 Equality Verification:
    Before training, RestrictedTeacher on (T1, T1ce, FLAIR) must produce
    identical outputs to FrozenBase on (T1, FLAIR) because T1ce columns are zero.
    """
    teacher.eval()
    frozen_base.eval()

    with torch.no_grad():
        x_base = {
            "T1": torch.randn(2, 1, 240, 240, device=device),
            "FLAIR": torch.randn(2, 1, 240, 240, device=device),
        }
        x_teacher = {
            "T1": x_base["T1"],
            "T1ce": torch.randn(2, 1, 240, 240, device=device),  # arbitrary random input
            "FLAIR": x_base["FLAIR"],
        }

        _, logits_base = frozen_base(x_base)
        logits_teacher = teacher(x_teacher)

        max_diff = torch.max(torch.abs(logits_teacher - logits_base)).item()

    if max_diff > tol:
        raise AssertionError(
            f"Day-0 Teacher Identity Failed! Max abs diff: {max_diff:.2e} (tol={tol:.2e})"
        )
    return max_diff


def calibrate_teacher_temperature(
    teacher: RestrictedTeacher,
    dataset: BraTSDataset,
    val_patient_ids: List[str],
    device: Union[str, torch.device] = "cpu",
    num_grid_points: int = 201,
    t_min: float = 0.25,
    t_max: float = 8.0,
    seed: int = 6601,
    max_voxels: int = 1_000_000,
) -> float:
    """
    Fits scalar teacher temperature T_T^{cal} on H1 validation data by NLL minimization (§16.4).
    """
    teacher.eval()
    device = torch.device(device)

    all_logits_list = []
    all_targets_list = []

    with torch.no_grad():
        for pid in val_patient_ids:
            vol = dataset.load_patient_volume(pid)
            lab_vol = vol["labels"]  # (155, 240, 240)
            num_slices = lab_vol.shape[0]

            for start in range(0, num_slices, 16):
                end = min(start + 16, num_slices)
                x_dict = {
                    m: torch.from_numpy(vol["modalities"][m][start:end]).float().unsqueeze(1).to(device)
                    for m in ["T1", "T1ce", "FLAIR"]
                }
                logits = teacher(x_dict)  # (B, 4, 240, 240)
                y_batch = torch.from_numpy(lab_vol[start:end]).long().to(device)

                # Nonzero brain voxels (where any modality is nonzero)
                brain_mask = (x_dict["T1"].squeeze(1) != 0) | (x_dict["FLAIR"].squeeze(1) != 0)
                if brain_mask.any():
                    logits_masked = logits.permute(0, 2, 3, 1)[brain_mask]  # (N_vox, 4)
                    y_masked = y_batch[brain_mask]  # (N_vox,)
                    all_logits_list.append(logits_masked.cpu())
                    all_targets_list.append(y_masked.cpu())

    if not all_logits_list:
        return 1.0

    cat_logits = torch.cat(all_logits_list, dim=0)
    cat_targets = torch.cat(all_targets_list, dim=0)

    # Subsample up to max_voxels using deterministic seed
    num_total = cat_logits.shape[0]
    if num_total > max_voxels:
        rng = np.random.Generator(np.random.PCG64(seed))
        indices = rng.choice(num_total, size=max_voxels, replace=False)
        cat_logits = cat_logits[indices]
        cat_targets = cat_targets[indices]

    cat_logits = cat_logits.to(device)
    cat_targets = cat_targets.to(device)

    # Grid search over 201 log-spaced temperatures
    temps = np.logspace(np.log10(t_min), np.log10(t_max), num=num_grid_points)
    best_nll = float("inf")
    best_t = 1.0

    with torch.no_grad():
        for t in temps:
            scaled_logits = cat_logits / t
            nll = F.cross_entropy(scaled_logits, cat_targets, reduction="mean").item()
            if nll < best_nll - 1e-6:
                best_nll = nll
                best_t = float(t)

    return best_t


def evaluate_teacher_viability(
    teacher: RestrictedTeacher,
    frozen_base: FrozenBase,
    dataset: BraTSDataset,
    release_audit_patient_ids: List[str],
    evaluator: PatientEvaluator,
    device: Union[str, torch.device] = "cpu",
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 7701,
    non_inferiority_delta: float = 0.02,
    et_confidence_threshold: float = 0.90,
) -> Tuple[int, Dict[str, float]]:
    """
    Evaluates Teacher Viability Gate (§10.2).
    Returns (q_T, metrics_dict).
    """
    teacher.eval()
    frozen_base.eval()
    device = torch.device(device)

    base_dice_et, base_dice_tc, base_dice_wt = [], [], []
    teach_dice_et, teach_dice_tc, teach_dice_wt = [], [], []

    with torch.no_grad():
        for pid in release_audit_patient_ids:
            vol = dataset.load_patient_volume(pid)
            lab_vol = vol["labels"]
            num_slices = lab_vol.shape[0]

            logits_b_list = []
            logits_t_list = []

            for start in range(0, num_slices, 16):
                end = min(start + 16, num_slices)
                x_base = {
                    "T1": torch.from_numpy(vol["modalities"]["T1"][start:end]).float().unsqueeze(1).to(device),
                    "FLAIR": torch.from_numpy(vol["modalities"]["FLAIR"][start:end]).float().unsqueeze(1).to(device),
                }
                x_teach = {
                    "T1": x_base["T1"],
                    "T1ce": torch.from_numpy(vol["modalities"]["T1ce"][start:end]).float().unsqueeze(1).to(device),
                    "FLAIR": x_base["FLAIR"],
                }

                _, lb = frozen_base(x_base)
                lt = teacher(x_teach)

                logits_b_list.append(lb.cpu())
                logits_t_list.append(lt.cpu())

            pred_b = torch.cat(logits_b_list, dim=0).permute(1, 0, 2, 3).numpy()
            pred_t = torch.cat(logits_t_list, dim=0).permute(1, 0, 2, 3).numpy()

            m_b = evaluator.evaluate_patient_volume(pred_b, lab_vol)
            m_t = evaluator.evaluate_patient_volume(pred_t, lab_vol)

            base_dice_et.append(m_b["dice_ET"])
            base_dice_tc.append(m_b["dice_TC"])
            base_dice_wt.append(m_b["dice_WT"])

            teach_dice_et.append(m_t["dice_ET"])
            teach_dice_tc.append(m_t["dice_TC"])
            teach_dice_wt.append(m_t["dice_WT"])

    base_et = np.array(base_dice_et)
    teach_et = np.array(teach_dice_et)
    delta_et = teach_et - base_et

    mean_delta_et = float(np.mean(delta_et))
    mean_delta_tc = float(np.mean(np.array(teach_dice_tc) - np.array(base_dice_tc)))
    mean_delta_wt = float(np.mean(np.array(teach_dice_wt) - np.array(base_dice_wt)))

    # Non-inferiority check: TC and WT do not drop by more than 0.02
    tc_non_inferior = mean_delta_tc >= -non_inferiority_delta
    wt_non_inferior = mean_delta_wt >= -non_inferiority_delta

    # Bootstrap test on ET Dice surge
    n_p = len(delta_et)
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    bootstrap_samples = rng.choice(delta_et, size=(bootstrap_resamples, n_p), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)

    prob_et_improved = float(np.mean(bootstrap_means > 0.0))

    passed = (
        tc_non_inferior
        and wt_non_inferior
        and prob_et_improved >= et_confidence_threshold
    )
    q_T = 1 if passed else 0

    metrics = {
        "q_T": q_T,
        "mean_delta_ET": mean_delta_et,
        "mean_delta_TC": mean_delta_tc,
        "mean_delta_WT": mean_delta_wt,
        "prob_ET_improved": prob_et_improved,
        "teacher_mean_ET": float(np.mean(teach_et)),
        "base_mean_ET": float(np.mean(base_et)),
    }
    return q_T, metrics
