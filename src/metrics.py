"""
CAMFS M1 — 3D Patient Metrics & Evaluation Module
==================================================

Implements §13.5 of the CAMFS M1 Specification.
Includes:
  - Label back-mapping: Model {0, 1, 2, 3} -> BraTS {0, 1, 2, 4}
  - 3D Tumor Region definitions: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)
  - 3D Patient-level Dice score with empty volume conventions (both empty=1.0, one empty=0.0)
  - 3D Patient-level 95th Percentile Hausdorff Distance (HD95) in physical mm with grid diagonal penalty
  - PatientEvaluator class for full test volume evaluation & macro metric calculation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

# Label mapping constants (§13.5)
MODEL_TO_BRATS_MAP = {0: 0, 1: 1, 2: 2, 3: 4}
BRATS_TO_MODEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}


def backmap_model_to_brats(model_mask: np.ndarray) -> np.ndarray:
    """
    Map model class predictions {0, 1, 2, 3} back to BraTS labels {0, 1, 2, 4}.
    """
    brats_mask = np.zeros_like(model_mask, dtype=np.uint8)
    brats_mask[model_mask == 1] = 1  # NCR/NET
    brats_mask[model_mask == 2] = 2  # ED
    brats_mask[model_mask == 3] = 4  # ET
    return brats_mask


def extract_tumor_region_mask(labels: np.ndarray, region: str) -> np.ndarray:
    """
    Extract binary mask for tumor region (§13.5):
      - WT (Whole Tumor): labels {1, 2, 4}
      - TC (Tumor Core): labels {1, 4}
      - ET (Enhancing Tumor): label {4}
    """
    region = region.upper()
    if region == "WT":
        return np.isin(labels, [1, 2, 4])
    elif region == "TC":
        return np.isin(labels, [1, 4])
    elif region == "ET":
        return labels == 4
    else:
        raise ValueError(f"Unknown region '{region}'. Must be one of ('WT', 'TC', 'ET').")


def compute_3d_dice(pred_binary: np.ndarray, target_binary: np.ndarray) -> float:
    """
    Compute 3D patient-level Dice score (§13.5).

    Conventions (§13.5):
      - Both empty (|P|=0 and |G|=0): Dice = 1.0
      - One empty (|P|=0 xor |G|=0): Dice = 0.0
    """
    p_count = int(np.sum(pred_binary))
    g_count = int(np.sum(target_binary))

    if p_count == 0 and g_count == 0:
        return 1.0
    if p_count == 0 or g_count == 0:
        return 0.0

    intersection = int(np.sum(pred_binary & target_binary))
    dice = (2.0 * intersection) / float(p_count + g_count)
    return float(dice)


def compute_grid_diagonal_penalty(shape: Tuple[int, ...], spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    """Compute physical grid diagonal length in mm for one-empty HD95 penalty (§13.5)."""
    return math.sqrt(sum((dim * sp) ** 2 for dim, sp in zip(shape, spacing)))


def compute_3d_hd95(
    pred_binary: np.ndarray,
    target_binary: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> Tuple[float, bool]:
    """
    Compute 3D patient-level 95th Percentile Hausdorff Distance (HD95) in physical mm (§13.5).

    Conventions (§13.5):
      - Both empty (|P|=0 and |G|=0): HD95 = 0.0 mm, is_one_empty = False
      - One empty (|P|=0 xor |G|=0): HD95 = GridDiagonalPenalty (mm), is_one_empty = True

    Returns:
        Tuple of (hd95_value_mm, is_one_empty_case)
    """
    p_count = int(np.sum(pred_binary))
    g_count = int(np.sum(target_binary))

    if p_count == 0 and g_count == 0:
        return 0.0, False

    if p_count == 0 or g_count == 0:
        penalty = compute_grid_diagonal_penalty(pred_binary.shape, spacing)
        return penalty, True

    # Compute Euclidean distance transforms in physical mm
    # Distance from background to closest target surface point
    dt_target = distance_transform_edt(~target_binary, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_binary, sampling=spacing)

    # Distances from predicted foreground points to target boundary
    dist_p_to_g = dt_target[pred_binary]
    # Distances from target foreground points to predicted boundary
    dist_g_to_p = dt_pred[target_binary]

    hd95_p = np.percentile(dist_p_to_g, 95)
    hd95_g = np.percentile(dist_g_to_p, 95)

    hd95 = float(max(hd95_p, hd95_g))
    return hd95, False


class PatientEvaluator:
    """
    Evaluates 3D patient volumes and computes region-wise Dice & HD95 metrics (§13.5).
    """

    def __init__(self, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> None:
        self.spacing = spacing

    def evaluate_patient_volume(
        self,
        pred_logits_3d: Union[torch.Tensor, np.ndarray],
        target_labels_3d: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, float]:
        """
        Evaluate a single 3D patient volume (155 x 240 x 240).

        Args:
            pred_logits_3d: 3D logits array of shape (4, D, H, W) or model prediction class mask (D, H, W).
            target_labels_3d: 3D ground truth labels of shape (D, H, W) in model {0,1,2,3} or BraTS {0,1,2,4}.

        Returns:
            Dict containing Dice and HD95 for WT, TC, ET, plus macro averages.
        """
        if isinstance(pred_logits_3d, torch.Tensor):
            pred_logits_3d = pred_logits_3d.detach().cpu().numpy()
        if isinstance(target_labels_3d, torch.Tensor):
            target_labels_3d = target_labels_3d.detach().cpu().numpy()

        # If logits provided (4, D, H, W), take argmax with lowest class index tie-breaking
        if pred_logits_3d.ndim == 4:
            model_pred_mask = np.argmax(pred_logits_3d, axis=0).astype(np.uint8)
        else:
            model_pred_mask = pred_logits_3d.astype(np.uint8)

        # Back-map predictions to BraTS label convention {0, 1, 2, 4}
        pred_brats = backmap_model_to_brats(model_pred_mask)

        # Ensure ground truth is in BraTS label convention {0, 1, 2, 4}
        target_brats = target_labels_3d.astype(np.uint8)
        if np.max(target_brats) == 3:  # Model convention was passed
            target_brats = backmap_model_to_brats(target_brats)

        metrics = {}
        one_empty_count = 0

        for region in ("WT", "TC", "ET"):
            p_mask = extract_tumor_region_mask(pred_brats, region)
            g_mask = extract_tumor_region_mask(target_brats, region)

            dice = compute_3d_dice(p_mask, g_mask)
            hd95, is_one_empty = compute_3d_hd95(p_mask, g_mask, spacing=self.spacing)

            metrics[f"dice_{region}"] = dice
            metrics[f"hd95_{region}"] = hd95

            if is_one_empty:
                one_empty_count += 1

        # Calculate macro averages over WT, TC, ET
        metrics["macro_dice"] = float(np.mean([metrics["dice_WT"], metrics["dice_TC"], metrics["dice_ET"]]))
        metrics["macro_hd95"] = float(np.mean([metrics["hd95_WT"], metrics["hd95_TC"], metrics["hd95_ET"]]))
        metrics["one_empty_case_count"] = float(one_empty_count)

        return metrics
