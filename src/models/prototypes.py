"""
CAMFS M1 — Prototype Bank & Alignment Module
=============================================

Implements §5.2 and §6.2 of the CAMFS M1 Specification.
Manages unimodal class prototypes (Proto_m^c ∈ ℝ²⁵⁶) and fused class prototypes
(FusedProto_S^c ∈ ℝ²⁵⁶) for classes c ∈ {0, 1, 2, 3}.
Handles L2 normalization, patient-support counting, zero-support masking,
and support-weighted federated prototype aggregation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Apply L2 normalization along dim=-1 or dim=1."""
    norm = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
    return x / (norm + eps)


class PrototypeBank(nn.Module):
    """
    Class prototype computation, storage, and aggregation (§5.2, §6.2).

    Parameters:
        num_classes: Number of segmentation classes (default: 4 for {0,1,2,3}).
        feature_dim: Dimensionality of bottleneck representation z (default: 256).
    """

    def __init__(self, num_classes: int = 4, feature_dim: int = 256) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

    @staticmethod
    def compute_local_prototypes(
        z: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int = 4,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute unimodal or fused class prototypes and patient-support counts for a local batch.

        Args:
            z: Bottleneck feature map of shape (B, C, H_b, W_b), e.g. (B, 256, 15, 15).
            labels: Segmentation mask of shape (B, 1, H, W) or (B, H, W), values in {0, 1, 2, 3}.
            num_classes: Total number of classes (default: 4).
            eps: Epsilon for L2 normalization.

        Returns:
            prototypes: Tensor of shape (num_classes, feature_dim), L2-normalized class prototypes.
                        Zero vector if class has 0 support.
            patient_counts: Tensor of shape (num_classes,) containing count of patients in batch
                            that have at least 1 pixel of class c.
        """
        if labels.ndim == 4:
            labels = labels.squeeze(1)  # (B, H, W)

        B, C, H_b, W_b = z.shape

        # Downsample labels to bottleneck spatial resolution (H_b, W_b) via nearest neighbor
        if (labels.shape[1], labels.shape[2]) != (H_b, W_b):
            labels_down = F.interpolate(
                labels.unsqueeze(1).float(),
                size=(H_b, W_b),
                mode="nearest",
            ).squeeze(1).long()
        else:
            labels_down = labels.long()

        # Reshape z to (B, H_b*W_b, C) and labels_down to (B, H_b*W_b)
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H_b * W_b, C)
        labels_flat = labels_down.reshape(B, H_b * W_b)

        prototypes = torch.zeros(num_classes, C, device=z.device, dtype=z.dtype)
        patient_counts = torch.zeros(num_classes, device=z.device, dtype=torch.long)

        for c in range(num_classes):
            mask_c = (labels_flat == c)  # (B, H_b*W_b)

            # Patient support: count patients that have at least 1 pixel of class c
            patient_has_c = mask_c.any(dim=1)  # (B,)
            patient_counts[c] = patient_has_c.sum()

            if mask_c.any():
                # Average bottleneck features where label == c
                # Extract all spatial tokens belonging to class c across all patients in batch
                features_c = z_flat[mask_c]  # (N_tokens, C)
                proto_raw = features_c.mean(dim=0)  # (C,)
                prototypes[c] = normalize_l2(proto_raw, eps=eps)

        return prototypes, patient_counts

    @staticmethod
    def aggregate_prototypes(
        client_prototypes: List[torch.Tensor],
        client_support_counts: List[torch.Tensor],
        previous_prototypes: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Aggregate local client prototypes on server using patient-support weighting (§6.2).

        Formula:
            Proto^c = L2Norm( (sum_i N_{i,c} * Proto_{i}^c) / (sum_i N_{i,c}) )
            If sum_i N_{i,c} == 0, retain previous_prototypes[c].

        Args:
            client_prototypes: List of client prototype tensors, each of shape (num_classes, C).
            client_support_counts: List of patient-support count tensors, each of shape (num_classes,).
            previous_prototypes: Optional previous round prototypes tensor of shape (num_classes, C).
            eps: Epsilon for L2 normalization.

        Returns:
            Aggregated prototypes tensor of shape (num_classes, C).
        """
        num_clients = len(client_prototypes)
        if num_clients == 0:
            if previous_prototypes is not None:
                return previous_prototypes
            raise ValueError("No client prototypes provided and no previous prototypes available.")

        num_classes, C = client_prototypes[0].shape
        device = client_prototypes[0].device
        dtype = client_prototypes[0].dtype

        aggregated = torch.zeros(num_classes, C, device=device, dtype=dtype)

        for c in range(num_classes):
            total_support = 0
            weighted_sum = torch.zeros(C, device=device, dtype=dtype)

            for i in range(num_clients):
                count_ic = client_support_counts[i][c].item()
                if count_ic > 0:
                    total_support += count_ic
                    weighted_sum += count_ic * client_prototypes[i][c]

            if total_support > 0:
                proto_raw = weighted_sum / float(total_support)
                aggregated[c] = normalize_l2(proto_raw, eps=eps)
            else:
                # Retain previous round prototype if no client observed class c in this round
                if previous_prototypes is not None:
                    aggregated[c] = previous_prototypes[c]
                else:
                    aggregated[c] = torch.zeros(C, device=device, dtype=dtype)

        return aggregated
