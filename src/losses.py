"""
CAMFS M1 — Loss Functions Module
================================

Implements §6.1, §8.2, and §15.1 of the CAMFS M1 Specification.
Includes:
  - MaskedInfoNCELoss: Unimodal & fused contrastive prototype loss with empty-prototype masking (§6.1).
  - SoftDiceCrossEntropyLoss: Soft Dice + Cross-Entropy composite loss matching exact §15.1 formula.
  - Phase1Loss: Phase 1 contrastive loss objective (λ1 = 1.0).
  - Phase2Loss: Phase 2 joint segmentation + fused prototype alignment objective (λ2 = 0.1).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.prototypes import normalize_l2


class MaskedInfoNCELoss(nn.Module):
    """
    Masked InfoNCE contrastive alignment loss with empty-prototype masking (§6.1).

    Formula:
        L_InfoNCE = - 1/|Ω| sum_{q ∈ Ω} log [ exp(sim(z(q), Proto^{y(q)}) / τ) / sum_{c ∈ C_active} exp(sim(z(q), Proto^c) / τ) ]

    Parameters:
        tau: Temperature hyperparameter (default: 0.1).
        eps: Small floor to prevent division by zero or log(0).
    """

    def __init__(self, tau: float = 0.1, eps: float = 1e-8) -> None:
        super().__init__()
        self.tau = tau
        self.eps = eps

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for InfoNCE loss.

        Args:
            z: Bottleneck feature map of shape (B, C, H_b, W_b), e.g. (B, 256, 15, 15).
            labels: Spatial labels of shape (B, 1, H, W) or (B, H, W) or (B, H_b, W_b).
            prototypes: Class prototypes of shape (num_classes, C), L2-normalized.

        Returns:
            Scalar InfoNCE loss tensor.
        """
        if labels.ndim == 4:
            labels = labels.squeeze(1)

        B, C, H_b, W_b = z.shape
        num_classes = prototypes.shape[0]

        # Downsample labels to (H_b, W_b) if needed
        if (labels.shape[1], labels.shape[2]) != (H_b, W_b):
            labels_down = F.interpolate(
                labels.unsqueeze(1).float(),
                size=(H_b, W_b),
                mode="nearest",
            ).squeeze(1).long()
        else:
            labels_down = labels.long()

        # Reshape z to (B * H_b * W_b, C) and labels to (B * H_b * W_b,)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = labels_down.reshape(-1)

        # L2-normalize bottleneck tokens z_flat along channel dimension
        z_norm = normalize_l2(z_flat, eps=self.eps)  # (N_tokens, C)
        proto_norm = normalize_l2(prototypes, eps=self.eps)  # (num_classes, C)

        # Compute cosine similarity matrix: (N_tokens, num_classes)
        sim_matrix = torch.matmul(z_norm, proto_norm.T) / self.tau  # (N_tokens, num_classes)

        # Identify active classes (classes with non-zero prototype vectors)
        proto_norms = torch.linalg.vector_norm(prototypes, ord=2, dim=1)
        active_classes = (proto_norms > self.eps)  # (num_classes,)

        if not active_classes.any():
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        # Mask out inactive/empty classes from denominator by setting sim to -inf
        mask_sim = sim_matrix.clone()
        mask_sim[:, ~active_classes] = -1e9

        # Log-softmax over active classes
        log_prob = F.log_softmax(mask_sim, dim=1)  # (N_tokens, num_classes)

        # Gather log-probability corresponding to target class y(q)
        # Filter tokens whose target class is active
        target_is_active = active_classes[labels_flat]

        if not target_is_active.any():
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        target_log_probs = log_prob[target_is_active, labels_flat[target_is_active]]
        loss = -target_log_probs.mean()

        return loss


class SoftDiceCrossEntropyLoss(nn.Module):
    """
    Soft Dice + Cross-Entropy composite segmentation loss (§15.1).

    Formula (§15.1):
        L_CE = - 1 / (N * |Ω|) sum_{n=1}^N sum_{q ∈ Ω} log p_{n, y_n(q)}(q)
        L_Dice = 1 - 1 / (3 * N) sum_{n=1}^N sum_{c ∈ C_T} [ (2 * sum p_{n,c} y_{n,c} + ε_D) / (sum p_{n,c} + sum y_{n,c} + ε_D) ]
        L_Dice+CE = L_CE + L_Dice

    Parameters:
        eps_d: Soft Dice smoothing constant (default: 1e-5).
        tumour_classes: Class indices for tumor regions after remapping (default: (1, 2, 3)).
    """

    def __init__(self, eps_d: float = 1e-5, tumour_classes: Sequence[int] = (1, 2, 3)) -> None:
        super().__init__()
        self.eps_d = eps_d
        self.tumour_classes = tumour_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for composite Soft Dice + CE loss.

        Args:
            logits: Output logits of shape (B, num_classes, H, W).
            targets: Target labels of shape (B, 1, H, W) or (B, H, W) in {0, 1, 2, 3}.

        Returns:
            Scalar loss tensor.
        """
        if targets.ndim == 4:
            targets = targets.squeeze(1)

        targets = targets.long()
        B, num_classes, H, W = logits.shape

        # Downsample/upsample targets if spatial resolution differs from logits (H, W)
        if (targets.shape[1], targets.shape[2]) != (H, W):
            targets = F.interpolate(
                targets.unsqueeze(1).float(),
                size=(H, W),
                mode="nearest",
            ).squeeze(1).long()

        # 1. Cross-Entropy Loss
        ce_loss = F.cross_entropy(logits, targets, reduction="mean")

        # 2. Soft Dice Loss over tumour classes {1, 2, 3} per batch sample
        probs = F.softmax(logits, dim=1)  # (B, num_classes, H, W)

        dice_scores_per_sample = []

        for b in range(B):
            sample_dice_sum = 0.0
            for c in self.tumour_classes:
                p_c = probs[b, c]  # (H, W)
                y_c = (targets[b] == c).float()  # (H, W)

                intersection = (p_c * y_c).sum()
                cardinality = p_c.sum() + y_c.sum()

                dice_c = (2.0 * intersection + self.eps_d) / (cardinality + self.eps_d)
                sample_dice_sum += dice_c

            # Average over the 3 tumor classes
            sample_mean_dice = sample_dice_sum / float(len(self.tumour_classes))
            dice_scores_per_sample.append(sample_mean_dice)

        mean_dice_over_batch = torch.stack(dice_scores_per_sample).mean()
        dice_loss = 1.0 - mean_dice_over_batch

        total_loss = ce_loss + dice_loss
        return total_loss


class Phase1Loss(nn.Module):
    """
    Phase 1 training objective (§6.1): Sum of unimodal InfoNCE contrastive losses.

    L^(1) = λ1 * sum_{m ∈ O(i)} L_InfoNCE(z_m, y, Proto_m)
    """

    def __init__(self, lambda1: float = 1.0, tau: float = 0.1) -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.infonce = MaskedInfoNCELoss(tau=tau)

    def forward(
        self,
        bottlenecks: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        prototypes: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            bottlenecks: Dict mapping modality 'T1' -> z_m tensor (B, 256, 15, 15).
            labels: Segmentation mask tensor (B, 240, 240).
            prototypes: Dict mapping modality 'T1' -> Proto_m tensor (4, 256).
        """
        total_loss = torch.tensor(0.0, device=labels.device)

        for m, z_m in bottlenecks.items():
            if m in prototypes:
                proto_m = prototypes[m]
                loss_m = self.infonce(z_m, labels, proto_m)
                total_loss = total_loss + loss_m

        return self.lambda1 * total_loss


class Phase2Loss(nn.Module):
    """
    Phase 2 training objective (§8.2): Dice+CE segmentation loss + λ2 * fused-alignment.

    L^(2) = L_Dice+CE(logits_S, y) + λ2 * L_InfoNCE(z_S, y, FusedProto_S)
    """

    def __init__(self, lambda2: float = 0.1, tau: float = 0.1, eps_d: float = 1e-5) -> None:
        super().__init__()
        self.lambda2 = lambda2
        self.seg_loss = SoftDiceCrossEntropyLoss(eps_d=eps_d)
        self.fused_align_loss = MaskedInfoNCELoss(tau=tau)

    def forward(
        self,
        logits_S: torch.Tensor,
        targets: torch.Tensor,
        z_S: Optional[torch.Tensor] = None,
        fused_prototypes_S: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits_S: Output segmentation logits (B, 4, 240, 240).
            targets: Target labels (B, 240, 240).
            z_S: Optional fused bottleneck tensor (B, 256, 15, 15).
            fused_prototypes_S: Optional fused prototypes tensor (4, 256).
        """
        l_seg = self.seg_loss(logits_S, targets)

        if z_S is not None and fused_prototypes_S is not None:
            l_align = self.fused_align_loss(z_S, targets, fused_prototypes_S)
            return l_seg + self.lambda2 * l_align

        return l_seg
