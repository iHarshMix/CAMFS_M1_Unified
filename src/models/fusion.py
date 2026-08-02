"""
CAMFS M1 — Subset Fusion Head Module
====================================

Implements §5.4 of the CAMFS M1 Specification.
Fuses modality-specific encoder features for a declared modality track subset S.
Features are concatenated in fixed universe order (T1, T1ce, T2, FLAIR).
A 1x1 convolution compresses |S| * C_r -> C_r channels at each of the 5 resolution
levels, followed by a ConvBlock.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import torch
import torch.nn as nn
from src.models.encoder import ConvBlock

# Fixed canonical universe order of modalities (§5.4)
CANONICAL_MODALITY_ORDER = ("T1", "T1ce", "T2", "FLAIR")


class SubsetFusionHead(nn.Module):
    """
    Track-isolated fusion head for a modality subset S (§5.4).

    Parameters:
        modality_subset: Tuple/List of modality names present in track S, e.g., ("T1", "T2").
        channel_list: Channel dimensions at the 5 resolution levels [32, 64, 128, 256, 256].
        num_groups: GroupNorm groups (default: 8).
    """

    def __init__(
        self,
        modality_subset: Sequence[str],
        channel_list: Tuple[int, ...] = (32, 64, 128, 256, 256),
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        # Order present modalities according to CANONICAL_MODALITY_ORDER
        self.modality_subset = [m for m in CANONICAL_MODALITY_ORDER if m in modality_subset]
        if not self.modality_subset:
            raise ValueError(f"Modality subset {modality_subset} contains no valid modalities.")

        self.num_modalities = len(self.modality_subset)
        self.channel_list = channel_list
        self.num_groups = num_groups

        # 5 fusion blocks for the 5 resolution levels
        self.fusion_blocks = nn.ModuleList()

        for level, c_r in enumerate(channel_list):
            in_channels = self.num_modalities * c_r
            # 1x1 compression followed by ConvBlock (Double 3x3 with GroupNorm + SiLU)
            level_block = nn.Sequential(
                nn.Conv2d(in_channels, c_r, kernel_size=1, bias=True),
                ConvBlock(c_r, c_r, num_groups=num_groups),
            )
            self.fusion_blocks.append(level_block)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Kaiming Normal weight initialization with mode='fan_out', nonlinearity='relu', zero bias (§15.1).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        modality_features: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for fusion.

        Args:
            modality_features: Dict mapping modality name (e.g. 'T1') to tuple of 5 feature maps
                               (h1, h2, h3, h4, z) from that modality's UnimodalEncoder.

        Returns:
            Tuple of 5 fused feature maps:
                f_S_1: Level 1 fused features (B, 32, 240, 240)
                f_S_2: Level 2 fused features (B, 64, 120, 120)
                f_S_3: Level 3 fused features (B, 128, 60, 60)
                f_S_4: Level 4 fused features (B, 256, 30, 30)
                z_S:   Level 5 fused bottleneck (B, 256, 15, 15)
        """
        fused_outputs = []

        # Iterate over the 5 resolution levels
        for level_idx in range(5):
            # Collect feature map at level_idx for each modality in canonical subset order
            level_feats = []
            for m in self.modality_subset:
                if m not in modality_features:
                    raise KeyError(f"Required modality '{m}' for track S is missing from modality_features dict.")
                level_feats.append(modality_features[m][level_idx])

            # Concatenate along channel dimension (dim=1)
            concat_feats = torch.cat(level_feats, dim=1)

            # Pass through 1x1 compression + ConvBlock
            fused = self.fusion_blocks[level_idx](concat_feats)
            fused_outputs.append(fused)

        return (
            fused_outputs[0],
            fused_outputs[1],
            fused_outputs[2],
            fused_outputs[3],
            fused_outputs[4],
        )
