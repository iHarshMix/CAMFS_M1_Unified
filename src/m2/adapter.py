"""
CAMFS M2 — CDRD Residual Adapter Architecture
==============================================

Implements §10.3 of the CAMFS M2 Specification:
- 5 Lateral Projections: Conv1x1(C_r -> 16, bias=False) + GN(8, 16) + SiLU
- Bilinear upsample all projections to 240x240 (align_corners=False)
- Concatenate with base logits (4 channels) -> 84 channels
- Residual Head:
    1. Depthwise-separable 3x3, 84 -> 32, GN(8, 32), SiLU
    2. Depthwise-separable 3x3, 32 -> 16, GN(8, 16), SiLU
    3. Conv1x1, 16 -> 4, bias=True (zero-initialized kernel and bias)
- Centering: Delta_ell = r_phi - (1/4) * sum_c(r_phi_c)
- Exactly 16,344 trainable parameters (~65.4 KB in FP32).
"""

from __future__ import annotations

from typing import Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableBlock(nn.Module):
    """
    Depthwise-separable 3x3 convolution block (§10.3):
    - Depthwise 3x3 (groups=in_channels, bias=False, padding=1)
    - Pointwise 1x1 (in_channels -> out_channels, bias=False)
    - GroupNorm (8 groups, affine=True)
    - SiLU activation
    """

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, affine=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.gn(x)
        return self.act(x)


class CDRDAdapter(nn.Module):
    """
    CDRD Detached Residual Adapter Sidecar (§10.3).
    """

    def __init__(
        self,
        channel_list: Tuple[int, ...] = (32, 64, 128, 256, 256),
        proj_channels: int = 16,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.channel_list = channel_list
        self.proj_channels = proj_channels
        self.num_classes = num_classes

        # 5 Lateral projections: Conv1x1(C_r -> 16, bias=False) + GN(8, 16) + SiLU
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c_r, proj_channels, kernel_size=1, bias=False)
            for c_r in channel_list
        ])
        self.lateral_gns = nn.ModuleList([
            nn.GroupNorm(num_groups=8, num_channels=proj_channels, affine=True)
            for _ in channel_list
        ])
        self.lateral_act = nn.SiLU(inplace=True)

        # Concatenated feature map channels: 5 * 16 + 4 = 84
        in_head_channels = len(channel_list) * proj_channels + num_classes

        # Residual head
        self.head_block1 = DepthwiseSeparableBlock(in_head_channels, 32, num_groups=8)
        self.head_block2 = DepthwiseSeparableBlock(32, 16, num_groups=8)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes weights according to §10.3:
        - Lateral convs & pointwise convs: Kaiming normal
        - Final 1x1 conv kernel & bias: ZERO-INITIALIZED for Day-0 M1 identity!
        """
        for m in self.lateral_convs:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for gn in self.lateral_gns:
            nn.init.ones_(gn.weight)
            nn.init.zeros_(gn.bias)

        # Final 1x1 zero-initialization (§10.3)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        logits_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            features: Tuple of 5 fused tensors (f1, f2, f3, f4, z_S) from FrozenBase.
            logits_B: Canonical base logits [B, 4, 240, 240].
        Returns:
            delta_ell: Class-zero-mean residual logit map [B, 4, 240, 240].
            logits_cand: Candidate augmented logits (ell_B + Delta_ell).
        """
        target_size = (logits_B.shape[2], logits_B.shape[3])  # (240, 240)

        proj_upsampled = []
        for i, (feat, conv, gn) in enumerate(zip(features, self.lateral_convs, self.lateral_gns)):
            p = self.lateral_act(gn(conv(feat)))
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            proj_upsampled.append(p)

        # Concat all 5 upsampled lateral features with base logits -> [B, 84, 240, 240]
        q = torch.cat(proj_upsampled + [logits_B], dim=1)

        # Residual head
        x = self.head_block1(q)
        x = self.head_block2(x)
        r_phi = self.final_conv(x)  # [B, 4, 240, 240]

        # Centering (§10.3): Delta_ell = r_phi - (1/C) * sum_c(r_phi_c)
        r_mean = torch.mean(r_phi, dim=1, keepdim=True)
        delta_ell = r_phi - r_mean

        logits_cand = logits_B + delta_ell
        return delta_ell, logits_cand

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
