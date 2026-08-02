"""
CAMFS M1 — UNet Decoder Architecture
=====================================

Implements §5.4 of the CAMFS M1 Specification.
Decodes fused bottleneck z_S and fused skip feature maps (f_S^1, f_S^2, f_S^3, f_S^4)
into 4-class segmentation logits via 2x bilinear upsampling and ConvBlocks.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoder import ConvBlock


class UNetDecoder(nn.Module):
    """
    UNet Decoder with skip connection fusion (§5.4).

    Input:
        z_S: Fused bottleneck tensor (B, 256, 15, 15)
        f_S_4: Fused skip level 4 tensor (B, 256, 30, 30)
        f_S_3: Fused skip level 3 tensor (B, 128, 60, 60)
        f_S_2: Fused skip level 2 tensor (B, 64, 120, 120)
        f_S_1: Fused skip level 1 tensor (B, 32, 240, 240)

    Output:
        Logits tensor of shape (B, num_classes, 240, 240), where num_classes=4.
    """

    def __init__(self, num_classes: int = 4, num_groups: int = 8) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_groups = num_groups

        # Level 4 Decoder: (256 upsampled + 256 skip) = 512 -> 128
        self.conv_up4 = ConvBlock(512, 128, num_groups=num_groups)

        # Level 3 Decoder: (128 upsampled + 128 skip) = 256 -> 64
        self.conv_up3 = ConvBlock(256, 64, num_groups=num_groups)

        # Level 2 Decoder: (64 upsampled + 64 skip) = 128 -> 32
        self.conv_up2 = ConvBlock(128, 32, num_groups=num_groups)

        # Level 1 Decoder: (32 upsampled + 32 skip) = 64 -> 32
        self.conv_up1 = ConvBlock(64, 32, num_groups=num_groups)

        # Output head: 32 -> num_classes logits
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1, bias=True)

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
        z_S: torch.Tensor,
        f_S_4: torch.Tensor,
        f_S_3: torch.Tensor,
        f_S_2: torch.Tensor,
        f_S_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for UNet Decoder.

        Args:
            z_S: (B, 256, 15, 15)
            f_S_4: (B, 256, 30, 30)
            f_S_3: (B, 128, 60, 60)
            f_S_2: (B, 64, 120, 120)
            f_S_1: (B, 32, 240, 240)

        Returns:
            Logits tensor (B, 4, 240, 240)
        """
        # 1. Upsample bottleneck 15x15 -> 30x30 and concat with f_S_4
        up4 = F.interpolate(z_S, scale_factor=2.0, mode="bilinear", align_corners=False)
        x4 = torch.cat([up4, f_S_4], dim=1)  # (B, 512, 30, 30)
        d4 = self.conv_up4(x4)  # (B, 128, 30, 30)

        # 2. Upsample 30x30 -> 60x60 and concat with f_S_3
        up3 = F.interpolate(d4, scale_factor=2.0, mode="bilinear", align_corners=False)
        x3 = torch.cat([up3, f_S_3], dim=1)  # (B, 256, 60, 60)
        d3 = self.conv_up3(x3)  # (B, 64, 60, 60)

        # 3. Upsample 60x60 -> 120x120 and concat with f_S_2
        up2 = F.interpolate(d3, scale_factor=2.0, mode="bilinear", align_corners=False)
        x2 = torch.cat([up2, f_S_2], dim=1)  # (B, 128, 120, 120)
        d2 = self.conv_up2(x2)  # (B, 32, 120, 120)

        # 4. Upsample 120x120 -> 240x240 and concat with f_S_1
        up1 = F.interpolate(d2, scale_factor=2.0, mode="bilinear", align_corners=False)
        x1 = torch.cat([up1, f_S_1], dim=1)  # (B, 64, 240, 240)
        d1 = self.conv_up1(x1)  # (B, 32, 240, 240)

        # 5. Output 1x1 conv -> 4 logits
        logits = self.out_conv(d1)  # (B, 4, 240, 240)

        return logits
