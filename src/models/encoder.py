"""
CAMFS M1 — Unimodal Encoder Network Architecture
=================================================

Implements §5.1 of the CAMFS M1 Specification.
Each modality (T1, T1ce, T2, FLAIR) uses an independent 2D UNet encoder
instance with 5 resolution levels [32, 64, 128, 256, 256].
"""

from __future__ import annotations

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Double 3x3 Convolution block with GroupNorm (8 groups) and SiLU activation.
    Preserves spatial dimensions via padding=1.
    """

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UnimodalEncoder(nn.Module):
    """
    2D UNet Encoder for a single MRI modality (§5.1).

    Channel progression: 1 -> 32 -> 64 -> 128 -> 256 -> 256
    Resolution progression: 240x240 -> 120x120 -> 60x60 -> 30x30 -> 15x15
    """

    def __init__(
        self,
        in_channels: int = 1,
        channel_list: Tuple[int, ...] = (32, 64, 128, 256, 256),
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channel_list = channel_list
        self.num_groups = num_groups

        # Level 1: 1 -> 32 (240x240)
        self.layer1 = ConvBlock(in_channels, channel_list[0], num_groups=num_groups)

        # Level 2: 32 -> 64 (120x120)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = ConvBlock(channel_list[0], channel_list[1], num_groups=num_groups)

        # Level 3: 64 -> 128 (60x60)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = ConvBlock(channel_list[1], channel_list[2], num_groups=num_groups)

        # Level 4: 128 -> 256 (30x30)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = ConvBlock(channel_list[2], channel_list[3], num_groups=num_groups)

        # Level 5 (Bottleneck): 256 -> 256 (15x15)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = ConvBlock(channel_list[3], channel_list[4], num_groups=num_groups)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Kaiming Normal weight initialization with mode='fan_out', nonlinearity='relu', zero bias (§15.1, §15.4).
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
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input 2D slice tensor of shape (B, 1, H, W), e.g., (B, 1, 240, 240).

        Returns:
            Tuple of (h1, h2, h3, h4, z):
                h1: Level 1 skip features of shape (B, 32, 240, 240)
                h2: Level 2 skip features of shape (B, 64, 120, 120)
                h3: Level 3 skip features of shape (B, 128, 60, 60)
                h4: Level 4 skip features of shape (B, 256, 30, 30)
                z:  Bottleneck representation of shape (B, 256, 15, 15)
        """
        h1 = self.layer1(x)
        h2 = self.layer2(self.pool1(h1))
        h3 = self.layer3(self.pool2(h2))
        h4 = self.layer4(self.pool3(h3))
        z = self.layer5(self.pool4(h4))
        return h1, h2, h3, h4, z
