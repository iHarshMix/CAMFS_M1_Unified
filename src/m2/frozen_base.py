"""
CAMFS M2 — Frozen Base Module & Stop-Gradient Interface
=======================================================

Implements §9.3 of the CAMFS M2 Specification:
Wraps the frozen M1 Track S3 model (encoders for T1 & FLAIR, fusion head, decoder).
Guarantees:
- eval() mode and requires_grad=False on all base parameters.
- GroupNorm statistics are static (no running buffers).
- Explicit stop-gradient (.detach()) on all exposed intermediate fused features
  and base logits before delivery to adapter sidecars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from src.models.encoder import UnimodalEncoder
from src.models.fusion import SubsetFusionHead
from src.models.decoder import UNetDecoder


class FrozenBase(nn.Module):
    """
    Read-only wrapper around M1 Track S3 components.

    Inputs:
        x_dict: Dict containing {'T1': Tensor, 'FLAIR': Tensor}, shape [B, 1, 240, 240].
    
    Outputs:
        features: Tuple of 5 detached fused feature tensors (f_1, f_2, f_3, f_4, z_S)
                  with channels (32, 64, 128, 256, 256) and spatial shapes
                  (240, 120, 60, 30, 15).
        logits_B: Detached canonical base logits [B, 4, 240, 240].
    """

    def __init__(
        self,
        encoder_t1: UnimodalEncoder,
        encoder_flair: UnimodalEncoder,
        fusion_head: SubsetFusionHead,
        decoder: UNetDecoder,
    ) -> None:
        super().__init__()
        self.encoder_t1 = encoder_t1
        self.encoder_flair = encoder_flair
        self.fusion_head = fusion_head
        self.decoder = decoder

        # Strict freezing of all parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def train(self, mode: bool = True) -> "FrozenBase":
        # Force eval mode at all times to prevent accidental mode switches
        super().train(False)
        return self

    @torch.no_grad()
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass through frozen Track S3.

        Returns:
            features: (f1, f2, f3, f4, z_S) all detached.
            logits_B: [B, 4, 240, 240] detached canonical base logits.
        """
        if "T1" not in x_dict or "FLAIR" not in x_dict:
            raise KeyError(f"Expected modalities 'T1' and 'FLAIR', got {list(x_dict.keys())}")

        feat_t1 = self.encoder_t1(x_dict["T1"])
        feat_flair = self.encoder_flair(x_dict["FLAIR"])

        mod_features = {
            "T1": feat_t1,
            "FLAIR": feat_flair,
        }

        # Fused skip connections and bottleneck: (f1, f2, f3, f4, z_S)
        f1, f2, f3, f4, z_s = self.fusion_head(mod_features)

        # Decoder pass
        logits_b = self.decoder(z_s, f4, f3, f2, f1)

        # Enforce stop-gradient on all returns
        features = (
            f1.detach(),
            f2.detach(),
            f3.detach(),
            f4.detach(),
            z_s.detach(),
        )
        return features, logits_b.detach()


def _extract_component_state_dict(sd: Any, key: str, alt_prefix: str) -> Dict[str, torch.Tensor]:
    """Helper to extract state dict whether stored as sub-dict or with prefixed keys."""
    if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
        return sd[key]
    out = {}
    for k, v in sd.items():
        if k.startswith(alt_prefix):
            out[k[len(alt_prefix):]] = v
    return out


def load_frozen_base(
    phase1_checkpoint_path: Union[str, Path],
    track_s3_checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> FrozenBase:
    """
    Instantiates and loads weights for the M1 Track S3 FrozenBase.
    """
    device = torch.device(device)

    # 1. Instantiate encoders
    encoder_t1 = UnimodalEncoder().to(device)
    encoder_flair = UnimodalEncoder().to(device)

    # Load Phase 1 checkpoint
    p1_ckpt = torch.load(phase1_checkpoint_path, map_location=device)
    p1_sd = p1_ckpt.get("state_dict", p1_ckpt)

    t1_sd = _extract_component_state_dict(p1_sd, "T1", "T1.")
    flair_sd = _extract_component_state_dict(p1_sd, "FLAIR", "FLAIR.")

    encoder_t1.load_state_dict(t1_sd)
    encoder_flair.load_state_dict(flair_sd)

    # 2. Instantiate Fusion Head for S3: ('T1', 'FLAIR')
    fusion_head = SubsetFusionHead(modality_subset=("T1", "FLAIR")).to(device)

    # 3. Instantiate UNetDecoder
    decoder = UNetDecoder(num_classes=4).to(device)

    # Load Track S3 checkpoint
    s3_ckpt = torch.load(track_s3_checkpoint_path, map_location=device)
    s3_sd = s3_ckpt.get("state_dict", s3_ckpt)

    fusion_sd = _extract_component_state_dict(s3_sd, "fusion", "fusion_head.")
    decoder_sd = _extract_component_state_dict(s3_sd, "decoder", "decoder.")

    fusion_head.load_state_dict(fusion_sd)
    decoder.load_state_dict(decoder_sd)

    base = FrozenBase(
        encoder_t1=encoder_t1,
        encoder_flair=encoder_flair,
        fusion_head=fusion_head,
        decoder=decoder,
    ).to(device)

    return base
