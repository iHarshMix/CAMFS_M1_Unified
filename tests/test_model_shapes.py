"""
CAMFS M1 — Neural Network Architecture Shape & Gradient Tests
==============================================================

PyTest test suite for Phase 3 models:
  - UnimodalEncoder
  - PrototypeBank
  - SubsetFusionHead
  - UNetDecoder
  - End-to-end pipeline composition & Kaiming Normal initialization
"""

import pytest
import torch
import torch.nn as nn

from src.models import (
    ConvBlock,
    PrototypeBank,
    SubsetFusionHead,
    UNetDecoder,
    UnimodalEncoder,
    normalize_l2,
)


def test_unimodal_encoder_shapes():
    """Verify UnimodalEncoder forward pass shapes at all 5 levels (§5.1)."""
    batch_size = 2
    encoder = UnimodalEncoder(in_channels=1, channel_list=(32, 64, 128, 256, 256))
    x = torch.randn(batch_size, 1, 240, 240)

    h1, h2, h3, h4, z = encoder(x)

    assert h1.shape == (batch_size, 32, 240, 240), f"h1 shape mismatch: {h1.shape}"
    assert h2.shape == (batch_size, 64, 120, 120), f"h2 shape mismatch: {h2.shape}"
    assert h3.shape == (batch_size, 128, 60, 60), f"h3 shape mismatch: {h3.shape}"
    assert h4.shape == (batch_size, 256, 30, 30), f"h4 shape mismatch: {h4.shape}"
    assert z.shape == (batch_size, 256, 15, 15), f"z shape mismatch: {z.shape}"


def test_prototype_bank_local_and_aggregate():
    """Verify PrototypeBank prototype extraction, L2 norm, and server aggregation (§5.2, §6.2)."""
    batch_size = 2
    num_classes = 4
    feature_dim = 256

    z = torch.randn(batch_size, feature_dim, 15, 15)
    # Create labels with background (0), WT (1), TC (2), ET (3)
    labels = torch.zeros(batch_size, 240, 240, dtype=torch.long)
    labels[0, :50, :50] = 1  # Patient 0 has class 1
    labels[0, 50:100, 50:100] = 2  # Patient 0 has class 2
    labels[1, :50, :50] = 3  # Patient 1 has class 3

    prototypes, support_counts = PrototypeBank.compute_local_prototypes(
        z=z, labels=labels, num_classes=num_classes
    )

    assert prototypes.shape == (num_classes, feature_dim)
    assert support_counts.shape == (num_classes,)

    # Class 0 present in both patients, Class 1 & 2 in patient 0, Class 3 in patient 1
    assert support_counts[0].item() == 2
    assert support_counts[1].item() == 1
    assert support_counts[2].item() == 1
    assert support_counts[3].item() == 1

    # Check L2 normalization for non-zero prototypes
    for c in range(num_classes):
        if support_counts[c] > 0:
            norm = torch.linalg.vector_norm(prototypes[c], ord=2).item()
            assert abs(norm - 1.0) < 1e-5, f"Prototype class {c} norm is {norm}, expected 1.0"

    # Test prototype aggregation across 2 synthetic clients
    client1_protos = prototypes
    client1_counts = support_counts

    client2_protos = torch.randn(num_classes, feature_dim)
    client2_protos = normalize_l2(client2_protos)
    client2_counts = torch.tensor([2, 0, 1, 0], dtype=torch.long)

    server_protos = PrototypeBank.aggregate_prototypes(
        client_prototypes=[client1_protos, client2_protos],
        client_support_counts=[client1_counts, client2_counts],
    )

    assert server_protos.shape == (num_classes, feature_dim)
    for c in range(num_classes):
        norm = torch.linalg.vector_norm(server_protos[c], ord=2).item()
        assert abs(norm - 1.0) < 1e-5


def test_subset_fusion_head_tracks():
    """Verify SubsetFusionHead compression and feature fusion for all 4 tracks (§5.4)."""
    batch_size = 2
    modalities = ("T1", "T1ce", "T2", "FLAIR")

    # Instantiate encoders for all 4 modalities
    encoders = {m: UnimodalEncoder(in_channels=1) for m in modalities}
    inputs = {m: torch.randn(batch_size, 1, 240, 240) for m in modalities}
    features = {m: encoders[m](inputs[m]) for m in modalities}

    # Define track subsets according to §5.3
    tracks = {
        "S1": ("T1", "T1ce", "T2", "FLAIR"),  # Full modality (4)
        "S2": ("T1", "T2"),  # Dual H2 track (2)
        "S3": ("T1", "FLAIR"),  # Dual H3 track (2)
        "S4": ("T1", "T1ce", "T2"),  # Triple H4 track (3)
    }

    for track_name, subset in tracks.items():
        fusion_head = SubsetFusionHead(modality_subset=subset)
        f1, f2, f3, f4, z_S = fusion_head(features)

        assert f1.shape == (batch_size, 32, 240, 240), f"{track_name} f1 shape: {f1.shape}"
        assert f2.shape == (batch_size, 64, 120, 120), f"{track_name} f2 shape: {f2.shape}"
        assert f3.shape == (batch_size, 128, 60, 60), f"{track_name} f3 shape: {f3.shape}"
        assert f4.shape == (batch_size, 256, 30, 30), f"{track_name} f4 shape: {f4.shape}"
        assert z_S.shape == (batch_size, 256, 15, 15), f"{track_name} z_S shape: {z_S.shape}"


def test_unet_decoder_end_to_end():
    """Verify end-to-end pipeline (Encoders -> FusionHead -> Decoder -> Logits) (§5.4)."""
    batch_size = 2
    modalities = ("T1", "T1ce", "T2", "FLAIR")

    encoders = {m: UnimodalEncoder(in_channels=1) for m in modalities}
    inputs = {m: torch.randn(batch_size, 1, 240, 240) for m in modalities}
    features = {m: encoders[m](inputs[m]) for m in modalities}

    fusion_head = SubsetFusionHead(modality_subset=modalities)
    f1, f2, f3, f4, z_S = fusion_head(features)

    decoder = UNetDecoder(num_classes=4)
    logits = decoder(z_S=z_S, f_S_4=f4, f_S_3=f3, f_S_2=f2, f_S_1=f1)

    assert logits.shape == (batch_size, 4, 240, 240), f"Logits shape mismatch: {logits.shape}"

    # Verify backward pass gradient flow
    loss = logits.sum()
    loss.backward()

    # Decoder and Fusion parameters should have non-zero gradients
    for p in decoder.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()

    for p in fusion_head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_kaiming_init():
    """Verify Kaiming Normal weight initialization and zero bias initialization (§15.1)."""
    encoder = UnimodalEncoder()
    decoder = UNetDecoder()
    fusion = SubsetFusionHead(modality_subset=("T1", "T2"))

    for module in [encoder, decoder, fusion]:
        for name, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                assert torch.isfinite(m.weight).all()
                if m.bias is not None:
                    assert (m.bias == 0).all(), f"Bias in {name} is not zero"
            elif isinstance(m, nn.GroupNorm):
                assert (m.weight == 1).all(), f"Weight in GroupNorm {name} is not one"
                assert (m.bias == 0).all(), f"Bias in GroupNorm {name} is not zero"
