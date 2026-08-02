"""
CAMFS M1 — Federated Client Local Training Loop
===============================================

Implements §6.2 and §8.4 of the CAMFS M1 Specification.
Manages local client training for a single hospital site:
  - Round 0 no-optimizer prototype bootstrap (§6.2)
  - Phase 1 unimodal contrastive training with fresh AdamW optimizer (3e-4 LR) each round
  - Phase 2 track-isolated fusion training with frozen encoders and fresh AdamW optimizer (1e-3 LR)
  - Post-local prototype recomputation pass in FP32 eval mode over all 155 slices
  - Generates update packets annotated with image_lineage set for governance auditing
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.augmentation import PairwiseAugmentation
from src.data.brats_dataset import BraTSDataset
from src.data.slice_sampler import SliceSampler
from src.losses import Phase1Loss, Phase2Loss
from src.models.decoder import UNetDecoder
from src.models.encoder import UnimodalEncoder
from src.models.fusion import SubsetFusionHead
from src.models.prototypes import PrototypeBank
from src.seed import seed_worker


class FederatedClient:
    """
    Federated Learning Client for a single hospital site (§6.2, §8.4).

    Parameters:
        hospital_id: Hospital identifier, e.g. 'H1'.
        owned_modalities: List of modalities owned locally, e.g. ['T1', 'T1ce', 'T2', 'FLAIR'].
        send_modalities: List of modalities permitted to send, e.g. ['T1', 'T2'].
        patient_ids: List of patient IDs assigned to this hospital.
        preprocessed_dir: Path to preprocessed patient `.npy` files.
        batch_size: Micro-batch size (default: 16).
        device: Torch device (e.g. 'cuda' or 'cpu').
        seed: Training seed for deterministic data augmentation and data loaders.
    """

    def __init__(
        self,
        hospital_id: str,
        owned_modalities: List[str],
        send_modalities: List[str],
        patient_ids: List[str],
        preprocessed_dir: Union[str, Path],
        batch_size: int = 16,
        device: Union[str, torch.device] = "cpu",
        seed: int = 17,
    ) -> None:
        self.hospital_id = hospital_id
        self.owned_modalities = list(owned_modalities)
        self.send_modalities = list(send_modalities)
        self.patient_ids = list(patient_ids)
        self.num_patients = len(self.patient_ids)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.seed = seed

        # Lazy dataset initialization
        self._dataset: Optional[BraTSDataset] = None

    def get_dataset(self) -> BraTSDataset:
        """Initialize and return local patient dataset."""
        if self._dataset is None:
            self._dataset = BraTSDataset(
                patient_ids=self.patient_ids,
                cache_root=self.preprocessed_dir,
                modalities=self.owned_modalities,
            )
        return self._dataset

    def bootstrap_round0_prototypes(
        self,
        encoders: Dict[str, UnimodalEncoder],
        max_patients: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Round 0 No-Optimizer Prototype Bootstrap (§6.2).
        Evaluates local 155 slices per patient in FP32 without augmentation.
        NO optimizer steps or weight updates occur.

        Returns:
            Tuple of (prototypes_dict, support_counts_dict) for permitted send modalities.
        """
        dataset = self.get_dataset()
        prototypes_out = {}
        support_counts_out = {}

        patient_list = self.patient_ids[:max_patients] if max_patients else self.patient_ids

        with torch.no_grad():
            for m in self.send_modalities:
                if m not in encoders:
                    continue

                encoder = encoders[m].to(self.device)
                encoder.eval()

                all_z = []
                all_labels = []

                for pid in patient_list:
                    vol_data = dataset.load_patient_volume(pid)
                    # vol_data['modalities'][m]: (155, 240, 240)
                    # vol_data['labels']: (155, 240, 240)
                    mod_vol = vol_data["modalities"][m]
                    lab_vol = vol_data["labels"]

                    # Process in slice batches of batch_size
                    num_slices = mod_vol.shape[0]
                    for start in range(0, num_slices, self.batch_size):
                        end = min(start + self.batch_size, num_slices)
                        x_batch = torch.from_numpy(mod_vol[start:end]).float().unsqueeze(1).to(self.device)
                        y_batch = torch.from_numpy(lab_vol[start:end]).long().to(self.device)

                        _, _, _, _, z_batch = encoder(x_batch)
                        all_z.append(z_batch.cpu())
                        all_labels.append(y_batch.cpu())

                if all_z:
                    cat_z = torch.cat(all_z, dim=0)
                    cat_y = torch.cat(all_labels, dim=0)

                    proto_m, counts_m = PrototypeBank.compute_local_prototypes(cat_z, cat_y)
                    prototypes_out[m] = proto_m
                    support_counts_out[m] = counts_m

        return prototypes_out, support_counts_out

    def train_phase1_round(
        self,
        global_encoders: Dict[str, UnimodalEncoder],
        global_prototypes: Dict[str, torch.Tensor],
        local_epochs: int = 1,
        lr: float = 3e-4,
    ) -> Dict:
        """
        Phase 1 Local Unimodal Contrastive Training Loop (§6.2).
        Uses fresh AdamW optimizer each round (no optimizer state persists across rounds).

        Returns:
            Dict containing updated encoder_state_dicts, prototypes, support_counts, and image_lineage.
        """
        dataset = self.get_dataset()

        # Copy global encoders for sending modalities
        local_encoders = {}
        trainable_params = []

        for m in self.send_modalities:
            if m in global_encoders:
                enc = copy.deepcopy(global_encoders[m]).to(self.device)
                enc.train()

                for p in enc.parameters():
                    p.requires_grad = True

                local_encoders[m] = enc
                trainable_params.extend(enc.parameters())

        if not trainable_params:
            raise ValueError(f"Hospital {self.hospital_id} has no trainable encoders for modalities {self.send_modalities}.")

        # Fresh local AdamW optimizer (§6.2, §15.1)
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )

        p1_loss_fn = Phase1Loss(lambda1=1.0, tau=0.1)
        augmenter = PairwiseAugmentation(modalities=self.send_modalities, is_training=True)

        # 1:1 Tumor vs. Non-Tumor Slice Sampler (§13.4)
        slice_sampler = SliceSampler(dataset, seed=self.seed)
        samples = slice_sampler.sample_slices_for_round(slices_per_patient=16)

        # Local training epochs
        for epoch in range(local_epochs):
            for pid, slice_idx in samples:
                vol = dataset.load_patient_volume(pid)
                
                # Build sample dict for pairwise augmentation (§13.4)
                sample_dict = {
                    m: torch.from_numpy(vol["modalities"][m][slice_idx]).unsqueeze(0)
                    for m in self.send_modalities
                    if m in vol["modalities"]
                }
                sample_dict["label"] = torch.from_numpy(vol["labels"][slice_idx])

                aug_sample = augmenter(sample_dict)
                y_tensor = aug_sample["label"].long().unsqueeze(0).to(self.device)  # (1, 240, 240)

                bottlenecks = {}
                for m, enc in local_encoders.items():
                    x_tensor = aug_sample[m].unsqueeze(0).float().to(self.device)  # (1, 1, 240, 240)
                    _, _, _, _, z_m = enc(x_tensor)
                    bottlenecks[m] = z_m

                # Prepare device prototypes
                device_protos = {m: global_prototypes[m].to(self.device) for m in self.send_modalities if m in global_prototypes}

                optimizer.zero_grad()
                loss = p1_loss_fn(bottlenecks, y_tensor, device_protos)
                loss.backward()
                optimizer.step()

        # Post-local prototype recomputation pass in FP32 eval mode over all 155 slices (§6.2)
        local_prototypes = {}
        local_counts = {}

        with torch.no_grad():
            for m, enc in local_encoders.items():
                enc.eval()
                all_z, all_labels = [], []

                for pid in self.patient_ids:
                    vol = dataset.load_patient_volume(pid)
                    mod_vol = vol["modalities"][m]
                    lab_vol = vol["labels"]
                    num_slices = mod_vol.shape[0]

                    for start in range(0, num_slices, self.batch_size):
                        end = min(start + self.batch_size, num_slices)
                        x_b = torch.from_numpy(mod_vol[start:end]).float().unsqueeze(1).to(self.device)
                        y_b = torch.from_numpy(lab_vol[start:end]).long().to(self.device)

                        _, _, _, _, z_b = enc(x_b)
                        all_z.append(z_b.cpu())
                        all_labels.append(y_b.cpu())

                if all_z:
                    cat_z = torch.cat(all_z, dim=0)
                    cat_y = torch.cat(all_labels, dim=0)
                    proto_m, count_m = PrototypeBank.compute_local_prototypes(cat_z, cat_y)
                    local_prototypes[m] = proto_m
                    local_counts[m] = count_m

        # Build update packet
        encoder_state_dicts = {m: enc.cpu().state_dict() for m, enc in local_encoders.items()}

        return {
            "hospital_id": self.hospital_id,
            "encoder_state_dicts": encoder_state_dicts,
            "prototypes": local_prototypes,
            "support_counts": local_counts,
            "image_lineage": set(self.send_modalities),
        }

    def train_phase2_round(
        self,
        track_id: str,
        track_modalities: Sequence[str],
        frozen_encoders: Dict[str, UnimodalEncoder],
        global_fusion_head: SubsetFusionHead,
        global_decoder: UNetDecoder,
        global_fused_prototypes: torch.Tensor,
        local_epochs: int = 1,
        lr: float = 1e-3,
    ) -> Dict:
        """
        Phase 2 Local Track-Isolated Fusion Training Loop (§8.4).
        Encoders are frozen in eval mode with stop-gradients. Uses fresh AdamW optimizer (1e-3 LR).

        Returns:
            Dict containing updated fusion_state_dict, decoder_state_dict, fused_prototypes, support_counts, and image_lineage.
        """
        dataset = self.get_dataset()
        track_modalities = list(track_modalities)

        # Copy global fusion head and decoder
        local_fusion = copy.deepcopy(global_fusion_head).to(self.device)
        local_decoder = copy.deepcopy(global_decoder).to(self.device)

        local_fusion.train()
        local_decoder.train()

        # Encoders are frozen in eval mode (§7)
        local_encoders = {}
        for m in track_modalities:
            if m in frozen_encoders:
                enc = copy.deepcopy(frozen_encoders[m]).to(self.device)
                enc.eval()
                for p in enc.parameters():
                    p.requires_grad = False
                local_encoders[m] = enc

        # Fresh local AdamW optimizer for fusion + decoder params only (§8.4)
        trainable_params = list(local_fusion.parameters()) + list(local_decoder.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )

        p2_loss_fn = Phase2Loss(lambda2=0.1, tau=0.1, eps_d=1e-5)
        augmenter = PairwiseAugmentation(modalities=track_modalities, is_training=True)
        slice_sampler = SliceSampler(dataset, seed=self.seed)
        samples = slice_sampler.sample_slices_for_round(slices_per_patient=16)

        device_fused_proto = global_fused_prototypes.to(self.device)

        for epoch in range(local_epochs):
            for pid, slice_idx in samples:
                vol = dataset.load_patient_volume(pid)

                # Build sample dict for pairwise augmentation (§13.4)
                sample_dict = {
                    m: torch.from_numpy(vol["modalities"][m][slice_idx]).unsqueeze(0)
                    for m in track_modalities
                    if m in vol["modalities"]
                }
                sample_dict["label"] = torch.from_numpy(vol["labels"][slice_idx])

                aug_sample = augmenter(sample_dict)
                y_tensor = aug_sample["label"].long().unsqueeze(0).to(self.device)

                # Extract features through frozen encoders with stop-gradient detach() (§7)
                mod_features = {}
                with torch.no_grad():
                    for m, enc in local_encoders.items():
                        x_tensor = aug_sample[m].unsqueeze(0).float().to(self.device)
                        h1, h2, h3, h4, z = enc(x_tensor)
                        # Wrap features with detach() to guarantee stop-gradient
                        mod_features[m] = (h1.detach(), h2.detach(), h3.detach(), h4.detach(), z.detach())

                # Pass through fusion head and decoder
                f1, f2, f3, f4, z_S = local_fusion(mod_features)
                logits = local_decoder(z_S=z_S, f_S_4=f4, f_S_3=f3, f_S_2=f2, f_S_1=f1)

                optimizer.zero_grad()
                loss = p2_loss_fn(logits_S=logits, targets=y_tensor, z_S=z_S, fused_prototypes_S=device_fused_proto)
                loss.backward()
                optimizer.step()

        # Post-local fused prototype recomputation pass in FP32 eval mode (§8.3)
        local_fused_proto = torch.zeros_like(global_fused_prototypes)
        local_counts = torch.zeros(4, dtype=torch.long)

        with torch.no_grad():
            local_fusion.eval()
            local_decoder.eval()
            all_z_S, all_labels = [], []

            for pid in self.patient_ids:
                vol = dataset.load_patient_volume(pid)
                lab_vol = vol["labels"]
                num_slices = lab_vol.shape[0]

                for start in range(0, num_slices, self.batch_size):
                    end = min(start + self.batch_size, num_slices)
                    y_b = torch.from_numpy(lab_vol[start:end]).long().to(self.device)

                    mod_feats_b = {}
                    for m, enc in local_encoders.items():
                        x_b = torch.from_numpy(vol["modalities"][m][start:end]).float().unsqueeze(1).to(self.device)
                        h1, h2, h3, h4, z = enc(x_b)
                        mod_feats_b[m] = (h1, h2, h3, h4, z)

                    _, _, _, _, z_S_b = local_fusion(mod_feats_b)
                    all_z_S.append(z_S_b.cpu())
                    all_labels.append(y_b.cpu())

            if all_z_S:
                cat_z_S = torch.cat(all_z_S, dim=0)
                cat_y = torch.cat(all_labels, dim=0)
                local_fused_proto, local_counts = PrototypeBank.compute_local_prototypes(cat_z_S, cat_y)

        return {
            "hospital_id": self.hospital_id,
            "track_id": track_id,
            "fusion_state_dict": local_fusion.cpu().state_dict(),
            "decoder_state_dict": local_decoder.cpu().state_dict(),
            "fused_prototypes": local_fused_proto,
            "support_counts": local_counts,
            "image_lineage": set(track_modalities),
        }
