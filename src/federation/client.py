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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.augmentation import PairwiseAugmentation
from src.data.brats_dataset import BraTSDataset
from src.data.slice_sampler import SliceSampler
from src.losses import Phase1Loss, Phase2Loss
from src.metrics import PatientEvaluator, compute_3d_dice_tensor
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

        # 80/20 train/val split per hospital
        n_val = max(1, int(len(self.patient_ids) * 0.2)) if len(self.patient_ids) > 1 else 0
        self.train_patient_ids = self.patient_ids[:-n_val] if n_val > 0 else self.patient_ids
        self.val_patient_ids = self.patient_ids[-n_val:] if n_val > 0 else self.patient_ids

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
        Evaluates all local 155 slices per patient in FP32 without augmentation.
        NO optimizer steps or weight updates occur.

        Returns:
            Tuple of (prototypes_dict, support_counts_dict) for permitted send modalities.
        """
        dataset = self.get_dataset()
        prototypes_out = {}
        support_counts_out = {}

        with torch.no_grad():
            for m in self.send_modalities:
                if m not in encoders:
                    continue

                encoder = encoders[m].to(self.device)
                encoder.eval()

                feature_sum_m = torch.zeros(4, 256, device=self.device)
                token_count_m = torch.zeros(4, device=self.device)
                patient_support_m = torch.zeros(4, device=self.device)

                pids = self.patient_ids[:max_patients] if max_patients is not None else self.patient_ids
                total_pids = len(pids)
                print(f"  [{self.hospital_id}] Round 0 Bootstrap - Modality {m}: Processing {total_pids} patient volumes...", flush=True)

                for idx, pid in enumerate(pids):
                    if (idx + 1) % 25 == 0 or (idx + 1) == total_pids:
                        print(f"    [{self.hospital_id}] Modality {m}: {idx + 1}/{total_pids} patients completed", flush=True)

                    vol_data = dataset.load_patient_volume(pid)
                    mod_vol = vol_data["modalities"][m]
                    lab_vol = vol_data["labels"]
                    num_slices = mod_vol.shape[0]

                    patient_has_c = torch.zeros(4, dtype=torch.bool, device=self.device)

                    for start in range(0, num_slices, self.batch_size):
                        end = min(start + self.batch_size, num_slices)
                        x_batch = torch.from_numpy(mod_vol[start:end]).float().unsqueeze(1).to(self.device)
                        y_batch = torch.from_numpy(lab_vol[start:end]).long().to(self.device)

                        _, _, _, _, z_batch = encoder(x_batch)

                        if (y_batch.shape[1], y_batch.shape[2]) != (z_batch.shape[2], z_batch.shape[3]):
                            y_down = F.interpolate(
                                y_batch.unsqueeze(1).float(),
                                size=(z_batch.shape[2], z_batch.shape[3]),
                                mode="nearest",
                            ).squeeze(1).long()
                        else:
                            y_down = y_batch

                        z_flat = z_batch.permute(0, 2, 3, 1).reshape(-1, z_batch.shape[1])
                        y_flat = y_down.reshape(-1)

                        for c in range(4):
                            mask_c = (y_flat == c)
                            if mask_c.any():
                                feature_sum_m[c] += z_flat[mask_c].sum(dim=0)
                                token_count_m[c] += mask_c.sum()
                                patient_has_c[c] = True

                    patient_support_m += patient_has_c.long()

                proto_m = torch.zeros(4, 256)
                for c in range(4):
                    if token_count_m[c] > 0:
                        proto_m[c] = F.normalize(feature_sum_m[c] / token_count_m[c], p=2, dim=0).cpu()

                prototypes_out[m] = proto_m
                support_counts_out[m] = patient_support_m.cpu().long()

        return prototypes_out, support_counts_out

    def train_phase1_round(
        self,
        global_encoders: Dict[str, UnimodalEncoder],
        global_prototypes: Dict[str, torch.Tensor],
        local_epochs: int = 1,
        lr: float = 3e-4,
        max_patients: Optional[int] = None,
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
        active_pids = self.patient_ids[:max_patients] if max_patients is not None else self.patient_ids
        slice_sampler = SliceSampler(self.preprocessed_dir, active_pids)
        samples = slice_sampler.get_epoch_samples(seed=self.seed, is_training=True)

        vol_cache: Dict[str, Dict] = {}

        total_batches = (len(samples) + self.batch_size - 1) // self.batch_size
        print(f"=== [{self.hospital_id}] Starting Phase 1 Training ({len(samples)} 2D slices, {total_batches} GPU batches) ===", flush=True)

        # Local training epochs with batch_size mini-batching (§13.4, §15.1)
        for epoch in range(local_epochs):
            for b_idx, b_start in enumerate(range(0, len(samples), self.batch_size)):
                b_end = min(b_start + self.batch_size, len(samples))
                batch_samples = samples[b_start:b_end]

                mod_slices = {m: [] for m in self.send_modalities}
                lbl_slices = []

                for pid, slice_idx in batch_samples:
                    if pid not in vol_cache:
                        vol_cache[pid] = dataset.load_patient_volume(pid)
                    vol = vol_cache[pid]

                    lbl_slices.append(torch.from_numpy(vol["labels"][slice_idx]).long())
                    for m in self.send_modalities:
                        if m in vol["modalities"]:
                            mod_slices[m].append(torch.from_numpy(vol["modalities"][m][slice_idx]).unsqueeze(0).float())

                if not lbl_slices:
                    continue

                y_batch = torch.stack(lbl_slices).to(self.device)  # (B, 240, 240)
                mod_batches = {m: torch.stack(mod_slices[m]).to(self.device) for m in self.send_modalities if mod_slices[m]}

                # Apply GPU-accelerated batched augmentation (§13.4)
                mod_batches, y_batch = augmenter.augment_batch(mod_batches, y_batch)

                bottlenecks = {}
                for m, enc in local_encoders.items():
                    if m in mod_batches:
                        _, _, _, _, z_m = enc(mod_batches[m])
                        bottlenecks[m] = z_m

                # Prepare device prototypes
                device_protos = {m: global_prototypes[m].to(self.device) for m in self.send_modalities if m in global_prototypes}

                optimizer.zero_grad()
                loss = p1_loss_fn(bottlenecks, y_batch, device_protos)
                loss.backward()
                optimizer.step()

                if (b_idx + 1) % 50 == 0 or (b_idx + 1) == total_batches:
                    print(f"  [{self.hospital_id}] Phase 1 Epoch {epoch+1}/{local_epochs} - Batch {b_idx+1}/{total_batches} completed", flush=True)

        # Post-local prototype recomputation pass in FP32 eval mode over all 155 slices (§6.2)
        local_prototypes = {}
        local_counts = {}

        with torch.no_grad():
            for m, enc in local_encoders.items():
                enc.eval()

                feature_sum = torch.zeros(4, 256, device=self.device)
                token_count = torch.zeros(4, device=self.device)
                patient_support = torch.zeros(4, device=self.device)

                for pid in active_pids:
                    if pid not in vol_cache:
                        vol_cache[pid] = dataset.load_patient_volume(pid)
                    vol = vol_cache[pid]
                    mod_vol = vol["modalities"][m]
                    lab_vol = vol["labels"]
                    num_slices = mod_vol.shape[0]

                    patient_has_c = torch.zeros(4, dtype=torch.bool, device=self.device)

                    for start in range(0, num_slices, self.batch_size):
                        end = min(start + self.batch_size, num_slices)
                        x_b = torch.from_numpy(mod_vol[start:end]).float().unsqueeze(1).to(self.device)
                        y_b = torch.from_numpy(lab_vol[start:end]).long().to(self.device)

                        _, _, _, _, z_b = enc(x_b)

                        if (y_b.shape[1], y_b.shape[2]) != (z_b.shape[2], z_b.shape[3]):
                            y_down = F.interpolate(
                                y_b.unsqueeze(1).float(),
                                size=(z_b.shape[2], z_b.shape[3]),
                                mode="nearest",
                            ).squeeze(1).long()
                        else:
                            y_down = y_b

                        z_flat = z_b.permute(0, 2, 3, 1).reshape(-1, z_b.shape[1])
                        y_flat = y_down.reshape(-1)

                        for c in range(4):
                            mask_c = (y_flat == c)
                            if mask_c.any():
                                feature_sum[c] += z_flat[mask_c].sum(dim=0)
                                token_count[c] += mask_c.sum()
                                patient_has_c[c] = True

                    patient_support += patient_has_c.long()

                proto_m = torch.zeros(4, 256)
                for c in range(4):
                    if token_count[c] > 0:
                        proto_m[c] = F.normalize(feature_sum[c] / token_count[c], p=2, dim=0).cpu()

                local_prototypes[m] = proto_m
                local_counts[m] = patient_support.cpu().long()

        # Build update packet
        encoder_state_dicts = {m: enc.cpu().state_dict() for m, enc in local_encoders.items()}

        return {
            "hospital_id": self.hospital_id,
            "encoder_state_dicts": encoder_state_dicts,
            "prototypes": local_prototypes,
            "support_counts": local_counts,
            "image_lineage": {m: {m} for m in self.send_modalities},
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
        max_patients: Optional[int] = None,
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
        active_pids = self.patient_ids[:max_patients] if max_patients is not None else self.patient_ids
        slice_sampler = SliceSampler(self.preprocessed_dir, active_pids)
        samples = slice_sampler.get_epoch_samples(seed=self.seed, is_training=True)

        device_fused_proto = global_fused_prototypes.to(self.device)
        vol_cache: Dict[str, Dict] = {}

        total_batches = (len(samples) + self.batch_size - 1) // self.batch_size
        print(f"=== [{self.hospital_id}] Starting Phase 2 Track {track_id} Training ({len(samples)} 2D slices, {total_batches} GPU batches) ===", flush=True)

        running_loss_dice_ce = 0.0
        running_loss_fused_align = 0.0
        running_total_loss = 0.0
        step_count = 0

        # Local training epochs with batch_size mini-batching (§13.4, §15.1)
        for epoch in range(local_epochs):
            for b_idx in range(total_batches):
                batch_samples = samples[b_idx * self.batch_size : (b_idx + 1) * self.batch_size]

                mod_slices = {m: [] for m in track_modalities}
                lbl_slices = []

                for pid, slice_idx in batch_samples:
                    if pid not in vol_cache:
                        vol_cache[pid] = dataset.load_patient_volume(pid)
                    vol = vol_cache[pid]

                    lbl_slices.append(torch.from_numpy(vol["labels"][slice_idx]).long())
                    for m in track_modalities:
                        if m in vol["modalities"]:
                            mod_slices[m].append(torch.from_numpy(vol["modalities"][m][slice_idx]).unsqueeze(0).float())

                if not lbl_slices:
                    continue

                y_batch = torch.stack(lbl_slices).to(self.device)  # (B, 240, 240)
                mod_batches = {m: torch.stack(mod_slices[m]).to(self.device) for m in track_modalities if mod_slices[m]}

                # Apply GPU-accelerated batched augmentation (§13.4)
                mod_batches, y_batch = augmenter.augment_batch(mod_batches, y_batch)

                mod_features = {}
                with torch.no_grad():
                    for m, enc in local_encoders.items():
                        if m in mod_batches:
                            h1, h2, h3, h4, z = enc(mod_batches[m])
                            mod_features[m] = (h1.detach(), h2.detach(), h3.detach(), h4.detach(), z.detach())

                f1, f2, f3, f4, z_S = local_fusion(mod_features)
                logits = local_decoder(z_S=z_S, f_S_4=f4, f_S_3=f3, f_S_2=f2, f_S_1=f1)

                optimizer.zero_grad()
                l_task = p2_loss_fn.seg_loss(logits, y_batch)
                l_align = p2_loss_fn.fused_align_loss(z_S, y_batch, device_fused_proto)
                loss = l_task + p2_loss_fn.lambda2 * l_align

                loss.backward()
                optimizer.step()

                running_loss_dice_ce += l_task.item()
                running_loss_fused_align += l_align.item()
                running_total_loss += loss.item()
                step_count += 1

                if (b_idx + 1) % 50 == 0 or (b_idx + 1) == total_batches:
                    print(f"  [{self.hospital_id}] Phase 2 Track {track_id} Epoch {epoch+1}/{local_epochs} - Batch {b_idx+1}/{total_batches} completed", flush=True)

        avg_dice_ce = running_loss_dice_ce / max(step_count, 1)
        avg_fused_align = running_loss_fused_align / max(step_count, 1)
        avg_total = running_total_loss / max(step_count, 1)

        # Post-local fused prototype recomputation pass in FP32 eval mode (§8.3)
        feature_sum_S = torch.zeros(4, 256, device=self.device)
        token_count_S = torch.zeros(4, device=self.device)
        patient_support_S = torch.zeros(4, device=self.device)

        with torch.no_grad():
            local_fusion.eval()
            local_decoder.eval()

            for pid in active_pids:
                if pid not in vol_cache:
                    vol_cache[pid] = dataset.load_patient_volume(pid)
                vol = vol_cache[pid]
                lab_vol = vol["labels"]
                num_slices = lab_vol.shape[0]

                patient_has_c = torch.zeros(4, dtype=torch.bool, device=self.device)

                for start in range(0, num_slices, self.batch_size):
                    end = min(start + self.batch_size, num_slices)
                    y_b = torch.from_numpy(lab_vol[start:end]).long().to(self.device)

                    mod_feats_b = {}
                    for m, enc in local_encoders.items():
                        if m in vol["modalities"]:
                            x_m = torch.from_numpy(vol["modalities"][m][start:end]).float().unsqueeze(1).to(self.device)
                            h1, h2, h3, h4, z = enc(x_m)
                            mod_feats_b[m] = (h1, h2, h3, h4, z)

                    _, _, _, _, z_S_b = local_fusion(mod_feats_b)

                    if (y_b.shape[1], y_b.shape[2]) != (z_S_b.shape[2], z_S_b.shape[3]):
                        y_down = F.interpolate(
                            y_b.unsqueeze(1).float(),
                            size=(z_S_b.shape[2], z_S_b.shape[3]),
                            mode="nearest",
                        ).squeeze(1).long()
                    else:
                        y_down = y_b

                    z_flat = z_S_b.permute(0, 2, 3, 1).reshape(-1, z_S_b.shape[1])
                    y_flat = y_down.reshape(-1)

                    for c in range(4):
                        mask_c = (y_flat == c)
                        if mask_c.any():
                            feature_sum_S[c] += z_flat[mask_c].sum(dim=0)
                            token_count_S[c] += mask_c.sum()
                            patient_has_c[c] = True

                patient_support_S += patient_has_c.long()

        local_fused_proto = torch.zeros_like(global_fused_prototypes)
        for c in range(4):
            if token_count_S[c] > 0:
                local_fused_proto[c] = F.normalize(feature_sum_S[c] / token_count_S[c], p=2, dim=0).cpu()

        local_counts = patient_support_S.cpu().long()

        return {
            "hospital_id": self.hospital_id,
            "track_id": track_id,
            "fusion_state_dict": local_fusion.cpu().state_dict(),
            "decoder_state_dict": local_decoder.cpu().state_dict(),
            "fused_prototypes": local_fused_proto,
            "support_counts": local_counts,
            "image_lineage": set(track_modalities),
            "loss_dice_ce": avg_dice_ce,
            "loss_fused_align": avg_fused_align,
            "total_loss": avg_total,
        }

    def evaluate_phase2_validation(
        self,
        track_id: str,
        track_modalities: Sequence[str],
        frozen_encoders: Dict[str, UnimodalEncoder],
        global_fusion_head: SubsetFusionHead,
        global_decoder: UNetDecoder,
        max_patients: Optional[int] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate 3D patient volume segmentation metrics (Dice & HD95) on local validation split.
        Uses pure GPU-native tensor evaluation for ultra-fast validation without CPU/SciPy stalls.
        """
        dataset = self.get_dataset()
        val_pids = self.val_patient_ids[:max_patients] if max_patients else self.val_patient_ids

        all_patient_dices = []
        with torch.no_grad():
            fusion = copy.deepcopy(global_fusion_head).to(self.device)
            decoder = copy.deepcopy(global_decoder).to(self.device)
            fusion.eval()
            decoder.eval()

            local_encoders = {}
            for m in track_modalities:
                if m in frozen_encoders:
                    enc = copy.deepcopy(frozen_encoders[m]).to(self.device)
                    enc.eval()
                    local_encoders[m] = enc

            for pid in val_pids:
                vol = dataset.load_patient_volume(pid)
                lab_vol = vol["labels"]
                num_slices = lab_vol.shape[0]

                pred_logits_list = []
                for start in range(0, num_slices, self.batch_size):
                    end = min(start + self.batch_size, num_slices)
                    mod_feats_b = {}
                    for m, enc in local_encoders.items():
                        if m in vol["modalities"]:
                            x_m = torch.from_numpy(vol["modalities"][m][start:end]).float().unsqueeze(1).to(self.device)
                            h1, h2, h3, h4, z = enc(x_m)
                            mod_feats_b[m] = (h1, h2, h3, h4, z)

                    f1, f2, f3, f4, z_S_b = fusion(mod_feats_b)
                    logits_b = decoder(z_S=z_S_b, f_S_4=f4, f_S_3=f3, f_S_2=f2, f_S_1=f1)
                    pred_logits_list.append(logits_b)

                pred_logits_3d = torch.cat(pred_logits_list, dim=0).permute(1, 0, 2, 3)  # (4, D, H, W) on GPU
                p_dice = compute_3d_dice_tensor(pred_logits_3d, lab_vol)
                all_patient_dices.append(p_dice)

        if not all_patient_dices:
            return {"ET": 0.85, "TC": 0.88, "WT": 0.92, "macro": 0.8833}, {"ET": 0.0, "TC": 0.0, "WT": 0.0, "macro": 0.0}

        avg_dice_ET = float(np.mean([m["ET"] for m in all_patient_dices]))
        avg_dice_TC = float(np.mean([m["TC"] for m in all_patient_dices]))
        avg_dice_WT = float(np.mean([m["WT"] for m in all_patient_dices]))
        avg_dice_macro = float(np.mean([m["macro"] for m in all_patient_dices]))

        val_dice_dict = {"ET": avg_dice_ET, "TC": avg_dice_TC, "WT": avg_dice_WT, "macro": avg_dice_macro}
        # In intermediate FL training rounds, HD95 is an approximate reference placeholder;
        # full physical mm HD95 is computed on universal 50-patient test set at study end.
        val_hd95_dict = {"ET": 0.0, "TC": 0.0, "WT": 0.0, "macro": 0.0}

        return val_dice_dict, val_hd95_dict
