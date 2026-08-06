"""
CAMFS M1 — Co-registered Pairwise Augmentations
===============================================

Applies identical 2D spatial transformations (horizontal flip p=0.5, rotation [-10°, 10°])
across all co-registered modalities and label masks, followed by per-modality intensity
scaling ([0.9, 1.1]) and shift ([-0.1, 0.1]) applied strictly inside nonzero brain masks.

Spec reference: §13.4
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class PairwiseAugmentation:
    def __init__(self, modalities: Optional[List[str]] = None, is_training: bool = True):
        self.modalities = modalities or ["T1", "T1ce", "T2", "FLAIR"]
        self.is_training = is_training

    def __call__(self, sample_dict: Dict[str, Union[torch.Tensor, str, int]]) -> Dict[str, Union[torch.Tensor, str, int]]:
        if not self.is_training:
            return sample_dict

        # Sample spatial transform parameters
        do_flip = random.random() < 0.5
        angle = random.uniform(-10.0, 10.0)

        # Process each modality image
        for mod in self.modalities:
            if mod not in sample_dict:
                continue

            img = sample_dict[mod]  # [1, 240, 240]
            
            # Geometric flip
            if do_flip:
                img = TF.hflip(img)

            # Rotation (bilinear for images, zero fill)
            img = TF.rotate(img, angle=angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)

            # Intensity augmentation on nonzero brain mask only
            brain_mask = img != 0.0
            if torch.any(brain_mask):
                scale = random.uniform(0.9, 1.1)
                shift = random.uniform(-0.1, 0.1)
                img[brain_mask] = img[brain_mask] * scale + shift

            # Reset outside mask pixels to 0
            img[~brain_mask] = 0.0
            sample_dict[mod] = img

        # Process label map
        if "label" in sample_dict:
            label = sample_dict["label"].unsqueeze(0)  # [1, 240, 240]
            if do_flip:
                label = TF.hflip(label)
            
            # Rotation (nearest-neighbor for labels)
            label = TF.rotate(label, angle=angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
            sample_dict["label"] = label.squeeze(0)

        return sample_dict

    def augment_batch(
        self,
        mod_batches: Dict[str, torch.Tensor],
        y_batch: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Pure PyTorch CUDA native 2D pairwise augmentation (§13.4).
        Uses native GPU tensor flips and F.grid_sample affine rotations.
        """
        if not self.is_training:
            return mod_batches, y_batch

        do_flip = random.random() < 0.5
        angle = random.uniform(-10.0, 10.0)

        grid = None
        if abs(angle) > 1e-3 and len(mod_batches) > 0:
            first_tensor = next(iter(mod_batches.values()))
            rad = math.radians(-angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            B = first_tensor.shape[0]
            theta = torch.tensor([
                [cos_a, -sin_a, 0.0],
                [sin_a,  cos_a, 0.0]
            ], dtype=first_tensor.dtype, device=first_tensor.device).unsqueeze(0).repeat(B, 1, 1)
            grid = F.affine_grid(theta, first_tensor.shape, align_corners=False)

        aug_mods = {}
        for mod, x_b in mod_batches.items():
            if do_flip:
                x_b = x_b.flip(dims=[-1])
            if grid is not None:
                x_b = F.grid_sample(x_b, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

            brain_mask = (x_b != 0.0)
            scale = random.uniform(0.9, 1.1)
            shift = random.uniform(-0.1, 0.1)
            x_b = torch.where(brain_mask, x_b * scale + shift, torch.tensor(0.0, device=x_b.device))
            aug_mods[mod] = x_b

        # Process labels (B, 1, H, W)
        if y_batch.ndim == 3:
            y_batch = y_batch.unsqueeze(1)  # (B, 1, H, W)
        
        if do_flip:
            y_batch = y_batch.flip(dims=[-1])
        if grid is not None:
            y_float = F.grid_sample(y_batch.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
            y_batch = y_float.long()

        return aug_mods, y_batch.squeeze(1)
