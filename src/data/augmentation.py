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
from typing import Dict, List, Optional, Union
import torch
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
