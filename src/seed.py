"""
CAMFS M1 — Deterministic Seed Management
=========================================

Handles framework, accelerator, and worker seed initialization to enforce
bitwise reproducibility on identical hardware and software environments.
Spec reference: §15.4
"""

import os
import random
import numpy as np
import torch

def set_deterministic(seed: int) -> None:
    """
    Set deterministic seeds across Python, NumPy, and PyTorch.
    Must be called at the very beginning of execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enforce deterministic cuDNN & PyTorch algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass

def seed_worker(worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoaders."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def make_generator(seed: int) -> torch.Generator:
    """Create a seeded PyTorch Generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
