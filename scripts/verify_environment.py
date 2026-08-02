"""
CAMFS M1 — Environment Verification Script
===========================================

Dumps and logs runtime Python, PyTorch, CUDA, cuDNN, NumPy, and GPU device details.
Spec reference: §1.3 of research code standards.
"""

import json
import sys
import numpy as np
import torch

def get_environment_dump() -> dict:
    """Collect runtime environment version details."""
    dump = {
        "python": sys.version.split("\n")[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    return dump

def main():
    dump = get_environment_dump()
    print("═" * 60)
    print("         CAMFS M1 Environment Runtime Version Dump")
    print("═" * 60)
    print(json.dumps(dump, indent=2))
    print("═" * 60)

if __name__ == "__main__":
    main()
