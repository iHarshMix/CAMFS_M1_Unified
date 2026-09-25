"""
CAMFS M2 — Canonical Base Registry & Verification
=================================================

Implements §9.2 of the CAMFS M2 Specification:
- Canonical Base Digest b = SHA256(ordered parameter tensor bytes of B_{S3})
- Rights Digest rho_b = SHA256(b || purpose || contributors)
- Invariance verification before and after M2 operations
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 hash over raw file bytes."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _flatten_state_dict(d: Any, prefix: str = "") -> List[Tuple[str, torch.Tensor]]:
    """Recursively flattens nested state dicts into (dotted_key, tensor) pairs."""
    items: List[Tuple[str, torch.Tensor]] = []
    if isinstance(d, dict):
        for k in sorted(d.keys()):
            new_prefix = f"{prefix}.{k}" if prefix else k
            items.extend(_flatten_state_dict(d[k], new_prefix))
    elif isinstance(d, torch.Tensor):
        items.append((prefix, d))
    return items


def compute_state_dict_sha256(state_dict: Any) -> str:
    """
    Computes a deterministic SHA-256 digest over ordered state_dict tensor bytes.
    Handles arbitrary nested dictionaries of tensors.
    """
    h = hashlib.sha256()
    flattened = _flatten_state_dict(state_dict)

    for key, tensor in flattened:
        t = tensor.detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())

    return h.hexdigest()


def compute_canonical_base_hash(
    phase1_checkpoint_path: Union[str, Path],
    track_s3_checkpoint_path: Union[str, Path],
) -> Dict[str, str]:
    """
    Computes canonical base digests for M1 frozen components.

    Returns:
        Dictionary containing:
            - 'phase1_file_sha256': file digest of phase1_frozen.pt
            - 'track_s3_file_sha256': file digest of best_track_S3.pt
            - 'phase1_weights_sha256': ordered parameter digest of phase1 encoders
            - 'track_s3_weights_sha256': ordered parameter digest of track S3
            - 'canonical_base_hash': composite base identifier b
    """
    p1_path = Path(phase1_checkpoint_path)
    s3_path = Path(track_s3_checkpoint_path)

    p1_file_hash = compute_file_sha256(p1_path)
    s3_file_hash = compute_file_sha256(s3_path)

    p1_ckpt = torch.load(p1_path, map_location="cpu")
    s3_ckpt = torch.load(s3_path, map_location="cpu")

    p1_weights = p1_ckpt.get("state_dict", p1_ckpt)
    s3_weights = s3_ckpt.get("state_dict", s3_ckpt)

    p1_w_hash = compute_state_dict_sha256(p1_weights)
    s3_w_hash = compute_state_dict_sha256(s3_weights)

    # Composite canonical base hash b
    composite = hashlib.sha256(f"{p1_w_hash}:{s3_w_hash}".encode("utf-8")).hexdigest()

    return {
        "phase1_file_sha256": p1_file_hash,
        "track_s3_file_sha256": s3_file_hash,
        "phase1_weights_sha256": p1_w_hash,
        "track_s3_weights_sha256": s3_w_hash,
        "canonical_base_hash": composite,
    }


def compute_rights_digest(
    base_hash: str,
    purpose: str = "CAMFS_M2_CDRD_H1_TO_H3",
    contributors: Optional[List[str]] = None,
) -> str:
    """Computes rights digest rho_b (§9.2)."""
    if contributors is None:
        contributors = ["H1", "H2", "H3", "H4"]
    contrib_str = ",".join(sorted(contributors))
    payload = f"{base_hash}:{purpose}:{contrib_str}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_base_integrity(
    phase1_checkpoint_path: Union[str, Path],
    track_s3_checkpoint_path: Union[str, Path],
    expected_base_hash: str,
) -> bool:
    """
    Verifies that the M1 base checkpoint parameters have not mutated.
    Raises ValueError if hashes do not match.
    """
    digests = compute_canonical_base_hash(phase1_checkpoint_path, track_s3_checkpoint_path)
    actual_hash = digests["canonical_base_hash"]
    if actual_hash != expected_base_hash:
        raise ValueError(
            f"M1 Base Integrity Check Failed!\n"
            f"Expected: {expected_base_hash}\n"
            f"Actual:   {actual_hash}"
        )
    return True
