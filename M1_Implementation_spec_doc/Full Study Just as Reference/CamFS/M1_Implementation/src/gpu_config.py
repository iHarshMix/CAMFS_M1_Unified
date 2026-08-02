"""
CAMFS M1 — Dynamic GPU & Environment Configuration
====================================================

This module auto-detects whether the code is running on:
  - Local WSL2 / workstation (small GPU, e.g. 6 GB)
  - HPC cluster via Slurm / PBS / LSF (large GPU, e.g. 90 GB)

It dynamically adjusts VRAM caps, batch sizes, precision, DataLoader
workers, and gradient accumulation so that the rest of the codebase
requires ZERO if/else branches for environment differences.

IMPORTANT — Import Order
-------------------------
This module MUST be imported BEFORE any other module that touches
``torch.cuda``.  The recommended pattern in your entrypoint is::

    # ── very first lines of run_primary.py / train.py ──
    from src.gpu_config import GPUConfig, setup_environment
    env = setup_environment()          # sets env vars + CUDA limits
    import torch                       # now safe
    from src.config import load_yaml   # etc.

If ``torch`` is imported before ``setup_environment()`` is called, the
module will still work but will emit a warning that CUDA allocator env
vars may not have taken effect.

Author : CAMFS research team
Spec   : Extends §15 reproducibility defaults with environment-adaptive
         runtime configuration.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Phase 0 — Pre-torch environment variables
# ──────────────────────────────────────────────────────────────────────
# These MUST be set before the CUDA allocator is initialised (i.e.
# before any torch.cuda call).  We set them at *module import time*
# so that importing this module first is sufficient.

_TORCH_IMPORTED_BEFORE_US = "torch" in sys.modules


def _set_pretorch_env_vars() -> None:
    """Set CUDA-allocator env vars.  Safe to call multiple times."""
    # expandable_segments reduces fragmentation on small-VRAM GPUs
    # (especially relevant for WSL2 where the VRAM ceiling is tight).
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    # Disable TF32 globally for FP32 reproducibility (§15 spec).
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")


_set_pretorch_env_vars()

# NOW it is safe to import torch
import torch  # noqa: E402
import torch.cuda  # noqa: E402
import torch.distributed as dist  # noqa: E402

if _TORCH_IMPORTED_BEFORE_US:
    warnings.warn(
        "gpu_config: torch was imported before gpu_config.  "
        "CUDA allocator env vars (PYTORCH_CUDA_ALLOC_CONF) may not "
        "have taken effect.  Move `from src.gpu_config import ...` "
        "above `import torch` in your entrypoint.",
        RuntimeWarning,
        stacklevel=2,
    )


# ──────────────────────────────────────────────────────────────────────
# Environment Detection Helpers
# ──────────────────────────────────────────────────────────────────────

def _is_wsl2() -> bool:
    """Detect WSL2 by inspecting /proc/version."""
    if platform.system() != "Linux":
        return False
    try:
        version_text = Path("/proc/version").read_text().lower()
        return "microsoft" in version_text or "wsl" in version_text
    except (OSError, PermissionError):
        return False


def _is_cluster() -> bool:
    """Detect HPC job scheduler (Slurm / PBS / LSF)."""
    return any(
        var in os.environ
        for var in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "SGE_TASK_ID")
    )


def _get_local_rank() -> int:
    """Return LOCAL_RANK for DDP; defaults to 0 for single-GPU."""
    for var in ("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    return 0


def _get_world_size() -> int:
    """Return total number of processes for DDP; defaults to 1."""
    for var in ("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    return 1


# ──────────────────────────────────────────────────────────────────────
# GPUConfig — Typed, Immutable Configuration Object
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GPUConfig:
    """
    Immutable snapshot of the runtime GPU environment.

    Every field is automatically populated by ``setup_environment()``.
    Training code reads these values instead of hard-coding assumptions.

    Attributes
    ----------
    device : torch.device
        Target device (``cuda:N`` or ``cpu``).
    is_cluster : bool
        True when running inside a Slurm / PBS / LSF job.
    is_wsl2 : bool
        True when running inside WSL2.
    env_label : str
        Human-readable label: ``"HPC Cluster"``, ``"WSL2"``, or ``"Workstation"``.
    total_vram_gb : float
        Total hardware VRAM of the selected GPU (GB).
    usable_vram_gb : float
        VRAM cap applied by this module (GB).  Equals total on cluster.
    local_rank : int
        GPU index for this process (DDP-aware).
    world_size : int
        Number of distributed processes.
    use_amp : bool
        Whether automatic mixed precision is enabled.
    amp_dtype : torch.dtype
        AMP dtype (``torch.float16`` or ``torch.bfloat16``).
    batch_size : int
        Per-GPU batch size (already scaled for available VRAM).
    grad_accum_steps : int
        Gradient accumulation steps to recover effective batch size.
    effective_batch_size : int
        ``batch_size × grad_accum_steps × world_size``.
    num_workers : int
        DataLoader worker count (reduced on WSL2).
    pin_memory : bool
        DataLoader pin_memory flag.
    gpu_name : str
        GPU device name string.
    """

    # ── Core device ──
    device: torch.device
    is_cluster: bool
    is_wsl2: bool
    env_label: str

    # ── VRAM ──
    total_vram_gb: float
    usable_vram_gb: float

    # ── Distributed ──
    local_rank: int
    world_size: int

    # ── Precision ──
    use_amp: bool
    amp_dtype: torch.dtype

    # ── Throughput ──
    batch_size: int
    grad_accum_steps: int
    effective_batch_size: int
    num_workers: int
    pin_memory: bool

    # ── Metadata ──
    gpu_name: str

    # ──────────────────────────────────────────────────────────────
    # Convenience helpers for training code
    # ──────────────────────────────────────────────────────────────

    def amp_context(self):
        """Return the AMP autocast context manager."""
        if self.use_amp and self.device.type == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)
        else:
            # No-op context manager
            return torch.amp.autocast(device_type="cuda", enabled=False)

    def grad_scaler(self) -> torch.amp.GradScaler:
        """Return a GradScaler configured for this environment."""
        return torch.amp.GradScaler(enabled=self.use_amp)

    def dataloader_kwargs(self) -> dict[str, Any]:
        """Return kwargs to pass to ``torch.utils.data.DataLoader``."""
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.num_workers > 0,
        }

    def summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for the run ledger."""
        return {
            "env_label": self.env_label,
            "gpu_name": self.gpu_name,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "usable_vram_gb": round(self.usable_vram_gb, 2),
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "use_amp": self.use_amp,
            "amp_dtype": str(self.amp_dtype),
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "effective_batch_size": self.effective_batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }

    def __str__(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║              CAMFS M1 — GPU Configuration               ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Environment : {self.env_label:<41s} ║",
            f"║  GPU         : {self.gpu_name:<41s} ║",
            f"║  VRAM        : {self.usable_vram_gb:.1f} GB / {self.total_vram_gb:.1f} GB"
            + " " * (29 - len(f"{self.usable_vram_gb:.1f} GB / {self.total_vram_gb:.1f} GB"))
            + "║",
            f"║  Device      : {str(self.device):<41s} ║",
            f"║  Rank        : {self.local_rank} / {self.world_size}"
            + " " * (37 - len(f"{self.local_rank} / {self.world_size}"))
            + "║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  AMP         : {'ON (' + str(self.amp_dtype).split('.')[-1] + ')' if self.use_amp else 'OFF':<41s} ║",
            f"║  Batch Size  : {self.batch_size:<41d} ║",
            f"║  Grad Accum  : {self.grad_accum_steps:<41d} ║",
            f"║  Effective BS: {self.effective_batch_size:<41d} ║",
            f"║  Workers     : {self.num_workers:<41d} ║",
            f"║  Pin Memory  : {str(self.pin_memory):<41s} ║",
            "╚══════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# OOM-Safe Training Step
# ──────────────────────────────────────────────────────────────────────

def oom_safe_forward(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    loss_fn,
    device: torch.device,
    amp_context,
    grad_scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    max_retries: int = 2,
) -> Optional[float]:
    """
    Attempt a forward + backward pass.  On CUDA OOM, clear the cache,
    halve the micro-batch, and retry up to ``max_retries`` times.

    Parameters
    ----------
    model : nn.Module
    batch : dict
        Must contain at minimum the keys expected by ``loss_fn``.
    loss_fn : callable
        ``loss_fn(model_output, targets) -> scalar loss``.
    device : torch.device
    amp_context : context manager from ``GPUConfig.amp_context()``.
    grad_scaler : GradScaler from ``GPUConfig.grad_scaler()``.
    optimizer : torch.optim.Optimizer
    max_retries : int
        Number of OOM retry attempts (each halves the micro-batch).

    Returns
    -------
    float or None
        The loss value, or None if all retries were exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            optimizer.zero_grad(set_to_none=True)
            with amp_context:
                outputs = model(batch)
                loss = loss_fn(outputs, batch)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            return loss.item()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt < max_retries:
                # Halve the batch along dim 0
                batch = {
                    k: v[:v.shape[0] // 2] if isinstance(v, torch.Tensor) and v.dim() > 0
                    else v
                    for k, v in batch.items()
                }
                logger.warning(
                    "CUDA OOM on attempt %d/%d.  Retrying with halved micro-batch.",
                    attempt + 1, max_retries + 1,
                )
            else:
                logger.error(
                    "CUDA OOM persists after %d retries.  Skipping batch.", max_retries + 1,
                )
                return None


# ──────────────────────────────────────────────────────────────────────
# Main Entry Point — setup_environment()
# ──────────────────────────────────────────────────────────────────────

def setup_environment(
    *,
    target_batch_size: int = 16,
    local_vram_cap_gb: float = 5.0,
    cluster_gpu_baseline_gb: float = 90.0,
    local_num_workers: int = 2,
    cluster_num_workers: int = 4,
    force_vram_gb: Optional[float] = None,
    force_amp: Optional[bool] = None,
    force_cpu: bool = False,
) -> GPUConfig:
    """
    Detect the runtime environment and return a fully configured
    ``GPUConfig``.

    Parameters
    ----------
    target_batch_size : int
        The desired *effective* per-GPU batch size on the cluster
        baseline GPU (``cluster_gpu_baseline_gb``).  On smaller local
        GPUs the per-GPU batch is reduced and ``grad_accum_steps`` is
        increased to preserve the effective batch size.
    local_vram_cap_gb : float
        Hard VRAM cap (GB) applied when running locally (WSL2 /
        workstation).  Set to ``None`` to use all available VRAM.
    cluster_gpu_baseline_gb : float
        Expected VRAM (GB) of cluster GPUs.  Used as the denominator
        when scaling batch size for smaller local GPUs.
    local_num_workers : int
        DataLoader ``num_workers`` on local / WSL2 (kept low to avoid
        fork / shared-memory issues).
    cluster_num_workers : int
        DataLoader ``num_workers`` on the cluster.
    force_vram_gb : float, optional
        Override the VRAM cap regardless of environment.
    force_amp : bool, optional
        Override automatic AMP decision.
    force_cpu : bool
        Force CPU mode even when a GPU is available.

    Returns
    -------
    GPUConfig
    """
    is_cluster = _is_cluster()
    is_wsl2 = _is_wsl2()

    if is_cluster:
        env_label = f"HPC Cluster (Job {os.environ.get('SLURM_JOB_ID', os.environ.get('PBS_JOBID', 'N/A'))})"
    elif is_wsl2:
        env_label = "WSL2 (Local)"
    else:
        env_label = "Workstation (Local)"

    # ── Device selection ──
    local_rank = _get_local_rank()
    world_size = _get_world_size()
    has_cuda = torch.cuda.is_available() and not force_cpu

    if has_cuda:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    else:
        device = torch.device("cpu")
        gpu_name = "CPU (no CUDA)"
        total_vram_gb = 0.0

    # ── VRAM cap ──
    if force_vram_gb is not None:
        usable_vram_gb = force_vram_gb
    elif not is_cluster and has_cuda:
        # Local: cap to leave headroom for the OS / display server
        usable_vram_gb = min(local_vram_cap_gb, total_vram_gb - 0.3)
        usable_vram_gb = max(usable_vram_gb, 0.5)  # floor at 0.5 GB
    else:
        usable_vram_gb = total_vram_gb  # cluster: use everything

    if has_cuda and usable_vram_gb < total_vram_gb:
        fraction = usable_vram_gb / total_vram_gb
        fraction = max(0.05, min(fraction, 1.0))
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
        logger.info(
            "VRAM capped at %.2f GB (%.1f%% of %.2f GB)",
            usable_vram_gb, fraction * 100, total_vram_gb,
        )

    # ── Precision ──
    # On local small GPUs: enable AMP to halve memory.
    # On cluster: respect spec §15 (fp32 default, but allow override).
    if force_amp is not None:
        use_amp = force_amp
    elif not is_cluster and has_cuda and usable_vram_gb < 12.0:
        use_amp = True  # Auto-enable on small GPUs
    else:
        use_amp = False  # Cluster: FP32 per spec §15

    # Prefer bfloat16 if supported (A100/H100), else float16
    if has_cuda and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    # ── Batch size scaling ──
    if has_cuda and usable_vram_gb > 0:
        # Scale relative to the cluster baseline
        vram_ratio = usable_vram_gb / cluster_gpu_baseline_gb
        # AMP roughly halves activation memory
        if use_amp:
            vram_ratio *= 1.8  # conservative 1.8× instead of 2×
        scaled_batch = max(1, round(target_batch_size * vram_ratio))
    else:
        # CPU: small batch
        scaled_batch = max(1, target_batch_size // 4)

    # Gradient accumulation to recover effective batch size
    if scaled_batch >= target_batch_size:
        grad_accum_steps = 1
        scaled_batch = target_batch_size  # don't exceed target
    else:
        # Round up so effective ≥ target
        grad_accum_steps = -(-target_batch_size // scaled_batch)  # ceil division

    effective_batch_size = scaled_batch * grad_accum_steps * world_size

    # ── DataLoader ──
    if is_wsl2 or (not is_cluster and not has_cuda):
        num_workers = local_num_workers
        pin_memory = has_cuda
    else:
        num_workers = cluster_num_workers
        pin_memory = has_cuda

    # ── TF32 reproducibility (§15.4) ──
    if has_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # ── Build config ──
    config = GPUConfig(
        device=device,
        is_cluster=is_cluster,
        is_wsl2=is_wsl2,
        env_label=env_label,
        total_vram_gb=round(total_vram_gb, 2),
        usable_vram_gb=round(usable_vram_gb, 2),
        local_rank=local_rank,
        world_size=world_size,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        batch_size=scaled_batch,
        grad_accum_steps=grad_accum_steps,
        effective_batch_size=effective_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        gpu_name=gpu_name,
    )

    # Print the summary box
    print(config)

    return config
