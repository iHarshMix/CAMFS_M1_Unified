"""
CAMFS M2 — Consent-Bounded Detached Residual Distillation (CDRD)
================================================================

Post-M1 knowledge transfer framework for cross-boundary MRI modality distillation.
Enables Hospital H3 (T1, FLAIR) to absorb contrast-enhancing tumor knowledge
from Hospital H1 (T1, T1ce, T2, FLAIR) without moving raw data or modifying M1.
"""

from __future__ import annotations

__version__ = "2.0.0"
