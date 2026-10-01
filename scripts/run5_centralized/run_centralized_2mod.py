"""
CAMFS M1 — RUN-5: Centralized Ceiling for 2-Modality Subset (H3 Ceiling)
========================================================================

Dedicated runner for Centralized Ceiling on 2 modalities:
  - Modalities: T1, FLAIR
  - Pooled Dataset: 256 train / 62 val patients
  - Target Evaluation: Ceiling for Hospital H3
  - Universal 50 pure held-out test evaluation via PatientEvaluator
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run5_centralized.run_centralized import run_centralized

if __name__ == "__main__":
    run_centralized(modality_config="2mod")
