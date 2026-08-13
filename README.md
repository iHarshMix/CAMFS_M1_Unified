# CAMFS M1 — Consent-Constrained Multimodal Federated Learning for Brain Tumor Segmentation

This repository contains the official implementation of **CAMFS M1** (Consent-Constrained Multimodal Federated Segmentation), fully reproducible under PyTorch 2.5.1 and Python 3.10.

---

## 1. Quick Start & One-Command Reproduction

### Environment Setup
```bash
conda env create -f environment.yml
conda activate camfs
```

### Preprocessing & Partitions
```bash
# Extract raw dataset (MICCAI_BraTS2020_TrainingData)
python scripts/unzip_dataset.py

# Preprocess and cache 2D axial slices with z-score normalization
python scripts/preprocess.py

# Generate patient-disjoint 4-hospital partition manifests
python scripts/make_partitions.py
```

### Verification & Unit Tests
```bash
# Run all 45 workspace unit tests across models, policy, loss, protocol, and ablations
PYTHONPATH=. pytest tests/ -v

# Audit pre-experiment verification gates (Gates 1-9)
PYTHONPATH=. python scripts/verify_pre_experiment_gates.py
```

---

## 2. Running Experiments

### Primary CAMFS M1 Run
```bash
# Run single CAMFS M1 experiment (Partition seed 1103, Training seed 17)
PYTHONPATH=. python scripts/run_primary.py --config configs/default.yaml --partition-seed 1103 --train-seed 17 --gpu 0

# Dry-run test mode (1 round Phase 1 & 2)
PYTHONPATH=. python scripts/run_primary.py --dry-run --gpu 0
```

### Ablation Studies (A1–A8)
```bash
# A1: Joint-training (no Phase 1/Phase 2 freeze separation)
PYTHONPATH=. python scripts/run_ablations.py --ablation A1_JointTraining --gpu 0

# A2: Delayed-site context reporting
PYTHONPATH=. python scripts/run_ablations.py --ablation A2_DelayedSite --gpu 0

# A3: Executable lineage audit self-verification
PYTHONPATH=. python scripts/run_ablations.py --ablation A3_LineageAudit --gpu 0

# A4: Unimodal alignment weight lambda1 sweep {0, 0.1, 0.5, 1.0}
PYTHONPATH=. python scripts/run_ablations.py --ablation A4_Lambda1Sweep --lambda1 0.0 --gpu 0

# A5: Cold-start variants (Tier 1, Tier 2, Tier 3)
PYTHONPATH=. python scripts/run_ablations.py --ablation A5_ColdStart --cold-start-tier 1 --gpu 0

# A6: Group-symmetric vs directional T2 policy
PYTHONPATH=. python scripts/run_ablations.py --ablation A6_Directional --gpu 0

# A7: Multi-track contribution toggle (R_contribute(H1, S3))
PYTHONPATH=. python scripts/run_ablations.py --ablation A7_MultiTrack --gpu 0

# A8: H2 reconnection (pull-only vs private head)
PYTHONPATH=. python scripts/run_ablations.py --ablation A8_Reconnection --h2-mode pull_only --gpu 0
```

---

## 3. Generating Paper Outputs (LaTeX Tables & PDF Figures)

```bash
# Generate LaTeX publication tables in outputs/tables/
PYTHONPATH=. python scripts/generate_tables.py

# Generate publication plots in outputs/figures/
PYTHONPATH=. python scripts/generate_figures.py
```

Outputs generated:
* `outputs/tables/primary_results.tex`
* `outputs/tables/ablation_results.tex`
* `outputs/tables/lineage_audit_summary.tex`
* `outputs/figures/phase1_convergence.pdf` (and `.png`)
* `outputs/figures/phase2_validation_curves.pdf` (and `.png`)
* `outputs/figures/ablation_a3_audit.pdf` (and `.png`)

---

## 4. Repository Structure

```
CAMFS_M1_Unified/
├── configs/
│   ├── default.yaml                 # Master §15 defaults
│   ├── policy_M1_PRIMARY_V1.json     # Exogenous versioned policy manifest
│   └── ablations/                    # A1–A8 ablation YAML overrides
├── src/
│   ├── data/                         # BraTS dataset, partitions, slice sampler, augmentations
│   ├── models/                       # UnimodalEncoder, SubsetFusionHead, UNetDecoder, PrototypeBank
│   ├── governance/                   # PolicyManager, ProvenanceLedger, LineageAuditor
│   ├── federation/                   # FederatedClient, FederatedServer, PhaseController
│   ├── losses.py                     # InfoNCE + Soft Dice+CE
│   ├── metrics.py                    # 3D Dice, HD95, PatientEvaluator
│   ├── statistics.py                 # 10,000 PCG64 bootstrap CIs, paired contrasts
│   ├── seed.py                       # Deterministic seed manager
│   └── logging.py                    # CSV loggers & CheckpointManager
├── scripts/
│   ├── run_primary.py                # Main primary experiment runner
│   ├── run_ablations.py              # Ablation studies runner
│   ├── generate_tables.py            # Paper LaTeX table generator
│   ├── generate_figures.py           # Publication PDF/PNG plot generator
│   └── verify_pre_experiment_gates.py# Pre-experiment gates audit
├── tests/                            # 45 PyTest sanity unit tests
└── outputs/                          # Logs, checkpoints, results, tables, figures
```

---

## 5. Specification & Research References

* **Research Spec:** [M1_CAMFS_Research_Implementation_Spec.md](file:///home/harsh/Research/Camfs/CAMFS_M1_Unified/M1_Implementation_spec_doc/M1_CAMFS_Research_Implementation_Spec.md)
* **Code Standards:** [m1_research_code_standards.md](file:///home/harsh/Research/Camfs/CAMFS_M1_Unified/M1_Implementation_spec_doc/m1_research_code_standards.md)
