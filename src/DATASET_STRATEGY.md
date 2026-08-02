# CAMFS M1 — Dataset Strategy & Implementation Execution Roadmap

> **Authoritative Specification:** [M1_CAMFS_Research_Implementation_Spec.md](../M1_Implementation_spec_doc/M1_CAMFS_Research_Implementation_Spec.md)  
> **Engineering Standards:** [m1_research_code_standards.md](../M1_Implementation_spec_doc/m1_research_code_standards.md)  
> **Date:** August 2026  

---

## Executive Summary

To ensure maximum engineering reliability, zero gradient leakage, and complete deterministic auditability before committing large compute resources, the **CAMFS M1** implementation follows a strict **two-stage dataset execution strategy**:

1. **Stage 1 — Pipeline Development, Audit & Pilot (BraTS 2020)**
   * **Dataset:** Local `dataset/BraTS2020_TrainingData.zip` (369 multi-parametric MRI training cases).
   * **Objective:** Build and audit the complete end-to-end software pipeline (data loading, preprocessing, unimodal encoders, contrastive prototypes, policy compiler, lineage ledger, FL server/client, 3D evaluation metrics, and PyTest sanity suite).
   * **Verification Gate:** Confirm Phase 1 contrastive convergence, Phase transition hard freeze, Phase 2 track-isolated segmentation, and zero accepted lineage audit violations.

2. **Stage 2 — Full Production Experiments (BraTS 2021)**
   * **Dataset:** Official BraTS 2021 dataset (1,251 labeled patient volumes).
   * **Objective:** Execute all 9 primary replication runs (3 partition seeds × 3 training seeds), 6 baseline comparisons (Local-Only, Blind FedAvg, DisentAFL, Hard Cohorts, FedAMM, Centralized Oracle), and 8 ablation studies (A1–A8).
   * **Switch Mechanism:** Seamlessly transition via single parameter update in `configs/default.yaml` (`dataset.name: "BraTS2021"`, `expected_labeled_patients: 1251`).

---

## Detailed Execution Steps

### Stage 1: BraTS 2020 Pilot & Pipeline Audit

| Milestone | Objective | Deliverables / Verification |
| :--- | :--- | :--- |
| **M1.1 — Config & Seeds** | Initialize environment and seed manager | `configs/default.yaml` (set for BraTS 2020), `src/seed.py`, `src/config.py` |
| **M1.2 — Preprocessing & Data** | Extract 2D axial slices, Z-score norm, 1:1 slice sampler | `src/data/brats_dataset.py`, `src/data/partition.py`, `src/data/slice_sampler.py`, `src/data/augmentation.py` |
| **M1.3 — Model Components** | 2D UNet encoders, fusion heads, decoders, prototypes | `src/models/encoder.py`, `src/models/fusion.py`, `src/models/decoder.py`, `src/models/prototypes.py` |
| **M1.4 — Governance & Audit** | Policy engine, SHA-256 ledger, lineage auditor | `src/governance/policy.py`, `src/governance/ledger.py`, `src/governance/lineage_audit.py` |
| **M1.5 — FL Protocol** | Phase 1 contrastive, freeze controller, Phase 2 fusion | `src/federation/server.py`, `src/federation/client.py`, `src/federation/phase_controller.py` |
| **M1.6 — Sanity Test Suite** | 7 mandatory PyTest pre-experiment verification tests | Run `pytest tests/ -v` (data, freeze, policy, lineage, loss, protocol, determinism) |
| **M1.7 — Pilot Run Audit** | End-to-end pilot run on BraTS 2020 | Verify Phase 1 drift < 0.01, positive validation Dice, zero lineage violations |

---

### Stage 2: BraTS 2021 Main Benchmark Training

| Task | Scope | Output |
| :--- | :--- | :--- |
| **Primary Experiments** | 9 replication runs (3 partition seeds × 3 training seeds) | `outputs/results/aggregated/primary_results.csv` |
| **Baseline Suite** | 6 comparative baselines (B1–B6) | `outputs/results/aggregated/baseline_comparisons.csv` |
| **Ablation Suite** | 8 ablation conditions (A1–A8) | `outputs/results/aggregated/ablation_results.csv` |
| **Final Tables & Figures** | Paper-ready LaTeX tables & PDF figures | `scripts/generate_tables.py` outputs |

---

## Configuration Mapping

```yaml
# configs/default.yaml (Stage 1 Pilot Setting)
dataset:
  name: "BraTS2020"
  root: "dataset/BraTS2020_TrainingData"
  expected_labeled_patients: 369
  input_size: [240, 240]
  modalities: ["T1", "T1ce", "T2", "FLAIR"]
  label_remap: {0: 0, 1: 1, 2: 2, 4: 3}
```
