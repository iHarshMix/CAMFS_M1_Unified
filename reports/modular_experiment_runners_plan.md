# Implementation Plan: Personalized Modular Experiment Runners for CAMFS-M1

> **Document Type:** Architecture & Implementation Plan  
> **Target Venue:** IEEE Journal of Biomedical and Health Informatics (JBHI) / IEEE Transactions on Medical Imaging (TMI)  
> **Governing Document:** [`reports/experiment_execution_plan.md`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/reports/experiment_execution_plan.md)  
> **Status:** Approved / In Progress

---

## 1. Goal & Motivation

### The Problem
During the initial test of RUN-2 (Local-Only Baseline), a single generic runner script ([`scripts/run_local_only.py`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/scripts/run_local_only.py)) was executed concurrently across hospitals using command-line arguments. Two critical issues emerged:
1. **Coupled Numerical Failure:** Training all modules end-to-end from scratch using generic Phase 2 hyperparameters (`lr = 1e-3` without gradient norm clipping) caused gradients to explode on Epoch 1, producing `NaN` losses across multiple hospital jobs simultaneously.
2. **Execution Coupling & Fragility:** When multiple hospitals or baselines share a monolithic script, diagnosing an issue or tweaking parameters for one hospital (e.g. H3 with 2 modalities and 52 patients vs H1 with 4 modalities and 102 patients) risks breaking or perturbing the others.

### The Objective
Restructure the experiment execution suite into **dedicated, personalized, self-contained scripts and matching Slurm submission files for each experiment** in the master execution plan:
- **Zero Interference:** Every run has dedicated, isolated checkpoints, logs, and evaluation metrics directories.
- **Personalized Hyperparameters:** Tailored learning rates, gradient clipping, schedulers, and modality configurations hardcoded per script.
- **One-Click Slurm Execution:** Pre-configured `.sbatch` scripts with correct partitions, memory allocations, and cluster wall-time limits (`--time=03:59:00` for QOS `mtech`).
- **Benchmark Invariance:** Strict adherence to the universal 3D evaluation standard ([`src/metrics.py::PatientEvaluator`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/src/metrics.py) over stacked 155 slices on the pure 50 test patients).

---

## 2. Architecture & Directory Organization

The experiment execution suite is organized under `scripts/` into specialized, isolated experiment packages matching Part C of [`reports/experiment_execution_plan.md`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/reports/experiment_execution_plan.md):

```
scripts/
├── run1_camfs_primary/
│   ├── run_primary.py                          # Existing primary runner (supports seeds 29, 43)
│   ├── submit_primary_s29.sbatch               # Pre-configured Slurm submission for Seed 29
│   └── submit_primary_s43.sbatch               # Pre-configured Slurm submission for Seed 43
│
├── run2_local_only/
│   ├── run_local_h1.py                          # Dedicated H1 runner (T1, T1ce, T2, FLAIR — 102 train)
│   ├── run_local_h2.py                          # Dedicated H2 runner (T1, T1ce, T2 — 64 train)
│   ├── run_local_h3.py                          # Dedicated H3 runner (T1, FLAIR — 52 train)
│   ├── run_local_h4.py                          # Dedicated H4 runner (T1, T1ce, T2 — 38 train)
│   ├── submit_h1.sbatch                         # Slurm submission: 1 GPU, 64GB RAM, time 03:59:00
│   ├── submit_h2.sbatch                         # Slurm submission: 1 GPU, 64GB RAM, time 03:59:00
│   ├── submit_h3.sbatch                         # Slurm submission: 1 GPU, 64GB RAM, time 03:59:00
│   ├── submit_h4.sbatch                         # Slurm submission: 1 GPU, 64GB RAM, time 03:59:00
│   └── submit_all_seeds.sh                     # Batch dispatch helper for seeds {17, 29, 43}
│
├── run3_fedavg/
│   ├── run_fedavg.py                           # Dedicated compliant FedAvg (zero-filled missing slots)
│   ├── submit_fedavg_s17.sbatch                # Slurm submission for Seed 17
│   ├── submit_fedavg_s29.sbatch                # Slurm submission for Seed 29
│   └── submit_fedavg_s43.sbatch                # Slurm submission for Seed 43
│
├── run4_fedamm/
│   ├── run_fedamm.py                           # Adapted FedAMM (prototype distillation & weighting)
│   ├── submit_fedamm_s17.sbatch                # Slurm submission for Seed 17
│   ├── submit_fedamm_s29.sbatch                # Slurm submission for Seed 29
│   └── submit_fedamm_s43.sbatch                # Slurm submission for Seed 43
│
├── run5_centralized/
│   ├── run_centralized_full4.py                # Centralized oracle on pooled data (4 modalities)
│   ├── run_centralized_3mod.py                 # Centralized oracle on pooled data (3 modalities)
│   ├── run_centralized_2mod.py                 # Centralized oracle on pooled data (2 modalities)
│   ├── submit_centralized_full4.sbatch
│   ├── submit_centralized_3mod.sbatch
│   └── submit_centralized_2mod.sbatch
│
└── run6_consent_audit/
    ├── run_consent_audit.py                    # Lineage tracking audit script (counts V)
    └── submit_audit.sbatch                     # 1-hour fast diagnostic run
```

---

## 3. Detailed Specifications per Experiment Module

### 3.1 RUN-2: Local-Only Baseline (`scripts/run2_local_only/`)

#### Distinct Characteristics & Personalization per Hospital
Each hospital trains completely standalone with its owned modalities. The script directly encapsulates the modalities, architecture, and split without reliance on fragile CLI overrides:

| Script | Hospital | Modalities | Architecture Config | Train / Val Patients |
|---|---|---|---|:---:|
| [`run_local_h1.py`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/scripts/run2_local_only/run_local_h1.py) | H1 | `T1`, `T1ce`, `T2`, `FLAIR` | 4 unimodal encoders + 4-modality fusion + UNet decoder | 102 / 25 |
| [`run_local_h2.py`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/scripts/run2_local_only/run_local_h2.py) | H2 | `T1`, `T1ce`, `T2` | 3 unimodal encoders + 3-modality fusion + UNet decoder | 64 / 16 |
| [`run_local_h3.py`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/scripts/run2_local_only/run_local_h3.py) | H3 | `T1`, `FLAIR` | 2 unimodal encoders + 2-modality fusion + UNet decoder | 52 / 12 |
| [`run_local_h4.py`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/scripts/run2_local_only/run_local_h4.py) | H4 | `T1`, `T1ce`, `T2` | 3 unimodal encoders + 3-modality fusion + UNet decoder | 38 / 9 |

#### Mandatory Numerical Stability Protocol
To prevent the `NaN` loss explosion observed in the earlier run:
1. **Learning Rate:** Set to `3e-4` (matching Phase 1 encoder rate in CAMFS specification) instead of `1e-3`.
2. **Gradient Norm Clipping:** Explicitly applied before every optimizer step:
   ```python
   torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
   ```
3. **Learning Rate Scheduler:** Cosine Annealing decay down to `1e-6` over the maximum 100 epochs:
   ```python
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
   ```
4. **Finite Loss Guard:** Validation check on loss tensor before stepping:
   ```python
   if not torch.isfinite(loss):
       print(f"Warning: Non-finite loss detected at epoch {epoch}, batch {b_idx}. Skipping step.", flush=True)
       optimizer.zero_grad()
       continue
   ```

#### Output Directory Isolation
Each hospital run script is assigned its own unique directory hierarchy:
- Checkpoints: `outputs/checkpoints/local_only_<H>__part1103__seed<S>/`
  - `latest_model.pt` (atomic auto-resume)
  - `best_model.pt` (highest validation macro Dice)
- Training Logs: `outputs/logs/local_only_<H>__part1103__seed<S>/training_metrics.csv`
- Test Results: `outputs/results/local_only_<H>__part1103__seed<S>/pure_50_test_patient_metrics.csv`
- Slurm Logs: `outputs/logs/slurm_local_<H>_s<S>_<jobid>.out` & `.err`

---

### 3.2 RUN-1: CAMFS Primary Multi-Seed (`scripts/run1_camfs_primary/`)
- Existing script: [`scripts/run_primary.py`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/scripts/run_primary.py) is already verified and tested for Seed 17.
- Dedicated Slurm scripts:
  - `submit_primary_s29.sbatch`: launches Seed 29 with `--partition-seed 1103 --train-seed 29 --max-gpu-memory-gb 75.0`
  - `submit_primary_s43.sbatch`: launches Seed 43 with `--partition-seed 1103 --train-seed 43 --max-gpu-memory-gb 75.0`
- Both jobs output to their own isolated directories:
  - `outputs/results/camfs_primary__part1103__seed29/`
  - `outputs/results/camfs_primary__part1103__seed43/`

---

### 3.3 RUN-3: Compliant FedAvg Baseline (`scripts/run3_fedavg/`)
- **Key Algorithmic Principle:**
  - Standard Federated Averaging (McMahan et al.) with unified 4-channel input (missing channels zero-filled).
  - **Consent Compliance:** During federated communication, H2 zero-fills its T1ce modality. At test time, H2 evaluates locally with all owned modalities present.
- **Dedicated Script:** `scripts/run3_fedavg/run_fedavg.py`
  - Manages client local epochs, weight averaging on server, model broadcast, and client evaluation.
  - Dedicated Slurm templates for seeds 17, 29, 43.

---

### 3.4 RUN-4: Adapted FedAMM Baseline (`scripts/run4_fedamm/`)
- **Key Algorithmic Principle:**
  - Implements FedAMM's intra-client prototype distillation $\mathcal{L}_{mb}$ and modality-weighted aggregation $\omega_k^m$.
  - Ported onto our unified 2D multi-encoder backbone to isolate the federated aggregation mechanism from 3D vs 2D capacity differences.
- **Dedicated Script:** `scripts/run4_fedamm/run_fedamm.py`
- Dedicated Slurm templates for seeds 17, 29, 43.

---

### 3.5 RUN-5: Centralized Ceiling Oracle (`scripts/run5_centralized/`)
- **Key Algorithmic Principle:**
  - Upper bound baseline: all 256 federation pool patients are pooled together on a single centralized server without federation.
- **Dedicated Scripts:**
  - `run_centralized_full4.py`: Pools all 4 modalities (upper bound ceiling for H1).
  - `run_centralized_3mod.py`: Pools T1, T1ce, T2 (upper bound ceiling for H2 and H4).
  - `run_centralized_2mod.py`: Pools T1, FLAIR (upper bound ceiling for H3).

---

### 3.6 RUN-6: Consent Violation Audit (`scripts/run6_consent_audit/`)
- **Key Algorithmic Principle:**
  - Traces gradient lineage over 3 rounds to empirically prove $V = 0$ for CAMFS-M1 and $V > 0$ for FedAvg.
- **Dedicated Script:** `scripts/run6_consent_audit/run_consent_audit.py`

---

## 4. Verification & Testing Plan

### 4.1 Automated Pre-Flight Sanity Checks (Local Dry-Run)
Before submitting multi-hour Slurm jobs:
1. Run a 1-epoch dry-run test for each script on 2 sample patients:
   ```bash
   python scripts/run2_local_only/run_local_h1.py --dry-run
   python scripts/run2_local_only/run_local_h2.py --dry-run
   python scripts/run2_local_only/run_local_h3.py --dry-run
   python scripts/run2_local_only/run_local_h4.py --dry-run
   ```
2. Verify:
   - Loss is a valid finite float (no `NaN` or `Inf`).
   - Gradient norms are within $[0.01, 1.0]$.
   - Checkpoints (`latest_model.pt` and `best_model.pt`) are written atomically.
   - Validation 3D Dice produces plausible non-zero numbers.
   - Test evaluation writes exactly 50 rows in `pure_50_test_patient_metrics.csv`.

### 4.2 Cluster Execution Plan (Immediate Phase)
1. **Clean corrupted outputs from failed jobs 6239 and 6240:**
   ```bash
   rm -rf outputs/logs/local_only_H* outputs/results/local_only_H* outputs/checkpoints/local_only_H*
   ```
2. **Submit RUN-2 (Seed 17) for all 4 hospitals:**
   ```bash
   sbatch scripts/run2_local_only/submit_h1.sbatch 17
   sbatch scripts/run2_local_only/submit_h2.sbatch 17
   sbatch scripts/run2_local_only/submit_h3.sbatch 17
   sbatch scripts/run2_local_only/submit_h4.sbatch 17
   ```
3. **Monitor first 5 epochs:** Ensure `Train Loss` decreases monotonically and `Val Dice` steadily climbs above 50% across all 4 hospitals.

---

## 5. Execution Status & Next Actions

- [x] Architecture design completed & saved to [`reports/modular_experiment_runners_plan.md`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/reports/modular_experiment_runners_plan.md)
- [ ] Build `scripts/run2_local_only/` (`run_local_h1.py`, `run_local_h2.py`, `run_local_h3.py`, `run_local_h4.py` + sbatch scripts)
- [ ] Clean invalid outputs from H1 and H2
- [ ] Run dry-run validation on H1..H4
- [ ] Submit H1, H2, H3, H4 (Seed 17) via Slurm
