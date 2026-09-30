# CAMFS-M1 Experiment Execution Plan

> **Master lookup document.** Every run, parameter, result, and dependency is here. Check off items as you complete them.

---

## Quick Reference

| Property | Value |
|---|---|
| **Target venue** | IEEE JBHI (stretch: IEEE TMI) |
| **Dataset** | BraTS 2020 — 368 patients (355 withdrawn) |
| **Partition seed** | 1103 (primary), {2207, 3301} reserved for extended study |
| **Training seeds** | {17, 29, 43} |
| **Test set** | 50 patients, PCG64 seed 901, SHA-256: `d100e44…` |
| **GPU Allocation** | `--gres=gpu:ugpg_3g47gb:1` (H100 NVL 47 GB MIG slice) |
| **Statistical protocol** | 10,000 PCG64 bootstrap CIs (seed 8803), paired sign-flip tests, Holm-Bonferroni correction |
| **Total new training runs** | ~25 (+ 7 already completed) |

---

## Part A: What Has Already Been Completed

All runs below used **partition 1103, training seed 17**, evaluated on the **same 50 held-out test patients**.

### A.1 Completed Runs Inventory

| # | Run ID | Type | Status | Checkpoints | CSV rows |
|---|---|---|---|---|---|
| ✅ | `camfs_primary__part1103__seed17` | Primary | **DONE** | 9 `.pt` files | 200 patients |
| ✅ | `ablation_a1_jointtraining__part1103__seed17` | Ablation A1 | **DONE** | 6 `.pt` files | 200 patients |
| ✅ | `ablation_a2_delayedsite__part1103__seed17` | Ablation A2 | **DONE** | 7 `.pt` files | 200 patients |
| ✅ | `ablation_a3_lineageaudit__part1103__seed17` | Ablation A3 | **DONE** | 7 `.pt` files | 200 patients |
| ✅ | `ablation_a4_lambda1sweep__part1103__seed17__lam0.0` | Ablation A4 | **DONE** | 7 `.pt` files | 200 patients |
| ✅ | `ablation_a5_coldstart__part1103__seed17__tier1` | Ablation A5 | **DONE** | 7 `.pt` files | 200 patients |
| ✅ | `ablation_a7_multitrack__part1103__seed17` | Ablation A7 | **DONE** | 7 `.pt` files | 200 patients |

### A.2 Existing Results (Seed 17 Only — Macro Dice / HD95 on 50 Test Patients)

| Run | H1 (S1) Dice | H2 (Local) Dice | H3 (S3) Dice | H4 (S4) Dice |
|---|---|---|---|---|
| **CAMFS Primary** | **0.8143** | **0.7845** | **0.6544** | **0.8107** |
| A1 Joint Training | 0.8046 | 0.7724 | 0.6227 | 0.7811 |
| A2 Delayed Site | 0.8208 | 0.7887 | 0.6518 | 0.8021 |
| A3 Lineage Audit | 0.8107 | 0.7902 | 0.6554 | 0.8103 |
| A4 λ₁=0 Sweep | 0.8181 | 0.7882 | 0.6638 | 0.8103 |
| A5 Cold Start | 0.8107 | 0.7894 | 0.6554 | 0.8223 |
| A7 Multi-Track | 0.8209 | 0.7826 | 0.6506 | 0.8059 |

**Per-class Dice for Primary (seed 17):**

| Track | ET | TC | WT | Macro |
|---|---|---|---|---|
| S1/H1 | 0.7373 | 0.8123 | 0.8933 | 0.8143 |
| H2_local | 0.7001 | 0.7894 | 0.8640 | 0.7845 |
| S3/H3 | 0.4209 | 0.6612 | 0.8812 | 0.6544 |
| S4/H4 | 0.7329 | 0.8268 | 0.8723 | 0.8107 |

> **Note:** These ablation results are NOT in the minimum publication set. They are kept as pre-computed answers for reviewer questions during revision. The primary seed-17 result IS directly used as 1 of 3 seeds in the main results table.

### A.3 Status of Existing Runs for Publication Use

| Run | Role in final paper |
|---|---|
| `camfs_primary__part1103__seed17` | ✅ **Directly used** — 1st of 3 seeds in main comparison table |
| `ablation_a1_jointtraining` | 🔄 Reserve — shows joint training hurts (reviewer Q&A material) |
| `ablation_a2_delayedsite` | 🔄 Reserve — delayed site joining dynamics |
| `ablation_a3_lineageaudit` | 🔄 Reserve — consent audit mechanism verification |
| `ablation_a4_lambda1sweep` | 🔄 Reserve — λ₁ insensitivity under AdamW |
| `ablation_a5_coldstart` | 🔄 Reserve — Net2Net initialization vs. cold start |
| `ablation_a7_multitrack` | 🔄 Reserve — multi-track contribution toggle |

---

## Part B: Shared Configuration (All New Runs Must Use These)

### B.1 Data Split

```
BraTS 2020: 368 patients
├── Fixed Test Set: 50 patients (PCG64 seed 901) — NEVER in any training/validation
└── Federation Pool: 318 patients (PCG64 seed 1103)
    ├── H1 (40%): 127 patients → 102 train / 25 val
    ├── H2 (25%):  80 patients →  64 train / 16 val
    ├── H3 (20%):  64 patients →  52 train / 12 val
    └── H4 (15%):  47 patients →  38 train /  9 val
```

### B.2 Hospital Modality Configuration

| Hospital | Owned | Shared (Consent) | Fed. Tracks |
|---|---|---|---|
| H1 | T1, T1ce, T2, FLAIR | all 4 | S1, contributes to S2, S3, S4 |
| H2 | T1, T1ce, T2 | T1, T2 only | S2, private local head for T1ce |
| H3 | T1, FLAIR | both | S3 |
| H4 | T1, T1ce, T2 | all 3 | S4, contributes to S2 |

### B.3 Model Architecture Parameters

| Parameter | Value | Source |
|---|---|---|
| Encoder channels | [32, 64, 128, 256, 256] | `configs/default.yaml` |
| GroupNorm groups | 8 | `configs/default.yaml` |
| Activation | SiLU | `configs/default.yaml` |
| Init | Kaiming Normal (fan_out, relu) | `configs/default.yaml` |
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=1e-8, wd=1e-4) | `configs/default.yaml` |
| Dice smoothing | 1e-5 | `configs/default.yaml` |

### B.4 Training Parameters

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| Learning rate | 3e-4 | 1e-3 |
| Loss weight (λ) | λ₁ = 1.0 | λ₂ = 0.1 |
| Temperature (τ) | 0.1 | 0.1 |
| Min rounds | 20 | 20 |
| Max rounds | 100 | 100 |
| Local epochs per round | 1 | 1 |
| Early stopping patience | 5 (drift) | 10 (improvement) |
| Improvement threshold | ε = 0.01 (drift) | 1e-4 |

### B.5 Evaluation Protocol (Strictly Unified Across ALL Runs)

> **Mandatory Benchmark Invariance:** Every single experiment without exception — CAMFS-M1, Local-Only (RUN-1), FedAvg (RUN-2), FedAMM (RUN-4), Centralized Oracle (RUN-5), and all Ablations (RUN-6–10) — **must use the exact same evaluation pipeline** (`src/metrics.py::PatientEvaluator` invoked via `src/experiment_utils.py`). No method is permitted to use an external or custom metric calculation script.

| Parameter | Value | Details |
|---|---|---|
| **Test set** | 50 patients | H3 pool partition, fixed PCG64 seed 901, zero test contamination |
| **Volumetric Reconstruction** | 155 slices → $(155, 240, 240)$ | All 2D slice predictions are stacked into the full 3D volume before `argmax` |
| **Statistical unit** | **Patient-level 3D volume** | Metrics are evaluated over the 3D volume, NOT averaged across 2D slices |
| **Target Regions** | WT ({1,2,4}), TC ({1,4}), ET ({4}) | Standard official BraTS Challenge hierarchy |
| **Metrics Computed** | 3D Dice & 3D HD95 | Per-region (WT, TC, ET) + Macro average across regions |
| **Physical Spacing** | $(1.0, 1.0, 1.0)\text{ mm}^3$ | Euclidean distance transform via `scipy.ndimage.distance_transform_edt` |
| **Boundary Penalties** | 373.1 mm for empty/non-empty mismatch | Bounded grid diagonal penalty ($\sqrt{240^2 + 240^2 + 155^2}$ mm) |
| **Evaluation Codebase** | `src/metrics.py` (`PatientEvaluator`) | Shared single-source-of-truth evaluator |
| **Output format** | Per-patient CSV | Columns: `patient_id, track_id, hospital_id, dice_ET, dice_TC, dice_WT, dice_macro, hd95_ET, hd95_TC, hd95_WT, hd95_macro` |

> [!NOTE]
> **2D Training ↔ 3D Evaluation Compatibility:** Although local hospital training operates on 2D axial slices (for computational feasibility on distributed clinical nodes and robust mini-batch gradient estimation), evaluation is strictly 3D volumetric: all 155 slice predictions are reconstructed into the full $(155 \times 240 \times 240)$ coordinate space before any metric computation. This matches the official BraTS Challenge evaluation protocol and ensures all reported Dice/HD95 numbers are directly comparable to any 3D-native method evaluated on the same test set.

### B.6 Reproducibility Settings

| Setting | Value |
|---|---|
| `torch.use_deterministic_algorithms` | True (warn_only for bilinear upsample) |
| TF32 | Disabled |
| Precision | FP32 |
| Bilinear upsampling | **Non-deterministic on GPU** (known, documented limitation) |

### B.7 Fault-Tolerance & Checkpoint Auto-Resumption Protocol

> **Zero Compute Waste on Slurm:** Cluster jobs can be interrupted at any time (hitting the `--time` limit, node failure, or preemption by higher-priority queues). The codebase implements an **atomic checkpointing and auto-resumption mechanism** across all runs.

| Feature | Mechanism | Behavior on Interruption |
|---|---|---|
| **Checkpoint Frequency** | Every round / epoch | Saved to `outputs/checkpoints/<run_id>/` |
| **Atomic Writing** | Save to `.tmp` then rename | Prevents checkpoint corruption if Slurm kills the job mid-write |
| **Phase 1 State** | `phase1_latest.pt` | Stores encoder weights, contrastive optimizer states, and bootstrapped prototypes |
| **Phase 1 Completion** | `phase1_frozen.pt` | When Phase 1 finishes, encoders are frozen; future resumes skip Phase 1 entirely |
| **Phase 2 State** | `phase2_latest.pt` & `best_track_*.pt` | Stores fusion head weights, decoder weights, and best validation Dice per track |
| **Resumption Action** | **Re-submit identical command** | The script automatically detects existing checkpoints and resumes from the exact interrupted round |
| **Baseline Requirement** | Mandatory | All newly built baseline scripts (`run_local_only.py`, `run_fedavg.py`, etc.) must implement this identical resumption protocol |

---

## Part C: New Runs Required (The Minimum Publication Set)

### Phase 0: Groundwork (No Training — Do First)

These are prerequisite tasks. No GPU time required.

---

#### GW-1: Config Reconciliation

- [ ] **DONE** · Result: ___

**What:** Verify that `configs/default.yaml` + `configs/policy_M1_PRIMARY_V1.json` match what the code actually used for the seed-17 primary run. Produce a single authoritative `configs/m1_primary.yaml`.

**Why blocking:** If τ, loss weights, round schedule, or encoder depth differ between spec/code/PDF, a reviewer who checks will reject.

**Steps:**
1. Check `τ` in code vs. config (spec says 0.1, check actual)
2. Check Phase 2 loss function (Dice+CE? Pure Dice?)
3. Check encoder depth matches `[32, 64, 128, 256, 256]`
4. Check slice sampling (all 155? subset?)
5. Write findings into `configs/m1_primary.yaml`
6. Log SHA-256 of final config in ledger

**Known discrepancy to resolve:** `default.yaml` says `expected_labeled_patients: 369` but actual dataset has 368.

---

#### GW-2: Time One Run + Check Determinism

- [ ] **DONE** · Result: ___ hours/run

**What:** Record wall-clock time for one full run (Phase 1 + Phase 2 + H2 head). Verify that the non-determinism from bilinear upsampling is within acceptable bounds.

**Why blocking:** You need to know total compute budget before committing to 25 runs. Also: A3 moved S1 by 0.4 Dice points and S3 ET by 3 points vs. Primary despite being an identical run — this variance needs to be understood.

**Steps:**
1. Search code for `use_deterministic_algorithms`, check if `warn_only=True`
2. Note expected variance range from seed-to-seed spread
3. Record hours for Phase 1, Phase 2, and H2 local head separately

**Expected:** ~X hours per full run on H100 (fill in after measurement).

---

#### GW-3: Per-Patient CSV Pipeline + Penalty Fix

- [ ] **DONE** · Result: ___

**What:** Ensure every future run outputs a per-patient CSV with all required columns. Fix HD95 penalty from whatever current value to 371.4 mm.

**Why blocking:** Without per-patient CSVs you cannot compute bootstrap CIs (the statistics step). Without correct penalty, HD95 numbers are wrong.

**Steps:**
1. Check current HD95 penalty value in code
2. Compute correct penalty: `sqrt(240² + 240² + 155²) = 371.4 mm`
3. Verify CSV output format matches §B.5 above
4. If existing seed-17 results used wrong penalty, decide: recompute from saved predictions or re-evaluate from checkpoint

---

### Phase 1: Core Comparison Table (23 Training Runs)

This is the heart of the paper — the main results table.

---

#### RUN-1: CAMFS Primary — Seeds 29 & 43

| Property | Value |
|---|---|
| **Runs needed** | 2 |
| **Script** | `python scripts/run_primary.py` |
| **What changes from seed 17** | Only `--train-seed` |
| **Depends on** | GW-1, GW-2, GW-3 |
| **Purpose** | Complete the 3-seed set for CAMFS. Without 3 seeds → no error bars → no publication. |

**Exact commands:**
```bash
# Interactive / Direct Execution:
# Seed 29
python scripts/run_primary.py \
  --config configs/default.yaml \
  --policy-config configs/policy_M1_PRIMARY_V1.json \
  --partition-seed 1103 \
  --train-seed 29 \
  --gpu 0 \
  --max-gpu-memory-gb 75.0

# Seed 43
python scripts/run_primary.py \
  --config configs/default.yaml \
  --policy-config configs/policy_M1_PRIMARY_V1.json \
  --partition-seed 1103 \
  --train-seed 43 \
  --gpu 0 \
  --max-gpu-memory-gb 75.0
```

**Slurm H100 Submission:**
```bash
# Seed 29 (submits via master sbatch script with mtech QOS time limit 03:59:00)
sbatch --job-name=camfs_s29 \
  scripts/submit_job.sbatch python scripts/run_primary.py \
    --config configs/default.yaml \
    --policy-config configs/policy_M1_PRIMARY_V1.json \
    --partition-seed 1103 \
    --train-seed 29 \
    --gpu 0 \
    --max-gpu-memory-gb 75.0

# Seed 43
sbatch --job-name=camfs_s43 \
  scripts/submit_job.sbatch python scripts/run_primary.py \
    --config configs/default.yaml \
    --policy-config configs/policy_M1_PRIMARY_V1.json \
    --partition-seed 1103 \
    --train-seed 43 \
    --gpu 0 \
    --max-gpu-memory-gb 75.0
```

**Expected output directories:**
- `outputs/results/camfs_primary__part1103__seed29/`
- `outputs/results/camfs_primary__part1103__seed43/`

**Tracking:**
- [ ] Seed 29 — started: ___ · finished: ___ · wall-clock: ___ hrs
  - H1 Dice: ___ · H2 Dice: ___ · H3 Dice: ___ · H4 Dice: ___
- [ ] Seed 43 — started: ___ · finished: ___ · wall-clock: ___ hrs
  - H1 Dice: ___ · H2 Dice: ___ · H3 Dice: ___ · H4 Dice: ___
- [ ] Both per-patient CSVs verified

---

#### RUN-2: Local-Only Baseline (Each Hospital Alone)

| Property | Value |
|---|---|
| **Runs needed** | 12 (4 hospitals × 3 seeds) |
| **Script** | **NEEDS TO BE BUILT** — no existing script |
| **Depends on** | GW-1, GW-3 |
| **Purpose** | Existential test: does federation help at ALL? If local-only ties or beats CAMFS on any hospital, the paper is dead. |

**What this run does:**
- Each hospital trains the CAMFS encoder+decoder architecture **alone**, using only its own data
- No federation, no communication, no aggregation
- H2 **includes T1ce** in local training (local use is allowed by consent policy)
- Model selection on hospital's own validation split
- Evaluate on the same 50 test patients

**Parameters per hospital:**

| Hospital | Modalities used | Train patients | Val patients | Architecture |
|---|---|---|---|---|
| H1 | T1, T1ce, T2, FLAIR | 102 | 25 | Full encoder+fusion+decoder |
| H2 | T1, T1ce, T2 | 64 | 16 | 3-modality encoder+fusion+decoder |
| H3 | T1, FLAIR | 52 | 12 | 2-modality encoder+fusion+decoder |
| H4 | T1, T1ce, T2 | 38 | 9 | 3-modality encoder+fusion+decoder |

**Training config:** AdamW optimizer, stable learning rate (3e-4) with gradient clipping (`torch.nn.utils.clip_grad_norm_(max_norm=1.0)`), CosineAnnealingLR, SoftDiceCrossEntropyLoss, max 100 epochs with patience 20. Decomposed into dedicated scripts per hospital to prevent interference (see [`reports/modular_experiment_runners_plan.md`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/reports/modular_experiment_runners_plan.md)).

**Exact commands (via dedicated modular runners):**
```bash
# Interactive / Direct Execution:
python scripts/run2_local_only/run_local_h1.py --partition-seed 1103 --train-seed 17 --gpu 0
python scripts/run2_local_only/run_local_h2.py --partition-seed 1103 --train-seed 17 --gpu 0
python scripts/run2_local_only/run_local_h3.py --partition-seed 1103 --train-seed 17 --gpu 0
python scripts/run2_local_only/run_local_h4.py --partition-seed 1103 --train-seed 17 --gpu 0
```

**Slurm H100 Submission:**
```bash
# Submit each hospital via its dedicated .sbatch script:
sbatch scripts/run2_local_only/submit_h1.sbatch 17
sbatch scripts/run2_local_only/submit_h2.sbatch 17
sbatch scripts/run2_local_only/submit_h3.sbatch 17
sbatch scripts/run2_local_only/submit_h4.sbatch 17

# Batch dispatch helper for all seeds:
bash scripts/run2_local_only/submit_all_seeds.sh
```

**Tracking:**

| Hospital | Seed 17 Dice | Seed 29 Dice | Seed 43 Dice | Mean ± CI |
|---|---|---|---|---|
| H1 | [ ] ___ | [ ] ___ | [ ] ___ | ___ |
| H2 | [ ] ___ | [ ] ___ | [ ] ___ | ___ |
| H3 | [ ] ___ | [ ] ___ | [ ] ___ | ___ |
| H4 | [ ] ___ | [ ] ___ | [ ] ___ | ___ |

**Key question this answers:** "For each hospital, is CAMFS Dice > Local-Only Dice?"

---

#### RUN-3: FedAvg Compliant Baseline

| Property | Value |
|---|---|
| **Runs needed** | 3 (seeds 17, 29, 43) |
| **Script** | **NEEDS TO BE BUILT** — no existing script |
| **Depends on** | GW-1, GW-3 |
| **Purpose** | Simplest federated baseline. Every FL paper compares against FedAvg. |

**What this run does:**
- Uses the **same CAMFS network architecture** (same encoder, same decoder, same parameter count)
- All 4 input slots always present; missing modalities are **zero-filled**
- **Compliant:** H2's T1ce channel is zero-filled during training (nothing H2 uploads is computed from T1ce)
- Standard FedAvg aggregation: weighted average of all hospital models each round
- At test time: H2 feeds in its own T1ce (nothing leaves the hospital)
- Single global model, no track isolation

**Parameters:**

| Parameter | Value |
|---|---|
| Architecture | Same U-Net encoder+decoder as CAMFS |
| Input | 4 channels (T1, T1ce, T2, FLAIR); missing = zero |
| Aggregation | FedAvg (patient-count-weighted) |
| LR | 1e-3 (same as CAMFS Phase 2) |
| Rounds | 100 max, patience 10 |
| Local epochs | 1 per round |
| H2 T1ce at train time | Zeroed out (compliant) |
| H2 T1ce at test time | Present (local use allowed) |

**Exact commands (once script is built):**
```bash
# Interactive / Direct Execution:
python scripts/run_fedavg.py \
  --partition-seed 1103 \
  --train-seed 17 \
  --gpu 0 \
  --max-gpu-memory-gb 75.0
# Repeat for seeds 29, 43
```

**Slurm H100 Submission:**
```bash
# Seed 17 (submits via master sbatch script with mtech QOS time limit 03:59:00)
sbatch --job-name=fedavg_s17 \
  scripts/submit_job.sbatch python scripts/run_fedavg.py \
    --partition-seed 1103 \
    --train-seed 17 \
    --gpu 0 \
    --max-gpu-memory-gb 75.0

# Repeat for seeds 29, 43 by updating --job-name and --train-seed
```

**Tracking:**
- [ ] Seed 17: H1=___ H2=___ H3=___ H4=___
- [ ] Seed 29: H1=___ H2=___ H3=___ H4=___
- [ ] Seed 43: H1=___ H2=___ H3=___ H4=___

**Key question this answers:** "Does track-isolated CAMFS beat naive FedAvg under the same consent constraints?"

---

#### RUN-4: FedAMM Compliant Baseline

| Property | Value |
|---|---|
| **Runs needed** | 3 (seeds 17, 29, 43) |
| **Script** | **NEEDS TO BE BUILT** — adapt from official FedAMM code |
| **Source code** | `github.com/13sky/FedAMM` |
| **Depends on** | GW-1, GW-3 |
| **Purpose** | Primary SOTA comparison — cited in introduction as closest prior work. |

**What this run does:**
- Implements **FedAMM's core algorithmic contributions** (intra-client prototype distillation $\mathcal{L}_{mb}$, inter-client prototype clustering $\mathcal{L}_{mc}$, and modality-weighted encoder aggregation $\omega_k^m$) adapted onto our unified 2D benchmark backbone.
- Evaluated strictly using our **unified CAMFS-M1 3D volumetric pipeline** (`src/metrics.py::PatientEvaluator` over stacked 155 slices on the exact same 50 test patients).
- **Compliant:** H2's T1ce treated as missing during federated training.
- **Why this adaptation is critical:** In the original MICCAI paper, FedAMM used a 3D volumetric RFNet backbone. Porting FedAMM's FL algorithms to our unified 2D benchmark isolates the federated learning mechanism from 3D-vs-2D architectural capacity differences, providing a rigorous, unassailable apples-to-apples baseline.

**Parameters:**

| Parameter | Value |
|---|---|
| Method logic | Official FedAMM (Prototype distillation + Modality-weighted aggregation) |
| Backbone | Unified 2D multi-encoder (same parameter capacity as CAMFS) |
| Data loader | Ours (same partition, same 2D preprocessing) |
| Evaluation | **Ours strictly** (`PatientEvaluator`, same 50 test patients, 155-slice stacked 3D Dice/HD95) |
| H2 T1ce at train time | Treated as missing (compliant) |
| H2 T1ce at test time | Present (local evaluation allowed) |
| Rounds/epochs | Match FedAMM paper defaults, document what they are |

**Exact commands (once adapted):**
```bash
# Interactive / Direct Execution:
python scripts/run_fedamm.py \
  --partition-seed 1103 \
  --train-seed 17 \
  --gpu 0 \
  --max-gpu-memory-gb 75.0
# Repeat for seeds 29, 43
```

**Slurm H100 Submission:**
```bash
# Seed 17 (submits via master sbatch script with mtech QOS time limit 03:59:00)
sbatch --job-name=fedamm_s17 \
  scripts/submit_job.sbatch python scripts/run_fedamm.py \
    --partition-seed 1103 \
    --train-seed 17 \
    --gpu 0 \
    --max-gpu-memory-gb 75.0

# Repeat for seeds 29, 43 by updating --job-name and --train-seed
```

**Tracking:**
- [ ] Seed 17: H1=___ H2=___ H3=___ H4=___
- [ ] Seed 29: H1=___ H2=___ H3=___ H4=___
- [ ] Seed 43: H1=___ H2=___ H3=___ H4=___
- [ ] Parameter count noted: ___

**Key question this answers:** "Does CAMFS outperform the closest SOTA method (FedAMM) under identical consent constraints?"

---

#### RUN-5: Centralized Ceiling (Oracle Upper Bound)

| Property | Value |
|---|---|
| **Runs needed** | 3 (one per modality config, or 3 seeds of 1 config) |
| **Script** | **NEEDS TO BE BUILT** — no existing script |
| **Depends on** | GW-1, GW-3 |
| **Purpose** | Defines the absolute ceiling. Required to compute the Compliance-Performance Gap metric. |

**What this run does:**
- Pool ALL training patients from all 4 hospitals (256 train patients total)
- Train one centralized model per modality configuration
- No federation, no privacy, no consent — all data in one place
- This is what you'd get if consent restrictions didn't exist

**Three modality configs needed:**

| Config | Modalities | Used to evaluate | Train patients |
|---|---|---|---|
| Full-4 | T1, T1ce, T2, FLAIR | H1 | 256 |
| 3-mod-A | T1, T1ce, T2 | H2, H4 | 256 |
| 2-mod | T1, FLAIR | H3 | 256 |

**Parameters:**

| Parameter | Value |
|---|---|
| Architecture | Same U-Net encoder+decoder as CAMFS |
| Optimizer | AdamW (same hyperparameters) |
| LR | 1e-3 |
| No federation | Direct training on pooled data |
| Validation | 20% of pooled patients |
| Seeds | Run each config once (3 total) OR run one config × 3 seeds |

**Exact commands (once script is built):**
```bash
# Interactive / Direct Execution:
python scripts/run_centralized.py \
  --modality-config full4 \
  --partition-seed 1103 \
  --train-seed 17 \
  --gpu 0 \
  --max-gpu-memory-gb 75.0
# Repeat for 3mod_a and 2mod configs
```

**Slurm H100 Submission:**
```bash
# Full-4 config
sbatch --job-name=oracle_full4 \
  scripts/submit_job.sbatch python scripts/run_centralized.py \
    --modality-config full4 \
    --partition-seed 1103 \
    --train-seed 17 \
    --gpu 0 \
    --max-gpu-memory-gb 75.0

# 3-mod config
sbatch --job-name=oracle_3mod \
  scripts/submit_job.sbatch python scripts/run_centralized.py \
    --modality-config 3mod_a \
    --partition-seed 1103 \
    --train-seed 17 \
    --gpu 0 \
    --max-gpu-memory-gb 75.0

# 2-mod config
sbatch --job-name=oracle_2mod \
  scripts/submit_job.sbatch python scripts/run_centralized.py \
    --modality-config 2mod \
    --partition-seed 1103 \
    --train-seed 17 \
    --gpu 0 \
    --max-gpu-memory-gb 75.0
```

**Tracking:**
- [ ] Full-4 (for H1): Dice=___
- [ ] 3-mod (for H2, H4): Dice=___
- [ ] 2-mod (for H3): Dice=___

**Key question this answers:** "How close does federated CAMFS get to the 'no privacy needed' oracle?"

---

### Phase 2: Consent Proof (2 Short Runs)

---

#### RUN-6: Consent Violation Audit

| Property | Value |
|---|---|
| **Runs needed** | 2 short (2–3 rounds each, not full training) |
| **Script** | Modify existing run to enable `image_lineage` tagging |
| **Depends on** | GW-3 |
| **Purpose** | **THE ENTIRE POINT OF THE PAPER.** Prove V=0 for CAMFS and V>0 for FedAvg. |

**What this run does:**
- Tag every weight update with `image_lineage` — which patient's data from which hospital contributed
- Run CAMFS for 2–3 rounds: count how many times a weight update that used H2's T1ce data appears in any model sent outside H2 → should be **V = 0**
- Run FedAvg for 2–3 rounds: count same → should be **V > 0** (because FedAvg's global model aggregates H2's T1ce-informed gradients and sends them to everyone)

**The measurement:**
- For CAMFS: trace every gradient update through the track system. H2 only contributes to S2 (T1, T2). T1ce never enters any shared track.
- For FedAvg: H2 trains on all owned modalities (including T1ce) → uploads model → server averages → sends to all hospitals. V > 0 by construction.

**Output:** A table like:

| Method | Violations (V) | Description |
|---|---|---|
| CAMFS | 0 | H2's T1ce never enters shared tracks |
| FedAvg | >0 | H2's T1ce gradients in global model |

**Exact commands:**
```bash
# Interactive / Direct Execution:
python scripts/run_primary.py \
  --config configs/default.yaml \
  --policy-config configs/policy_M1_PRIMARY_V1.json \
  --max-p1-rounds 3 \
  --max-p2-rounds 3 \
  --gpu 0

# Slurm H100 Submission:
sbatch --job-name=audit_camfs \
  --output=outputs/logs/slurm_audit_camfs_%j.out \
  --error=outputs/logs/slurm_audit_camfs_%j.err \
  --gres=gpu:ugpg_3g47gb:1 --cpus-per-task=8 --mem=64G --time=01:00:00 \
  --wrap="python scripts/run_primary.py --config configs/default.yaml --policy-config configs/policy_M1_PRIMARY_V1.json --max-p1-rounds 3 --max-p2-rounds 3 --gpu 0"
```

**Tracking:**
- [ ] CAMFS audit complete: V = ___
- [ ] FedAvg audit complete: V = ___

---

### Phase 3: Statistics (After All Runs Complete)

---

#### STAT-1: Bootstrap CIs + Hypothesis Tests

| Property | Value |
|---|---|
| **Runs needed** | 0 (computation only) |
| **Script** | `python scripts/generate_tables.py` (exists) |
| **Depends on** | ALL of Phase 1 + Phase 2 completed |
| **Purpose** | IEEE JBHI/TMI requires statistical testing. "Mean ± std across 3 seeds" without formal tests won't pass review. |

**Statistical protocol:**
- 10,000 PCG64 percentile-bootstrap resamples over patient identities (seed 8803)
- Paired sign-flip tests for CAMFS vs. each baseline
- Holm-Bonferroni correction for multiple comparisons
- Report: 95% CI, p-value, effect size

**What goes into the final table:**

| Method | H1 Dice [95% CI] | H2 Dice [95% CI] | H3 Dice [95% CI] | H4 Dice [95% CI] | V |
|---|---|---|---|---|---|
| CAMFS | x.xx [a, b] | x.xx [a, b] | x.xx [a, b] | x.xx [a, b] | 0 |
| FedAvg | x.xx [a, b] | x.xx [a, b] | x.xx [a, b] | x.xx [a, b] | >0 |
| FedAMM | x.xx [a, b] | x.xx [a, b] | x.xx [a, b] | x.xx [a, b] | >0 |
| Local-Only | x.xx [a, b] | x.xx [a, b] | x.xx [a, b] | x.xx [a, b] | — |
| Centralized | x.xx [—] | x.xx [—] | x.xx [—] | x.xx [—] | — |

**Tracking:**
- [ ] All CSVs collected from all runs
- [ ] `generate_tables.py` executed successfully
- [ ] CIs computed
- [ ] P-values computed
- [ ] Holm correction applied

---

## Part D: Master Run Table

Every run in one place — what it is, what it tests, and its current status.

| # | Run Name | Category | Seeds | Total Runs | What It Tests | Status |
|---|---|---|---|---|---|---|
| — | `camfs_primary__seed17` | Primary | 17 | 1 | CAMFS method | ✅ DONE |
| — | `ablation_a1` | Ablation | 17 | 1 | Joint training leaks | ✅ DONE (reserve) |
| — | `ablation_a2` | Ablation | 17 | 1 | Delayed site joining | ✅ DONE (reserve) |
| — | `ablation_a3` | Ablation | 17 | 1 | Lineage audit overhead | ✅ DONE (reserve) |
| — | `ablation_a4` | Ablation | 17 | 1 | λ₁ sensitivity | ✅ DONE (reserve) |
| — | `ablation_a5` | Ablation | 17 | 1 | Cold start vs Net2Net | ✅ DONE (reserve) |
| — | `ablation_a7` | Ablation | 17 | 1 | Multi-track contribution | ✅ DONE (reserve) |
| GW-1 | Config reconciliation | Groundwork | — | 0 | Config consistency | ⬜ TODO |
| GW-2 | Timing + determinism | Groundwork | — | 0 | Variance understanding | ⬜ TODO |
| GW-3 | CSV pipeline + penalty | Groundwork | — | 0 | Correct metrics | ⬜ TODO |
| RUN-1 | CAMFS seeds 29, 43 | Primary | 29, 43 | 2 | Error bars for CAMFS | ⬜ TODO |
| RUN-2 | Local-only | Baseline | 17, 29, 43 | 12 | Does federation help? | ⬜ TODO |
| RUN-3 | FedAvg compliant | Baseline | 17, 29, 43 | 3 | vs simplest FL method | ⬜ TODO |
| RUN-4 | FedAMM compliant | Baseline | 17, 29, 43 | 3 | vs closest SOTA | ⬜ TODO |
| RUN-5 | Centralized ceiling | Baseline | 17 | 3 | Upper bound | ⬜ TODO |
| RUN-6 | Consent audit | Proof | — | 2 | V=0 for CAMFS | ⬜ TODO |
| STAT-1 | Bootstrap + tests | Stats | — | 0 | Publication-grade stats | ⬜ TODO |
| | | | **TOTAL NEW** | **25** | | |

---

## Part E: Execution Order & Dependencies

### Dependency Graph

```
GW-1 (Config) ──┬──→ GW-2 (Timing) ──→ RUN-1 (CAMFS seeds 29, 43)
                 │
GW-3 (CSV fix) ─┼──→ RUN-2 (Local-Only, 12 runs)
                 │
                 ├──→ RUN-3 (FedAvg, 3 runs)      ──→ RUN-6 (Consent Audit)
                 │
                 ├──→ RUN-4 (FedAMM, 3 runs)
                 │
                 └──→ RUN-5 (Centralized, 3 runs)
                 
ALL RUNS COMPLETE ──→ STAT-1 (Bootstrap CIs + Tests)
```

### Suggested Execution Schedule

**Day 1 — No GPU (Groundwork)**
1. ⬜ GW-1: Config reconciliation (~1 hour)
2. ⬜ GW-3: CSV pipeline + penalty fix (~1–2 hours)
3. ⬜ GW-2: Time one run — start a dry-run to measure (~30 min)

**Day 2 — Start CAMFS Seeds (2 runs)**
4. ⬜ RUN-1a: `camfs_primary__part1103__seed29`
5. ⬜ RUN-1b: `camfs_primary__part1103__seed43`

> **STOP/GO GATE:** Compare seed 29 and 43 to seed 17. If macro Dice varies by >5 points across seeds for the same hospital, investigate before proceeding.

**Days 3–4 — Local-Only (12 runs, likely fastest)**
6. ⬜ RUN-2: All 12 local-only runs (4 hospitals × 3 seeds)
   - These are smaller (no federation overhead, fewer patients per run)
   - Can potentially batch multiple on same GPU sequentially

> **STOP/GO GATE:** If any hospital's local-only Dice ≥ CAMFS Dice (mean across 3 seeds), debug before proceeding.

**Days 5–6 — FedAvg + FedAMM (6 runs)**
7. ⬜ RUN-3: FedAvg × 3 seeds
8. ⬜ RUN-4: FedAMM × 3 seeds

**Day 7 — Centralized + Consent (5 runs)**
9. ⬜ RUN-5: Centralized × 3 configs
10. ⬜ RUN-6: Consent audit (2 short runs)

**Day 8 — Statistics + Table**
11. ⬜ STAT-1: Run `generate_tables.py` on all results
12. ⬜ Final paper table populated

> **Note:** This schedule assumes ~X hours per full run on H100. Adjust based on GW-2 timing measurement. If runs take 2 hours each, total GPU time ≈ 50 hours (roughly 3 days sequential).

---

## Part F: Scripts to Build

These scripts are organized into dedicated modular packages per experiment (see full architecture in [`reports/modular_experiment_runners_plan.md`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/reports/modular_experiment_runners_plan.md)):

| Experiment Module | Scripts & SLURM Files | Key characteristics |
|---|---|---|
| **RUN-2: Local-Only** | `scripts/run2_local_only/run_local_h{1..4}.py`<br>`scripts/run2_local_only/submit_h{1..4}.sbatch` | Dedicated runner per hospital. Modalities, data splits, and architecture hardcoded per hospital. Stable LR (3e-4) + gradient clipping (1.0). Zero cross-talk. |
| **RUN-3: FedAvg** | `scripts/run3_fedavg/run_fedavg.py`<br>`scripts/run3_fedavg/submit_fedavg_s{17,29,43}.sbatch` | Single global model, FedAvg aggregation, 4-channel input with zero-fill for missing modalities. Compliant H2 T1ce masking during federation. |
| **RUN-4: FedAMM** | `scripts/run4_fedamm/run_fedamm.py`<br>`scripts/run4_fedamm/submit_fedamm_s{17,29,43}.sbatch` | Official FedAMM prototype distillation + modality-weighted aggregation ported to our unified 2D multi-encoder backbone. |
| **RUN-5: Centralized** | `scripts/run5_centralized/run_centralized_{full4,3mod,2mod}.py`<br>`scripts/run5_centralized/submit_centralized_{full4,3mod,2mod}.sbatch` | Upper-bound oracle. Pools all 256 federation pool patients. Trains separate models for 4-modality, 3-modality, and 2-modality subsets. |
| **RUN-6: Consent Audit** | `scripts/run6_consent_audit/run_consent_audit.py`<br>`scripts/run6_consent_audit/submit_audit.sbatch` | Diagnostic lineage verification to prove empirical violation count $V=0$ for CAMFS and $V>0$ for FedAvg. |

---

## Part G: What Gets Cut (Safe to Defer)

| Cut item | Original plan # | Why it's safe | When to run it |
|---|---|---|---|
| Prototype cosine similarity | Item 5 | Remove "shared language" claim from text instead | If reviewer asks |
| FedMEPD / FedMEMA | Item 9 | FedAMM is primary SOTA comparison | Revision round |
| H2 alternatives (borrow S4, fine-tune) | Items 12–14 | Core table already shows H2's score | Revision round |
| All ablations at seeds 29, 43 | Items 16–21 | Seed-17 ablation results already banked | If reviewer requests multi-seed ablations |
| Check S4 init in ledger | Item 4 | Bookkeeping, not blocking | Anytime |

---

## Part H: Final Paper Table Template

After all runs and STAT-1 complete, the main results table should look like this:

### Table: Main Comparison on 50 Held-Out BraTS 2020 Test Patients

| Method | Params | V | H1 Dice | H2 Dice | H3 Dice | H4 Dice | Mean |
|---|---|---|---|---|---|---|---|
| Local-Only | — | — | _±_ | _±_ | _±_ | _±_ | _±_ |
| FedAvg | _M | >0 | _±_ | _±_ | _±_ | _±_ | _±_ |
| FedAMM | _M | >0 | _±_ | _±_ | _±_ | _±_ | _±_ |
| **CAMFS (ours)** | _M | **0** | _±_ | _±_ | _±_ | _±_ | _±_ |
| Centralized (oracle) | _M | — | _._ | _._ | _._ | _._ | _._ |

*Values are mean Macro Dice ± 95% bootstrap CI across 3 training seeds. V = consent violation count. Bold = best federated method.*

---

## Part I: Slurm H100 Cluster Execution Guide

This guide summarizes the cluster operational rules established from our Slurm configuration on the H100 node.

### I.1 Critical Cluster Parameters & Traps

| Parameter | Value | Critical Guidance |
|---|---|---|
| **Account & QOS Tier** | `Account=mtech`, `QOS=mtech` | Your user group is governed by QOS `mtech`. |
| **QOS Max Wall Limit** | `MaxWall = 04:00:00` (4 hours) | ⚠️ **DO NOT REQUEST >4 HOURS!** Any request $>04:00:00$ (e.g. `12:00:00`) is immediately blocked with `QOSMaxWallDurationPerJobLimit`. Always specify `--time=03:59:00`. |
| **Partition** | `--partition=h100` | Mandatory partition flag for the H100 GPU node. |
| **GPU Allocation** | `--gres=gpu:ugpg_3g47gb:1` | **MANDATORY MIG SLICE:** Always specify `--gres=gpu:ugpg_3g47gb:1` to explicitly bind to the designated 47 GB HBM3 / 42 SM slice on the H100 NVL. |
| **PyTorch VRAM Flag** | `--max-gpu-memory-gb 45.0` | **Mandatory on H100:** Scripts default to 20 GB. Set to `45.0` to match the 47 GB MIG slice. |
| **CPU & System RAM** | `--cpus-per-task=8 --mem=64G` | Ensures fast multi-worker slice loading without CPU bottlenecking. |
| **Handling Long Runs** | Auto-resumption | Runs exceeding 4 hours (e.g., CAMFS Primary) save checkpoints each round. When a 4-hour job finishes, re-submitting resumes from the exact saved round. |

> [!IMPORTANT]
> **MANDATORY CLUSTER RULE: Always Use `gres/gpu:ugpg_3g47gb:1`**
> From now on, all batch scripts, interactive sessions, and runner dispatches MUST specify `#SBATCH --gres=gpu:ugpg_3g47gb:1`. Never use generic `--gres=gpu:1` to prevent unpredictable scheduling onto non-designated slices.

---

### I.2 Reusable `.sbatch` Master Template

Saved at `scripts/submit_job.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=camfs_job
#SBATCH --output=outputs/logs/slurm_%x_%j.out
#SBATCH --error=outputs/logs/slurm_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ugpg_3g47gb:1
#SBATCH --partition=h100
#SBATCH --time=03:59:00
#SBATCH --mem=64G

# Navigate to project root
cd /home/harshyadav/Research/CAMFS/CAMFS_M1_Unified

# Environment initialization
mkdir -p outputs/logs outputs/checkpoints outputs/results

echo "=== Slurm Job ID: $SLURM_JOB_ID on $(hostname) ==="
echo "=== GPU Device: $CUDA_VISIBLE_DEVICES ==="
nvidia-smi

# Activate conda environment
source /home/shared/miniconda/etc/profile.d/conda.sh
conda activate camfs

# Execute training command passed via argument or script
"$@"
```

**Usage:**
```bash
sbatch scripts/submit_job.sbatch python scripts/run_primary.py --config configs/default.yaml --policy-config configs/policy_M1_PRIMARY_V1.json --partition-seed 1103 --train-seed 29 --gpu 0 --max-gpu-memory-gb 75.0
```

---

### I.3 Slurm Monitoring & Management Cheatsheet

```bash
# 1. View your active and queued jobs:
squeue -u $USER

# 2. View all currently running jobs across the cluster with GPU allocations:
squeue -t R -o "%.8i %.10u %.12P %.20j %.2t %.10M %.6D %10b %R"

# 3. Stream live output logs from a running job:
tail -f outputs/logs/slurm_<job_name>_<job_id>.out

# 4. Check live GPU utilization on your allocated compute node:
srun --jobid=<job_id> --pty nvidia-smi

# 5. Cancel a specific job:
scancel <job_id>

# 6. Cancel ALL your running and pending jobs:
scancel -u $USER
```

---

### I.4 Handling Slurm Timeouts, Preemption & Auto-Resuming

If a job is killed by Slurm (e.g., job hits time limit `DUE TO TIME LIMIT`, node preemption, or cluster maintenance):

1. **Check the checkpoint status:**
   ```bash
   ls -la outputs/checkpoints/<experiment_id>/
   ```
   You will see the latest saved checkpoints:
   - `phase1_latest.pt` (Phase 1 active training state)
   - `phase1_frozen.pt` (Phase 1 completed; frozen encoders)
   - `phase2_latest.pt` (Phase 2 active training state)
   - `best_track_S*.pt` (Best validation checkpoints per track)

2. **How to resume:**
   Simply **re-submit the exact same `sbatch` command**:
   ```bash
   sbatch scripts/submit_job.sbatch python scripts/run_primary.py \
     --config configs/default.yaml \
     --policy-config configs/policy_M1_PRIMARY_V1.json \
     --partition-seed 1103 \
     --train-seed 29 \
     --gpu 0 \
     --max-gpu-memory-gb 75.0
   ```
   The runner will automatically detect the checkpoint and output:
   ```
   === Found Frozen Phase 1 Checkpoint (phase1_frozen.pt). Skipping Phase 1 and Jumping Directly to Phase 2 ===
   === Resuming Phase 2 from Checkpoint (phase2_latest.pt) ===
   ```
   It will pick up right where it left off, restoring optimizer state, client prototypes, and best evaluation scores without restarting from round 0.

---

## Appendix: File Locations Quick Reference

| What | Path |
|---|---|
| Master config | `configs/default.yaml` |
| Policy config | `configs/policy_M1_PRIMARY_V1.json` |
| Partition manifests | `outputs/partitions/partition_{1103,2207,3301}.json` |
| Test set manifest | `outputs/partitions/h3_test_patients.json` |
| Preprocessed data | `outputs/preprocessed/` |
| Results (per run) | `outputs/results/<run_id>/` |
| Checkpoints (per run) | `outputs/checkpoints/<run_id>/` |
| Primary run script | `scripts/run_primary.py` |
| Ablation run script | `scripts/run_ablations.py` |
| Test set evaluator | `scripts/evaluate_pure_50_test_set.py` |
| Table generator | `scripts/generate_tables.py` |
| Figure generator | `scripts/generate_figures.py` |
| Statistics module | `src/statistics.py` |
| Partition generator | `src/data/partition.py` |
| Federation client | `src/federation/client.py` |
