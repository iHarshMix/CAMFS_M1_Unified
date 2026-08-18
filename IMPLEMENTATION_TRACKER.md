# CAMFS M1 — Complete Implementation Tracker & Progress Log

> **Source of truth:** [M1_CAMFS_Research_Implementation_Spec.md](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/M1_Implementation_spec_doc/M1_CAMFS_Research_Implementation_Spec.md), [M1_CAMFS_Unified_Architecture_Complete.md](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/M1_Implementation_spec_doc/M1_CAMFS_Unified_Architecture_Complete.md), and [CAMFS_M1_Complete_Specification.md](file:///home/harshyadav/.gemini/antigravity-ide/brain/23be6503-9e84-4368-863a-6dde390b9f1e/CAMFS_M1_Complete_Specification.md)
>
> **Current Stage:** Stage 1 Pilot on BraTS 2020 (368 patients) — Pipeline fully built, verified, and primary run evaluated on NVIDIA H100 GPU.
> **Next Stage:** Multi-seed primary runs, comparative baselines (B1–B6), and ablation campaign (A1–A8).
>
> **MANDATORY WORKING RULE:** After every step, experiment, or phase completed, log a detailed progress report in the "Execution Progress Log" section at the end of this document.

---

## Executive Status Dashboard

| Layer | Component | Status | Details |
| :--- | :--- | :---: | :--- |
| **Code Architecture** | Phases 1–10 Software Implementation | 🟢 **100% COMPLETE** | All 10 phases built, 45/45 PyTests passing, 9/9 pre-experiment verification gates passed. |
| **Primary Experiment** | Partition 1103, Seed 17 | 🟢 **COMPLETED & EVALUATED** | Phase 1 (100 rds) + Phase 2 Track FL + Post-Phase 2 H2 Local Head (Tier-1 Net2Net). |
| **Benchmark Evaluation** | Standardized Pure 50-Patient Test Set | 🟢 **COMPLETED & VERIFIED** | **`76.60%`** Macro Dice (**+6.89%** over FedAMM SOTA **`69.71%`**). |
| **Primary Campaign** | Multi-Seed / Multi-Partition Runs | 🟡 **1 / 9 COMPLETED** | Partition 1103, Seed 17 done. Remaining 8 partition/seed pairs pending. |
| **Ablation Studies** | 8 Ablation Experiments (A1–A8) | 🟡 **CODE READY / RUNS PENDING** | `run_ablations.py` & all 8 YAML configs verified. GPU training runs pending. |
| **Comparative Baselines** | 6 Baselines (B1–B6) | 🔴 **PENDING IMPLEMENTATION/RUNS** | Comparative runs against Local-Only, FedAvg, DisentAFL, RELIEF, FedAMM retrain, Oracle. |
| **Paper Outputs** | LaTeX Tables & Statistical Figures | 🟡 **SCRIPTS READY** | `generate_tables.py` & `generate_figures.py` built. Final aggregation awaits 9-run completion. |

---

## Master Implementation Checklist

### Phase 1 — Environment & Infrastructure Setup *(Code Standards §1–§4)*

- [x] Create Conda environment `camfs` with Python 3.10
- [x] Install PyTorch 2.5.1 + torchvision via `cu121` index
- [x] Install core packages: `numpy`, `nibabel`, `SimpleITK`, `scipy`, `scikit-image`, `pandas`, `matplotlib`, `tqdm`, `pyyaml`, `pytest`
- [x] Create human-maintained `environment.yml`
- [x] Export resolved `environment.lock.yml` via `conda env export --no-builds`
- [x] Export resolved `requirements.lock.txt` via `pip freeze`
- [x] Create runtime version dump script `scripts/verify_environment.py` (§1.3)
- [x] Verify runtime: Python 3.10.20, PyTorch 2.5.1+cu121, CUDA 12.1, cuDNN 90100, GPU detected
- [x] Initialize Git repository and push to GitHub remote (`main` branch)
- [x] Configure comprehensive `.gitignore` (raw data, outputs, caches, spec folder)
- [x] Create project directory structure matching §2 spec layout
- [x] Create master config `configs/default.yaml` with all §15 hyperparameters
- [x] Implement deterministic seed manager `src/seed.py` with `torch.use_deterministic_algorithms(True, warn_only=True)` (§4)
- [x] Implement YAML config loader `src/config.py` with CLI arg parser (§3)

---

### Phase 2 — Dataset Preprocessing Pipeline *(Spec §13.3–§13.4, Code Standards §5)*

- [x] Extract raw BraTS 2020 dataset via `scripts/unzip_dataset.py`
- [x] Implement NIfTI loader supporting `.nii` and `.nii.gz` in `src/data/brats_dataset.py`
- [x] Implement per-patient, per-modality z-score normalization on nonzero brain voxels with $\epsilon=10^{-8}$ floor
- [x] Implement intensity clipping to $[-5, 5]$, outside-brain stays 0, no spatial cropping
- [x] Implement label remapping $\{0,1,2,4\} \to \{0,1,2,3\}$
- [x] Implement canonical unaugmented `.npy` caching (155×240×240 float32 per modality, uint8 labels)
- [x] Implement memory-mapped lazy loader `BraTSDataset` via `np.load(path, mmap_mode="r")` with in-memory volume caching
- [x] Run `scripts/preprocess.py` — successfully cached 368/368 patients to `outputs/preprocessed/`
- [x] Implement PCG64 patient-disjoint partitioner in `src/data/partition.py` (§13.3)
- [x] Generate fixed universal held-out test set (50 patients, PCG64 seed 901)
- [x] Generate federation partitions for seeds $\{1103, 2207, 3301\}$ with SHA-256 digests
- [x] Run `scripts/make_partitions.py` — manifests saved to `outputs/partitions/`
- [x] Implement 1:1 Tumor:Non-Tumor 2D axial slice sampler in `src/data/slice_sampler.py` with class-level index caching
- [x] Implement GPU CUDA-accelerated co-registered augmentations (`F.grid_sample` & horizontal flip) in `src/data/augmentation.py`
- [x] Write and pass `tests/test_data_pipeline.py` — 4/4 tests passed

---

### Phase 3 — Neural Network Architectures *(Spec §5.1–§5.4, Code Standards §2)*

- [x] Create `src/models/__init__.py` package exporter
- [x] Implement `UnimodalEncoder` in `src/models/encoder.py` (§5.1)
  - [x] 5 downsampling levels: channels [32, 64, 128, 256, 256]
  - [x] Resolutions: 240→120→60→30→15 via 2×2 max-pool
  - [x] Double 3×3 ConvBlocks with 8-group GroupNorm and SiLU activation
  - [x] Kaiming-normal init with `mode="fan_out"`, `nonlinearity="relu"`, zero bias
  - [x] Return skip connections at all 5 levels ($h^{(1)}, h^{(2)}, h^{(3)}, h^{(4)}, z$)
- [x] Implement `PrototypeBank` in `src/models/prototypes.py` (§5.2, §6.2)
  - [x] Unimodal class prototype computation ($\text{Proto}_m^c \in \mathbb{R}^{256}$)
  - [x] Fused class prototype computation ($\text{FusedProto}_S^c \in \mathbb{R}^{256}$)
  - [x] $L_2$ normalization for cosine similarity
  - [x] Patient-support counting per class & zero-support masking
  - [x] Prototype aggregation with patient-weighted rule (§6.2)
- [x] Implement `SubsetFusionHead` in `src/models/fusion.py` (§5.4)
  - [x] Concatenate modality features in fixed universe order (`"T1", "T1ce", "T2", "FLAIR"`), skipping absent slots
  - [x] $1\times1$ convolution compression ($|S|\cdot C_r \to C_r$) + ConvBlock at each of 5 levels
  - [x] Support variable subset sizes ($S_1=4$ mods, $S_2'=2$ mods, $S_3=2$ mods, $S_4=3$ mods, $\text{Track}_{\text{local},2}=3$ mods)
- [x] Implement `net2net_widen_fusion_head` in `src/models/fusion.py` (§10.2 Tier 1)
  - [x] Net2Net channel-widening ($1\times1$ conv expansion with zero-initialized new modality slots)
  - [x] Function-preserving Day 0 initialization for $S_2' \to \text{Track}_{\text{local},2}$ and $S_2' \to S_4$
- [x] Implement `UNetDecoder` in `src/models/decoder.py` (§5.4)
  - [x] Start at fused bottleneck $z_S$ ($256\times15\times15$)
  - [x] Bilinear 2× upsampling with `align_corners=False`
  - [x] Skip connection concatenation with $f_S^{(4)}, f_S^{(3)}, f_S^{(2)}, f_S^{(1)}$
  - [x] ConvBlocks producing 256→128→64→32 channels
  - [x] Final $1\times1$ convolution to 4 class logits
- [x] Write and pass `tests/test_model_shapes.py` (5/5 tests passed)

---

### Phase 4 — Loss Functions & 3D Evaluation *(Spec §6.1, §8.2, §13.5, §15.1)*

- [x] Implement `src/losses.py`
  - [x] Masked InfoNCE contrastive loss with cosine similarity and $\tau=0.1$ (§6.1)
  - [x] Empty-prototype masking from denominator (§6.1)
  - [x] Dice + Cross-Entropy segmentation loss matching exact §15.1 formula
  - [x] Dice smoothing $\epsilon_D = 10^{-5}$, tumour classes $\{1,2,3\}$
  - [x] Phase 1 objective: $\lambda_1 \sum_m \mathcal{L}_{\text{uni}}(z_m)$ (§6.1)
  - [x] Phase 2 objective: $\mathcal{L}_{\text{Dice+CE}} + \lambda_2 \mathcal{L}_{\text{fused-align}}$ (§8.2)
- [x] Implement `src/metrics.py`
  - [x] 3D patient-level Dice for regions WT=\{1,2,4\}, TC=\{1,4\}, ET=\{4\} (§13.5)
  - [x] Both-empty=1, one-empty=0 Dice convention
  - [x] 3D HD95 surface distance in physical mm (§13.5)
  - [x] Both-empty HD95=0, one-empty HD95=grid diagonal penalty
  - [x] One-empty case counting
  - [x] Label back-mapping: model $\{0,1,2,3\} \to \text{BraTS } \{0,1,2,4\}$
  - [x] FP32 inference logits, argmax with lowest-class-index tie-breaking
- [x] Implement `scripts/evaluate_pure_50_test_set.py`
  - [x] Standardized head-to-head evaluation of all 4 hospital models on the universal 50-patient test set
  - [x] Per-patient 3D metric logging to CSV
- [x] Write and pass `tests/test_loss.py` (6/6 tests passed)

---

### Phase 5 — Governance, Policy & Provenance *(Spec §5.3, §5.6, §14.4, Code Standards §8)*

- [x] Create `configs/policy_M1_PRIMARY_V1.json` matching §5.3 tables exactly
  - [x] Encoder cohorts: $\kappa_{\text{T1}}$ (H1,H2,H3,H4), $\kappa_{\text{T1ce}}$ (H1,H4), $\kappa_{\text{T2}}$ (H1,H2,H4), $\kappa_{\text{FLAIR}}$ (H1,H3)
  - [x] Track cohorts: $S_1$ (H1), $S_2'$ (H2+H1,H4 support), $S_3$ (H3+H1 support), $S_4$ (H4+H1 support)
  - [x] $R_{\text{send}}, R_{\text{recv}}, R_{\text{send}}^{\text{track}}, R_{\text{recv}}^{\text{track}}, R_{\text{contribute}}$ matrices
  - [x] H2 private T1ce copy rules
- [x] Create `src/governance/__init__.py`
- [x] Implement `src/governance/policy.py` (§5.3)
  - [x] Policy manifest loader and SHA-256 digest computation
  - [x] Consent cohort compiler from pairwise policies
  - [x] Cohort closure verification: $\text{Closed}(m, \kappa)$ rule
  - [x] Track receive safety rule: $S \subseteq O(i)$ AND $i$ accepts every contributor
  - [x] Send-gated track routing (§9)
- [x] Implement `src/governance/ledger.py` (§5.6)
  - [x] Append-only SHA-256 chained JSONL provenance log
  - [x] Canonical JSON with stable key order and compact separators
  - [x] Cryptographic hash chaining: $h_k = \text{SHA256}(h_{k-1} \| \text{CanonicalJSON}(r_k))$
  - [x] Event types: `POLICY_MANIFEST`, `PHASE1_AGGREGATION`, `PHASE_TRANSITION`, `TRACK_CREATION`, `CHECKPOINT_SELECTED`, `H2_LOCAL_HEAD_SEED`, `H2_LOCAL_HEAD_TRAINED`
- [x] Implement `src/governance/lineage_audit.py` (§14.4)
  - [x] Packet-level `image_lineage` set tracking
  - [x] Encoder lineage rule: $\text{ImageLineage}(\theta_{E_m}) \subseteq \{m\}$
  - [x] Track lineage rule: $\text{ImageLineage}(\theta_{F_S}) \cup \text{ImageLineage}(\theta_{D_S}) \subseteq S$
  - [x] CAMFS mode: reject before deserialization
  - [x] B3 shadow mode: log violations without altering routing
- [x] Write and pass `tests/test_policy.py` (4/4 tests passed)
- [x] Write and pass `tests/test_lineage.py` (4/4 tests passed)

---

### Phase 6 — Federation Protocol & Phase Controller *(Spec §5.5, §6.2, §7, §8.4–§8.5)*

- [x] Create `src/federation/__init__.py`
- [x] Implement `src/federation/phase_controller.py` (§5.5, §7)
  - [x] State machine: `PHASE1` $\to$ `FROZEN` $\to$ `PHASE2` $\to$ `RELEASED`
  - [x] Freeze procedure: serialize & hash encoder state, set `eval()` mode, `requires_grad=False`
  - [x] Stop-gradient wrapper: $z_m = \text{detach}(E_m^*(x_m))$
  - [x] Destroy Phase 1 optimizer state and communication routes
  - [x] Create fresh Phase 2 optimizer
- [x] Implement `src/federation/client.py` (§6.2, §8.4)
  - [x] Phase 1 local training loop: contrastive InfoNCE on unimodal encoders (batch size 16)
  - [x] Phase 2 local training loop: Dice+CE + fused-alignment on fusion+decoder (batch size 16)
  - [x] Round 0 no-optimizer prototype bootstrap (§6.2)
  - [x] Fresh local AdamW optimizer each round (no state persists across rounds)
  - [x] Post-local prototype recomputation via GPU streaming accumulation
- [x] Implement `src/federation/server.py` (§6.2, §8.5)
  - [x] Phase 1 cohort aggregation: patient-weighted encoder averaging (§6.2)
  - [x] Phase 1 prototype aggregation: class-specific patient-support weighted (§6.2)
  - [x] Phase 2 track aggregation: send-keyed patient-weighted fusion/decoder averaging (§8.5)
  - [x] Phase 2 fused-prototype bootstrap and aggregation (§8.3)
  - [x] Track-cohort closure rule enforcement (§8.5)
  - [x] Zero-support prototype retention
  - [x] No server optimizer
- [x] Implement Phase 1 stopping criterion: drift $< \epsilon=0.01$ for $K=5$ rounds after round 20 (§6.3)
- [x] Implement Phase 2 model selection and stopping: highest validation macro Dice, patience 10 (§8.6)
- [x] Write and pass `tests/test_freeze.py` (2/2 tests passed)
- [x] Write and pass `tests/test_protocol.py` (3/3 tests passed)
- [x] Write and pass `tests/test_determinism.py` (1/1 tests passed)

---

### Phase 7 — Metrics Logging & Per-Round CSV *(Code Standards §6)*

- [x] Create `src/logging.py`
  - [x] `Phase1CSVLogger`: per-round CSV for Phase 1 contrastive training
  - [x] `Phase2CSVLogger`: per-round CSV for Phase 2 track fusion training
  - [x] `EvaluationCSVLogger`: patient-level 3D test evaluation CSV
- [x] Implement checkpoint saving strategy (§15.3)
  - [x] `outputs/checkpoints/{experiment_id}/phase1_latest.pt`
  - [x] `outputs/checkpoints/{experiment_id}/phase1_frozen.pt`
  - [x] `outputs/checkpoints/{experiment_id}/best_track_{track_id}.pt`
  - [x] `outputs/checkpoints/{experiment_id}/best_H2_local_head.pt`
  - [x] Atomic checkpoint write (`.tmp` $\to$ `os.replace`)
- [x] Write and pass `tests/test_logging.py` (4/4 tests passed)

---

### Phase 8 — Experiment Runner Scripts & Ablation Configurations *(Code Standards §7.2)*

- [x] Implement `scripts/run_primary.py`
  - [x] Full Phase 1 + Freeze + Phase 2 multi-track federated execution
  - [x] **Post-Phase 2 H2 Private Local Head Training Stage** (Tier-1 Net2Net channel-widening, early stopping, provenance logging)
  - [x] CLI options: `--config`, `--partition-seed`, `--train-seed`, `--gpu`, `--dry-run`, `--max-gpu-memory-gb`
- [x] Implement `scripts/run_ablations.py` supporting all 8 ablation studies:
  - [x] A1: Joint-training (no freeze)
  - [x] A2: Delayed-site context (H4 dynamic onboarding at $t=30$)
  - [x] A3: Executable lineage audit self-verification (reject mode)
  - [x] A4: $\lambda_1$ sweep $\{0, 0.1, 0.5, 1.0\}$
  - [x] A5: Cold-start variants (Tier 1 Net2Net, Tier 2 warm start, Tier 3 fresh init)
  - [x] A6: Group-symmetric vs directional T2 policy
  - [x] A7: Multi-track contribution toggle ($R_{\text{contribute}}(\text{H1}, S_3) = 1 \text{ vs } 0$)
  - [x] A8: H2 reconnection (private head vs pull-only $S_4$ load)
- [x] Create all 8 ablation config YAMLs in `configs/ablations/` (`a1`–`a8`)
- [x] Write and pass `tests/test_ablations.py` (9/9 tests passed)

---

### Phase 9 — Statistical Analysis & Paper Output *(Spec §14.5, Code Standards §11)*

- [x] Implement `src/statistics.py` (§14.5)
  - [x] 10,000 PCG64 percentile-bootstrap resamples over patient identities (seed 8803)
  - [x] Paired patient-level contrasts $d_{p,r,s}$ averaged over seeds then partitions
  - [x] Hierarchical run-aware sensitivity intervals (seed 8804)
  - [x] Per-hospital statistics for $H_1, H_2, H_3, H_4$
- [x] Implement `scripts/generate_tables.py` (LaTeX table outputs: `primary_results.tex`, `ablation_results.tex`, `lineage_audit_summary.tex`)
- [x] Implement `scripts/generate_figures.py` (Matplotlib publication figures in `outputs/figures/`)
- [x] Write and pass `tests/test_statistics.py` (4/4 tests passed)

---

### Phase 10 — Pre-Experiment Gates & Full Experiment Verification *(Spec §19.3)*

- [x] **Gate 1:** Data loaded, preprocessed, and partitioned with registered seeds
- [x] **Gate 2:** All models initialized with registered Kaiming-normal seeds
- [x] **Gate 3:** All 45 sanity tests pass (`pytest tests/ -v`)
- [x] **Gate 4:** Single Phase 1 round produces valid loss and prototype updates
- [x] **Gate 5:** Phase 1 converges (prototype drift < 0.01 for 5 rounds)
- [x] **Gate 6:** Phase 1 → Freeze → Phase 2 produces positive Dice on validation
- [x] **Gate 7:** Lineage audit log correctly chained; synthetic forbidden packet rejected
- [x] **Gate 8:** All endpoints, tests, and A1–A8 ablation configurations frozen before inspecting test results
- [x] **Gate 9:** `generate_tables.py` produces valid LaTeX from dummy CSV data
- [x] Automated auditor script `scripts/verify_pre_experiment_gates.py` confirms **ALL 9 GATES PASSED**
- [x] Create root `README.md` with one-command reproduction instructions

---

## Scientific Experiment Roadmap & Execution Tracker

To complete the full empirical study required for publication in **IEEE TMI / MICCAI 2026** per [M1_CAMFS_Unified_Architecture_Complete.md](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/M1_Implementation_spec_doc/M1_CAMFS_Unified_Architecture_Complete.md) and [CAMFS_M1_Complete_Specification.md](file:///home/harshyadav/.gemini/antigravity-ide/brain/23be6503-9e84-4368-863a-6dde390b9f1e/CAMFS_M1_Complete_Specification.md), the following experiment execution matrix is tracked:

### 1. Primary 9-Run Federation Matrix *(Spec §13.3, §15.2)*

| Run # | Partition Seed | Training Seed | Phase 1 (100 rds) | Phase 2 Track FL | Post-P2 H2 Head | 50 Pure Test Eval | Output Directory | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **1** | **1103** | **17** | ✅ Done | ✅ Done | ✅ Done | ✅ **76.60%** | `outputs/results/camfs_primary__part1103__seed17/` | 🟢 **COMPLETE** |
| **2** | 1103 | 29 | ⏳ Pending | ⏳ Pending | ⏳ Pending | ⏳ Pending | `outputs/results/camfs_primary__part1103__seed29/` | 🔴 To-Do |
| **3** | 1103 | 43 | ⏳ Pending | ⏳ Pending | ⏳ Pending | ⏳ Pending | `outputs/results/camfs_primary__part1103__seed43/` | 🔴 To-Do |
| **4** | 2207 | 17 | ⏳ Pending | ⏳ Pending | ⏳ Pending | ⏳ Pending | `outputs/results/camfs_primary__part2207__seed17/` | 🔴 To-Do |
| **5** | 2207 | 29 | ⏳ Pending | ⏳ Pending | ⏳ Pending | ⏳ Pending | `outputs/results/camfs_primary__part2207__seed29/` | 🔴 To-Do |
| **6** | 2207 | 43 | ⏳ Pending | ⏳ Pending | ⏳ Pending | ⏳ Pending | `outputs/results/camfs_primary__part2207__seed43/` | 🔴 To-Do |
| **7** | 3301 | 17 | ⏳ Pending | ⏳ Pending | ⏳ Pending | ⏳ Pending | `outputs/results/camfs_primary__part3301__seed17/` | 🔴 To-Do |
| **8** | 3301 | 29 | ⏳ Pending | ⏳ Pending | ⏳ Pending | ⏳ Pending | `outputs/results/camfs_primary__part3301__seed29/` | 🔴 To-Do |
| **9** | 3301 | 43 | ⏳ Pending | ⏳ Pending | ⏳ Pending | ⏳ Pending | `outputs/results/camfs_primary__part3301__seed43/` | 🔴 To-Do |

---

### 2. The 8 Ablation Studies (A1–A8) *(Spec §14.3)*

| # | Ablation Study | Core Scientific Question Tested | Config YAML | Script Command | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **A1** | **Joint-Training Alternative** | Does two-phase freezing improve purity without sacrificing task accuracy? | `a1_joint_training.yaml` | `python scripts/run_ablations.py --ablation A1_JointTraining` | 🟡 Ready to Run |
| **A2** | **Delayed-Site Context** | How does dynamic onboarding of H4 at $t=30$ impact convergence vs day-0 joining? | `a2_delayed_site.yaml` | `python scripts/run_ablations.py --ablation A2_DelayedSite` | 🟡 Ready to Run |
| **A3** | **Hard vs Soft Gating Lineage Audit** | **THE KILLER EXPERIMENT:** Does DisentAFL-style soft routing silently violate institutional consent? | `a3_lineage_audit.yaml` | `python scripts/run_ablations.py --ablation A3_LineageAudit` | 🟡 Ready to Run |
| **A4** | **$\lambda_1$ Sweep** | Does unimodal alignment weight $\lambda_1 \in \{0, 0.1, 0.5, 1.0\}$ affect feature space purity? | `a4_lambda1_sweep.yaml` | `python scripts/run_ablations.py --ablation A4_Lambda1Sweep` | 🟡 Ready to Run |
| **A5** | **Cold-Start Seeding Tiers** | Does Tier 1 Net2Net channel-widening outperform Tier 2 warm-start and Tier 3 fresh init? | `a5_cold_start.yaml` | `python scripts/run_ablations.py --ablation A5_ColdStart` | 🟡 Ready to Run |
| **A6** | **Group-Symmetric vs Directional** | What is the accuracy/communication cost of enforcing a strict one-way T2 policy? | `a6_directional.yaml` | `python scripts/run_ablations.py --ablation A6_Directional` | 🟡 Ready to Run |
| **A7** | **Multi-Track Contribution** | Does superset donation ($R_{\text{contribute}}(\text{H1}, S_3)=1$) boost the minority track $S_3$? | `a7_multi_track.yaml` | `python scripts/run_ablations.py --ablation A7_MultiTrack` | 🟡 Ready to Run |
| **A8** | **H2 Reconnection** | Does Option (b) private local head outperform Option (a) pull-only inference loading of $S_4$? | `a8_reconnection.yaml` | `python scripts/run_ablations.py --ablation A8_Reconnection` | 🟡 Ready to Run |

---

### 3. Comparative Baseline Experiments (B1–B6) *(Spec §14.1)*

| # | Baseline Name | Description & Comparison Purpose | Implementation & Execution Plan | Status |
| :---: | :--- | :--- | :--- | :---: |
| **B1** | **Local-Only** | No federation. Each hospital trains independently on its local dataset. **Lower bound.** | Run client-isolated training loop with local optimizer. | 🔴 To-Do |
| **B2** | **Policy-Blind FedAvg** | Aggregates all encoders and tracks ignoring consent policies. **Non-compliant reference.** | Standard FedAvg with masked inputs across all sites. | 🔴 To-Do |
| **B3** | **DisentAFL Reproduction** | Soft accuracy-driven routing. Subjected to shadow lineage audit. **Primary comparative target.** | Adapt DisentAFL routing objective to 4-class BraTS. | 🔴 To-Do |
| **B4** | **Availability Hard Cohort (RELIEF)** | Cohorts formed solely by exact modality availability without consent matrices. | Availability grouping without multi-track contribution. | 🔴 To-Do |
| **B5** | **FedAMM Local Reproduction** | Per-combination prototype clustering and dynamic zero-masking on the same data split. | Local run with FedAMM prototype aggregation equations. | 🔴 To-Do |
| **B6** | **Centralized Oracle** | Centralized model trained on all patients with all 4 modalities. **Theoretical ceiling.** | Centralized training loop on full 4-modality training pool. | 🔴 To-Do |

---

### 4. Statistical Aggregation & Final Publication Outputs *(Spec §14.5)*

- [ ] Execute remaining 8 primary runs (Partitions 1103, 2207, 3301 × Seeds 17, 29, 43).
- [ ] Run `scripts/generate_tables.py` on all 9 completed runs to compute 10,000 PCG64 bootstrap CIs and sign-flip test $p$-values.
- [ ] Generate final publication LaTeX tables: `primary_results.tex`, `ablation_results.tex`, `lineage_audit_summary.tex`.
- [ ] Generate final publication vector figures: `phase1_convergence.pdf`, `phase2_validation_curves.pdf`, `ablation_a3_audit.pdf`.

---

## Execution Progress Log

### Log Entry 1 — Phase 1: Environment & Infrastructure Setup
- **Completed:** 2026-08-02
- **Accomplishments:**
  - Created Conda environment `camfs` with Python 3.10.20 and PyTorch 2.5.1+cu121.
  - Configured deterministic seeds, master YAML configuration loader, and initialized GitHub repository.

### Log Entry 2 — Phase 2: Dataset Preprocessing Pipeline
- **Completed:** 2026-08-02
- **Accomplishments:**
  - Preprocessed and cached 368 BraTS 2020 patient volumes (1,840 `.npy` arrays) with z-score normalization on nonzero voxels and $[-5, 5]$ clipping.
  - Generated partition manifests for seeds $\{1103, 2207, 3301\}$ and fixed 50-patient test set (seed 901).
  - Implemented 1:1 slice sampler and GPU CUDA augmentations. 4/4 PyTests passed.

### Log Entry 3 — Phase 3: Neural Network Architectures
- **Completed:** 2026-08-02
- **Accomplishments:**
  - Implemented `UnimodalEncoder`, `PrototypeBank`, `SubsetFusionHead`, and `UNetDecoder`.
  - 5/5 PyTest model shape and gradient flow tests passed.

### Log Entry 4 — Phase 4: Loss Functions & 3D Evaluation
- **Completed:** 2026-08-02
- **Accomplishments:**
  - Implemented InfoNCE masked contrastive loss, Soft Dice + CE loss, 3D region evaluator (WT, TC, ET), and HD95 surface metric.
  - 6/6 PyTest loss and metric tests passed.

### Log Entry 5 — Phase 5: Governance, Policy & Provenance
- **Completed:** 2026-08-02
- **Accomplishments:**
  - Implemented `PolicyManager` (`policy_M1_PRIMARY_V1.json`), `ProvenanceLedger` (SHA-256 hash chaining), and `LineageAuditor` (packet-level lineage tracking).
  - 8/8 PyTest governance and policy tests passed.

### Log Entry 6 — Phase 6: Federation Protocol & Phase Controller
- **Completed:** 2026-08-02
- **Accomplishments:**
  - Implemented `PhaseController` state machine (`PHASE1` $\to$ `FROZEN` $\to$ `PHASE2` $\to$ `RELEASED`).
  - Implemented `FederatedClient` and `FederatedServer` with pre-mix consent cohorts.
  - 5/5 PyTest freeze, determinism, and protocol tests passed.

### Log Entry 7 — Phase 7: Metrics Logging & Checkpointing
- **Completed:** 2026-08-02
- **Accomplishments:**
  - Implemented per-round CSV loggers and atomic checkpoint manager (`CheckpointManager`).
  - 4/4 PyTest logger tests passed.

### Log Entry 8 — Phase 8: Runner Scripts & Ablation Configurations
- **Completed:** 2026-08-02
- **Accomplishments:**
  - Implemented `scripts/run_primary.py`, `scripts/run_ablations.py`, and created all 8 ablation YAML configs (`a1`–`a8`).
  - 9/9 PyTest ablation tests passed.

### Log Entry 9 — Phase 9: Statistical Analysis & Paper Output
- **Completed:** 2026-08-02
- **Accomplishments:**
  - Implemented `src/statistics.py` (10,000 PCG64 bootstrap CIs, paired sign-flip tests), `scripts/generate_tables.py`, and `scripts/generate_figures.py`.
  - 4/4 PyTest statistical tests passed.

### Log Entry 10 — Phase 10: Pre-Experiment Gates Verification
- **Completed:** 2026-08-02
- **Accomplishments:**
  - Verified `scripts/verify_pre_experiment_gates.py` — ALL 9 GATES PASSED.
  - All 45 unit tests in workspace passed. Created root `README.md`.

### Log Entry 11 — Primary Campaign Execution, Post-Phase 2 Net2Net & Pure 50 Test Evaluation
- **Completed:** 2026-08-06 / 2026-08-13
- **Accomplishments:**
  - Executed primary experiment `camfs_primary__part1103__seed17` on NVIDIA H100 GPU (`NVIDIA H100 NVL MIG 4g.47gb`).
  - Phase 1 completed 100 rounds of contrastive unimodal alignment and executed permanent freeze snapshot (`phase1_frozen.pt`).
  - Phase 2 completed track-isolated federated fusion training across $S_1, S_2', S_3, S_4$.
  - Implemented and executed **Post-Phase 2 H2 Private Local Head** via Tier-1 Net2Net channel-widening (`net2net_widen_fusion_head`), saving `best_H2_local_head.pt` with `network_transmission: NEVER` provenance logging.
  - Implemented and ran `scripts/evaluate_pure_50_test_set.py` across all four hospital models on the universal 50-patient test set (`seed 901`):
    - **$H_1$ ($S_1$, 4 mods)**: **`81.43%`** Macro Dice (**`2.59 mm`** Med HD95)
    - **$H_4$ ($S_4$, 3 mods)**: **`81.07%`** Macro Dice (**`2.44 mm`** Med HD95)
    - **$H_2$ ($\text{Track}_{\text{local},2}$, Private)**: **`78.45%`** Macro Dice (**`3.14 mm`** Med HD95) — **+15.91%** over 2-modality baseline.
    - **$H_3$ ($S_3$, 2 mods)**: **`65.44%`** Macro Dice (**`5.82 mm`** Med HD95).
    - **Standardized Federated Mean**: **`76.60%`** Macro Dice (**+6.89%** total gain over FedAMM SOTA **`69.71%`**).
  - Synchronized master specifications (`M1_CAMFS_Unified_Architecture_Complete.md` and `M1_CAMFS_Research_Implementation_Spec.md`) and merged all progress into `main` branch.
