# CAMFS M1 — Research Code Standard Practices

> **Purpose:** Define every engineering convention the codebase must follow so that running 9 experiments produces paper-ready tables, figures, and audit logs with zero manual post-processing.
>
> **Authoritative spec:** [M1_CAMFS_Research_Implementation_Spec.md](M1_CAMFS_Research_Implementation_Spec.md)

---

## 1. Isolated Python Environment

### 1.1 Conda environment (not pip global)

Use one project-local Conda environment. `environment.yml` is the human-maintained source of truth; `environment.lock.yml` and `requirements.lock.txt` are generated, version-pinned artifacts recorded with every result bundle.

```bash
conda create -n camfs python=3.10 -y
conda activate camfs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy nibabel SimpleITK scipy scikit-image pandas matplotlib tqdm pyyaml pytest
conda env export --no-builds > environment.lock.yml
pip freeze > requirements.lock.txt
```

### 1.2 Dependency files to maintain

| File | Purpose | When to update |
| :--- | :--- | :--- |
| `environment.yml` | Human-maintained Conda environment definition | When an intentional dependency changes |
| `environment.lock.yml` | Resolved Conda environment (`conda env export --no-builds`) | After every intentional dependency change |
| `requirements.lock.txt` | Pip-resolved fallback (`pip freeze`) | After every intentional dependency change |
| `README.md` | Setup instructions, one-command reproduce | Before submission |

### 1.3 Lock the exact versions

The spec (§15.4) requires recording *"framework, accelerator, and kernel versions."* Add a runtime version dump at the start of every run:

```python
# Logged automatically at run start
{
    "python": sys.version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "numpy": np.__version__,
    "gpu": torch.cuda.get_device_name(0)
}
```

---

## 2. Project Directory Structure

Modelled after FedAMM's structure but adapted to M1's two-phase consent-cohort design:

```
c:\Users\ihars\Downloads\Research\CamFS\M1_Implementation\
│
├── environment.yml              # Human-maintained environment source
├── environment.lock.yml         # Resolved Conda environment
├── requirements.lock.txt        # Resolved pip packages
├── README.md                    # One-command reproduce instructions
│
├── configs/
│   ├── default.yaml             # All §15 defaults in one YAML
│   ├── policy_M1_PRIMARY_V1.json # Versioned policy manifest (§5.3)
│   ├── baseline_manifests/      # Source/version/config manifests for B3 and B5
│   │   ├── disentafl.json
│   │   └── fedamm.json
│   └── ablations/
│       ├── a1_joint_training.yaml
│       ├── a2_delayed_site.yaml
│       ├── a3_lineage_audit.yaml
│       ├── a4_lambda1_sweep.yaml
│       ├── a5_cold_start.yaml
│       ├── a6_directional.yaml
│       ├── a7_multi_track.yaml
│       └── a8_reconnection.yaml
│
├── src/
│   ├── __init__.py
│   ├── config.py                # YAML loader + CLI arg parser
│   ├── seed.py                  # Deterministic seed manager (§15.4)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── brats_dataset.py     # BraTS 2021 NIfTI loader + z-norm (§13.4)
│   │   ├── partition.py         # PCG64 patient-disjoint splitter (§13.3)
│   │   ├── slice_sampler.py     # Tumour/non-tumour 1:1 axial sampler (§13.4)
│   │   └── augmentation.py      # Geometric + intensity aug (§13.4)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py           # Unimodal 2D UNet encoder (§5.1)
│   │   ├── fusion.py            # Concat + 1×1 fusion heads (§5.4)
│   │   ├── decoder.py           # UNet decoder with skip connections (§5.4)
│   │   └── prototypes.py        # Unimodal & fused prototype bank (§5.2)
│   │
│   ├── governance/
│   │   ├── __init__.py
│   │   ├── policy.py            # Policy matrix manager + cohort compiler (§5.3)
│   │   ├── ledger.py            # SHA-256 chained JSONL provenance log (§5.6)
│   │   └── lineage_audit.py     # Packet-level image_lineage tracker (§14.4)
│   │
│   ├── federation/
│   │   ├── __init__.py
│   │   ├── server.py            # Cohort aggregation + prototype merge (§6.2, §8.5)
│   │   ├── client.py            # Local training loop (Phase 1 & 2)
│   │   └── phase_controller.py  # State machine: P1 → FROZEN → P2 → RELEASED (§5.5)
│   │
│   ├── losses.py                # InfoNCE + Dice+CE with exact §15.1 formula
│   ├── metrics.py               # 3D patient-level Dice, HD95 (§13.5)
│   └── statistics.py            # Bootstrap CIs, sign-flip tests, Holm (§14.5)
│
├── baselines/
│   ├── local_only.py            # B1
│   ├── policy_blind_fedavg.py   # B2
│   ├── disentafl/               # B3 — adapted reproduction
│   ├── availability_cohort.py   # B4
│   ├── fedamm/                  # B5 — adapted reproduction
│   └── centralized_oracle.py    # B6
│
├── scripts/
│   ├── run_primary.py           # Runs all 9 partition×seed M1 experiments
│   ├── run_baselines.py         # Runs all 6 baselines × 9 runs
│   ├── run_ablations.py         # Runs all 8 ablations
│   ├── preprocess.py             # Writes canonical unaugmented patient cache
│   ├── make_partitions.py        # Writes and hashes patient manifests
│   └── generate_tables.py       # Produces LaTeX tables from results/
│
├── tests/                       # Sanity verification scripts
│   ├── test_data_pipeline.py    # Verify shapes, splits, normalization
│   ├── test_freeze.py           # Verify irreversible Phase-1 encoder freeze
│   ├── test_policy.py           # Verify policy matrix rejects forbidden routes
│   ├── test_lineage.py          # Verify audit log catches violations
│   ├── test_loss.py             # Verify exact masked InfoNCE and Dice+CE
│   ├── test_protocol.py         # Verify bootstrap, aggregation, and fresh-optimizer rules
│   └── test_determinism.py      # Verify deterministic same-device replay
│
├── outputs/                     # ALL experiment outputs (gitignored)
│   ├── partitions/              # Patient manifest JSONs + SHA-256 hashes
│   ├── preprocessed/            # Canonical unaugmented per-patient cache
│   ├── checkpoints/             # Model .pt files per run
│   ├── ledger/                  # Provenance JSONL logs per run
│   ├── metrics/                 # Per-round CSV metrics per run
│   └── results/                 # Final aggregated tables + bootstrap CIs
│
└── M1_CAMFS_Research_Implementation_Spec.md  # The authoritative spec
```

---

## 3. Configuration Management (Not Hardcoded)

### 3.1 Single YAML config file

Every hyperparameter from §15 lives in `configs/default.yaml`, not scattered across Python files. This is how FedAMM and all standard FL repos handle it:

```yaml
# configs/default.yaml
experiment:
  name: "camfs_m1_primary"
  partition_seeds: [1103, 2207, 3301]
  training_seeds: [17, 29, 43]
  bootstrap_seed: 8803
  sensitivity_seed: 8804

dataset:
  name: "BraTS2021"
  root: "/path/to/BraTS2021"
  expected_labeled_patients: 1251
  cache_root: "outputs/preprocessed"
  cache_format: "per_patient_npy"
  cache_contains_augmentation: false
  input_size: [240, 240]
  modalities: ["T1", "T1ce", "T2", "FLAIR"]
  label_remap: {0: 0, 1: 1, 2: 2, 4: 3}
  slices_per_patient: 155
  batch_size: 16
  num_workers: 4

partition:
  patient_id_order: "official_identifier_lexicographic"
  rng: "PCG64"
  fixed_h3_final_test_seed: 901
  fixed_h3_final_test_patients: 50
  h3_final_test_excluded_from: ["train", "validation", "calibration", "model_selection"]

governance:
  policy_manifest: "configs/policy_M1_PRIMARY_V1.json"
  policy_version: "M1_PRIMARY_V1"
  audit_mode: "reject"       # Primary CAMFS; B3 uses its documented shadow override
  ledger_canonical_json: true

reproducibility:
  deterministic_algorithms: true
  allow_tf32: false
  precision: "fp32"

primary_exclusions:
  class_weighting: false
  deep_supervision: false
  learning_rate_schedule: false
  gradient_clipping: false
  postprocessing: false

hospitals:
  H1: {patients: 500, owned: ["T1","T1ce","T2","FLAIR"], send: ["T1","T1ce","T2","FLAIR"], split: [350, 50, 100]}
  H2: {patients: 313, owned: ["T1","T1ce","T2"],         send: ["T1","T2"],                 split: [219, 31, 63]}
  H3: {patients: 250, owned: ["T1","FLAIR"],              send: ["T1","FLAIR"],              split: [175, 25, 50]}
  H4: {patients: 188, owned: ["T1","T1ce","T2"],          send: ["T1","T1ce","T2"],          split: [132, 19, 37]}

phase1:
  lr: 3.0e-4
  lambda1: 1.0
  tau: 0.1
  bottleneck_samples: "all_15x15"
  min_rounds: 20
  max_rounds: 100
  drift_epsilon: 0.01
  drift_patience: 5
  local_epochs: 1

phase2:
  lr: 1.0e-3
  lambda2: 0.1
  tau: 0.1
  validation_selection: "highest_contributor_patient_weighted_macro_dice"
  tie_break: "earliest_exact_tie"
  min_rounds: 20
  max_rounds: 100
  patience: 10
  improvement_threshold: 1.0e-4
  local_epochs: 1

optimizer:
  type: "AdamW"
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 1.0e-4

federation_protocol:
  phase1_round0: "no_optimizer_prototype_bootstrap"
  new_phase2_track_bootstrap: "no_optimizer_fused_prototype_bootstrap"
  fresh_local_optimizer_each_round: true
  retain_local_optimizer_state_between_rounds: false
  server_optimizer: false
  prototype_recompute: "eval_fp32_all_training_patients_all_155_slices_no_augmentation"
  aggregation_weight: "distinct_local_patients_used_in_round"
  prototype_aggregation_weight: "patients_with_supported_class"

loss:
  dice_smoothing: 1.0e-5
  tumour_classes: [1, 2, 3]  # after label remap

model:
  encoder_channels: [32, 64, 128, 256, 256]
  groupnorm_groups: 8
  activation: "SiLU"
  init: "kaiming_normal"
  init_mode: "fan_out"
  init_nonlinearity: "relu"

```

### 3.2 Ablation configs override only what changes

Each ablation YAML imports the default and overrides one setting:

```yaml
# configs/ablations/a4_lambda1_sweep.yaml
_base_: "../default.yaml"
experiment:
  name: "ablation_a4_lambda1"
phase1:
  lambda1: 0.0   # swept across {0, 0.1, 0.5, 1.0}
```

### 3.3 CLI interface

```bash
python scripts/run_primary.py --config configs/default.yaml --partition-seed 1103 --train-seed 17 --gpu 0
```

### 3.4 Reference-baseline manifests are mandatory

Before a B3 (DisentAFL) or B5 (FedAMM) result is run or reported, create its immutable manifest in `configs/baseline_manifests/`. It records the cited paper version, official/author-provided source URL, commit or archive SHA-256, every changed file, and every hyperparameter changed for the 2D four-class BraTS setting. Hash this manifest into the run manifest and ledger. If this evidence is unavailable, label the condition an **adaptation**, never a literal reproduction.

---

## 4. Deterministic Seed Management

`PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG` must be set before Python imports PyTorch or initializes CUDA; setting them inside the function below is too late. Launch each primary run as follows (with the actual run training seed):

```powershell
$env:PYTHONHASHSEED = "17"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
python scripts/run_primary.py --config configs/default.yaml --partition-seed 1103 --train-seed 17 --gpu 0
```

Your spec §15.4 requires this. Implement a single `seed.py` module called once at the start of every run:

```python
# src/seed.py
import os
import random

import numpy as np
import torch

def set_deterministic(seed: int) -> None:
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError("Launch with PYTHONHASHSEED equal to the training seed")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG before starting Python")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)
```

Every `DataLoader` uses `worker_init_fn=seed_worker` and a run-specific `generator=make_generator(seed)`. Record the launch environment, seed, device, and version dump in the run manifest. Determinism is required for a replay on the **same hardware, driver, PyTorch build, and configuration**; do not claim bitwise identity across different GPUs or software stacks.

> [!IMPORTANT]
> **FedAMM uses `torch.manual_seed()` and `numpy.random.seed()` but does NOT enable `torch.use_deterministic_algorithms(True)`.** Your spec explicitly requires it (§15.4). This is stricter and better — it catches non-deterministic CUDA kernels at runtime.

---

## 5. Dataset Preprocessing Pipeline

### 5.1 How FedAMM does it (3D patches)

FedAMM loads BraTS volumes as 3D $128\times128\times128$ patches using MONAI/SimpleITK, applies z-score normalization on brain voxels, and uses dynamic modality masking to simulate missing modalities per client.

### 5.2 How CAMFS M1 does it differently (2D axial slices)

Your spec (§13.4) requires a fundamentally different pipeline:

| Step | CAMFS M1 Specification | Implementation |
| :--- | :--- | :--- |
| **Input format** | BraTS 2021 NIfTI volumes ($240\times240\times155$) | Use `nibabel` to load `.nii.gz` |
| **Normalization** | Per-patient, per-modality z-score on nonzero brain voxels, clip $[-5,5]$, outside-brain stays 0 | Custom function, NOT MONAI default |
| **Cropping** | **No crop** — keep full $240\times240$ canvas | Different from FedAMM which crops to bounding box |
| **Slice extraction** | 2D axial slices, patient-level split before slicing | Pre-split patients, then extract per-patient |
| **Training sampling** | Tumour-containing slices + equal-count non-tumour brain slices (1:1 ratio) per epoch | Custom `SliceSampler` class |
| **Validation/test** | All 155 slices, no sampling, no augmentation | Simple sequential loader |
| **Augmentation** | Geometric (flip, rotation $\pm10°$) + intensity (scale $[0.9,1.1]$, shift $[-0.1,0.1]$) on brain mask only | Custom transforms, applied identically to all co-registered modalities |
| **Label mapping** | BraTS $\{0,1,2,4\} \to$ model $\{0,1,2,3\}$ | Simple remap before any loss/metric |

### 5.3 Canonical, unaugmented cache is a one-time offline step

```
Raw BraTS NIfTI volumes
    │
    ▼
[preprocess.py]  ──→  outputs/preprocessed/
    │                    ├── patient_BraTS21_00001/
    │                    │     ├── t1.npy       (155×240×240 float32)
    │                    │     ├── t1ce.npy     (155×240×240 float32)
    │                    │     ├── t2.npy       (155×240×240 float32)
    │                    │     ├── flair.npy    (155×240×240 float32)
    │                    │     └── labels.npy   (155×240×240 uint8, remapped)
    │                    ├── patient_BraTS21_00002/
    │                    ...
    │
    ▼
[partition.py]   ──→  outputs/partitions/
                        ├── partition_1103.json   (H1/H2/H3/H4 patient ID lists)
                        ├── partition_2207.json
                        ├── partition_3301.json
                        ├── h3_test_patients.json (fixed 50 patients, seed 901)
                        └── manifests_sha256.json
```

> [!IMPORTANT]
> Pre-slice only the **canonical, unaugmented** normalized images and remapped labels. Store each array as an uncompressed `.npy` file, then read it lazily with `np.load(path, mmap_mode="r")`; compressed `.npz` files do not provide the required memory-mapped access. Apply tumour/non-tumour sampling and all augmentation at runtime, never in the cache. The full cache is roughly 178 GB at FP32, so it must not be loaded into RAM as a whole.

---

## 6. Metrics Logging & Results Storage

### 6.1 Per-round CSV logging (during training)

Every federated round writes one row to a CSV. This is what reviewers and you need for loss curves and convergence plots:

```
outputs/metrics/{run_id}/phase1_rounds.csv
outputs/metrics/{run_id}/phase2_rounds.csv
```

**Phase 1 columns:**
```
run_id, config_hash, policy_digest, phase1_state_hash, round, cohort, contributors, contributor_patient_counts, recipients, drift_t1, drift_t1ce, drift_t2, drift_flair, mean_drift, loss_h1, loss_h2, loss_h3, wall_time_sec
```

**Phase 2 columns:**
```
run_id, config_hash, policy_digest, phase1_state_hash, round, track, contributors, contributor_patient_counts, val_dice_wt, val_dice_tc, val_dice_et, val_macro_dice, train_loss, fused_align_loss, best_round, wall_time_sec
```

### 6.2 Final test evaluation (after training)

```
outputs/results/{run_id}/test_results.csv
```

**Columns (one row per patient):**
```
run_id, condition, partition_seed, training_seed, config_hash, policy_digest, patient_id, hospital, track, dice_wt, dice_tc, dice_et, hd95_wt, hd95_tc, hd95_et
```

Write a separate run summary with `one_empty_case_count`, `one_empty_case_rate`, selected checkpoint hash, final ledger hash, patient-manifest digest, and baseline source-manifest digest where applicable. A metric row is not interpretable without this provenance.

### 6.3 Aggregated results across all 9 runs

```
outputs/results/aggregated/
    ├── primary_results.csv        # Mean ± bootstrap CI per hospital per condition
    ├── baseline_comparisons.csv   # Paired contrasts + Holm-adjusted p-values
    ├── ablation_results.csv       # A1–A8 results
    └── lineage_audit_summary.csv  # Pass/fail + violation counts per run
```

### 6.4 What NOT to use

| Tool | Verdict for CAMFS M1 |
| :--- | :--- |
| **Weights & Biases** | Optional for personal tracking; do NOT depend on it for paper results. Results must be reproducible from local CSVs alone. |
| **TensorBoard** | Optional for debugging loss curves during development. Not required for final paper. |
| **MLflow** | Overkill for a single-researcher Master's project. |
| **Plain CSV + JSON** | **This is the standard.** FedAMM, FeTS-AI, and most MICCAI papers use this. Reviewers can `pandas.read_csv()` your results. |

---

## 7. Experiment Run Management

### 7.1 Run ID convention

Every run gets a unique, human-readable ID:

```
{condition}__part{partition_seed}__seed{train_seed}
```

Examples:
```
camfs_primary__part1103__seed17
camfs_primary__part1103__seed29
baseline_b1_local__part1103__seed17
ablation_a3_audit__part2207__seed43
```

### 7.2 Automated experiment scripts

```bash
# Run all 9 primary CAMFS M1 experiments
python scripts/run_primary.py --config configs/default.yaml --gpu 0

# Run all baselines
python scripts/run_baselines.py --config configs/default.yaml --gpu 0

# Run specific ablation
python scripts/run_ablations.py --config configs/ablations/a3_lineage_audit.yaml --gpu 0

# Generate paper tables from results
python scripts/generate_tables.py --results-dir outputs/results/aggregated/
```

### 7.3 Checkpoint and manifest strategy

Save the **selected best checkpoint** (by validation macro Dice) and only the minimal restart/debug artefacts needed for the study:

```
outputs/checkpoints/{run_id}/
    ├── phase1_final.pt         # Final frozen encoder state
    ├── phase2_best_{track}.pt  # Best validation Dice checkpoint per track
    ├── phase2_last_{track}.pt  # Last round checkpoint (for debugging)
    └── run_config.yaml         # Exact config used for this run
```

Each checkpoint directory also contains `run_manifest.json` and `file_hashes.json`. The run manifest records the run ID; configuration, policy, and patient-manifest digests; phase-1 frozen state hash; selected checkpoint hash; final ledger hash; seeds; environment/version dump; and, for B3/B5, the source/version/commit/config-change manifest digest. Hash every saved model and manifest in `file_hashes.json`. Do not select a test-set checkpoint.

### 7.4 Freeze is a destructive state transition

`phase_controller.py` must implement `PHASE1 → FROZEN → PHASE2 → RELEASED` as a checked state machine. At `FROZEN`, serialize and hash the selected Phase-1 state; set each encoder to evaluation mode and `requires_grad=False`; use stop-gradient in every later forward; discard the Phase-1 optimizer and its communication route; then create a fresh Phase-2 optimizer that contains only the permitted fusion, decoder, and fused-prototype parameters. A later encoder update, mutable encoder buffer, Phase-1 optimizer restoration, or unlogged transfer is a test failure, not a recoverable warning.

---

## 8. Provenance Ledger & Lineage Audit

### 8.1 The ledger is a research artifact, not a database

Your spec §5.6 requires a SHA-256 chained JSONL log. This is lightweight:

```jsonl
{"record_id":0,"event":"POLICY_MANIFEST","run_id":"camfs_primary__part1103__seed17","policy_digest":"a3f8...","prev_hash":"0000...","record_hash":"b7c2..."}
{"record_id":1,"event":"PHASE1_AGGREGATION","run_id":"camfs_primary__part1103__seed17","round":1,"cohort":"kappa_T1","packet_ids":["pkt-001","pkt-002","pkt-003"],"contributors":["H1","H2","H3"],"recipients":["H1","H2","H3"],"decision":"accepted","state_hash":"91de...","image_lineage":["T1"],"prev_hash":"b7c2...","record_hash":"65af..."}
```

Canonicalize JSON (stable key order and compact separators) before hashing. `record_hash` is SHA-256 of that record with the `record_hash` field omitted; each later `prev_hash` equals the preceding `record_hash`. Log packet IDs, sender/recipient, policy decision, image lineage, and the state hash for every transfer or aggregation. Store the final record hash in `run_manifest.json`; `test_lineage.py` must recompute the entire chain.

### 8.2 Lineage audit for Ablation A3

The audit report names the policy digest, audited run ID, route/packet IDs, state hash, decision, and violation count. It is an execution-path claim, not a claim that image lineage mathematically proves label privacy.

Per §14.4, every local update packet carries an `image_lineage` set. The audit is a deterministic pass/fail:

- **CAMFS:** Runtime rejects any packet with forbidden lineage before deserialization.
- **B3 (DisentAFL):** Shadow mode — logs violations but does not alter B3's routing.

---

## 9. Sanity Verification Tests

Run these BEFORE launching the 9 full experiments. These correspond to §19.3 pre-experiment gates:

```bash
# Run all sanity checks
python -m pytest tests/ -v
```

| Test | What it verifies | Spec reference |
| :--- | :--- | :--- |
| `test_data_pipeline.py` | Shapes ($4\times240\times240$), z-norm range, label remap, partition patient counts | §13.3, §13.4 |
| `test_freeze.py` | After Phase 1 freeze: all encoder parameters have `.requires_grad == False`, encoder is in eval mode, outputs are detached, and a Phase-2 optimizer step cannot alter encoder parameters | §7 |
| `test_policy.py` | Policy compiler accepts the listed primary routes (including H1 masked contribution to S3) and rejects H3 receiving T1ce, H2 contributing T1ce, H4 pre-freeze contribution, and every unlisted route | §5.3, §5.5 |
| `test_lineage.py` | Lineage audit catches a synthetically injected forbidden-modality packet | §14.4 |
| `test_loss.py` | InfoNCE with empty-prototype masking produces valid gradients; Dice+CE matches the exact §15.1 formula | §6.1, §15.1 |
| `test_protocol.py` | Round 0 has no optimizer step; aggregation uses the registered patient/support weights; and no client/server optimizer state persists across rounds | §6.2, §8.3–§8.5 |
| `test_determinism.py` | Two same-device replays with the same environment and seed produce the same Phase-1 round-1 loss and state hash | §15.4 |

---

## 10. How This Differs from FedAMM

This is a CAMFS scope comparison, not a substitute for verifying a particular FedAMM release. The registered B5 source/configuration manifest controls the reproduction; do not make paper claims about FedAMM behavior that are not supported by that cited source.

| Dimension | FedAMM (MICCAI 2025) | CAMFS M1 (Your Paper) |
| :--- | :--- | :--- |
| **Input** | 3D patches ($128^3$) | 2D axial slices ($240\times240$) |
| **Modality handling** | Dynamic masking (soft) | Hard structural absence (no input slot) |
| **Aggregation** | Global cluster-weighted prototype averaging | Closed pre-mix consent cohorts ($\kappa$) |
| **Policy/governance** | None | Exogenous versioned manifest + SHA-256 ledger |
| **Routing** | Learned soft gates | Exogenous hard policy matrices |
| **Evaluation** | DSC only | DSC + HD95 + lineage audit pass/fail |
| **Statistics** | Mean ± std across runs | 10,000-resample bootstrap CIs + Holm-adjusted sign-flip tests |
| **Reproducibility** | `torch.manual_seed()` | Full deterministic: `torch.use_deterministic_algorithms(True)` + PCG64 partition seeds |
| **Config** | `argparse` CLI flags | YAML config files |

---

## 11. Paper-Ready Output Generation

### 11.1 LaTeX table generation

`scripts/generate_tables.py` reads CSVs and outputs LaTeX:

```latex
% Auto-generated by generate_tables.py
\begin{table}[t]
\caption{Per-hospital segmentation performance (Dice \%).}
\begin{tabular}{lccc|ccc}
\toprule
 & \multicolumn{3}{c}{WT} & \multicolumn{3}{c}{TC} \\
Method & H1 & H2 & H3 & H1 & H2 & H3 \\
\midrule
Local-Only (B1)   & 84.2 & 79.1 & 71.3 & ... \\
CAMFS M1 (Ours)   & \textbf{87.1} & ... \\
\bottomrule
\end{tabular}
\end{table}
```

### 11.2 Figure generation

Use `matplotlib` with a consistent style:

```python
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
```

Save figures as both PDF (for LaTeX) and PNG (for slides):

```
outputs/figures/
    ├── phase1_convergence.pdf
    ├── phase2_validation_curves.pdf
    ├── compliance_gap_bar.pdf
    └── ablation_a3_audit.pdf
```

---

## 12. Summary Checklist

Before running the first real experiment, verify:

- [ ] Environment source and both resolved lock files created; runtime version dump is written
- [ ] `configs/default.yaml` matches every §15 value exactly
- [ ] `configs/policy_M1_PRIMARY_V1.json` matches §5.3 tables exactly
- [ ] BraTS 2021 data downloaded and preprocessed
- [ ] Patient partition manifests generated and SHA-256 hashed
- [ ] All 7 sanity tests pass (`pytest tests/ -v`)
- [ ] Single Phase 1 round produces valid loss and prototype updates
- [ ] Phase 1 reaches its registered drift criterion and Phase 1→Freeze→Phase 2 produces positive validation Dice on one partition
- [ ] Lineage audit log is correctly chained, policy/audit code is hashed, and a synthetic forbidden packet is rejected
- [ ] B1–B6 are adapted to the 4-hospital BraTS federation; B3/B5 source/configuration manifests are registered and hashed
- [ ] All endpoints, tests, and A1–A8 configurations are frozen before inspecting test results
- [ ] `generate_tables.py` produces valid LaTeX from dummy CSV data
