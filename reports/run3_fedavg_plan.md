# Implementation Plan: RUN-3 Compliant FedAvg Baseline (`scripts/run3_fedavg/`)

> **Document Type:** Experiment Architecture & Implementation Plan  
> **Target Venue:** IEEE JBHI / IEEE TMI  
> **Master Reference:** [`reports/experiment_execution_plan.md`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/reports/experiment_execution_plan.md) (§RUN-3) & [`reports/modular_experiment_runners_plan.md`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/reports/modular_experiment_runners_plan.md)  
> **Status:** Pending User Approval  

---

## 1. Goal & Scientific Rationale

### The Role of FedAvg (RUN-3) in the Paper
In the scientific hierarchy of the paper:
- **Upper Bound (Ceiling):** RUN-5 Centralized Oracle (all 256 patients pooled).
- **Proposed Method:** CAMFS-M1 Primary (multi-track privacy/consent isolation).
- **Competitor Baseline (RUN-3):** **FedAvg (Compliant Zero-Filled)**.
- **Lower Bound (Floor):** RUN-2 Local-Only Baseline (individual hospitals).

FedAvg is the ubiquitous standard baseline in federated learning. Every reviewer will look for it first.
The critical scientific question answered by RUN-3 is:
> **"Does track-isolated CAMFS outperform standard Federated Averaging under identical patient consent constraints?"**

---

## 2. Core Algorithmic & Architectural Design

### 2.1 Network Architecture (Strict Parameter Parity)
To ensure unassailable, apples-to-apples comparison with CAMFS-M1, FedAvg uses the **exact same neural network components**:
- **4 Unimodal Encoders:** One per modality (`T1`, `T1ce`, `T2`, `FLAIR`), each with 5 resolution levels `[32, 64, 128, 256, 256]`, GroupNorm (8 groups), and SiLU activation.
- **1 4-Modality Fusion Head:** `SubsetFusionHead(modality_subset=['T1', 'T1ce', 'T2', 'FLAIR'])` fusing multi-level encoder feature maps via $1 \times 1$ compression and ConvBlocks.
- **1 UNet Decoder:** `UNetDecoder(num_classes=4)` with skip connection fusion and bilinear upsampling to output 4-class logits $(B, 4, 240, 240)$.
- **Single Global Model:** $\theta_{\text{global}} = \{\text{encoders}, \text{fusion\_head}, \text{decoder}\}$. Total parameter count is identical to CAMFS full track.

### 2.2 Input Handling & Zero-Filling
All 4 modality slots are always fed to the network. When a hospital does not possess or share a modality, the input slice is **zero-filled** (`torch.zeros_like(x)`):
```python
for mod in ["T1", "T1ce", "T2", "FLAIR"]:
    if mod in client_available_mods:
        input_tensors[mod] = sample_dict[mod].to(device)
    else:
        input_tensors[mod] = torch.zeros((B, 1, 240, 240), dtype=torch.float32, device=device)
```

### 2.3 Consent Compliance Protocol
- **Hospital $H_1$** (102 train / 25 val): Owns and shares `T1`, `T1ce`, `T2`, `FLAIR`. All 4 inputs are real MRI data.
- **Hospital $H_2$** (64 train / 16 val): Owns `T1`, `T1ce`, `T2`. By consent policy, **$T_1ce$ is private and strictly unshared**.
  - **During federated training:** $T_1ce$ is **ZERO-FILLED** (treated as missing). Nothing $H_2$ uploads to the global model is derived from $T_1ce$. $FLAIR$ is also zero-filled.
  - **During local testing / evaluation:** $H_2$ feeds in its own local $T_1ce$ (local use allowed; nothing leaves the hospital).
- **Hospital $H_3$** (52 train / 12 val): Owns `T1`, `FLAIR`. $T_1ce$ and $T_2$ are zero-filled.
- **Hospital $H_4$** (38 train / 9 val): Owns and shares `T1`, `T1ce`, `T2`. $FLAIR$ is zero-filled.

### 2.4 Federated Aggregation (FedAvg Math)
In round $t$:
1. Server distributes $\theta_{\text{global}}^{(t)}$ to all 4 hospitals.
2. Each hospital $k \in \{1, 2, 3, 4\}$ trains locally for $E = 1$ epoch with AdamW (`lr = 3e-4`, gradient clipping `max_norm = 1.0`).
3. Server aggregates client updates using patient-weighted averaging:
   $$\theta_{\text{global}}^{(t+1)} = \sum_{k=1}^4 \frac{N_k}{N} \theta_k^{(t+1)}$$
   where $N_1=102, N_2=64, N_3=52, N_4=38$, total $N = 256$:
   - $w_1 = 102/256 \approx 0.3984$
   - $w_2 = 64/256 = 0.2500$
   - $w_3 = 52/256 \approx 0.2031$
   - $w_4 = 38/256 \approx 0.1484$

### 2.5 Validation & Model Selection
- Each round, evaluate the aggregated global model on each hospital's local validation set (using its local evaluation modalities).
- Track weighted global validation macro Dice:
  $$\text{Macro}_{\text{val}} = \sum_{k=1}^4 w_k \text{Macro}_{\text{val}}^{(k)}$$
- Save `best_model.pt` when $\text{Macro}_{\text{val}}$ reaches a new peak.
- Early stopping patience: 20 rounds without improvement (max 100 rounds).

### 2.6 Universal 3D Test Set Evaluation
- Reload `best_model.pt`.
- Evaluate on the **50 pure held-out test patients** (`outputs/partitions/h3_test_patients.json`).
- Evaluated for each hospital separately with that hospital's diagnostic modalities:
  - $H_1$: `[T1, T1ce, T2, FLAIR]`
  - $H_2$: `[T1, T1ce, T2]` (FLAIR zero-filled)
  - $H_3$: `[T1, FLAIR]` (T1ce, T2 zero-filled)
  - $H_4$: `[T1, T1ce, T2]` (FLAIR zero-filled)
- Metric calculation via unified [`src/metrics.py::PatientEvaluator`](file:///home/harshyadav/Research/CAMFS/CAMFS_M1_Unified/src/metrics.py) (stacked 155 slices per patient, full 3D Dice and 3D HD95 with 371.4 mm penalty).
- Output: `outputs/results/fedavg__part1103__seed<seed>/pure_50_test_patient_metrics.csv` (contains 200 rows: 50 patients $\times$ 4 hospital perspectives).

---

## 3. Directory Layout & File Artifacts

```
scripts/run3_fedavg/
├── run_fedavg.py               # Complete, self-contained FedAvg runner script
├── submit_fedavg_s17.sbatch    # Slurm submission script for Seed 17
├── submit_fedavg_s29.sbatch    # Slurm submission script for Seed 29
├── submit_fedavg_s43.sbatch    # Slurm submission script for Seed 43
└── submit_all_fedavg.sh        # Batch submission helper for all seeds
```

### Outputs Structure:
- Checkpoints: `outputs/checkpoints/fedavg__part1103__seed<seed>/`
  - `latest_model.pt` (atomic auto-resume per round)
  - `best_model.pt` (best validation macro Dice)
- Logs: `outputs/logs/fedavg__part1103__seed<seed>/`
  - `training_metrics.csv` (round loss, per-hospital val Dice, global val Dice)
  - `slurm_fedavg_s<seed>_<jobid>.out` & `.err`
- Results: `outputs/results/fedavg__part1103__seed<seed>/`
  - `pure_50_test_patient_metrics.csv` (full 200-row benchmark evaluation)

---

## 4. Verification & Execution Roadmap

### Step 1: Pre-Flight Smoke Test (`--dry-run`)
Run 2 dry-run rounds with 2 patients per hospital on GPU:
```bash
python scripts/run3_fedavg/run_fedavg.py --dry-run --gpu 0
```
Verify:
1. Zero-filling correctly handles missing modalities without tensor shape mismatch.
2. Parameter averaging produces finite weights (no NaNs).
3. Checkpoint auto-resume functions as expected.
4. Test evaluation produces 3D Dice and HD95 metrics.

### Step 2: Parallel Dual-Compute Execution
- **Seed 17 Execution on V100:**
  Because `v100` is currently idle with 32 GB VRAM and full dataset local cache, we can launch **Seed 17 FedAvg directly on `v100`** in the background!
- **Seeds 29 & 43 on H100 / V100:**
  Once Seed 17 completes, Seeds 29 and 43 can be dispatched across both machines to complete all 3 seeds rapidly.
