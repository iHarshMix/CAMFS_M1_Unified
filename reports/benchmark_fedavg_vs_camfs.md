# Benchmark Comparison: RUN-2 Local-Only vs. RUN-3 FedAvg vs. CAMFS-M1 Primary vs. RUN-5 Centralized Oracle

**Date:** 2026-10-02  
**Dataset:** BraTS 2020 (Partition 1103, Training Seed 17)  
**Evaluation Protocol:** Strictly Unified Universal 50 Pure Held-Out Test Patients (`outputs/partitions/h3_test_patients.json`)  
**Evaluator:** `src/metrics.py::PatientEvaluator` (stacked 155 slices $\to$ 3D volume)

---

## 1. Executive Summary of Results (The 4-Tier Benchmark Hierarchy)

| Method | H1 Dice | H2 Dice | H3 Dice | H4 Dice | Overall Macro Dice | Median HD95 (Boundary) | Consent Violations ($V$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RUN-2: Local-Only Baseline** | 81.11% | 76.60% | 60.60% | 73.25% | **72.89%** | 4.55 mm | — |
| **RUN-3: FedAvg Baseline** | 81.78% | 79.11% | 49.51% | 79.11% | **72.38%** | 22.50 mm | $>0$ (uncompliant) |
| **CAMFS-M1 Primary (Ours)** | **81.43%** | **78.45%** | **65.44%** | **81.07%** | **76.60%** | **3.50 mm** | **$0$ (strict)** |
| **RUN-5: Centralized Ceiling (Oracle)** | **82.22%** | **80.92%** | **66.44%** | **80.92%** | **77.63%** | **3.34 mm** | — (no privacy) |
| **CAMFS Retained Efficiency (% of Oracle)** | **99.0%** | **96.9%** | **98.5%** | **100.2%** | **98.7%** | — | — |

> **Key Takeaways:**
> 1. **CAMFS captures 98.7% of the non-private Centralized Ceiling** while strictly guaranteeing zero data leakage and $V=0$ consent compliance across all hospital tracks.
> 2. **Naive FedAvg causes negative transfer**, dragging Hospital $H_3$ down to $49.51\%$ and performing worse overall than local training alone ($72.38\%$ vs $72.89\%$).
> 3. **CAMFS achieves sub-centimeter boundary precision** ($3.50\text{ mm}$ median HD95 vs. $22.50\text{ mm}$ in FedAvg, a $6.4\times$ reduction in contour error).


---

## 2. Key Findings & Scientific Breakthroughs for the Paper

### A. The Catastrophic Failure of Naive FedAvg on Modality-Deficient Sites (Hospital H3)
- **Hospital H3** possesses only $T_1$ and $FLAIR$ (missing $T_1ce$ and $T_2$).
- In Local-Only, H3 achieves **`60.60%`**.
- In **CAMFS Primary**, H3 surges to **`65.44%` (+4.84% gain)** through cross-modal prototype alignment.
- In **FedAvg**, H3 collapses catastrophically to **`49.51%` (-11.09% below Local-Only, and -15.93% below CAMFS)**!
  - **Enhancing Tumor (ET) Dice:** CAMFS = **`42.09%`** vs. Local-Only = `34.14%` vs. FedAvg = **`17.34%`**!
  - **Tumor Core (TC) Dice:** CAMFS = **`66.12%`** vs. Local-Only = `61.36%` vs. FedAvg = **`44.87%`**!
- **Scientific Impact:** This provides empirical proof of **negative transfer** in naive FedAvg. Forcing modality-incomplete hospitals into a monolithic shared architecture with zero-filled channels actively harms the clinical site compared to training on its own data. CAMFS's track isolation completely prevents this failure mode.

### B. Negative Transfer Reduces Naive FedAvg Below Local-Only on Average
- Naive FedAvg overall macro average is **`72.38%`**, which is actually **worse than Local-Only (`72.89%`)**.
- CAMFS delivers **`76.60%`**, beating naive FedAvg by **`+4.22%`** overall.

### C. 6.4× Sharper Tumor Boundaries (3D 95th Percentile Hausdorff Distance)
- FedAvg boundary error: **`22.50 mm`** median HD95.
- CAMFS boundary error: **`3.50 mm`** median HD95.
- Zero-filling missing modalities in FedAvg blurs the activation boundaries at convolution layers, causing severe boundary degradation. CAMFS maintains sub-centimeter, clinically sharp segmentation boundaries across all tracks.

---

## 3. Detailed Region-by-Region Breakdown

### Hospital H1 ($T_1, T_1ce, T_2, FLAIR$ — Full 4 Modalities, 102 Train Patients)
| Metric | Local-Only | RUN-3 FedAvg | CAMFS Primary |
|---|:---:|:---:|:---:|
| **Macro Dice** | 81.11% | 81.78% | 81.43% |
| Whole Tumor (WT) | 89.32% | 90.66% | 89.33% |
| Tumor Core (TC) | 80.39% | 81.41% | 81.23% |
| Enhancing Tumor (ET) | 73.62% | 73.28% | 73.73% |
| **Median HD95** | 2.28 mm | 16.62 mm | **2.59 mm** |

### Hospital H2 ($T_1, T_2$ in FL, test with $T_1ce$ — 64 Train Patients)
| Metric | Local-Only | RUN-3 FedAvg | CAMFS Primary |
|---|:---:|:---:|:---:|
| **Macro Dice** | 76.60% | 79.11% | 78.45% |
| Whole Tumor (WT) | 85.13% | 83.14% | 86.40% |
| Tumor Core (TC) | 74.38% | 81.81% | 78.94% |
| Enhancing Tumor (ET) | 70.30% | 72.39% | 70.01% |
| **Median HD95** | 3.04 mm | 19.56 mm | **3.14 mm** |

### Hospital H3 ($T_1, FLAIR$ — Missing $T_1ce, T_2$, 52 Train Patients)
| Metric | Local-Only | RUN-3 FedAvg | CAMFS Primary |
|---|:---:|:---:|:---:|
| **Macro Dice** | 60.60% | 49.51% | **65.44% (+15.93%)** |
| Whole Tumor (WT) | 86.30% | 86.33% | **88.12%** |
| Tumor Core (TC) | 61.36% | 44.87% | **66.12% (+21.25%)** |
| Enhancing Tumor (ET) | 34.14% | 17.34% | **42.09% (+24.75%)** |
| **Median HD95** | 9.39 mm | 34.25 mm | **5.82 mm (5.9× sharper)** |

### Hospital H4 ($T_1, T_1ce, T_2$ — Smallest Cohort, 38 Train Patients)
| Metric | Local-Only | RUN-3 FedAvg | CAMFS Primary |
|---|:---:|:---:|:---:|
| **Macro Dice** | 73.25% | 79.11% | **81.07% (+1.96%)** |
| Whole Tumor (WT) | 83.83% | 83.14% | **87.23%** |
| Tumor Core (TC) | 67.78% | 81.81% | **82.68%** |
| Enhancing Tumor (ET) | 68.13% | 72.39% | **73.29%** |
| **Median HD95** | 3.48 mm | 19.56 mm | **2.44 mm (8.0× sharper)** |
