# CAMFS-M1 Data Split Intuition Report

> **Purpose:** Explain *why* this specific federation data split is the right one to evaluate CAMFS-M1's claims — and why no simpler alternative would work.

---

## 1. The Core Problem CAMFS Solves

In the real world, hospitals don't all have the same MRI scanners, and even when they do, legal/ethical consent rules may forbid sharing certain scans. The central question is:

> *Can you still train a competitive brain tumour segmentation model when hospitals have different modalities AND different consent policies about what they're allowed to share?*

The data split is engineered so that **every type of heterogeneity that exists in the real world is simultaneously present** — and each hospital isolates a different stress case.

---

## 2. Dataset Overview

| Property | Value |
|---|---|
| **Dataset** | BraTS 2020 Training Set |
| **Total patients** | 368 (patient 355 withdrawn — known BraTS issue) |
| **Modalities per patient** | T1, T1ce, T2, FLAIR (4 co-registered MRI sequences) |
| **Segmentation labels** | Background, NCR/NET, Edema, Enhancing Tumour |
| **Slices per patient** | 155 axial slices |
| **Input resolution** | 240 × 240 per slice |

---

## 3. The Split at a Glance

```
BraTS 2020 Total:              368 patients (355 withdrawn)
┌─────────────────────────────────────────────┐
│ Fixed H3 Test Set:           50  (PCG64@901)│ ← never trained on, never validated on
├─────────────────────────────────────────────┤
│ Remaining for federation:    318             │
│   H1 (40%): 127 → 102 train / 25 val       │
│   H2 (25%):  80 →  64 train / 16 val       │
│   H3 (20%):  64 →  52 train / 12 val       │
│   H4 (15%):  47 →  38 train /  9 val       │
└─────────────────────────────────────────────┘
```

### Deterministic Partitioning

- **Test set:** 50 patients selected by `PCG64(seed=901)` — fixed across all experiments, never seen during training or validation.
- **Federation split:** Remaining 318 patients shuffled per partition seed `{1103, 2207, 3301}` using `PCG64`, then divided by the 40/25/20/15 ratio.
- **Within-hospital:** Each hospital applies an 80/20 train/val split internally (last 20% of sorted patient IDs).
- **SHA-256 manifests** are stored for every partition for full reproducibility.

---

## 4. What Each Hospital Tests

```
              Modalities                     
Hospital   Owned            Shared (consent)   Size    Role in evaluation
─────────────────────────────────────────────────────────────────────────
H1         T1,T1ce,T2,FLAIR  all 4             127     The "anchor" — rich & cooperative
H2         T1,T1ce,T2        T1,T2 only        80      The "consent bottleneck"
H3         T1,FLAIR           both              64+50   The "modality-starved" site
H4         T1,T1ce,T2        all 3              47      The "small cooperative" site
```

### H1 — The Rich Anchor (40%, all 4 modalities, shares everything)

This is your large academic medical center. It has the most data and the most complete imaging protocol.

**What it tests:** Does CAMFS at least not *hurt* the hospital that already has everything? If H1's performance degrades under federation vs. training alone, something is broken. H1 is the federation's anchor — it contributes the most knowledge and should benefit least (since it already has all modalities locally).

### H2 — The Consent Bottleneck (25%, owns T1ce but *refuses* to share it)

This is **the single most important hospital for the paper's thesis.** H2 *has* T1ce (the contrast-enhanced scan that is critical for delineating enhancing tumour) but its consent policy says: *"you may not send T1ce outside this hospital."*

**What it tests:** Can CAMFS respect this restriction *and* still give H2 a good model? H2 should *receive* T1ce knowledge from H1 and H4 (who share T1ce), but must *never contribute its own T1ce* to the federation. This is the free-rider problem that CAMFS's policy matrices (`R_send`, `R_recv`) exist to solve.

**Why this matters clinically:** T1ce highlights the enhancing tumour boundary. Withholding it simulates a hospital whose patients consented to MRI scans for treatment but did *not* consent to having their contrast-enhanced images shared for research. This is a real and common consent scenario.

### H3 — The Modality-Starved Site (20%, only T1 + FLAIR)

H3 is missing both T1ce and T2 — it only has 2 out of 4 modalities. This is common in smaller regional hospitals that lack contrast-agent protocols or certain scanner configurations.

**What it tests:** Can CAMFS's fusion head produce a usable segmentation with *half the input channels missing*? This is where the track-isolated architecture matters — H3 can only participate in the T1 and FLAIR tracks, and its fusion head must learn to segment from just those two feature maps.

**Why this is the hardest case:** T1ce is the primary signal for enhancing tumour (ET), and T2 helps delineate peritumoral edema (ED). H3 has neither. If CAMFS can produce reasonable segmentation for H3, it proves the architecture's value under severe deprivation.

### H4 — The Small Cooperative (15%, 3 modalities, shares all)

The smallest hospital, fully cooperative but data-poor. It has T1, T1ce, T2 but no FLAIR.

**What it tests:** Do small hospitals actually *gain* from federation? If H4's federated model doesn't beat its local-only model, federation has no value for small sites — which kills the paper's practical argument. H4 also validates that patient-count-weighted aggregation doesn't drown out the smallest contributor.

---

## 5. Why the Ratios Matter (40/25/20/15, not 25/25/25/25)

Real federations are **imbalanced**. One large teaching hospital dominates; community clinics contribute far less data. If we used equal splits:

- FedAvg would work reasonably well (it's designed for roughly balanced settings)
- We'd be testing an artificially easy scenario
- Reviewers would immediately object: *"This doesn't reflect real deployment"*

The 40% → 15% spread creates a **2.7× imbalance** between the largest and smallest site. This forces the aggregation weighting (`distinct_local_patients_used_in_round`) to handle the asymmetry — and lets us show whether small hospitals are lifted or drowned by larger partners.

### The imbalance also interacts with modality availability:

| Hospital | Data share | Modality richness | Effect |
|---|---|---|---|
| H1 | Largest (40%) | Richest (4/4) | Dominates naive aggregation |
| H4 | Smallest (15%) | Medium (3/4) | Risks being overwhelmed |
| H3 | Small (20%) | Poorest (2/4) | Double disadvantage: less data AND fewer modalities |

This double heterogeneity (data quantity × modality availability) is exactly what real federations face and what CAMFS is designed to handle.

---

## 6. The Modality Landscape — Why No Simpler Split Works

```
         T1    T1ce    T2    FLAIR
H1        ✓      ✓      ✓      ✓     ← complete
H2        ✓      ✓*     ✓      ✗     ← *owns T1ce but won't SHARE it
H3        ✓      ✗      ✗      ✓     ← severe deprivation
H4        ✓      ✓      ✓      ✗     ← missing FLAIR
```

**No two hospitals have the same modality profile.** This means:

1. **There is no single "shared encoder"** that all hospitals can jointly train on all modalities — CAMFS must form **track-specific cohorts** per modality.

2. **The consent matrix creates asymmetry** even between hospitals with the same *ownership*: H2 and H4 both own T1ce, but only H4 shares it. A system that only knows "who has what" (ignoring consent) would incorrectly include H2 in the T1ce sharing cohort.

3. **FLAIR is only at H1 and H3**, so the FLAIR track cohort is {H1, H3} — a 127:64 imbalanced mini-federation within the federation.

4. **T1 is the only universally shared modality** (all 4 hospitals own and share it), making the T1 track the one place where all hospitals can collaborate.

### Track cohort formation under the policy:

| Modality Track | Sending cohort | Receiving cohort |
|---|---|---|
| T1 | {H1, H2, H3, H4} | {H1, H2, H3, H4} |
| T1ce | {H1, H4} | {H1, H2, H4} |
| T2 | {H1, H2, H4} | {H1, H2, H4} |
| FLAIR | {H1, H3} | {H1, H3} |

Note: H2 *receives* T1ce knowledge but never *sends* it. This is the consent asymmetry that CAMFS's `R_send` / `R_recv` matrices encode.

If we simplified this (e.g., all hospitals have 3 modalities, only 1 missing), we'd lose the ability to test:
- **Consent vs. availability** (H2 proves consent matters, not just what you own)
- **Severe deprivation** (H3 with only 2/4 modalities)
- **Asymmetric track participation** (some tracks have 2 members, some have 3–4)

---

## 7. Why 50 Test Patients from H3's Pool Specifically

This is the sharpest design choice. The test set comes from the **hardest scenario**:

1. These 50 patients have **all 4 modalities available** (they're BraTS patients with complete scans).
2. But they are evaluated using **each hospital's own model**, which sees only that hospital's owned modalities at inference.
3. So when H3's model is tested on these patients, it only gets T1 + FLAIR (the 2 modalities H3 owns).

This lets us build the paper's key comparison table:

| Model | Modalities at inference | Dice on same 50 patients |
|---|---|---|
| H1 model | T1, T1ce, T2, FLAIR | (best, upper bound) |
| H2 model | T1, T1ce, T2 | (near H1?) |
| H3 model | T1, FLAIR only | (the stress test) |
| H4 model | T1, T1ce, T2 | (small site, does federation help?) |
| Centralized | all 4, all 368 patients | (absolute ceiling) |

**All evaluated on the exact same 50 patients → directly comparable, no confounds.**

### Why not test patients from all hospitals?

- H1, H2, H4 don't have a separate held-out test set — their val sets are used for model selection (early stopping).
- Using H3's fixed test set as the universal benchmark ensures that **every comparison is on identical data**, eliminating patient-level confounds.
- Since these patients have all 4 ground-truth modalities, we can also compute a "what if this hospital had more data" oracle comparison.

---

## 8. Integrity Verification Summary

| Check | Result |
|---|---|
| Patient count on disk | 368 ✓ (BraTS20_Training_355 withdrawn — known issue) |
| Test set isolation | 0 patients overlap between test and any hospital's train/val ✓ |
| Test set stability | Identical 50 patients across all 3 partition seeds ✓ |
| Hospital ratios | H1=39.94%, H2=25.16%, H3=20.13%, H4=14.78% (within rounding of 40/25/20/15) ✓ |
| Inter-hospital leakage | Zero overlap between any two hospitals within each partition ✓ |
| SHA-256 manifests | Generated and stored for all partitions + test set ✓ |
| Within-hospital train/val | 80/20 split, deterministic (last 20% of sorted IDs) ✓ |

---

## 9. The One-Line Summary

> *"This split creates the minimal federation where every real-world heterogeneity — data quantity, modality availability, and consent restrictions — is simultaneously present and independently testable, so each hospital's result isolates a different claim of the paper."*

---

## Appendix: Full Test Patient List (PCG64, seed 901)

50 patients, lexicographically sorted:

```
BraTS20_Training_002, BraTS20_Training_004, BraTS20_Training_009,
BraTS20_Training_020, BraTS20_Training_024, ...
(full list in outputs/partitions/h3_test_patients.json)
```

SHA-256 of test manifest: `d100e44e4aba12ec339105b2d8727d356248f5ea60c6f4d8621dae8f5bc809ec`
