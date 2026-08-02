# M1 Implementation Specification — Summary

> **Authoritative document:** [M1_CAMFS_Research_Implementation_Spec.md](M1_CAMFS_Research_Implementation_Spec.md)  
> **Last updated:** 2026-07-31  
> **Status:** Standalone M1 research specification; ready for implementation after the listed pre-experiment gates are satisfied.

## Design position

M1 is a standalone research study of consent-constrained multimodal federated segmentation. It neither uses nor depends on M2 components. M2 is a follow-up study that retrains its own M1 base with an M2-specific H3 split; the papers share methodology, not checkpoints.

This summary is non-normative. The detailed specification is the sole source of truth for implementation, configuration, and paper claims.

## M1 method in one view

1. Train one modality-specific encoder bank through closed, pre-mix consent cohorts.
2. Freeze encoder parameters, buffers, prototypes, optimizer state, and encoder communication.
3. Train structurally isolated fusion/decoder tracks keyed by the permitted send subset.
4. Enforce the versioned `M1_PRIMARY_V1` policy manifest and record all state lineage in a hashed research ledger.

The protected guarantee is limited to image/update-path lineage: excluded image tensors and updates derived from them cannot shape an authorized M1 encoder or track state. It is not a privacy, legal-compliance, malicious-client, or label-provenance guarantee.

## Standalone M1 protocol

| Item | M1 choice |
| :--- | :--- |
| Dataset | BraTS 2021, 1,251 labelled patients |
| Hospital allocation | H1 500; H2 313; H3 250; H4 188 |
| H3 roles | 175 train / 25 validation / 50 final test |
| Federation seeds | 1103, 2207, 3301 |
| Training seeds | 17, 29, 43 |
| Architecture | Four 2D modality encoders; concat + \(1\times1\) fusion; isolated subset tracks |
| Phase 1 | AdamW, \(3\times10^{-4}\), \(\tau=0.1\), \(\lambda_1=1\), one local epoch per round |
| Phase 2 | AdamW, \(10^{-3}\), \(\lambda_2=0.1\), one local epoch per round |
| Loss | Soft Dice + cross-entropy, \(\epsilon_D=10^{-5}\) |
| Evaluation | Patient-level 3D WT/TC/ET Dice and HD95; no postprocessing |
| Replication | 3 partitions × 3 training seeds = 9 runs |

## What was corrected from the older unified design

- Pairwise receive policies compile into closed pre-mix cohorts; incompatible updates are never filtered after they have mixed.
- A recipient may load a track only when it owns every track modality and accepts every contributor.
- Empty class prototypes are explicitly masked before they receive support.
- Phase 2 now defines fused-prototype bootstrap, InfoNCE, full-round order, closed track aggregation, and checkpoint selection.
- The M1 primary model is the concat + \(1\times1\) design, not an unresolved cross-attention default.
- The former CKA leakage test was removed: the shared T1 encoder makes its same-modality CKA null identically one. A packet-level execution-lineage audit replaces it.

## Primary research evidence

M1 evaluates six baselines (Local-Only, policy-blind subset FedAvg, DisentAFL reproduction, availability-only hard cohorts, FedAMM reproduction, and centralized full-modality oracle) and eight precisely defined ablations.

The primary compliance evidence is the executable lineage audit. CAMFS must accept zero policy-violating encoder or track states across all nine runs. DisentAFL is observed in shadow-audit mode; the paper does not assume in advance that it will violate the M1 policy.

External baselines require a preregistered source/configuration manifest before any result is reported.

## Relationship to M2

| Shared methodology | Separate work |
| :--- | :--- |
| Architecture, cohort aggregation, preprocessing, seeds, hyperparameters, evaluator | H3 role split, M1 checkpoint, experiments, grants, teacher, adapter, and calibration |
| Same BraTS 2021 patient pool | M2 retrains M1 using 125 train / 25 validation / 25 calibration / 25 acceptance-validation / 50 test H3 roles |

M1 therefore remains a complete standalone study. M2 may cite it as the methodological base but cannot claim to reuse the literal M1-paper checkpoint.

## Pre-experiment gates

- Freeze and hash the patient manifests, `M1_PRIMARY_V1` policy manifest, audit code, evaluator, and external-baseline configuration manifests.
- Verify all initialization, data-loader, augmentation, and sampler seeds are deterministic.
- Verify Phase-1 prototype convergence, Phase-2 validation behavior, policy enforcement, seed-lineage checks, and frozen-encoder tests.
- Preregister primary endpoints, baseline adaptations, and all ablations before H3 final-test evaluation.
