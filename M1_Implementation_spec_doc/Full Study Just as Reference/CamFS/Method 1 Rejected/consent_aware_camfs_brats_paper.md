# Consent-Aware Multimodal Federated Segmentation (CAMFS)
### Auditable Institutional Policy Gating for Asymmetric Modality Sharing in Federated Brain Tumor Segmentation

**Working paper draft — architecture instantiated on BraTS2020/BraTS2021**

---

## Abstract

Federated learning (FL) for multimodal medical imaging almost always treats a missing modality as an accident of data collection — a sensor that failed, a scan that wasn't ordered — and tries to compensate for it (imputation, hallucination, robust fusion). In many real multi-institutional deployments, however, a modality is absent not by accident but by **institutional design**: an ethics board has not cleared gadolinium-contrast imaging for a pediatric cohort, a rural site's scanner is physically incapable of a sequence, a data-sharing MoU permits pooling of structural sequences but not contrast-enhanced ones. In this setting the "right" behavior is not to reconstruct the missing signal — it is to guarantee, verifiably, that the disallowed signal never reaches a party that isn't permitted to have it, even at a measurable accuracy cost. We term this **consent-constrained** federated learning, as distinct from the much larger literature on **performance-constrained** (missing-modality-robust) federated learning. We formalize the distinction, instantiate a working architecture — a bank of modality-specific encoders combined with hard-isolated, subset-specific fusion tracks, gated by an explicit, auditable institutional policy matrix — and describe a full experimental and ablation plan on BraTS2020/BraTS2021 simulating a three-to-four-hospital federation with clinically motivated, asymmetric modality access. We do not claim the aggregation mechanism itself is unprecedented — close mechanisms have appeared independently in DisentAFL, RELIEF, and FedMM-style hard-gated designs. Our contribution is the **problem framing and the policy layer**: treating consent as a first-class, non-negotiable, auditable object that governs routing, and empirically measuring the accuracy cost of enforcing it against methods that only optimize for performance.

---

## 1. Problem Statement

### 1.1 The Real-World Setting

Consider three hospitals collaborating on a federated brain tumor segmentation model, each contributing patients scanned with some subset of the four standard multiparametric MRI (mpMRI) sequences — T1-weighted (T1), contrast-enhanced T1-weighted (T1ce/T1Gd), T2-weighted (T2), and T2-FLAIR:

- **Hospital A — Regional Contrast-Restricted Center.** Serves a population with a high proportion of pediatric and renal-impaired patients for whom gadolinium-based contrast agents are contraindicated or restricted by internal ethics policy. It **only ever acquires T2 and FLAIR**. It has no T1ce data, cannot acquire it, and its data-sharing agreement explicitly forbids incorporating T1ce-conditioned signal into any model it deploys, regardless of whether doing so would improve accuracy.
- **Hospital B — Academic Comprehensive Center.** Runs the full mpMRI protocol (T1, T1ce, T2, FLAIR) and has an MoU permitting full participation in the federation.
- **Hospital C — Rural Low-Resource Site.** Operates an older scanner that cannot run the T2 sequence used at the other sites and does not perform contrast studies. It **only ever has T1 and FLAIR**.

This is not a corrupted-data problem. No amount of imputation "fixes" Hospital A's missing T1ce, because the constraint is not statistical, it is **legal and ethical**: Hospital A is not permitted to benefit from, or leak into, T1ce-conditioned representations, full stop. A federated learning method that quietly routes T1ce-derived gradient signal to Hospital A because it improves Dice score is not a better method here — it is a **non-compliant** one.

### 1.2 Why Existing FL Assumptions Fail Here

The dominant multimodal FL literature (FedMFS, MAFS, FedProto, FedAMM, MMiC, FedCMD, DisentAFL, and others) is built to answer: *"given that modalities are missing, how do we still get the best possible shared model?"* Their gating, clustering, or routing decisions — even when "hard" in mechanism — are ultimately justified and tuned by **performance**: a modality is included in a fusion path because it helps, excluded because it's unavailable or noisy, or routed by a learned gate that optimizes accuracy. None of them treat "must never be combined, even if it would help" as a first-class constraint with its own audit trail.

This produces two failure modes if applied naively to the hospital scenario:
1. **Silent violation.** A soft/learned gate (e.g., DisentAFL-style asymmetric routing) may discover that leaking T1ce signal into Hospital A's pathway improves its Dice score, and — because the gate is optimized for accuracy, not compliance — will do exactly that. The violation is invisible; there is no artifact a compliance officer or IRB could inspect to confirm it isn't happening.
2. **Under-specification.** Even hard-isolation mechanisms that do exist (cohort-wise aggregation, hard-gated experts) isolate along *data availability* boundaries, not *permission* boundaries. In practice these often coincide (a site without a scanner can't send that modality's gradients), but they are not the same variable: two sites might both have T1ce data yet be forbidden by a data-use agreement from being pooled with each other specifically, which availability-based clustering cannot express.

### 1.3 Formal Problem Statement

We define the problem CAMFS is built to solve:

> Given a federation of clients \(C = \{1, \dots, N\}\), each client \(i\) has a fixed, structurally-owned modality subset \(O(i) \subseteq \mathcal{M}\) (a modality is either always present for that client, or never — the *complete missing-modality* assumption, distinct from *partial* per-sample missingness). In addition to \(O(i)\), each client (or an external governance body) specifies a **consent policy** — a set of hard, exogenous, non-learnable constraints on which cross-client signal may be aggregated into which pathway. The learning system must:
> 1. Achieve the best possible task performance (segmentation Dice/HD95) for every client, **subject to** never violating any specified consent constraint, even when violating it would improve performance;
> 2. Make the enforced constraint set **auditable** — inspectable by a third party as an explicit object, not inferable only from training dynamics; and
> 3. Degrade gracefully — a client whose exact modality subset has not been seen before should not need to train an entirely new pathway from nothing.

We distinguish two policy regimes, both of which the architecture below must support:

- **Group-symmetric consent** \(R_{\text{recv}}(i, j, m) = 1 \iff m \in O(i) \cap O(j)\) and both clients have opted into the shared pool for modality \(m\) — i.e., all owners of a modality either pool together or don't, but there is no finer-grained per-pair asymmetry within an owning group.
- **Fully directional consent** \(R_{\text{recv}}(i, j, m) \in \{0, 1\}\), specified per ordered pair \((i, j)\) and modality \(m\), independent of \(R_{\text{recv}}(j, i, m)\) — e.g., Hospital B accepts contributions from everyone but does not reciprocate, or two T1ce-owning hospitals refuse to pool with each other specifically due to a competitive data-use restriction, despite both being willing to pool with a third party.

---

## 2. Positioning and Novelty Defense Statement

**We are explicit about what is, and is not, new here, because the closest prior art has moved quickly through 2024–2026 and a paper that claims architectural novelty alone will not survive review.**

> We do not claim novelty in the *mechanism* of hard, isolated aggregation by modality subset. Structurally similar mechanisms exist: DisentAFL (AAAI 2024) performs learned, asymmetric gated routing between clients; RELIEF (2026) aggregates modality-specific blocks only within the cohort of devices that possess that modality, explicitly to eliminate cross-modal gradient interference, with an accompanying convergence bound; hard-gated, modality-routed expert selection has also appeared in computational-pathology and phishing-detection federated systems inspired by the FedMM line of work. What none of these treat as a first-class object is **consent as a variable independent of performance or availability** — an exogenous, auditable, institution-defined policy matrix that can forbid a pairing even when it would help, and that can be inspected by a compliance officer rather than inferred from training behavior. That is our contribution: not a new mechanism, but a new *problem framing*, a concrete formalization of it (\(R_{\text{send}}\), \(R_{\text{recv}}\), hard vs. soft constraints, complete vs. partial missingness), an architecture assembled — deliberately, and without claiming otherwise — from known components to satisfy that framing, and an empirical protocol that quantifies the accuracy cost of enforcing consent against methods that only chase accuracy.

### 2.1 Closest prior art, side by side

| Work | Mechanism | Explicit consent/policy object? | What it optimizes for |
|---|---|---|---|
| FedMFS / MAFS | Send-side modality masking | No | Robustness to missing modalities |
| FedProto / MFCPL | Prototype-based alignment | No | Cross-client generalization |
| FedCMD | Unimodal encoders → late fusion (per-combination) | No | Handling arbitrary combinations, single institution focus |
| FedAMM | Per-combination centroid prototypes, single global model | No | Global accuracy across combinations |
| MMiC | Modality-combination clustering | Implicit grouping, not policy-driven | Reducing interference between combinations |
| DisentAFL (2024) | Learned, asymmetric gated knowledge routing | No — opaque, learned | Maximizing beneficial transfer |
| RELIEF (2026) | Cohort-isolated aggregation (hard, by data availability) | No | Convergence speed / efficiency on IoT edge |
| FedMM-style hard-gated pathology / phishing FL | Hard expert selection by modality label | No | Stability under modality heterogeneity |
| **CAMFS (this work)** | Hard-isolated bank + subset-fusion, gated by explicit \(R_{\text{send}}, R_{\text{recv}}\) | **Yes — auditable, institution-authored matrix, independent of availability** | **Compliance guarantee first; performance second** |

### 2.2 What would falsify our claim, and what we are doing about it

If a future review finds a paper that (a) hard-isolates by *permission* rather than *availability*, (b) exposes that permission structure as an inspectable artifact, and (c) reports the accuracy cost of enforcing it — our claim collapses to "we did this for medical imaging, they did it first elsewhere." We treat this as a real risk, not a hypothetical one, given how fast this sub-area moved in the last two years, and recommend a fresh literature check immediately before submission.

---

## 3. Related Work

*(Condensed here to the entries most load-bearing for positioning; expand with full citation detail in the camera-ready related-work section.)*

**Missing-modality-robust FL (performance-constrained):** FedMFS and MAFS handle missing modalities via send-side masking and robust aggregation; FedProto and MFCPL align clients in prototype space rather than parameter space to tolerate modality heterogeneity; CreamFL and FedMAC address cross-modal contrastive alignment under partial modality availability; Task Arithmetic and FedMSplit explore modular composition of modality-specific parameters. None of these encode a non-learnable, institution-specified permission constraint.

**Combination-aware architectures:** FedCMD trains unimodal encoders and fuses only within an institution's own available combination — closest in *mechanism* to our Unimodal Bank + Subset-Fusion Track design (Approach 1). FedAMM generalizes this to per-combination prototypes feeding a single global head. MMiC clusters clients by modality combination for symmetric aggregation.

**Hard-isolation / gated mechanisms (2024–2026):** DisentAFL's asymmetric gated routing is the closest thing to a *directional* mechanism in the literature, but it is learned and performance-driven, not policy-driven. RELIEF, appearing in 2026, formalizes cohort-wise hard isolation with a convergence proof, but frames the isolation purely as a way to eliminate destructive gradient interference from unavailable modalities — an availability constraint, not a permission constraint. Hard-gated expert-selection designs inspired by FedMM appear in both computational pathology and, more recently, phishing-webpage detection, again motivated by stability under heterogeneity rather than auditable compliance.

**Consent/governance in FL generally:** Differential privacy and secure aggregation address *what* is shared (protecting individual records), not *whether a modality-conditioned pathway may exist between two specific parties at all* — a structural, not statistical, question. To our knowledge no prior work formalizes modality-level, per-pair, auditable consent as a routing variable in multimodal FL.

---

## 4. Preliminaries and Notation

| Symbol | Meaning |
|---|---|
| \(C = \{1, \dots, N\}\) | Set of federation clients (simulated hospitals) |
| \(\mathcal{M} = \{\text{T1, T1ce, T2, FLAIR}\}\) | Full modality universe (BraTS instantiation) |
| \(O(i) \subseteq \mathcal{M}\) | Modality subset structurally owned by client \(i\) (fixed for all of \(i\)'s data — the *complete* missing-modality assumption) |
| \(S\) | A specific modality subset acting as a "fusion track" identity, e.g. \(S = \{\text{T2, FLAIR}\}\) |
| \(R_{\text{send}}(i, m) \in \{0,1\}\) | Client \(i\) is permitted to contribute modality-\(m\) encoder updates to the federation |
| \(R_{\text{recv}}(i, j, m) \in \{0,1\}\) | Client \(i\) is permitted to receive/incorporate modality-\(m\) signal that originated (directly or indirectly) from client \(j\) |
| \(\text{Encoder}_m\) | Modality-specific encoder for modality \(m\), shared across all clients with \(R_{\text{send}}(\cdot, m) = 1\) |
| \(\text{Proto}_m^c\) | Global class-\(c\) prototype in \(\text{Encoder}_m\)'s embedding space |
| \(\text{FusionHead}_S\), \(\text{Decoder}_S\) | Fusion and decoding modules specific to subset track \(S\), shared only among clients whose owned subset is \(S\) (or a superset routed down to \(S\)) |
| \(\text{FusedProto}_S^c\) | Class-\(c\) prototype in the fused embedding space of track \(S\) |
| Hard constraint | \(R_{\text{send}}, R_{\text{recv}}\) are **exogenous** — set by governance, never learned, never a function of the loss |
| Soft constraint (contrast case) | A learned gate whose value is optimized to improve task loss (e.g., DisentAFL-style) — explicitly *not* what CAMFS uses for consent enforcement |

---

## 5. Proposed Architecture: CAMFS

### 5.1 Design Principle

Two things must be true simultaneously, and no single existing mechanism guarantees both:

1. **A client only ever computes forward/backward passes through parameters trained on modalities it is permitted to receive.** This is enforced *structurally* (by which global parameters are broadcast to it), not by a penalty term it could in principle ignore.
2. **The permission structure is a first-class artifact** — a matrix, versioned and inspectable — not an emergent property of a learned gate's weights.

CAMFS satisfies (1) by combining two previously-analyzed approaches:

- **Approach 1 (Unimodal Bank):** every modality has its own encoder, aggregated only across clients permitted to contribute to and receive from it.
- **Approach 3 (Hard-Isolated Subset-Fusion Tracks):** every distinct *owned subset* that appears in the federation gets its own fusion head, decoder, and fused prototype set, aggregated only within that exact subset's owner group (or an explicitly-authorized cross-subset link).

The Unimodal Bank maximizes data efficiency (a T2 scan helps every T2-owning client, not just clients with an identical subset). The Subset-Fusion Tracks guarantee that fusion — where cross-modal leakage risk is highest — never crosses a permission boundary.

### 5.2 Component Diagram — BraTS Instantiation

```
                         ┌─────────────────────────────────────────┐
                         │           UNIMODAL ENCODER BANK          │
                         │  (aggregated per-modality, gated by      │
                         │   R_send / R_recv — availability AND     │
                         │   permission, not just availability)     │
                         │                                           │
                         │  Encoder_T1    Encoder_T1ce               │
                         │  Encoder_T2    Encoder_FLAIR               │
                         │                                           │
                         │  Proto_T1^c  Proto_T1ce^c                  │
                         │  Proto_T2^c  Proto_FLAIR^c                 │
                         └───────────────┬───────────────────────────┘
                                         │  (skip-connection features,
                                         │   routed ONLY to tracks the
                                         │   owning client's subset feeds)
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
   ┌──────────▼─────────┐    ┌───────────▼───────────┐   ┌──────────▼─────────┐
   │ TRACK S_A           │    │ TRACK S_B (full)       │   │ TRACK S_C           │
   │ {T2, FLAIR}         │    │ {T1,T1ce,T2,FLAIR}     │   │ {T1, FLAIR}          │
   │                     │    │                        │   │                      │
   │ FusionHead_S_A      │    │ FusionHead_S_B          │   │ FusionHead_S_C       │
   │ Decoder_S_A         │    │ Decoder_S_B             │   │ Decoder_S_C          │
   │ FusedProto_S_A^c    │    │ FusedProto_S_B^c        │   │ FusedProto_S_C^c     │
   │                     │    │                        │   │                      │
   │ HARD-ISOLATED —     │    │ HARD-ISOLATED —         │   │ HARD-ISOLATED —      │
   │ never aggregated    │    │ never aggregated with   │   │ never aggregated     │
   │ with S_B or S_C     │    │ S_A or S_C parameters   │   │ with S_A or S_B      │
   └──────────┬──────────┘    └───────────┬────────────┘   └──────────┬───────────┘
              │                           │                           │
        Hospital A                  Hospital B                  Hospital C
   (Contrast-restricted:        (Academic center:            (Rural low-resource:
    T2, FLAIR only;             full mpMRI protocol,          T1, FLAIR only;
    no T1ce, ever)               consents to share all)        no T1ce or T2 acquired)
```

### 5.3 Component Definitions (BraTS-specific)

- **Backbone:** 2D U-Net-style encoder/decoder (axial slices) for tractability at federation scale; a 3D variant is architecturally identical but noted as a compute-heavier optional variant, not evaluated by default.
- **Encoder\(_m\)** (\(m \in \{\text{T1, T1ce, T2, FLAIR}\}\)): 4-stage down-sampling path (conv-conv-downsample blocks), producing a bottleneck embedding and skip-connection feature maps at each resolution.
- **Proto\(_m^c\)**: mean bottleneck embedding for class \(c\), computed per-round from local batches and aggregated identically to Encoder\(_m\)'s parameters. Classes follow the standard BraTS label set: background, necrotic/non-enhancing tumor core (NCR/NET, label 1), peritumoral edema (ED, label 2), enhancing tumor (ET, label 4).
- **FusionHead\(_S\):** for each track \(S\), a lightweight cross-attention (or concatenation + 1×1 conv, as a cheaper ablation variant) module combining the skip-connection features of exactly the encoders in \(S\).
- **Decoder\(_S\):** standard U-Net up-sampling path from the fused bottleneck/skip features to a 4-class segmentation map, specific to track \(S\).
- **FusedProto\(_S^c\):** mean fused-bottleneck embedding for class \(c\) within track \(S\), used for the fusion-alignment loss term below.

### 5.4 Loss Functions

For a client \(i\) with owned subset \(O(i) = S\):

\[
\mathcal{L}_i = \mathcal{L}_{\text{task}}(y, \hat{y}) \;+\; \lambda_1 \sum_{m \in S} \mathcal{L}_{\text{unimodal-align}}(z_m, \text{Proto}_m^{c}) \;+\; \lambda_2\, \mathcal{L}_{\text{fusion-align}}(z_S, \text{FusedProto}_S^{c})
\]

- \(\mathcal{L}_{\text{task}}\): combined Dice + cross-entropy over the 4-class BraTS label map (or the composite WT/TC/ET regions, reported separately at evaluation time as is standard practice).
- \(\mathcal{L}_{\text{unimodal-align}}\): pulls each modality's bottleneck embedding \(z_m\) toward its class prototype, encouraging the shared Encoder\(_m\) to stay class-discriminative on its own, independent of which track it feeds — this term is the main lever in **Ablation A4** (gradient/representation purity).
- \(\mathcal{L}_{\text{fusion-align}}\): analogous alignment in the fused embedding space, specific to track \(S\), never shared across tracks.

### 5.5 Aggregation Rules

At round \(t\), for modality \(m\):

\[
\theta_{\text{Encoder}_m}^{t+1} = \sum_{i \,:\, m \in O(i),\; R_{\text{send}}(i,m)=1} \frac{n_i}{N_m}\, \theta_{\text{Encoder}_m,i}^{t}
\]

where \(n_i\) is client \(i\)'s local sample count and \(N_m = \sum_{i: m \in O(i), R_{\text{send}}(i,m)=1} n_i\). Crucially, the *broadcast* of \(\theta_{\text{Encoder}_m}^{t+1}\) back down is itself gated: client \(i\) only receives the average over the subset of contributors it is permitted to receive from,

\[
\theta_{\text{Encoder}_m \to i}^{t+1} = \sum_{j \,:\, m \in O(j),\; R_{\text{recv}}(i,j,m)=1} \frac{n_j}{N_m^{(i)}}\, \theta_{\text{Encoder}_m,j}^{t}, \qquad N_m^{(i)} = \sum_{j: R_{\text{recv}}(i,j,m)=1} n_j
\]

In the **group-symmetric special case** used as the default for the three-hospital BraTS simulation, \(R_{\text{recv}}(i,j,m) = 1\) for all \(i,j\) that both own \(m\) and have opted in — i.e., every T2-owning, opted-in hospital receives the *same* aggregate. Track-level aggregation for FusionHead\(_S\), Decoder\(_S\), and FusedProto\(_S^c\) follows the identical rule restricted to clients whose owned subset is exactly \(S\).

### 5.6 Training Loop (pseudocode)

```
for round t = 1 .. T:
    for each client i in parallel:
        S ← O(i)
        pull Encoder_m for each m in S, gated by R_recv(i, ·, m)
        pull FusionHead_S, Decoder_S  (only if track S already exists; else see 5.7)
        local_train():
            z_m ← Encoder_m(x_m)  for m in S
            z_S ← FusionHead_S({z_m : m in S})
            y_hat ← Decoder_S(z_S)
            loss ← L_task + λ1 * unimodal_align + λ2 * fusion_align
            backprop, local SGD steps
        push updated Encoder_m deltas (only for m where R_send(i, m) = 1)
        push updated FusionHead_S, Decoder_S deltas (track S only)
        push local Proto_m^c, FusedProto_S^c estimates

    server:
        for each modality m: aggregate Encoder_m per R_send/R_recv (eq. 5.5)
        for each track S: aggregate FusionHead_S, Decoder_S within owner group of S only
        update Proto_m^c, FusedProto_S^c as gated running averages
```

### 5.7 Handling Unseen Modality Subsets (Cold Start)

If a new client joins with an owned subset \(S'\) that has never appeared before (e.g., a fourth hospital owning only \(\{\text{T1ce, T2}\}\)), there is no existing FusionHead\(_{S'}\)/Decoder\(_{S'}\) to pull. CAMFS initializes these by averaging the parameters of the smallest existing **superset** track(s) that contain \(S'\), restricted to the input channels in \(S'\) (channel-subsetting the fusion head's attention/concatenation dimensions), then fine-tunes locally for a short warm-up period before the new track enters normal federated aggregation. This is treated explicitly as a **practical patch**, not a solved problem — see Ablation A5 and Limitations.

### 5.8 From Group-Symmetric to Fully Directional Consent

The group-symmetric rule in §5.5 is a special case of the fully general \(R_{\text{recv}}(i,j,m)\) defined in §1.3 and §4. The architecture requires no structural change to support full directionality — only that the aggregation sum in eq. (5.5) be computed per-recipient \(i\) rather than once globally. The cost is purely communicational: instead of one global \(\theta_{\text{Encoder}_m}^{t+1}\), the server computes and stores up to \(N\) distinct per-recipient aggregates per modality per round. **Ablation A6** quantifies this cost against the accuracy benefit of true directionality.

### 5.9 Communication and Compute Overhead

Relative to a naive single-global-model FedAvg baseline, CAMFS's overhead scales with the number of distinct owned subsets \(|\{O(i) : i \in C\}|\) (each requiring its own FusionHead/Decoder) plus, under full directionality, a factor up to \(N\) in server-side aggregate storage. For the 3–4 hospital BraTS simulation this overhead is modest; it is flagged as a scalability question for federations with many distinct ownership patterns (see Limitations).

---

## 6. Experimental Setup

### 6.1 Datasets

- **BraTS2020** (primary, domain-fidelity + full-scale runs): 369 training cases with public ground truth, four co-registered, skull-stripped mpMRI sequences (T1, T1ce, T2, FLAIR) per case, standard BraTS label set.
- **BraTS2021** (scale-up + head-to-head comparison against FedAMM/DisentAFL baselines): 1,251 training cases with public ground truth (219 validation and 570 test cases exist but are not publicly labeled, so all reported splits are carved out of the 1,251 labeled training cases).

**Disclosed simplification:** BraTS's four "modalities" are four MRI *sequences* from a single scanner session, not four independent imaging technologies (unlike, e.g., MRI/PET/CT). This is stated explicitly, not left for a reviewer to discover, and is precisely why the paper's introduction and motivating scenario should lean on the qualitative hospital narrative while treating BraTS as the large-scale, comparable-to-prior-work benchmark rather than the sole evidence of real-world fidelity.

### 6.2 Simulated Institutional Split

Patient-level (not slice-level) disjoint split into three simulated hospitals, sized roughly proportional to plausible institution scale, with fixed owned subsets:

| Hospital | Owned subset \(O(i)\) | Rationale | Approx. share of patients |
|---|---|---|---|
| A — Contrast-Restricted Regional Center | {T2, FLAIR} | High pediatric/renal-impaired population; no gadolinium contrast ever acquired | 30% |
| B — Academic Comprehensive Center | {T1, T1ce, T2, FLAIR} | Full protocol, consents to full participation | 45% |
| C — Rural Low-Resource Site | {T1, FLAIR} | Scanner cannot run T2 or contrast sequences | 25% |

A fourth hospital **D — {T1ce, T2}** (an uncommon combination) is added only for Ablation A2/A5 (scale and cold-start), not the main comparison, to avoid confounding the primary three-way result.

Patients are split disjointly (no patient appears in more than one simulated hospital) to reflect genuine separate institutional populations; a shared-pool variant (same patients, different modality "views" withheld per simulated hospital) is run as a controlled check in Ablation A2 to isolate the effect of the consent-routing mechanism from confounding population shift.

### 6.3 Preprocessing

Standard BraTS-provided preprocessing is used as-is (co-registration to a common anatomical template, skull-stripping, 1mm³ isotropic resampling). Additional steps: per-scan z-score intensity normalization, cropping to the non-zero brain region, 2D axial slice extraction with empty-background slices discarded, patient-level train/val/test split (e.g., 70/10/20 within each simulated hospital's patient pool).

### 6.4 Baselines

| Baseline | What it represents |
|---|---|
| Local-only | Each hospital trains independently, no federation — lower bound |
| FedAvg-all (policy-ignoring) | Pools everything regardless of consent — upper bound on accuracy, **not** a valid deployable comparison, included only to quantify the compliance cost |
| FedMFS / MAFS-style | Send-side masking robustness baseline |
| FedAMM-style | Per-combination prototypes, single global model |
| Modality-subset clustering (MMiC-style, symmetric) | Groups by combination, no directional or policy-driven asymmetry |
| DisentAFL-style learned gate ("soft consent") | Learned, performance-optimized asymmetric routing — the key baseline for showing what a non-consent-aware method *would* do |
| RELIEF-style hard cohort isolation (availability-only) | Hard isolation mechanism identical in spirit to Approach 3, but gated purely by data availability, with no permission layer or audit artifact |
| Oracle | Single model trained as if every hospital had every modality (not achievable in practice — theoretical ceiling) |
| **CAMFS (proposed)** | Hard-isolated bank + subset tracks, gated by explicit, auditable \(R_{\text{send}}, R_{\text{recv}}\) |

### 6.5 Metrics

- **Dice score** and **HD95**, reported per composite region (WT, TC, ET), per simulated hospital.
- **Compliance-Performance Gap** (proposed metric): \(\text{Gap} = \text{Dice}_{\text{Oracle}} - \text{Dice}_{\text{CAMFS}}\), reported per hospital — the explicit "price of auditable compliance."
- **Consent-violation rate** (for baselines without hard gating): fraction of gradient/representation signal in a restricted client's pathway attributable to a forbidden modality, measured via a **purity probe** — a linear classifier trained to predict, from a client's encoder or fused embeddings, whether a forbidden modality was present anywhere upstream in that pathway's training history. High probe accuracy on a supposedly-restricted client's embeddings indicates leakage.
- **Representation drift:** cosine distance / CKA similarity between a client's Encoder\(_m\) embeddings and a reference Encoder\(_m\) trained in complete isolation from any non-consented client, quantifying contamination.

### 6.6 Implementation Details (to be finalized at run-time)

U-Net backbone with 4 down/up stages, base channel width 32 (doubling per stage); Adam optimizer; local epochs per round = 1–3 (swept); total federated rounds swept per compute budget; \(\lambda_1, \lambda_2\) swept in \(\{0, 0.1, 0.5, 1.0\}\) as part of Ablation A4. All hyperparameters to be reported in a final configuration table once tuning is complete — **placeholders only at this stage.**

---

## 7. Ablation Studies

### A1 — Routing Asymmetry Sweep
**Question:** How does Dice per hospital change as \(R_{\text{recv}}\) moves from fully symmetric (group-level) to maximally asymmetric (fully directional, per-pair)?
**Setup:** Fix owned subsets; vary only the consent matrix across 4–5 configurations from fully open (equivalent to FedAvg-all) to fully closed (equivalent to Local-only), with CAMFS's group-symmetric default and 1–2 genuinely directional configurations in between.
**Expected use:** Establishes that CAMFS's accuracy sits on the compliance-respecting side of this sweep by construction, and quantifies how much of the Oracle-to-Local gap is recoverable at each consent level.

### A2 — Number of Clients / Ownership Diversity
**Question:** Does the architecture's benefit hold as the number of distinct owned subsets grows (3 → 4 with Hospital D → further synthetic splits)?
**Setup:** Add Hospital D ({T1ce, T2}) and, synthetically, additional subset patterns by further partitioning Hospital B's data into smaller sub-populations with narrower subsets.
**Also includes the shared-pool control:** re-run the 3-hospital split with the *same* patients but different modality views withheld per hospital, to separate the effect of population disjointness from the effect of the consent mechanism itself.

### A3 — Hard Consent vs. Soft/Learned Gating ("Price of Auditable Compliance")
**Question:** Does a performance-optimized soft gate (DisentAFL-style) actually violate the stated consent constraint when given the chance, and how much accuracy does CAMFS sacrifice by refusing to?
**Setup:** Train the DisentAFL-style baseline with **no** consent constraint programmed in — only a performance objective — on the same three-hospital split. Measure, via the purity probe (§6.5), whether Hospital A's pathway ends up carrying detectable T1ce-attributable signal. If it does, record the Dice improvement this leakage bought the soft baseline, and compare directly to CAMFS's Dice under the enforced constraint. This is the paper's central empirical claim: **hard consent is a different problem from hard isolation-for-availability, and the difference is measurable.**

### A4 — Effect of Unimodal Alignment Loss (\(\lambda_1\)) on Representation Purity
**Question:** Does the unimodal alignment term reduce cross-track contamination, independent of the hard architectural isolation already in place?
**Setup:** Sweep \(\lambda_1 \in \{0, 0.1, 0.5, 1.0\}\); measure purity-probe accuracy and CKA similarity to an isolated reference encoder at each value.
**Rationale:** Structural isolation alone prevents *parameter* leakage; this ablation checks whether the *representations themselves* still drift toward encoding cross-modal correlations picked up indirectly (e.g., through label co-occurrence), which the alignment term is designed to suppress.

### A5 — Cold-Start Behavior for an Unseen Subset
**Question:** How well does the superset-averaging initialization (§5.7) perform for Hospital D's novel {T1ce, T2} track, compared to training that track entirely from scratch?
**Setup:** Compare Dice trajectory over federated rounds for (a) cold-start-initialized Hospital D track vs. (b) from-scratch Hospital D track vs. (c) Hospital D trained fully locally (no federation benefit at all).
**Expected use:** Quantifies how much the cold-start patch actually helps, and how many rounds it takes to converge — directly informs the honesty of the Limitations section's claim that this is "a patch, not a solved problem."

### A6 — Group-Symmetric vs. Fully Directional Consent (§5.8 Reconciliation)
**Question:** What is the practical accuracy and communication-cost difference between the simpler group-symmetric special case and full per-pair directionality?
**Setup:** Construct one scenario where true directionality matters (e.g., two T1ce-owning academic centers, B and a synthetic B′, are mutually competitive and refuse to pool with each other specifically, while both are willing to pool with a third, non-competing site). Compare CAMFS-group-symmetric (forced to either pool all T1ce owners or none) against CAMFS-directional (correctly expresses the asymmetric refusal).
**Expected use:** Demonstrates when the added complexity of full directionality (§5.8) is actually necessary versus when the simpler default suffices.

---

## 8. Results — Reporting Template (Not Yet Populated)

**No experiments have been run yet.** The tables below are the structure results should be reported in once the experimental plan in Sections 6–7 is executed; all cell values are placeholders and must not be read as findings.

### 8.1 Main Comparison (BraTS2020, 3-hospital split)

| Method | Hospital A Dice (WT/TC/ET) | Hospital B Dice (WT/TC/ET) | Hospital C Dice (WT/TC/ET) | Mean HD95 | Consent-violation rate |
|---|---|---|---|---|---|
| Local-only | *TBD* | *TBD* | *TBD* | *TBD* | 0% (no sharing) |
| FedAvg-all | *TBD* | *TBD* | *TBD* | *TBD* | *TBD (expected high)* |
| FedAMM | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| DisentAFL-style (soft) | *TBD* | *TBD* | *TBD* | *TBD* | *TBD — central to A3* |
| RELIEF-style (hard, availability-only) | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **CAMFS (proposed)** | *TBD* | *TBD* | *TBD* | *TBD* | **Expected 0% by construction** |
| Oracle | *TBD* | *TBD* | *TBD* | *TBD* | n/a |

### 8.2 Compliance-Performance Gap (per hospital)

| Hospital | Oracle Dice | CAMFS Dice | Gap |
|---|---|---|---|
| A | *TBD* | *TBD* | *TBD* |
| B | *TBD* | *TBD* | *TBD* |
| C | *TBD* | *TBD* | *TBD* |

### 8.3 Ablation Summary Table

| Ablation | Key metric | Expected qualitative trend (hypothesis, not result) |
|---|---|---|
| A1 (asymmetry sweep) | Dice vs. openness level | Monotonic increase in Dice as consent opens, CAMFS sits strictly below FedAvg-all and strictly above Local-only |
| A2 (client scale) | Dice vs. #subsets | Graceful degradation, not collapse, as subset diversity grows |
| A3 (hard vs. soft) | Leakage-driven Dice gain vs. Gap | Soft baseline shows nonzero purity-probe accuracy (leakage); CAMFS shows near-chance probe accuracy |
| A4 (\(\lambda_1\) sweep) | Purity probe accuracy vs. \(\lambda_1\) | Probe accuracy decreases as \(\lambda_1\) increases, task Dice roughly stable until \(\lambda_1\) too large |
| A5 (cold start) | Rounds-to-convergence | Cold-start track converges faster than from-scratch, slower than a mature track |
| A6 (directional vs. group) | Dice + communication overhead | Directional variant recovers accuracy group-symmetric variant cannot, at higher server-side cost |

---

## 9. Discussion

The central empirical bet of this paper is Ablation A3: if the soft/learned baseline does **not** actually leak forbidden signal in practice (i.e., the purity probe finds nothing), the paper's motivating distinction — that hard consent is a meaningfully different problem from hard isolation-by-availability — is considerably weaker, and the contribution shrinks to "an auditability wrapper around an existing mechanism." That is still a legitimate, if more modest, contribution (auditability itself has real deployment value in regulated healthcare settings), but the framing and abstract would need to shift accordingly. This should be treated as a go/no-go checkpoint early in experimentation, not discovered late.

---

## 10. Limitations and Open Problems

- **No convergence theory.** RELIEF provides a convergence bound for its (availability-based) hard-isolation scheme; CAMFS's directional, permission-based variant has no equivalent proof here. This is flagged as future work, not resolved.
- **Policy matrix privacy leakage.** The consent matrix itself, if published for auditability, could leak sensitive information about which institutions refuse to collaborate with which — a governance question, not just a technical one.
- **Free-rider dynamics.** A hospital consenting only to receive, never to send, is representable in \(R_{\text{send}}/R_{\text{recv}}\) but its game-theoretic and fairness implications are not analyzed here.
- **Cold start is a patch.** §5.7's superset-averaging initialization is a practical heuristic, not a principled solution; Ablation A5 quantifies but does not eliminate its cost.
- **BraTS pseudo-modality caveat.** All four "modalities" are MRI sequences from one scanner session, not independent imaging technologies — the paper's real-world motivation should not be read as fully validated by BraTS results alone.
- **Scalability of distinct subsets.** Overhead (§5.9) scales with the number of distinct owned subsets and, under full directionality, with \(N\); untested beyond a small number of simulated hospitals here.

---

## 11. Broader Impact and Ethical Considerations

This work is motivated by, and intended to support, real institutional constraints around data governance in healthcare — not to work around them. A system that makes non-sharing verifiable rather than merely assumed is intended to *increase* institutional willingness to participate in federated learning, particularly for sites serving vulnerable populations (pediatric, renal-impaired) where contrast-agent-related constraints are already a matter of clinical ethics, not preference. A misuse risk worth naming explicitly: the same hard-routing machinery could, in principle, be repurposed to enforce exclusionary policies unrelated to genuine clinical or legal constraints (e.g., encoding discriminatory data-sharing refusals as "consent"). The paper should state plainly that the architecture enforces whatever policy it is given, and is not itself a guarantee that the policy is ethically sound — that judgment remains with the institutions and governance bodies authoring the matrix.

---

## 12. Conclusion

We formalize consent-constrained federated learning as distinct from the more heavily studied performance-constrained (missing-modality-robust) setting, and instantiate a working architecture — a shared unimodal encoder bank combined with hard-isolated, subset-specific fusion tracks, gated by an explicit and auditable institutional consent matrix — on a clinically motivated, three-to-four-hospital BraTS federation. We are explicit that the underlying aggregation mechanism echoes recent independent work (DisentAFL, RELIEF, FedMM-style hard gating); our claimed contribution is the problem framing, the policy-as-artifact formalization, and an experimental protocol — centered on Ablation A3 — designed to measure whether "hard consent" is empirically distinguishable from "hard isolation for availability," and at what accuracy cost compliance is bought.

---

## References

*(Author/venue detail as established in the preceding literature analysis; verify all entries against current preprint/publication status before submission, as several are recent 2025–2026 preprints likely to have moved.)*

1. FedMFS — send-side modality masking for federated learning under missing modalities.
2. MAFS — modality-adaptive federated segmentation.
3. FedProto — prototype-based federated learning for heterogeneous clients.
4. MFCPL — multimodal federated contrastive prototype learning.
5. FedCMD — unimodal-encoder, combination-aware federated fusion.
6. FedAMM — per-combination centroid prototypes for federated multimodal learning.
7. MMiC — modality-combination clustering for federated learning.
8. CreamFL, FedMAC — cross-modal contrastive alignment under federated partial-modality settings.
9. Task Arithmetic; FedMSplit — modular composition of modality-specific parameters.
10. DisentAFL (AAAI 2024) — disentangled, asymmetric gated knowledge routing in federated learning.
11. RELIEF (arXiv:2604.04243, 2026) — turning missing modalities into training acceleration via cohort-isolated aggregation for federated learning on heterogeneous IoT edge devices.
12. FedMM — federated multi-modal learning with modality heterogeneity in computational pathology.
13. Role-aware multi-modal federated learning system for phishing webpage detection (arXiv:2509.22369) — hard-gated expert selection inspired by FedMM.
14. BraTS 2020/2021 challenge datasets and associated benchmark papers (Menze et al.; Bakas et al.; Baid et al.) — dataset description, preprocessing protocol, and standard WT/TC/ET evaluation convention.

---

*End of working draft. Sections 8 (Results) and the final hyperparameter table in §6.6 are explicitly placeholders pending actual experimental runs — do not present populated-looking numbers as findings until real experiments have been executed.*
