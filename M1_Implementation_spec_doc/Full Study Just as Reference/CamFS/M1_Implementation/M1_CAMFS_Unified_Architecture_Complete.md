# CAMFS: Consent-Aware Multimodal Federated Segmentation
## Complete Unified Architecture Specification (Method 2 + Send-Gated Patch)

> **Idea Source:** Professor Jerry  
> **Researcher:** Harsh Yadav  
> **Date:** July 2026  
> **Domain:** Multimodal Federated Learning — Consent-Constrained Medical Image Segmentation  
> **Dataset:** BraTS 2020/2021 (Brain Tumor Segmentation)

---

## Table of Contents

1. [Problem Statement & Motivation](#1-problem-statement--motivation)
2. [Literature Gaps](#2-five-literature-gaps)
3. [Formal Requirements](#3-three-formal-requirements)
4. [Notation](#4-notation)
5. [Component Inventory](#5-component-inventory)
6. [Phase 1 — Unimodal Contrastive Alignment](#6-phase-1--unimodal-contrastive-alignment)
7. [Phase Transition — Freeze Procedure](#7-phase-transition--freeze-procedure)
8. [Phase 2 — Track-Isolated Fusion Training](#8-phase-2--track-isolated-fusion-training)
9. [Send-Gated Track Routing (The Patch Extension)](#9-send-gated-track-routing-the-patch-extension)
10. [Cold-Start & Lineage Rule](#10-cold-start--lineage-rule)
11. [Multi-Track Contribution](#11-multi-track-contribution)
12. [Formal Purity Guarantees](#12-formal-purity-guarantees)
13. [The 4-Client Experimental Federation](#13-the-4-client-experimental-federation)
14. [Experimental Plan](#14-experimental-plan)
15. [Communication Profile](#15-communication-profile)
16. [Open Questions & Deferred Items](#16-open-questions--deferred-items)

---

## 1. Problem Statement & Motivation

### 1.1 The Core Problem

In real-world healthcare federated learning, **missing modalities are not random accidents — they are institutional decisions.** A hospital may lack an MRI scanner (hardware constraint), may have ethics board restrictions against sharing PET-derived data (legal constraint), or may be bound by a Memorandum of Understanding that limits what imaging data it contributes to a collaboration (contractual constraint).

Every existing multimodal federated learning method treats modality routing as a **performance optimization problem**: route knowledge wherever it improves accuracy. CAMFS treats it as a **compliance problem**: route knowledge only where institutional policy permits, even when blocking would measurably improve performance.

### 1.2 Real-World Motivations

| Motivation | Example |
| :--- | :--- |
| **Hardware Absence** | A hospital without an MRI scanner should not have its model pulled toward MRI-driven representations — its patients will never be scanned that way. |
| **Ethics Board Clearance** | A hospital may have consent approval for PET and CT data but not MRI; it legally cannot use MRI-infused information, even indirectly through federated model updates. |
| **Institutional MoU Agreements** | Hospitals share only what their data-sharing agreements permit. A well-equipped hospital may absorb all knowledge but restrict what it exports — the flow is inherently asymmetric. |
| **Regulatory Compliance** | The EU AI Act (2025) and HIPAA scrutiny of FL model updates demand auditable, inspectable governance of what information enters a deployed clinical model. |

### 1.3 What Makes This Different from Existing Work

| Dimension | Existing Work | CAMFS |
| :--- | :--- | :--- |
| Who decides what to share? | Performance/efficiency metrics (Shapley, bandwidth) | **Client-defined institutional policy** (MoU, ethics, hardware) |
| Direction of concern | Send-side only | **Both send-side AND receive-side** |
| Type of missing modality | Partial (some samples missing a modality) | **Complete** (a client will *never* have that modality) |
| Aggregation | Symmetric, global | **Asymmetric, personalized per client** |
| Policy layer | None (learned gate or implicit) | **Explicit, exogenous, auditable consent matrix** |
| Routing optimization target | Maximize accuracy | **Enforce compliance** (accuracy is optimized *within* the constraint) |

---

## 2. Five Literature Gaps

| Gap | Description |
| :--- | :--- |
| **Gap 1 — Receive-side filtering** | Every existing paper considers only what a client *sends*. No one has formalized what a client is *permitted to receive* as an independent policy decision. |
| **Gap 2 — Asymmetric bidirectional gradient flow** | No paper models a training round where Hospital B accepts all modality signals, Hospital A accepts only $\{T2, \text{FLAIR}\}$-infused signals, and Hospital C accepts only $\{T1, \text{FLAIR}\}$-infused signals simultaneously. This is a directed graph of gradient flows, not symmetric averaging. |
| **Gap 3 — Gradient purity by modality** | When a client trains a joint encoder on multiple modalities, its gradient carries mixed modality signal. Even a T2 encoder gradient, trained in the presence of T1ce data through a shared fusion head, carries indirect T1ce signal. The architectural fix is strictly separate modality encoders with temporal isolation. |
| **Gap 4 — MoU/consent matrix as a formal FL component** | No paper has introduced a policy/consent matrix as a first-class architectural component sitting above the aggregation mechanism. The legal and institutional framing is entirely absent from FL literature. |
| **Gap 5 — Convergence under asymmetric aggregation** | FedAvg and FedProx convergence proofs assume symmetric, global aggregation. Asymmetric routing means each client receives a different personalized model — the convergence theory needs to be rebuilt. *(Deferred to future work.)* |

---

## 3. Three Formal Requirements

Given a federation of clients $C = \{1, \dots, N\}$, where each client $i$ has a fixed, structurally owned modality subset $O(i) \subseteq \mathcal{M}$ (complete missing-modality assumption), and an exogenous, non-learnable consent policy, the system must satisfy:

| # | Requirement | Description |
| :--- | :--- | :--- |
| **R1** | **Performance Under Compliance** | Achieve the best possible task performance (segmentation Dice/HD95) for every client, **subject to** never violating any specified consent constraint — even when violating it would improve performance. |
| **R2** | **Auditability** | Make the enforced constraint set **auditable** — inspectable by a third party (IRB, compliance officer) as an explicit, exogenous, versioned object. Not inferable only from training dynamics or learned gates. |
| **R3** | **Graceful Degradation** | A client whose exact modality subset has not been seen before should not need to train an entirely new pathway from nothing. Cold-start initialization should leverage existing compatible components. |

---

## 4. Notation

| Symbol | Meaning |
| :--- | :--- |
| $C = \{1, \dots, N\}$ | Federation clients (hospitals) |
| $\mathcal{M} = \{\text{T1, T1ce, T2, FLAIR}\}$ | Full modality universe (BraTS MRI sequences) |
| $O(i) \subseteq \mathcal{M}$ | Client $i$'s fixed owned modality subset |
| $\text{Send}(i) \subseteq O(i)$ | Modalities $i$ is willing to contribute to the federation |
| $\text{Recv}(i) \subseteq O(i)$ | Modalities $i$ wants pooled-in benefit for |
| $S, S'$ | A modality subset identifying a fusion track ($S'$ denotes a send-keyed track) |
| $\mathcal{I}(S') = \{i \in C : \text{Send}(i) = S'\}$ | Contributor set for track $S'$ |
| $C_{\text{cls}} = 4$ | Segmentation classes: $c \in \{0: \text{BG},\; 1: \text{NCR/NET},\; 2: \text{ED},\; 4: \text{ET}\}$ |
| $\text{Encoder}_m$, $\text{Encoder}_m^*$ | Modality-$m$ encoder; $^*$ denotes the frozen post-Phase-1 snapshot |
| $\text{Proto}_m^c$ | Global unimodal class-$c$ prototype for modality $m$ ($\in \mathbb{R}^{256}$) |
| $\text{FusionHead}_S$, $\text{Decoder}_S$ | Track-$S$ fusion module and decoder |
| $\text{FusedProto}_S^c$ | Fused class-$c$ prototype for track $S$ |
| $z_m(p)$ | Encoder-$m$ bottleneck feature at spatial location $p$ |
| $P_m$, $P_m^c$ | Sampled bottleneck locations for modality $m$; subset with ground-truth class $c$ |
| $\tau$ | Contrastive temperature (default: $0.1$) |
| $\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$ | Cosine similarity |
| $\text{detach}(\cdot)$ | Stop-gradient operator — forward value preserved, backward gradient path severed |
| $\text{Lineage}(\theta)$ | Union of all modality sets touched by any computation that ever shaped $\theta$ |
| $R_{\text{send}}(i, m)$ | Client $i$ may contribute modality-$m$ encoder updates |
| $R_{\text{recv}}(i, j, m)$ | Client $i$ may receive modality-$m$ signal originating from $j$ |
| $R_{\text{send}}^{\text{track}}(i, S')$ | Client $i$ may contribute to track $S'$'s fusion/decoder aggregate |
| $R_{\text{recv}}^{\text{track}}(i, j, S)$ | Client $i$ may receive track-$S$ signal from client $j$ |
| $R_{\text{contribute}}(i, S)$ | Superset-owning client $i$ opts in to help train non-native track $S$ |

**Well-formedness constraints:**
- $\text{Send}(i) \subseteq O(i)$: A client cannot send what it does not own.
- $\text{Recv}(i) \subseteq O(i)$: A client cannot receive benefit for a modality it does not own.
- Degenerate case $\text{Send}(i) = \text{Recv}(i) = O(i)$ recovers the original single-matrix behavior. The Send/Recv decomposition is a strict generalization.

---

## 5. Component Inventory

CAMFS consists of **6 architectural components**:

### 5.1 Unimodal Encoder Bank

Four independent encoders, one per modality: $\text{Encoder}_{\text{T1}}$, $\text{Encoder}_{\text{T1ce}}$, $\text{Encoder}_{\text{T2}}$, $\text{Encoder}_{\text{FLAIR}}$.

**Architecture:** Each is a 2D U-Net-style downsampling path with 4 conv-conv-downsample stages. Base channel width 32, doubling per stage:

| Stage | Input Resolution | Output Channels | Output Resolution |
| :--- | :--- | :--- | :--- |
| Input | $240 \times 240$ | 1 (raw grayscale) | $240 \times 240$ |
| Stage 1 | $240 \times 240$ | 32 | $120 \times 120$ |
| Stage 2 | $120 \times 120$ | 64 | $60 \times 60$ |
| Stage 3 | $60 \times 60$ | 128 | $30 \times 30$ |
| Stage 4 (Bottleneck) | $30 \times 30$ | 256 | $15 \times 15$ |

**Information conservation:** $1 \times 240 \times 240 = 57{,}600$ input values $\longrightarrow$ $256 \times 15 \times 15 = 57{,}600$ bottleneck values. Spatial resolution is traded for 256-dimensional semantic channel depth.

Each encoder produces:
- A **bottleneck embedding** $z_m \in \mathbb{R}^{256 \times 15 \times 15}$
- **Skip-connection feature maps** at each of the 4 resolutions (for the decoder)

**Scope:** Aggregated globally across every client that owns modality $m$ and consents ($R_{\text{send}}(i,m)=1$), independent of what else that client owns.

**Lifecycle:** Trainable in Phase 1; frozen and static from Phase 2 onward.

### 5.2 Unimodal Prototype Bank

$\text{Proto}_m^c$ for each of 4 modalities $\times$ 4 classes = **16 prototype vectors**, each in $\mathbb{R}^{256}$.

**Computation:**

$$P_m^c = \{p \in \{0, \dots, 14\}^2 \mid y_{\text{down}}(p) = c\}$$

$$\text{Proto}_m^c = \frac{1}{|P_m^c|} \sum_{p \in P_m^c} z_m(p) \quad \in \mathbb{R}^{256}$$

where $y_{\text{down}}$ is the ground-truth mask nearest-neighbor-downsampled to bottleneck resolution ($15 \times 15$).

**The "Shared Language" Property:** The InfoNCE contrastive loss (§6) forces all 4 unimodal encoders to project the *same anatomical class* into the *same region* of $\mathbb{R}^{256}$:

$$\text{Proto}_{\text{T1}}^c \approx \text{Proto}_{\text{T1ce}}^c \approx \text{Proto}_{\text{T2}}^c \approx \text{Proto}_{\text{FLAIR}}^c$$

This cross-modal alignment is what makes fusion possible in Phase 2 — different encoders speak the same "256-D language" about anatomy.

**Lifecycle:** Updated in Phase 1 only; frozen alongside the encoders at the phase transition.

### 5.3 Policy Matrices

All matrices are **exogenous, versioned, non-learned, and inspectable** — no matrix is shaped by loss gradients or training dynamics.

| Matrix | Governs | Default |
| :--- | :--- | :--- |
| $R_{\text{send}}(i, m)$ | Client $i$ may contribute modality-$m$ encoder updates | $1$ if $m \in \text{Send}(i)$ |
| $R_{\text{recv}}(i, j, m)$ | Client $i$ may receive modality-$m$ signal originating from $j$ | $1$ if $m \in O(i) \cap O(j)$ and both opted in |
| $R_{\text{send}}^{\text{track}}(i, S')$ | Client $i$ may contribute to track $S'$'s fusion/decoder aggregate | $1$ when $\text{Send}(i) = S'$ |
| $R_{\text{recv}}^{\text{track}}(i, j, S)$ | Client $i$ may receive track-$S$ signal from same-subset client $j$ | $1$ when $O(i) \supseteq S$ or $O(j) = S$ |
| $R_{\text{contribute}}(i, S)$ | Superset client $i$ opts in to help train non-native track $S$ | Default $0$ (opt-in only) |

### 5.4 Subset-Fusion Tracks

One track per distinct $\text{Send}(i)$ value present in the federation. Each track owns:

- **$\text{FusionHead}_S$** — Cross-attention module (default) or concatenation + $1 \times 1$ conv (ablation variant). Combines *exactly* the skip-connection features of the encoders in $S$, applied at each of the 4 resolutions plus the bottleneck.
- **$\text{Decoder}_S$** — Standard U-Net upsampling path from fused features to a $C_{\text{cls}}=4$-class segmentation map.
- **$\text{FusedProto}_S^c$** — Mean fused-bottleneck embedding per class, local to track $S$.

**Hard Isolation:** No parameter or gradient ever crosses between tracks. This is a structural invariant, not a soft constraint.

**Lifecycle:** Not instantiated in the computation graph during Phase 1. Trainable from Phase 2 onward.

### 5.5 Phase Controller

A server-side state machine managing the two-phase lifecycle:

```
Global State: { CURRENT_PHASE ∈ {1, 2}, round t, proto_history[m][c] }

On each round-end:
  if CURRENT_PHASE == 1:
      check stopping criterion (§6.5)
      if met: execute freeze procedure (§7); CURRENT_PHASE ← 2
  broadcast round instructions to clients:
      - which components to pull (Encoder/Proto only, vs. FusionHead/Decoder/FusedProto)
      - which loss terms to compute
```

**Generalized (per-track) lifecycle:** Each fusion track additionally has its own lifecycle state:

```python
Per-track state: { LIFECYCLE ∈ {WARMING, STABLE}, round t_local, seed_record }

On track S' creation:
    LIFECYCLE ← WARMING
    execute seed procedure, log to Provenance Ledger

Each round, per active track:
    if LIFECYCLE == WARMING:
        run L_fusion-align-only round (no task loss)
        if warm-start stopping criterion met: LIFECYCLE ← STABLE
    if LIFECYCLE == STABLE:
        run normal Phase-2 round (L_task + λ2 · L_fusion-align)
```

### 5.6 Provenance Ledger

An auditable log recording how every fusion track was initialized. Written *before* any seed operation executes. Satisfies Requirement R2 (auditability) for the cold-start pathway.

**Ledger Entry Format:**
```yaml
Track: S' = {T1, T2}
Created: round t0
Seed mechanism: Tier 2 (warm-start only; no Tier 1/1b source available)
Per-slot lineage:
    FusionHead_S'.T1_slot  → fresh_init
    FusionHead_S'.T2_slot  → fresh_init
    Decoder_S'             → fresh_init
Warm-start rounds: W = <swept>
Contributors (Send-gated): {i, ...}
Lineage check: PASS  (∅ ⊆ {T1,T2})
```

---

## 6. Phase 1 — Unimodal Contrastive Alignment

### 6.1 Purpose

Train the 4 unimodal encoders and build the 16-vector prototype bank. **No fusion module exists in the computation graph during this phase.** Each encoder trains independently using a contrastive loss that aligns its bottleneck embeddings with class prototypes from the same modality only.

### 6.2 Loss Function

For client $i$, modality $m \in O(i)$:

**InfoNCE Unimodal Alignment Loss:**

$$\mathcal{L}_{\text{unimodal-align}}(z_m) = -\frac{1}{|P_m|} \sum_{p \in P_m} \log \frac{\exp\left(\text{sim}(z_m(p),\; \text{Proto}_m^{c(p)}) / \tau\right)}{\sum_{c'=1}^{C_{\text{cls}}} \exp\left(\text{sim}(z_m(p),\; \text{Proto}_m^{c'}) / \tau\right)}$$

where:
- $z_m(p)$ is the 256-D bottleneck feature vector at spatial location $p$
- $c(p) = y_{\text{down}}(p)$ is the ground-truth class at location $p$ (from the downsampled mask)
- $\text{Proto}_m^{c(p)}$ is the **positive pair** — the prototype of the *same* class as location $p$
- The denominator sums over all 4 class prototypes — these are the negatives
- $\tau = 0.1$ is the contrastive temperature

**Intuition:** At each spatial location $p$, the loss pulls $z_m(p)$ toward the prototype of its ground-truth class $c(p)$ (positive pair) and pushes it away from prototypes of all other classes (negative pairs). Across all modalities and all clients, this creates the "shared 256-D language" where the same anatomical structure maps to the same prototype region regardless of which MRI sequence captured it.

**Total Phase 1 Client Loss:**

$$\mathcal{L}_i^{\text{Phase\;1}} = \lambda_1 \sum_{m \in O(i)} \mathcal{L}_{\text{unimodal-align}}(z_m)$$

### 6.3 Phase 1 Aggregation Equations

**Server-internal global aggregate (Encoder):**

$$\theta_{\text{Encoder}_m}^{t+1} = \sum_{\substack{i:\; m \in O(i),\\ R_{\text{send}}(i,m)=1}} \frac{n_i}{N_m} \; \theta_{\text{Encoder}_m, i}^{t}$$

**Personalized broadcast to client $k$ (Encoder):**

$$\theta_{\text{Encoder}_m \to k}^{t+1} = \sum_{\substack{j:\; m \in O(j),\; R_{\text{send}}(j,m)=1,\\ R_{\text{recv}}(k,j,m)=1}} \frac{n_j}{N_m^{(k)}} \; \theta_{\text{Encoder}_m, j}^{t}$$

$$N_m^{(k)} = \sum_{\substack{j:\; R_{\text{send}}(j,m)=1,\\ R_{\text{recv}}(k,j,m)=1}} n_j$$

**Evidence-weighted prototype aggregation:**

$$\text{Proto}_m^{c,t+1} \to k = \sum_{j \in (\cdot)} \frac{n_j^{c,m}}{N_m^{c,(k)}} \; \text{Proto}_{m,j}^{c,t}$$

where $n_j^{c,m} = |P_{m,j}^c|$ (count of spatial locations of class $c$ in client $j$'s bottleneck for modality $m$), using the identical $R_{\text{send}}/R_{\text{recv}}$-gated client set as the encoder broadcast.

**Key property for restricted-send clients:** A client like Hospital 2 with $R_{\text{send}}(2, \text{T1ce}) = 0$ still **downloads** the global $\text{Proto}_{\text{T1ce}}^c$ and $\text{Encoder}_{\text{T1ce}}$ to train locally. It just **never uploads** its T1ce encoder updates to the server. This is how it benefits from the pooled T1ce knowledge without contributing its own T1ce signal.

### 6.4 Stopping Criterion

Run a minimum $T_1$ rounds (candidate range: 20–50), then continue until prototype drift falls below threshold:

$$\text{Drift}_t = \frac{1}{4 \cdot C_{\text{cls}}} \sum_{m} \sum_{c} \|\text{Proto}_m^{c,t} - \text{Proto}_m^{c,t-1}\|_2 < \epsilon$$

for $K$ consecutive rounds. Defaults: $\epsilon = 0.01$, $K = 5$.

### 6.5 Phase 1 Pseudocode

```python
for round t = 1 .. (until stopping criterion met):
    for each client i in parallel:
        pull Encoder_m, {Proto_m^c} for m in O(i), gated by R_send/R_recv
        for m in O(i):
            z_m ← Encoder_m(x_m)                    # bottleneck: [256 × 15 × 15]
            Proto_m_local^c ← mean(z_m[p] for p where y_down(p) == c)
            loss_m ← L_unimodal_align(z_m, {Proto_m^c})
        loss ← λ1 * Σ_m loss_m
        backprop through Encoder_m only
        # NO Decoder, NO FusionHead exist in graph
        local SGD steps (Adam, lr swept, 1-3 local epochs)
        push updated Encoder_m deltas (m where R_send(i,m)=1)
        push local Proto_m^c estimates + evidence counts n_i^{c,m}

    server:
        aggregate Encoder_m per §6.3
        aggregate Proto_m^c per §6.3 (evidence-weighted)
        Phase Controller: check stopping criterion
```

---

## 7. Phase Transition — Freeze Procedure

Executed **once** by the Phase Controller when the stopping criterion is met. This is the critical architectural moment that closes the gradient leakage channel identified in Method 1 (Flaw 8).

### The 5-Step Freeze Procedure

1. **Server finalizes snapshot:** Server records $\theta_{\text{Encoder}_m}^*$ and $\text{Proto}_m^{c,*}$ for all $m$, $c$ — the last Phase 1 aggregate becomes the permanent, immutable snapshot.

2. **Final gated broadcast:** Server broadcasts $\text{Encoder}_m^*$ to every client currently permitted to receive it (same $R_{\text{recv}}$ gating as the final Phase 1 round). **This is the last time an encoder is ever transmitted.**

3. **Client-side gradient stopping & detachment:** Each client sets `requires_grad=False` on all locally-held encoder parameters **AND** wraps every encoder forward call with `detach()`:
   $$z_m = \text{detach}(\text{Encoder}_m^*(x_m))$$
   The guarantee holds at the **computation graph level**, not only via optimizer configuration.

4. **Unimodal Bank locked:** The Unimodal Bank is now static for the remainder of training. No further Phase 1 rounds occur.

5. **Phase transition & track instantiation:** Phase Controller sets `CURRENT_PHASE ← 2`. Fusion tracks are instantiated (freshly initialized or cold-started per §10).

### Zero Gradient Leakage Proof

After the freeze, for any task loss computed in Phase 2:

$$\frac{\partial \mathcal{L}_{\text{task}}}{\partial \theta_{\text{Encoder}_m}^*} = \frac{\partial \mathcal{L}_{\text{task}}}{\partial z_m} \cdot \underbrace{\frac{\partial z_m}{\partial \theta_{\text{Encoder}_m}^*}}_{= \; 0 \;\text{(detach)}} \equiv \mathbf{0}$$

The `detach()` operation severs the gradient path at the computation graph level. Even if `requires_grad=False` were accidentally omitted, no gradient can reach the frozen encoders.

---

## 8. Phase 2 — Track-Isolated Fusion Training

### 8.1 Purpose

Train independent fusion heads and decoders for each modality-subset track. Each track produces a segmentation map using only the frozen encoder outputs from its constituent modalities. **No parameter or gradient crosses between tracks.**

### 8.2 Forward Pass (Phase 2)

For client $i$ with track $S$:

```
for m in S:
    z_m = detach(Encoder_m*(x_m))           # frozen encoder, detached output
z_S = FusionHead_S({z_m : m in S})          # cross-attention over skip features
y_hat = Decoder_S(z_S)                       # upsample to segmentation map
```

The `detach()` wall ensures the backward pass from the task loss terminates at the encoder output — no gradient reaches the frozen encoder parameters.

### 8.3 Loss Function

$$\mathcal{L}_i^{\text{Phase\;2}} = \mathcal{L}_{\text{task}}(y, \hat{y}) + \lambda_2 \cdot \mathcal{L}_{\text{fusion-align}}(z_S, \{\text{FusedProto}_S^c\}_c)$$

where:

- $\mathcal{L}_{\text{task}}$: Standard segmentation loss (Dice + Cross-Entropy) on the $C_{\text{cls}}=4$-class output.
- $\mathcal{L}_{\text{fusion-align}}$: InfoNCE contrastive loss in the fused embedding space:

$$\mathcal{L}_{\text{fusion-align}} = -\frac{1}{|P_S|} \sum_{p \in P_S} \log \frac{\exp(\text{sim}(z_S(p),\; \text{FusedProto}_S^{c(p)}) / \tau)}{\sum_{c'} \exp(\text{sim}(z_S(p),\; \text{FusedProto}_S^{c'}) / \tau)}$$

Note: $\lambda_1$'s term is **absent by construction** — there is no encoder parameter left to regularize in Phase 2.

### 8.4 Phase 2 Aggregation Equations (Send-Keyed)

Under the unified architecture, track aggregation is keyed by $\text{Send}(i)$, not $O(i)$:

**Server aggregate for track $S'$:**

$$\theta_{\text{FusionHead}_{S'}}^{t+1} = \sum_{\substack{i:\; \text{Send}(i) = S',\\ R_{\text{send}}^{\text{track}}(i, S') = 1}} \frac{n_i}{N_{S'}} \; \theta_{\text{FusionHead}_{S'}, i}^{t}$$

**Personalized broadcast to client $k$:**

$$\theta_{\text{FusionHead}_{S'} \to k}^{t+1} = \sum_{\substack{j:\; \text{Send}(j) = S',\; R_{\text{send}}^{\text{track}}(j, S') = 1,\\ R_{\text{recv}}^{\text{track}}(k, j, S') = 1}} \frac{n_j}{N_{S'}^{(k)}} \; \theta_{\text{FusionHead}_{S'}, j}^{t}$$

Identical structure applies to $\text{Decoder}_{S'}$ and $\text{FusedProto}_{S'}^c$.

### 8.5 Phase 2 Pseudocode

```python
# Phase Controller broadcasts frozen Encoder_m* once (§7); not repeated.

for round t = 1 .. T_max:
    for each client i in parallel:
        S' ← Send(i)    # track keyed by send-set, not ownership
        pull FusionHead_S', Decoder_S', FusedProto_S'^c
             gated by R_send^track / R_recv^track
        local_train():
            for m in S': z_m ← detach(Encoder_m*(x_m))   # frozen, detached
            z_S' ← FusionHead_S'({z_m : m in S'})
            y_hat ← Decoder_S'(z_S')
            loss ← L_task(y, y_hat) + λ2 · L_fusion-align(z_S', FusedProto_S'^c)
            backprop → updates FusionHead_S', Decoder_S' only
            local SGD steps
        push updated FusionHead_S', Decoder_S' deltas (if R_send^track(i,S')=1)
        push local FusedProto_S'^c estimates

    server:
        for each track S': aggregate FusionHead_S', Decoder_S' per §8.4
        update FusedProto_S'^c as evidence-weighted average
```

### 8.6 Combined Full Training Loop

```python
PHASE_CONTROLLER.CURRENT_PHASE ← 1

while CURRENT_PHASE == 1:
    run one Phase 1 round (§6.5)
    if stopping criterion met (§6.4):
        execute freeze procedure (§7)
        CURRENT_PHASE ← 2

while CURRENT_PHASE == 2 and t < T_max:
    run one Phase 2 round (§8.5)
    t += 1

# INVARIANT: No round ever executes both phases' losses simultaneously.
# INVARIANT: No parameter is ever active in both phases.
```

---

## 9. Send-Gated Track Routing (The Patch Extension)

### 9.1 The Gap Addressed (Flaw 15)

The base Method 2 architecture assumes $\text{Send}(i) = O(i)$ — a client contributes updates for every modality it owns. But a real institution may want to:

> "I own T1ce and use it for my local patients, but I refuse to let my T1ce-derived signal leave my institution (ethics restriction). I still want the benefit of the pooled T1ce encoder from other consenting hospitals."

This creates a policy triple: $O(i) = \{T1, T1ce, T2\}$, $\text{Send}(i) = \{T1, T2\}$, $\text{Recv}(i) = \{T1, T1ce, T2\}$.

### 9.2 Why the Unimodal Layer Already Handles This

At the encoder level (Layer 1), $\text{Encoder}_{\text{T1ce}}$'s aggregation is gated by $R_{\text{send}}(j, \text{T1ce})$. If client $i$ has $R_{\text{send}}(i, \text{T1ce}) = 0$, it never enters the aggregation sum — its T1ce data cannot shape the global encoder. But $i$ can still *pull* the frozen $\text{Encoder}_{\text{T1ce}}^*$ afterward (gated by $R_{\text{recv}}$) to encode its own T1ce scans locally. **No contradiction at Layer 1.**

### 9.3 Why the Fusion Layer (Layer 2) Was Broken

At the fusion track level, with cross-attention (or any nonlinear fusion), $\partial\mathcal{L}_{\text{task}} / \partial\theta_{\text{FusionHead}_S}$ is a **joint function of every input modality simultaneously**. There is no way to decompose an update to $\text{FusionHead}_S$ into "the part attributable to T1/T2" versus "the part attributable to T1ce." Both settings of the single flag violate the client's policy:

- $R_{\text{send}}^{\text{track}}(i, S) = 1$: Client's full, T1ce-entangled update reaches peers. **Violates send restriction.**
- $R_{\text{send}}^{\text{track}}(i, S) = 0$: Client contributes nothing, including the T1/T2 signal it was willing to share. **Over-restrictive.**

### 9.4 The Solution: Track Re-Keying

A client with $\text{Send}(i) \subsetneq O(i)$ is re-keyed into a **different, smaller, dedicated track** for the send side. Rather than contributing to Track $\{T1, T1ce, T2\}$, it contributes to Track $S' = \text{Send}(i) = \{T1, T2\}$.

**Contribution Path** (client $i$, $\text{Send}(i) = S' = \{T1, T2\}$):

```python
for each round t in Phase 2:
    pull FusionHead_S', Decoder_S'
    local_train():
        z_T1 = detach(Encoder_T1*(x_T1))
        z_T2 = detach(Encoder_T2*(x_T2))
        # x_T1ce NEVER enters this subgraph — no input slot exists for it
        z_S' = FusionHead_S'({z_T1, z_T2})
        y_hat = Decoder_S'(z_S')
        loss = L_task(y, y_hat) + λ2 · L_fusion-align(z_S', FusedProto_S'^c)
        backprop → updates FusionHead_S', Decoder_S' only
    push FusionHead_S', Decoder_S' deltas
    push local FusedProto_S'^c estimate
```

T1ce is **absent**, not suppressed — there is no input slot to plug it into.

### 9.5 Receive-Side Reconnection: Two Options

**(a) Pull-Only (lighter, conditional on a peer existing):**  
If a peer $j$ with $\text{Send}(j) = \{T1, T1ce, T2\}$ fully sends to Track $S$, client $i$ can set $R_{\text{recv}}^{\text{track}}(i, j, S) = 1$ and use $j$'s $\text{FusionHead}_S / \text{Decoder}_S$ **at inference only** — never trains against it, never uploads anything derived from it.

**(b) Personalized Local Head (default, always available):**  
Client $i$ trains a **private** $\text{FusionHead}_{O(i),i}^{\text{local}}$ and $\text{Decoder}_{O(i),i}^{\text{local}}$ using all its frozen encoders ($\text{Encoder}_{\text{T1}}^*$, $\text{Encoder}_{\text{T2}}^*$, $\text{Encoder}_{\text{T1ce}}^*$), trained purely on $i$'s own data. **Never transmitted, never aggregated.** Safety is definitional — purely local computation has always been safe in this architecture.

---

## 10. Cold-Start & Lineage Rule

### 10.1 The Lineage Rule

When a new track $S'$ is created, its initial parameters must come from somewhere. The **Lineage Rule** prevents contaminated initialization:

$$\theta \text{ is a valid seed for track } S' \iff \text{Lineage}(\theta) \subseteq S'$$

where $\text{Lineage}(\theta)$ is the union of all modality sets touched by any computation that ever shaped $\theta$'s current value — including direct training *and* any transfer/inheritance operation.

**Why naive superset seeding fails:** Copying weights from Track $\{T1, T1ce, T2, \text{FLAIR}\}$ and channel-subsetting to $\{T1, T2\}$ preserves weights that were *learned jointly with T1ce and FLAIR present*. The retained weights encode functional dependence on excluded modalities without those modalities' tensors ever appearing again. $\text{Lineage} = \{T1, T1ce, T2, \text{FLAIR}\} \not\subseteq \{T1, T2\}$. **Rejected.**

### 10.2 Four Cold-Start Tiers

| Tier | Name | Condition | Mechanism |
| :--- | :--- | :--- | :--- |
| **Tier 1** | Grow from existing strict subset | Some track $S'' \subsetneq S'$ already exists and was validly seeded | Net2Net-style channel-widening: new input slots get fresh init; existing slots keep $S''$'s weights. $\text{Lineage}(S'') \subseteq S'' \subseteq S'$ by transitivity. |
| **Tier 1b** | Composable multi-source stitching | Pure single-modality or smaller subset tracks exist for individual input slots | Take T1 slot from $\{T1\}$-only track, T2 slot from $\{T2\}$-only track. Each slot independently satisfies $\text{Lineage} \subseteq S'$. |
| **Tier 2** | Track-local warm-start | Always available | Run $W$ bootstrap rounds using $\mathcal{L}_{\text{fusion-align}}$ only (no task loss), with frozen encoders. Structurally a second Phase Controller instance scoped to the new track. |
| **Tier 3** | Fresh random init (the floor) | Always available | Random initialization of $\text{FusionHead}_{S'}$ / $\text{Decoder}_{S'}$. Only retrains fusion-and-decode stage, not encoders (already frozen). Bounds worst-case cost. |

### 10.3 Server-Side Seed-Selection Procedure

```python
On track S' creation:
    if a valid Tier 1 source exists (some S'' ⊊ S', already validly seeded):
        seed via channel-widening from S''
    else if valid Tier 1b sources exist for some/all slots:
        seed those slots from their pure sources; fresh-init remaining slots
    always: run Tier 2 warm-start (fusion-align only, no task loss) for W rounds
    then: enter normal Phase-2 task-loss training
    log entire operation to Provenance Ledger
```

---

## 11. Multi-Track Contribution

A client $i$ with $O(i) \supset S$ for some existing track $S$ may, **only if it opts in** via $R_{\text{contribute}}(i, S) = 1$, additionally run a **masked forward/backward pass** — dropping every channel outside $S$ — through $\text{FusionHead}_S / \text{Decoder}_S$.

**Example:**

```python
Hospital 1 (owns {T1, T1ce, T2, FLAIR}), Phase 2, per round:
    pass 1 (native):  x_{T1,T1ce,T2,FLAIR} → FusionHead_S1 → Decoder_S1
    pass 2 (opt-in,   x_{T1,FLAIR}          → FusionHead_S3 → Decoder_S3
      if R_contribute(1, S3) = 1):
```

**Why this doesn't weaken the consent guarantee:** Track $S_3$'s isolation invariant is "parameters shaped only by $\{T1, \text{FLAIR}\}$-restricted signal." Hospital 1's masked pass provides exactly this — the T1ce and T2 data it also owns are simply not used in this pass. Default is $R_{\text{contribute}} = 0$ (opt-in only).

---

## 12. Formal Purity Guarantees

### 12.1 Encoder Purity (Method 2, §11)

**Claim:** $\theta_{\text{Encoder}_m}^*$ (the value broadcast to any client) is provably a function of Phase 1 data alone.

**Argument:**
- *Phase 1:* The only loss computed is $\mathcal{L}_{\text{unimodal-align}}(z_m, \text{Proto}_m^c)$, dependent only on $\text{Encoder}_m$ and same-modality prototypes. No fusion module or decoder exists in the computation graph. No cross-modal gradient path can exist.
- *Phase transition:* $\theta_{\text{Encoder}_m}^*$ is fixed permanently.
- *Phase 2:* Encoder outputs are wrapped with `detach()`. The backward pass terminates at the detach point. `requires_grad=False` provides a second safety layer. **Zero gradient reaches the frozen encoders.**

### 12.2 Send-Restricted Track Purity Lemma

**Setup:** $\mathcal{I}(S') = \{i \in C : \text{Send}(i) = S'\}$. For $i \in \mathcal{I}(S')$, the local Phase 2 pass is:

$$z_m = \text{detach}(\text{Encoder}_m^*(x_i^m)), \; m \in S' \;\to\; z_{S'} = \text{FusionHead}_{S'}(\{z_m\}) \;\to\; \hat{y} = \text{Decoder}_{S'}(z_{S'})$$

$$\mathcal{L}_i = \mathcal{L}_{\text{task}}(y, \hat{y}) + \lambda_2 \cdot \mathcal{L}_{\text{fusion-align}}(z_{S'}, \{\text{FusedProto}_{S'}^c\})$$

No $x_i^m$ for $m \notin S'$ appears anywhere. $\text{FusionHead}_{S'}$ takes exactly $|S'|$ inputs — there is no slot to plug an excluded modality into.

**Seed condition:** At round $t_0$, $\text{Lineage}(\theta^{t_0}) \subseteq S'$ (enforced via Provenance Ledger).

**Lemma (Send-Restricted Track Purity).** Given the standing honest-execution assumption and the seed condition, for every round $t \geq t_0$:

$\theta_{\text{FusionHead}_{S'}}^t$, $\theta_{\text{Decoder}_{S'}}^t$, $\text{FusedProto}_{S'}^{c,t}$ are, as mathematical functions, **invariant to** $x_i^m$ for every $i \in C$, $m \in \mathcal{M} \setminus S'$.

**Proof:**
1. **Forward-pass exclusion:** No node in the computation subgraph takes $m \notin S'$ data as input. So $\mathcal{L}_i$ and every $\nabla_\theta \mathcal{L}_i$ are constant in $x_i^m$ for $m \notin S'$.
2. **Aggregation preserves it:** $\theta_{\text{FusionHead}_{S'}}^{t+1} = \sum_{i \in \mathcal{I}(S')} w_i \cdot \theta_{\text{FusionHead}_{S'},i}^t$ — a weighted average of quantities each independent of excluded-modality data. The weights $n_i / N_{S'}$ are sample counts (not content), hence also independent. A weighted average of $v$-independent terms is $v$-independent.
3. **Induction:** Step 2 is the inductive step; the seed condition supplies a true base case at $t_0$. $\blacksquare$

**Corollary (Composition):** Combined with the encoder purity guarantee (§12.1), Track $S'$'s entire trained state — encoders it consumes, fusion head, decoder, fused prototype — carries **zero signal** from any modality outside $S'$, at every layer.

---

## 13. The 4-Client Experimental Federation

### 13.1 Client Matrix

| Client | Owned $O(i)$ | Send $\text{Send}(i)$ | Receive $\text{Recv}(i)$ | Joins At | Architectural Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hospital 1** | $\{T1, T1ce, T2, FLAIR\}$ | $\{T1, T1ce, T2, FLAIR\}$ | $\{T1, T1ce, T2, FLAIR\}$ | Phase 1 | **Full anchor.** Trains all 4 encoders. Native to Track $S_1$. Opts into Track $S_3$ via $R_{\text{contribute}}$. Tests: full participation, multi-track contribution, standard routing. |
| **Hospital 2** | $\{T1, T1ce, T2\}$ | **$\{T1, T2\}$** | $\{T1, T1ce, T2\}$ | Phase 1 | **Send-Gated star case.** Withholds T1ce on send. Re-keyed to Track $S_2'=\{T1,T2\}$. Trains private local head. Tests: Send-Gated routing, $\text{Recv} > \text{Send}$, personalized local head, track-level purity. |
| **Hospital 3** | $\{T1, FLAIR\}$ | $\{T1, FLAIR\}$ | $\{T1, FLAIR\}$ | Phase 1 | **Restricted-receive & purity probe target.** Only 2 modalities, structurally blocked from T1ce/T2. Tests: receive filtering, standard routing, minority track (beneficiary of $R_{\text{contribute}}$), Ablation A3 purity probe. |
| **Hospital 4** | $\{T1, T1ce, T2\}$ | $\{T1, T1ce, T2\}$ | $\{T1, T1ce, T2\}$ | **Phase 2 (delayed)** | **Dynamic cold-start joiner.** Track $S_4$ created dynamically. Seeded from stabilized $S_2'$ via Tier 1. Tests: cold-start, Lineage Rule, Provenance Ledger, Phase Controller per-track lifecycle. |

### 13.2 Tracks That Emerge

| Track | Subset | Native Contributors | Created At | Seed Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| $S_1$ | $\{T1, T1ce, T2, FLAIR\}$ | Hospital 1 | Phase 2 start | Tier 3 (fresh) + Tier 2 warm-start |
| $S_2'$ | $\{T1, T2\}$ | Hospital 2 (send-gated) | Phase 2 start | Tier 3 (fresh) + Tier 2 warm-start |
| $S_3$ | $\{T1, FLAIR\}$ | Hospital 3 + Hospital 1 ($R_{\text{contribute}}$) | Phase 2 start | Tier 3 (fresh) + Tier 2 warm-start |
| $S_4$ | $\{T1, T1ce, T2\}$ | Hospital 4 | Phase 2, **delayed** | **Tier 1** (from stabilized $S_2'$) + Tier 2 warm-start |

### 13.3 Phase 1 Encoder Training Participation

| Encoder | Trained By | Contributors |
| :--- | :--- | :--- |
| $\text{Encoder}_{\text{T1}}$ | H1, H2, H3 | 3 clients (T1 is the most shared modality) |
| $\text{Encoder}_{\text{T1ce}}$ | H1, H2 | 2 clients (H2 downloads but does NOT upload — $R_{\text{send}}(2, \text{T1ce}) = 0$) |
| $\text{Encoder}_{\text{T2}}$ | H1, H2 | 2 clients |
| $\text{Encoder}_{\text{FLAIR}}$ | H1, H3 | 2 clients |

**Note:** Hospital 2 trains $\text{Encoder}_{\text{T1ce}}$ locally using downloaded global prototypes, but never uploads its T1ce updates. Hospital 4 is absent from Phase 1 entirely — it receives frozen encoders when it joins.

### 13.4 BraTS Dataset Split

**Dataset:** BraTS 2020 (369 labeled training cases) or BraTS 2021 (1,251 labeled cases).

**Patient-level disjoint splitting:**
- Hospital 1: ~40% of patients (largest cohort — full-modality academic center)
- Hospital 2: ~25% of patients (3-modality center with T1ce send restriction)
- Hospital 3: ~20% of patients (2-modality rural site)
- Hospital 4: ~15% of patients (joins dynamically mid-training)

**Preprocessing:**
1. Standard BraTS preprocessing: co-registration, skull-stripping, 1mm³ isotropic resampling
2. Per-scan z-score intensity normalization
3. Crop to non-zero brain bounding box
4. 2D axial slice extraction (discarding empty background slices)
5. Patient-level train/val/test split within each hospital (70/10/20)

**Ground-truth classes:** $c \in \{0: \text{BG},\; 1: \text{NCR/NET},\; 2: \text{ED},\; 4: \text{ET}\}$

**Clinical evaluation sub-regions:**
- Whole Tumor (WT) $= 1 \cup 2 \cup 4$
- Tumor Core (TC) $= 1 \cup 4$
- Enhancing Tumor (ET) $= 4$

---

## 14. Experimental Plan

### 14.1 Baselines

| # | Baseline | Description |
| :--- | :--- | :--- |
| 1 | **Local-Only** | No collaboration. Each client trains independently on its own data. Lower bound. |
| 2 | **FedAvg-All** | All encoder weights aggregated symmetrically, ignoring consent policies. Non-compliant upper bound. |
| 3 | **FedMFS** | Shapley-based send-side modality selection. No receive-side filtering. |
| 4 | **MMiC-style Clustering** | Symmetric grouping by modality combination. Parameter substitution across clusters. |
| 5 | **DisentAFL-style Soft Gate** | Learned, accuracy-driven asymmetric routing. Opaque, non-auditable gate. **The primary comparison target.** |
| 6 | **RELIEF-style Hard Cohort** | Availability-based hard isolation without policy governance. |
| 7 | **FedAMM-style** | Per-combination centroid prototypes, weighted aggregation by modality proportion. Same dataset (BraTS). |
| 8 | **Oracle** | Single model trained as if all hospitals had all modalities. Theoretical ceiling. |

### 14.2 Evaluation Metrics

| Metric | Description |
| :--- | :--- |
| **Dice Score** | Per client, per region (WT, TC, ET). Primary metric. |
| **HD95** | 95th-percentile Hausdorff Distance per region. |
| **Compliance-Performance Gap** | $\text{Gap}(i) = \text{Dice}_{\text{Oracle}} - \text{Dice}_{\text{CAMFS}}(i)$. Per-client price of auditable compliance. |
| **Purity Probe Accuracy** | Linear classifier on bottleneck embeddings predicting whether a forbidden modality was present upstream. Should be at null-baseline for CAMFS, elevated for soft baselines. |
| **Representation Drift** | CKA similarity and cosine distance vs. isolated reference encoder. |

### 14.3 Ablation Studies

| # | Ablation | Question | Setup |
| :--- | :--- | :--- | :--- |
| **A1** | 1-Phase vs 2-Phase | Does two-phase training improve purity without sacrificing accuracy? | Train Method 1 (joint) vs Method 2 (two-phase) on same federation. |
| **A2** | Client Diversity | Does performance scale with ownership diversity? | 3-client (H1,H2,H3) vs 4-client (+H4). Shared-pool control variant. |
| **A3** | Hard vs Soft Gating (**THE KILLER EXPERIMENT**) | Does DisentAFL-style soft gating silently violate consent? | Purity probe on H3 and Track $S_2'$. If soft baseline leaks → strong empirical narrative. |
| **A4** | $\lambda_1$ Sweep | Does unimodal alignment strength affect representation purity? | Sweep $\lambda_1 \in \{0, 0.1, 0.5, 1.0\}$. |
| **A5** | Cold-Start Tiers | Does Tier 1 seeding outperform Tier 2/3? | Compare H4's Track $S_4$: Tier 1 (from $S_2'$) vs Tier 2 (warm-start only) vs Tier 3 (fresh init). |
| **A6** | Group-Symmetric vs Directional | What's the accuracy/communication trade-off of full directionality? | Compare group-symmetric default $R_{\text{recv}}$ vs per-pair directional $R_{\text{recv}}$. |
| **A7** | Multi-Track Contribution | Does $R_{\text{contribute}}$ help minority tracks? | Track $S_3$ Dice with $R_{\text{contribute}}(1, S_3) = 1$ vs $= 0$. |
| **A8** | Personalized Local Head | Does Option (b) outperform Option (a) pull-only? | Hospital 2 local-head Dice vs pull-only Dice vs send-track-only Dice. |

### 14.4 Purity Probe Protocol

**Purpose:** Empirically verify zero cross-modal leakage, and demonstrate that soft baselines (DisentAFL) *do* leak.

**Setup:**
1. Extract bottleneck embeddings $z_m$ from a target client's encoder.
2. Train a linear classifier (logistic regression) to predict whether a forbidden modality (e.g., T1ce for Hospital 3) was present anywhere upstream in that pathway's training history.
3. **Null baseline:** Probe accuracy on a clean/clean pair (same-anatomy, different random seed) — establishes the chance-level reference.
4. **Expected result for CAMFS:** Probe accuracy sits at the null baseline by construction (structural guarantee).
5. **Expected result for soft baselines:** Probe accuracy is elevated above null baseline → evidence of silent leakage.

**Anatomy-confound control:** Report probe accuracy *relative to* the null baseline, not absolute 50% chance, because anatomical features shared across modalities can inflate raw accuracy without indicating leakage.

---

## 15. Communication Profile

| Dimension | Phase 1 | Phase 2 |
| :--- | :--- | :--- |
| **Components transmitted** | $\text{Encoder}_m$, $\text{Proto}_m^c$ only | $\text{FusionHead}_{S'}$, $\text{Decoder}_{S'}$, $\text{FusedProto}_{S'}^c$ only (plus one final frozen-encoder broadcast at transition) |
| **Payload per round** | 4 encoders' worth, gated by modality ownership | Up to $|\{\text{Send}(i) : i \in C\}|$ tracks' worth, gated by subset membership |
| **Peak concurrent traffic** | Lower than Method 1 joint round (fusion/decoder payloads not yet moving) | Lower than Method 1 joint round (encoder payloads have stopped) |

Total communication volume across both phases combined is the same order as Method 1's single-phase training — Method 2 time-slices what gets synchronized rather than adding to it.

---

## 16. Open Questions & Deferred Items

| Item | Status | Impact on Implementation |
| :--- | :--- | :--- |
| Convergence theory under asymmetric aggregation | **Deferred** — flagged as future work | Paper-writing, not implementation |
| BraTS label annotation bias (ET annotated from T1ce) | **Known** — performance on ET region is T1ce-dependent | Disaggregate WT/TC/ET results; note in discussion |
| Graduated free-rider dynamics | **Deferred** — out of scope for honest execution | Analysis concern, not implementation |
| Cross-track prototype alignment | **Explicitly rejected** — would reopen cross-modal leaks | N/A |
| Cold-Start Tier 1b empirical validation | **Deferred** — requires single-modality sites not in BraTS | Theoretically proven via Lineage Rule; note in paper |
| Adversarial/malicious clients | **Out of scope** — honest execution assumption (A1) | N/A |
| Missing reference: `camfs_method2_theoretical_foundations.md` | **Not available** — contains convergence theory (Theorem 1, G1–G4) | No impact on implementation |

### Hyperparameter Defaults

| Parameter | Default / Range | Notes |
| :--- | :--- | :--- |
| $T_1$ (min Phase 1 rounds) | 20–50 (start at 30) | Use stopping criterion to extend |
| $\epsilon$ (proto-drift threshold) | 0.01 | Standard convergence-check default |
| $K$ (consecutive rounds below $\epsilon$) | 5 | Conservative |
| $\tau$ (contrastive temperature) | 0.1 | Standard InfoNCE default |
| $\lambda_1$ (unimodal alignment weight) | Sweep $\{0, 0.1, 0.5, 1.0\}$; start at 0.5 | |
| $\lambda_2$ (fusion alignment weight) | Sweep $\{0, 0.1, 0.5, 1.0\}$; start at 0.5 | |
| $W$ (Tier 2 warm-start rounds) | 5–10 | Use proto-drift criterion to govern |
| $T_{\text{max}}$ (Phase 2 total rounds) | Unspecified | Use validation-loss early stopping |
| Local epochs per round | 1–3 (start at 1) | |
| Optimizer | Adam | Learning rate swept |
| Input mode | 2D axial slices | 3D noted as optional extension |
| FusionHead variant | Cross-attention (default) | Concat+1×1 conv as ablation |

---

*Document version: July 2026*  
*Architecture status: Conceptually complete. Implementation-ready. No architectural blockers.*
