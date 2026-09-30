# Solution: Gap 6 / R4 — Consented Cross-Boundary Knowledge Absorption
## Complete Specification — Living Document

> **Status:** Working draft — iterate until ready for merger into [M2_CAMFS_Unified_Problem(R4,Gap6).md](file:///c:/Users/ihars/Downloads/Research/CamFS/M2_CAMFS_Unified/M2_CAMFS_Unified_Problem%28R4,Gap6%29.md)  
> **Approach:** D (Local Cross-Absorb Head) — sole MVP  
> **Deferred:** Approach C (Federated Prototype Reg), Approach B' (Self-Distillation Adapter)  
> **Date:** July 2026

---

## Table of Contents

1. [Problem Restatement](#1-problem-restatement)
2. [Solution Overview](#2-solution-overview)
3. [New Policy Matrices](#3-new-policy-matrices)
4. [New Prototype Variant](#4-new-prototype-variant--cross-exposed-prototype)
5. [The 7th Component: Cross-Boundary Absorption Layer](#5-the-7th-component-cross-boundary-absorption-layer)
6. [Formal Guarantees](#6-formal-guarantees)
7. [Provenance Ledger Extension](#7-provenance-ledger-extension)
8. [Updated Experimental Federation](#8-updated-experimental-federation)
9. [New Ablation & Probe Extensions](#9-new-ablation--probe-extensions)
10. [The Primary Empirical Risk](#10-the-primary-empirical-risk)
11. [Escalation Path: Approach B'](#11-escalation-path-approach-b-self-distillation-adapter)
12. [Deferred: Approach C Analysis](#12-deferred-approach-c-federated-prototype-regularization)
13. [Literature Positioning](#13-literature-positioning)
14. [What Changes vs What Doesn't](#14-what-changes-vs-what-doesnt)
15. [Pre-Implementation Checklist](#15-pre-implementation-checklist)
16. [Open Questions](#16-open-questions)

---

## 1. Problem Restatement

### 1.1 The Exact Constraint Being Relaxed

**M1 Well-Formedness Constraint:**

$$\text{Recv}(i) \subseteq O(i) \quad \text{— a client cannot receive benefit for a modality it does not own.}$$

**M2 Relaxation:**

$$\text{Recv}(i) \subseteq \mathcal{M} \quad \text{— a client may receive benefit for any modality, including non-owned, if explicitly consented.}$$

This creates the partition:

$$\text{Recv}_{\text{own}}(i) = \text{Recv}(i) \cap O(i) \qquad \text{Recv}_{\text{cross}}(i) = \text{Recv}(i) \setminus O(i)$$

When $\text{Recv}_{\text{cross}}(i) = \emptyset$ for all clients, this reduces **exactly** to M1 behavior.

### 1.2 The Concrete Scenario

Hospital 3 owns $O(3) = \{T1, \text{FLAIR}\}$. Its patients will *never* have T1ce scans. Yet Hospital 3 wants its segmentation model to benefit from T1ce-derived knowledge:

$$\text{Recv}(3) = \{T1, \text{FLAIR}, \text{T1ce}\}, \quad \text{Recv}_{\text{cross}}(3) = \{\text{T1ce}\}$$

**The challenge:** Hospital 3 has no T1ce input data ($x_{\text{T1ce}}$ doesn't exist for its patients), no T1ce encoder input slot in its fusion head, and no way to run $\text{Encoder}_{\text{T1ce}}^*$ at inference. The knowledge must transfer through a mechanism that requires only the **shared 256-D embedding space** — not raw modality data.

### 1.3 Why the Current Architecture Blocks This

| Blocker | Description |
| :--- | :--- |
| **No input slot** | $\text{FusionHead}_{S_3}$ accepts exactly 2 inputs (T1, FLAIR). There is no slot for T1ce. |
| **Track hard isolation** | "No parameter or gradient ever crosses between tracks" (§5.4). |
| **Purity Lemma** | §12.2 proves Track $S'$ parameters carry zero signal from $m \notin S'$. |
| **Cross-track proto alignment rejected** | §16 explicitly rejects cross-track prototype alignment — "would reopen cross-modal leaks." |

**All four blockers were designed for M1's stricter $\text{Recv}(i) \subseteq O(i)$ constraint.** Under M2's relaxation, the Purity Lemma for the *federated* track still holds exactly — but a *local-only* pathway is deliberately allowed to absorb consented cross-boundary signal.

---

## 2. Solution Overview

### 2.1 Design Philosophy

> **The M1 architecture is complete and correct for R1–R3. The solution for R4 is a bounded, additive 7th component that sits _downstream_ of everything else and touches nothing that already works.**

This follows the established **FedPer / FedRep / FedBABU** personalization paradigm in federated learning: a shared representation layer (encoders + fusion track — collaboratively trained, aggregated) paired with a personalized head (cross-absorb layer — trained locally, never uploaded). The specific pattern of providing detached prototypes from the shared layer to a private local head has a very close precedent in **pFedDKS (2026)**.

### 2.2 Component Inventory Update

| # | Component | Source | Aggregated? | New in M2? |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Unimodal Encoder Bank | §5.1 | Yes (Phase 1) | No |
| 2 | Unimodal Prototype Bank | §5.2 | Yes (Phase 1) | No |
| 3 | Policy Matrices | §5.3 | No (exogenous) | ⚠️ Extended |
| 4 | Subset-Fusion Tracks | §5.4 | Yes (Phase 2) | No |
| 5 | Phase Controller | §5.5 | No (server-side) | No |
| 6 | Provenance Ledger | §5.6 | No (audit log) | ⚠️ Extended |
| **7** | **Cross-Boundary Absorption Layer** | **This document** | **No (local only)** | **✅ NEW** |

---

## 3. New Policy Matrices

### 3.1 Why Not Overload `R_recv`

$R_{\text{recv}}(i, j, m)$ is **peer-indexed** — it governs "may client $i$ receive modality-$m$ signal from *this specific client $j$*." Cross-boundary absorption consumes a single, already-aggregated, anonymized artifact ($\text{Proto}_m^{c,\text{cross}}$), not a specific peer's contribution. The peer dimension is semantically meaningless for this relationship.

The architecture already has precedent for introducing new matrices when semantics differ: $R_{\text{send}}$ vs $R_{\text{send}}^{\text{track}}$ vs $R_{\text{contribute}}$ are all separate because they govern genuinely different consent decisions. Two new matrices follow this pattern.

### 3.2 Matrix Definitions

| Matrix | Shape | Governs | Default |
| :--- | :--- | :--- | :--- |
| $R_{\text{recv}}^{\text{cross}}(i, m)$ | $\|C\| \times \|\mathcal{M}\|$ | Client $i$ (with $m \notin O(i)$) may locally consume modality-$m$'s cross-exposed prototype for cross-boundary knowledge absorption | **$0$** (opt-in only) |
| $R_{\text{expose}}^{\text{cross}}(j, m)$ | $\|C\| \times \|\mathcal{M}\|$ | Client $j$ (already sending $m$ via $R_{\text{send}}$) additionally permits its modality-$m$ contribution to be used for cross-boundary absorption by non-owners | **$0$** (conservative, opt-in only) |

### 3.3 Well-Formedness Constraints

- $R_{\text{recv}}^{\text{cross}}(i, m) = 1 \implies m \notin O(i)$: Cross-absorb is only for modalities the client does not own. If $m \in O(i)$, use the existing $R_{\text{recv}}(i, j, m)$ pathway.
- $R_{\text{expose}}^{\text{cross}}(j, m) = 1 \implies R_{\text{send}}(j, m) = 1$: You must already be a contributor of modality $m$ to allow cross-boundary exposure of your contribution.
- Both matrices are **exogenous, versioned, non-learned, and inspectable** — same auditability standard as all existing policy matrices (§5.3).

### 3.4 The Two-Gate Consent Model

Cross-boundary absorption requires consent from **both sides**:

```
Receive-side:   R_recv_cross(i, m) = 1   →  "I (Hospital 3) want T1ce knowledge"
Send-side:      R_expose_cross(j, m) = 1  →  "I (Hospital 1) allow my T1ce proto
                                                to be consumed by non-T1ce-owners"

Proto_T1ce^{c,cross} is transmitted to Hospital 3 IFF:
    R_recv_cross(3, T1ce) = 1      AND
    ∃ j : R_expose_cross(j, T1ce) = 1
```

**Why the send-side gate matters:** $R_{\text{send}}(1, \text{T1ce}) = 1$ means "Hospital 1 contributes T1ce updates *within the T1ce-owning pool*." That is a different institutional consent decision from "Hospital 1 allows its T1ce-derived aggregate statistics to be consumed by *anyone, including non-T1ce-owners*." Real MoUs distinguish "share within the consortium" from "share aggregate statistics externally." The architecture should too.

### 3.5 Relationship to Existing Recv Notation

$$\text{Recv}_{\text{cross}}(i) = \{m \in \mathcal{M} \setminus O(i) : R_{\text{recv}}^{\text{cross}}(i, m) = 1\}$$

$$\text{Recv}_{\text{own}}(i) = \text{Recv}(i) \cap O(i) \quad \text{(governed by existing } R_{\text{recv}}(i,j,m) \text{ as before)}$$

$$\text{Recv}(i) = \text{Recv}_{\text{own}}(i) \cup \text{Recv}_{\text{cross}}(i)$$

---

## 4. New Prototype Variant — Cross-Exposed Prototype

### 4.1 Definition

For each modality $m$ and class $c$, the **cross-exposed prototype** is computed from the subset of modality-$m$ contributors who have opted into cross-boundary exposure:

$$\text{Proto}_m^{c,\text{cross}} = \sum_{\substack{j:\; R_{\text{send}}(j,m)=1 \\ \wedge\; R_{\text{expose}}^{\text{cross}}(j,m)=1}} \frac{n_j^{c,m}}{N_m^{c,\text{cross}}} \; \text{Proto}_{m,j}^{c}$$

where:
- $\text{Proto}_{m,j}^c$ is client $j$'s local prototype for modality $m$, class $c$ (computed in Phase 1 as per §5.2)
- $n_j^{c,m}$ is the number of class-$c$ spatial locations in client $j$'s modality-$m$ bottleneck
- $N_m^{c,\text{cross}} = \sum_{j:\, R_{\text{send}}(j,m)=1 \,\wedge\, R_{\text{expose}}^{\text{cross}}(j,m)=1} n_j^{c,m}$ is the normalization

### 4.2 Key Properties

- **Same formula as §6.3** — evidence-weighted aggregation — just with a tighter contributor filter.
- **No new aggregation mechanism.** The server already computes $\text{Proto}_m^c$ from all contributors. $\text{Proto}_m^{c,\text{cross}}$ is the same computation with the contributor set filtered by $R_{\text{expose}}^{\text{cross}}$.
- **Frozen after Phase 1.** Like all prototypes, these are computed at the end of Phase 1 and never updated.
- **Tiny payload.** Per modality per class: one $\mathbb{R}^{256}$ vector = 1 KB. For $\text{Recv}_{\text{cross}}(3) = \{\text{T1ce}\}$: 4 classes × 256 floats × 4 bytes = **4 KB total**.
- **Chain of custody.** The Provenance Ledger records exactly which contributors are in the $R_{\text{expose}}^{\text{cross}}$-filtered set, so the audit trail is complete.

### 4.3 When $R_{\text{expose}}^{\text{cross}}$ Is Unanimous vs. Partial

| Scenario | Consequence |
| :--- | :--- |
| All modality-$m$ contributors set $R_{\text{expose}}^{\text{cross}} = 1$ | $\text{Proto}_m^{c,\text{cross}} = \text{Proto}_m^c$ (identical to the standard prototype) |
| Some but not all contributors opt in | $\text{Proto}_m^{c,\text{cross}}$ is computed from the consenting subset only — may differ from $\text{Proto}_m^c$ |
| No contributors opt in | $\text{Proto}_m^{c,\text{cross}}$ does not exist → cross-absorb for modality $m$ is blocked regardless of $R_{\text{recv}}^{\text{cross}}$ |

---

## 5. The 7th Component: Cross-Boundary Absorption Layer

### 5.1 Architecture

The Cross-Boundary Absorption Layer consists of two private, never-aggregated sub-components:

| Sub-Component | Architecture | Parameters | Aggregated? |
| :--- | :--- | :--- | :--- |
| **CrossAbsorbHead** | Cross-attention or MLP (see §5.3) | Local only | **Never** |
| **Decoder_local** | U-Net upsampling path (same architecture as track decoders) | Local only | **Never** |

### 5.2 Full Phase 2 Pseudocode with Cross-Absorb

```python
# Hospital 3, Phase 2, per round
# S = Send(3) = {T1, FLAIR}

# ═══════════════════════════════════════════════════════════════
# PATH 1: FEDERATED (byte-identical to M1 — §8.2/§8.5)
# ═══════════════════════════════════════════════════════════════

# Pull federated track parameters (gated by R_send^track / R_recv^track)
pull FusionHead_S, Decoder_S, FusedProto_S^c

# Forward pass
for m in S:
    z_m = detach(Encoder_m*(x_m))               # frozen encoder, detached output
z_S = FusionHead_S({z_m : m in S})              # cross-attention over skip features
y_hat_fed = Decoder_S(z_S)                       # upsample to segmentation map

# Federated loss (unchanged from M1)
L_fed = L_task(y, y_hat_fed) + λ2 · L_fusion_align(z_S, {FusedProto_S^c})

# Backprop → updates FusionHead_S, Decoder_S only
# These updates ARE pushed to server for aggregation

# ═══════════════════════════════════════════════════════════════
# PATH 2: LOCAL CROSS-ABSORB (new in M2 — never aggregated)
# Only runs when: (a) Recv_cross(i) ≠ ∅, (b) Track S.LIFECYCLE == STABLE
# ═══════════════════════════════════════════════════════════════

if Recv_cross(i) != {} and Track_S.LIFECYCLE == STABLE:

    # SECOND DETACH WALL — severs gradient path to federated parameters
    z_S_frozen = detach(z_S)

    # Cross-absorb head uses consented non-owned prototypes
    cross_protos = {Proto_m^{c,cross} : m in Recv_cross(i), c in C_cls}
    z_enhanced = CrossAbsorbHead(z_S_frozen, cross_protos)

    # Local decoder produces enhanced segmentation
    y_hat_local = Decoder_local(z_enhanced)

    # Local loss — trains CrossAbsorbHead and Decoder_local ONLY
    L_local = L_task(y, y_hat_local)

    # Backprop → updates CrossAbsorbHead, Decoder_local only
    # These updates are NEVER pushed to server. NEVER aggregated.
    # Logged as an audit event in Provenance Ledger, not a track update.
```

### 5.3 CrossAbsorbHead Design Options

#### Option 1: Cross-Attention (Recommended Default)

The CrossAbsorbHead treats $z_{S,\text{frozen}}$ feature vectors as **queries** and the cross-exposed prototypes as **keys/values**:

$$\text{CrossAbsorbHead}(z_{S,\text{frozen}},\; \{\text{Proto}_m^{c,\text{cross}}\}) = \text{MultiHeadAttn}(Q = z_{S,\text{frozen}},\; K = V = P_{\text{cross}}) + z_{S,\text{frozen}}$$

where $P_{\text{cross}} \in \mathbb{R}^{|\text{Recv}_{\text{cross}}| \times C_{\text{cls}} \times 256}$ is the matrix of all cross-exposed prototypes, and the residual connection preserves the original federated features.

**Why cross-attention:** Consistent with $\text{FusionHead}_S$'s own architecture (§5.4), allowing direct comparison. Each spatial location $p$ in $z_S$ attends to the cross-exposed prototypes based on learned relevance — e.g., spatial locations near enhancing tumor boundaries attend more strongly to $\text{Proto}_{\text{T1ce}}^{\text{ET},\text{cross}}$.

**Parameter count:** For a single-head attention with $d = 256$: $W_Q, W_K, W_V, W_O \in \mathbb{R}^{256 \times 256}$ = 4 × 256² = **262K parameters**. Tiny relative to the full model.

#### Option 2: Prototype-Conditioned MLP (Simpler Ablation)

$$\text{CrossAbsorbHead}(z_{S,\text{frozen}},\; \{\text{Proto}_m^{c,\text{cross}}\}) = \text{MLP}([z_{S,\text{frozen}}(p) \;\|\; \text{Proto}_m^{c_{\text{nearest}}(p),\text{cross}}])$$

where $c_{\text{nearest}}(p)$ is the class whose standard prototype is closest to $z_S(p)$ (nearest-neighbor class assignment), and $\|$ denotes concatenation. This is simpler but less expressive — useful as an ablation to measure whether the attention mechanism adds value.

### 5.4 Gradient Flow Verification

```
PATH 1 (Federated):
L_fed   → ∂/∂θ(Decoder_S)       ✓ updated, uploaded
        → ∂/∂θ(FusionHead_S)    ✓ updated, uploaded
        → detach(Encoder_m*)     ✗ gradient severed at encoder boundary

PATH 2 (Local Cross-Absorb):
L_local → ∂/∂θ(Decoder_local)     ✓ updated, NEVER uploaded
        → ∂/∂θ(CrossAbsorbHead)   ✓ updated, NEVER uploaded
        → detach(z_S)              ✗ gradient severed — FusionHead_S UNTOUCHED

INVARIANT: No gradient from L_local ever reaches FusionHead_S or Decoder_S.
           No parameter from CrossAbsorbHead or Decoder_local ever enters
           any aggregation sum.
           ∴ The federated track is byte-identical to what it would be
             without cross-absorb.
```

### 5.5 Lifecycle Gating — The Moving-Target Fix

**Problem (identified by FedAS, CVPR 2024):** $\text{FusionHead}_S$ is retrained and re-aggregated every Phase 2 round. During warm-start rounds, $z_S$'s distribution shifts rapidly. If CrossAbsorbHead trains against this non-stationary target, it chases a moving distribution and struggles to converge.

**Fix:** Gate CrossAbsorbHead activation to Track $S$'s **STABLE** lifecycle state. The Phase Controller (§5.5) already tracks per-track lifecycle:

```python
# Updated Phase Controller per-track logic:
Per-track state: { LIFECYCLE in {WARMING, STABLE}, round t_local, seed_record }

Each round, per active track:
    if LIFECYCLE == WARMING:
        run L_fusion-align-only round (no task loss, no cross-absorb)
        if warm-start stopping criterion met: LIFECYCLE = STABLE
    if LIFECYCLE == STABLE:
        run normal Phase-2 round (L_task + lambda2 * L_fusion-align)
        if Recv_cross(i) != {}:                                          # NEW
            activate CrossAbsorbHead + Decoder_local training (L_local)  # NEW
```

**Why this works:** Once STABLE, the federated track's parameter drift is limited to normal federated learning drift (small per-round updates), which personalized-FL methods handle routinely. The CrossAbsorbHead is not exposed to the large distribution shifts that occur during warm-start.

### 5.6 Inference-Time Behavior

At inference, Hospital 3 deploys the **local** pathway (not the federated pathway):

```python
# Hospital 3, Inference:
for m in O(3):  # {T1, FLAIR}
    z_m = Encoder_m*(x_m)                    # no detach needed at inference
z_S = FusionHead_S({z_m : m in S})           # federated track forward pass
z_enhanced = CrossAbsorbHead(z_S, {Proto_T1ce^{c,cross}})
y_hat = Decoder_local(z_enhanced)            # final segmentation output

# Only T1 and FLAIR inputs are required. No T1ce scan exists or is needed.
# Proto_T1ce^{c,cross} was received once at Phase 1 end and stored locally.
```

---

## 6. Formal Guarantees

### 6.1 Lemma: Cross-Boundary Non-Propagation

**Statement.** For client $i$ with $\text{Recv}_{\text{cross}}(i) \neq \emptyset$, the shared state $\theta_{\text{FusionHead}_S}^t$, $\theta_{\text{Decoder}_S}^t$, $\text{FusedProto}_S^{c,t}$ satisfies the **§12.2 Send-Restricted Track Purity Lemma exactly as originally stated** — unchanged, for every round $t$.

**Proof.** Composes two existing guarantees:

1. **Forward-pass exclusion (§12.2 argument 1):** The federated forward pass (PATH 1) is byte-identical to M1. No $x_i^m$ for $m \notin S$ appears anywhere in $\mathcal{L}_{\text{fed}}$'s computation subgraph. $\mathcal{L}_{\text{local}}$'s computation subgraph is irrelevant because:

2. **Gradient isolation:** $\mathcal{L}_{\text{local}}$'s backward pass is walled off by `detach()` at $z_S$. `CrossAbsorbHead` and `Decoder_local` are never transmitted to the server, hence never enter any aggregation sum (§12.2 argument 2). A weighted average of federated-only terms — each independent of $\text{Proto}_m^{c,\text{cross}}$ — remains independent of $\text{Proto}_m^{c,\text{cross}}$.

3. **Induction (§12.2 argument 3):** The seed condition is unchanged (Lineage Rule applies to the federated track only, not the local head). The inductive step is unchanged. $\blacksquare$

### 6.2 Corollary: Pareto Non-Harm

> Client $i$'s *personal, deployed* model is additionally a function of $\text{Proto}_m^{c,\text{cross}}$ for $m \in \text{Recv}_{\text{cross}}(i)$ — an artifact with fully logged lineage $\{j : R_{\text{send}}(j,m)=1 \wedge R_{\text{expose}}^{\text{cross}}(j,m)=1\}$.
>
> **No other client's shared or personal model is affected.** The cross-absorb layer is structurally incapable of costing anyone else anything — it's a strict Pareto improvement gated by bilateral consent.

**Why this matters for the paper:** This is a clean, publishable result that directly supports R1 (Performance Under Compliance). Cross-absorb can only help the consenting client; every other client's guarantees remain byte-identical.

### 6.3 Lineage Rule Interaction

The **Lineage Rule (§10.1)** governs track seeding and prevents contaminated initialization. Under Approach D:

- $\text{Lineage}(\text{FusionHead}_S)$ is **unchanged** — the federated track parameters are shaped only by modalities in $S$.
- $\text{Lineage}(\text{CrossAbsorbHead})$ and $\text{Lineage}(\text{Decoder\_local})$ include modalities in $S \cup \text{Recv}_{\text{cross}}(i)$. But these components are **never seeded from, never used as Tier 1 sources, and never enter any cold-start procedure** — they are purely local and have no role in the Lineage Rule's scope.

**Result:** The Lineage Rule requires zero modification for Approach D. It remains exactly as specified in §10.

---

## 7. Provenance Ledger Extension

### 7.1 New Entry Type: Cross-Boundary Absorption Event

```yaml
Event: CROSS_BOUNDARY_ABSORPTION
Client: i = Hospital 3
Round activated: t_absorb (first round where Track S3.LIFECYCLE == STABLE)
Recv_cross(3): {T1ce}
Proto_T1ce^{c,cross} source:
    Contributing clients: {j : R_expose_cross(j, T1ce) = 1}
    Contributor list: [Hospital 1]       # H2 has R_expose_cross(2, T1ce) = 0
    Contributor sample counts: {H1: n_1^{c,T1ce}}
    Proto frozen at: Phase 1 end, round t_freeze
Components created:
    CrossAbsorbHead_3  -> fresh_init at round t_absorb
    Decoder_local_3    -> fresh_init at round t_absorb
Aggregation status: NEVER_AGGREGATED
```

### 7.2 Audit Trail Properties

- **Who consented (receive-side):** $R_{\text{recv}}^{\text{cross}}(3, \text{T1ce}) = 1$ — logged, versioned
- **Who consented (send-side):** $R_{\text{expose}}^{\text{cross}}(1, \text{T1ce}) = 1$ — logged, versioned
- **What was transmitted:** $\text{Proto}_{\text{T1ce}}^{c,\text{cross}}$ — frozen artifact, 4 KB, no patient data
- **What was affected:** `CrossAbsorbHead_3`, `Decoder_local_3` — purely local, never aggregated
- **What was NOT affected:** $\text{FusionHead}_{S_3}$, $\text{Decoder}_{S_3}$, $\text{FusedProto}_{S_3}^c$ — untouched

This satisfies **Requirement R2 (Auditability)** — a compliance officer can inspect the ledger and verify exactly which cross-boundary absorptions occurred, who consented on both sides, and confirm that no federated component was affected.

---

## 8. Updated Experimental Federation

### 8.1 Updated Client Matrix

| Client | Owned $O(i)$ | Send | $\text{Recv}_{\text{own}}$ | $\text{Recv}_{\text{cross}}$ | $R_{\text{expose}}^{\text{cross}}$ | Joins At |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hospital 1** | $\{T1, T1ce, T2, FL\}$ | $\{T1, T1ce, T2, FL\}$ | $\{T1, T1ce, T2, FL\}$ | $\emptyset$ | $R_{\text{expose}}^{\text{cross}}(1, \text{T1ce}) = 1$ | Phase 1 |
| **Hospital 2** | $\{T1, T1ce, T2\}$ | $\{T1, T2\}$ | $\{T1, T1ce, T2\}$ | $\emptyset$ | $R_{\text{expose}}^{\text{cross}}(2, \text{T1ce}) = 0$ | Phase 1 |
| **Hospital 3** | $\{T1, FL\}$ | $\{T1, FL\}$ | $\{T1, FL\}$ | $\{\text{T1ce}\}$ | N/A (doesn't own T1ce) | Phase 1 |
| **Hospital 4** | $\{T1, T1ce, T2\}$ | $\{T1, T1ce, T2\}$ | $\{T1, T1ce, T2\}$ | $\emptyset$ | Default $0$ | Phase 2 |

**Notable consent dynamics:**
- **Hospital 1** allows T1ce cross-exposure → its $\text{Proto}_{\text{T1ce}}^c$ contribution enters $\text{Proto}_{\text{T1ce}}^{c,\text{cross}}$
- **Hospital 2** does NOT allow T1ce cross-exposure (despite owning T1ce) → its contribution is excluded from the cross-exposed prototype
- **Hospital 3** consents to receive T1ce knowledge it doesn't own → activates cross-absorb
- This demonstrates exactly how fine-grained the consent model is: H2 contributes T1ce to the in-cohort pool but withholds it from cross-boundary consumption

### 8.2 Proto Transmission at Phase 1 End

At the Phase 1 → Phase 2 freeze transition, the server:

1. Broadcasts frozen $\text{Encoder}_m^*$ and $\text{Proto}_m^c$ to all clients as before (unchanged)
2. **Additionally** computes and transmits $\text{Proto}_{\text{T1ce}}^{c,\text{cross}}$ to Hospital 3 — filtered to include only Hospital 1's contribution (since $R_{\text{expose}}^{\text{cross}}(2, \text{T1ce}) = 0$)
3. Logs the transmission event to Provenance Ledger

**Payload:** 4 classes × 256 floats × 4 bytes = **4 KB**. Negligible relative to encoder payloads (~10 MB each).

---

## 9. New Ablation & Probe Extensions

### 9.1 Ablation A9: Cross-Boundary Absorption Impact

| # | Ablation | Question | Setup |
| :--- | :--- | :--- | :--- |
| **A9** | Cross-Absorb On vs Off | Does cross-boundary T1ce prototype absorption improve Hospital 3's segmentation? | Hospital 3 Dice disaggregated by WT/TC/ET, with $\text{Recv}_{\text{cross}}(3) = \{\text{T1ce}\}$ vs $\text{Recv}_{\text{cross}}(3) = \emptyset$ |

**Expected outcome and interpretation:**
- **WT (Whole Tumor):** Small or no improvement — T1/FLAIR already capture WT boundaries well
- **TC (Tumor Core):** Possible small improvement — T1ce helps delineate NCR/NET vs ED
- **ET (Enhancing Tumor):** **This is where the signal should appear** — ET is clinically defined by gadolinium contrast enhancement visible only in T1ce. If cross-absorb helps anywhere, it's here.
- **If all three show no improvement:** The prototype distance is too small (see §10) → escalate to Approach B'

### 9.2 Extended Purity Probe (A3 Extension)

The existing Purity Probe (§14.4) checks whether forbidden modality signal enters a pathway. With cross-absorb, we need TWO probes on Hospital 3:

| Probe Target | Expected Result | Interpretation |
| :--- | :--- | :--- |
| Hospital 3's **federated** track output ($z_{S_3}$ from $\text{FusionHead}_{S_3}$) | **At null baseline** | Confirms §12.2 Purity Lemma still holds — the federated track carries zero T1ce signal |
| Hospital 3's **personalized** output ($z_{\text{enhanced}}$ from CrossAbsorbHead) | **Elevated above null** | Confirms cross-absorb IS transferring T1ce signal — this is the *desired* outcome, not a leak |

**If the federated probe is elevated:** Something is architecturally wrong — the `detach()` wall has a bug. This should never happen under honest execution.

**If the personalized probe is NOT elevated:** The prototype doesn't carry enough T1ce-specific signal to matter → escalate to Approach B'.

### 9.3 Additional Ablation Suggestion: H2 Cross-Exposure Toggle

| # | Ablation | Question | Setup |
| :--- | :--- | :--- | :--- |
| **A10** | H2 Cross-Exposure Toggle | Does including H2's T1ce contribution in $\text{Proto}_{\text{T1ce}}^{c,\text{cross}}$ measurably change Hospital 3's Dice? | Toggle $R_{\text{expose}}^{\text{cross}}(2, \text{T1ce})$ from 0 to 1, re-run A9 |

**Why this matters:** Demonstrates that the consent model is not governance theater — different exposure policies produce measurably different outcomes.

---

## 10. The Primary Empirical Risk

### 10.1 The Self-Defeating Dynamic

Phase 1's InfoNCE contrastive loss aligns all encoders to project the same anatomy into the same 256-D region (§5.2, "Shared Language" property):

$$\text{Proto}_{\text{T1}}^c \approx \text{Proto}_{\text{T1ce}}^c \approx \text{Proto}_{\text{T2}}^c \approx \text{Proto}_{\text{FLAIR}}^c$$

**The more successful Phase 1 is at its own objective, the smaller the residual distance $\|\text{Proto}_{\text{T1ce}}^c - \text{Proto}_{\text{T1}}^c\|$ becomes, and the less information the cross-absorb mechanism has to work with.**

This isn't a bug — it's a fundamental tension built into the architecture. Any design that transfers knowledge through a shared embedding space hits this same ceiling.

### 10.2 Evidence from Literature

A November 2025 brain-MRI foundation model paper measured cross-modality embedding cosine distance before and after modality-invariant contrastive training:

| Condition | Cross-Modality Cosine Distance |
| :--- | :--- |
| Without alignment pressure | ~0.50 – 0.60 |
| With effective modality-invariant training | **~0.007** |

Phase 1's entire purpose IS modality-invariant contrastive alignment. If it works as designed, residual inter-modal prototype distance could be near-zero.

### 10.3 What $\text{Proto}_m^c$ Actually Is — and What It Can't Transfer

$\text{Proto}_m^c$ is a **single mean-pooled centroid** in $\mathbb{R}^{256}$ — one point per class, all spatial and boundary structure averaged away by construction:

$$\text{Proto}_m^c = \frac{1}{|P_m^c|} \sum_{p \in P_m^c} z_m(p)$$

**What it CAN transfer:** Classification-consistency signal — "the cluster center for class $c$ in T1ce encoder space sits *here* in 256-D."

**What it CANNOT transfer:** Spatial boundary information, fine-grained tumor morphology, or any structure finer than a class-level centroid.

### 10.4 The Pre-Check Protocol

**Immediately after Phase 1 completes, before any Phase 2 experiments:**

```python
# Pre-check: compute per-class inter-modal prototype distance
for c in C_cls:                      # {BG, NCR/NET, ED, ET}
    for (m1, m2) in [("T1", "T1ce"), ("T1", "T2"),
                     ("T1", "FLAIR"), ("T1ce", "FLAIR")]:
        d_cosine = 1 - cosine_similarity(Proto[m1][c], Proto[m2][c])
        d_L2     = np.linalg.norm(Proto[m1][c] - Proto[m2][c])
        print(f"Class {c}, {m1}-{m2}: cos_dist={d_cosine:.6f}, L2={d_L2:.4f}")
```

**Decision gate:**
- If $\|\text{Proto}_{\text{T1ce}}^c - \text{Proto}_{\text{T1}}^c\|$ is meaningfully non-zero **especially for the ET class**: proceed with Approach D. The prototype carries transferable signal.
- If near-zero across all classes: Approach D's ceiling is low. Report this honestly as a finding. Escalate to Approach B' (self-distillation adapter). The governance framework (matrices, ledger, consent) still applies — only the CrossAbsorbHead's interior mechanism changes.

### 10.5 Per-Class Hypothesis

| Class | Expected Proto Distance | Reasoning |
| :--- | :--- | :--- |
| BG (Background) | Very small | Background looks similar across all MRI sequences |
| NCR/NET (Necrosis) | Small–medium | Some contrast difference, but not T1ce-specific |
| ED (Edema) | Small–medium | FLAIR is the dominant modality for edema, not T1ce |
| **ET (Enhancing Tumor)** | **Largest** | ET is *defined* by gadolinium contrast enhancement — visible as bright rim in T1ce, absent in T1. Most likely to retain residual inter-modal distance. |

### 10.6 Why This Risk Isn't Architectural

This is a question about **information physics** — what residual modality-specific signal survives contrastive alignment in 256-D — and **MRI contrast enhancement imaging** — whether T1ce's distinctive gadolinium enhancement pattern is distinguishable from T1 at the class-centroid level.

**Any design** that tries to give a client benefit from a non-owned modality through a shared embedding space hits this same ceiling — MFCPL hits it, pFedDKS hits it, a from-scratch redesign would hit it. Rebuilding the architecture doesn't buy an exemption from this question; it just costs everything M1 already solved for zero progress on the one question that's actually still open.

---

## 11. Escalation Path: Approach B' (Self-Distillation Adapter)

### 11.1 Motivation

If the pre-check (§10.4) shows that prototype distance is near-zero, the prototype-based CrossAbsorbHead provides minimal benefit. The escalation is a **masked self-distillation adapter** that transfers richer knowledge — spatial, boundary-level, not just class-centroid-level.

### 11.2 Mechanism

Hospital 1 already runs masked forward passes via §11 (Multi-Track Contribution). Approach B' adds:

1. **Teacher pass (local to Hospital 1):**
   $x_{\{T1, T1ce, T2, \text{FLAIR}\}} \to \text{FusionHead}_{S_1} \to \text{Decoder}_{S_1} \to y_{\text{teacher}}$

2. **Student pass (local to Hospital 1):**
   $x_{\{T1, \text{FLAIR}\}} \to \text{FusionHead}_{S_3} \to \text{Decoder}_{S_3} \to y_{\text{student}}$

3. **Self-distillation:** Train a small adapter on the gap between $y_{\text{teacher}}$ and $y_{\text{student}}$ — capturing what 4-modality knowledge adds beyond 2-modality knowledge, distilled into a few parameters.

4. **Transmit adapter:** The adapter (parameters, not logits, not patient data) is shared with Hospital 3 via the same consent-gated, Provenance-Ledger-logged protocol.

### 11.3 Why This Is Deferred, Not Built Now

- Adds a new component type (adapter) not currently in the architecture
- Requires a teacher-student training protocol
- The adapter carries $\text{Lineage} = \{T1, T1ce, T2, \text{FLAIR}\}$ — consent gating applies
- **Approach D is simpler and may be sufficient** — measure first, escalate if needed
- The governance framework (matrices, ledger, consent) is identical either way — only the CrossAbsorbHead's interior changes

### 11.4 Status

> **Labeled as deferred future option. Not rejected, not foreclosed.** If prototype distance proves insufficient, B' becomes the primary mechanism, using the same governance shell Approach D established.

---

## 12. Deferred: Approach C (Federated Prototype Regularization)

### 12.1 What It Would Be

Add $\lambda_3 \cdot \mathcal{L}_{\text{cross-absorb}}$ directly to the **federated** track's Phase 2 loss, pulling the track's fused features toward non-owned prototypes. This shapes the aggregated $\text{FusionHead}_S$ parameters with cross-boundary signal — benefiting all track participants.

### 12.2 Why It's Deferred

#### Problem 1: Dynamic Membership

If Track $S_3$ bakes T1ce signal into its federated weights via Approach C:
- A new client joins Track $S_3$ later **without** T1ce cross-absorb consent
- That client receives T1ce-contaminated weights — consent violation
- "Fall back to Approach D" doesn't undo what's already baked in

**Unanimity is a point-in-time property; the architecture needs it to be an invariant.**

#### Problem 2: Lineage Rule Contamination

If federated C is used, $\text{Lineage}(\text{FusionHead}_{S_3}) = \{T1, \text{FLAIR}, \text{T1ce}\}$. A future Tier 1 cold-start from $S_3$ to a $\{T1, \text{FLAIR}\}$-only track would fail the Lineage check (correctly, if the Ledger is updated), or silently inherit T1ce contamination (if not).

#### Problem 3: Primary Empirical Risk

Even if the membership and Lineage problems are solved, Approach C hits the same prototype-distance ceiling as Approach D — and with more architectural complexity.

### 12.3 Resolution (If Ever Pursued)

**Key tracks on $(\text{Send}(i),\; \text{Recv}_{\text{cross}}(i))$ jointly**, not $\text{Send}(i)$ alone:
- Client with $\text{Send} = \{T1, \text{FLAIR}\}$, $\text{Recv}_{\text{cross}} = \{\text{T1ce}\}$ → Track $S_{3,+\text{T1ce}}$
- Client with $\text{Send} = \{T1, \text{FLAIR}\}$, $\text{Recv}_{\text{cross}} = \emptyset$ → Track $S_3$ (pure)

This reuses §9's re-keying pattern. No track ever contains members with mismatched cross-absorb consent. But creates track proliferation — deferred until needed.

---

## 13. Literature Positioning

### 13.1 What's Already Published

| Paper | Year | Mechanism | Relationship to CAMFS |
| :--- | :--- | :--- | :--- |
| **FedPer / FedRep / FedBABU** | 2020–2022 | Shared body + private head personalization | Approach D follows this paradigm exactly |
| **pFedDKS** | 2026 | Detached prototypes → private local head | Almost a description of our specific mechanism |
| **MFCPL** | April 2025 | Cross-modal prototype regularization in FL for missing modalities | The underlying technique — should be cited directly |
| **FedAS** | CVPR 2024 | Moving-target problem with personalized heads over evolving shared representations | Validates the STABLE-gating fix |
| **FedCMD** | ACM TIST 2024 | Cross-modal KD with prototype alignment | Validates prototype-based cross-modal transfer in FL |
| **Pro MoE-FL** | 2025 | Prototype-conditioned expert routing for missing features | Validates prototype as proxy for absent modality |

### 13.2 What's Still Novel

> **The technique (prototype-based cross-modal regularization) is not the novel part. The consent governance is.**
>
> No existing work combines missing-modality knowledge transfer with:
> - Exogenous, non-learned, per-modality consent matrices
> - Separate send-side and receive-side consent gates
> - Bilateral opt-in (both $R_{\text{recv}}^{\text{cross}}$ AND $R_{\text{expose}}^{\text{cross}}$ must be set)
> - Provenance-ledger auditability for every cross-boundary absorption event
> - Formal purity guarantee for the federated track alongside the personalized cross-absorb pathway

**This is Gap 6 as stated:** "No existing work provides a consent-governed, auditable mechanism for a client to selectively absorb knowledge derived from modalities it does not physically possess."

MFCPL provides the mechanism. CAMFS provides the governance. Both are needed; neither alone is sufficient.

### 13.3 How to Cite

In the Related Work section:
- Cite MFCPL, FedCMD, Pro MoE-FL as prototype-based cross-modal transfer methods
- Cite FedPer/FedRep/FedBABU as the personalization paradigm
- Cite FedAS for the moving-target insight
- Position CAMFS's contribution as: **the first to wrap prototype-based cross-modal transfer in a formal, auditable, bilateral consent framework** — the "what to transfer" is known, the "who gets to decide" is new

---

## 14. What Changes vs What Doesn't

### 14.1 Unchanged from M1 (Zero Modification)

| Component | Why It's Unchanged |
| :--- | :--- |
| **Encoder Bank** (§5.1) | Frozen after Phase 1. Cross-absorb operates in Phase 2 only, downstream of encoders. |
| **Phase 1 Training** (§6) | Identical — unimodal encoders and prototypes train the same way. |
| **Freeze Procedure** (§7) | Identical — encoders and prototypes freeze at the same transition point. |
| **Phase 2 Federated Loss** (§8.3) | Identical — $\mathcal{L}_{\text{fed}}$ is byte-identical to M1. |
| **Track Hard Isolation** (§5.4) | Unchanged — no parameter or gradient crosses between federated tracks. |
| **Send-Gated Routing** (§9) | Unchanged — send-side keying logic is identical. |
| **Cold-Start / Lineage Rule** (§10) | Unchanged — local head never enters Lineage chain. |
| **Multi-Track Contribution** (§11) | Unchanged. |
| **Purity Lemma** (§12.2) | **Holds exactly as stated.** New lemma is added alongside, not replacing it. |
| **Requirements R1–R3** | Composed with, not modified. |

### 14.2 New in M2

| Addition | Type | Description |
| :--- | :--- | :--- |
| $R_{\text{recv}}^{\text{cross}}(i, m)$ | Policy matrix | Receive-side cross-boundary consent |
| $R_{\text{expose}}^{\text{cross}}(j, m)$ | Policy matrix | Send-side cross-boundary exposure consent |
| $\text{Proto}_m^{c,\text{cross}}$ | Prototype variant | Same aggregation, filtered by $R_{\text{expose}}^{\text{cross}}$ |
| CrossAbsorbHead | Component (7th) | Private, `detach()`-separated, never aggregated |
| Decoder_local | Component (part of 7th) | Private decoder for cross-absorb output |
| Cross-Boundary Non-Propagation Lemma | Formal guarantee | Proves federated track is untouched |
| Pareto Corollary | Formal guarantee | R4 can only help, never harm |
| Provenance Ledger cross-absorb event | Audit entry | Records all cross-boundary absorption events |
| Ablation A9 | Experiment | Cross-absorb on vs off |
| Ablation A10 | Experiment | H2 cross-exposure toggle |
| Extended Purity Probe | Experiment | Dual probe: federated (null) vs personalized (elevated) |
| Requirement R4 | Formal requirement | Cross-boundary knowledge absorption under consent |
| Gap 6 | Literature gap | Consent-governed cross-boundary absorption |

---

## 15. Pre-Implementation Checklist

| # | Action | Blocks | Status |
| :--- | :--- | :--- | :--- |
| 1 | **Run Phase 1** and measure $\|\text{Proto}_{\text{T1ce}}^c - \text{Proto}_{\text{T1}}^c\|$ per class | Determines prototype-based ceiling | 🔴 Before Phase 2 R4 code |
| 2 | **Implement** $R_{\text{recv}}^{\text{cross}}$ and $R_{\text{expose}}^{\text{cross}}$ in federation config | All cross-absorb logic | 🔴 Before implementation |
| 3 | **Compute** $\text{Proto}_m^{c,\text{cross}}$ at Phase 1 end with $R_{\text{expose}}^{\text{cross}}$ filter | CrossAbsorbHead input | 🔴 Before implementation |
| 4 | **Gate** CrossAbsorbHead activation to Track STABLE lifecycle | Moving-target correctness | 🔴 Before implementation |
| 5 | **Implement** CrossAbsorbHead + Decoder_local with double `detach()` | Core component | 🔴 Implementation |
| 6 | **Decide** Decoder_local init: warm from Decoder_S vs fresh | Design choice (see §16 Q4) | 🟡 During implementation |
| 7 | **Run** Ablation A9 (H3 Dice WT/TC/ET, cross-absorb on/off) | Empirical value | 🟡 After implementation |
| 8 | **Run** dual purity probe (federated=null, personalized=elevated) | Validates guarantees | 🟡 After implementation |
| 9 | **Run** Ablation A10 (H2 exposure toggle) | Consent granularity demo | 🟡 After implementation |
| 10 | **If proto distance near-zero:** design Approach B' adapter | Escalation path | 🟠 Conditional |

---

## 16. Open Questions

| # | Question | Impact | Proposed Resolution |
| :--- | :--- | :--- | :--- |
| 1 | **Does per-class proto distance survive Phase 1?** | Core viability of prototype-based mechanism | Measure immediately after Phase 1. §10.4 protocol. The single most important empirical gate. |
| 2 | **Cross-attention vs MLP for CrossAbsorbHead?** | Component design | Default cross-attention (consistent with FusionHead). MLP as ablation variant. |
| 3 | **Learning rate for local head vs federated track?** | Training stability | The two losses have separate backward passes. The local head can use an independent learning rate. Start with the same LR as the federated track; tune if needed. |
| 4 | **Should Decoder_local be initialized from Decoder_S?** | Cold-start of local components | Decoder_S is available, Lineage = $S$. Initializing from it gives a warm start. Fresh init is conservative. **Suggest: warm-start from Decoder_S, ablate against fresh.** |
| 5 | **If only one contributor exposes, is the cross-exposed proto representative?** | Statistical robustness | With only Hospital 1, $\text{Proto}_{\text{T1ce}}^{c,\text{cross}}$ reflects H1's patient population only. Note limitation; not fixable without more contributors opting in. |
| 6 | **How does cross-absorb interact with BraTS ET annotation bias?** | Known issue (§16 of main doc) | ET is annotated from T1ce. If H3 absorbs T1ce proto and improves on ET, the improvement is partially due to label bias. Disaggregate WT/TC/ET and note in discussion. |
| 7 | **Should cross-absorb also apply at skip-connection resolutions?** | Potential enhancement | Currently CrossAbsorbHead operates at bottleneck (256×15×15). Skip-connection features at higher resolutions carry more spatial detail. Could extend cross-absorb to operate on skip features too. **Deferred — start with bottleneck-only.** |

---

*Document version: v1.0 — July 2026*  
*Status: Working draft. Ready for review and iteration.*
