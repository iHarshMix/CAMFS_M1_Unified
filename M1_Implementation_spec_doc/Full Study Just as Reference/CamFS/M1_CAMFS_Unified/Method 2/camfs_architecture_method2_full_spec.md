# CAMFS Full Architecture Specification — Rebuilt Around Method 2 (Two-Phase Training)

**Scope:** This document supersedes and expands `consent_aware_camfs_brats_paper.md` §5 (Proposed Architecture) specifically, rebuilding it around **Method 2 — Two-Phase / Sequential Training**, as recommended in `camfs_training_methods_report.md`. Everything outside §5 of the original paper draft (problem statement, novelty positioning, experimental plan, ablations) is unchanged and not repeated here.

**Traceability convention:** where a design choice below directly resolves a numbered flaw from `camfs_deep_audit_method1.md`, it is marked **[Resolves Flaw N]**. Where this document introduces something genuinely new — not stated anywhere in the four source files — it is marked **[Proposed — new in this document]** so it's clear what's an established decision versus a fresh recommendation to bring to Professor Jerry.

---

## 1. Research Objective (Grounding Restatement)

CAMFS targets **consent-constrained** federated segmentation: a federation of hospitals, each structurally owning a fixed subset of four MRI sequences (T1, T1ce, T2, FLAIR), governed by an explicit, exogenous, non-learnable policy matrix (`R_send`, `R_recv`) rather than by what would improve accuracy. The architectural requirement (paper §5.1) is that two things hold *simultaneously*:

1. A client only ever computes forward/backward passes through parameters trained on modalities it is permitted to receive — enforced **structurally**, not by a penalty term.
2. The permission structure is a first-class, inspectable artifact.

The deep audit (Flaw 8) showed the original joint-training design satisfies (2) but not (1): task-loss gradients from a full-modality client's fusion head still shape the *shared* encoders, which are then broadcast to restricted clients. Method 2 exists specifically to make (1) true as well. Everything below is the architecture that results.

---

## 2. Design Principles Under Method 2

Two principles are added to the paper's original design principle (§5.1):

- **Principle 3 — Temporal isolation, not just structural isolation.** Approach 3 (subset-fusion tracks) already gives spatial/parameter isolation *between* tracks. Method 2 adds isolation *in time*: the encoder bank and the fusion tracks are never in the same optimization step, so there is no round in which a task-loss gradient can reach a shared encoder at all.
- **Principle 4 — The freeze is a provable property of the computation graph, not an optimizer setting alone.** Setting `requires_grad=False` prevents an optimizer step from updating a frozen parameter, but gradients can still be *computed* through it for downstream layers. This architecture additionally `detach()`s encoder outputs at the point they enter a fusion head in Phase 2 (§8.2), so the "no gradient path" claim is visible directly in the graph, not just enforced by training-loop discipline.

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| $C = \{1,\dots,N\}$ | Federation clients (hospitals) |
| $\mathcal{M} = \{\text{T1, T1ce, T2, FLAIR}\}$ | Full modality universe |
| $O(i) \subseteq \mathcal{M}$ | Client $i$'s fixed owned subset |
| $S$ | A modality subset identifying a fusion track |
| $C_{\text{cls}} = 4$ | Segmentation classes: background, NCR/NET (label 1), ED (label 2), ET (label 4) |
| $R_{\text{send}}(i,m)$, $R_{\text{recv}}(i,j,m)$ | Modality-level policy gates (paper §4) |
| $R_{\text{send}}^{\text{track}}(i,S)$, $R_{\text{recv}}^{\text{track}}(i,j,S)$ | **[Proposed — new]** track-level policy gates, §8.4 |
| $R_{\text{contribute}}(i,S)$ | **[Proposed — new]** opt-in for a superset-owning client to help train a track it doesn't natively belong to, §14 |
| $\text{Encoder}_m$, $\text{Encoder}_m^{*}$ | Modality encoder; $^{*}$ denotes the frozen, post–Phase-1 snapshot |
| $\text{Proto}_m^c$ | Global unimodal class-$c$ prototype for modality $m$ |
| $\text{FusionHead}_S$, $\text{Decoder}_S$, $\text{FusedProto}_S^c$ | Track-$S$ fusion, decoding, and fused prototype |
| $z_m(p)$ | Encoder-$m$ bottleneck feature at spatial location $p$ |
| $P_m$, $P_m^c$ | Sampled bottleneck locations for modality $m$; subset with ground-truth class $c$ |
| $\tau$ | Contrastive temperature |
| $\text{sim}(\cdot,\cdot)$ | Cosine similarity |
| $t$, $T_1$ | Current round; number of Phase 1 rounds |
| $\text{detach}(\cdot)$ | Stop-gradient operator — value is kept, gradient path is not |

---

## 4. Component Inventory

### 4.1 Unimodal Encoder Bank

Four independent encoders, one per modality — $\text{Encoder}_{\text{T1}}$, $\text{Encoder}_{\text{T1ce}}$, $\text{Encoder}_{\text{T2}}$, $\text{Encoder}_{\text{FLAIR}}$ — each a 2D U-Net-style down-sampling path (paper §5.3): 4 conv-conv-downsample stages, base channel width 32 doubling per stage (32→64→128→256), producing a bottleneck embedding plus skip-connection feature maps at each of the 4 resolutions. **Global scope**: aggregated across every client that owns that modality and consents, independent of what else that client owns.

**Lifecycle under Method 2:** trainable in Phase 1; frozen and static for the remainder of training from Phase 2 onward (§7).

### 4.2 Unimodal Prototype Bank

$\text{Proto}_m^c$ for each of the 4 modalities × 4 classes = 16 vectors, each in $\mathbb{R}^{256}$ (bottleneck channel width). Computed as the mean bottleneck embedding at ground-truth class-$c$ locations (§6.3), aggregated across the same consenting group as $\text{Encoder}_m$.

**Lifecycle under Method 2:** updated in Phase 1 only; frozen alongside the encoders at the phase transition, since they're only used as Phase 1's alignment target and have no further role once fusion training begins.

### 4.3 Policy Matrices

| Matrix | Governs | Existing / Proposed |
|---|---|---|
| $R_{\text{send}}(i,m)$ | Client $i$ may contribute modality-$m$ encoder updates | Existing (paper §4) |
| $R_{\text{recv}}(i,j,m)$ | Client $i$ may receive modality-$m$ signal originating from $j$ | Existing (paper §4) |
| $R_{\text{send}}^{\text{track}}(i,S)$ | Client $i$ may contribute to track $S$'s fusion/decoder aggregate | **[Proposed — new, resolves Flaw 6]** |
| $R_{\text{recv}}^{\text{track}}(i,j,S)$ | Client $i$ may receive track-$S$ signal from same-subset client $j$ | **[Proposed — new, resolves Flaw 6]** |
| $R_{\text{contribute}}(i,S)$ | Superset-owning client $i$ opts in to help train a non-native track $S$ | **[Proposed — new, §14]** |

All matrices are exogenous, versioned, and inspectable — none are learned or loss-conditioned, consistent with the paper's core requirement (§4, "Hard constraint" row). The two track-level additions default to 1 whenever $O(i)=O(j)=S$, exactly reproducing the paper's original assumption unless governance explicitly restricts further — so this is a strict extension, not a behavior change, for anyone who doesn't need it.

### 4.4 Subset-Fusion Tracks

One track per distinct owned subset present in the federation. For the BraTS instantiation: $S_A=\{\text{T2,FLAIR}\}$, $S_B=\{\text{T1,T1ce,T2,FLAIR}\}$, $S_C=\{\text{T1,FLAIR}\}$, plus $S_D=\{\text{T1ce,T2}\}$ (ablation-only, Hospital D). Each track owns:

- **FusionHead$_S$** — cross-attention (default) or concatenation + 1×1 conv (cheaper ablation variant), combining *exactly* the skip-connection features of the encoders in $S$, applied at each of the 4 resolutions plus the bottleneck.
- **Decoder$_S$** — standard U-Net up-sampling path from fused features to a $C_{\text{cls}}=4$-class segmentation map.
- **FusedProto$_S^c$** — mean fused-bottleneck embedding per class, local to track $S$.

**Hard isolation** (Approach 3, unchanged): no parameter or gradient ever crosses between tracks. **Lifecycle under Method 2:** untrained and not instantiated in the computation graph during Phase 1; trainable from Phase 2 onward.

### 4.5 Phase Controller **[Proposed — new]**

A lightweight server-side state machine, not present in the original paper draft, needed once training has two distinct regimes:

```
State: { CURRENT_PHASE ∈ {1, 2}, round t, proto_history[m][c] }

On each round-end:
  if CURRENT_PHASE == 1:
      check stopping criterion (§6.5)
      if met: execute freeze procedure (§7); CURRENT_PHASE ← 2
  broadcast round instructions to clients:
      - which components to pull (Encoder/Proto only, vs. FusionHead/Decoder/FusedProto)
      - which loss terms to compute (per §6.3 or §8.3)
```

This is the only genuinely new *server-side component*; everything else is a re-scoped version of an existing one.

---

## 5. System-Level Component Diagram (Phase-Agnostic)

```
                                    SERVER
   ┌─────────────────────────────────────────────────────────────────┐
   │  PHASE CONTROLLER  (tracks current phase, triggers freeze)       │
   ├─────────────────────────────────────────────────────────────────┤
   │  UNIMODAL BANK                        │  SUBSET-FUSION TRACKS    │
   │  Encoder_T1   Encoder_T1ce            │  Track S_A {T2,FLAIR}    │
   │  Encoder_T2   Encoder_FLAIR           │  Track S_B {T1,T1ce,     │
   │  Proto_T1^c   Proto_T1ce^c            │            T2,FLAIR}     │
   │  Proto_T2^c   Proto_FLAIR^c           │  Track S_C {T1,FLAIR}    │
   │  [trainable Phase 1 → frozen Phase 2] │  Track S_D {T1ce,T2}     │
   │                                        │  (ablation only)         │
   │                                        │  [inert Phase 1 →        │
   │                                        │   trainable Phase 2]     │
   ├─────────────────────────────────────────────────────────────────┤
   │  POLICY LAYER:  R_send, R_recv (modality) | R_send/recv^track    │
   │                 (track) | R_contribute (multi-track, §14)         │
   └─────────────────────────────────────────────────────────────────┘
                    │                              │
        Phase 1 only: Encoder_m,        Phase 2 only: FusionHead_S,
        Proto_m^c broadcast,            Decoder_S, FusedProto_S^c
        gated by R_send/R_recv          broadcast, gated by track
                    │                    policy — plus a static,
                    │                    frozen Encoder_m snapshot
                    ▼                              ▼
       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
       │ Hospital A   │  │ Hospital B   │  │ Hospital C   │  │ Hospital D   │
       │ {T2,FLAIR}   │  │ {T1,T1ce,    │  │ {T1,FLAIR}   │  │ {T1ce,T2}    │
       │              │  │  T2,FLAIR}   │  │              │  │ (ablation)   │
       └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

---

## 6. Phase 1 — Unimodal Stabilization

### 6.1 Objective & Active/Frozen State

**Goal:** project each modality into a stable, class-discriminative latent space, using *only* same-modality signal — no fusion module exists in the graph, so there is nothing for cross-modal conditioning to travel through.

| Component | State |
|---|---|
| $\text{Encoder}_m$ (all 4) | **Trainable** |
| $\text{Proto}_m^c$ (all 16) | **Trainable** (running local means, aggregated) |
| $\text{FusionHead}_S$, $\text{Decoder}_S$, $\text{FusedProto}_S^c$ (all tracks) | **Not instantiated** — no forward pass, no parameters exist yet |

### 6.2 Data & Gradient Flow (per client)

```
   x_m (client i's local scans, m ∈ O(i))
        │
        ▼
   Encoder_m  ──────────────► z_m  (bottleneck + skip features)
        ▲                       │
        │                       ▼
        │              L_unimodal-align(z_m, {Proto_m^c}_c)   ◄── ground-truth
        │                       │                                  mask (class
        └───────────────────────┘                                  labels only,
              gradient flows only here — no Decoder_S,               resolution-
              no FusionHead_S, no task loss exists in this graph      matched)
```

No modality's encoder gradient can ever be a function of another modality's features in Phase 1, because no operation in the graph combines them. This is the structural fix for Flaw 8's root cause.

### 6.3 Loss Function **[Resolves Flaws 1, 2, 3]**

For client $i$, modality $m \in O(i)$: let $\tilde{y}$ be the ground-truth mask nearest-neighbor-downsampled to bottleneck resolution, $P_m$ a sampled set of bottleneck locations, and $c(p)=\tilde{y}(p)$ the ground-truth class at location $p$.

$$\mathcal{L}_{\text{unimodal-align}} = -\frac{1}{|P_m|}\sum_{p \in P_m} \log \frac{\exp\big(\text{sim}(z_m(p),\, \text{Proto}_m^{c(p)})/\tau\big)}{\sum_{c'=1}^{C_{\text{cls}}} \exp\big(\text{sim}(z_m(p),\, \text{Proto}_m^{c'})/\tau\big)}$$

$$\mathcal{L}_i^{\text{Phase 1}} = \lambda_1 \sum_{m \in O(i)} \mathcal{L}_{\text{unimodal-align}}(z_m)$$

This is an InfoNCE-style matched-class contrastive loss — each location is pulled toward its *own* ground-truth-class prototype and pushed from the other $C_{\text{cls}}-1$ prototypes inside a single softmax term. This resolves the three ambiguities the audit flagged together:

- **Flaw 1** (missing $\sum_c$) — resolved by making explicit that alignment is *matched-class*, not summed over all classes: each location contributes exactly one contrastive term, with the sum over classes appearing inside the softmax denominator (the standard negative-set), not as an outer sum that would otherwise incorrectly pull one pixel toward every class simultaneously.
- **Flaw 2** (undefined functional form) — resolved as InfoNCE-style contrastive, consistent with the CMPR/CMPC-style contrastive terms MFCPL uses for the same purpose (analysis file §11), rather than the weaker cosine-pull or MSE alternatives.
- **Flaw 3** (spatial-vs-vector mismatch) — resolved by defining the loss per sampled *location*, using the downsampled ground-truth mask to determine each location's class — the audit's own recommended "standard approach in prototype learning (as in FedAMM/RFNet)."

**Local prototype computation** (replaces the undefined "mean bottleneck embedding"):

$$\text{Proto}_{m,i}^{c,t} = \frac{1}{|P_m^c|}\sum_{p \in P_m^c} z_m(p), \qquad P_m^c = \{p \in P_m : c(p)=c\}$$

### 6.4 Aggregation & Broadcast **[Resolves Flaw 5]**

Server-internal global aggregate:
$$\theta_{\text{Encoder}_m}^{t+1} = \sum_{i:\,m\in O(i),\,R_{\text{send}}(i,m)=1} \frac{n_i}{N_m}\,\theta_{\text{Encoder}_m,i}^{t}$$

Personalized broadcast to client $k$ — with the missing $R_{\text{send}}$ check added:

$$\theta_{\text{Encoder}_m \to k}^{t+1} = \sum_{\substack{j:\,m\in O(j),\; R_{\text{send}}(j,m)=1,\\ R_{\text{recv}}(k,j,m)=1}} \frac{n_j}{N_m^{(k)}}\,\theta_{\text{Encoder}_m,j}^{t}, \qquad N_m^{(k)} = \sum_{\substack{j:\,R_{\text{send}}(j,m)=1,\\R_{\text{recv}}(k,j,m)=1}} n_j$$

A client $j$ who has *not* consented to send modality $m$ can no longer be silently included in another client's aggregate, closing the exact gap the audit identified.

**Prototype aggregation** — refined to weight by class-evidence rather than patient count, since a client contributing few class-$c$ pixels shouldn't count equally to one contributing many:

$$\text{Proto}_m^{c,t+1}\!\to\! k = \sum_{j \in (\cdot)} \frac{n_j^{c,m}}{N_m^{c,(k)}}\,\text{Proto}_{m,j}^{c,t}, \qquad n_j^{c,m} = |P_{m,j}^c|$$

using the identical $R_{\text{send}}/R_{\text{recv}}$-gated client set as above.

### 6.5 Stopping Criterion **[Proposed — new]**

Run a minimum $T_1$ rounds (swept; candidate starting range 20–50), then continue until prototype drift falls below threshold for $K$ consecutive rounds:

$$\frac{1}{4\,C_{\text{cls}}}\sum_{m}\sum_{c}\big\|\text{Proto}_m^{c,t} - \text{Proto}_m^{c,t-1}\big\|_2 < \epsilon \quad \text{for } K \text{ consecutive rounds}$$

$\epsilon$, $K$ are hyperparameters (open — see Appendix C). This gives the Phase Controller a concrete, checkable signal for when representations have "stabilized," rather than an arbitrarily fixed round count.

### 6.6 Phase 1 Pseudocode

```
for round t = 1 .. (until stopping criterion met):
    for each client i in parallel:
        pull Encoder_m, {Proto_m^c} for m in O(i), gated by R_send/R_recv (§6.4)
        for m in O(i):
            z_m ← Encoder_m(x_m)
            loss_m ← L_unimodal-align(z_m, {Proto_m^c})
        loss ← λ1 * Σ_m loss_m
        backprop through Encoder_m only  (no Decoder_S, no FusionHead_S exist)
        local SGD steps
        push updated Encoder_m deltas (m where R_send(i,m)=1)
        push local Proto_m^c estimates + counts n_i^{c,m}

    server:
        aggregate Encoder_m per §6.4
        aggregate Proto_m^c per §6.4 (evidence-weighted)
        Phase Controller: check stopping criterion (§6.5)
```

---

## 7. Phase Transition — Freeze Procedure

Executed once, by the Phase Controller, when Phase 1's stopping criterion is met:

1. Server finalizes $\theta_{\text{Encoder}_m}^{*}$ and $\text{Proto}_m^{c,*}$ for all $m$, $c$ — the last Phase-1 aggregate becomes the permanent snapshot.
2. Server broadcasts $\text{Encoder}_m^{*}$ to every client currently permitted to receive it (same $R_{\text{recv}}$ gating as the final Phase 1 round) — this is the *only* time an encoder is transmitted after this point; there is no further re-broadcast.
3. Each client sets `requires_grad=False` on all locally-held encoder parameters **and** wraps every encoder forward call with `detach()` on its output before that output is consumed downstream — **[Principle 4, §2]** — so the guarantee holds at the graph level, not only via optimizer configuration.
4. The Unimodal Bank is now static for the remainder of training. No further Phase 1 rounds occur.
5. Phase Controller sets `CURRENT_PHASE ← 2`; fusion tracks are instantiated (freshly initialized, or cold-started per §13 if the owned subset is new).

From this point forward, $\text{Encoder}_m$'s broadcast weights are, **by construction**, a pure function of Phase 1 data only — see the formal argument in §11.

---

## 8. Phase 2 — Isolated Multimodal Fusion

### 8.1 Objective & Active/Frozen State

**Goal:** each hospital learns to fuse its own frozen, already-stable unimodal features into a segmentation, entirely within its own hard-isolated track.

| Component | State |
|---|---|
| $\text{Encoder}_m^{*}$ | **Frozen** — used only for inference (feature extraction), `detach()`-ed at output |
| $\text{Proto}_m^c$ | **Frozen** — no longer used at all in Phase 2 |
| $\text{FusionHead}_S$, $\text{Decoder}_S$, $\text{FusedProto}_S^c$ | **Trainable**, isolated per track |

### 8.2 Data & Gradient Flow (per client, owned subset $S$)

```
   x_m  (m ∈ S)
     │
     ▼
   Encoder_m*  [FROZEN]
     │
     ▼
   z_m ──────► detach() ⊘  ◄── gradient path physically ends here;
                  │              nothing upstream of this point is
                  ▼              ever touched by Phase 2 backprop
             FusionHead_S
                  │
                  ▼
               z_S ──────► L_fusion-align(z_S, FusedProto_S^c)
                  │
                  ▼
             Decoder_S
                  │
                  ▼
                ŷ  ──────► L_task(y, ŷ)
                  │
                  ▼
      backprop: Decoder_S ← FusionHead_S   (stops at the ⊘ detach point —
                                              never reaches Encoder_m*)
```

The `⊘` marks the exact point that used to be the leakage channel (Flaw 8's step 2–3): task-loss gradient now terminates there instead of continuing into the shared encoder.

### 8.3 Loss Function

$$\mathcal{L}_i^{\text{Phase 2}} = \mathcal{L}_{\text{task}}(y,\hat y) + \lambda_2\,\mathcal{L}_{\text{fusion-align}}(z_S, \{\text{FusedProto}_S^c\}_c)$$

$\lambda_1$'s term is **absent by construction** — there is no encoder parameter left to regularize toward a prototype in Phase 2, so including it would be a computed no-op. $\mathcal{L}_{\text{fusion-align}}$ takes the same resolved contrastive form as $\mathcal{L}_{\text{unimodal-align}}$ (§6.3), applied in the fused embedding space, for consistency:

$$\mathcal{L}_{\text{fusion-align}} = -\frac{1}{|P_S|}\sum_{p\in P_S} \log \frac{\exp(\text{sim}(z_S(p), \text{FusedProto}_S^{c(p)})/\tau)}{\sum_{c'} \exp(\text{sim}(z_S(p), \text{FusedProto}_S^{c'})/\tau)}$$

### 8.4 Aggregation & Broadcast **[Resolves Flaw 6]**

The audit noted the paper's track-aggregation rule was prose-only and didn't address two same-subset clients that might still refuse to pool. Resolved with the track-level matrices from §4.3:

$$\theta_{\text{FusionHead}_S}^{t+1} = \sum_{i:\,O(i)=S,\,R_{\text{send}}^{\text{track}}(i,S)=1} \frac{n_i}{N_S}\,\theta_{\text{FusionHead}_S,i}^{t}$$

$$\theta_{\text{FusionHead}_S \to k}^{t+1} = \sum_{\substack{j:\,O(j)=S,\,R_{\text{send}}^{\text{track}}(j,S)=1,\\R_{\text{recv}}^{\text{track}}(k,j,S)=1}} \frac{n_j}{N_S^{(k)}}\,\theta_{\text{FusionHead}_S,j}^{t}$$

— identical structure for $\text{Decoder}_S$ and $\text{FusedProto}_S^c$. Default value $1$ whenever $O(i)=O(j)=S$ reproduces the paper's original assumption exactly; the matrix only does anything when governance actively restricts it, so nothing changes for the base BraTS 3-hospital setup unless Hospital-D-style ablations need it.

### 8.5 Phase 2 Pseudocode

```
Phase Controller broadcasts frozen Encoder_m* once (§7); not repeated below.

for round t = 1 .. T2:
    for each client i in parallel:
        S ← O(i)
        pull FusionHead_S, Decoder_S, FusedProto_S^c
             gated by R_send^track / R_recv^track (§8.4)
             [if track S is new: cold-start init, §13]
        local_train():
            for m in S: z_m ← detach(Encoder_m*(x_m))     # frozen, detached
            z_S ← FusionHead_S({z_m : m in S})
            y_hat ← Decoder_S(z_S)
            loss ← L_task + λ2 * L_fusion-align(z_S, FusedProto_S^c)
            backprop  →  updates FusionHead_S, Decoder_S only
            local SGD steps
        push updated FusionHead_S, Decoder_S deltas (track S only)
        push local FusedProto_S^c estimates

    server:
        for each track S: aggregate FusionHead_S, Decoder_S per §8.4
        update FusedProto_S^c as evidence-weighted average (as §6.4, applied to fused space)
```

---

## 9. Full Combined Two-Phase Training Loop

```
PHASE_CONTROLLER.CURRENT_PHASE ← 1

while CURRENT_PHASE == 1:
    run one Phase 1 round (§6.6)
    if stopping criterion met (§6.5):
        execute freeze procedure (§7)
        CURRENT_PHASE ← 2

while CURRENT_PHASE == 2 and t < T_max:
    run one Phase 2 round (§8.5)
    t += 1

# No round ever executes both phases' losses simultaneously.
# No parameter is ever active in both phases.
```

---

## 10. Per-Hospital Instantiation

```
                          PHASE 1  (rounds 1 .. T1)
  ┌────────────┐   ┌────────────┐   ┌────────────┐
  │ Hospital A  │   │ Hospital B  │   │ Hospital C  │
  │ trains:     │   │ trains:     │   │ trains:     │
  │ Enc_T2      │   │ Enc_T1      │   │ Enc_T1      │
  │ Enc_FLAIR   │   │ Enc_T1ce    │   │ Enc_FLAIR   │
  │ (Proto_T2,  │   │ Enc_T2      │   │ (Proto_T1,  │
  │  Proto_FLAIR)│  │ Enc_FLAIR   │   │  Proto_FLAIR)│
  └────────────┘   │ (all 4 Proto)│   └────────────┘
                    └────────────┘
        A never touches Enc_T1ce — it's never broadcast to A,
        because A never owns T1ce and never requests it.

                    ── FREEZE (§7) ──

                          PHASE 2  (rounds T1+1 .. T_max)
  ┌────────────────┐  ┌────────────────────┐  ┌────────────────┐
  │ Hospital A       │  │ Hospital B           │  │ Hospital C       │
  │ Enc_T2*, Enc_    │  │ Enc_T1*, Enc_T1ce*,  │  │ Enc_T1*, Enc_    │
  │ FLAIR*  [FROZEN] │  │ Enc_T2*, Enc_FLAIR*  │  │ FLAIR*  [FROZEN] │
  │        ⊘         │  │      [FROZEN]  ⊘     │  │        ⊘         │
  │ FusionHead_S_A   │  │ FusionHead_S_B       │  │ FusionHead_S_C   │
  │ Decoder_S_A      │  │ Decoder_S_B          │  │ Decoder_S_C      │
  │ FusedProto_S_A^c │  │ FusedProto_S_B^c     │  │ FusedProto_S_C^c │
  │ [TRAINABLE]      │  │ [TRAINABLE]          │  │ [TRAINABLE]      │
  └────────────────┘  └────────────────────┘  └────────────────┘
    HARD-ISOLATED — never aggregated across S_A / S_B / S_C
```

Hospital D ($S_D=\{\text{T1ce,T2}\}$) joins in Phase 2 only, for Ablations A2/A5, using cold-start (§13) since $S_D$ has no existing track.

---

## 11. Formal Argument: Why Method 2 Closes Flaw 8

**Claim:** under this architecture, $\theta_{\text{Encoder}_m}^{*}$ (the value ever broadcast to any client) is provably a function of Phase 1 data alone, regardless of what happens afterward.

**Argument:**
- *Phase 1:* the only loss term computed is $\mathcal{L}_{\text{unimodal-align}}(z_m, \text{Proto}_m^c)$, a function of $z_m$ (from $\text{Encoder}_m$ alone) and $\text{Proto}_m^c$ (aggregated only from same-modality clients). No other modality's encoder, no fusion head, and no decoder exist in the computation graph during Phase 1 — there is no tensor operation through which a different modality's data could influence $\partial \mathcal{L}/\partial\theta_{\text{Encoder}_m}$. This closes the channel described in Flaw 8, steps 2–3 (gradient conditioned on a co-present modality at the fusion input) at the source, because the fusion input doesn't exist yet.
- *Phase transition:* $\theta_{\text{Encoder}_m}^{*}$ is fixed at this point and never updated again.
- *Phase 2:* $\text{Encoder}_m^{*}(x_m)$ is computed and immediately `detach()`-ed. Even though $\text{FusionHead}_S$ and $\text{Decoder}_S$ *do* mix modalities and *do* receive task-loss gradient, that gradient's backward pass terminates at the detach point (§8.2 diagram) — it is never accumulated into $\theta_{\text{Encoder}_m}^{*}$, and `requires_grad=False` additionally guarantees no optimizer step could apply it even if it were somehow computed.

Since $\theta_{\text{Encoder}_m}^{*}$ is fixed before Phase 2 begins and provably receives zero gradient contribution during Phase 2, it cannot carry any information about modalities it wasn't trained on in Phase 1 — closing Flaw 8 as a structural guarantee rather than the soft, $\lambda_1$-weighted mitigation the audit correctly identified as insufficient.

---

## 12. Directional Consent Under Two-Phase Training

§5.8 of the original paper (group-symmetric as a special case of fully directional $R_{\text{recv}}$) applies unchanged to both phases — the aggregation equations in §6.4 and §8.4 are already written in the general per-recipient form; the group-symmetric BraTS default is simply the case where $R_{\text{recv}}(i,j,m)=1$ for every pair that both owns $m$ and has opted in. No structural change is needed for either phase to support full directionality — only, as the original paper notes, a shift in server-side storage from one aggregate per modality/track to up to $N$ per-recipient aggregates.

---

## 13. Cold-Start Under Two-Phase Training

The original cold-start patch (§5.7: initialize a new track by channel-subsetting the nearest superset track, then fine-tune) applies in Phase 2 only, since fusion tracks are the only thing being cold-started. Two properties change relative to Method 1, both favorably:

- **Existing subsets are now trivial to onboard.** A new client whose owned subset already has a track just pulls the current $\text{FusionHead}_S$/$\text{Decoder}_S$ and the already-frozen $\text{Encoder}_m^{*}$ set — no encoder bootstrapping is needed at all, since the Unimodal Bank stopped changing after Phase 1.
- **Genuinely new subsets should cold-start onto a more stable target.** In Method 1, a superset track being channel-subsetted for initialization is itself still a moving target (encoders and decoder co-adapting every round). Under Method 2, by the time any cold start happens, $\text{Encoder}_m^{*}$ is already frozen — so the features a new track is bootstrapping onto aren't shifting under it. This should make Ablation A5's cold-start convergence faster and more stable than under Method 1, though this is a hypothesis to verify empirically, not an assumed result.

---

## 14. Proposed Extension: Multi-Track Contribution for Superset Clients **[Resolves Flaw 9]**

The audit flagged that Hospital B (owning all 4 modalities) is assigned only to its own full track $S_B$, leaving the minority tracks $S_A$ and $S_C$ trained on just Hospital A's (~30%) and Hospital C's (~25%) patients respectively — small pools for entire decoder pathways.

**Mechanism:** a client $i$ with $O(i) \supset S$ for some other existing track $S$ may, *only if it opts in* via $R_{\text{contribute}}(i,S)=1$, additionally run a masked forward/backward pass — dropping every channel outside $S$ — through $\text{FusionHead}_S$/$\text{Decoder}_S$, contributing to Track $S$'s aggregate exactly as if it were a native $S$-owner for that pass.

```
Hospital B, Phase 2, per round:
    pass 1 (native):    x_{T1,T1ce,T2,FLAIR} → FusionHead_S_B → Decoder_S_B
    pass 2 (opt-in,     x_{T2,FLAIR}          → FusionHead_S_A → Decoder_S_A
      if R_contribute
      (B, S_A) = 1):
```

**Why this doesn't weaken the consent guarantee:** Track $S_A$'s isolation invariant is "parameters shaped only by $\{\text{T2,FLAIR}\}$-restricted signal" — and that's exactly what Hospital B's masked pass provides. Hospital A never receives anything beyond what a genuine $\{\text{T2,FLAIR}\}$-owning hospital would have sent; the fact that the underlying patient also happens to have a T1ce scan Hospital B chose not to use for this pass is irrelevant to Track $S_A$'s guarantee. Default is **off** ($R_{\text{contribute}}=0$) — this is a data-usage expansion for Hospital B specifically and should never be assumed silently, consistent with every other policy gate in this architecture being opt-in and exogenous.

This directly answers the audit's open question ("does B also contribute to Track $S_A$/$S_C$? ... the architecture as described doesn't support a client contributing to multiple tracks simultaneously") by adding exactly that support, gated behind its own consent flag.

---

## 15. Communication & Compute Profile by Phase

| | Phase 1 | Phase 2 |
|---|---|---|
| **Components transmitted** | $\text{Encoder}_m$, $\text{Proto}_m^c$ only | $\text{FusionHead}_S$, $\text{Decoder}_S$, $\text{FusedProto}_S^c$ only (plus one final frozen-encoder broadcast at transition) |
| **Payload per round** | 4 encoders' worth, gated by modality ownership | Up to $|\{O(i):i\in C\}|$ tracks' worth, gated by subset membership |
| **Peak concurrent traffic** | Lower than Method 1's joint round (fusion/decoder payloads aren't moving yet) | Lower than Method 1's joint round (encoder payloads have stopped) |

Total communication volume across both phases combined is the same order as Method 1's single-phase training summed over an equivalent number of rounds — Method 2 **time-slices** what gets synchronized rather than adding to it. This is worth stating explicitly since a natural (incorrect) assumption is that two phases means twice the communication cost.

---

## Appendix A — Purity Probe Formalization **[Resolves Flaw 12]**

Not part of the trained model — an evaluation-time diagnostic used in Ablation A3.

- **Training data:** embeddings from two encoder snapshots trained under identical conditions except modality exposure — one "clean" (trained only ever seeing consented modalities) and one "contaminated" (trained with a forbidden modality artificially permitted) reference run per restricted hospital.
- **Labels:** binary — clean vs. contaminated — at the level of the encoder snapshot, not per-sample, since contamination is a property of training history, not of any individual input.
- **Extraction layer:** the bottleneck embedding $z_m$, matching where $\text{Proto}_m^c$ is defined, for consistency with the rest of the architecture's notion of "representation."
- **Anatomy-confound control:** because MRI sequences share underlying anatomy, some probe accuracy above chance is expected even with zero true leakage. Report probe accuracy **relative to a same-anatomy, different-random-seed clean/clean pair** (same clean training procedure, two random seeds) as the null baseline, rather than relative to literal 50% chance — a violation is probe accuracy meaningfully above *that* baseline, not above 50%.
- **Expected result under this architecture:** because $\theta_{\text{Encoder}_m}^{*}$ is provably a function of Phase 1 data only (§11), a purity probe run against CAMFS-Method-2 encoders should sit at the clean/clean null baseline by construction — this is now a confirmatory check on the architecture's own guarantee, not just a comparison point against the DisentAFL-style soft baseline.

---

## Appendix B — Flaw Resolution Map

| Flaw (audit) | Resolved by | Section |
|---|---|---|
| 1 — class-summation ambiguity | Matched-class contrastive form | §6.3 |
| 2 — undefined loss functional form | InfoNCE-style contrastive, specified | §6.3 |
| 3 — spatial/vector mismatch | Location-sampled, mask-matched loss | §6.3 |
| 4 — Eq. 2/3 redundancy | Explicit server-internal vs. broadcast split | §6.4 |
| 5 — missing $R_{\text{send}}$ in broadcast | Added condition | §6.4 |
| 6 — track-aggregation unformalized | $R_{\text{send}}^{\text{track}}$/$R_{\text{recv}}^{\text{track}}$ + explicit equations | §8.4 |
| 7 — "gated running average" undefined | Evidence-weighted average, explicit formula | §6.4 |
| **8 — encoder leakage channel** | **Structural: no fusion graph in Phase 1; detach + freeze in Phase 2** | **§6–§8, formal argument §11** |
| 9 — Hospital B superset problem | Multi-track contribution, opt-in | §14 |
| 10 — no gradient stopping | `detach()` at fusion input, Phase 2 | §8.2 |
| 12 — purity probe unformalized | Full protocol, anatomy-confound control | Appendix A |
| 11, 13, 14 | Not addressed by this document — orthogonal to the training-method question (convergence theory, BraTS label bias, baseline engineering effort) | — |

---

## Appendix C — Remaining Open Design Choices

Honest flags on what's proposed-but-unvalidated in this document:

- Exact $T_1$, $\epsilon$, $K$ values for the Phase 1 stopping criterion (§6.5) — mechanism is specified, thresholds are empirical.
- Whether $\tau$ (contrastive temperature) should be swept jointly with $\lambda_1$/$\lambda_2$ or fixed — suggested starting point $\tau=0.1$ is a common default, not derived here.
- Whether $R_{\text{contribute}}$ (§14) should be per-track or a single federation-wide flag per client — specified here as per-(client, track) for maximum expressiveness, which is more storage than a coarser design would need.
- Whether Ablation A4 ($\lambda_1$ sweep) still needs re-deriving now that Phase 1 has no competing task loss (flagged in the prior synthesis report; not resolved here since it's an experimental-design question, not an architectural one).
- Whether staged/frozen-then-fine-tune training has direct precedent elsewhere in federated multimodal learning specifically (as opposed to general transfer learning) — worth a targeted literature check before presenting Method 2 as a clean, uncontested pivot in the paper itself.
