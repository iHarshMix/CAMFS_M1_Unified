# CAMFS M1: Consent-Aware Multimodal Federated Segmentation
## Complete Research Implementation Specification

1. [Research objective and scope](#1-research-objective-and-scope)
2. [Literature gaps](#2-literature-gaps)
3. [Formal requirements](#3-formal-requirements)
4. [Notation](#4-notation)
5. [Component inventory](#5-component-inventory)
6. [Phase 1 — Unimodal contrastive alignment](#6-phase-1--unimodal-contrastive-alignment)
7. [Phase transition — Freeze procedure](#7-phase-transition--freeze-procedure)
8. [Phase 2 — Track-isolated fusion training](#8-phase-2--track-isolated-fusion-training)
9. [Send-gated track routing](#9-send-gated-track-routing)
10. [Cold-start and lineage](#10-cold-start-and-lineage)
11. [Multi-track contribution](#11-multi-track-contribution)
12. [Formal purity guarantees](#12-formal-purity-guarantees)
13. [Four-hospital federation](#13-four-hospital-federation)
14. [Experimental plan](#14-experimental-plan)
15. [Reproducible implementation defaults](#15-reproducible-implementation-defaults)
16. [Communication profile](#16-communication-profile)
17. [Paper claims and limitations](#17-paper-claims-and-limitations)
18. [Forward compatibility with M2](#18-forward-compatibility-with-m2)
19. [Completeness and go/no-go checklist](#19-completeness-and-gono-go-checklist)

---

## 1. Research objective and scope

### 1.1 The Core Problem
In real-world healthcare federated learning, **missing modalities are not random accidents — they are institutional decisions.** A hospital may lack an MRI scanner (hardware constraint), may have ethics board restrictions against sharing PET-derived data (legal constraint), or may be bound by a Memorandum of Understanding that limits what imaging data it contributes to a collaboration (contractual constraint).

Every existing multimodal federated learning method treats modality routing as a **performance optimization problem**: route knowledge wherever it improves accuracy. CAMFS treats it as a **compliance problem**: route knowledge only where institutional policy permits, even when blocking would measurably improve performance.

### 1.2 The M1 Hard Constraint
M1 solves consent-aware federated learning by keeping modality encoders temporally isolated and fusion tracks structurally isolated. Its hard constraint is:

\[
\operatorname{Recv}(i)\subseteq O(i).
\]

A client can receive pooled benefit only for modalities it physically owns. 

Explicitly, the relaxation of this constraint (cross-boundary absorption of non-owned modalities) is deferred to a follow-up paper (M2). This document serves as a standalone, publishable paper specification focusing purely on the consent-constrained architecture that strictly enforces this boundary.

---

## 2. Literature gaps

| Gap | Description |
| :--- | :--- |
| **Gap 1 — Receive-side filtering** | Every existing paper considers only what a client *sends*. No one has formalized what a client is *permitted to receive* as an independent policy decision. |
| **Gap 2 — Asymmetric bidirectional gradient flow** | No paper models a training round where Hospital B accepts all modality signals, Hospital A accepts only \(\{T2, \text{FLAIR}\}\)-infused signals, and Hospital C accepts only \(\{T1, \text{FLAIR}\}\)-infused signals simultaneously. This is a directed graph of gradient flows, not symmetric averaging. |
| **Gap 3 — Gradient purity by modality** | When a client trains a joint encoder on multiple modalities, its gradient carries mixed modality signal. Even a T2 encoder gradient, trained in the presence of T1ce data through a shared fusion head, carries indirect T1ce signal. The architectural fix is strictly separate modality encoders with temporal isolation. |
| **Gap 4 — MoU/consent matrix as a formal FL component** | No paper has introduced a policy/consent matrix as a first-class architectural component sitting above the aggregation mechanism. The legal and institutional framing is entirely absent from FL literature. |
| **Gap 5 — Convergence under asymmetric aggregation** | FedAvg and FedProx convergence proofs assume symmetric, global aggregation. Asymmetric routing means each client receives a different personalized model — the convergence theory needs to be rebuilt. *(Deferred to future work.)* |

*(Note: Gap 6 regarding cross-boundary absorption is deferred to the M2 follow-up paper).*

---

## 3. Formal requirements

Given a federation of clients \(C = \{1, \dots, N\}\), where each client \(i\) has a fixed, structurally owned modality subset \(O(i) \subseteq \mathcal{M}\) (complete missing-modality assumption), and an exogenous, non-learnable consent policy, the system must satisfy:

| # | Requirement | Description |
| :--- | :--- | :--- |
| **R1** | **Performance Under Compliance** | Empirically optimize and report task performance (segmentation Dice/HD95) for every client, **subject to** never violating any specified consent constraint — even when violating it would improve performance. This is not a claim of a global optimum for the non-convex FL objective. |
| **R2** | **Auditability** | Make the enforced constraint set **auditable** — inspectable by a third party (IRB, compliance officer) as an explicit, exogenous, versioned object. Not inferable only from training dynamics or learned gates. |
| **R3** | **Graceful Degradation** | A client whose exact modality subset has not been seen before should not need to train an entirely new pathway from nothing. Cold-start initialization should leverage existing compatible components. |

---

## 4. Notation

| Symbol | Meaning |
| :--- | :--- |
| \(C=\{1,\ldots,N\}\) | Hospitals |
| \(\mathcal M=\{\mathrm{T1},\mathrm{T1ce},\mathrm{T2},\mathrm{FLAIR}\}\) | Modality universe |
| \(O(i)\subseteq\mathcal M\) | Modalities physically owned by hospital \(i\) |
| \(\operatorname{Send}(i)\subseteq O(i)\) | Modalities hospital \(i\) contributes in M1 |
| \(\operatorname{Recv}_{\mathrm{M1}}(i)\subseteq O(i)\) | Owned modalities for which \(i\) accepts M1 pooled benefit |
| \(S, S'\) | A modality subset identifying a fusion track (\(S'\) denotes a send-keyed track) |
| \(C_{\text{cls}} = 4\) | Segmentation classes: \(c \in \{0: \text{BG},\; 1: \text{NCR/NET},\; 2: \text{ED},\; 4: \text{ET}\}\) |
| \(\text{Encoder}_m\), \(\text{Encoder}_m^*\) | Modality-\(m\) encoder; \(^*\) denotes the frozen post-Phase-1 snapshot |
| \(\text{Proto}_m^c\) | Global unimodal class-\(c\) prototype for modality \(m\) (\(\in \mathbb{R}^{256}\)) |
| \(F_S,D_S\) | M1 fusion head and decoder for track \(S\) |
| \(\text{FusedProto}_S^c\) | Fused class-\(c\) prototype for track \(S\) |
| \(z_m(p)\) | Encoder-\(m\) bottleneck feature at spatial location \(p\) |
| \(P_m\), \(P_m^c\) | Sampled bottleneck locations for modality \(m\); subset with ground-truth class \(c\) |
| \(\tau\) | Contrastive temperature (default: \(0.1\)) |
| \(\operatorname{sim}(u, v)\) | Cosine similarity |
| \(\operatorname{sg}(\cdot)\) | Stop-gradient operator |
| \(\operatorname{ImageLineage}(\theta)\) | Image modalities directly present in forward/training paths shaping \(\theta\) |
| \(R_{\text{send}}(i, m)\) | Client \(i\) may contribute modality-\(m\) encoder updates |
| \(R_{\text{recv}}(i, j, m)\) | Client \(i\) may receive modality-\(m\) signal originating from \(j\) |
| \(R_{\text{send}}^{\text{track}}(i, S')\) | Client \(i\) may contribute to track \(S'\)'s fusion/decoder aggregate |
| \(R_{\text{recv}}^{\text{track}}(i, j, S)\) | Client \(i\) may receive track-\(S\) signal from client \(j\) |
| \(R_{\text{contribute}}(i, S)\) | Superset-owning client \(i\) opts in to help train non-native track \(S\) |
| \(\kappa\) | Consent cohort key compiling pairwise policies into a closed set |

---

## 5. Component inventory

CAMFS M1 consists of **6 architectural components**:

### 5.1 Unimodal Encoder Bank
Four independent encoders, one per modality: \(\text{Encoder}_{\text{T1}}\), \(\text{Encoder}_{\text{T1ce}}\), \(\text{Encoder}_{\text{T2}}\), \(\text{Encoder}_{\text{FLAIR}}\).

Define \(\operatorname{ConvBlock}_{C_{\mathrm{in}}\rightarrow C_{\mathrm{out}}}\) as two \(3\times3\), stride-1, padding-1 convolutions, each followed by 8-group GroupNorm and SiLU. Each downsampling operation is \(2\times2\) max-pooling.

| Tensor | Operation | Channels | Resolution |
| :--- | :--- | ---: | :--- |
| \(h_m^{(1)}\) | ConvBlock before pool 1 | 32 | \(240\times240\) |
| \(h_m^{(2)}\) | ConvBlock before pool 2 | 64 | \(120\times120\) |
| \(h_m^{(3)}\) | ConvBlock before pool 3 | 128 | \(60\times60\) |
| \(h_m^{(4)}\) | ConvBlock before pool 4 | 256 | \(30\times30\) |
| \(z_m\) | bottleneck ConvBlock after pool 4 | 256 | \(15\times15\) |

No external pretrained checkpoint is used: every freshly initialized convolution uses PyTorch Kaiming-normal with `mode="fan_out"`, `nonlinearity="relu"`, and zero bias under a registered seed.

Encoders are trained only in Phase 1 and frozen before any segmentation task loss.

### 5.2 Same-modality prototypes
For downsampled class \(c\):

\[
\operatorname{Proto}_{m}^{c}
=
\frac{1}{|P_m^c|}
\sum_{p\in P_m^c} z_m(p).
\]

These prototypes organize each modality's features by class, updated in Phase 1 and frozen at the phase transition.

### 5.3 Policy Matrices
All matrices are **exogenous, versioned, non-learned, and inspectable**.

| Matrix | Governs | Default |
| :--- | :--- | :--- |
| \(R_{\text{send}}(i, m)\) | Client \(i\) may contribute modality-\(m\) encoder updates | \(1\) if \(m \in \text{Send}(i)\) |
| \(R_{\text{recv}}(i, j, m)\) | Client \(i\) may receive modality-\(m\) signal originating from \(j\) | \(1\) if \(m \in O(i) \cap O(j)\) and both opted in |
| \(R_{\text{send}}^{\text{track}}(i, S')\) | Client \(i\) may contribute to track \(S'\)'s fusion/decoder aggregate | \(1\) when \(\text{Send}(i) = S'\) |
| \(R_{\text{recv}}^{\text{track}}(i, j, S)\) | Client \(i\) may receive track-\(S\) signal from client \(j\) | \(1\) when \(S \subseteq O(i)\) and \(i\) explicitly accepts \(j\)'s track-\(S\) contribution |
| \(R_{\text{contribute}}(i, S)\) | Superset client \(i\) opts in to help train non-native track \(S\) | Default \(0\) (opt-in only) |

**Track receive safety rule:** A recipient \(i\) may load track \(S\) only if \(S\subseteq O(i)\) **and** \(i\) accepts every contributor in that track's cohort. The condition \(O(j)=S\) alone is **not** sufficient — that would allow H3 to receive H1's full-modality track merely because H1 trains it. This distinction is load-bearing for M1's purity guarantee (§12).

#### Primary policy manifest

The primary experiment uses the versioned manifest `M1_PRIMARY_V1`. This table is the complete allowed policy; every unlisted send, receive, track-receive, or contribution edge is zero. A listed recipient accepts **every** contributor shown for that state. "Delayed read" occurs only after the Phase-1 freeze and never makes the recipient a contributor to that cohort.

| State | Cohort contributors | Aggregate recipients | Delayed/private read-only recipients |
| :--- | :--- | :--- | :--- |
| \(\kappa_{\mathrm{T1}}\) | H1, H2, H3 | H1, H2, H3 | H4 |
| \(\kappa_{\mathrm{T1ce}}\) | H1 | H1 | H2 private copy; H4 |
| \(\kappa_{\mathrm{T2}}\) | H1, H2 | H1, H2 | H4 |
| \(\kappa_{\mathrm{FLAIR}}\) | H1, H3 | H1, H3 | none |

| Track cohort | Contributors | Authorized receivers | Additional rule |
| :--- | :--- | :--- | :--- |
| \(S_1=\{\mathrm{T1,T1ce,T2,FLAIR}\}\) | H1 | H1 | Native H1 track only |
| \(S_2=\{\mathrm{T1,T2}\}\) | H2 | H2 | Native H2 send-keyed track only |
| \(S_3=\{\mathrm{T1,FLAIR}\}\) | H3 and H1 masked to \(S_3\) | H1, H3 | \(R_{\mathrm{contribute}}(\mathrm{H1},S_3)=1\); both contributors mutually accept |
| \(S_4=\{\mathrm{T1,T1ce,T2}\}\) | H4 | H4 | Created only after the selected \(S_2\) checkpoint is frozen |

H2's full-owned-set private head is never a track cohort and is never transmitted. During ablation A8, H2 may additionally load H4's \(S_4\) track for inference only after H4 has released it; that edge is enabled only for A8 and is not part of the primary policy.

### 5.4 Subset fusion tracks
Each track \(S\) contains \(F_S\), \(D_S\), and \(\operatorname{FusedProto}_S^c\). Modalities are concatenated in the fixed universe order T1, T1ce, T2, FLAIR, skipping absent slots. For level \(r\in\{1,2,3,4,5\}\), with \(h_m^{(5)}=z_m\) and \(C_r=(32,64,128,256,256)\):

\[
u_S^{(r)}=\operatorname{Concat}_{m\in S}h_m^{(r)},
\qquad
f_S^{(r)}=
\operatorname{ConvBlock}_{C_r\rightarrow C_r}
\left(
\operatorname{Conv}_{1\times1}^{|S|C_r\rightarrow C_r}(u_S^{(r)})
\right).
\]

Write \(z_S=f_S^{(5)}\). The decoder \(D_S\) starts at \(z_S\), then repeats bilinear \(2\times\) upsampling with `align_corners=False`, concatenation with \(f_S^{(4)},f_S^{(3)},f_S^{(2)},f_S^{(1)}\), and a ConvBlock producing respectively 256, 128, 64, and 32 channels. A final \(1\times1\) convolution maps 32 channels to the four class logits.

### 5.5 Phase Controller
The research state machine manages the two-phase lifecycle:
```
PHASE_1_UNIMODAL
  -> PHASE_1_FROZEN
  -> PHASE_2_TRACK_TRAINING
  -> M1_RELEASED
```

### 5.6 Provenance Ledger
An auditable log records every cohort, track, and initialization decision. It is a research artifact, not a production audit service. Before training, write the canonical JSON policy manifest and its SHA-256 digest. Append one canonical JSONL record for each phase transition, cohort aggregation, track creation, and selected checkpoint containing at least `run_id`, `round`, `policy_digest`, `state_id`, `contributors`, `recipients`, `image_lineage`, `seed_mechanism`, `random_seed`, and the previous-record hash. The chain entry is \(h_k=\operatorname{SHA256}(h_{k-1}\Vert\operatorname{CanonicalJSON}(r_k))\). This makes accidental policy or lineage drift detectable within the released research bundle.

---

## 6. Phase 1 — Unimodal contrastive alignment

### 6.1 Objective
Each encoder trains independently against same-modality class prototypes:

\[
\mathcal L_{\mathrm{uni}}(z_m)
=
-\frac{1}{|P_m|}
\sum_{p\in P_m}
\log
\frac{
\exp(\operatorname{sim}(z_m(p),\operatorname{Proto}_m^{c(p)})/\tau)
}{
\sum_{c'\in\mathcal C_{\mathrm{supported}}}
\exp(\operatorname{sim}(z_m(p),\operatorname{Proto}_m^{c'})/\tau)
}.
\]

where \(\mathcal C_{\mathrm{supported}}\) contains only classes whose prototype has received at least one support update. **Empty prototype entries are masked out of the denominator.** A class with no cohort-wide support before its first observed round remains masked and does not contribute to the contrastive loss. This prevents undefined behaviour when a class has not yet been seen by any contributor in the cohort.

The client objective is:

\[
\mathcal L_i^{(1)}
=
\lambda_1
\sum_{m\in O(i)}
\mathcal L_{\mathrm{uni}}(z_m).
\]

### 6.2 Aggregation and Consent Cohorts

Pairwise receive policy cannot be applied after incompatible donor updates have already been mixed. M1 therefore compiles policy into **consent cohorts**. A cohort \(\kappa\) for modality \(m\) has one fixed eligible-sender set:

\[
\mathcal A_{m,\kappa}
=
\{j:m\in O(j),\ R_{\mathrm{send}}(j,m)=1,\ j\text{ is listed in }\kappa\}.
\]

It is admissible only when every contributor accepts the common state it will train:

\[
\operatorname{Closed}(m,\kappa)
\iff
\forall r,j\in\mathcal A_{m,\kappa},
\quad R_{\mathrm{recv}}(r,j,m)=1,
\]

with self-receive defined as true. Recipient \(i\) may load this cohort snapshot only if:

\[
m\in O(i)
\quad\land\quad
\forall j\in\mathcal A_{m,\kappa},\ R_{\mathrm{recv}}(i,j,m)=1.
\]

For one cohort:

\[
\theta_{E_{m,\kappa}}^{t+1}
=
\sum_{j\in\mathcal A_{m,\kappa}}
\frac{n_{j,m,\kappa}}{N_{m,\kappa}}
\theta_{E_{m,\kappa},j}^{t},
\qquad
N_{m,\kappa}=\sum_{j\in\mathcal A_{m,\kappa}}n_{j,m,\kappa},
\]

where \(n_{j,m,\kappa}\) is the number of distinct local patients used in that round.

For class prototypes, let \(s_{j,m,\kappa}^{c}\) be the number of patients containing at least one downsampled voxel of class \(c\), and let \(P_{j,m,\kappa}^{c}\) be the normalized mean embedding over their sampled class-\(c\) locations:

\[
P_{m,\kappa}^{c,t+1}
=
\operatorname{norm}
\left(
\frac{
\sum_{j\in\mathcal A_{m,\kappa}}
s_{j,m,\kappa}^{c}P_{j,m,\kappa}^{c,t}
}{
\sum_{j\in\mathcal A_{m,\kappa}}s_{j,m,\kappa}^{c}
}
\right).
\]

A site with \(s_{j,m,\kappa}^{c}=0\) contributes nothing for that class. If the cohort denominator is zero, the previous prototype is retained; before its first supported update, the entry is explicitly marked unset and remains masked from the loss.

**Round 0** is a no-optimizer bootstrap: sites run the registered initialized encoder once, compute supported local class means, and initialize the cohort prototypes. Contrastive encoder updates begin at round 1.

**For every round \(t\ge1\), the order is fixed:**

1. broadcast \(\theta_{E_{m,\kappa}}^{t}\) and \(P_{m,\kappa}^{c,t}\) to the cohort;
2. each contributor detaches and holds the received prototypes fixed for the complete local epoch;
3. reset a fresh local AdamW optimizer and update encoder parameters only;
4. after the last optimizer step, switch the post-local encoder to evaluation mode and recompute local class means/support counts without gradient;
5. upload the post-local encoder, means, and counts; then form \(\theta^{t+1}\) and \(P^{t+1}\) with the equations above.

There is no server optimizer and no local Adam/AdamW moment state persists across rounds.

Prototype recomputation is deterministic and independent of the stochastic training loader. Each contributor processes every one of the 155 axial slices from every local training patient exactly once, ordered first by patient identifier and then by slice index, with the fixed preprocessing but no augmentation. Masks are resized to the \(15\times15\) bottleneck grid by nearest-neighbour interpolation; every grid location is used. The encoder is in evaluation mode, accumulation is FP32, and only the final local means plus patient-support counts are uploaded.

**Primary paper cohorts:**

| Modality | Contributors | Recipients |
| :--- | :--- | :--- |
| T1 | H1, H2, H3 | H1, H2, H3 |
| T1ce | H1 only | H1 (H2 may fit a private copy after download; never aggregated) |
| T2 | H1, H2 | H1, H2 |
| FLAIR | H1, H3 | H1, H3 |

H4 is absent in Phase 1. It receives frozen encoder snapshots when it joins in Phase 2.

### 6.3 Stopping criterion
Let \(\mathcal K_m\) be the active consent cohorts for modality \(m\). After a minimum number of rounds, stop when mean prototype drift:

\[
\operatorname{Drift}_t
=
\frac{1}{C_{\mathrm{cls}}\sum_m|\mathcal K_m|}
\sum_m\sum_{\kappa\in\mathcal K_m}\sum_c
\left\|
\operatorname{Proto}_{m,\kappa}^{c,t}
-
\operatorname{Proto}_{m,\kappa}^{c,t-1}
\right\|_2
\]

remains below \(\epsilon\) for \(K\) rounds. Default: \(\epsilon=0.01\), \(K=5\).

---

## 7. Phase transition — Freeze procedure

The server finalizes every authorized \(E_{m,\kappa}^*\) and prototype bank. Each client:
1. loads the final authorized encoder snapshot;
2. sets encoder parameters to non-trainable;
3. sets frozen modules to evaluation mode;
4. disables mutation of normalization buffers;
5. wraps encoder outputs in stop-gradient.

\[
z_m=\operatorname{sg}(E_m^*(x_m)).
\]

For any Phase-2 loss:
\[
\frac{\partial\mathcal L^{(2)}}{\partial\theta_{E_m^*}}=0.
\]

Destroy the Phase-1 optimizer state and communication route before Phase 2 starts. The frozen encoder snapshots may be read by an authorized Phase-2 client but are never uploaded, aggregated, or used to seed a later encoder state.

---

## 8. Phase 2 — Track-isolated fusion training

### 8.1 Forward path
For track \(S\):

\[
\{h_m^{(r)},z_m\}_{m\in S}
\xrightarrow{F_S}
\{f_S^{(r)},z_S\}
\xrightarrow{D_S}
\ell_B
\xrightarrow{\operatorname{softmax}}
\hat y_B.
\]

Only modalities in \(S\) occur in this graph.

### 8.2 Objective
\[
\mathcal L_i^{(2)}
=
\mathcal L_{\mathrm{Dice+CE}}(y,\hat y_B)
+
\lambda_2\mathcal L_{\mathrm{fused-align}}
\left(z_S,\{\operatorname{FusedProto}_S^c\}_c\right).
\]

For fused bottleneck locations \(P_S\), let \(\mathcal C_{S,\mathrm{supported}}\) contain exactly the classes whose fused prototype has support. The alignment term is the same class-InfoNCE form used in Phase 1:

\[
\mathcal L_{\mathrm{fused-align}}
=
-\frac{1}{|P_S|}
\sum_{p\in P_S}
\log
\frac{
\exp(\operatorname{sim}(z_S(p),\operatorname{FusedProto}_S^{c(p)})/\tau)
}{
\sum_{c'\in\mathcal C_{S,\mathrm{supported}}}
\exp(\operatorname{sim}(z_S(p),\operatorname{FusedProto}_S^{c'})/\tau)
}.
\]

Each embedding and prototype is L2-normalized for cosine similarity. A location whose class is not yet supported is omitted from this term; the Dice + CE term still uses every labeled pixel. Only \(F_S,D_S,\operatorname{FusedProto}_S\) update. Encoders remain frozen with stop-gradient throughout.

### 8.3 Fused-prototype bootstrap

Before the first task update of a newly created track, run the same no-optimizer prototype bootstrap as Phase 1 on \(z_S\): each contributor processes all training patients in deterministic order, computes fused-bottleneck class means, and uploads means with patient-support counts. The server aggregates them using the same patient-weighted rule as §6.2.

**Unsupported-class masking:** A class with no cohort-wide bootstrap support remains masked from \(\mathcal L_{\mathrm{fused-align}}\) until its first supported round. This prevents dividing by zero or computing InfoNCE against an undefined prototype. When computing the fused-alignment loss, sum only over classes \(c\) for which \(\operatorname{FusedProto}_S^c\) has nonzero support; mask out the rest from the denominator.

### 8.4 Full round protocol

For every track round \(t\ge1\), the order is fixed:

1. broadcast the track state \(\theta_{F_{S,\kappa}}^{t}\), \(\theta_{D_{S,\kappa}}^{t}\), and fused prototypes to the cohort;
2. each contributor detaches and holds the received fused prototypes fixed for the complete local epoch;
3. reset a fresh local AdamW optimizer and update only \(F_S, D_S\) for one local epoch (batch size 16);
4. after the last optimizer step, switch the updated track to evaluation mode and recompute local fused class means/support counts using the deterministic full-training-patient pass (§6.2) — no augmentation, all 155 slices per patient, FP32 accumulation;
5. upload the post-local track parameters, fused means, and support counts; then aggregate with the equations below.

No track optimizer moment state persists across rounds.

### 8.5 Send-keyed aggregation

For track cohort \((S,\kappa)\), let:

\[
\mathcal I(S,\kappa)=
\left\{
i:
 i\text{ is listed in }\kappa
\land
R_{\mathrm{send}}^{\mathrm{track}}(i,S)=1
\land
\left[
\operatorname{Send}(i)=S
\ \lor\ 
\left(S\subsetneq\operatorname{Send}(i)
\land R_{\mathrm{contribute}}(i,S)=1\right)
\right]
\right\}.
\]

**Track-cohort closure rule:** The track cohort is admissible only if every member of \(\mathcal I(S,\kappa)\) accepts every other member's track update. A recipient may load the complete snapshot only if it accepts every contributor in that set. Otherwise a separate isolated cohort must be trained. No fused checkpoint is filtered after aggregation.

Then:

\[
\theta_{F_{S,\kappa}}^{t+1}
=
\sum_{i\in\mathcal I(S,\kappa)}
\frac{n_{i,S,\kappa}}{N_{S,\kappa}}
\theta_{F_{S,\kappa},i}^{t},
\qquad
N_{S,\kappa}
=
\sum_{i\in\mathcal I(S,\kappa)}n_{i,S,\kappa},
\]

where \(n_{i,S,\kappa}\) is the number of distinct patients in the native or masked \(S\)-only pass. The same patient weighting applies to \(D_S\). Fused prototypes use the class-specific patient-support rule from §6.2, including its zero-support behavior.

### 8.6 Model selection and stopping

- **Selection metric:** Highest contributor-patient-weighted validation macro Dice.
- **Improvement threshold:** Validation macro Dice increases by more than \(10^{-4}\); retain the earliest checkpoint on an exact tie.
- **Stopping:** No validation improvement for 10 consecutive rounds after round 20.
- **Round limits:** Minimum 20, maximum 100.

---

## 9. Send-gated track routing

A client with \(\text{Send}(i) \subsetneq O(i)\) cannot simply use cross-attention over its full modality set because \(\partial\mathcal{L}_{\text{task}} / \partial\theta_{\text{FusionHead}_S}\) would entangle all input modalities, violating the send restriction. 

Instead, the client is re-keyed into a **different, smaller, dedicated track**. Rather than contributing to a full track, it contributes to Track \(S' = \text{Send}(i)\). The excluded modality is structurally absent from this track's input slots.

On the receive side, the client has two options to regain the benefit of its withheld modality:
- **(a) Pull-Only**: If a peer fully sends to the larger track, the client can pull it for inference only (no training).
- **(b) Personalized Local Head (default)**: The client trains a private fusion head and decoder locally on all its frozen encoders. It is never transmitted.

---

## 10. Cold-start and lineage

A valid M1 seed for track \(S\) must satisfy the Lineage Rule:
\[
\operatorname{ImageLineage}(\theta_{\mathrm{seed}})\subseteq S.
\]

M1 removes tier ambiguity for the canonical setup:
- \(S_1,S_2,S_3\) start from registered Kaiming initialization.
- Delayed H4 (\(S_4 = \{\mathrm{T1},\mathrm{T1ce},\mathrm{T2}\}\)) joins only after \(S_2 = \{\mathrm{T1},\mathrm{T2}\}\) stabilizes. 
- At all five fusion levels, widen the first \(1\times1\) input from `[T1,T2]` to `[T1,T1ce,T2]`, zero-initialize the new T1ce block, and copy the remaining fusion and decoder blocks. Train \(S_4\) only with H4 updates.

#### Cold-start ablation definitions (A5)

The following three variants are the only A5 conditions. They all use H4's 132/19/37 split, the Phase-2 optimizer and stopping rule in §15.3, and the same frozen encoder snapshots.

| Variant | Initialization | Pre-task procedure |
| :--- | :--- | :--- |
| **Tier 1 — subset growth** | The canonical \(S_2\rightarrow S_4\) widening above, using the selected frozen \(S_2\) checkpoint | Run the normal Phase-2 procedure directly after the mandatory fused-prototype bootstrap. |
| **Tier 2 — track-local warm start** | Fresh registered Kaiming initialization for every \(S_4\) fusion/decoder tensor | After bootstrap, run 10 rounds of \(\mathcal L_{\mathrm{fused-align}}\) only, then begin the normal task-plus-alignment rounds. The 10 warm-start rounds do not count toward the 20-round minimum task-training budget. |
| **Tier 3 — fresh task training** | Fresh registered Kaiming initialization for every \(S_4\) fusion/decoder tensor | After bootstrap, begin normal task-plus-alignment rounds immediately. |

Every variant writes its source checkpoint, copied tensor names, zero-initialized tensor names, and image lineage to the provenance ledger. A source state with image lineage outside \(S_4\) is invalid and the run must stop before training.

---

## 11. Multi-track contribution

A superset-owning donor may train a smaller pure track only when:
\[
R_{\mathrm{contribute}}(i,S)=1.
\]
It executes a masked pass using exactly \(x_S\). For example, H1 uses this mechanism to help train the H3 pure track \(S_3=\{\mathrm{T1},\mathrm{FLAIR}\}\). The static policy sets mutual acceptance for this two-member track cohort.

---

## 12. Formal purity guarantees

### 12.1 Scope and assumptions

The following result is an **image-and-update-path** guarantee. It means excluded image tensors and updates whose local forward path read excluded image tensors cannot shape protected M1 state. It does not claim that BraTS labels are information-theoretically independent of T1ce: annotation provenance is a separate limitation (§13.6).

The result assumes: (a) the versioned manifest is enforced by the server and clients; (b) clients honestly report the modalities read by each local forward path; (c) no out-of-band parameter, optimizer, or checkpoint transfer occurs; (d) every seed satisfies the lineage rule; and (e) frozen encoders remain in evaluation mode with no mutable buffers.

### 12.2 Image-path purity theorem

For a track \(S\), define \(\operatorname{ImageLineage}(\theta)\) as the union of image modalities read by every local forward path, checkpoint seed, or aggregated update that shaped \(\theta\). Under the assumptions above:

\[
\operatorname{ImageLineage}(\theta_{F_{S,\kappa}}^t)
\cup
\operatorname{ImageLineage}(\theta_{D_{S,\kappa}}^t)
\subseteq S
\qquad\text{for every authorized track round }t.
\]

**Proof sketch.** The base case holds because fresh initialization has empty image lineage and every Tier-1 source is required to satisfy the lineage rule. In a local track update, the graph reads exactly \(x_S\); the frozen encoders, fusion head, decoder, and fused-prototype recomputation therefore add only modalities in \(S\). The track cohort closure rule aggregates only those valid local updates, and weighted averaging cannot add a modality absent from every summand. Induction over rounds proves the statement.

### 12.3 Encoder purity and composition

Phase 1 has no fusion module and each encoder \(E_m\) reads only \(x_m\) and same-modality prototypes. At Phase 2, stop-gradient gives \(\partial\mathcal L^{(2)}/\partial\theta_{E_m^*}=0\). Thus an authorized track has both a modality-pure frozen encoder interface and the image-path-pure fusion/decoder state proved above. This is a structural routing guarantee, not a privacy, legal-compliance, malicious-client, or convergence guarantee.

---

## 13. Four-hospital federation

### 13.1 Hospital Matrix

| Hospital | Owned \(O(i)\) | Send \(\text{Send}(i)\) | Receive \(\text{Recv}(i)\) | Joins | Lineage-audit role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H1** | \(\{T1, T1ce, T2, FLAIR\}\) | \(\{T1, T1ce, T2, FLAIR\}\) | \(\{T1, T1ce, T2, FLAIR\}\) | Phase 1 | Full anchor; contributes to \(S_3\) via \(R_{\mathrm{contribute}}\) |
| **H2** | \(\{T1, T1ce, T2\}\) | \(\{T1, T2\}\) | \(\{T1, T1ce, T2\}\) | Phase 1 | Send-gated; owns T1ce but withholds it |
| **H3** | \(\{T1, FLAIR\}\) | \(\{T1, FLAIR\}\) | \(\{T1, FLAIR\}\) | Phase 1 | Restricted target; audit target |
| **H4** | \(\{T1, T1ce, T2\}\) | \(\{T1, T1ce, T2\}\) | \(\{T1, T1ce, T2\}\) | Phase 2 | Delayed joiner |

### 13.2 Tracks
- \(S_1\): \(\{T1, T1ce, T2, FLAIR\}\) (H1)
- \(S_2\): \(\{T1, T2\}\) (H2)
- \(S_3\): \(\{T1, FLAIR\}\) (H3 + H1 masked)
- \(S_4\): \(\{T1, T1ce, T2\}\) (H4 delayed)

### 13.3 Dataset and Allocation
- **Dataset:** BraTS 2021 (1,251 labeled patients). BraTS 2020 may be used for code pilots only.
- **Allocation:** H1: 500 (40%), H2: 313 (25%), H3: 250 (20%), H4: 188 (15%)
- **Federation partition construction:** Sort official patient identifiers lexicographically. Use PCG64 seed 901 to select one fixed 50-patient H3 final-test cohort; these identities are excluded from training, validation, and calibration in every run. For federation-partition seeds \(\{1103,2207,3301\}\), independently permute the remaining 1,201 identities and assign: first 500 → H1, next 313 → H2, next 200 → H3 non-test, final 188 → H4.
- **Within-hospital splits:**
  - H1: 350 train / 50 validation / 100 held-out (for future release-audit; not used in M1 experiments but never trained on)
  - H2: 219 train / 31 validation / 63 test
  - H4: 132 train / 19 validation / 37 test
  - **H3: 175 train / 25 validation / 50 final-test** (test set fixed across partitions via PCG64 seed 901)

> **Note on M2 transition:** M2 will run its own M1 training with its own H3 split (125 train / 25 val / 25 M2-cal / 25 M2-accept-val / 50 test) to carve out calibration data. M1 maximizes its own H3 training data. The cost is that M2's base checkpoint is a separate run, not byte-identical to M1's. This is standard practice — extension papers retrain the base system with matched specs. Reviewers expect the same methodology, which is guaranteed by shared architecture, hyperparameters, dataset, and preprocessing (§18).

- **Training seeds within each partition:** \(\{17, 29, 43\}\)
- **Total experimental runs:** 3 partitions × 3 seeds = 9 M1 runs

### 13.4 Preprocessing

Use the released co-registered, skull-stripped, \(1\,\mathrm{mm}^3\) BraTS volumes on the \(240\times240\times155\) grid. For each patient and modality, compute mean and standard deviation over nonzero brain voxels, apply z-normalization with \(10^{-8}\) denominator floor, clip to \([-5,5]\), and keep outside-brain voxels at zero. Do not crop.

Split patients before extracting slices. Training uses aligned axial slices. Each epoch draws every tumour-containing slice once and an equal-sized, seeded sample of non-tumour brain-containing slices (1:1 ratio). Sample without replacement when the non-tumour pool is large enough and with replacement only when it is smaller than the tumour-slice count. Validation and testing retain all 155 slices, including empty slices.

**Augmentation:** Apply the same geometric transform to every co-registered input, its nonzero-brain mask, and the label: left-right flip with probability 0.5 and rotation sampled uniformly from \([-10^\circ,10^\circ]\), keeping the \(240\times240\) canvas, using bilinear interpolation for images and nearest-neighbour for masks/labels. Fill rotated-out image locations with 0 and label locations with 0. Independently per modality, sample intensity scale from \([0.9,1.1]\) and additive normalized shift from \([-0.1,0.1]\), apply both only where the transformed brain mask is nonzero, then reset every outside-mask pixel to 0.

### 13.5 3D Evaluation

At validation and test time, process all 155 axial slices without augmentation, stack predictions in original z-order, take the four-class argmax, and map model classes back to BraTS labels \(\{0,1,2,4\}\). Inference logits and softmax are FP32; an exact argmax tie resolves to the lowest class index. No connected-component or other postprocessing is used. Compute 3D patient-level regions:

\[
WT=\{1,2,4\},\qquad TC=\{1,4\},\qquad ET=\{4\}.
\]

For Dice, both prediction and reference empty gives 1 and exactly one empty gives 0. Define a 3D surface as foreground voxel centres with at least one 6-connected neighbour outside the mask or at a volume boundary. For HD95, when both surfaces are nonempty, compute Euclidean nearest-neighbour distances in physical millimetres in both directions, concatenate, and take the 95th percentile with linear interpolation. Both empty gives HD95 0; exactly one empty receives the fixed grid diagonal \(\sqrt{239^2+239^2+154^2}\) mm as a finite failure penalty. Report the count of one-empty cases and freeze the evaluator code hash before inspecting test results.

### 13.6 BraTS label caveat

BraTS enhancing tumour is defined using contrast-enhanced T1ce information. ET improvement is expected to be especially tied to T1ce-derived supervision. Report WT, TC, and ET separately. Do not average away region-level harm.

---

## 14. Experimental plan

**Primary question:** "Does CAMFS M1's consent-constrained architecture maintain competitive segmentation performance while providing an explicit, auditable image/update-path policy guarantee?"

### 14.1 Baselines

Every condition uses the fixed patient manifests, preprocessing, evaluator, and nine partition/seed pairs in §§13 and 15. No condition may use H3 final-test identities for model selection. The word "non-compliant" means the condition is an experimental reference, not a deployable alternative.

| # | Baseline | Exact comparison rule |
|---|----------|-------------|
| B1 | Local-Only | Each hospital trains its own encoders and native owned-set fusion/decoder on its local training patients only. It uses the same initialization, loss, local epoch count, validation selection, and maximum epochs as M1, but sends or receives no model state. H2's local model may use all of \(O(\mathrm{H2})\). |
| B2 | Policy-blind subset FedAvg | Use the M1 encoder and concat-fusion topology but ignore every M1 send, receive, and contribution policy. Each modality cohort contains every Phase-1 owner; each subset track accepts every site whose owned set contains that subset, using the masked subset input. This includes H2's T1ce updates and H1's masked contribution wherever structurally possible. It is a policy-blind reference, not an upper bound for every site. |
| B3 | DisentAFL reference reproduction | Reproduce the cited DisentAFL method from a registered source implementation and retain its published routing/gating objective. Adapt only its input/output heads to the four-class, 2D BraTS task. It receives no CAMFS policy matrix or lineage constraints. |
| B4 | Availability-only hard cohort | Use the M1 encoder/fusion topology and the same Phase separation, but construct all cohorts solely from modality availability: every owner of a modality contributes, and a fusion track contains only exact owned-set peers. No \(R\) matrix, explicit acceptance, or superset opt-in exists. |
| B5 | FedAMM reference reproduction | Reproduce the cited FedAMM method from a registered source implementation, using its per-combination prototype and aggregation rules. Adapt only the input/output heads, label mapping, data manifests, and evaluation procedure required for this BraTS task. |
| B6 | Centralized full-modality oracle | Train the same \(S_1\) concat-fusion architecture centrally on the union of all training patients with all four source volumes available. It is an intentionally policy-noncompliant descriptive ceiling, not a clinical or statistical upper bound. |

For B3 and B5, the preregistration bundle must name the paper version, official or author-provided code source, commit/archive hash, all changed files, and every hyperparameter changed for BraTS. A method must never be reported as a literal reproduction if this manifest is unavailable; in that case it is labelled an adaptation and its complete configuration is released. This avoids an undefined "-style" comparator.

### 14.2 Evaluation metrics
- Dice Score and HD95 per client per region (WT, TC, ET)
- Compliance-Performance Gap = \(\text{Dice}_{\text{Oracle}} - \text{Dice}_{\text{CAMFS}}(i)\)
- Policy and lineage-audit pass/fail, plus the number of rejected forbidden routes
- Representation drift (CKA) as a descriptive diagnostic only, never as evidence of forbidden-modality leakage

### 14.3 Ablation studies
- **A1 — Joint-training alternative:** Do not freeze after Phase 1. In each common round, optimize \(\mathcal L_{\mathrm{Dice+CE}}+\lambda_1\sum_m\mathcal L_{\mathrm{uni}}+\lambda_2\mathcal L_{\mathrm{fused-align}}\) and update encoders, fusion, and decoder together under the same routing matrix. This is deliberately not encoder-pure and is reported as a mechanism ablation, not an M1-compliant condition.
- **A2 — Delayed-site context:** Report the primary H1–H3 federation and the H4 delayed \(S_4\) result separately. H4 does not alter \(S_1,S_2,S_3\) under M1 isolation, so this is a cold-start/context study rather than a claim that merely adding a client improves every track.
- **A3 — Hard-policy versus soft-routing audit:** Run the executable lineage audit in §14.4 for CAMFS and B3. It tests actual routed update paths; it does not assume that every soft-routing method leaks.
- **A4 — Unimodal-alignment weight:** Run \(\lambda_1\in\{0,0.1,0.5,1.0\}\) on validation data only. The primary M1 value remains \(1\) regardless of this exploratory sweep.
- **A5 — Cold start:** Compare the three fully specified variants in §10.
- **A6 — Group-symmetric versus directional:** Keep all primary policy edges except T2. In the directional condition, H1 receives only its own T2 cohort, while H2 may download H1's frozen T2 snapshot and fit a private, non-uploaded T2 copy. No aggregate containing H2's T2 update is returned to H1. This tests the cost of a one-way policy without illegal post-mix filtering.
- **A7 — Multi-track contribution:** Compare the primary \(S_3\) cohort with \(R_{\mathrm{contribute}}(\mathrm{H1},S_3)=1\) against H3-only \(S_3\) training. All other settings are unchanged.
- **A8 — H2 reconnection:** Compare (a) H2's private full-owned-set head trained on its 219 training patients and selected on its 31 validation patients, using the §15.3 optimizer and stopping rule, with (b) inference-only loading of H4's released \(S_4\) track. The pull-only condition performs no H2 update on \(S_4\). Report it only after H4 has completed its own track training.

### 14.4 Executable lineage audit (Ablation A3)

**Purpose:** Verify the actual computation and aggregation routes, rather than infer leakage from representation similarity. A CKA comparison cannot prove modality leakage here because H1 and H3 legitimately load the same frozen T1 cohort encoder; on the same image it would make the same-modality reference identically one.

**Instrumentation:** Every local update packet carries an immutable `image_lineage` set. Before a backward pass, the client records every image modality tensor read by that forward graph. The packet's lineage is their union. A seed inherits its recorded lineage; aggregation, copying, and broadcasting take the union of every input lineage. The server records the input packet IDs and resulting lineage in the §5.6 ledger. The instrumentation is conservative: if a joint fusion update reads T1ce, the complete update is tagged with T1ce even if a learned gate later assigns it a low weight.

**Audit rule:** For every received encoder state, require \(\operatorname{ImageLineage}(\theta_{E_m})\subseteq\{m\}\). For every received track state, require \(\operatorname{ImageLineage}(\theta_{F_S})\cup\operatorname{ImageLineage}(\theta_{D_S})\subseteq S\). In CAMFS, the runtime rejects any packet or checkpoint that fails its applicable rule before deserialization. In B3, the identical instrumentation runs in shadow mode: it logs what would violate the M1 manifest but never alters B3's routing or optimization.

**Protocol:**

1. Run CAMFS M1 and B3 on the same nine partition/seed pairs.
2. Freeze the policy manifest and audit implementation hash before any training run.
3. For every round, retain the lineage log, accepted/rejected route decision, contributing site IDs, and model-state hash.
4. Report for each method and run: attempted forbidden routes, rejected forbidden routes, accepted violations, and the first violating route if any.
5. Treat CAMFS as a success only when accepted violations equal zero in all nine runs. For B3, report the observed result without assuming a violation; a zero-violation B3 result narrows the paper's empirical contrast but does not invalidate CAMFS's structural guarantee.

This is an execution-path audit under the honest-execution assumptions of §12. It is not a privacy attack, proof of legal compliance, or a claim that a parameter tensor is information-theoretically free of all T1ce-correlated label content.

### 14.5 Statistics

- The lineage audit is a deterministic pass/fail execution test, not a patient-level statistical endpoint. Report its result separately for all nine runs.
- Use the unique **patient** as the statistical unit; slices and repeated model runs are never independent samples.
- Use the three registered federation partitions and three training seeds per partition (9 total runs). Conditions are paired within the same partition, base, seed, and fixed H3 test patient.
- For patient \(p\), partition \(r\), and seed \(s\), compute the paired contrast \(d_{p,r,s}\). Average first over seeds and then partitions to obtain one \(d_p\) per unique test patient.
- Form the primary two-sided \(95\%\) confidence interval by bootstrapping patient identities and recomputing the mean. Use **10,000 PCG64 percentile-bootstrap resamples with seed 8803**.
- For pairwise comparisons (CAMFS vs each baseline), use **one-sided paired sign-flip randomization tests** with 100,000 sign vectors. Apply **Holm adjustment** across the baseline comparison family at \(\alpha=0.05\).
- Report per-hospital results; **never average away harm to one site**.
- Add a run-aware sensitivity interval with hierarchical resamples (PCG64 seed 8804): resample partition indices, training-seed indices within each partition, and patient identities.

### 14.6 Failure interpretation

| Result | Interpretation |
| :--- | :--- |
| CAMFS passes every audit + B3 accepts a forbidden route + competitive Dice | **Strongest outcome.** M1 provides an auditable structural guarantee where the unconstrained route demonstrably does not. |
| CAMFS passes every audit + moderate gap versus B2/B6 | **Still strong.** The Compliance-Performance Gap measures the observed cost of enforcing the declared policy. |
| B3 also has zero accepted violations | **Narrower empirical contrast.** Report the result; M1's contribution is the explicit policy artifact and proof-carrying routing, not a claim that every soft method leaks. |
| Poor Dice vs RELIEF (B4) | Hard isolation is sound, but M1's contrastive alignment adds no value over naive hard cohorts. Investigate \(\lambda_1\). |
| A1 shows 1-Phase ≈ 2-Phase | Two-phase freezing is unnecessary overhead. Simplify the architecture. |
| A7 shows no \(R_{\text{contribute}}\) benefit | Multi-track contribution doesn't help minority tracks. Simplify. |

---

## 15. Reproducible implementation defaults

### 15.1 Shared training defaults
- **Input:** 2D axial \(240\times240\)
- **Batch size:** 16 slices
- **Optimizer:** AdamW (\(\beta=(0.9,0.999)\), \(\epsilon=10^{-8}\))
- **Weight decay:** \(10^{-4}\)
- **Loss:** Soft Dice + Cross-Entropy
- **Dice smoothing:** \(\epsilon_D=10^{-5}\); tumour-class set \(\mathcal C_T=\{1,2,3\}\) after label remapping
- **Primary numeric precision:** FP32; mixed precision is an ablation
- **Other primary-model choices:** no class weighting, deep supervision, learning-rate schedule, gradient clipping, or postprocessing

\[
\mathcal L_{\mathrm{Dice+CE}}
=
-\frac{1}{N|\Omega|}\sum_{n=1}^{N}\sum_{q\in\Omega}\log p_{n,y_n(q)}(q)
+
1-\frac{1}{3N}\sum_{n=1}^{N}\sum_{c\in\mathcal C_T}
\frac{2\sum_{q\in\Omega}p_{n,c}(q)y_{n,c}(q)+\epsilon_D}
{\sum_{q\in\Omega}p_{n,c}(q)+\sum_{q\in\Omega}y_{n,c}(q)+\epsilon_D}.
\]

### 15.2 Phase 1 defaults
- **LR:** \(3\times10^{-4}\)
- **Temperature:** \(\tau=0.1\)
- **Weight:** \(\lambda_1=1\)
- **Bottleneck samples:** all \(15\times15\) locations of each sampled slice
- **Local work:** 1 local epoch per cohort round
- **Rounds:** Minimum 20, maximum 100. Stop on drift \(<0.01\) for 5 consecutive rounds after round 20.

### 15.3 Phase 2 defaults
- **Encoders:** frozen, evaluation mode, stop-gradient
- **LR:** \(10^{-3}\)
- **Alignment Weight:** \(\lambda_2=0.1\)
- **Local work:** 1 native or masked local epoch per track-cohort round, batch size 16
- **Selection:** highest contributor-patient-weighted validation macro Dice; improvement threshold \(10^{-4}\), earliest exact tie
- **Rounds:** Minimum 20, maximum 100. Stop when no validation improvement for 10 consecutive rounds after round 20.

### 15.4 Deterministic seed handling

Each registered training seed initializes:
- model parameters (Kaiming-normal);
- the axial-slice sampler (tumour/non-tumour sampling);
- augmentation draws (flip, rotation, intensity);
- data-loader worker generators.

Enable deterministic framework algorithms (`torch.use_deterministic_algorithms(True)`), seed each worker from the registered run seed, and record the framework, accelerator, and kernel versions. Map raw BraTS labels \(\{0,1,2,4\}\) to model indices \(\{0,1,2,3\}\). Learning-rate schedules are constant; gradient clipping is not used. These choices must be identical across matched conditions.

---

## 16. Communication profile

- **Phase 1:** Transmits \(\text{Encoder}_m\) and \(\text{Proto}_m^c\) only. Gated by modality ownership.
- **Phase 2:** Transmits \(\text{FusionHead}_{S'}\), \(\text{Decoder}_{S'}\), and \(\text{FusedProto}_{S'}^c\) only. Gated by subset membership.
- **Total volume:** Same order as single-phase training, but time-sliced.

---

## 17. Paper claims and limitations

### Claims
1. Formal problem formulation of consent-constrained FL.
2. The policy abstraction mechanism (MoU routing matrices).
3. Provable structural image/update-path purity guarantee for authorized M1 state.
4. The Compliance-Performance Gap metric.
5. Executable lineage-audit evidence showing which actual routes satisfy or violate the declared policy.

### Limitations
1. BraTS dataset represents simulated silos, not true institutional deployments.
2. ET label bias heavily relies on T1ce presence.
3. No convergence theory under asymmetric aggregation (deferred).
4. Honest execution is assumed.
5. The lineage theorem is an image/update-path guarantee, not a privacy, legal-compliance, or information-theoretic label-purity guarantee.

---

## 18. Forward compatibility with M2

M1 is a **standalone research study**. It does not depend on M2 and does not sacrifice any result for M2 compatibility. However, M2 will build on M1's methodology, and the following shared specifications act as the **compatibility contract**:

### 18.1 What is shared (the compatibility contract)

| Setting | Value | Why it matters for M2 |
| :--- | :--- | :--- |
| Architecture | Concat + \(1\times1\) fusion, 4 encoders, same UNet | M2 loads the same architecture for its frozen base |
| Aggregation | Pre-mix consent cohorts | M2 inherits this; does not invent its own |
| Dataset | BraTS 2021, 1,251 patients | Same patient pool |
| Partition seeds | \(\{1103, 2207, 3301\}\) | M2 uses the same federation partitions |
| Training seeds | \(\{17, 29, 43\}\) | M2 uses the same training seeds |
| Preprocessing | z-norm, clip \([-5,5]\), same augmentation | Identical input pipeline |
| Hyperparameters | All Phase 1 and Phase 2 defaults | M2's M1 retrain is methodologically identical |
| Loss formula | Exact Dice + CE formula (§15.1) | Bit-identical loss computation |
| Evaluation | 3D Dice, HD95, same empty-region rules | Comparable metrics |

### 18.2 What is NOT shared

- **H3 patient split:** M1 uses 175/25/50. M2 will use 125/25/25/25/50 (carving out calibration data).
- **Checkpoint:** M2 retrains M1 with its own H3 split. M2's E0 is methodologically equivalent to M1's result, but is a separate run.
- **M2-specific components:** Grants, CDRD, donor teacher, adapters, calibration gates — none of these exist in M1.

### 18.3 Narrative continuity

M1's limitation section states:

> *"M1 enforces \(\operatorname{Recv}(i)\subseteq O(i)\), which prevents Hospital 3 from benefiting from T1ce knowledge it does not own. Relaxing this constraint under bilateral consent — allowing a client to absorb knowledge from a non-owned modality while preserving the M1 base — is future work."*

This seeds M2's research question directly. A reviewer of M2 will see a clean progression:
1. M1 paper: consent-constrained isolation (this document)
2. M2 paper: consented cross-boundary absorption (follow-up)

The two papers are connected by shared methodology, not by shared checkpoints.

---

## 19. Completeness and go/no-go checklist

### 19.1 Architecture completeness

- [x] Unimodal Phase 1 with consent cohorts and empty-prototype masking
- [x] Hard freeze transition with stop-gradient proof
- [x] Phase 2 with fused-prototype bootstrap and unsupported-class masking
- [x] Phase 2 track-cohort closure rule
- [x] Phase 2 model selection with tie-breaking and early stopping
- [x] Send-gated track routing with corrected receive rule
- [x] Concat + \(1\times1\) canonical fusion (not cross-attention)
- [x] Multi-track \(R_{\text{contribute}}\) for H1→S3
- [x] Cold-start lineage rule for H4 delayed join
- [x] Track receive safety rule prevents cross-track leakage
- [x] Primary versioned policy manifest and append-only research provenance ledger
- [x] Image/update-path purity theorem with explicit scope and assumptions

### 19.2 Experimental completeness

- [x] 6 baselines (B1–B6) defined
- [x] 8 ablations (A1–A8) defined
- [x] Executable lineage audit with packet-level route records and reject-before-load rule
- [x] Statistics: patient-bootstrap CIs, sign-flip tests, Holm adjustment
- [x] Failure interpretation table

### 19.3 Pre-experiment gates

- [ ] Data loaded, preprocessed, and partitioned with registered seeds
- [ ] All models initialized with registered Kaiming-normal seeds
- [ ] Baselines (B1–B6) adapted to BraTS 2021 and the 4-hospital federation
- [ ] Phase 1 converges (prototype drift < 0.01 for 5 rounds)
- [ ] Phase 2 produces valid segmentation (positive Dice on validation)
- [ ] DisentAFL baseline faithfully reimplemented for the shadow lineage audit
- [ ] Baseline source/configuration manifests registered and hashed
- [ ] Lineage-audit code and policy manifest frozen and hashed before any result inspection
- [ ] All endpoints, tests, and ablation configurations preregistered

### 19.4 Paper success criteria

Minimum publishable result:

1. **Purity:** CAMFS accepts zero policy-violating encoder or track states in every one of the nine runs. The released ledger must reproduce that result from packet IDs, state hashes, and the frozen manifest.
2. **Utility:** On the preregistered H3 primary endpoint, CAMFS has a positive paired mean macro-Dice contrast against B1 and a one-sided paired sign-flip \(p<0.05\) after the registered Holm adjustment. Report the B2 and B6 gaps per hospital without claiming non-inferiority from an unregistered margin.
3. **Compliance:** All policy-routing, seed-lineage, and frozen-encoder tests pass in all nine runs.

Whether B3 accepts a forbidden route is a reported empirical comparison, not a precondition for M1's validity or publishability.

**Status:** Architecture and implementation specification complete. Code and experiments may begin once the pre-experiment gates in §19.3 are fulfilled.
