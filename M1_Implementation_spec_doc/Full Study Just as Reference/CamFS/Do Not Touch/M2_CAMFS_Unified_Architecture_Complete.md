# CAMFS M2: Consent-Governed Cross-Boundary Knowledge Absorption
## Complete Unified Research-Paper Architecture Specification

**Canonical research scope:** M1 + Gap 6 / R4 + Consent-Bounded Detached Residual Distillation (CDRD)

**Purpose:** this is the standalone M2 specification to implement and evaluate for a research paper. It inherits M1's two-phase consent-aware federation, then adds one bounded post-M1 knowledge-transfer phase. It does not claim to implement a production hospital governance platform.

**Status:** architecture complete; empirical claims remain hypotheses until the specified experiments are run.

---

## Table of Contents

1. Research objective and scope  
2. Gap 6, R4, and paper contributions  
3. Formal requirements  
4. Notation  
5. Unified component inventory  
6. Phase 1 — M1 unimodal training  
7. M1 freeze transition  
8. Phase 2 — M1 track-isolated fusion  
9. M1 routing, lineage, and canonical release  
10. Phase 3 — M2 CDRD  
11. M2-Lite calibrated prototype variant  
12. Paper policy and provenance  
13. Formal guarantees  
14. Four-hospital federation  
15. Experimental plan  
16. Reproducible implementation defaults  
17. Communication and compute  
18. Paper claims and limitations  
19. Deferred full-system scope  
20. Completeness and go/no-go checklist  

---

## 1. Research objective and scope

### 1.1 Research objective

M1 solves consent-aware federated learning by keeping modality encoders temporally isolated and fusion tracks structurally isolated. Its hard constraint is:

\[
\operatorname{Recv}(i)\subseteq O(i).
\]

A client can receive pooled benefit only for modalities it physically owns. Gap 6 asks whether that binary can be relaxed safely:

> Can a client explicitly opt into knowledge derived from one non-owned modality, while every other non-owned modality remains blocked and the original M1 federation remains uncontaminated?

M2 studies this question on the four-hospital BraTS federation. The principal case is:

\[
O(\mathrm{H3})=\{\mathrm{T1},\mathrm{FLAIR}\},
\qquad
U_{\mathrm{H3}}=\{\mathrm{T1ce}\}.
\]

Hospital 3 never receives a T1ce image and never requires T1ce at inference. It receives, under a simulated institution-level policy, a detached model artifact trained by an eligible donor that owns paired T1, FLAIR, and T1ce.

### 1.2 Relationship to M1

M2 does not repair, retrain, or weaken an active M1 federation. It uses M1 as a completed base:

\[
\text{M1 Phase 1}
\rightarrow
\text{M1 Phase 2}
\rightarrow
\text{canonical frozen M1 release}
\rightarrow
\text{M2 Phase 3}.
\]

M2 does not update a released M1 parameter or cross the M1/M2 namespace boundary. This unified specification preserves M1's phases, components, hard-routing intent, cold-start lineage, and purity boundary, while making M1's previously stated pairwise receive semantics executable through pre-mix consent cohorts. That cohort formalization corrects the earlier single-aggregate equations; it is an M1 implementation clarification required before producing the canonical base, not an M2 gradient path into a released base.

The M2 paper explicitly selects M1's documented **concatenation + \(1\times1\) fusion ablation** as its canonical fusion variant and retrains E0 with the exact five-tap architecture in Section 5. Earlier M1 checkpoints using the prior cross-attention default are historical references, not byte-compatible M2 bases. All claims in this document compare against the newly content-addressed concat-fusion E0; no result silently mixes the two variants.

Throughout this document, “M1-pure” means that excluded image tensors and excluded-modality model updates are absent from the M1 execution path. It does not mean BraTS annotations are information-theoretically independent of T1ce; label provenance is tracked separately.

### 1.3 Paper scope versus full-system scope

The research paper implements the left column and treats the right column as future deployment work.

| Implemented and evaluated in the paper | Deferred scalable hospital system |
| :--- | :--- |
| Static pairwise donor/recipient policy manifest | Institutional identity infrastructure and key management |
| Exact base-checkpoint and interface hash | Signed certificates, encrypted artifact transport, replay protection |
| One H1-to-H3 T1ce grant | Large-scale multi-donor and multi-recipient orchestration |
| Donor-local restricted teacher | Trusted build enclaves and remote attestation |
| CDRD adapter and recipient-local calibration | Dynamic grants, time windows, legal workflow integration |
| No-upload and no-seed enforcement in the experiment code | Production artifact registry and distributed audit service |
| M1 hash check and rollback-to-M1 test | Formal deletion acknowledgements across organizations |
| Static grant revocation simulation | Machine-unlearning service for patient-level withdrawal |
| Privacy-risk disclosure | Formal DP accounting, secure aggregation, and full attack red-teaming |

The paper must call its policies **simulated institution-level data-use policies**. It must not claim patient informed consent, HIPAA/GDPR compliance, or production security.

### 1.4 Claims and non-claims

M2 is designed to support these claims:

1. a bilateral policy can select one non-owned modality and one donor-recipient path;
2. a donor can distil \(S\cup U\) knowledge into an artifact that consumes only \(S\) at recipient inference;
3. the detached artifact cannot update or seed M1 under the stated execution assumptions;
4. CDRD can be evaluated causally against capacity-, donor-, and pairing-matched controls.

M2 does not guarantee:

- positive performance for every recipient;
- recovery or reconstruction of an unavailable MRI sequence;
- privacy merely because raw data are not transferred;
- cryptographic enforcement against malicious parties;
- clinical generalization from artificial BraTS silos.

---

## 2. Gap 6, R4, and paper contributions

### 2.1 Gap 6

Hard-isolation methods block all non-owned-modality influence. Missing-modality methods usually transfer or synthesize information according to an optimization rule, without allowing an institution to select a specific source modality and refuse another.

Gap 6 is:

> Within the reviewed consent-aware and missing-modality federated-learning literature, we found no evaluated mechanism in the M1 setting that combines an explicit bilateral policy, a bounded artifact shaped by one selected non-owned modality, an unchanged M1 fallback, and enforced non-propagation of that artifact into the federation.

This is a narrower and more defensible claim than saying no prior work has policy-aware FL. PoliFL already studies heterogeneous privacy policies. The paper's contribution is the combination of modality-scoped bilateral governance, detached missing-modality transfer, and M1 non-propagation.

### 2.2 Requirement R4

The original performance wording is interpreted as an empirical objective rather than a theorem:

> If donor and recipient both authorize a modality-bounded artifact, the architecture shall provide an auditable transfer route. The artifact is deployed only if it improves or is non-inferior to the frozen M1 base under the preregistered validation rule.

The existence of a compliant route is architectural. Positive utility is experimental.

### 2.3 Paper contributions

The paper should claim four contributions:

1. **Problem formulation:** consented cross-boundary knowledge absorption for complete missing-modality federations.
2. **Policy abstraction:** a static bilateral artifact manifest identifying donor, recipient, allowed modality lineage, purpose, and exact M1 base.
3. **Mechanism:** CDRD, a donor-trained dense residual adapter that operates only on recipient-owned inputs and remains detached from M1.
4. **Evaluation:** matched controls that distinguish non-owned-modality transfer from extra capacity, donor cohort knowledge, initialization, and paired-data effects.

CDRD itself is not presented as the invention of teacher-student distillation. The novelty is its placement and governance inside CAMFS.

### 2.4 Geometry correction inherited from the external review

M1 Phase 1 supplies class structure within each modality, but it does not mathematically identify one shared coordinate frame across modalities. For any modality-specific orthogonal transform \(Q_m\):

\[
z_m\mapsto Q_mz_m,
\qquad
\operatorname{Proto}_m^c\mapsto Q_m\operatorname{Proto}_m^c,
\]

the same-modality cosine objective is unchanged.

This does not prove that empirical alignment is impossible. Shared initialization or shared anatomy may encourage it. However:

- M1 aggregation is performed separately for each \(m\), not across T1 and T1ce encoders;
- common class labels do not remove the rotational symmetry;
- a learned fusion head can map different encoder spaces through learned projections;
- therefore successful fusion does not establish that raw T1ce prototypes are directly comparable to fused T1/FLAIR features.

The paper should use this wording:

> M1 leaves raw cross-modal prototype geometry unidentified and unvalidated. Prototype transfer is an empirical lightweight variant; CDRD is the primary method because it does not require direct comparison of uncoupled latent coordinates.

The large-scale MRI result reporting approximately \(0.0066\) cross-modal cosine distance used explicit co-localized cross-modality positive pairs; its ordinary contrastive baseline was far less aligned. It also found that alignment alone did not reliably improve lesion segmentation. See [large-scale modality-invariant brain MRI](https://arxiv.org/html/2511.11311). MFCPL likewise uses explicit projection and cross-modal alignment. See [MFCPL](https://arxiv.org/html/2401.13898).

---

## 3. Formal requirements

| Requirement | Paper-level definition |
| :--- | :--- |
| **R1 — Performance under policy** | Optimize segmentation only within the active static policy manifest. Report every recipient separately. |
| **R2 — Auditability** | Represent policy and artifact lineage as an inspectable configuration plus a released, content-hashed experiment log. |
| **R3 — Graceful degradation** | Preserve M1's track cold-start and lineage mechanisms; M2 failure always degrades to the released M1 base. |
| **R4 — Cross-boundary absorption** | Permit an authorized \(S\cup U\rightarrow S\) detached artifact, while unlisted modalities remain absent and the M1 state remains unchanged. |

R4 has three independent success tests:

1. **Policy success:** only the authorized donor, recipient, and modality are routed.
2. **Isolation success:** M1 hashes and baseline predictions are unchanged.
3. **Utility success:** CDRD exceeds the matched controls on held-out H3 patients.

An experiment can pass the first two and fail the third. That is a valid negative utility result, not an architectural policy failure.

---

## 4. Notation

| Symbol | Meaning |
| :--- | :--- |
| \(C=\{1,\ldots,N\}\) | Hospitals |
| \(\mathcal M=\{\mathrm{T1},\mathrm{T1ce},\mathrm{T2},\mathrm{FLAIR}\}\) | Modality universe |
| \(O(i)\subseteq\mathcal M\) | Modalities physically owned by hospital \(i\) |
| \(\operatorname{Send}(i)\subseteq O(i)\) | Modalities hospital \(i\) contributes in M1 |
| \(\operatorname{Recv}_{\mathrm{M1}}(i)\subseteq O(i)\) | Owned modalities for which \(i\) accepts M1 pooled benefit |
| \(S\) | M1 track subset and recipient inference subset |
| \(U\subseteq\mathcal M\setminus O(i)\) | Non-owned modalities listed in one M2 grant |
| \(E_m^*\) | Frozen post-Phase-1 encoder for modality \(m\) |
| \(\operatorname{Proto}_m^c\) | Same-modality class prototype |
| \(F_S,D_S\) | M1 fusion head and decoder for track \(S\) |
| \(B_S^b\) | Canonical released M1 base for \(S\), identified by hash \(b\) |
| \(H_S^b(x_S)\) | Detached multi-scale feature taps and logits from \(B_S^b\) |
| \(T_j^{S\cup U}\) | Donor \(j\)'s private restricted teacher |
| \(A_{\phi}^{S,U,b}\) | CDRD residual adapter bound to \(S,U,b\) |
| \(\Delta\ell\) | Dense residual logit map |
| \(\alpha\) | Recipient-local per-class calibration gate |
| \(d_g\in\{0,1\}\) | Recipient-local deployment switch for grant \(g\); zero bypasses M2 exactly |
| \(q_T\in\{0,1\}\) | Donor-teacher viability flag; zero forces the simulated deployment route to M1 |
| \(\rho_b\) | Digest of the standing M1 reuse-right records keyed by base hash \(b\) and purpose \(p\); stored separately from \(b\) |
| \(g=(j,i,S,U,\Gamma,a,p,b,\rho_b,v)\) | Paper artifact grant: donor \(j\), recipient \(i\), recipient inputs \(S\), permitted non-owned modalities \(U\), authorized upstream site IDs \(\Gamma\) across update/data/label/initialization provenance, artifact type \(a\), purpose \(p\), base/interface hash \(b\), rights digest \(\rho_b\), and versioned state \(v\in\{\mathrm{ACTIVE},\mathrm{REVOKED}\}\) |
| \(R_{\mathrm{out}}^{\mathrm{M2}}(g)\) | Donor authorizes the grant |
| \(R_{\mathrm{in}}^{\mathrm{M2}}(g)\) | Recipient accepts the grant |
| \(R_{\mathrm{reuse}}^{\mathrm{M1}}(k,b,p)\) | Contributor \(k\)'s standing permission to reuse released M1 base \(b\) for purpose \(p\) |
| \(\operatorname{sg}(\cdot)\) | Stop-gradient |
| \(\operatorname{ImageLineage}(\theta)\) | Image modalities directly present in forward/training paths shaping \(\theta\) |
| \(\operatorname{UpdateLineage}(\theta)\) | Sites and model updates inherited by \(\theta\) |
| \(\operatorname{LabelLineage}(\theta)\) | Annotation source and protocol shaping task losses |
| \(\operatorname{InitLineage}(\theta)\) | Checkpoints used to initialize \(\theta\) |

The paper grant deliberately omits production fields such as legal identity certificates, timestamps, encryption keys, and deletion receipts.

---

## 5. Unified component inventory

M2 has ten research components. The first six are inherited from M1.

| # | Component | Phase | Trainable? | Federated? |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Unimodal encoder bank | 1 | Yes in Phase 1 | Yes, modality-gated |
| 2 | Same-modality prototype bank | 1 | Updated in Phase 1 | Yes, modality-gated |
| 3 | M1 policy matrices | 1–2 | No | Exogenous |
| 4 | Subset fusion tracks | 2 | Yes in Phase 2 | Yes, track-gated |
| 5 | M1 phase controller | 1–2 | No | Server state |
| 6 | M1 provenance/lineage record | 1–2 | No | Audit metadata |
| 7 | Static M2 artifact grant | 3 | No | Exogenous |
| 8 | Canonical frozen-base registry | 3 | No | Checkpoint metadata |
| 9 | Donor-local restricted teacher | 3 | Yes at donor | Never |
| 10 | CDRD adapter and local gate | 3 | Adapter at donor; gate at recipient | Never |

### 5.1 Unimodal encoders

M1 uses one independent 2D U-Net downsampling encoder per MRI modality. This paper freezes the exact five-tensor interpretation of M1's four downsampling stages.

Define \(\operatorname{ConvBlock}_{C_{\mathrm{in}}\rightarrow C_{\mathrm{out}}}\) as two \(3\times3\), stride-1, padding-1 convolutions, each followed by 8-group GroupNorm and SiLU. Each downsampling operation is \(2\times2\) max-pooling.

| Tensor | Operation | Channels | Resolution |
| :--- | :--- | ---: | :--- |
| \(h_m^{(1)}\) | ConvBlock before pool 1 | 32 | \(240\times240\) |
| \(h_m^{(2)}\) | ConvBlock before pool 2 | 64 | \(120\times120\) |
| \(h_m^{(3)}\) | ConvBlock before pool 3 | 128 | \(60\times60\) |
| \(h_m^{(4)}\) | ConvBlock before pool 4 | 256 | \(30\times30\) |
| \(z_m\) | bottleneck ConvBlock after pool 4 | 256 | \(15\times15\) |

Thus \(h_m^{(4)}\) and \(z_m\) are distinct computations, not two names for one tensor. No external pretrained checkpoint is used in the primary paper: every freshly initialized convolution uses PyTorch Kaiming-normal with `mode="fan_out"`, `nonlinearity="relu"`, and zero bias under a registered seed. Any pretraining ablation must declare its dataset and checkpoint in image, label, and initialization lineage.

Encoders are trained only in Phase 1 and frozen before any segmentation task loss.

### 5.2 Same-modality prototypes

For downsampled class \(c\):

\[
\operatorname{Proto}_{m}^{c}
=
\frac{1}{|P_m^c|}
\sum_{p\in P_m^c} z_m(p).
\]

These prototypes organize each modality's features by class. M2 does not assume that prototypes from two different modalities share coordinates unless an explicit post-M1 calibration test establishes it.

### 5.3 M1 policies

M1 retains:

\[
R_{\mathrm{send}}(i,m),\quad
R_{\mathrm{recv}}(i,j,m),\quad
R_{\mathrm{send}}^{\mathrm{track}}(i,S),\quad
R_{\mathrm{recv}}^{\mathrm{track}}(i,j,S),\quad
R_{\mathrm{contribute}}(i,S).
\]

M2 does not reinterpret these matrices. Its grant is a separate purpose-specific object.

### 5.4 Subset tracks

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

Write \(z_S=f_S^{(5)}\). The decoder starts at \(z_S\), then repeats bilinear \(2\times\) upsampling with `align_corners=False`, concatenation with \(f_S^{(4)},f_S^{(3)},f_S^{(2)},f_S^{(1)}\), and a ConvBlock producing respectively 256, 128, 64, and 32 channels. A final \(1\times1\) convolution maps 32 channels to the four class logits. No postprocessing layer is part of the model.

\(\operatorname{FusedProto}_S^c\) is computed from \(z_S\). Track parameters are isolated and keyed by \(S\) and by the consent-cohort key defined in Section 6.2.

### 5.5 Extended phase controller

The research state machine is:

    PHASE_1_UNIMODAL
      -> PHASE_1_FROZEN
      -> PHASE_2_TRACK_TRAINING
      -> M1_RELEASED
      -> PHASE_3_M2_RESEARCH

Phase 3 cannot begin from the M1 STABLE flag alone. It requires a final released checkpoint hash.

### 5.6 Paper provenance record

For M1, the record preserves seed and track lineage. For M2, a simple experiment record adds:

    grant_id
    donor
    recipient
    target_subset
    authorized_nonowned_modalities
    base_hash
    base_release_rights_digest
    teacher_config_hash
    adapter_config_hash
    artifact_content_hash
    random_seed
    allow_decision
    validation_decision
    m1_hash_before
    m1_hash_after

Records are canonical JSON objects stored in an append-only JSONL file. Each record stores

\[
h_k=\operatorname{SHA256}(h_{k-1}\Vert\operatorname{CanonicalJSON}(r_k)).
\]

This content-hashed chain detects accidental or retrospective modification within the released experiment bundle. It is not independently timestamped, signed, or a production audit ledger.

---

## 6. Phase 1 — M1 unimodal training

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
\sum_{c'}
\exp(\operatorname{sim}(z_m(p),\operatorname{Proto}_m^{c'})/\tau)
}.
\]

\(P_m\) contains the bottleneck locations of the current sampled slices, \(c(p)\) is the nearest-neighbour-downsampled four-class label, and \(\operatorname{sim}\) is cosine similarity after L2-normalizing the embedding and prototype. Empty prototype entries are masked out of the denominator.

The client objective is:

\[
\mathcal L_i^{(1)}
=
\lambda_1
\sum_{m\in O(i)}
\mathcal L_{\mathrm{uni}}(z_m).
\]

No fusion head or decoder participates in this phase.

### 6.2 Aggregation

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

with self-receive defined as true.

Recipient \(i\) may load this cohort snapshot only if:

\[
m\in O(i)
\quad\land\quad
\forall j\in\mathcal A_{m,\kappa},\ R_{\mathrm{recv}}(i,j,m)=1.
\]

Clients contributing to a cohort train from that cohort's common incoming checkpoint. If two recipients accept different donor sets, the policy compiler creates two isolated cohort keys; it never attempts post-aggregation subtraction. For one cohort:

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

For class prototypes, let \(s_{j,m,\kappa}^{c}\) be the number of those patients containing at least one downsampled voxel of class \(c\), and let \(P_{j,m,\kappa}^{c}\) be the normalized mean embedding over their sampled class-\(c\) locations. Then:

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

A site with \(s_{j,m,\kappa}^{c}=0\) contributes nothing for that class. If the cohort denominator is zero, the previous prototype is retained; before the first supported update it is an explicitly masked, unusable entry.

Round 0 is a no-optimizer bootstrap: sites run the registered initialized encoder once, compute supported local class means, and initialize the cohort prototypes with the equation above. Contrastive encoder updates begin at round 1. A class with no cohort-wide bootstrap support remains masked until its first supported round.

For every round \(t\ge1\), the order is fixed:

1. broadcast \(\theta_{E_{m,\kappa}}^{t}\) and \(P_{m,\kappa}^{c,t}\) to the cohort;
2. each contributor detaches and holds the received prototypes fixed for the complete local epoch;
3. reset a fresh local AdamW optimizer and update encoder parameters only;
4. after the last optimizer step, switch the post-local encoder to evaluation mode and recompute local class means/support counts without gradient;
5. upload the post-local encoder, means, and counts; then form \(\theta^{t+1}\) and \(P^{t+1}\) with the equations above.

There is no server optimizer and no local Adam/AdamW moment state persists across rounds.

Prototype recomputation is deterministic and independent of the stochastic training loader. Each contributor processes every one of the 155 axial slices from every local training patient exactly once, ordered first by patient identifier and then by slice index, with the fixed preprocessing but no augmentation. Masks are resized to the \(15\times15\) bottleneck grid by nearest-neighbour interpolation; every grid location is used. The encoder or track is in evaluation mode, accumulation is FP32, and only the final local means plus patient-support counts are uploaded.

The primary paper uses one compatible cohort per modality: T1 contributors \(\{\mathrm{H1},\mathrm{H2},\mathrm{H3}\}\), T1ce contributors \(\{\mathrm{H1}\}\), T2 contributors \(\{\mathrm{H1},\mathrm{H2}\}\), and FLAIR contributors \(\{\mathrm{H1},\mathrm{H3}\}\). H4 is absent in Phase 1. Every recipient of these cohort snapshots accepts every listed contributor. Consequently, for any site \(q\notin\mathcal A_{m,\kappa}\), the cohort state is invariant to every update from \(q\).

A recipient that owns \(m\) but is not in \(\mathcal A_{m,\kappa}\), such as H2 for T1ce, may fit a private copy after download. That private state is never aggregated, never selected as the canonical cohort snapshot, and cannot initialize the H1-to-H3 teacher.

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

remains below \(\epsilon\) for \(K\) rounds. Initial paper defaults remain \(\epsilon=0.01\), \(K=5\), subject to validation.

---

## 7. M1 freeze transition

The server finalizes every authorized \(E_{m,\kappa}^*\) and prototype bank. In later equations, \(E_m^*\) is shorthand for the exact cohort snapshot named by the track/base manifest. Each client:

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

The Phase-1 optimizer and communication route are destroyed before Phase 2 starts.

---

## 8. Phase 2 — M1 track-isolated fusion

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

Only \(F_S,D_S,\operatorname{FusedProto}_S\) update.

Before the first task update of a newly created track, run the same no-optimizer prototype bootstrap as Phase 1 on \(z_S\). Unsupported classes remain masked from \(\mathcal L_{\mathrm{fused-align}}\) until observed.

For every later track round, broadcast the track state and fused prototypes; detach and hold the prototypes fixed; reset local AdamW; update only \(F_S,D_S\) for one epoch; then recompute post-local fused class means with the updated track in evaluation mode using the deterministic full-training-patient pass in Section 6.2 and aggregate them with the declared patient supports. No track optimizer moment persists across rounds.

### 8.3 Send-keyed aggregation

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

where \(n_{i,S,\kappa}\) is the number of distinct patients in the native or masked \(S\)-only pass. The same patient weighting applies to \(D_S\). Fused prototypes use the class-specific patient-support rule from Section 6.2, including its zero-support behavior.

The track cohort is admissible only if every member of \(\mathcal I(S,\kappa)\) accepts every other member's track update. A recipient may load the complete snapshot only if it accepts every contributor in that set. Otherwise a separate isolated cohort must be trained. No fused checkpoint is filtered after aggregation.

### 8.4 Multi-track contribution

A superset-owning donor may train a smaller pure track only when:

\[
R_{\mathrm{contribute}}(i,S)=1.
\]

It executes a masked pass using exactly \(x_S\). H1 uses this mechanism to help train the H3 pure track:

\[
S_3=\{\mathrm{T1},\mathrm{FLAIR}\}.
\]

Consequently, the primary \(S_3\) cohort has

\[
\mathcal I(S_3,\kappa_{13})=\{\mathrm{H1},\mathrm{H3}\}.
\]

The static paper policy sets \(R_{\mathrm{contribute}}(\mathrm{H1},S_3)=1\) and \(R_{\mathrm{send}}^{\mathrm{track}}(\mathrm{H1},S_3)=1\); both H1 and H3 mutually accept this two-member track cohort.

This matters in M2 because it makes the final canonical \(S_3\) checkpoint available to both H1 and H3 under the paper policy, avoiding a new recipient-private checkpoint handoff.

---

## 9. M1 routing, lineage, and canonical release

### 9.1 Track lineage

A valid M1 seed for track \(S\) must satisfy:

\[
\operatorname{ImageLineage}(\theta_{\mathrm{seed}})\subseteq S.
\]

M1 cold-start may grow from a strict subset, stitch compatible slots, warm-start within \(S\), or use fresh initialization. A superset checkpoint containing an excluded modality is invalid.

The primary paper removes tier ambiguity: \(S_1,S_2,S_3\) fusion/decoder tracks start from registered Kaiming initialization. Delayed H4 joins only after \(S_2\) has met its stopping rule and its selected checkpoint has been frozen. At that point instantiate \(S_4=\{\mathrm{T1},\mathrm{T1ce},\mathrm{T2}\}\) from the finalized \(S_2=\{\mathrm{T1},\mathrm{T2}\}\) track. At all five fusion levels, widen the first \(1\times1\) input from `[T1,T2]` to `[T1,T1ce,T2]`, copy the T1/T2 column blocks to their named slots, zero-initialize the new T1ce block, and copy the remaining fusion ConvBlocks and decoder. Train \(S_4\) only with H4 updates for at least 20 and at most 100 rounds, selecting on H4 validation macro Dice and stopping after 10 rounds without improvement. This delayed track is an M1 cold-start context result only: it never changes \(S_3\), the M2 base hash, the grant, the teacher, or an M2 artifact. This is the only cold-start rule used in the primary results; other tiers are ablations.

### 9.2 Canonical release for the paper

After Phase 2 model selection, the server releases:

\[
B_{S_3}^{b}
=
\left(
E_{\mathrm{T1}}^*,
E_{\mathrm{FLAIR}}^*,
F_{S_3}^{b},
D_{S_3}^{b}
\right).
\]

For the paper cohorts:

\[
\operatorname{ImageLineage}(B_{S_3}^{b})
=
\{\mathrm{T1},\mathrm{FLAIR}\},
\qquad
\operatorname{UpdateLineage}(B_{S_3}^{b})
=
\{\mathrm{H1},\mathrm{H2},\mathrm{H3}\}.
\]

The hash \(b\) covers:

- all parameters and buffers;
- architecture configuration;
- feature-tap names and shapes;
- encoder and track consent-cohort identifiers;
- preprocessing configuration;
- class order;
- software version relevant to serialization.

It deliberately excludes policy rights. After computing \(b\), canonicalize the sorted records \(\{R_{\mathrm{reuse}}^{\mathrm{M1}}(k,b,p)\}_k\) and hash them separately as \(\rho_b\). The released base record binds the pair \((b,\rho_b)\). This order avoids a self-referential hash: rights refer to the already computed model hash, while changing a right changes \(\rho_b\) without changing the model bytes.

The same binary is loaded by H1 for donor adapter training and by H3 for inference. If the binaries differ, Phase 3 is invalid.

### 9.3 Read-only feature interface

The base exposes:

\[
H_{S_3}^{b}(x_{S_3})
=
\operatorname{sg}
\left(
f_{S_3}^{(1)},
f_{S_3}^{(2)},
f_{S_3}^{(3)},
f_{S_3}^{(4)},
z_{S_3},
\ell_B
\right).
\]

These are taps from an unchanged M1 forward pass. The interface changes what is returned to the local process, not what M1 computes or trains.

---

## 10. Phase 3 — M2 CDRD

### 10.1 Static bilateral grant

The paper grant is:

\[
g=(j,i,S,U,\Gamma,a,p,b,\rho_b,v).
\]

Authorization has a pre-build gate and a post-build lineage gate. Before any teacher or adapter build:

\[
\operatorname{AllowBuild}(g)
=
\operatorname{Registered}(g)
\land
\operatorname{Active}(v)
\land
\operatorname{SafeRights}(g)
\land
\operatorname{BaseReusable}(B_S^b,p,\rho_b)
\land
R_{\mathrm{out}}^{\mathrm{M2}}(g)
\land
R_{\mathrm{in}}^{\mathrm{M2}}(g)
\land
S\subseteq O(i)
\land
S\cup U\subseteq O(j)
\land
U\cap O(i)=\varnothing
\land
\left[
\operatorname{UpdateLineage}(B_S^b)
\cup
\operatorname{UpdateLineage}(E_U^*)
\right]
\subseteq
\Gamma
\land
\operatorname{DeclaredNewCrossSites}(g)=\{j\}
\land
\operatorname{HashMatch}(b).
\]

After artifact \(A\) has been built but before export or load:

\[
\begin{aligned}
\operatorname{AllowRelease}(A,g)
={}&
\operatorname{AllowBuild}(g)\\
&\land
\operatorname{ActualSourceSites}(A)\subseteq\Gamma\\
&\land
\operatorname{ActualNewCrossSites}(A\mid B)
\subseteq
\operatorname{DeclaredNewCrossSites}(g)\\
&\land
\operatorname{ImageLineage}(A)\subseteq S\cup U\\
&\land
\operatorname{ArtifactBoundTo}(A,b,\rho_b,a,p).
\end{aligned}
\]

\(\operatorname{Registered}(g)\) means every immutable field exactly matches a preregistered static grant; there is no wildcard modality or artifact type. The primary CDRD-Frozen grant uses \(a=\texttt{CDRD_FROZEN}\); CDRD-P uses a second otherwise parallel grant with \(a=\texttt{CDRD_P}\); and the forced-on raw-prototype diagnostic uses a third with \(a=\texttt{RAW_PROTO_DIAGNOSTIC}\). \(\operatorname{SafeRights}(g)\) requires `allow_upload=false`, `allow_m1_seed=false`, and `allow_onward_transfer=false`. Therefore an artifact that declares T2 does not match the registered \(U=\{\mathrm{T1ce}\}\) grant and is denied even if H1 owns T2.

\[
\operatorname{BaseReusable}(B_S^b,p,\rho_b)
\iff
\left[
\bigwedge_{k\in\operatorname{UpdateLineage}(B_S^b)}
R_{\mathrm{reuse}}^{\mathrm{M1}}(k,b,p)=1
\right]
\land
\rho_b=
\operatorname{SHA256}
\left(
\operatorname{CanonicalRights}(b,p)
\right).
\]

All `*Sites` functions are sets of hospital identifiers, never tensors or checkpoints. \(\operatorname{ActualSourceSites}(A)\) is the union of site IDs recorded across A's update, image-data, label, and initialization provenance. \(\operatorname{ActualNewCrossSites}(A\mid B)\) contains sites providing any new post-base image, label, teacher, encoder, or adapter-training role; a site remains listed even if it also contributed to the frozen base. In the primary experiment it is \(\{\mathrm{H1}\}\).

\(\operatorname{AllowRelease}\) is a provenance/scope decision for the research artifact, not a utility decision. \(q_T\) and H3 acceptance control only the simulated deployment route; they never retroactively exclude an artifact from the registered forced-on scientific analysis.

Paper instance:

    grant_id: H1_to_H3_T1ce_CDRD
    donor: H1
    recipient: H3
    target_subset: [T1, FLAIR]
    authorized_nonowned_modalities: [T1ce]
    artifact_type: CDRD_FROZEN
    purpose: BraTS_segmentation_research
    base_hash: <canonical-S3-hash>
    base_release_rights_digest: <standing-M1-rights-hash>
    authorized_upstream_sites: [H1, H2, H3]
    base_update_contributors: [H1, H2, H3]
    base_track_contributors: [H1, H3]
    cross_encoder_update_contributors: [H1]
    declared_new_cross_sites: [H1]
    grant_version: 1
    grant_state: ACTIVE
    donor_allows: true
    recipient_accepts: true
    allow_upload: false
    allow_m1_seed: false
    allow_onward_transfer: false

T2 is absent. H2 and H4 also lack the paired set \(S_3\cup\{\mathrm{T1ce}\}\) because neither owns FLAIR. H2 additionally withholds the relevant T1ce use.

### 10.2 Donor restricted teacher

For H1:

\[
T_{\omega}^{S_3\cup\{\mathrm{T1ce}\}}
\left(x_{\mathrm{T1}},x_{\mathrm{FLAIR}},x_{\mathrm{T1ce}}\right)
\rightarrow \ell_T.
\]

The teacher uses:

- frozen \(E_{\mathrm{T1}}^*,E_{\mathrm{FLAIR}}^*,E_{\mathrm{T1ce}}^*\);
- a private fusion head for \(\{\mathrm{T1},\mathrm{FLAIR},\mathrm{T1ce}\}\);
- a private decoder.

For the primary federation, the \(S_3\) fusion head and decoder have H1/H3 update provenance. Its frozen T1 encoder has H1/H2/H3 provenance, its FLAIR encoder has H1/H3 provenance, and the final T1ce encoder has H1 provenance because H2 withholds T1ce updates and H4 was absent from Phase 1. Hence \(\Gamma=\{\mathrm{H1},\mathrm{H2},\mathrm{H3}\}\).

The new cross-boundary release is still bilateral H1\(\rightarrow\)H3: H2 supplies no T1ce-derived artifact and grants no new M2 right. The paper explicitly assumes that every contributor to the already released M1 base authorized frozen downstream research use of that base when M1 was released. The manifest records those inherited contributors, while H1 and H3 separately authorize the new T1ce-shaped sidecar. If that standing M1 release right is absent in an implementation, the build is invalid until H2 authorizes derivative use or an H2-free consent-cohort base is retrained.

Initialization uses an exact M1 subset-growth mapping. At each of the five fusion levels, widen the first \(1\times1\) fusion convolution from \(2C_r\) inputs ordered `[T1, FLAIR]` to \(3C_r\) inputs ordered `[T1, T1ce, FLAIR]`. Copy the old T1 column block to the new T1 block, copy the old FLAIR block to the new FLAIR block, initialize the new T1ce column block to zero, and copy the output bias unchanged. Copy every following fusion ConvBlock and every decoder tensor byte-for-byte from \(B_{S_3}^{b}\). Then train all teacher fusion and decoder parameters locally with all three encoders frozen.

Because the new column block is zero, the widened teacher produces the canonical base logits before teacher training, up to deterministic numerical tolerance. This initialization-equality test is mandatory.

This is valid because:

\[
\operatorname{ImageLineage}(B_{S_3}^b)
\subseteq
S_3
\subset
S_3\cup\{\mathrm{T1ce}\}.
\]

The teacher must not use H1's full T1/T1ce/T2/FLAIR track. That would add unauthorized T2 lineage.

Teacher loss:

\[
\mathcal L_T
=
\mathcal L_{\mathrm{Dice+CE}}
\left(y,\operatorname{softmax}(\ell_T)\right).
\]

After locking the teacher checkpoint, hyperparameters, and \(T_T^{\mathrm{cal}}\), compare the teacher and frozen \(S_3\) base once on the separate 100-patient H1 release-audit set. The simulated release is viable when

\[
\operatorname{LCB}_{95\%}
\left[
\operatorname{MacroDice}_{H1}(T)-
\operatorname{MacroDice}_{H1}(B)
\right]>0
\]

and no region's teacher point estimate is more than \(\delta_T=0.01\) Dice below the base. Set \(q_T=1\) only when this gate passes. This is a deployment flag, not an experimental exclusion rule. Failed seeds have \(q_T=0\) and are marked `FALLBACK_ONLY`; their adapter is still trained, transferred under the research-only grant, and evaluated as forced-on \(E4_{\mathrm{aug}}\), but their simulated deployment output is unconditionally M1 with \(d_g=0\). Thus teacher gating cannot silently remove unfavorable runs or enable a failed artifact.

Compute that lower bound from 10,000 patient-identity percentile-bootstrap resamples of the 100 H1 release-audit patients using PCG64 seed 7711. The same 3D reconstruction and empty-region conventions as H3 evaluation apply.

### 10.3 Exact CDRD adapter

The base provides five distinct feature tensors: four pre-pooling fused skip resolutions plus the post-pooling bottleneck. Define:

\[
(f_S^{(1)},f_S^{(2)},f_S^{(3)},f_S^{(4)},f_S^{(5)})
=
(f_S^{(1)},f_S^{(2)},f_S^{(3)},f_S^{(4)},z_S),
\]

with channel widths \((32,64,128,256,256)\) and spatial sizes \((240,120,60,30,15)\) under the documented M1 implementation. For each:

\[
p_r
=
\operatorname{SiLU}
\left(
\operatorname{GN}
\left(
\operatorname{Conv}_{1\times1}^{C_r\rightarrow16}
(f_S^{(r)})
\right)
\right).
\]

The projected features are bilinearly upsampled to \(240\times240\) with `align_corners=False` and concatenated with the four-channel base logit map:

\[
q=
\operatorname{Concat}
\left(
\operatorname{Up}(p_1),\ldots,\operatorname{Up}(p_5),\ell_B
\right)
\in\mathbb R^{84\times240\times240}.
\]

The residual head is:

1. depthwise-separable \(3\times3\), \(84\rightarrow32\), GroupNorm, SiLU;
2. depthwise-separable \(3\times3\), \(32\rightarrow16\), GroupNorm, SiLU;
3. \(1\times1\), \(16\rightarrow4\).

Every convolution followed by GroupNorm is bias-free. Each depthwise-separable block is a `bias=False`, padding-1 depthwise \(3\times3\) convolution with `groups=input_channels`, followed by a `bias=False` pointwise \(1\times1\) convolution, 8-group GroupNorm with `affine=True`, and SiLU. All five lateral GroupNorm layers also use 8 groups and `affine=True`. The final \(1\times1\) convolution has a four-value bias; its kernel and bias are initialized to zero. Remove the softmax-invariant common class offset from its raw output \(r_\phi\):

\[
\Delta\ell
=
r_\phi-\frac{1}{C_{\mathrm{cls}}}\sum_c r_{\phi,c},
\qquad
\ell_{\mathrm{cand}}=\ell_B+\Delta\ell.
\]

Therefore the initial augmented predictor equals M1 and every predicted residual is class-zero-mean.

With the documented channel widths, the adapter has exactly \(16{,}344\) trainable parameters, excluding the four H3 gates, and a payload of roughly \(65.4\) KB in FP32 or \(32.7\) KB in FP16.

GroupNorm is used instead of BatchNorm so the sidecar has no shared running-statistic mutation.

### 10.4 Adapter objective

Independently trained teacher and base logits can have different additive offsets and confidence scales. The deployed correction is added to the **raw canonical base logits**, so the target must use that same base coordinate. Fit only a scalar teacher temperature \(T_T^{\mathrm{cal}}\) on H1 validation patients and define:

\[
\ell_T^{\mathrm{cal}}
=
\ell_T/T_T^{\mathrm{cal}}.
\]

The exact identifiable residual target is:

\[
\Delta\ell^\star
=
\operatorname{sg}
\left(
\operatorname{center}
\left[
\ell_T^{\mathrm{cal}}-\ell_B
\right]
\right).
\]

Here \(\operatorname{center}(v)=v-\frac{1}{C_{\mathrm{cls}}}\sum_c v_c\).

This coordinate is identifiable and serving-consistent. If \(\Delta\ell=\Delta\ell^\star\), then \(\ell_B+\Delta\ell\) and \(\ell_T^{\mathrm{cal}}\) differ only by a common per-pixel class offset and therefore have identical softmax probabilities. Calibrating the base inside the target while serving raw \(\ell_B\) is forbidden because it would break that equality and the exact M1 fallback.

The donor trains:

\[
\begin{aligned}
\mathcal L_A
&=
\lambda_{\mathrm{seg}}
\mathcal L_{\mathrm{Dice+CE}}
\left(y,\operatorname{softmax}(\ell_{\mathrm{cand}})\right)\\
&+
\lambda_{\mathrm{KD}}T^2
\operatorname{KL}
\left[
\operatorname{softmax}(\ell_T^{\mathrm{cal}}/T)
\;\Vert\;
\operatorname{softmax}(\ell_{\mathrm{cand}}/T)
\right]\\
&+
\lambda_{\Delta}
\operatorname{SmoothL1}
\left(\Delta\ell,\Delta\ell^\star\right)\\
&+
\lambda_{\mathrm{TV}}
\operatorname{TV}(\Delta\ell).
\end{aligned}
\]

The operators above are fixed, not implementation-dependent. Let \(\Omega\) contain all spatial positions in a batch, let \(C_{\mathrm{cls}}=4\), let \(P=\operatorname{softmax}(\ell_T^{\mathrm{cal}}/T)\), and let \(Q=\operatorname{softmax}(\ell_{\mathrm{cand}}/T)\). Then

\[
\operatorname{KL}(P\Vert Q)
=
\frac{1}{|\Omega|}
\sum_{q\in\Omega}\sum_{c=1}^{4}
P_c(q)\left[\log P_c(q)-\log Q_c(q)\right],
\]

with the class sum inside the spatial mean and \(0\log 0=0\). It is computed from FP32 `log_softmax` values. For \(r=\Delta\ell-\Delta\ell^\star\), SmoothL1 uses \(\beta=1\):

\[
\operatorname{SmoothL1}(\Delta\ell,\Delta\ell^\star)
=
\frac{1}{4|\Omega|}\sum_{q,c}
\begin{cases}
\tfrac12 r_c(q)^2,& |r_c(q)|<1,\\
|r_c(q)|-\tfrac12,& \text{otherwise}.
\end{cases}
\]

For a tensor with shape \(N\times4\times H\times W\), total variation is the sum of the separately normalized horizontal and vertical forward differences:

\[
\operatorname{TV}(\Delta\ell)
=
\frac{\sum|\Delta\ell_{:,:,:,1:}-\Delta\ell_{:,:,:,:-1}|}{4NH(W-1)}
+
\frac{\sum|\Delta\ell_{:,:,1:,:}-\Delta\ell_{:,:,:-1,:}|}{4N(H-1)W}.
\]

Registered primary values:

\[
\lambda_{\mathrm{seg}}=1,\quad
\lambda_{\mathrm{KD}}=1,\quad
\lambda_{\Delta}=0.1,\quad
\lambda_{\mathrm{TV}}=10^{-5},\quad
T=2.
\]

These values are fixed across the matched E3/E4/E5 confirmatory runs. Section 16 defines a separately labelled exploratory sweep that cannot change the primary checkpoints.

### 10.5 Recipient calibration

Primary paper variant, CDRD-Frozen:

- freeze \(B_{S_3}^b\);
- freeze donor adapter \(\phi\);
- fit only four per-class gate logits on the dedicated H3 M2-calibration split.

\[
\alpha_c=\sigma(a_c),
\qquad
\ell_{\mathrm{aug},c}
=
\ell_{B,c}
+
d_g\alpha_c\Delta\ell_c.
\]

Initialize \(a_c=-4\), so \(\alpha_c\approx0.018\). During gate fitting set \(d_g=1\), train for the fixed epoch budget, and keep the last checkpoint; the M2 acceptance-validation split is not used for gate fitting or early stopping. After the gate is locked, H3 evaluates the complete augmented-versus-fallback acceptance rule once on that untouched split.

Use suffixes consistently:

- \(E\!k_{\mathrm{aug}}\) is the registered forced-on scientific output with its learned frozen gate and \(d_g=1\), regardless of deployment qualification;
- \(E\!k_{\mathrm{deploy}}\) is the simulated routed output, with \(d_g=1\) only after every donor and H3 acceptance gate passes and \(d_g=0\) otherwise.

Primary causal tests use the `_aug` estimand so validation fallback cannot hide a harmful or failed artifact. Operational results report `_deploy` separately. For E4, \(q_T=0\) forces \(E4_{\mathrm{deploy}}=E0\).

Secondary ablation, CDRD-FT:

- locally fine-tune \(\phi\) for 50 epochs with \(d_g=1\), \(\alpha_c=1\), and Dice + cross-entropy, then freeze it;
- reinitialize \(a_c=-4\) and fit only the four gates for the same 50-epoch schedule used by CDRD-Frozen;
- use this same two-stage local schedule for the E1 local-capacity control;
- delete the entire sidecar on simulated revocation.

CDRD-Frozen is the primary causal transfer result because it cannot turn the donor artifact into a large H3-trained local model.

### 10.6 Inference

H3 inference uses only:

\[
x_{\mathrm{T1}},x_{\mathrm{FLAIR}}
\rightarrow
B_{S_3}^b
\rightarrow
A_{\phi}^{S_3,\mathrm{T1ce},b}
\rightarrow
\ell_{\mathrm{aug}}.
\]

No T1ce image, donor logit, donor feature, or teacher is present at H3.

If the grant is disabled, the artifact fails validation, or rollback is requested:

\[
d_g=0,
\qquad
\ell_{\mathrm{aug}}=\ell_B.
\]

### 10.7 Phase-3 pseudocode

    assert M1.state == RELEASED
    verify canonical base hash b and rights digest rho_b at H1 and H3
    load static grant g

    if not AllowBuild(g):
        record BUILD_DENIED
        stop

    H1:
        train restricted teacher T_(S3+T1ce) with frozen encoders
        lock teacher, hyperparameters, and temperature
        evaluate teacher viability once against B_S3 on H1 release-audit patients
        if viability fails:
            q_T = 0
            record FALLBACK_ONLY but continue the registered offline experiment
        else:
            q_T = 1

        for paired donor batches (x_S3, x_T1ce, y):
            H_B, logits_B = stop_gradient(B_S3(x_S3))
            logits_T = stop_gradient(T_(S3+T1ce)(x_S3, x_T1ce))
            delta = A_phi(H_B, logits_B)
            update phi with L_A

        package A_phi plus research manifest locally
        recompute actual artifact lineage
        compute artifact content hash over immutable manifest fields and tensor bytes
        if not AllowRelease(A_phi, g):
            record RELEASE_DENIED
            stop
        export only A_phi plus research manifest

    H3:
        recompute AllowRelease(A_phi, g) from manifest and local base
        recompute and match the artifact content hash before deserialization
        load A_phi into M2-only parameter namespace
        train four per-class gates for the fixed schedule on H3 M2-calibration split
        lock the final gate checkpoint
        acceptance_pass = evaluate once on untouched H3 M2 acceptance-validation split
        d_deploy = 1 only if q_T == 1 and acceptance_pass else 0
        evaluate E4_aug on H3 test with d_g = 1 for every registered run
        evaluate E4_deploy on the same test with d_g = d_deploy

    verify:
        M1 hash before == M1 hash after
        M2 namespace absent from upload serialization
        revoked output == pure M1 output

---

## 11. M2-Lite calibrated prototype variant

The prototype route is retained as a lightweight comparison, not dismissed as inherently impossible.

### 11.1 Raw prototype diagnostic

E6a reproduces the original raw CrossAbsorb hypothesis:

\[
z_{S_3}
\leftrightarrow
\operatorname{Proto}_{\mathrm{T1ce}}^c.
\]

The four H1 T1ce prototype tensors cross to H3 only under a separately preregistered `RAW_PROTO_DIAGNOSTIC` grant with the same donor, recipient, \(S\), \(U\), \(\Gamma\), purpose, base hash, rights digest, content-hash rule, and no-upload/no-seed/no-onward flags as CDRD-Frozen. The manifest binds their class order, shape, dtype, encoder/prototype-bank hash, and H1 T1ce lineage. This grant is research-only: it authorizes the forced-on E6a comparison but never qualifies a deployment route, so its simulated deployed output remains E0.

For reproducibility, use the original bottleneck-only default. Flatten the frozen \(256\times15\times15\) fused bottleneck into 225 spatial queries; use the four raw \(256\)-dimensional H1 T1ce class prototypes as keys and values; apply one-head scaled dot-product attention with scale \(1/\sqrt{256}\) and learned \(W_Q,W_K,W_V,W_O\in\mathbb R^{256\times256}\); and add the output residually to \(z_{S_3}\). Copy the canonical \(S_3\) decoder, feed it the enhanced bottleneck plus the same four detached canonical skip tensors, and train the attention head and copied decoder on the 25 H3 M2-calibration patients with Dice + cross-entropy, AdamW at \(10^{-3}\), weight decay \(10^{-4}\), for exactly 50 epochs; keep the final checkpoint without consulting acceptance validation. They remain outside M1 and are deleted on revocation.

E6a is explicitly labelled empirically contingent and capacity-confounded because M1 does not establish the shared coordinates and the local decoder adds substantially more trainable state than CDRD-Frozen. Report norm-matched random-prototype and class-permuted-prototype controls. It is a historical diagnostic, not the serious lightweight comparison.

### 11.2 Calibrated prototype variant

E6b, called CDRD-P, learns an explicit paired projection at H1:

\[
q_S(p)=
\operatorname{norm}(W_Sz_S(p)),
\qquad
q_U(p)=
\operatorname{norm}(W_Uz_U(p)),
\]

where bias-free \(W_S,W_U:\mathbb R^{256}\rightarrow\mathbb R^{16}\). Train the projections with symmetric paired InfoNCE at temperature 0.1 on co-registered H1 locations. Same-patient, same-location features are positives; other patient/locations are negatives. Select on H1 validation correspondence, then freeze both projections. A shuffled-pair control must fail this correspondence test.

Construct four projected T1ce prototypes:

\[
\widetilde P_U^c
=
\operatorname{norm}
\left(
\frac{1}{|P_U^c|}
\sum_{p:y(p)=c}q_U(p)
\right).
\]

Require nonzero H1 training support for every class; otherwise mark E6b unavailable rather than inventing a prototype.

At H3, only \(W_S\) and the four projected prototypes are required:

\[
a_c(p)
=
\operatorname{softmax}_c
\left(
q_S(p)^\top\widetilde P_U^c/\tau_p
\right),
\qquad
\tau_p=0.1.
\]

Bilinearly upsample the four attention maps to \(240\times240\) with `align_corners=False`, concatenate them with the four M1 baseline logits, and apply:

1. bias-free \(3\times3\) convolution, \(8\rightarrow16\), 8-group GroupNorm, SiLU;
2. `bias=True` \(1\times1\) convolution, \(16\rightarrow4\), with both its kernel and four-value bias initialized to zero.

For raw head output \(r_P\), use the same serving coordinate as CDRD:

\[
\Delta\ell_P=\operatorname{center}(r_P),
\qquad
\ell_{\mathrm{cand},P}=\ell_B+\Delta\ell_P.
\]

With the projections and prototypes frozen, train only this residual head at H1 using the same segmentation, calibrated-teacher KD, centered residual-target, and TV objective as CDRD. Release \(W_S\), four projected prototypes, and the residual head; \(W_U\) remains at H1. The released coefficient tensors contain \(5{,}348\) values plus 64 prototype values, or approximately \(21.7\) KB FP32. CDRD-P uses a separately preregistered grant whose donor, recipient, \(S\), \(U\), \(\Gamma\), purpose, base hash, rights digest, and safety flags are identical to the primary grant, but whose artifact type is exactly `CDRD_P`. The recipient-calibration procedure, no-upload rule, and fallback are unchanged.

### 11.3 Positioning

The paper presents:

- **M2-Lite:** calibrated prototype conditioning, low communication and limited spatial bandwidth;
- **M2-Full:** CDRD, higher communication and a dense-output-capable residual adapter; donor activations and residual maps are never exported.

Results determine the accuracy/complexity trade-off. The paper does not assume M2-Lite succeeds.

---

## 12. Paper policy and provenance

### 12.1 Policy cases

At minimum, run:

| Case | Donor allows | Recipient accepts | Modality listed | Expected result |
| :--- | :---: | :---: | :---: | :--- |
| P0 | 0 | 1 | T1ce | Denied |
| P1 | 1 | 0 | T1ce | Denied |
| P2 | 1 | 1 | T1ce | H1-to-H3 artifact allowed |
| P3 | 1 | 1 | T2 | Denied for H3 experiment |
| P4 | 1 | 1 | T1ce, wrong base hash | Denied |
| P5 | 1 | 1 | T1ce, artifact disabled | Pure M1 fallback |
| P6 | 1 | 1 | Registered T1ce grant, but actual artifact lineage also contains T2 | Build may complete locally; release and load denied |

### 12.2 Policy soundness in the paper

Under honest execution:

\[
\neg\operatorname{AllowBuild}(g)
\Rightarrow
\text{no teacher or adapter build},
\]

and:

\[
\neg\operatorname{AllowRelease}(A,g)
\Rightarrow
\text{no artifact export and no artifact load}.
\]

This is a program/configuration invariant, not a cryptographic theorem.

### 12.3 Minimal artifact manifest

    artifact_id
    artifact_content_hash
    artifact_type
    donor
    recipient
    target_subset
    authorized_nonowned_modalities
    purpose
    grant_version
    grant_state
    label_schema
    base_hash
    base_update_contributors
    base_track_contributors
    base_release_rights_digest
    cross_encoder_update_contributors
    actual_source_sites
    declared_new_cross_sites
    actual_new_cross_sites
    image_lineage
    label_lineage
    initialization_lineage
    adapter_architecture_hash
    training_seed
    donor_validation_metrics
    no_upload
    no_seed
    no_onward_transfer

No patient identifiers, per-patient logits, or per-patient features are exported.

The content hash is SHA-256 over the canonical ordered tensor names, shapes, dtypes, raw tensor bytes, and every immutable manifest field except `artifact_content_hash` itself. H1 computes it before release and H3 recomputes it before loading. This detects accidental substitution or corruption in the released experiment bundle; without signatures it does not authenticate a malicious sender.

---

## 13. Formal guarantees

### 13.1 M1 shared-state non-propagation

Let \(\theta_B\) contain every M1 parameter and mutable buffer. Let \(\phi,\alpha,d_g\) contain all M2 recipient-side state.

\[
H=\operatorname{sg}(B_{\theta_B}(x_S)),
\qquad
\mathcal L_{\mathrm{M2}}
=
\mathcal L(\phi,\alpha,d_g;H).
\]

If:

1. \(B_{\theta_B}\) is evaluation-only;
2. all base outputs are detached;
3. M1 and M2 optimizers and serialization namespaces are disjoint;
4. no M2 object enters aggregation, seeding, routing, or M1 checkpoint selection;

then:

\[
\frac{\partial\mathcal L_{\mathrm{M2}}}{\partial\theta_B}=0,
\qquad
\theta_B^{\mathrm{after}}=\theta_B^{\mathrm{before}}.
\]

The implementation verifies content hashes and fixed-probe predictions before and after M2.

### 13.2 Lineage

M2 separates input/update purity from annotation provenance:

\[
\operatorname{ImageLineage}(B_{S_3}^{b})
=
\{\mathrm{T1},\mathrm{FLAIR}\},
\]

\[
\operatorname{ImageLineage}(A_{\phi}^{S_3,\mathrm{T1ce},b})
=
\{\mathrm{T1},\mathrm{FLAIR},\mathrm{T1ce}\}.
\]

Their annotation protocol is the same, but the roles are recorded explicitly:

\[
\operatorname{LabelLineage}(B_{S_3}^{b})
=
\{\text{BraTS multiparametric expert protocol; assigned H1/H2/H3 base labels}\},
\]

\[
\operatorname{LabelLineage}(A_{\phi})
=
\operatorname{LabelLineage}(B_{S_3}^{b})
\cup
\{\text{H1 teacher and adapter labels}\}.
\]

\(\operatorname{InitLineage}(A_{\phi})\) records the canonical base hash, the H1-provenance T1ce encoder hash, the exact zero-column teacher growth rule, and the registered random seed. No external checkpoint is implicit.

The M1 claim is therefore **excluded-image/update-path purity**, not information-theoretic absence of T1ce-related information from labels. E3 and E4 use the same donor labels, so \(E4_{\mathrm{aug}}-E3_{\mathrm{aug}}\) isolates the incremental T1ce image pathway under fixed annotation provenance.

The manifest records H1/H2/H3 total base-update provenance, H1/H3 \(S_3\)-track provenance, H1 T1ce-encoder provenance, the standing M1 reuse-rights digest, and canonical-base initialization. The augmented H3 composite is intentionally cross-boundary and must not be called a pure M1 track.

### 13.3 M2 firewall

For any M1 track \(S\), an M2 artifact is rejected from M1 if:

\[
\operatorname{ImageLineage}(A)\nsubseteq S.
\]

The server allowlist contains only M1 object types. A test attempts to serialize the adapter as an M1 update and must fail.

### 13.4 Fallback equivalence

With \(d_g=0\):

\[
\ell_{\mathrm{aug}}=\ell_B.
\]

The rollback test bypasses the adapter call entirely. It must be bit-identical under deterministic FP32 CPU inference; under the registered deterministic accelerator kernel, require maximum absolute logit difference \(\le10^{-6}\). Report the kernel, precision, and observed maximum rather than calling approximate equality byte identity.

### 13.5 No recipient Pareto theorem

The architecture protects the M1 base and other clients. It does not prove H3 improves. CDRD is accepted only by validation and all rejected artifacts are reported.

---

## 14. Four-hospital federation

### 14.1 Hospitals

| Hospital | Owned modalities | M1 send set | M1 role | M2 role |
| :--- | :--- | :--- | :--- | :--- |
| H1 | T1, T1ce, T2, FLAIR | T1, T1ce, T2, FLAIR | Full anchor; contributes masked data to \(S_3\) | Authorized T1ce donor for H3 |
| H2 | T1, T1ce, T2 | T1, T2 | Send-gated T1ce case | Not a T1ce donor in the primary experiment |
| H3 | T1, FLAIR | T1, FLAIR | Pure restricted track and target | T1ce artifact recipient |
| H4 | T1, T1ce, T2 | T1, T1ce, T2 | Delayed M1 joiner | No primary M2 role |

Selected Phase-1 cohort audiences are fixed: H1 receives the T1, T1ce, T2, and FLAIR cohorts listed in Section 6.2; H2 receives the T1, T1ce, and T2 cohorts; H3 receives the T1 and FLAIR cohorts; and delayed H4 receives the frozen T1, T1ce, and T2 snapshots when it joins. Every listed audience edge is true in the static M1 receive matrix; no unlisted modality is delivered.

### 14.2 M1 tracks

| Track | Modalities | Contributors |
| :--- | :--- | :--- |
| \(S_1\) | T1, T1ce, T2, FLAIR | H1 |
| \(S_2\) | T1, T2 | H2 |
| \(S_3\) | T1, FLAIR | H3 plus masked H1 contribution |
| \(S_4\) | T1, T1ce, T2 | H4 after delayed join |

M2 uses the final canonical \(S_3\) release.

### 14.3 Dataset

Dataset: **BraTS 2021**, fixed for the main paper, using all \(1{,}251\) labelled training patients as patient-disjoint artificial silos. BraTS 2020 may be used for code pilots only and must not be mixed into the primary result.

Sort the official patient identifiers lexicographically. Use PCG64 seed 901 to select one fixed 50-patient H3 final-test cohort; these identities are excluded from training, validation, calibration, and teacher fitting in every run. For federation-partition seeds \(\{1103,2207,3301\}\), independently permute the remaining \(1{,}201\) identities and assign the first 500 to H1, next 313 to H2, next 200 to the non-test portion of H3, and final 188 to H4. Retain the within-hospital order produced by that permutation and take the contiguous role counts below in their listed order; no second unspecified split seed is used. Use training seeds \(\{17,29,43\}\) within each partition. Release the resulting patient-ID manifests and their SHA-256 hashes before model comparison.

Each of three preregistered federation partitions uses these counts:

| Hospital | Patients | Approximate share |
| :--- | ---: | ---: |
| H1 | 500 | 40% |
| H2 | 313 | 25% |
| H3 | 250 | 20% |
| H4 | 188 | 15% |

Within each federation partition, the deterministic patient-level splits are: H1 350/50/100 for train/model-selection-validation/release-audit, H2 219/31/63 for train/validation/test, and H4 132/19/37 for train/validation/test. H1's 100 release-audit patients remain untouched until the M1 base and teacher are selected and the primary adapter hyperparameters and teacher-temperature procedure are locked; they are never used to update a model or choose an adapter setting.

H3 reserves distinct roles:

| H3 role | Patients | Use |
| :--- | ---: | :--- |
| M1 training | 125 | Train the pure \(S_3\) track |
| M1 validation | 25 | Select the canonical M1 base before M2 |
| M2 calibration | 25 | Fit the four CDRD gate values |
| M2 acceptance validation | 25 | Choose augmented route or fallback |
| Final test | 50 | Fixed across all partitions; locked scientific evaluation |

Never split 2D slices from one patient across roles. Use identical preprocessing, partitions, and seeds for every M2 condition.

The original T1ce volumes assigned to H3 are hidden from every deployable condition. E7 alone may access them at final evaluation as an explicitly non-deployable full-input donor-teacher reference; it is not called an oracle or upper bound.

### 14.4 BraTS label caveat

BraTS enhancing tumour is defined using contrast-enhanced T1Gd/T1ce information. ET improvement is expected to be especially tied to T1ce-derived supervision. Report WT, TC, and ET separately and use macro-region performance as the primary endpoint. Do not claim that CDRD reconstructs T1ce or creates label-pure evidence.

The official dataset description is available from [BraTS](https://www.med.upenn.edu/cbica/brats2021/).

### 14.5 Preprocessing and 2D-to-3D evaluation

Use the released co-registered, skull-stripped, \(1\,\mathrm{mm}^3\) BraTS volumes on the \(240\times240\times155\) grid. For each patient and modality, compute mean and standard deviation over nonzero brain voxels, apply z-normalization with \(10^{-8}\) denominator floor, clip to \([-5,5]\), and keep outside-brain voxels at zero. Do not crop in the primary experiment.

Split patients before extracting slices. Training uses aligned axial slices. Each epoch draws every tumour-containing slice once and an equal-sized, seeded sample of non-tumour brain-containing slices. Sample non-tumour slices without replacement when enough exist and with replacement only if the pool is smaller than the tumour-slice count. A batch therefore follows a fixed 1:1 tumour/non-tumour sampling target; empty slices are not removed from validation or testing.

Apply the same geometric transform to every co-registered input, its stored nonzero-brain mask, and the label: left-right flip with probability 0.5 and rotation sampled uniformly from \([-10^\circ,10^\circ]\), keeping the \(240\times240\) canvas, using bilinear interpolation for images and nearest-neighbour interpolation for masks/labels, and filling outside-image locations with image value 0 and label 0. Independently per modality, sample intensity scale from \([0.9,1.1]\) and additive normalized shift from \([-0.1,0.1]\), apply both only where the transformed brain mask is nonzero, and reset all other pixels to zero afterward. E5 performs patient-level pairing before these transforms.

At validation and test time, process all 155 axial slices without augmentation, stack predictions in original z-order, take the four-class argmax, and map model classes back to BraTS labels \(\{0,1,2,4\}\). No connected-component or other postprocessing is used. Compute 3D patient-level regions:

\[
WT=\{1,2,4\},\qquad TC=\{1,4\},\qquad ET=\{4\}.
\]

For Dice, both prediction and reference empty gives 1 and exactly one empty gives 0. Define a 3D surface as foreground voxel centres with at least one 6-connected neighbour outside the mask or at a volume boundary. When both surfaces are nonempty, compute Euclidean nearest-neighbour distances in physical millimetres in both directions, concatenate the directed distances, and take the 95th percentile with linear interpolation. Both empty gives HD95 0; exactly one empty receives the fixed grid diagonal \(\sqrt{239^2+239^2+154^2}\) mm as a finite failure penalty. Report the count of one-empty cases. Inference logits and softmax are FP32; class argmax resolves an exact tie to the lowest class index. Freeze and hash this evaluator code before any result is inspected.

---

## 15. Experimental plan

### 15.1 Primary question

> Does a consented H1 T1ce-derived CDRD artifact improve H3 beyond the same local capacity and donor knowledge without T1ce, while preserving the M1 base?

The preregistered primary contrast is:

\[
\Delta_{\mathrm{primary}}
=
\operatorname{MacroDice}_{H3}(E4_{\mathrm{aug}})
-
\operatorname{MacroDice}_{H3}(E3_{\mathrm{aug}}).
\]

Here `_aug` means the locked learned gate with \(d_g=1\) for every run. Validation-selected deployment output is a separate operational estimand, \(E4_{\mathrm{deploy}}\), and never replaces a forced-on observation in the causal analysis.

### 15.2 Conditions

| ID | Condition | Purpose |
| :--- | :--- | :--- |
| E0 | Frozen M1 \(S_3\) base | Main compliant baseline |
| E1 | Forced-on same CDRD architecture trained only at H3 from random initialization, then gate-calibrated | Extra local capacity control |
| E2 | Same-shape random adapter with each layer's Frobenius norm matched to E4; gate calibrated at H3 | Imported object/scale placebo |
| E3 | Forced-on H1 adapter distilled from an otherwise identical teacher whose T1ce slot is hard-zeroed | Donor cohort, labels, capacity, and pipeline without T1ce |
| E4 | Forced-on CDRD-Frozen from H1 \(S_3+\mathrm{T1ce}\) teacher | Primary causal method; route-selected output reported as \(E4_{\mathrm{deploy}}\) |
| E4b | CDRD-FT | Recipient fine-tuning ablation |
| E5 | Forced-on H1 T1ce paired by seeded patient-level derangement | Paired cross-modal semantic control |
| E6a | Raw prototype CrossAbsorb | Empirically contingent lightweight diagnostic |
| E6b | Calibrated M2-Lite prototypes | Defensible lightweight variant |
| E7 | The E4 restricted teacher evaluated zero-shot on hidden H3 T1ce test inputs | Non-deployable full-input donor-teacher reference, not an upper bound |

E0, E1, E3, E4, and E5 are confirmatory and receive the full \(3\times3\) replication. E7 is a required non-deployable reference using the corresponding E4 teachers. E6a/E6b are secondary mechanism comparisons, E4b is a fine-tuning ablation, and E2 is an appendix placebo; their reduced replication, if compute-limited, must be fixed before any result inspection and cannot support the main claim.

Control construction is fixed as follows:

- **E1:** freeze the same canonical M1 base and initialize the same CDRD sidecar without a donor artifact. First set \(d_g=1\) and fix \(\alpha_c=1\), then train the sidecar alone on the 25 H3 M2-calibration patients for exactly 50 epochs with Dice + cross-entropy and keep the final checkpoint. Next freeze the sidecar, initialize four fresh gate logits at \(-4\), and train only those gates on the same patients for the registered 50-epoch calibration schedule. Never consult acceptance validation during either stage. This avoids optimizing a zero-output sidecar through a nearly closed gate and deliberately makes E1 a strong local-capacity baseline.
- **E2:** draw independent Gaussian tensors under the registered seed and rescale each tensor to the corresponding E4 tensor's Frobenius norm. Freeze this placebo adapter and train only the same four H3 gates. No E4 coefficient position is retained.
- **E3:** use E4's exact three-slot teacher topology, H1 patients, labels, initialization mapping, optimizer, adapter architecture, fixed loss-weight tuple, and schedule. Replace the T1ce input by a constant zero tensor before fusion and keep its new input-column blocks fixed at zero; no T1ce image is loaded. All remaining parameters train exactly as in E4.
- **E5:** construct a patient-level, no-fixed-point derangement of H1 T1ce volumes within ET-presence and tumour-volume-quartile strata. Deterministically merge an adjacent volume stratum when fewer than two patients make a derangement impossible. Train on \((x_{S_3}^{n},x_{\mathrm{T1ce}}^{\pi(n)},y^n)\). Never shuffle slices independently, never cross a data split, and apply co-registered augmentation after pairing. Use the exact E4 topology, fixed loss-weight tuple, and schedule.
- **E7:** reuse the frozen E4 H1 teacher and infer directly on the fixed H3 test cohort with its hidden T1ce volumes. This diagnoses the donor teacher under H1-to-H3 domain shift; it is neither deployable nor a mathematical upper bound. A separately trained H3 full-input model, if added, is labelled E7b and explicitly policy-noncompliant.

The indispensable causal comparison is:

\[
E4_{\mathrm{aug}}-E3_{\mathrm{aug}}.
\]

If \(E4_{\mathrm{aug}}\) does not exceed \(E3_{\mathrm{aug}}\), the experiment has not shown modality-specific T1ce transfer. If \(E4_{\mathrm{aug}}\) does not exceed \(E5_{\mathrm{aug}}\), it has not shown that correct paired T1ce anatomy is responsible.

### 15.3 External baselines

External methods are optional contextual utility comparisons and are not part of the R4 success gate. Subject to available official implementations and matched data assumptions:

- Local-only;
- FedAvg-All, labelled policy-noncompliant;
- MFCPL, adapted to the dataset and labelled governance-unaware;
- PLOT as the incomplete-modality distillation baseline;
- FedAFD, if a faithful public-data/server-distillation adaptation is feasible, labelled governance-unaware;
- OmniFM, if its heterogeneous-modality medical FL pipeline can be matched to the fixed BraTS task, labelled governance-unaware;
- SimMLM, if its centralized missing-modality assumptions can be matched without changing the primary federation;
- the original M1 architecture;
- M2-Lite and CDRD.

Before any run, freeze the exact repository URL, commit, environment, input policy, and CAMFS adaptation in the experiment registry. Label each method as consent-compatible, utility-only/policy-noncompliant, adapted under the CDRD grant, or non-deployable. Do not force an external method into the table if its input assumptions cannot be reproduced fairly; report the infeasibility instead. The complete primary protocol remains E0/E1/E3/E4/E5 plus containment tests even if no external port is feasible.

### 15.4 Metrics

Primary endpoint:

\[
\text{H3 macro Dice}
=
\frac{\text{Dice}_{WT}+\text{Dice}_{TC}+\text{Dice}_{ET}}{3}.
\]

Secondary:

- Dice for WT, TC, ET separately;
- HD95 for each region;
- four-class Brier score over nonzero brain voxels;
- artifact size and additional inference compute;
- M1 hash/prediction invariance;
- policy-routing pass/fail results.

As a sensitivity result, also report ET Dice separately for patients whose ground-truth ET region is present and absent. This does not replace or modify the registered macro-Dice primary endpoint.

For each patient, Brier score is \(\frac{1}{4|\Omega_{\mathrm{brain}}|}\sum_{q\in\Omega_{\mathrm{brain}}}\sum_c(p_c(q)-y_c(q))^2\), then averaged across patients. No lesion-component endpoint is part of the primary protocol.

### 15.5 Statistics

- Use the unique patient as the statistical unit; slices and repeated model runs are never independent samples.
- Use the three registered federation partitions and three training seeds per partition for E0, E1, E3, E4, and E5. Conditions are paired within the same partition, base, seed, and fixed H3 test patient.
- For patient \(p\), partition \(r\), and seed \(s\), compute the paired contrast \(d_{p,r,s}\). Average first over seeds and then partitions to obtain one \(d_p\) per unique test patient. The primary estimand is explicitly the mean patient effect of this fixed nine-model ensemble; it is conditional on the three registered partitions and three registered training seeds.
- Form the primary two-sided \(95\%\) confidence interval by bootstrapping the 50 unique H3 test patient identities and recomputing the mean of \(d_p\). Use 10,000 PCG64 percentile-bootstrap resamples with seed 8803. Report between-seed and between-partition standard deviations separately; do not count nine repeated predictions as nine patients.
- Add a run-aware sensitivity interval with 10,000 hierarchical resamples (PCG64 seed 8804): resample the three partition indices, resample three training-seed indices within each selected partition, resample the 50 patient identities, and recompute the full mean contrast. This sensitivity reflects the finite registered partition/seed variation but is not presented as population-wide hospital uncertainty.
- For the three key secondary contrasts \(E4_{\mathrm{aug}}-E1_{\mathrm{aug}}\), \(E4_{\mathrm{aug}}-E0\), and \(E4_{\mathrm{aug}}-E5_{\mathrm{aug}}\), use one-sided paired sign-flip randomization tests on the corresponding 50 averaged \(d_p\) values, with 100,000 sign vectors and PCG64 seeds 8811, 8812, and 8813 respectively. For observed mean \(\bar d\), compute \(p=[1+\#\{\bar d_{\mathrm{flip}}\ge\bar d\}]/100{,}001\). Apply Holm adjustment across this three-test family at \(\alpha=0.05\), and report ordinary two-sided 95% patient-bootstrap effect intervals alongside the adjusted p-values. Treat the WT/TC/ET versions of the primary E4-E3 contrast as a separate Holm-adjusted three-test family.
- As a further sensitivity analysis, fit a crossed mixed model with fixed condition and random intercepts for patient identity, partition, and partition-by-seed. With only three partitions, this is not the primary inference.
- Predefine every endpoint and test before inspecting H3 final-test results.
- Report each hospital separately for M1; never average away harm to one site.

### 15.6 Validation rule

Set \(d_{\mathrm{deploy}}=1\) only when:

1. the donor viability flag is \(q_T=1\);
2. locked \(E4_{\mathrm{aug}}\) H3 acceptance-validation macro Dice is higher than E0;
3. no acceptance-validation region loses more than the preregistered experimental margin \(\delta=0.01\) Dice;
4. the artifact passes all policy and hash checks.

The \(0.01\) value is an experimental noninferiority margin, not a clinically validated threshold.

Otherwise set \(d_{\mathrm{deploy}}=0\). Final claims come only from the untouched H3 test set. Always report both forced-on \(E4_{\mathrm{aug}}\) and route-selected \(E4_{\mathrm{deploy}}\); the latter equals E0 for a rejected run and never replaces the former in causal contrasts.

### 15.7 Required ablations

1. loss terms: remove KD, residual target, and TV separately;
2. temperature \(T\);
3. adapter width \(d\in\{8,16,32\}\);
4. frozen versus recipient-fine-tuned adapter;
5. raw versus calibrated prototype route;
6. correct versus shuffled donor pairing;
7. canonical base hash match versus mismatch;
8. grant on versus off.
9. for CDRD-P: learned prototypes versus zero maps, norm-matched random prototypes, and class-permuted prototypes;
10. CDRD-P paired projection versus a shuffled-pair projection.

### 15.8 Failure interpretation

| Result | Interpretation |
| :--- | :--- |
| \(E4_{\mathrm{aug}}>E3_{\mathrm{aug}},E1_{\mathrm{aug}},E5_{\mathrm{aug}}\) | Evidence for useful, correctly paired T1ce-derived transfer |
| \(E4_{\mathrm{aug}}\approx E3_{\mathrm{aug}}\) | Donor knowledge transfers, but T1ce-specific contribution is unsupported |
| \(E4_{\mathrm{aug}}\approx E1_{\mathrm{aug}}\) | Improvement is explained by local capacity |
| \(E4_{\mathrm{aug}}<E0\) | Artifact is harmful in the forced-on test analysis; a run rejected by the preregistered acceptance rule falls back in \(E4_{\mathrm{deploy}}\) |
| E6 succeeds | M1/projection geometry is empirically usable in this setting |
| E6 fails but \(E4_{\mathrm{aug}}\) succeeds | Dense distillation is preferable to prototype conditioning |
| Policy/hash test fails | Implementation is not compliant with the M2 specification |

---

## 16. Reproducible implementation defaults

### 16.1 Shared training defaults

| Setting | Initial value |
| :--- | :--- |
| Input | 2D axial \(240\times240\) |
| Dataset | BraTS 2021 |
| Batch size | 16 slices |
| Segmentation loss | soft Dice + cross-entropy |
| Optimizer | AdamW |
| Weight decay | \(10^{-4}\) |
| Numeric precision | FP32 only in the primary protocol; mixed precision is an ablation |
| M2 early stopping | H1 teacher/adapter selection only; H3 calibration uses fixed epochs; M1 phase-specific stopping is defined below |
| Replication | 3 federation partitions \(\times\) 3 training seeds |

Adam and AdamW use \(\beta=(0.9,0.999)\) and \(\epsilon=10^{-8}\). Learning-rate schedules are constant; gradient clipping is not used. These choices must be identical across matched conditions.

Each registered training seed initializes model parameters, the axial-slice sampler, augmentation draws, data-loader generators, and any condition-specific derangement. Enable deterministic framework algorithms, seed each worker from the registered run seed, and record the framework, accelerator, and kernel versions. Unless a section names another selection metric or fixes the final epoch explicitly, “improvement” means validation macro Dice increases by more than \(10^{-4}\); retain the earliest checkpoint on an exact tie.

Map raw BraTS labels \(\{0,1,2,4\}\) to model indices \(\{0,1,2,3\}\). For a batch of \(N\) slices, spatial index set \(\Omega\), softmax probabilities \(p\), one-hot labels \(y\), tumour classes \(\mathcal C_T=\{1,2,3\}\), and \(\epsilon_D=10^{-5}\), every occurrence of Dice + cross-entropy means:

\[
\mathcal L_{\mathrm{Dice+CE}}
=
-\frac{1}{N|\Omega|}\sum_{n=1}^{N}\sum_{q\in\Omega}\log p_{n,y_n(q)}(q)
+
1-\frac{1}{3N}\sum_{n=1}^{N}\sum_{c\in\mathcal C_T}
\frac{2\sum_{q\in\Omega}p_{n,c}(q)y_{n,c}(q)+\epsilon_D}
{\sum_{q\in\Omega}p_{n,c}(q)+\sum_{q\in\Omega}y_{n,c}(q)+\epsilon_D}.
\]

No class weighting, deep supervision, or postprocessing is used in the primary model.

### 16.2 M1 Phase-1 defaults

| Setting | Fixed value |
| :--- | :--- |
| Initialization | Kaiming-normal, no external pretraining |
| Optimizer | AdamW, learning rate \(3\times10^{-4}\), weight decay \(10^{-4}\) |
| Local work | 1 local epoch per cohort round, batch size 16 |
| Contrastive temperature | \(\tau=0.1\) |
| Objective weight | \(\lambda_1=1\) |
| Bottleneck samples | all \(15\times15\) locations of each sampled slice |
| Rounds | minimum 20, maximum 100 |
| Stop | prototype drift \(<0.01\) for 5 consecutive rounds after round 20 |

Every cohort has independent model/prototype state. Local optimizer state is freshly reset each round as specified in Section 6.2. Prototype means are recomputed from the post-local model and aggregated using the class-specific patient-support rule.

### 16.3 M1 Phase-2 defaults

| Setting | Fixed value |
| :--- | :--- |
| Encoders | frozen, evaluation mode |
| Optimizer | AdamW, learning rate \(10^{-3}\), weight decay \(10^{-4}\) |
| Local work | 1 native or masked local epoch per track-cohort round, batch size 16 |
| Fusion alignment | same class-InfoNCE form as Phase 1, on \(z_S\), temperature 0.1 |
| Objective weight | \(\lambda_2=0.1\) |
| Rounds | minimum 20, maximum 100 |
| Selection | highest contributor-patient-weighted validation macro Dice |
| Stop | no validation improvement for 10 consecutive rounds after round 20 |

Run M1 once for each registered federation-partition/training-seed pair. Every E-condition sharing that pair loads the same content-addressed M1 binary; M1 is not retrained separately per M2 condition.

### 16.4 Teacher defaults

| Setting | Initial value |
| :--- | :--- |
| Trainable state | private fusion head and decoder only |
| Encoders | frozen M1 encoders |
| Optimizer | AdamW, learning rate \(3\times10^{-4}\), weight decay \(10^{-4}\) |
| Batch size | 16 |
| Local epochs | maximum 50, early stopping patience 10 |
| Initialization | exact subset-growth from canonical \(S_3\); five new T1ce input-column blocks initialized to zero |
| Selection | highest H1 validation macro Dice; improvement threshold \(10^{-4}\), earliest exact tie |
| Calibration | fit \(T_T^{\mathrm{cal}}\) by H1-validation NLL; keep the canonical base coordinate unchanged |

After the teacher checkpoint is selected, fit its one scalar temperature deterministically. From all nonzero-brain H1-validation voxels, sample without replacement at most \(1{,}000{,}000\) voxel logits using PCG64 seed 6601. Evaluate mean NLL at 201 temperatures equally spaced in \(\log T\) over \([0.25,8]\), select the lowest-NLL value, and choose the smaller temperature on an exact tie. No base temperature is fitted.

### 16.5 Adapter defaults

| Setting | Initial value |
| :--- | :--- |
| Projection width | 16 |
| Learning rate | \(10^{-3}\) |
| Batch size | 16 |
| Maximum epochs | 30, early stopping patience 7 |
| Selection | highest H1 validation macro Dice; improvement threshold \(10^{-4}\), earliest exact tie |
| \(T\) | 2 |
| \(\lambda_{\mathrm{seg}}\) | 1 |
| \(\lambda_{\mathrm{KD}}\) | 1 |
| \(\lambda_{\Delta}\) | 0.1 |
| \(\lambda_{\mathrm{TV}}\) | \(10^{-5}\) |
| Output initialization | zero |

The confirmatory E3/E4/E5 runs use the fixed tuple above in every partition and seed; none of the three conditions reselects its own loss weights. The following grid is an exploratory adapter ablation only and cannot replace a confirmatory checkpoint:

\[
T\in\{1,2,4\},\quad
\lambda_{\mathrm{KD}}\in\{0.5,1,2\},\quad
\lambda_{\Delta}\in\{0,0.1,0.5\}.
\]

Evaluate this exploratory grid only on H1 validation and report every registered cell. H3 validation and test data remain inaccessible, and grid results do not alter the primary configuration.

### 16.6 Recipient calibration defaults

| Setting | Initial value |
| :--- | :--- |
| Trainable parameters | 4 gate logits only |
| Loss | the same Dice + cross-entropy definition as M1 |
| Optimizer | Adam, learning rate \(10^{-2}\), weight decay 0 |
| Batch size | 16 slices |
| Epochs | exactly 50; keep final checkpoint, no early stopping |
| Initialization | \(a_c=-4\), so \(\alpha_c\approx0.018\) |

All controls receive the same gate-calibration schedule. E1 first completes the fixed-open-gate sidecar stage in Section 15.2, freezes that sidecar, and only then runs this gate-only schedule from \(a_c=-4\); it never jointly optimizes a zero-output sidecar through the near-closed gate. The M2 acceptance-validation set is consulted only after the final calibration checkpoint is locked. The final test set is never used for fitting, checkpoint selection, or route acceptance.

### 16.7 M2-Lite defaults

For CDRD-P, downsample the H1 segmentation mask to the \(15\times15\) bottleneck grid by nearest-neighbour interpolation. In every 16-slice training batch, sample without replacement at most 32 bottleneck locations per present class per slice, using the registered training seed. The positive for each sampled \(S_3\) feature is the T1ce feature from the same patient, slice, and grid location. Every other sampled T1ce location in the batch is a negative; the reverse T1ce-to-\(S_3\) direction uses the analogous candidate set. The projection loss is the arithmetic mean of these two cross-entropies, with normalized embeddings and temperature 0.1.

Train \(W_S,W_U\) with AdamW at learning rate \(10^{-3}\), weight decay \(10^{-4}\), batch size 16, for at most 30 epochs with patience 7. Select by H1 validation top-1 paired-location retrieval accuracy, averaged over both directions; improvement must exceed \(10^{-4}\), with the earliest exact tie retained. Before training, register 100 patient-level derangements of the T1ce validation identities. CDRD-P passes its correspondence gate only if the patient-bootstrap 95% lower confidence bound for paired retrieval accuracy minus the mean accuracy of those 100 deranged controls is greater than zero. Compute it with 10,000 patient-identity percentile-bootstrap resamples and PCG64 seed 7723. A failed gate makes E6b unavailable for release and is reported; it is not repaired using H3 data.

For the required shuffled-projection ablation, train a separate projection pair under the identical schedule after applying one fixed, no-fixed-point patient derangement to the H1 training T1ce identities; keep slice and bottleneck coordinates unchanged after patient pairing. It receives no correctly paired location during fitting and is evaluated by the same locked H1 validation retrieval protocol.

After selection, freeze both projections, form the four prototypes from H1 training patients only, and train the residual head on H1 with the adapter optimizer, epoch limit, patience, objective reductions, and validation grid in Section 16.5. Select the head by H1 validation macro Dice. Release only \(W_S\), the four frozen prototypes, and that head; \(W_U\) and every donor feature remain at H1. Fit the H3 class gates using Section 16.6 without changing any released CDRD-P tensor.

---

## 17. Communication and compute

### 17.1 Paper workload

One deployable H1-to-H3 artifact adds:

1. one H1 restricted teacher training job, comparable to one local Phase-2 fusion/decoder run because encoders remain frozen;
2. one small adapter training job;
3. one H3 four-parameter calibration job.

That is the deployment unit, not the total paper budget. The confirmatory protocol trains nine M1 bases (three partitions by three seeds). For every base it trains the E3, E4, and E5 donor-teacher/adapter pipelines, the staged E1 local sidecar, and recipient gates. Thus the minimum confirmatory core adds 27 teacher jobs, 27 donor-adapter jobs, nine local E1 sidecar jobs, and the associated small calibration jobs. The appendix E2 placebo and secondary E6 variants add further jobs. M1 is reused across matched conditions within a partition/seed and no M2 condition trains an encoder.

### 17.2 Adapter cost

Approximate adapter:

- parameters: \(16{,}344\), plus four recipient-local gate scalars;
- adapter payload: approximately \(65.4\) KB FP32 or \(32.7\) KB FP16;
- additional compute: approximately \(0.30\) GMAC per \(240\times240\) slice, implementation-dependent.

Measure actual latency and memory on the experiment hardware rather than presenting estimates as measurements.

### 17.3 Scaling limitation

Let \(G\) be the number of grants and \(K\) the number of distinct donor/subset/modality teacher configurations.

- teachers scale approximately as \(O(K)\);
- adapters scale as \(O(G)\) when recipient bases differ;
- one adapter can technically be reused for recipients sharing the same canonical base, but every release still requires a separate policy grant.

The deployed topology has \(K=1,G=1\). Counterfactual teachers used only for causal experiments do not represent additional grants, but they do increase compute as stated above. Large-federation optimization is deferred.

### 17.4 M2-Lite communication

The four projected \(16\)-dimensional FP32 prototypes require \(256\) bytes. Including \(W_S\) and the residual head, the complete CDRD-P payload is approximately \(21.7\) KB FP32, versus approximately \(65.4\) KB for CDRD. Metadata is reported separately.

---

## 18. Paper claims and limitations

### 18.1 Defensible paper claim

> CAMFS M2 adds a static, bilateral, modality-scoped transfer route to an excluded-image/update-path M1 federation. A donor-local teacher distils privileged-modality knowledge into a detached residual adapter that consumes only the recipient's modalities. The adapter is auditable, removable, and excluded from M1 aggregation and seeding.

### 18.2 Novelty boundary

Do not claim novelty in:

- teacher-student distillation alone;
- prototypes alone;
- policy-aware FL in general;
- missing-modality segmentation in general.

Claim novelty in the combined formulation and enforcement:

- explicit recipient-side acceptance of a non-owned modality;
- named donor-to-recipient artifact scope;
- exact modality lineage;
- preserved unaugmented M1 fallback;
- non-propagation into shared tracks;
- causal controls for modality-specific benefit.

Recent multimodal methods reinforce the need for this narrow claim. FedAFD combines adversarial representation alignment, fusion, and server-side ensemble distillation; OmniFM targets modality-robust, task-agnostic medical FL through spectral knowledge; and SimMLM addresses centralized missing-modality learning with a dynamic mixture of experts. They are useful performance comparators where assumptions can be matched, but they do not directly supply CAMFS M2's recipient-bound, post-release artifact route with an unchanged, removable M1 fallback.

### 18.3 Main limitations

1. BraTS hospitals are artificial silos, not real institutional deployments.
2. ET labels are strongly tied to T1ce.
3. One donor-recipient grant does not demonstrate large-scale efficiency.
4. Honest execution is assumed.
5. No formal privacy guarantee is made for adapter weights.
6. A static configuration is audit-friendly but not a legal consent system.
7. The teacher sees paired modalities available only at the donor.
8. CDRD may fail under severe donor-recipient domain shift.
9. M1 cross-modal prototype geometry remains empirical.
10. Divergent pairwise receive policies can require multiple pre-mix M1 cohort replicas; the paper evaluates only the declared small set and does not solve cohort explosion.

### 18.4 Relevant research anchors

- [Cross-modal MRI distillation for missing sequences](https://pubmed.ncbi.nlm.nih.gov/34941496/)
- [PLOT federated incomplete multimodal segmentation](https://pubmed.ncbi.nlm.nih.gov/40030851/)
- [MFCPL cross-modal prototypes and explicit alignment](https://arxiv.org/abs/2401.13898)
- [FedAFD multimodal federated adversarial fusion and distillation](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_FedAFD_Multimodal_Federated_Learning_via_Adversarial_Fusion_and_Distillation_CVPR_2026_paper.html)
- [OmniFM modality-robust and task-agnostic medical federated learning](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_OmniFM_Toward_Modality-Robust_and_Task-Agnostic_Federated_Learning_for_Heterogeneous_Medical_CVPR_2026_paper.html)
- [SimMLM centralized multi-modal learning with missing modality](https://openaccess.thecvf.com/content/ICCV2025/html/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.html)
- [pFedDKS detached personalized knowledge sharing](https://researchportal.hkust.edu.hk/en/publications/pfeddks-detached-knowledge-sharing-for-personalized-federated-lea/)
- [FedHide and prototype leakage](https://eccv.ecva.net/virtual/2024/poster/2697)
- [PoliFL heterogeneous privacy policies](https://arxiv.org/abs/2003.06612)

---

## 19. Deferred full-system scope

The following belong in future work or a separate systems paper:

1. public-key signatures and institutional certificate authorities;
2. encrypted artifact delivery and hardware-backed key storage;
3. trusted execution for recipient-specific builds;
4. dynamic start/end times and automated policy reconciliation;
5. multi-donor artifact composition and revocation dependencies;
6. secure aggregation for recipient-specific prototype cohorts;
7. patient-level differential privacy and privacy accounting;
8. systematic membership, reconstruction, and property-inference red-teaming;
9. patient withdrawal and verified machine unlearning;
10. production audit retention, deletion receipts, and regulator interfaces;
11. adversarial or Byzantine hospitals;
12. real cross-site prospective clinical validation.
13. optimization of consent-cohort replica growth for large policy graphs.

These are important deployment requirements. They are explicitly outside the empirical claims of this paper.

---

## 20. Completeness and go/no-go checklist

### 20.1 Architecture completeness

- [x] Pairwise receive policies compiled into closed pre-mix consent cohorts
- [x] Documented concat + \(1\times1\) M1 ablation selected as the canonical M2 base architecture
- [x] Exact five-tap M1 encoder, fusion, decoder, and subset-growth mapping defined
- [x] M1 Phase 1 retained and frozen before task training
- [x] M1 Phase 2 retained with send-keyed isolated tracks
- [x] Final canonical \(S_3\) checkpoint defined
- [x] H2's inherited T1 update lineage and standing M1 reuse right recorded
- [x] Static bilateral M2 policy defined
- [x] Exact H1 teacher modality set defined
- [x] T2 lineage explicitly excluded
- [x] CDRD adapter architecture defined
- [x] Loss and initial hyperparameters defined
- [x] Recipient calibration and fallback defined
- [x] M2-Lite prototype comparison defined
- [x] Non-propagation assumptions and tests defined
- [x] Causal controls and statistics defined
- [x] Patient splits, preprocessing, 2D-to-3D reconstruction, and empty-region metrics defined
- [x] Paper versus system scope separated

### 20.2 Pre-experiment gates

- [ ] Retrain and reproduce the canonical concat-fusion M1 \(S_3\) baseline; do not reuse a historical cross-attention checkpoint
- [ ] Verify H1 and H3 canonical base hashes match
- [ ] Verify H1 restricted teacher passes the locked 100-patient release-audit gate
- [ ] Verify M2 serialization cannot enter M1 update messages
- [ ] Freeze data partitions, preprocessing, metrics, and primary endpoint
- [ ] Register all E0–E7 configurations before H3 test evaluation

### 20.3 Paper success criteria

Minimum architecture result:

- every policy/hash/isolation test passes in all nine partition/seed configurations; revocation restores the same base hash and the Section 13.4 prediction tolerance.

Minimum utility result:

\[
\operatorname{LCB}_{95\%}
(E4_{\mathrm{aug}}-E3_{\mathrm{aug}})>0
\]

on H3 test macro Dice for the preregistered primary contrast. In addition, each observed key-secondary mean difference against E1, E0, and E5 must be positive and its one-sided paired sign-flip p-value must remain below 0.05 after the Section 15.5 Holm adjustment. Ordinary 95% effect intervals are reported but are not mislabelled as simultaneous Holm-adjusted bounds. Finally, no forced-on WT, TC, or ET Dice point estimate may be more than \(0.01\) below E0; this is a deterministic experimental safety heuristic, not a claim of statistical or clinical noninferiority. Report \(E4_{\mathrm{deploy}}\) and its qualification rate separately; it is not substituted into these tests.

If the utility criterion fails, do not claim R4 performance success. Report the compliant route, the negative result, and which control explains the outcome.

---

## Final unified decision

The earlier CDRD document is complete as a broad architecture and governance design, but it is larger than one research paper. This unified M2 specification extracts the publishable core:

\[
\boxed{
\text{M1 frozen excluded-image-path base}
+
\text{static bilateral grant}
+
\text{restricted donor teacher}
+
\text{detached CDRD sidecar}
+
\text{matched causal evaluation}
}
\]

That is the canonical M2 research architecture for Gap 6 / R4. It is fully specified for implementation, but it becomes a completed research result only after the checklist experiments produce and validate evidence.
