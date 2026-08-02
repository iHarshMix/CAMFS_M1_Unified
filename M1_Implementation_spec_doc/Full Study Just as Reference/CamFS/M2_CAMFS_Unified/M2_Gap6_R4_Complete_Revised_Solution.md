# M2 Gap 6 / R4 — Complete Revised Solution

## Consent-Bounded Detached Residual Distillation (CDRD)

**Status:** research-grounded canonical candidate for M2. This document intentionally leaves M1 and the earlier M2 prototype draft intact. It specifies a replacement primary mechanism for Gap 6 / R4, not a minor extension of the existing CrossAbsorbHead.

**Decision in one sentence:** preserve M1 as an immutable, modality-pure baseline; satisfy R4 through a post-M1, bilateral-consent, donor-trained residual sidecar that uses only the recipient's owned modalities at inference and is never federated or merged back into M1.

---

## 1. Executive decision

The current M2 proposal has the right containment instinct: non-owned-modality knowledge must not flow back into shared M1 tracks. Its proposed transfer object, however, is not strong enough to be the canonical solution:

1. M1 Phase 1 aligns each modality only to that modality's own prototype bank. It does not establish a valid common coordinate system in which a fused T1/FLAIR representation can attend directly to a T1ce prototype.
2. Four per-class centroids do not carry patient-specific geometry or dense tumour-boundary information.
3. A fresh local decoder/head is a major capacity confound unless it receives M1's complete multi-scale interface and has matched controls.
4. A coarse source-wide exposure flag is not a deployable bilateral consent agreement.

The recommended M2 mechanism is therefore **Consent-Bounded Detached Residual Distillation (CDRD)**:

1. Finish M1 normally and pin a final immutable snapshot for the recipient's pure track.
2. A specifically authorized donor trains, only on its own paired data, a private teacher using exactly the recipient subset plus the authorized non-owned modality.
3. The donor distils the teacher's incremental dense prediction into a small residual adapter that consumes only the pinned M1 feature pyramid and logits.
4. The adapter is released only through a named donor-to-recipient artifact grant with a purpose, base-model hash, expiry, privacy metadata, and revocation path.
5. The recipient optionally calibrates the sidecar locally. It always retains the untouched M1 baseline and deploys the sidecar only after a preregistered local validation gate.

This meets the useful interpretation of R4: it creates a selective, auditable path by which a recipient can use a consented modality-derived artifact without contaminating the shared federation. It does **not** promise that every such artifact improves every recipient; that empirical benefit must be demonstrated against matched controls.

---

## 2. Scope, terminology, and claims

### 2.1 What this solves

For recipient hospital \(i\), let \(S_i\) be the input subset used by its deployed M1 track. Let \(m \notin O(i)\) be a modality it does not own, such as T1ce for Hospital 3. CDRD permits an approved donor \(j\) that owns \(m\) to release a derived, recipient-bound model artifact that improves an \(S_i\)-only predictor if the evidence supports doing so.

At recipient inference:

\[
x_{S_i} \longrightarrow \text{pinned M1 base} \longrightarrow \text{private CDRD sidecar} \longrightarrow \hat y_{\mathrm{aug}}.
\]

The recipient never needs \(x_m\), does not synthesize a scan, and does not receive raw donor data.

### 2.2 What this does not claim

- It is **not** a proof that a recipient cannot lose accuracy. The recipient can overfit, receive a poor artifact, or face domain shift.
- It is **not** a claim that an adapter, prototype, or model weight is anonymous or free of patient-data risk.
- It is **not** a claim that T1ce information has been biologically reconstructed from T1/FLAIR. It is a model-lineage claim: the sidecar was trained using an authorized \(S_i \cup \{m\}\) donor pathway but operates from \(S_i\) alone.
- It is **not** a modification of M1's pure tracks. The augmented composite is intentionally not modality-pure; the immutable M1 baseline remains pure.
- It is **not** a blanket permission to use all information from a full-modality donor. Every modality that shapes the teacher or released artifact must be listed in its artifact manifest.

### 2.3 Terms

| Term | Meaning |
| :--- | :--- |
| Pure M1 base | The frozen M1 route for \(S_i\), with lineage restricted to \(S_i\). |
| Donor teacher | A donor-local model using exactly \(S_i \cup U\), where \(U\) is the explicitly authorized non-owned modality set. |
| CDRD adapter | A dense residual-logit sidecar trained at the donor to transfer the teacher's incremental benefit to an \(S_i\)-only base. |
| Artifact grant | A bilateral, versioned policy record authorizing one artifact to be released from a named donor to a named recipient for a defined purpose. |
| Base hash | Hash of the exact frozen M1 checkpoint, architecture, preprocessing contract, and feature-tap interface expected by the adapter. |
| Operational revocation | Immediate removal of the sidecar from serving and deletion/rebuild of every local component dependent on it. |

---

## 3. Why the existing CrossAbsorb prototype mechanism is not the primary solution

### 3.1 The key geometry failure

M1's stated Phase-1 loss is separately applied within each modality:

\[
\mathcal L_m =
\operatorname{InfoNCE}\left(z_m,\{\operatorname{Proto}_m^c\}_c\right).
\]

For any modality-specific orthogonal matrix \(Q_m\), replacing

\[
z_m \mapsto Q_m z_m,
\qquad
\operatorname{Proto}_m^c \mapsto Q_m\operatorname{Proto}_m^c
\]

leaves all within-modality cosine similarities, and therefore that loss, unchanged. Because M1 contains no loss term coupling \(m\) and \(m'\), each modality can have a different \(Q_m\). Consequently, M1's loss does not identify the claimed common coordinate system:

\[
\operatorname{Proto}_{\mathrm{T1}}^c
\approx
\operatorname{Proto}_{\mathrm{T1ce}}^c.
\]

M1's learned fusion head can still combine modality-specific encoder outputs. That does not imply that a raw T1ce prototype is semantically comparable to an already fused T1/FLAIR feature. Direct cross-attention from \(z_S\) to \(\operatorname{Proto}_{\mathrm{T1ce}}^c\) is therefore ungrounded under M1 as written.

This is a structural issue, not merely a hyperparameter concern. MFCPL explicitly introduces a shared projection and cross-modal alignment for prototype transfer rather than assuming unimodal prototype learning creates it. See [MFCPL](https://arxiv.org/abs/2401.13898).

### 3.2 Four global centroids are too weak for the stated task

The proposed cross-exposed object is four \(256\)-dimensional means. It contains no patient-specific spatial structure, no lesion shape, and no boundary map. A new attention head plus a fresh local decoder can improve because it adds local capacity, not because it carries useful T1ce-derived information.

That makes the existing prototype head appropriate as a low-bandwidth ablation only. It is not appropriate as the principal mechanism claimed to recover dense missing-modality benefit.

### 3.3 The decoder contract is incomplete

M1's decoder consumes a fused skip-feature pyramid at four resolutions plus the bottleneck. A bottleneck-only local branch followed by a new U-Net decoder is not equivalent to M1's decoder interface. It introduces a capacity and architecture change unrelated to knowledge transfer.

CDRD instead consumes read-only taps from the complete frozen M1 feature pyramid and emits a residual logit map. The original M1 decoder remains untouched.

### 3.4 A STABLE track is still a moving target

The earlier M2 proposal starts local absorption when a track reaches STABLE. Under M1, STABLE still permits continued Phase-2 updates. A private branch then trains against a changing feature distribution.

CDRD starts only after final M1 model selection and release. A changed M1 base hash invalidates the CDRD artifact and requires retraining or revalidation.

### 3.5 The current policy is too coarse

A source-wide flag such as \(R^{\mathrm{cross}}_{\mathrm{expose}}(j,m)\) effectively permits release to every non-owner. It cannot express:

- which hospital may receive the artifact;
- which exact modality set and labels shaped it;
- whether the artifact is a prototype, adapter, or teacher-derived residual;
- which M1 snapshot it is compatible with;
- whether local fine-tuning is allowed;
- when it expires or how revocation is handled.

R4 needs an artifact-level bilateral policy, not a global exposed-prototype pool.

---

## 4. Revised formal contract for R4

### 4.1 Requirement interpretation

The phrase in R4, “should be able to benefit,” cannot be a universal performance theorem. No learning algorithm can guarantee positive benefit under arbitrary donor quality, sample size, or domain shift.

The implementable and testable form is:

> If both parties authorize a bounded artifact, the system shall make a selective, auditable, non-propagating transfer path available. The recipient may deploy it only after it passes a preregistered local validation rule relative to its immutable M1 baseline.

This strengthens compliance and makes benefit an empirical hypothesis with a clear falsification protocol.

### 4.2 M1 remains unchanged

Keep all M1 policy matrices and their M1 meanings:

\[
R_{\mathrm{send}},\quad
R_{\mathrm{recv}},\quad
R_{\mathrm{send}}^{\mathrm{track}},\quad
R_{\mathrm{recv}}^{\mathrm{track}},\quad
R_{\mathrm{contribute}}.
\]

They continue to govern M1 Phase 1 and Phase 2 routing. M2 does not overload them, does not relax a pure M1 track, and does not inject cross-boundary artifacts into any M1 aggregation or seed operation.

### 4.3 M2 artifact-grant namespace

For an artifact grant \(g\), define:

\[
g =
\left(
j,\ i,\ S,\ U,\ a,\ p,\ b,\ d,\ q,\ v,\ t_{\mathrm{start}},\ t_{\mathrm{end}},\ \rho
\right),
\]

where:

| Field | Meaning |
| :--- | :--- |
| \(j\) | Named donor hospital, or an explicitly named closed donor group. |
| \(i\) | Named recipient hospital, or an explicitly named closed recipient group. |
| \(S\) | Recipient-owned input subset used by the pinned M1 base. |
| \(U\) | Every non-owned source modality allowed to shape the artifact. |
| \(a\) | Artifact type, initially residual-distillation-adapter. |
| \(p\) | Purpose, such as research-only local adaptation or brain-tumour segmentation inference. |
| \(b\) | Pinned M1 base and feature-interface hash. |
| \(d\) | Donor cohort/data version and label schema version. |
| \(q\) | Training code, container, preprocessing, and configuration hash. |
| \(v\) | Policy version and signed policy digest. |
| \(t_{\mathrm{start}},t_{\mathrm{end}}\) | Validity window. |
| \(\rho\) | Rights and prohibitions: local use, optional local fine-tuning, no upload, no aggregation, no seeding, no onward transfer. |

The two independent signatures are:

\[
R_{\mathrm{out}}^{\mathrm{M2}}(g)
\quad\text{and}\quad
R_{\mathrm{in}}^{\mathrm{M2}}(g).
\]

The effective permit at time \(t\) is:

\[
\operatorname{Allow}(g,t)
=
R_{\mathrm{out}}^{\mathrm{M2}}(g)
\land
R_{\mathrm{in}}^{\mathrm{M2}}(g)
\land
\operatorname{PolicyValid}(g,t)
\land
\operatorname{ArtifactMatchesManifest}(g).
\]

Well-formedness requires:

\[
S \subseteq O(i),\qquad
U \subseteq O(j),\qquad
U \cap O(i) = \varnothing.
\]

The compiler must reject an artifact if any modality, label source, training scope, base hash, recipient identity, purpose, or expiry does not match the signed grant.

### 4.4 Relationship to M1 send restrictions

M1 update sharing and one-time M2 artifact release are different actions. Permission for one must not be inferred from permission for the other.

- A positive M1 \(R_{\mathrm{send}}(j,m)\) does **not** automatically authorize M2 artifact export.
- A negative M1 \(R_{\mathrm{send}}(j,m)\) must be treated as a prohibition on M2 export whenever the institution marks it as an all-purpose restriction.
- If the institution explicitly scopes M1 \(R_{\mathrm{send}}\) only to M1 aggregation, it may independently authorize M2 export with \(R_{\mathrm{out}}^{\mathrm{M2}}\). That is a new, visible decision, never a silent override.

This resolves the earlier H2 counterfactual inconsistency. Hospital 2 cannot be treated as a T1ce donor merely by toggling an exposure bit. Its policy must explicitly authorize the precise M2 artifact or it is excluded.

### 4.5 Example grant

    grant_id: H1_to_H3_T1ce_CDRD_v1
    donor: H1
    recipient: H3
    target_subset: [T1, FLAIR]
    source_modalities: [T1, FLAIR, T1ce]
    source_nonowned_modalities_for_H3: [T1ce]
    artifact_type: residual_distillation_adapter
    task_and_purpose: brats_segmentation_local_inference
    base_hash: sha256:<pinned-H3-T1-FLAIR-base>
    base_delivery: canonical_pure_track_release
    preprocessing_hash: sha256:<fixed-preprocessing>
    policy_version: 1
    rights:
      local_inference: true
      local_calibration: true
      local_adapter_finetuning: false
      aggregate: false
      m1_seed: false
      onward_transfer: false
    valid_from: <timestamp>
    expires_at: <timestamp>
    revocation_id: <opaque-id>
    signed_by: [H1_authority, H3_authority]

The donor teacher in this example may use T1, FLAIR, and T1ce. It must not use T2 or be initialized from a full T1/T1ce/T2/FLAIR teacher, because then T2 also shapes the released artifact.

---

## 5. CDRD architecture

### 5.1 Phase 3 begins only after M1 release

For recipient \(i\), choose its final M1 base:

\[
B_{S,i}^{b} =
\left(
\{\operatorname{Encoder}_{s}^{*}\}_{s\in S},
\operatorname{FusionHead}_{S,i}^{b},
\operatorname{Decoder}_{S,i}^{b}
\right).
\]

The base is:

- selected before any M2 artifact is trained or evaluated;
- content-hashed;
- frozen in evaluation mode, including mutable normalization buffers;
- immutable to all M2 optimizers;
- retained as the permanent fallback.

The base exposes a read-only feature interface already computed by its forward pass:

\[
H_{S,i}^{b}(x_S)
=
\operatorname{sg}
\left(
h_S^{(1)},h_S^{(2)},h_S^{(3)},h_S^{(4)},z_S,\ell_B
\right),
\]

where \(h_S^{(r)}\) are fused multi-scale skip/decoder features, \(z_S\) is the fused bottleneck, \(\ell_B\) is the pre-softmax baseline logit map, and \(\operatorname{sg}\) is stop-gradient.

Exposing these tensors is an interface/tapping change only. It adds no input edge to an excluded modality and modifies no M1 parameter, buffer, or decoder computation.

### 5.1a Exact-base availability is an explicit precondition

The donor must train the adapter against the exact feature interface that the recipient will serve. A hash alone is not sufficient if the donor cannot lawfully obtain the corresponding pure M1 checkpoint.

There are only two valid release patterns:

1. **Canonical pure-track release.** The M1 registry publishes a pure-track binary \(B_{S}^{b}\) that both parties are already authorized to receive. The recipient serves exactly that binary.
2. **Recipient-authorized base handoff.** The recipient separately authorizes a read-only, content-addressed copy of its pure \(B_{S,i}^{b}\) to be supplied to the donor or an approved build enclave. No recipient patient data, activations, or labels are transferred. The base handoff itself is recorded in the M2 grant.

If the donor cannot train against the exact base because neither pattern is permitted, the artifact must not be released. A merely “similar” donor track is not an acceptable substitute: it can make the learned residual incompatible and defeats the base-hash safety contract.

### 5.2 Donor-local masked teacher

For grant \(g\), donor \(j\) trains a private teacher:

\[
T_{\omega_j}^{S\cup U}(x_S,x_U)
\longrightarrow
\ell_T.
\]

It uses exactly the modal inputs listed in the grant. Its architecture may use:

- the frozen M1 unimodal encoder outputs for \(S\cup U\);
- a fresh, donor-private fusion module for \(S\cup U\);
- a fresh donor-private decoder.

The teacher parameters \(\omega_j\) are never uploaded, aggregated, routed through M1, or released to the recipient. It is trained locally with:

\[
\mathcal L_{\mathrm{teacher}}
=
\mathcal L_{\mathrm{seg}}
\left(y,\operatorname{softmax}(\ell_T)\right).
\]

For the principal H1-to-H3 case:

\[
S=\{\mathrm{T1},\mathrm{FLAIR}\},
\qquad
U=\{\mathrm{T1ce}\}.
\]

The teacher therefore uses only T1, FLAIR, and T1ce. It must not use T2, a full-track checkpoint with T2 lineage, or a hidden shortcut that carries T2-dependent state.

### 5.3 Donor-trained residual adapter

The released CDRD adapter has parameters \(\phi_{j\rightarrow i}^{g}\) and consumes no non-owned input:

\[
A_{\phi_{j\rightarrow i}^{g}}
\left(
H_{S,i}^{b}(x_S)
\right)
=
\Delta\ell
\in \mathbb R^{C_{\mathrm{cls}}\times H\times W}.
\]

It is a small multi-scale residual network, for example an FPN-like decoder-side module with projected inputs at each of the four frozen scales. Its output is a dense residual logit map, not a prototype token and not a synthetic MRI volume.

The donor's candidate augmented logit is:

\[
\ell_{\mathrm{cand}}
=
\ell_B + \Delta\ell.
\]

Train \(\phi\) only on donor paired samples \((x_S,x_U,y)\):

\[
\begin{aligned}
\mathcal L_{\mathrm{CDRD}}
&=
\lambda_{\mathrm{seg}}
\mathcal L_{\mathrm{seg}}
\left(y,\operatorname{softmax}(\ell_{\mathrm{cand}})\right)\\
&+
\lambda_{\mathrm{KD}}T^2
\operatorname{KL}
\left[
\operatorname{softmax}(\ell_T/T)
\ \Vert\
\operatorname{softmax}(\ell_{\mathrm{cand}}/T)
\right]\\
&+
\lambda_{\Delta}
\left\|
\Delta\ell -
\operatorname{sg}(\ell_T-\ell_B)
\right\|_1\\
&+
\lambda_{\mathrm{TV}}\operatorname{TV}(\Delta\ell).
\end{aligned}
\]

The first term anchors the student to donor labels, the second transfers the full teacher's soft prediction, the third learns the teacher's incremental correction over the exact pure M1 base, and the fourth discourages implausibly noisy dense residuals.

Optional feature-relation losses may be tested within this donor-local paired setting. They are not required for the core design and must not be confused with uncalibrated raw prototype injection.

### 5.4 Why residual logits rather than a replacement decoder

A residual-logit sidecar is deliberately conservative:

- the M1 decoder stays intact;
- the sidecar uses the same complete feature pyramid that the M1 decoder uses;
- the visible baseline logits make the correction's role explicit;
- parameter count can be matched in controls;
- revocation can remove one independent sidecar without touching M1;
- inference uses only \(x_S\).

A feature-residual variant is permissible only if it has the exact same multi-scale decoder interface and a separate private decoder warm-started from the pinned M1 decoder. It should be treated as an ablation after the logit-residual version is established.

### 5.5 Recipient-private calibration

After receiving a valid artifact, recipient \(i\) may train a small per-artifact gate/calibrator \(\gamma_i^g\) on its own \(S\)-only labels:

\[
\alpha_{\gamma_i^g}
\left(H_{S,i}^{b},\Delta\ell\right)
\in [0,1]^{C_{\mathrm{cls}}\times H\times W}.
\]

The augmented output becomes:

\[
\ell_{\mathrm{aug}}
=
\ell_B+
\alpha_{\gamma_i^g}\odot\Delta\ell,
\qquad
\hat y_{\mathrm{aug}}
=
\operatorname{softmax}(\ell_{\mathrm{aug}}).
\]

Default policy:

- freeze the donor adapter;
- train only \(\gamma_i^g\) and a small calibration bias;
- never upload either object.

An optional policy may permit recipient fine-tuning of \(\phi\). If used, every modified adapter parameter becomes a dependent M2 object and must be deleted or rebuilt on revocation.

### 5.6 Multiple authorized donors

For multiple artifacts, preserve separate modules:

\[
\ell_{\mathrm{aug}}
=
\ell_B+
\sum_{g\in G_i}
\alpha_{\gamma_i^g}\odot
\Delta\ell_g.
\]

Do not average independently trained neural-adapter weights. Their parameter spaces are not generally aligned. A local combination gate is permissible, but it must be rebuilt from a clean checkpoint if any source grant is revoked.

---

## 6. Full CDRD lifecycle

### 6.1 State machine

    M1_PHASE_1
        -> M1_PHASE_2
        -> M1_RELEASED(base_hash b)
        -> M2_POLICY_COMPILED
        -> DONOR_TEACHER_TRAINED
        -> DONOR_ADAPTER_BUILT
        -> PRIVACY_RELEASE_REVIEWED
        -> ARTIFACT_RELEASED
        -> RECIPIENT_CALIBRATED
        -> VALIDATION_ACCEPTED or FALLBACK_ONLY
        -> DEPLOYED
        -> EXPIRED or REVOKED

No M2 state may transition backwards into M1 Phase 1 or Phase 2.

### 6.2 Donor procedure

    Input: valid grant g, pinned recipient-compatible base B_S,i^b,
           donor paired data D_j^(S union U)

    1. Verify Allow(g, now), base hash, preprocessing hash, and policy scope.
    2. Instantiate a fresh donor-private teacher T_omega^(S union U).
    3. Train the teacher only on modalities listed in g.
    4. Run the pinned base B_S,i^b on donor x_S in evaluation mode.
    5. Train the residual adapter A_phi on detached base feature taps,
       teacher logits, and donor labels using L_CDRD.
    6. Execute release privacy checks and donor-held-out utility checks.
    7. Package only the approved adapter and signed manifest.
    8. Encrypt the package to the named recipient; record release in the ledger.

### 6.3 Recipient procedure

    Input: artifact package P_g, recipient S-only data D_i^S,
           immutable base B_S,i^b

    1. Verify signature, recipient binding, policy validity, expiry,
       modality lineage, purpose, and exact base/interface hashes.
    2. Reject P_g if any check fails. Do not deserialize it into an M1 process.
    3. Load P_g into an isolated M2 sidecar process or namespace.
    4. Freeze B_S,i^b and the default donor adapter.
    5. Train only local gate/calibration parameters on recipient validation-train data.
    6. Evaluate pure M1 and augmented outputs on a held-out recipient validation set.
    7. Deploy augmented output only if the acceptance rule passes.
       Otherwise retain M1 fallback only and retain or delete the artifact
       according to policy.
    8. Never upload, aggregate, use as an M1 seed, or forward the artifact.

### 6.4 Inference

    x_S -> immutable M1 base B_S,i^b -> (H_S, baseline logits)
                                         -> valid CDRD sidecar
                                         -> augmented logits
                                         -> accepted deployment selector

The deployment selector chooses the pre-approved augmented route or the pure M1 route. It must not silently switch per patient based on unvalidated convenience metrics. If a dynamic uncertainty gate is researched later, it requires a separate prospective validation protocol.

---

## 7. Formal containment and lineage guarantees

### 7.1 Shared-state non-propagation theorem

Let \(\theta_B\) be every M1 parameter and mutable buffer in the pinned base. Let \(\Phi\) contain all donor teacher, adapter, recipient gate, and calibration state. In M2:

\[
H=\operatorname{sg}(B_{\theta_B}(x_S)),
\]

and the M2 optimizer updates only \(\Phi\):

\[
\Phi^{t+1}
=
\operatorname{Optimizer}_{\mathrm{M2}}
\left(
\Phi^t,\nabla_{\Phi}\mathcal L_{\mathrm{M2}}
\right).
\]

Under the following enforceable assumptions:

1. \(\theta_B\) and all M1 buffers are frozen and evaluation-only;
2. M2 receives detached feature taps;
3. M2 and M1 have disjoint parameter sets, optimizers, checkpoints, and mutable buffers;
4. M2 artifacts, metrics, optimizer state, and calibration weights are excluded from every M1 upload, aggregation, seed, and checkpoint-selection path;
5. the M1 base hash is selected independently of M2 results;

then:

\[
\frac{\partial\mathcal L_{\mathrm{M2}}}{\partial\theta_B}=0,
\qquad
\theta_B^{t+1}=\theta_B^t.
\]

Therefore, a valid M2 run cannot change M1 shared state or any other client's M1 model through the M2 route.

The system should claim **shared-state non-propagation**, not automatic byte identity. Byte-for-byte equality additionally requires deterministic kernels, fixed random-number state, fixed scheduling, and no hidden side effects. Implement hash checks before and after M2 and deterministic prediction checks on a fixed probe set.

### 7.2 Lineage rule

For the pure base:

\[
\operatorname{Lineage}(B_{S,i}^b)\subseteq S.
\]

For an M2 artifact:

\[
\operatorname{Lineage}(A_{\phi_{j\rightarrow i}^{g}})
\subseteq
S\cup U\cup\{\text{donor labels}\}.
\]

The label source is explicit because teacher and adapter task losses are label-shaped. This is a lineage statement, not a claim that the label itself is a modality.

Add the M2 firewall invariant:

> Any object whose lineage is not a subset of a target M1 track's modality set is permanently ineligible for M1 aggregation, M1 cold-start seeding, M1 model selection, M1 routing, and M1 provenance inheritance.

The deployed composite deliberately has cross-boundary lineage. It must be labelled as a consented M2 augmentation, not as a pure M1 track.

### 7.3 What non-harm is actually supportable

Other clients and the pure M1 route are structurally protected under the theorem's assumptions. The recipient is not guaranteed to improve.

The correct safeguard is:

1. preserve the M1 fallback;
2. use a preregistered validation acceptance rule;
3. deploy the sidecar only if the rule passes;
4. report both accepted and rejected artifacts.

This is validation-selected noninferiority, not a universal Pareto theorem.

---

## 8. Artifact release, privacy, and governance

### 8.1 Treat artifacts as sensitive derived health-data objects

No model-derived object is automatically harmless because it is small or contains no raw image. A prototype, feature statistic, or adapter can expose membership, cohort properties, or reconstructible information. FedHide directly treats true prototype sharing as a leakage problem rather than assuming it is private; see [FedHide](https://eccv.ecva.net/virtual/2024/poster/2697).

CDRD makes no privacy guarantee merely by avoiding raw data transfer.

### 8.2 Minimum artifact manifest

Every package must contain:

- opaque artifact identifier and content hash;
- donor and recipient pseudonymous identifiers or commitments;
- source modality set and label/cohort lineage;
- target subset \(S\), task, purpose, and allowed rights;
- exact M1 base, architecture, preprocessing, and feature-interface hashes;
- policy digest, two signatures, issue date, expiry, and revocation identifier;
- code/configuration hash and training-risk summary;
- privacy release mechanism and accounting if a formal privacy claim is made;
- artifact-specific validation summary;
- deletion/revocation instructions.

The ledger records at least:

    POLICY_GRANTED
    ARTIFACT_BUILT
    PRIVACY_REVIEW_PASSED or PRIVACY_REVIEW_FAILED
    ARTIFACT_RELEASED
    ARTIFACT_ACCEPTED or ARTIFACT_REJECTED
    M2_BRANCH_TRAINED
    VALIDATION_ACCEPTED or FALLBACK_ONLY
    DEPLOYED
    EXPIRED
    REVOKED
    ROLLBACK_OR_DELETION_ACK

### 8.3 Threat model and controls

The minimum threat model is an honest-but-curious relay/server and recipient: they follow routing rules but may inspect released artifacts. A stronger clinical deployment must also consider malicious recipients, model substitution, replay, and artifact exfiltration.

Required controls:

- authenticated, encrypted artifact transport;
- recipient-bound encryption or key wrapping;
- signed manifests and anti-replay identifiers;
- allowlist rejection at the M1 aggregation and seed boundaries;
- model inversion, membership-inference, and site-property inference evaluation for adapters and prototype baselines;
- release-rate limits and overlap accounting to limit differencing attacks;
- secure aggregation only when a true aggregate is constructed; it is not a substitute for a privacy argument;
- patient-level clipping and a documented DP-SGD or release mechanism if a formal \((\epsilon,\delta)\) claim is made.

For aggregate artifacts, additionally require:

- at least \(k\) eligible source institutions;
- minimum patient-level and per-class support, not merely voxel counts;
- rare-class suppression where support is inadequate;
- a recipient-specific aggregate assembled only from grants valid for that recipient.

A single-donor H1-to-H3 artifact can be allowed under explicit institutional authority, but must never be called anonymous or aggregate-private.

### 8.4 Revocation and expiry

Sidecar isolation makes revocation tractable:

1. Authority revokes \(g\), or its expiry is reached.
2. The relay stops distribution and sends a signed revocation event.
3. Recipient disables the augmented route immediately and serves the immutable M1 fallback.
4. Recipient deletes the artifact, every fine-tuned copy, and every gate/calibrator that depended on it; alternatively it restores a clean pre-M2 checkpoint.
5. Recipient records a rollback/deletion acknowledgement in the ledger.
6. If a donor patient withdrawal requires an artifact rebuild, the donor rebuilds without that patient and affected recipient sidecars are rebuilt or kept disabled.

Deleting the received file alone does not establish unlearning if its influence has been mixed into another local module. That is why CDRD keeps each source in a separately removable sidecar and forbids it from entering M1. The importance of revocation readiness in clinical AI is also emphasized in [Machine unlearning as a governance imperative for clinical AI](https://pubmed.ncbi.nlm.nih.gov/42471427/).

---

## 9. Concrete H1-to-H3 scenario

### 9.1 Policies

| Hospital | Owned modalities | M1 role | M2 R4 role |
| :--- | :--- | :--- | :--- |
| H1 | T1, T1ce, T2, FLAIR | Full M1 participant and optional pure-track contributor | Candidate donor of a T1ce-derived artifact to H3. |
| H2 | T1, T1ce, T2 | M1 send-gated case; T1ce may be withheld from M1 aggregation | Not a T1ce donor unless it signs a separate M2 export grant with the required scope. |
| H3 | T1, FLAIR | Pure M1 target track \(S_3\) | Named recipient that can opt in to T1ce but not T2. |
| H4 | T1, T1ce, T2 | Delayed M1 joiner | Separate future donor/recipient only through its own grants. |

The H3 grant is:

\[
S_3=\{\mathrm{T1},\mathrm{FLAIR}\},
\qquad
U=\{\mathrm{T1ce}\}.
\]

H3 does not authorize T2. A teacher using T2, a full-modality H1 checkpoint, or a fused artifact with undisclosed T2 lineage fails policy compilation.

### 9.2 Donor execution

H1 trains:

\[
T_{\mathrm{H1}}^{\{\mathrm{T1},\mathrm{FLAIR},\mathrm{T1ce}\}}.
\]

It runs the pinned H3-compatible pure M1 base \(B_{S_3,\mathrm{H3}}^b\) over H1's T1/FLAIR inputs and distils the teacher correction into:

\[
A_{\mathrm{H1}\rightarrow\mathrm{H3}}^{\mathrm{T1ce}\mid\{\mathrm{T1},\mathrm{FLAIR}\}}.
\]

Only after the signed grant and release checks does H1 transfer the adapter package.

### 9.3 Recipient execution

H3 uses:

\[
x_{\mathrm{T1}},x_{\mathrm{FLAIR}}
\longrightarrow
B_{S_3,\mathrm{H3}}^b
\longrightarrow
A_{\mathrm{H1}\rightarrow\mathrm{H3}}.
\]

T1ce is absent at H3 training and inference. The M1 base stays exactly as it was before M2.

### 9.4 BraTS caveat

BraTS enhancing tumour (ET) is defined using contrast-enhanced T1Gd/T1ce information. The official BraTS description identifies ET as label 4 and describes its relationship to T1Gd; see [BraTS data description](https://www.med.upenn.edu/cbica/brats2021/). An ET gain alone therefore does not prove clinically general missing-modality recovery or label-pure transfer.

Report whole tumour, tumour core, and ET separately. Frame the result as a consented input/model-lineage study under a BraTS benchmark, not as proof that an unavailable clinical sequence has been reconstructed.

---

## 10. Required evaluation design

### 10.1 Primary causal question

The primary question is not merely whether CDRD beats bare M1:

> Does a T1ce-authorized donor artifact improve H3 beyond an identical-capacity local personalization route that has no T1ce-derived information?

All conditions must use the same frozen M1 base, patient-level splits, recipient tuning budget, parameter count where applicable, and test protocol.

### 10.2 Required conditions

| ID | Condition | What it controls |
| :--- | :--- | :--- |
| E0 | Frozen pure M1 base \(B_{S,i}^b\) | Compliance-preserving baseline. |
| E1 | M1 plus matched-capacity H3-local residual sidecar, no donor artifact | Extra local capacity and recipient labels. |
| E2 | M1 plus same-shape, same-norm random artifact | Artifact shape and initialization. |
| E3 | H1 donor adapter trained with an \(S\)-only teacher | Donor cohort, labels, and transfer pipeline without T1ce. |
| E4 | Proposed CDRD adapter trained with \(S\cup\{\mathrm{T1ce}\}\) teacher | Authorized T1ce-derived transfer. |
| E5 | T1ce-pairing shuffle or class/teacher permutation control | Whether actual cross-modal pairing and semantic structure matter. |
| E6 | Existing raw cross-prototype CrossAbsorbHead, only after alignment pre-check | Low-bandwidth legacy ablation. |
| E7 | Full \(S\cup\{\mathrm{T1ce}\}\) teacher/oracle | Non-deployable upper bound. |

If an existing literature baseline is adapted, label it clearly as compliant or non-compliant with the same artifact policy. MFCPL, prototype-conditioned synthesis, and missing-modality distillation methods are performance comparisons, not automatic consent-governance baselines.

### 10.3 Split and statistical protocol

- Split at the patient level, never at the 2D slice level.
- Keep donor and recipient patients disjoint.
- Use a held-out recipient validation set for all model/route selection and an untouched recipient test set for final reporting.
- Use multiple random federation partitions and random seeds.
- Use paired per-patient effects and stratified or hierarchical bootstrap confidence intervals.
- Pre-register one primary endpoint and multiplicity handling before reading the test set.
- Report every recipient separately. Do not average away a harmed site.

Report:

- Dice for whole tumour, tumour core, and ET;
- HD95;
- calibration such as ECE or Brier score;
- sensitivity, specificity, lesion-level failures, and failure-case review;
- adaptation compute, artifact size, release latency, and privacy budget if applicable;
- policy, revocation, hash, and leakage-audit results.

Artificial BraTS silo splits show feasibility only. They do not establish clinical transportability across genuine scanner, protocol, and population shifts.

### 10.4 Deployment acceptance rule

Let \(Q\) be a preregistered recipient metric and \(\delta\) a clinician-approved noninferiority margin. Deploy only if:

\[
\operatorname{LCB}_{95\%}
\left[
Q(\mathrm{CDRD})-Q(\mathrm{M1})
\right]
\geq -\delta
\]

for every mandatory safety endpoint, and only claim superiority when the held-out benefit also exceeds the matched-capacity and randomized-artifact controls.

The exact margins and primary endpoint must be chosen before result inspection. A failed gate is a valid result: retain the M1 fallback and report that the R4 artifact did not justify deployment.

### 10.5 Mandatory systems tests

| Test | Pass condition |
| :--- | :--- |
| Source denial | No artifact is built or released. |
| Recipient denial | No artifact is delivered or loaded. |
| Unapproved modality | Policy compiler rejects teacher/artifact lineage containing it. |
| Base-hash mismatch | Recipient rejects artifact before execution. |
| M1 non-propagation | Shared-state hashes and fixed-probe M1 predictions are unchanged. |
| Upload firewall | Serialization and server allowlists reject all M2 object types. |
| Prototype geometry pre-check | A prototype path must beat shuffled paired controls on held-out cross-modal correspondence after a learned calibration; centroid distance alone is insufficient. |
| Privacy evaluation | Membership, inversion/reconstruction, and site-property inference are reported for released artifacts. |
| Revocation | Augmented route is disabled; rollback output equals pure M1; dependent objects are deleted/rebuilt. |
| Donor ablation | E4 is compared with E3 to isolate the authorized modality's contribution. |

---

## 11. Legacy prototype path: allowed only as an ablation

The earlier CrossAbsorb design can remain useful as a research baseline if all of the following conditions are met:

1. it is never called the canonical R4 solution;
2. it uses the same artifact-grant policy, lineage, privacy review, expiry, and revocation mechanics as CDRD;
3. it does not claim M1 has already learned cross-modal coordinates;
4. it first learns a post-M1, donor-local projection/calibration map from paired data;
5. held-out spatial correspondence or retrieval beats shuffled controls;
6. it includes matched-capacity, random-prototype, class-permuted, and own-modality prototype controls;
7. any fresh decoder receives the full multi-scale M1 interface.

A raw L2 or cosine distance between independently trained M1 modality prototypes is not a viability test. A recent modality-invariant MRI study found that very close cross-modality embeddings did not automatically improve lesion segmentation, reinforcing that alignment alone is not evidence of dense clinical utility; see [Large-scale modality-invariant representation learning for MRI](https://arxiv.org/html/2511.11311).

---

## 12. Integration plan

### 12.1 What remains exactly M1

| M1 component | M2 treatment |
| :--- | :--- |
| Phase-1 encoder/prototype training | No change. |
| Phase-2 track training and routing | No change. |
| M1 send/receive policy semantics | No change. |
| Pure track lineage rule | No change; M2 artifacts are explicitly ineligible as M1 seeds. |
| M1 aggregation | No M2 parameter or statistic may enter it. |
| Frozen M1 fallback | Preserved and deployable at all times. |

### 12.2 New M2 components

| Component | Responsibility |
| :--- | :--- |
| M2 policy compiler | Validates bilateral artifact grants and blocks policy/scope mismatches. |
| Frozen-base registry | Publishes recipient-compatible base/interface hashes. |
| Donor-local masked teacher | Learns \(S\cup U\) benefit without federation. |
| CDRD adapter trainer | Distils the teacher correction into an \(S\)-only dense residual sidecar. |
| Artifact registry/relay | Delivers signed encrypted packages and records state; never aggregates them. |
| Recipient-private calibration | Selects whether and how strongly to use a valid artifact. |
| M2 provenance and revocation ledger | Records artifact lifecycle, dependencies, expiry, and deletion acknowledgements. |
| M1 firewall | Rejects M2 types at aggregation, seeding, routing, and checkpoint selection boundaries. |

### 12.3 Implementation order

1. Add immutable M1 snapshot and feature-interface hashing.
2. Add an M1 firewall test before adding any transfer method.
3. Implement the policy schema and signed manifest validator.
4. Implement donor-private \(S\cup U\) teacher training.
5. Implement residual-logit CDRD with only frozen base taps.
6. Implement recipient-local gate and M1 fallback.
7. Implement policy, hash, revocation, and serialization tests.
8. Run E0-E5 before prototype or multi-donor extensions.
9. Perform privacy attacks and release review before any external artifact exchange.

---

## 13. Research positioning

The contribution should be described narrowly and accurately:

> A bilateral, artifact-level, modality/purpose/version-bounded governance layer for donor-specific detached missing-modality transfer, with frozen-base compatibility, provenance, revocation containment, and a shared-state non-propagation guarantee.

Avoid these claims:

- “No prior work has a policy matrix.” PoliFL already supports heterogeneous privacy policies; see [PoliFL](https://arxiv.org/abs/2003.06612).
- “Prototypes are private because they are small.” Prototype leakage is an active concern.
- “FedAS proves the M1 STABLE lifecycle.” FedAS addresses personalized/shared inconsistency and stragglers, not this particular frozen-base lifecycle; see [FedAS](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_FedAS_Bridging_Inconsistency_in_Personalized_Federated_Learning_CVPR_2024_paper.html).
- “R4 always improves the recipient.” The correct statement is validation-selected optional benefit.
- “The released artifact is T1ce-only.” The exact claim is that its declared lineage is \(S\cup\{\mathrm{T1ce}\}\) plus source labels and no undisclosed modality.

Relevant research anchors:

- [MFCPL](https://arxiv.org/abs/2401.13898): cross-modal prototype work that explicitly adds alignment.
- [Cross-modal distillation for missing MRI sequences](https://pubmed.ncbi.nlm.nih.gov/34941496/): supports teacher-to-limited-input transfer for brain-tumour segmentation.
- [PLOT for federated incomplete multimodal brain-tumour segmentation](https://pubmed.ncbi.nlm.nih.gov/40030851/): supports distillation as a serious missing-modality mechanism, while lacking CDRD's artifact governance and non-propagation design.
- [pFedDKS](https://researchportal.hkust.edu.hk/en/publications/pfeddks-detached-knowledge-sharing-for-personalized-federated-lea/): supports detached knowledge sharing in personalized FL.
- [FedHide](https://eccv.ecva.net/virtual/2024/poster/2697): shows why learned prototypes require leakage analysis.
- [ProMoE-FL](https://arxiv.org/abs/2607.06633): a recent prototype-conditioned missing-feature synthesis comparison, not a replacement for consent-bounded transfer.

---

## 14. Final recommendation

Do not merge the current raw prototype CrossAbsorbHead as the canonical M2 update.

Adopt M2 as a third, post-M1 phase:

\[
\text{M1 Phase 1}
\longrightarrow
\text{M1 Phase 2}
\longrightarrow
\text{immutable M1 release}
\longrightarrow
\text{M2 CDRD policy-gated sidecar}.
\]

The complete recommended R4 path is:

1. bilateral artifact grant;
2. exact recipient-base hash;
3. donor-private teacher on exactly \(S\cup U\);
4. donor-trained dense residual adapter operating on frozen \(S\)-only M1 features;
5. recipient-private calibration and M1 fallback;
6. no aggregation, seeding, or onward transfer;
7. artifact-level provenance, privacy review, expiry, and revocation;
8. matched-capacity and placebo controls before any performance claim.

This is the strongest solution compatible with your M1 architecture: it preserves the original purity guarantees while allowing a client to explicitly and audibly opt into a bounded cross-boundary benefit.
