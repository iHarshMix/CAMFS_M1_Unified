# Gap 6 / R4 Feasibility Analysis (v1 Initial Draft)
## Can Cross-Boundary Knowledge Absorption Be Solved Within the Existing CAMFS Architecture?

---

## 1. The Core Challenge — Precisely Stated

Hospital 3 owns $O(3) = \{T1, \text{FLAIR}\}$, wants $\text{Recv}(3) = \{T1, \text{FLAIR}, \text{T1ce}\}$.

**The fundamental tension:** Hospital 3 will *never* have T1ce input data ($x_{\text{T1ce}}$ does not exist for its patients). Yet it wants its model — its fusion head, decoder, and segmentation output — to be *shaped by* T1ce-derived knowledge from the federation. How can knowledge from a modality you'll never have as input improve your model?

---

## 2. What the Existing Architecture Provides (and Where It Blocks)

### 2.1 The Enablers (Things That Already Work in Our Favor)

| Existing Property | Why It Helps for R4 |
| :--- | :--- |
| **Shared 256-D language** (§5.2) | Phase 1 contrastive alignment forces $\text{Proto}\_{\text{T1}}^c \approx \text{Proto}\_{\text{T1ce}}^c$ — all encoders map the same anatomy to the same region of $\mathbb{R}^{256}$. This means T1ce prototype knowledge is **expressible in the same language** Hospital 3's T1/FLAIR features already speak. |
| **Frozen Proto Bank** (§5.2) | After Phase 1, $\text{Proto}\_{\text{T1ce}}^c$ is frozen, public, and tiny (4 vectors × 256-D = 1 KB). Transmitting it to Hospital 3 is trivially cheap and carries no raw patient data — only class-level distributional information. |
| **Frozen Encoders + detach()** (§7) | Encoders are immutable after the freeze. Any R4 mechanism operates entirely in Phase 2, at the fusion/decoder layer. No encoder needs to change. |
| **Personalized Local Head** (§9.5b) | The architecture already has a precedent for a **private, never-aggregated** component that trains on local data only. This pattern can be extended. |
| **FusedProto per track** (§5.4) | Each track already maintains its own fused prototype bank. A richer track's $\text{FusedProto}\_{S\_1}^c$ encodes what 4-modality fusion looks like for each class — potentially transferable knowledge. |

### 2.2 The Blockers (Where the Architecture Structurally Prevents R4)

| Blocker | Why It Prevents R4 | Severity |
| :--- | :--- | :--- |
| **No T1ce input slot** | $\text{FusionHead}\_{S\_3}$ has exactly 2 input slots (T1, FLAIR). There is literally no place to plug a T1ce tensor. Even if Hospital 3 received $\text{Encoder}\_{\text{T1ce}}^*$, it has no $x\_{\text{T1ce}}$ to feed through it. | **Architectural** — the fusion head dimensions are fixed by the track's modality set |
| **Track hard isolation** (§5.4) | "No parameter or gradient ever crosses between tracks" — pulling parameters from Track $S_1$ into Track $S_3$ violates this invariant | **Structural invariant** — but was designed for M1's stricter constraint |
| **Purity Lemma** (§12.2) | Proves Track $S'$ parameters carry zero signal from $m \notin S'$. Cross-boundary absorption *intentionally* breaks this for consented modalities. | **By design** — R4 explicitly relaxes this for $\text{Recv}\_{\text{cross}}$ |
| **Cross-track prototype alignment rejected** (§16) | M1 explicitly rejected cross-track prototype alignment because "it would reopen cross-modal leaks." | **Policy decision** — valid for M1, but R4 deliberately reopens this channel under consent governance |

---

## 3. Six Solution Approaches — Evaluated

### Approach A: Pull a Larger Track's Parameters Directly

**Idea:** Hospital 3 pulls $\text{FusionHead}\_{S_1}$ and $\text{Decoder}\_{S_1}$ from Track $S_1 = \{T1, T1ce, T2, \text{FLAIR}\}$ and uses them at inference.

**Why it fails:**
- $\text{FusionHead}\_{S\_1}$ expects 4 modality inputs. Hospital 3 has 2. The cross-attention dimensions don't match.
- Zero-filling or mean-filling the missing modality slots produces garbage: the attention weights were trained assuming real feature maps, not constants.
- The decoder was trained on 4-modality fused features and will misinterpret 2-modality fused features.

> **Verdict: ❌ REJECTED.** Dimensionality mismatch is fatal. Not fixable without retraining.

---

### Approach B: Server-Side Knowledge Distillation (Logit/Feature Transfer)

**Idea:** Hospital 1 (full-modality) computes segmentation predictions using Track $S_1$. These predictions (soft logits) are shared with Hospital 3 as teacher targets. Hospital 3 trains its Track $S_3$ model to mimic the teacher's output.

**Analysis:**
- Hospital 3 cannot compute teacher outputs itself (missing T1ce/T2 inputs), so the teacher must run at another site or on the server.
- Requires **shared proxy data** or synthetic data that both teacher and student can process — adds significant complexity.
- In a federated setting, sharing logits on patient data raises privacy concerns (logit leakage attacks).
- This is essentially FedMD / FedDF territory — well-studied but adds a whole new mechanism.

**Architectural impact:** Requires a new distillation protocol, proxy dataset management, and logit communication — this is a significant architectural addition, not a small extension.

> **Verdict: ⚠️ POSSIBLE but HEAVY.** Requires substantial new infrastructure. Not "within the existing architecture."

---

### Approach C: Prototype-Mediated Cross-Modal Regularization (⭐ MOST PROMISING)

**Idea:** During Phase 2, Hospital 3 adds an additional loss term that pulls its fused representations toward the frozen $\text{Proto}\_{\text{T1ce}}^c$ vectors from Phase 1. No T1ce input data is needed — only the prototype vectors (which are already frozen, public, 256-D, and in the shared language).

**How it works:**

```python
# Hospital 3, Phase 2, per round:
z_T1    = detach(Encoder_T1*(x_T1))
z_FLAIR = detach(Encoder_FLAIR*(x_FLAIR))
z_S3    = FusionHead_S3({z_T1, z_FLAIR})
y_hat   = Decoder_S3(z_S3)

# Standard losses (unchanged from M1)
L_task        = DiceCE(y, y_hat)
L_fusion_align = InfoNCE(z_S3, {FusedProto_S3^c})

# NEW: Cross-absorb loss for consented non-owned modalities
# For each m in Recv_cross(i):
L_cross_absorb = InfoNCE(z_S3, {Proto_T1ce^c})  # frozen Phase-1 prototypes

L_total = L_task + λ2 · L_fusion_align + λ3 · L_cross_absorb
```

**Why it works (the shared language is the key):**
- Phase 1 forced $\text{Proto}\_{\text{T1}}^c \approx \text{Proto}\_{\text{T1ce}}^c$ — they're in the same 256-D region.
- But they're not *identical*. $\text{Proto}\_{\text{T1ce}}^c$ captures subtle distributional differences specific to T1ce contrast enhancement — particularly for the ET (enhancing tumor) class, where T1ce provides sharper boundary information.
- By pulling $z\_{S\_3}(p)$ toward $\text{Proto}\_{\text{T1ce}}^c$ (in addition to $\text{FusedProto}\_{S\_3}^c$), Hospital 3's fusion head learns to produce representations that are compatible with what T1ce "sees" for each class.
- This is not hallucinating T1ce features — it's asking "what would T1ce expect this anatomy to look like?" and nudging the T1/FLAIR fusion toward that target.

**Architectural impact:** Minimal. Adds one loss term. No new components. No new communication channels (prototypes are already transmitted at Phase 1 end). The consent gate is just: "only add $\mathcal{L}\_{\text{cross-absorb}}$ for $m \in \text{Recv}\_{\text{cross}}(i)$."

**Concern — Purity Lemma interaction:** This shapes the federated track $S_3$'s parameters with T1ce signal. If $S_3$ is aggregated, *all* participants of Track $S_3$ get T1ce-infused weights. This is fine IF all Track $S_3$ participants consented to T1ce absorption. If one participant didn't, we need per-client personalized aggregation — or we need to push the cross-absorb to a local-only component (see Approach C+D hybrid below).

> **Verdict: ✅ STRONG CANDIDATE.** Minimal architectural change, leverages the existing shared language, consent-gated naturally. Main open question: does prototype-level alignment actually transfer enough knowledge to improve task performance?

---

### Approach D: Local Cross-Absorb Head (Private Refinement Layer)

**Idea:** Keep the federated Track $S_3$ completely pure (M1 Purity Lemma holds unchanged). Add a **private, never-aggregated** refinement layer *after* the federated track that uses non-owned prototypes.

**Architecture:**

```python
# Hospital 3, Phase 2, per round:

# ── Federated path (pure, unchanged, aggregated normally) ──
z_T1    = detach(Encoder_T1*(x_T1))
z_FLAIR = detach(Encoder_FLAIR*(x_FLAIR))
z_S3    = FusionHead_S3({z_T1, z_FLAIR})        # federated, pure

# ── Local cross-absorb path (private, never aggregated) ──
# CrossAbsorbHead attends z_S3 features against Proto_T1ce^c
z_S3_enhanced = CrossAbsorbHead(z_S3, {Proto_T1ce^c : c})   # local only
y_hat = Decoder_local(z_S3_enhanced)                          # local only

# Loss trains both paths, but only FusionHead_S3 updates are uploaded
L_total = L_task(y, y_hat) + λ2 · L_fusion_align(z_S3, FusedProto_S3^c)
# backprop flows through Decoder_local → CrossAbsorbHead → z_S3 → FusionHead_S3
# but ONLY FusionHead_S3/Decoder_S3 deltas are pushed to server
```

**Wait — there's a problem.** If backprop flows through the local cross-absorb head all the way to FusionHead_S3, then the FusionHead_S3 updates that get uploaded to the server ARE shaped by the cross-absorb head (and thus by Proto_T1ce). This contaminates the federated track.

**Fix: detach the boundary:**

```python
z_S3_detached = detach(z_S3)                                  # sever gradient
z_S3_enhanced = CrossAbsorbHead(z_S3_detached, {Proto_T1ce^c})
y_hat_local   = Decoder_local(z_S3_enhanced)

# Federated loss (trains FusionHead_S3, Decoder_S3 — pure)
L_fed   = L_task(y, y_hat_fed) + λ2 · L_fusion_align(z_S3, FusedProto_S3^c)

# Local loss (trains CrossAbsorbHead, Decoder_local — never uploaded)
L_local = L_task(y, y_hat_local)

# Two separate backward passes, no gradient leakage between them
```

This is cleaner: the federated track remains completely pure, the local cross-absorb head trains independently on local data, and at inference Hospital 3 uses the enhanced path.

**Concern:** CrossAbsorbHead + Decoder_local are trained only on Hospital 3's local data — potentially limited for a small hospital (20% of patients). But this is a performance concern, not an architectural one.

> **Verdict: ✅ STRONG CANDIDATE.** Preserves all M1 purity guarantees for the federated track. Adds cross-boundary absorption as a strictly local, private extension. Pattern matches the existing "Personalized Local Head" (§9.5b). Requires `detach()` boundary to prevent gradient leakage into federated parameters.

---

### Approach E: Cross-Track Prototype Alignment (Rehabilitated Under Consent)

**Idea:** M1 rejected cross-track prototype alignment because it "would reopen cross-modal leaks." Under M2, reopening this channel *under explicit consent* is exactly the point of R4.

**Mechanism:** Server computes alignment between $\text{FusedProto}\_{S\_3}^c$ and $\text{FusedProto}\_{S\_1}^c$ and broadcasts a regularization target to Track $S_3$'s participants.

**Analysis:**
- $\text{FusedProto}\_{S\_1}^c$ encodes what 4-modality fused representations look like for each class — richer than the unimodal $\text{Proto}\_{\text{T1ce}}^c$ used in Approach C.
- But this creates a server-side cross-track information flow, which is architecturally more invasive.
- Direction is one-way: $S_3$ pulls from $S_1$, not vice versa. $S_1$ is unaffected.

> **Verdict: ⚠️ POSSIBLE.** Strictly more powerful than Approach C (fused proto > unimodal proto), but more invasive. Could be an optional enhancement on top of C or D.

---

### Approach F: Feature Hallucination (Synthesize Missing Modality)

**Idea:** Hospital 3 learns a local module that *hallucinates* T1ce-like features from T1/FLAIR features, then feeds these hallucinated features into a larger track's fusion head.

**Analysis:**
- Requires a hallucination network: $\hat{z}\_{\text{T1ce}} = \text{Hallucinate}(z\_{\text{T1}}, z\_{\text{FLAIR}})$
- The shared language makes this somewhat plausible — all encoders project into the same $\mathbb{R}^{256}$ space, so the "target space" for hallucination is well-defined.
- But training the hallucination network requires paired data (T1/FLAIR + T1ce ground truth), which Hospital 3 doesn't have.
- Could use the frozen prototypes as targets: "hallucinate a feature map whose class-level mean matches $\text{Proto}\_{\text{T1ce}}^c$" — but this produces spatially uniform/blurry hallucinations.

> **Verdict: ❌ REJECTED for now.** Requires paired data Hospital 3 doesn't have. Prototype-only hallucination is too coarse. Much more complex than C or D for uncertain benefit.

---

## 4. Recommended Solution: Approach C + D Hybrid

The cleanest, most architecturally compatible solution combines Approaches C and D:

### Layer 1: Federated Track (Approach C — Prototype Regularization)

Add $\mathcal{L}\_{\text{cross-absorb}}$ to the federated track's loss, pulling fused features toward non-owned prototypes. This operates at the **federated level** and benefits from aggregation.

**Consent requirement:** ALL participants on the track must consent to the same cross-boundary modalities for this to be used. If consent is not unanimous, fall back to Approach D.

### Layer 2: Local Refinement (Approach D — Private Cross-Absorb Head)

Add a private, `detach()`-separated refinement head that uses non-owned prototypes for client-specific cross-boundary absorption. This operates at the **local level** and never touches the federation.

**Always available** regardless of what other track participants consent to.

### Why the Hybrid Works

| Concern | How It's Handled |
| :--- | :--- |
| Federated track purity | Approach C: all participants consent → track carries consented cross-signal. Approach D: `detach()` boundary → federated track stays pure. |
| Single-client consent | Approach D handles it — purely local, no federation impact |
| Unanimous consent | Approach C handles it — federated cross-absorb, benefits from aggregation |
| No T1ce input data needed | Both approaches use frozen $\text{Proto}\_{\text{T1ce}}^c$ only — no raw modality data |
| Auditability (R2) | Consent is in the policy matrix. Provenance Ledger records which prototypes were absorbed. |
| Backward compatibility | When $\text{Recv}\_{\text{cross}}(i) = \emptyset$ for all clients, both loss terms vanish → exactly M1 |

---

## 5. What Changes in the Architecture (Summary)

| Component | Change Type | Description |
| :--- | :--- | :--- |
| **Well-formedness constraints** (§4) | ✅ Already updated | $\text{Recv}(i) \subseteq \mathcal{M}$, partitioned into $\text{Recv}\_{\text{own}}$ and $\text{Recv}\_{\text{cross}}$ |
| **Phase 1** (§6) | ❌ No change | Unimodal encoders and prototypes train exactly as before |
| **Phase Transition** (§7) | ⚠️ Minor addition | Server additionally broadcasts $\text{Proto}\_m^c$ for $m \in \text{Recv}\_{\text{cross}}(i)$ to consenting clients (tiny payload — 256 floats per prototype) |
| **Phase 2 Loss** (§8.3) | ✅ New term | Add $\lambda\_3 \cdot \mathcal{L}\_{\text{cross-absorb}}$ for federated cross-modal regularization |
| **Local Cross-Absorb Head** | ✅ New component (7th) | Private, `detach()`-separated refinement module. Never aggregated. Uses $\text{Proto}\_m^c$ for $m \in \text{Recv}\_{\text{cross}}(i)$ as attention keys/regularization targets. |
| **Purity Lemma** (§12.2) | ⚠️ Updated, not broken | Original lemma holds for federated track when Approach D is used. Extended lemma: track parameters carry signal from $S' \cup \text{Recv}\_{\text{cross-unanimous}}$ when Approach C is used — but this is explicitly consented. |
| **Policy Matrices** (§5.3) | ⚠️ Minor extension | $R\_{\text{recv}}$ default logic updated: $R\_{\text{recv}}(i, j, m) = 1$ now possible for $m \notin O(i)$ |
| **Provenance Ledger** (§5.6) | ⚠️ New entry type | Records which cross-boundary prototypes were absorbed and by which client |
| **Experimental Federation** (§13) | ⚠️ Client matrix update | Hospital 3 gets $\text{Recv}(3) = \{T1, \text{FLAIR}, \text{T1ce}\}$ with $\text{Recv}\_{\text{cross}}(3) = \{\text{T1ce}\}$ |

---

## 6. What DOESN'T Need to Change

- **Encoder Bank** — frozen, untouched
- **Phase 1 training** — identical
- **Freeze procedure** — identical
- **Track isolation for non-cross-absorb tracks** — unchanged
- **Send-Gated routing** — unchanged
- **Cold-Start / Lineage Rule** — unchanged (Lineage only applies to track seeding, not prototype absorption)
- **Multi-Track Contribution** — unchanged

---

## 7. Open Questions for the Solution

| Question | Impact | Suggested Resolution |
| :--- | :--- | :--- |
| **Does prototype-level alignment actually improve task performance?** | Core empirical question. Proto alignment is a very coarse signal (4 class-level vectors, not spatial detail). | Run ablation: Track $S_3$ with vs. without $\mathcal{L}\_{\text{cross-absorb}}$, especially on ET region (T1ce-dependent). |
| **What should the CrossAbsorbHead architecture be?** | Design decision | Cross-attention (z_S3 queries, Proto_T1ce^c keys/values) is the natural choice — consistent with the existing FusionHead design |
| **Should $\lambda_3$ be swept or fixed?** | Hyperparameter | Add to the sweep: $\lambda_3 \in \{0, 0.01, 0.1, 0.5\}$ |
| **What if Proto_T1ce^c ≈ Proto_T1^c already (shared language)?** | If prototypes are *too* similar, the cross-absorb term adds no new information | Measure inter-modal prototype distance after Phase 1. If $\|\text{Proto}\_{\text{T1ce}}^c - \text{Proto}\_{\text{T1}}^c\|$ is small, the knowledge transfer is inherently limited — but this is an empirical finding, not a flaw. |
| **How does this interact with the Purity Probe (Ablation A3)?** | The probe measures whether forbidden modality signal enters a pathway. With R4, T1ce signal *intentionally* enters Hospital 3's local pathway. | Need a new probe variant: "unconsented purity" (should be at baseline) vs. "consented cross-absorb" (should be elevated — that's the point). |

---

## 8. Bottom Line

> [!IMPORTANT]
> **Yes — Gap 6 / R4 is solvable within the existing CAMFS architecture with targeted, modular extensions.** The architecture does not need to be rebuilt. The "shared 256-D language" property established in Phase 1 is the critical enabler — it makes cross-boundary knowledge expressible without requiring the missing modality's raw input data. The recommended solution (Approach C + D hybrid) adds one loss term and one local-only component, preserves all existing federated purity guarantees via `detach()` boundaries, and reduces exactly to M1 behavior when $\text{Recv}_{\text{cross}}(i) = \emptyset$.
