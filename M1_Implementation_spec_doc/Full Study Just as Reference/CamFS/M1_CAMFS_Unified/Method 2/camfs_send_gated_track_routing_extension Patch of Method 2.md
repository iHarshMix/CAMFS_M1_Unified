# CAMFS Extension: Send-Gated Track Routing
### Decoupling Ownership, Send-Consent, and Receive-Consent at the Fusion-Track Layer

**Supersedes/extends:**
- `consent_aware_camfs_brats_paper.md` §1.3 (Formal Problem Statement), §5.7 (Cold Start)
- `camfs_architecture_method2_full_spec.md` §4.3 (Policy Matrices), §8.4 (Aggregation & Broadcast), §13 (Cold-Start), §4.5 (Phase Controller)

Everything in the source documents not explicitly modified below is unchanged and not repeated here.

**Standing assumption for this entire document — Honest, Consent-Aware Execution.** Every client faithfully executes only the training procedure its own declared policy permits: no client routes withheld-modality data through covert channels (augmentation/crop parameters, sampling curriculum, RNG/dropout seeding, early stopping, hyperparameter choice, or any other side channel). Adversarial or malicious-client behavior is explicitly **out of scope** for this document. This assumption is what the source discussion called "(A1)"; here it is elevated from a hedged caveat to a standing scope condition of the whole extension, stated once, not re-litigated per section.

**Traceability convention:** `[Inherited]` = unchanged from Method 2 spec. `[Extended]` = an existing CAMFS object, modified here. `[New]` = introduced in this document, with no precedent upstream.

---

## 0. Summary of What Changed and Why

| # | Item | Status |
|---|---|---|
| 1 | Gap: a client cannot withhold one owned modality from a fusion track's shared contribution while still receiving that modality's federation benefit | Identified, named **Flaw 15** (numbering inferred from Appendix B sequence, not confirmed against the audit file) |
| 2 | Mechanism: **Send-Gated Track Routing** — re-key track *contribution* by $\text{Send}(i)$ rather than $O(i)$ | Specified, §3 |
| 3 | Compliance guarantee for the re-keyed track | Proven — **Lemma (Send-Restricted Track Purity)**, §9 |
| 4 | Cold-start purity condition for a re-keyed track's initial parameters | Generalized into the **Lineage Rule**, §6 |
| 5 | Enforcement mechanism for the Lineage Rule | **Provenance Ledger**, §7 `[New]` |
| 6 | Per-track lifecycle management | **Phase Controller generalized** to one instance per track, §8 `[Extended]` |
| 7 | Consequence for §1.3's own contract | Trade-off between auditability and graceful degradation is real but **bounded**, not eliminated — §10 |

---

## 1. The Gap (Restated Precisely)

### 1.1 What the original formal statement admits

CAMFS's problem statement (`consent_aware_camfs_brats_paper.md` §1.3) defines, for each client $i$, a fixed owned subset $O(i) \subseteq \mathcal{M}$, and governs cross-client sharing through two matrices:

- $R_{\text{send}}(i,m)$ — client $i$ may contribute modality-$m$ signal
- $R_{\text{recv}}(i,j,m)$ — client $i$ may receive modality-$m$ signal originating from client $j$

Both matrices govern **inter-client** permission. Neither expresses an asymmetry a single client might want on a modality **it itself owns**: use it locally, don't contribute it, still receive the pooled benefit of it. $O(i)$ implicitly does double duty as both "what I have" and "what I'll act on," and nothing in §1.3 separates the two.

### 1.2 The gap, as a concrete policy triple

For client $i$:

$$O(i) = \{T1, T2, T1ce\}, \quad \text{Send}(i) = \{T1, T2\}, \quad \text{Recv}(i) = \{T1, T2, T1ce\}$$

Client $i$ owns T1ce and uses it locally. It refuses to let its own T1ce-derived signal leave its walls (ethics/MoU restriction on that modality specifically), while still wanting the benefit of the **pooled** T1ce encoder trained from other consenting clients' contributions.

### 1.3 Why Layer 1 already handles this, and Layer 2 doesn't

**Unimodal Encoder Bank (Layer 1):** already correct, no changes needed. $\text{Encoder}_{T1ce}$'s aggregation sum (`camfs_architecture_method2_full_spec.md` §6.4) is gated by $R_{\text{send}}(j,\text{T1ce})=1$. If $i \notin \text{Send}(i)$ for T1ce, $i$ simply never enters that sum — $i$'s T1ce data cannot shape the global $\text{Encoder}_{T1ce}^{*}$. $i$ can still *pull* the frozen $\text{Encoder}_{T1ce}^{*}$ afterward, gated by $R_{\text{recv}}$, to encode its own T1ce scans locally. No contradiction.

**Subset-Fusion Track (Layer 2):** broken. Track-level policy has exactly one flag, $R_{\text{send}}^{\text{track}}(i,S) \in \{0,1\}$, covering $i$'s **entire** contribution to track $S=O(i)$. There is no sub-flag for "contribute the parts of $S$ that exclude T1ce." This is not a bookkeeping omission — it is forced by how fusion works: with cross-attention (or any nonlinear fusion), $\partial\mathcal{L}_{\text{task}}/\partial\theta_{\text{FusionHead}_S}$ is a joint function of every input modality simultaneously. There is no way to retroactively decompose an update to $\text{FusionHead}_S$ into "the part attributable to T1/T2" versus "the part attributable to T1ce." Both available settings of the single flag violate $i$'s stated policy in one direction:

- $R_{\text{send}}^{\text{track}}(i,S)=1$ → $i$'s full, T1ce-entangled update reaches every peer who also owns $S$. Violates the send restriction.
- $R_{\text{send}}^{\text{track}}(i,S)=0$ → $i$ contributes nothing, including the T1/T2 signal it was willing to share. Over-restrictive relative to its actual policy.

§14 of the Method 2 spec (multi-track contribution) is the closest existing mechanism, but solves the mirror-image problem: a **superset** client opting **in** to a smaller track it doesn't natively own. What's needed here is the opposite: a **native** track member opting **out** of one modality's contribution to a track it does natively own. That case is not specified anywhere upstream.

---

## 2. Extended Notation

| Symbol | Meaning | Status |
|---|---|---|
| $O(i) \subseteq \mathcal{M}$ | Client $i$'s fixed owned modality subset | Inherited |
| $\text{Send}(i) \subseteq O(i)$ | Modalities $i$ is willing to contribute to the federation | **New — promoted to first-class** |
| $\text{Recv}(i) \subseteq O(i)$ | Modalities $i$ wants pooled-in benefit for | **New — promoted to first-class** |
| $S'$ | A track identity keyed by a client's $\text{Send}(i)$ value, distinct from $O(i)$-keyed tracks | New |
| $\mathcal{I}(S') = \{i \in C : \text{Send}(i) = S'\}$ | Contributor set for track $S'$ | New — replaces $\{i : O(i)=S\}$ in aggregation gating |
| $\text{Lineage}(\theta)$ | Union, across every computation that ever shaped $\theta$'s value, of the modality sets touched | New, §6 |

**Well-formedness constraint:** $\text{Recv}(i) \subseteq O(i)$. A client cannot meaningfully "receive" fusion-level benefit for a modality it doesn't own at all — that would be modality hallucination, which is precisely what CAMFS's motivating framing rejects. Degenerate case $\text{Send}(i)=\text{Recv}(i)=O(i)$ recovers the original single-matrix behavior exactly; this extension is a strict generalization, not a replacement.

---

## 3. Architecture: Send-Gated Track Routing

### 3.1 Design Principle

**A client's contribution to the shared fusion-track pool is keyed by what it will *send*, not by what it *owns*.** $O(i)$ continues to govern only $i$'s own local computation (which encoders it can run). This is not "the same mechanism as the encoder layer, applied one layer down" — it is a distinct move: rather than making a single track accept a partial contribution (which the fusion module's fixed input arity makes structurally impossible), a client with $\text{Send}(i) \subsetneq O(i)$ is re-keyed into a **different, smaller, dedicated track** for the send side, and reconnects to the fuller federation only on the receive side.

### 3.2 Component Diagram

```
                                   SERVER
   ┌───────────────────────────────────────────────────────────────────┐
   │  UNIMODAL BANK  (unchanged — global per modality, R_send/R_recv)   │
   │  Encoder_T1   Encoder_T1ce   Encoder_T2   Encoder_FLAIR            │
   │  [Frozen after Phase 1, per Method 2]                              │
   ├───────────────────────────────────────────────────────────────────┤
   │  SUBSET-FUSION TRACKS — now keyed by Send(i), not O(i)             │
   │                                                                     │
   │   Track S = {T1,T2,T1ce}          Track S' = {T1,T2}               │
   │   FusionHead_S, Decoder_S         FusionHead_S', Decoder_S'        │
   │   contributors: {j : Send(j)=S}   contributors: {i : Send(i)=S'}   │
   └──────────────┬────────────────────────────┬─────────────────────────┘
                  │                            │
         (native full sender)          (native restricted sender —
                                         e.g. client i, owns T1ce,
                                         withholds it on send)
                  │                            │
                  └───────────  receive-side reconnection  ───────────┐
                                                                        │
                                            i additionally either:     │
                                            (a) pulls Track S's output │
                                                at inference (if        │
                                                R_recv^track(i,·,S)=1), │
                                                or                      │
                                            (b) trains a private,       │
                                                never-transmitted        │
                                                personalized local head  │
                                                using Encoder_T1*,       │
                                                Encoder_T2*, Encoder_T1ce* ◄
```

### 3.3 Contribution Path (client $i$, $\text{Send}(i)=S'=\{T1,T2\}$)

```
for each round t in Phase 2:
    pull FusionHead_S', Decoder_S'    (gated by R_send^track / R_recv^track, keyed on Send)
    local_train():
        z_T1 = detach(Encoder_T1*(x_T1))
        z_T2 = detach(Encoder_T2*(x_T2))
        # x_T1ce never enters this subgraph — no input slot exists for it
        z_S' = FusionHead_S'({z_T1, z_T2})
        y_hat = Decoder_S'(z_S')
        loss = L_task(y, y_hat) + λ2 · L_fusion-align(z_S', FusedProto_S'^c)
        backprop → updates FusionHead_S', Decoder_S' only
    push FusionHead_S', Decoder_S' deltas, gated by R_send^track(i, S')=1
    push local FusedProto_S'^c estimate
```

T1ce is **absent**, not suppressed — the same property that makes §11 of the Method 2 spec (frozen-encoder detach argument) hold, applied here to a track membership decision rather than a phase boundary.

### 3.4 Receive Path — Two Options

**(a) Pull-only (lighter, conditional on a peer existing).** If some peer $j$ with $O(j)=\{T1,T2,T1ce\}$ fully sends to Track $S$, $i$ sets $R_{\text{recv}}^{\text{track}}(i,j,S)=1$ and uses $j$'s $\text{FusionHead}_S/\text{Decoder}_S$ **at inference only** — never trains against it, never uploads anything derived from it. Simple, but not personalized to $i$'s own population, and unavailable if no such peer exists.

**(b) Personalized local head (default, always available).** $i$ additionally trains a private $\text{FusionHead}_{O(i),i}^{\text{local}}$, $\text{Decoder}_{O(i),i}^{\text{local}}$ using all three frozen encoders ($\text{Encoder}_{T1}^{*}, \text{Encoder}_{T2}^{*}, \text{Encoder}_{T1ce}^{*}$), trained purely on $i$'s own data. **Never transmitted, never aggregated** — its safety is definitional, not proof-dependent, for the same reason purely local computation has always been safe in this architecture. No lemma is required for this piece.

---

## 4. Policy Matrices — Updated

`camfs_architecture_method2_full_spec.md` §4.3 specified a single default rule anchored on ownership: "default to 1 whenever $O(i)=O(j)=S$." Once send and receive can diverge from ownership, one rule is no longer sufficient — this is split into two.

| Matrix | Governs | Default rule |
|---|---|---|
| $R_{\text{send}}^{\text{track}}(i,S')$ | $i$ may contribute to track $S'$'s aggregate | **1 whenever $\text{Send}(i) = S'$** *(send-keyed — new)* |
| $R_{\text{recv}}^{\text{track}}(i,j,S)$ | $i$ may receive track-$S$ signal from $j$ | 1 whenever $O(i) \supseteq S$ or $O(j)=S$ per existing semantics *(receive-keyed — unchanged in spirit, re-derived from $O$ not $\text{Send}$)* |

Both remain exogenous, versioned, inspectable — consistent with §4's "Hard constraint" row in the Method 2 spec. This is a strict extension: for any client with $\text{Send}(i)=O(i)$ (the default, unrestricted case), both rules collapse to the original single-flag behavior exactly.

---

## 5. Aggregation & Broadcast — Updated Equations

`camfs_architecture_method2_full_spec.md` §8.4's aggregation sum was gated on $i:O(i)=S$. Re-keyed:

$$\theta_{\text{FusionHead}_{S'}}^{t+1} = \sum_{i \,:\, \text{Send}(i)=S',\; R_{\text{send}}^{\text{track}}(i,S')=1} \frac{n_i}{N_{S'}}\,\theta_{\text{FusionHead}_{S'},i}^{t}$$

$$\theta_{\text{FusionHead}_{S'} \to k}^{t+1} = \sum_{\substack{j:\,\text{Send}(j)=S',\,R_{\text{send}}^{\text{track}}(j,S')=1,\\ R_{\text{recv}}^{\text{track}}(k,j,S')=1}} \frac{n_j}{N_{S'}^{(k)}}\,\theta_{\text{FusionHead}_{S'},j}^{t}$$

— identical structure for $\text{Decoder}_{S'}$ and $\text{FusedProto}_{S'}^c$. This is the same equation *form* as §8.4, with its membership condition changed, not "reused verbatim."

---

## 6. Track Initialization: The Lineage Rule

### 6.1 Why cold-start needs its own purity condition

The forward-pass-exclusion argument in §3.3/§9 only guarantees purity for **rounds after** a track exists. If Track $S'$ is newly created, its *initial* parameters must come from somewhere, and a naive reuse of §13's literal rule — "average the parameters of the smallest existing superset track" — would seed $S'$ from a track whose history included the very modality $S'$ was created to exclude. Channel-subsetting the tensor shape does not fix this: the retained weights were *learned jointly* with the excluded modality present, and can encode a functional dependence on its statistics without that modality's tensor ever again appearing in a forward pass. This is the same weight-disentanglement fragility already flagged elsewhere for additive/vector-arithmetic fusion approaches — it resurfaces here via initialization rather than post-hoc subtraction.

### 6.2 The General Rule

$$\theta \text{ is a valid seed for track } S' \iff \text{Lineage}(\theta) \subseteq S'$$

where $\text{Lineage}(\theta)$ is the union, across every forward/backward computation that ever shaped $\theta$'s current value — direct training *or* any transfer/inheritance operation — of the modality sets those computations touched. Fresh, data-independent initialization has $\text{Lineage}=\emptyset$, trivially valid for any $S'$.

### 6.3 Rejected: Sibling Parameter Transfer

Tempting shortcut: copy a per-modality sub-component (e.g., a cross-attention projection $W_Q^{T2}$) from an existing track that happens to include that modality — e.g., Track $S_A=\{T2,\text{FLAIR}\}$'s $W_Q^{T2}$, into Track $S'=\{T1,T2\}$. Fails the rule: $\text{Lineage}(W_Q^{T2}) = \{T2,\text{FLAIR}\} \not\subseteq \{T1,T2\}$, because $W_Q^{T2}$'s gradient during $S_A$'s training passed through cross-attention mixing and a shared decoder that also saw FLAIR — it is co-adapted to FLAIR in the same unmeasurable way a channel-subsetted superset track is co-adapted to whatever it's trimmed away. Labeling a parameter "T2's projection" does not make it a pure function of T2 data if it trained inside a graph that also contained FLAIR.

### 6.4 Tier 1 — Grow From an Existing Strict Subset

If some track $S'' \subsetneq S'$ already exists and was itself validly seeded, its entire history satisfies $\text{Lineage}(S'') \subseteq S'' \subseteq S'$ by transitivity. Net2net-style channel-widening: new input slot(s) for the added modality get fresh, data-independent init; existing slots keep $S''$'s weights unchanged. Satisfies the rule exactly. Limitation: only helps when a suitable strict-subset track happens to already exist.

### 6.5 Tier 1b — Composable Multi-Source Stitching

A generalization of Tier 1 that the rule makes available but a simple "chain of subsets" framing would miss: the rule only requires **each slot's own** lineage to satisfy containment, not that one single existing track cover the whole of $S'$. If a pure $\{T1\}$-only track and a pure $\{T2\}$-only track both exist, $S'=\{T1,T2\}$ can be seeded by taking the T1 slot from one and the T2 slot from the other — each individually satisfies $\text{Lineage}\subseteq S'$, and a tuple of independently-pure components is itself pure. Stacks with Tier 2 (fresh-init any slot lacking a pure source, then warm-start on top).

### 6.6 Tier 2 — Track-Local Warm Start (Always Available)

Run a short bootstrap using only $\mathcal{I}(S')$ and the already-frozen $\text{Encoder}_m^{*}$ for $m \in S'$, using $\mathcal{L}_{\text{fusion-align}}$ alone (no task loss) for a handful of rounds before task-loss training begins — structurally a second, smaller instance of the Phase Controller's own job (§8), scoped to one new track. Every round uses only $S'$-permitted data, so it satisfies the rule by the identical inductive argument the main lemma uses, applied to $S'$'s own bootstrap history. Always available regardless of what other tracks exist. Should converge faster than pure random init — a hypothesis to verify empirically, not a proven bound.

### 6.7 Tier 3 — The Floor

Even the genuine worst case — no subset track, no warm-start, pure random initialization of $\text{FusionHead}_{S'}$/$\text{Decoder}_{S'}$ — only retrains the fusion-and-decode stage, not the encoders, which are already converged and frozen for free regardless of which track is cold-starting. This materially bounds how bad "from scratch" can be relative to Method 1 or naive multimodal FL, where "from scratch" would mean relearning unimodal representations too.

### 6.8 Seed-Selection Procedure (server-side)

```
On track S' creation:
    if a valid Tier 1 source exists (some S'' ⊊ S', already validly seeded):
        seed via channel-widening from S''
    else if valid Tier 1b sources exist for some/all slots:
        seed those slots from their pure sources; fresh-init remaining slots
    always: run Tier 2 warm-start (fusion-align only, no task loss) for W rounds
    then: enter normal Phase-2 task-loss training
```

---

## 7. Provenance Ledger `[New]`

§1.3 point 2 demands an **inspectable object**, not something inferable only from training dynamics. Without a record, whether a given track's $\theta^{t_0}$ came from fresh init, Tier 1, Tier 1b, or an invalid seed is exactly the kind of fact only recoverable from training history — the thing point 2 forbids relying on.

**Component:** one ledger entry per track, written *before* any seed operation is permitted to execute (not audited after the fact):

```
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

This is the natural enforcement point for the Lineage Rule — it makes §1.3 point 2 checkable, not merely satisfiable, and is the correct place to record Tier 1b's mixed-source slot bookkeeping, which needs per-slot rather than per-track granularity.

**Added to Component Inventory (`camfs_architecture_method2_full_spec.md` §4) as a sixth entry**, alongside the Phase Controller.

---

## 8. Phase Controller — Generalized to Per-Track Lifecycle `[Extended]`

**Original (`camfs_architecture_method2_full_spec.md` §4.5):** one global state, `{ CURRENT_PHASE ∈ {1,2}, round t, proto_history[m][c] }`, governing a single federation-wide Phase 1 → Phase 2 transition for the entire encoder bank.

**Generalized here:** Tier 2 warm-start (§6.6) is structurally the same pattern — stabilize using only permitted signal, then transition to task-loss training — applied per-track instead of federation-wide. Rather than treating this as a second, different mechanism that happens to rhyme, the Phase Controller is generalized to manage **one lifecycle instance per track**, each on its own clock, triggered by its own creation event:

```
Per-track state: { LIFECYCLE ∈ {WARMING, STABLE}, round t_local, seed_record }

On track S' creation:
    LIFECYCLE ← WARMING
    execute seed procedure (§6.8), log to Provenance Ledger (§7)

Each round, per active track:
    if LIFECYCLE == WARMING:
        run L_fusion-align-only round (§6.6)
        if warm-start stopping criterion met: LIFECYCLE ← STABLE
    if LIFECYCLE == STABLE:
        run normal Phase-2 round (L_task + λ2·L_fusion-align)
```

Under this reframing, the **original global Phase 1 → Phase 2 transition becomes just the Unimodal Bank's own instance of the identical pattern** — a "track" of one federation-wide unit, rather than a structurally special case. Original §13 cold-start and Tier 2 warm-start stop being two mechanisms that happen to resemble each other and become the same state machine, parameterized differently per scope.

---

## 9. Formal Argument: Send-Restricted Track Purity

**Setup.** $\mathcal{I}(S') = \{i \in C : \text{Send}(i)=S'\}$. For $i \in \mathcal{I}(S')$, the local Phase-2 pass is exactly §3.3's:

$$z_m = \text{detach}(\text{Encoder}_m^{*}(x_i^m)),\ m\in S' \;\to\; z_{S'}=\text{FusionHead}_{S'}(\{z_m\}) \;\to\; \hat y = \text{Decoder}_{S'}(z_{S'})$$
$$\mathcal{L}_i = \mathcal{L}_{\text{task}}(y,\hat y) + \lambda_2\,\mathcal{L}_{\text{fusion-align}}(z_{S'}, \{\text{FusedProto}_{S'}^c\})$$

No $x_i^m$ for $m \notin S'$ appears anywhere in this expression, and structurally cannot — $\text{FusionHead}_{S'}$ is parameterized for exactly $|S'|$ inputs, with no slot to plug a third modality into even accidentally.

**Precondition** (satisfied by this document's standing Honest-Execution Assumption, §0): no client routes excluded-modality data through any side channel (loss, augmentation, sampling, RNG, early stopping, hyperparameter choice) in this pass's pipeline.

**Seed condition (§6.2):** at Track $S'$'s creation round $t_0$, its parameters satisfy $\text{Lineage}(\theta^{t_0}) \subseteq S'$ — enforced via the Provenance Ledger (§7), explicitly **excluding** naive superset cold-start per §6.1/§6.3.

**Lemma (Send-Restricted Track Purity).** Given the standing assumption and the seed condition, for every round $t \geq t_0$, $\theta_{\text{FusionHead}_{S'}}^t$, $\theta_{\text{Decoder}_{S'}}^t$, $\text{FusedProto}_{S'}^{c,t}$ are, as mathematical functions, invariant to $x_i^m$ for every $i \in C$, $m \in \mathcal{M}\setminus S'$.

*Proof.*
- *Step 1 (forward-pass exclusion, per client per round).* No node in the pass's subgraph takes $m \notin S'$ data as input. So $\mathcal{L}_i$, and by the chain rule every $\nabla_\theta \mathcal{L}_i$ computed locally, is constant in $x_i^m$ for $m \notin S'$.
- *Step 2 (aggregation preserves it).* $\theta_{\text{FusionHead}_{S'}}^{t+1} = \sum_{i\in\mathcal{I}(S')} w_i\,\theta_{\text{FusionHead}_{S'},i}^{t}$ — a weighted average of quantities each independent of excluded-modality data, given $\theta^{t}$ already was. The weights $n_i/N_{S'}$ are counts, not content, hence also independent. A weighted average of $v$-independent terms is $v$-independent. Identical argument for $\text{Decoder}_{S'}$ and the forward-only $\text{FusedProto}_{S'}^{c}$.
- *Step 3 (induction).* Step 2 is the inductive step; the seed condition supplies a true base case at $t_0$. Without it, contamination at $t_0$ would propagate forward through every later round faithfully — the induction would be sound but start from a false premise. With it, the guarantee holds for all $t \geq t_0$. $\blacksquare$

**Corollary (composition).** Combined with the Unimodal Bank's own structural guarantee (`camfs_architecture_method2_full_spec.md` §11 — verified directly against the source document in this thread) that $\text{Encoder}_m^{*}$ for $m\in S'$ is a function of Phase-1 data from consenting $m$-owners only, Track $S'$'s entire trained state — encoders it consumes, fusion head, decoder, fused prototype — carries zero signal from any modality outside $S'$, at every layer. This composition is **consistent with** §11's argument; it does not constitute independent verification of a numbered "Theorem 1," since the file containing that theorem's stated assumptions (G1–G4) was not available in this session.

**Scope note — data-purity, not annotation-purity.** This lemma proves no excluded-modality *scan* ever enters the graph. It says nothing about the *ground-truth label* $y$, which for BraTS is typically annotated by radiologists referencing all four sequences. Track $S'$'s performance on the enhancing-tumor (ET) region may still sit at whatever T1ce-dependent annotation ceiling exists independent of routing — a property of the label, not of anything this lemma governs. Disaggregating WT/TC/ET results for any Send-Gated track is advisable so a ceiling effect isn't misread as a routing failure.

**(b) does not need a version of this lemma.** The personalized local head is never transmitted or aggregated; its safety is definitional, not proof-dependent.

---

## 10. Consequence: The §1.3 Points 2/3 Trade-off, Bounded

§1.3 requires both:
> 2. auditable — inspectable... not inferable only from training dynamics
> 3. graceful degradation — a client whose exact modality subset has not been seen before should not need to train an entirely new pathway from nothing

Point 3 was written for a client's **ownership** being novel, not for an existing client's **send policy** diverging from its ownership — Send-Gated Track Routing stretches language written for a different case. Whenever a track $S'=\text{Send}(i)$ has no valid Tier 1/1b source (as in the worked example below), the Lineage Rule forces a genuine cost: no clean cold-start substitute exists, and the best available fallback is Tier 2/3 — retraining the fusion-and-decode stage, warm-started if possible, but never the full stack, since the encoders remain reusable and frozen throughout. This is **bounded**, not eliminated: worse than an impure superset seed would give in raw convergence speed, but materially better than "train an entirely new pathway from nothing," since "from nothing" here never includes the encoders. This trade-off should be stated plainly in any presentation of this extension, not folded silently into a blanket "compliance has costs" line.

---

## 11. Complexity / Compute-Communication Accounting

| Item | Cost relative to Method 2 baseline |
|---|---|
| Restricted-send pass ($i \in \mathcal{I}(S')$) | Same order as any normal Phase-2 pass — one forward/backward through a smaller-arity fusion head |
| Personalized local head (receive-side, option (b)) | One additional local forward/backward pass per round, never transmitted — pure local compute, zero communication cost |
| Pull-only receive (option (a)) | Inference-only pull of a peer track's parameters — no additional training compute |
| Tier 2 warm-start rounds | $W$ extra rounds per newly created track, fusion-align loss only, no task loss |
| Track proliferation | Scales with the diversity of distinct $\text{Send}(i)$ values across the federation, on top of the existing $|\{O(i)\}|$-driven proliferation already flagged in the original paper (§10) and Method 2 spec (§15) |

Recommend consolidating this table with the original paper's §5.9 and the Method 2 spec's §15 into one accounting document if/when this extension is merged into project files.

---

## 12. Worked Example — Full Walkthrough

**Policy:** $O(i)=\{T1,T2,T1ce\}$, $\text{Send}(i)=\{T1,T2\}$, $\text{Recv}(i)=\{T1,T2,T1ce\}$. Existing tracks in the base BraTS instantiation: $S_A=\{T2,\text{FLAIR}\}$, $S_B=\{T1,T1ce,T2,\text{FLAIR}\}$, $S_C=\{T1,\text{FLAIR}\}$, $S_D=\{T1ce,T2\}$ (ablation-only).

1. **Contribution.** $i$ is re-keyed to Track $S'=\{T1,T2\}$. No existing track equals $S'$.
2. **Seed search (§6.8).** Is any existing track a strict subset of $\{T1,T2\}$? $S_A, S_B, S_C, S_D$ — none is $\subsetneq \{T1,T2\}$. Tier 1 unavailable. Is there a pure single-modality-only source for T1 or T2 individually? $S_A$'s T2 slot carries FLAIR-lineage; $S_C$'s T1 slot carries FLAIR-lineage; $S_D$'s T2 slot carries T1ce-lineage; $S_B$'s slots carry everything. No single-modality-pure source exists anywhere in this track list. Tier 1b unavailable.
3. **Fallback: Tier 2.** Fresh-init $\text{FusionHead}_{S'}$, $\text{Decoder}_{S'}$; run $W$ rounds of fusion-align-only warm-start using $\text{Encoder}_{T1}^{*}$, $\text{Encoder}_{T2}^{*}$ before task-loss training begins. Provenance Ledger entry recorded as shown in §7.
4. **Ongoing training.** $i$ (and any other client with $\text{Send}=\{T1,T2\}$) contributes to $S'$'s aggregate per §5's updated equations. By the Lemma (§9), $S'$'s entire trained state is provably free of T1ce influence.
5. **Receive side.** $i$ separately trains a personalized local head using all three frozen encoders (option (b)) to get fusion-level benefit from its own T1ce data — never transmitted, unconditionally safe.
6. **Net result:** $i$'s stated policy — own it, use it, don't send it, still benefit from the pool where possible — is honored exactly, with a disclosed, bounded cold-start cost recorded in §10.

---

## 13. Open Problems (Explicitly Deferred, Not Resolved Here)

- **Graduated free-rider dynamics.** The original paper's §10 names only the binary case — "a hospital consenting only to receive, never to send." Send-Gated Track Routing makes a **graduated** version possible for the first time at the fusion layer: many clients each restricting to *some* smaller subset, none necessarily sending zero, potentially fragmenting the track space into many small, undertrained tracks. Not modeled here.
- **Purity ≠ usefulness.** The Lemma and Lineage Rule prove every tier produces a *pure* seed; none of them prove a *functionally good* one. Tier 1b in particular stitches together components that were never jointly optimized with each other — whether that produces a coherent starting point before fine-tuning, versus just a differently-shaped random init, is untested.
- **Prototype alignment across siblings.** A speculative idea considered and shelved: aligning $\text{FusedProto}$ across tracks that share some modalities but where neither contains the other (e.g., $S_A=\{T2,\text{FLAIR}\}$ and $S'=\{T1,T2\}$) would pull each toward information the Lineage Rule says the other isn't licensed to have — it would reopen the exact leak this extension closes. Not recommended without a containment-respecting reformulation.
- **Reconciliation with `camfs_method2_theoretical_foundations.md`.** This file (containing the actual Theorem 1 and assumptions G1–G4) was not available in this session. All claims of consistency with it are secondhand and should be checked directly before this extension is presented as formally reconciled.
- **Fused-embedding purity probe.** `camfs_architecture_method2_full_spec.md` Appendix A defines a purity probe at the unimodal bottleneck $z_m$ only. Extending it to the fused embedding $z_{S'}$ — and running it in two arms (Tier-2-seeded vs., counterfactually, an incorrectly superset-seeded Track $S'$) — would give a direct, visible measurement of what the Lineage Rule's seed condition is preventing.
- **Malicious/adversarial client robustness.** Out of scope by this document's standing assumption (§0). Should be revisited explicitly if the honest-execution assumption is ever relaxed.

---

## Appendix A — Full Symbol Table (This Extension)

| Symbol | Meaning |
|---|---|
| $O(i)$ | Owned modality subset (inherited) |
| $\text{Send}(i) \subseteq O(i)$ | Modalities $i$ contributes to the federation |
| $\text{Recv}(i) \subseteq O(i)$ | Modalities $i$ draws federation benefit for |
| $S'$ | Track keyed by $\text{Send}(i)$ |
| $\mathcal{I}(S')$ | Contributor set for track $S'$ |
| $R_{\text{send}}^{\text{track}}(i,S')$ | Send-keyed track policy gate |
| $R_{\text{recv}}^{\text{track}}(i,j,S)$ | Receive-keyed track policy gate |
| $\text{Lineage}(\theta)$ | Union of all modality sets ever touched by any computation shaping $\theta$ |
| $\text{FusionHead}_{S,i}^{\text{local}}$, $\text{Decoder}_{S,i}^{\text{local}}$ | Personalized, never-transmitted local head |

## Appendix B — Change Map

| Source location | Original rule | Change in this document |
|---|---|---|
| Paper §1.3 | $O(i)$, $R_{\text{send}}(i,m)$, $R_{\text{recv}}(i,j,m)$ only | $\text{Send}(i)$, $\text{Recv}(i)$ promoted to first-class, §2 |
| Method 2 spec §4.3 | One default rule, keyed on $O(i)=O(j)=S$ | Split into send-keyed and receive-keyed defaults, §4 |
| Method 2 spec §8.4 | Aggregation gated on $i:O(i)=S$ | Gated on $i:\text{Send}(i)=S'$, §5 |
| Method 2 spec §13 | Cold-start = channel-subset nearest superset, unconditionally | Superseded by general Lineage Rule + Tier 1/1b/2/3 procedure, §6 |
| Method 2 spec §4.5 | One global Phase 1/2 state | Generalized to per-track lifecycle; global transition is one instance of it, §8 |
| Method 2 spec §4 (Component Inventory) | Five components | Sixth added: Provenance Ledger, §7 |
| Method 2 spec Appendix A | Probes $z_m$ only | Extension to fused $z_{S'}$ proposed, not built, §13 |
| Paper §10 | Binary free-rider case named, not analyzed | Graduated case newly possible, still not analyzed, §13 |

---

*This document consolidates a single conversation thread's reasoning into a standalone specification. It has not been cross-checked against `camfs_method2_theoretical_foundations.md` or any audit file beyond the three source documents supplied in this session. Claims of consistency with material outside those three files are flagged as such throughout and should be verified before this extension is merged into the project's canonical documents.*
