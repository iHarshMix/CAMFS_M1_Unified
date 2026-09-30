# Consent-Aware Multimodal Federated Learning
## Research Idea Analysis — Session Summary

> **Idea Source:** Professor Jerry  
> **Researcher:** Harsh Yadav  
> **Date:** June 2026  
> **Domain:** Multimodal Federated Learning — Missing Modalities

---

## 1. The Core Idea (Professor's Proposal)

A framework for **asymmetric, policy-based modality gradient routing** in multimodal FL, motivated by real-world institutional constraints.

### The Setup

| Client | Modalities Owned | Will Send | Will Receive |
|--------|-----------------|-----------|--------------|
| A | PET (P), CT (C) | P, C only | Only P,C-infused gradients — blocks M |
| B | MRI (M), PET (P), CT (C) | M, P, C | All gradients from all clients |
| C | MRI (M), CT (C) | M, C only | Only M,C-infused gradients — blocks P |

### Real-World Motivations

- **Hardware absence** — A hospital without an MRI machine should not have its model pulled toward MRI-driven representations; its patients will never be scanned that way.
- **Ethics board clearance** — A hospital may have consent approval for PET and CT data but not MRI; it legally cannot use MRI-infused information, even indirectly through gradients.
- **Institutional MoU agreements** — Like Google and Facebook sharing user data selectively to protect competitive advantage, hospitals share only what their agreements permit.
- **Asymmetry is natural** — A well-equipped hospital (Client B) may absorb all knowledge but restrict what it exports; the flow is not symmetric.

---

## 2. What Makes This Different from Existing Work

### The Fundamental Shift

| Dimension | Existing Work | Professor's Idea |
|-----------|--------------|-----------------|
| Who decides what to share? | Performance/efficiency metrics (Shapley, bandwidth) | Client-defined policy (MoU, ethics, hardware) |
| Direction of concern | Send-side only | **Both send-side AND receive-side** |
| Type of missing modality | Partial (some samples missing a modality) | **Complete** (client will never have that modality) |
| Aggregation | Symmetric, global | **Asymmetric, personalized per client** |
| Policy layer | None | **Explicit MoU/consent matrix** |

### Why FedAMM is NOT the Same

FedAMM handles partial missing modalities by aligning **all** modality-specific encoders toward a shared multimodal teacher prototype. This means:
- All clients' encoders are pulled toward a joint representation space that includes modalities they may never have.
- A client without MRI has its CT encoder implicitly shaped by MRI-driven teacher signals.
- The professor's idea does the opposite — isolate the representation space to only the modalities a client owns and consents to.

---

## 3. Literature Landscape

### Closest Existing Papers and Their Gaps

**FedMFS** (Purdue, 2023)
- Selects which modalities to upload using Shapley value importance × model size trade-off.
- *Gap:* Send-side only. No receive-side filtering. Motivation is communication efficiency, not consent.

**MAFS** (2024)
- Clients choose which modalities they consider "insensitive" and are willing to share.
- *Gap:* Semi-supervised setting. Send-side only. No asymmetric bidirectional policy.

**MMiC** (2025)
- Substitutes parameters of incomplete-modality clients with those from complete-modality clients.
- *Gap:* Assumes receiving more modality signal is always beneficial — the opposite assumption to this idea.

**Latent-Space Consensus approaches** (2026)
- Partitions clients into modality-combination subsets; only same-subset clients collaborate.
- *Gap:* Static symmetric clusters. Cannot handle asymmetric receive policies (B takes all, A takes subset).

**CreamFL, FedMAC, FedMRR**
- Focus on partial missing modalities, contrastive alignment, or cross-modal aggregation.
- *Gap:* None model the consent/permission layer.

---

## 4. Five Literature Gaps This Idea Addresses

**Gap 1 — Receive-side filtering**  
Every existing paper thinks only about what a client *sends*. No one has formalized what a client is *permitted to receive* as an independent policy decision.

**Gap 2 — Asymmetric bidirectional gradient flow**  
No paper models a training round where B accepts all, A accepts only {P,C}-infused, and C accepts only {M,C}-infused gradients simultaneously. This is a directed graph of gradient flows, not symmetric averaging.

**Gap 3 — Gradient purity by modality**  
When a client trains a joint encoder on multiple modalities, its gradient carries mixed modality signal. No paper has formalized "modality gradient contamination" — that even a PET encoder gradient, trained in the presence of MRI data, carries indirect MRI signal. The architectural fix is strictly separate modality encoders.

**Gap 4 — MoU/agreement as a formal FL component**  
No paper has introduced a policy/consent matrix as a first-class architectural component sitting above the aggregation mechanism. The legal and institutional framing is entirely absent from the FL literature.

**Gap 5 — Convergence under asymmetric aggregation**  
FedAvg and FedProx convergence proofs assume symmetric, global aggregation. Asymmetric routing means each client receives a different personalized model — the convergence theory needs to be rebuilt.

---

## 5. Proposed Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│                        SERVER                           │
│                                                         │
│   Policy Routing Matrix R[i][j][m]                     │
│   R[i][j][m] = 1 → Client i accepts encoder_m          │
│                     weights from Client j               │
│                                                         │
│   Aggregation for client i, modality m:                 │
│   Aggregate encoder_m from all j where R[i][j][m] = 1  │
└──────────┬─────────────────────────────┬────────────────┘
           │                             │
    Upload only                   Personalized
    owned modality                aggregated encoders
    encoders                      per client
           │                             │
┌──────────▼──────────┐   ┌─────────────▼───────────────┐
│      CLIENT A       │   │         CLIENT B             │
│                     │   │                              │
│  Encoder_P (PET)    │   │  Encoder_M (MRI)            │
│  Encoder_C (CT)     │   │  Encoder_P (PET)            │
│                     │   │  Encoder_C (CT)             │
│  Receives: P, C     │   │  Receives: M, P, C          │
│  Blocks:   M        │   │  (from all clients)         │
└─────────────────────┘   └──────────────────────────────┘
```

### Key Architectural Decisions

- **Strictly separate modality encoders per client** — no cross-encoder gradient flow locally. This is what enforces gradient purity. Each modality encoder trains independently on local data.
- **Shared task head** — fuses available modality representations at inference. Can be personalized or global.
- **Policy matrix R** — defined at initialization based on MoU agreements. R[i][j][m] = 1 means client i will accept encoder_m weights that client j has trained.
- **Personalized aggregation** — client i's encoder_m is the weighted average of encoder_m from all j where R[i][j][m] = 1. No single global model exists.

### Training Loop

```
For each round t:
  1. Server broadcasts current personalized encoder sets to each client
  2. Each client trains its owned modality encoders locally (independently)
  3. Each client uploads only its owned modality encoder weights to server
  4. Server applies routing matrix R:
       for each client i, modality m:
           aggregate encoder_m from {j : R[i][j][m] = 1}
  5. Server sends personalized aggregated encoders back to each client
  Repeat until convergence
```

---

## 6. Hard Technical Challenges

### Challenge 1 — Gradient Disentanglement
**The problem:** If a client trains encoders on multiple modalities even independently, the task head's gradients flow back through all encoders simultaneously during joint backpropagation. Pure separation requires either (a) truly independent per-modality training objectives, or (b) gradient masking during backpropagation.

**Mitigation:** Train each modality encoder with a modality-specific auxiliary loss in addition to the joint task loss, and stop gradients between encoder branches.

### Challenge 2 — No Single Global Model
**The problem:** Each client converges to a different personalized model. Standard convergence analysis does not apply. This is technically a **personalized FL** problem layered on a **routing** problem.

**Mitigation:** Frame convergence as each client converging to its own local optimum under the constraint of the routing policy. Draw on personalized FL convergence literature (pFedMe, Ditto).

### Challenge 3 — Free-Rider Imbalance
**The problem:** Client B contributes MRI, PET, and CT but receives all. Client A contributes only PET and CT but still benefits from Client B's full encoder. This creates a fairness imbalance.

**Mitigation:** Weight the aggregation contribution of each client proportional to what it shares × the number of clients that accept its contribution. Incorporate Shapley-based contribution scoring.

### Challenge 4 — Policy Leakage
**The problem:** The routing matrix R reveals institutional relationships. If R[A][B][M] = 0, the server knows Hospital A has no MRI machines. This is sensitive information.

**Mitigation:** This is a later-stage concern. For the core paper, treat R as a trusted initialization parameter. In follow-up work, explore secure computation or masked policy encoding.

---

## 7. Baselines and Evaluation

### Baselines

| Baseline | Description |
|----------|-------------|
| Local Only | No collaboration. Each client trains on its own data. |
| FedAvg (all modalities) | All encoder weights aggregated symmetrically, ignoring policies. |
| FedMFS | Shapley-based send-side selection, no receive-side filtering. |
| Modality-Subset Clustering | Group clients by identical modality sets; collaborate within groups only. |
| Oracle | Full sharing, no restrictions. Upper bound on performance. |

### Dataset: BraTS 2023 (Simulated FL Split)

BraTS provides four MRI sub-modalities per patient: T1, T2, FLAIR, T1ce. Treat each sub-modality as a different "imaging machine" and simulate the professor's scenario:

| Simulated Hospital | Available Modalities | Receive Policy |
|--------------------|---------------------|----------------|
| Hospital A | T2, FLAIR | Block T1, T1ce |
| Hospital B | T1, T2, FLAIR, T1ce | Accept all |
| Hospital C | T1, FLAIR | Block T2, T1ce |

### Metrics

- **Primary:** Segmentation Dice score per client on its owned modalities
- **Secondary:** Representation drift — cosine distance of encoder embeddings vs. oracle
- **Ablation 1:** Vary asymmetry of routing matrix R from fully symmetric to maximally asymmetric
- **Ablation 2:** Vary the number of clients and modality ownership patterns

---

## 8. Novelty Assessment

**Verdict: Genuinely novel as a unified framework.**

The combination of:
1. Receive-side policy filtering (new)
2. Asymmetric bidirectional gradient flow (new)
3. Gradient purity via strict encoder separation (new framing)
4. MoU/consent matrix as a formal FL component (new)
5. Complete (not partial) missing modality assumption (distinct from most existing work)

...has not been published as a unified framework. Individual components exist in isolation in adjacent work, but no paper assembles them together or motivates them through the institutional consent angle.

---

## 9. Research Value and Suggested Positioning

### Framing Options

**Option A — "Consent-Aware Multimodal Federated Learning"**  
Emphasizes the ethical/legal motivation. Good for MICCAI, NeurIPS healthcare workshops, or journals at the intersection of AI and healthcare policy.

**Option B — "Policy-Gated Gradient Routing for Multimodal FL"**  
Emphasizes the systems/algorithmic contribution. Better for ICLR, ICML, or NeurIPS main track (if convergence theory is included).

### What a Full Paper Needs

| Component | Status |
|-----------|--------|
| Architectural design (separate encoders + routing matrix) | Conceptually clear, needs implementation |
| Routing matrix formalization | Needs mathematical definition |
| Aggregation algorithm | Needs pseudocode + proof of correctness |
| Convergence theory under asymmetric aggregation | Open — critical for top-tier venues |
| BraTS 2023 simulation setup | Feasible, needs engineering |
| Baseline comparisons | Clear list, needs implementation |
| Ablation study | Clear design |
| Privacy hardening (DP, Secure Aggregation) | Later-stage follow-up, not required for core paper |

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Separate encoder constraint reduces local fusion quality | Medium | Show collaborative gain offsets local fusion loss |
| No convergence theory | High for top venues | Empirically strong results + theory as future work for workshop track |
| Free-rider problem | Medium | Contribution-weighted aggregation |
| Dataset simulation is artificial | Low | BraTS is widely accepted; simulate with documented splits |

---

## 10. Open Questions for Next Steps

1. How exactly does the routing matrix R get initialized — is it a hyperparameter, or learned?
2. Can the aggregation under asymmetric R be shown to converge to a stable point? What conditions are needed?
3. How do you handle a new client joining the federation mid-training with a previously unseen modality combination?
4. Is there a way to enforce gradient purity without strictly separate encoders — for example, using gradient masking or modality-specific batch normalization?
5. What happens when the routing matrix creates isolated sub-graphs — clients that share nothing with each other — does the framework degenerate?

---

*Analysis completed: June 2026*  
*Literature searched: FedMFS, MAFS, FedAMM, MMiC, CreamFL, FedMAC, FedMRR, Latent-Space Consensus approaches, FedMPO, RCSR, FedRecon*

---
---

# Part II — Literature Survey of Three Candidate Solution Approaches

> **Context:** After the Part I analysis, three candidate architectures were proposed as ways to actually *implement* the consent-aware routing framework, inspired by re-reading FedCMD and FedAMM. Each is checked here against current literature (searched July 2026).

---

## 11. Approach 1 — Federated Unimodal Encoders + Unimodal Prototypes with Client-Side Fusion

### The Idea
Split the problem into unimodal contrastive learning. The server maintains and distributes **global unimodal encoder weights** and **global unimodal (class-wise) prototype distributions** — a client downloads only the ones it needs. Once a client has these, it performs **all multimodal fusion locally**, aligning its own encoders to the received unimodal prototypes. No fused/multimodal weights are ever shared globally — after the unimodal broadcast, everything left is client-side alignment.

### Closest Prior Work

**FedCMD — "Cross-Modal Federated Learning among Unimodal Devices" (2025, ACM IMWUT/UbiComp).**
This is a near-exact structural match. It performs unimodal federated learning first to learn unimodal encoders, then calculates and shares per-modality prototypes across devices for cross-modal feature alignment, with those prototypes serving as stand-ins for missing modalities so a downstream fusion-and-classification network can be trained. The resulting network supports both unimodal and any-missing-modality multimodal inputs. This is architecturally almost identical to what you described: unimodal-first training → shared prototypes → downstream fusion.

> **Naming collision warning:** There are two unrelated papers both called "FedCMD" — this unimodal-devices one, and a separate *Federated Cross-Modal Distillation* paper for driver emotion recognition (physiological signal ↔ video, teacher-student split across vehicle/edge). If you're citing FedCMD, double check which one you mean — your idea matches the unimodal-devices paper, not the emotion-recognition one.

**FedProto (Tan et al., AAAI 2022).**
The foundational mechanism underneath all of this: share class-level *prototypes* instead of full model weights, dramatically cutting communication and sidestepping architecture-heterogeneity issues. This is the base primitive that FedCMD, FedAMM, and MFCPL (below) all build on.

**MFCPL — "Cross-Modal Prototype based Multimodal Federated Learning under Severely Missing Modality" (Le et al., arXiv 2401.13898, 2024).**
Extends FedProto specifically for multimodal missing-modality settings. Builds "complete prototypes" combining class-wise, modality-shared, and modality-specific representations, and aligns local features to them via a contrastive loss (CMPR/CMPC) — critically, when a client lacks modality *m*, that modality's contrastive term is simply excluded from the loss, rather than approximated or hallucinated. That's functionally very close to what you want for "complete missing modality" clients.

### What This Approach Solves — and What It Doesn't

This is worth being precise about, because it changes how you should think about the original five gaps from Part I:

- **It solves Gap 1 (receive-side filtering) almost for free, architecturally.** In a pull-based design, a client that doesn't have MRI simply never requests the MRI encoder or MRI prototype — there's no "unwanted gradient" to block, because nothing unwanted is ever sent. The consent problem partially dissolves rather than needing to be solved head-on.
- **It does *not* solve Gap 3 (gradient purity).** If Client B trains its PET encoder using a contrastive loss against a *complete* prototype (one informed by MRI-having clients), B's "PET encoder" can still be indirectly shaped by MRI-derived signal. Pull-based architecture stops unwanted modalities from *arriving*, but doesn't stop a client's own encoder from being *influenced* by cross-modal alignment terms during training.
- It still doesn't give you the explicit MoU/policy matrix as a first-class object — consent is implicit (only ask for what you need), not an enforceable, auditable rule.

### Verdict

| | |
|---|---|
| **Already implemented?** | **Yes, substantially.** FedCMD (unimodal-devices version) is essentially this architecture; MFCPL adds the missing-modality-aware contrastive alignment machinery on top. |
| **Feasibility** | **High** — this is proven, working technique, not speculative. |
| **Residual novelty** | Low on the architecture itself. The novel piece, if you keep this direction, has to be the explicit consent/policy layer gating *which* unimodal components a client is even permitted to request — not the unimodal-encoder-plus-prototype mechanism itself. |

---

## 12. Approach 2 — Extracting Modality-Specific Gradients via Vector Subtraction

### The Idea
Since fusing modalities in a shared prototype space is "just a vector operation," train **global unimodal prototypes/encoders** *and* a **global fused (multimodal) prototype/encoder** in parallel. Every client receives all of these. Then, using vector arithmetic, subtract the known unimodal components from the fused representation to isolate the modality-specific signal a client is missing or needs.

### Closest Prior Work

**Task Arithmetic (Ilharco et al., "Editing Models with Task Arithmetic," ICLR 2023 / arXiv 2212.04089).**
This is the foundational result your intuition is closest to. A *task vector* is defined as fine-tuned weights minus pretrained weights. These vectors can be added or subtracted: adding improves multi-task performance, negating (subtracting) suppresses/unlearns a capability. This is exactly the "vector operation" intuition you're describing, just not yet applied to modality-gradient purification in FL.

**Weight disentanglement — the property that decides whether this works at all.**
Task arithmetic only behaves cleanly when the model is *weight-disentangled*: each task vector's effect stays confined to its own task's input domain, with negligible bleed into others. This is **not automatically true** — it's an empirical property, stronger under certain conditions (shared pretrained initialization, NTK/linearized training regimes) and weaker under others (very different tasks, heavy nonlinear interaction between them). Concretely for your case: if a "PET-infused gradient" and an "MRI-infused gradient" interact nonlinearly inside a shared fusion module, subtracting one from the joint gradient will *not* cleanly recover the other — the residual will carry interaction terms, not pure signal.

**DisentAFL — "On Disentanglement of Asymmetrical Knowledge Transfer for Modality-Task Agnostic Federated Learning" (Chen & Zhang, AAAI 2024). This is the most important single finding of this search pass — see §14 below.**
DisentAFL explicitly performs a two-stage **Knowledge Disentanglement and Gating** process that decomposes an asymmetric inter-client information-sharing scheme into independent components, each tied to a specific semantic knowledge type (which includes modality-specific knowledge), then *gates* which decomposed component goes to which client. This is functionally the closest thing in the literature to what you're proposing — except it uses a **learned** disentanglement (via something like mixture-of-experts routing) rather than **closed-form subtraction arithmetic**.

**"Towards Diverse Device Heterogeneous Federated Learning via Task Arithmetic Knowledge Integration" (arXiv 2409.18461).**
Directly imports task-arithmetic-style vector operations into FL, formalizing a "task arithmetic property" adapted for federated knowledge transfer across heterogeneous devices. Confirms the general technique (treating model deltas as combinable/subtractable vectors) already has precedent in FL — just not for modality-gradient purification specifically.

**Whether "fusion is just a vector operation" holds — it depends entirely on the fusion architecture.**
Some multimodal architectures genuinely do use additive/**score-level fusion** (literally summing unimodal embeddings) — this shows up in retrieval-style architectures and in embedding-arithmetic setups (e.g., ImageBind-style averaging). *If* your fusion module is additive, your subtraction premise is mathematically well-founded, because the gradient of a sum is the sum of gradients. But most higher-performing modern fusion is **feature-level / cross-attention** — a genuinely nonlinear joint function of both modalities — where "fused gradient minus unimodal gradient" does not cleanly recover the other modality's pure contribution.

### Verdict

| | |
|---|---|
| **Already implemented?** | **Not as literal closed-form subtraction, in FL, for this purpose.** The *goal* (decompose shared knowledge into modality-attributable pieces) is implemented in DisentAFL — but via a learned mechanism, not arithmetic. |
| **Feasibility** | **Medium-low as literally specified.** It hinges on weight disentanglement holding, which is an imperfect, architecture-dependent property — and on using additive fusion, which trades away some fusion expressiveness relative to attention-based approaches. |
| **Residual novelty** | Real, but risky. Nobody has combined task-arithmetic-style closed-form subtraction with multimodal FL gradient purification exactly this way. It's a legitimate stretch contribution, not a safe one — you'd want to validate the disentanglement assumption empirically early, before betting the whole paper on it. |

---

## 13. Approach 3 — Modality-Subset-Wise Fusion Aggregation (FedAMM-style, Routed by Subset)

### The Idea
Instead of FedAMM's per-*class* aggregation, maintain separate fusion/prototype tracks per modality **subset** — one track for {M,P}, one for {P,C}, one for {M,P,C} — and route each client to the track matching its available/required modality subset, keeping tracks isolated from each other.

### Closest Prior Work — including an important correction to Part I

**FedAMM, re-examined.** The original Part I analysis characterized FedAMM as aligning everything toward a single shared multimodal teacher prototype. A closer look this pass shows that's not quite the full picture: FedAMM actually computes **global centroid prototypes indexed by modality combination**, and does "modality-weighted aggregation" and "modality-specific encoder aggregation" — samples are compared against the centroid prototype for *their own* modality combination, not one universal centroid. So FedAMM already has some sub-structure by modality combination.

*What still differentiates your idea from FedAMM, then, is not the existence of subset-aware prototypes — FedAMM has that — but the hard-separation requirement.* FedAMM still converges to **one global model** that generates predictions for any modality combination at inference time; the per-combination prototypes are internal alignment aids inside a shared model, not hard partition walls that block information from crossing between subsets. Your version, by contrast, would need subsets to stay genuinely isolated — no eventual re-merging into one global model.

**FedMSplit (Chen & Zhang, KDD 2022).**
Uses a dynamic, multi-view graph structure to adaptively capture correlations among multimodal client models, without assuming similar active sensors across clients — i.e., it adaptively decides who should aggregate with whom based on learned correlation, rather than a fixed subset partition. Close in spirit (non-uniform, modality-aware routing) but the routing signal is learned graph structure, not an explicit fixed subset.

**MMiC (2025) — clustered FL for modality incompleteness** (already in Part I). Groups clients and replaces parameters of incomplete-modality clients using rate-of-variation signals from complete-modality clients. Same "cluster/route by modality availability" spirit, different mechanism (parameter substitution vs. subset-routed prototype alignment).

**General clustered FL** (CFL, IFCA, PACFL, FedSPD, Multi-Center FL) is a mature toolbox — clustering by gradient similarity, parameter similarity, or loss — but none of it clusters specifically by **modality subset** as the signal, combined with FedAMM-style prototype alignment as the within-cluster mechanism.

### Verdict

| | |
|---|---|
| **Already implemented?** | **Not as this exact combination**, but both halves (modality-subset clustering à la MMiC, and prototype-based alignment à la FedAMM) are independently mature and proven. |
| **Feasibility** | **High — the most implementable of the three ideas.** It's a natural, low-risk extension rather than a leap: swap MMiC's clustering signal from "similarity" to "exact modality subset," and use FedAMM's existing per-combination prototype machinery inside each cluster instead of across all of them. |
| **Residual novelty** | Moderate. The genuine novel piece is enforcing *hard* isolation between subsets (never re-merging into one shared model), which existing subset-aware methods don't do. |

---

## 14. Critical Update: DisentAFL's Effect on the Part I Novelty Verdict

This needs to be said plainly, because it matters for how you and Professor Jerry position the paper: **DisentAFL was not found during the Part I search, and it is close enough to the *original* professor's idea — not just to Approach 2 above — that it changes the novelty picture from §5 of Part I.**

DisentAFL formulates "Modality-task Agnostic FL," explicitly built around **asymmetric knowledge relationships among clients** caused by modality gaps, task gaps, and domain shifts — and proposes a two-stage Knowledge Disentanglement and Gating mechanism that decomposes an asymmetric inter-client sharing scheme into independent, semantically-typed components, each routed only to the clients that benefit from it. That is genuinely close to "asymmetric, policy-based modality gradient routing."

**What's still different, and still yours to claim:**

- **Optimization target is different.** DisentAFL's gating is *learned to maximize positive transfer and minimize negative transfer* — it routes knowledge wherever it helps performance. The professor's framing is the opposite kind of constraint: a client should be blocked from a modality **even if it would help**, because of an ethics-board limit or an MoU, not because the model decided it wasn't useful. Hard consent constraints and soft performance-driven gating are different problems that happen to produce structurally similar architectures.
- **No explicit policy/consent matrix.** DisentAFL's gate is an internal, learned, opaque mechanism. It's not an auditable, institution-defined R[i][j][m] matrix that a hospital's legal team could inspect and sign off on. That auditability is arguably essential for the real-world motivation here.
- **No healthcare/hospital framing**, and no explicit treatment of complete (vs. partial) missing modality as the operating assumption.

**Practical implication:** any future paper draft — regardless of which of the three approaches above ends up in the architecture — **must cite and explicitly differentiate against DisentAFL.** It's the single closest prior work found across both search passes, closer than FedMFS or MAFS from Part I. It doesn't eliminate the novelty of the consent-aware framing, but it substantially narrows the "nobody has done anything like this" claim from Part I §5. That claim should be softened to: *"asymmetric knowledge-routing exists (DisentAFL), but it optimizes for performance transfer, not for hard institutional consent enforcement — no existing work treats the routing policy as an external, auditable constraint rather than a learned, performance-driven one."*

---

## 15. Summary Table

| Approach | Core Mechanism | Closest Prior Work | Already Implemented? | Feasibility | Key Risk |
|---|---|---|---|---|---|
| **1. Unimodal encoders + prototypes, client-side fusion** | Server distributes unimodal-only components; fusion happens locally | FedCMD (unimodal-devices), FedProto, MFCPL | Yes, substantially | High | Low residual novelty on its own |
| **2. Gradient subtraction / vector arithmetic** | Extract modality-pure gradient by subtracting unimodal from fused | Task Arithmetic, weight disentanglement, DisentAFL (learned analog), additive/score-level fusion | Not as literal arithmetic for this purpose | Medium–Low | Depends on weight disentanglement holding + additive fusion assumption |
| **3. Modality-subset-wise aggregation** | FedAMM-style prototypes, but hard-routed by modality subset instead of class | FedAMM (re-examined), FedMSplit, MMiC, general clustered FL | Not as this exact combination | High | Needs the "hard isolation" piece to differentiate from FedAMM |

---

## 16. Recommendation — How These Fit Together

None of these three ideas needs to be chosen in isolation — and honestly, none of them is where the paper's novelty should live.

- **Approach 1 gives you a working architecture** (client-side fusion over server-distributed unimodal components) — use it as the backbone, but recognize it's largely "borrowed," not new.
- **Approach 3 is the natural refinement on top of it** — route unimodal component distribution by modality subset rather than treating every client identically, and keep subsets hard-isolated rather than letting FedAMM-style methods re-merge everything into one global model.
- **Approach 2 is the intellectually interesting stretch goal** — worth a paragraph as a theoretical framing or an ablation ("what if fusion were additive — could we recover pure modality gradients?"), but too risky to build the core paper around, given how load-bearing the weight-disentanglement assumption is.
- **The actual novel contribution, across all three, is the same as it was in Part I: the explicit, auditable, institution-defined consent/MoU matrix that hard-blocks routing regardless of whether blocking helps performance.** That's the piece absent from FedCMD, FedProto, MFCPL, FedAMM, FedMSplit, MMiC, *and* DisentAFL. Build the architecture from (1) + (3), and spend the paper's actual argumentative energy on why hard consent constraints are a different — and under-addressed — problem from learned performance-driven gating.

---

## 17. Reading List Additions (for `idea_analysis_instructions.md`)

Papers surfaced this pass that are directly relevant and worth reading in full:

- **FedCMD** — *Cross-Modal Federated Learning among Unimodal Devices* (2025) — confirmed read by user
- **FedProto** — Tan et al., AAAI 2022
- **MFCPL** — *Cross-Modal Prototype based Multimodal Federated Learning under Severely Missing Modality*, Le et al., arXiv 2401.13898 (2024)
- **DisentAFL** — Chen & Zhang, *On Disentanglement of Asymmetrical Knowledge Transfer for Modality-Task Agnostic Federated Learning*, AAAI 2024 — **highest priority read; closest prior work found to date**
- **Task Arithmetic** — Ilharco et al., *Editing Models with Task Arithmetic*, ICLR 2023 (arXiv 2212.04089)
- **FedMSplit** — Chen & Zhang, KDD 2022

---

*Part II literature searched: FedCMD (both versions), FedProto, MFCPL, Task Arithmetic, weight disentanglement literature, DisentAFL, FedMSplit, MMiC, FedAMM (re-examined), general clustered FL (CFL, IFCA, PACFL, FedSPD), additive/score-level multimodal fusion, FL consent/governance literature.*

---
---

# Part III — Combined Architecture (Approach 1 + Approach 3) and Dataset Selection

---

## 18. Combined Architecture: Federated Unimodal Encoders + Hard-Isolated Subset-Fusion Tracks

### 18.1 Design Principle

The two approaches slot together along a natural seam in a segmentation (or classification) network: **everything upstream of fusion stays global per modality (Approach 1); everything from fusion onward stays isolated per modality subset (Approach 3).**

- **Encoders are global, per modality.** Encoder_M is aggregated across every consenting client that owns MRI, regardless of what else they own. A PET/CT-only client (Client A) never touches Encoder_M at all — it's never broadcast to them, because they never request it.
- **Fusion + decoder are local to a modality-subset track, never merged across tracks.** Client A (owns P, C) and Client C (owns M, C) never share a fusion module, even though both include CT — because their *subsets* differ. This is the hard-isolation piece that FedAMM doesn't have (FedAMM's per-combination prototypes still feed into one eventual global model; here, subset tracks never re-merge).

### 18.2 Component Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                              SERVER                                  │
│                                                                        │
│  UNIMODAL BANK  (Approach 1 — global per modality, policy-gated)     │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐                    │
│  │ Encoder_M  │   │ Encoder_P  │   │ Encoder_C  │                    │
│  │ Proto_M^c  │   │ Proto_P^c  │   │ Proto_C^c  │                    │
│  └────────────┘   └────────────┘   └────────────┘                    │
│   avg over all      avg over all      avg over all                   │
│   M-owning,          P-owning,         C-owning,                     │
│   consenting clients consenting clients consenting clients            │
│                                                                        │
│  SUBSET-FUSION TRACKS  (Approach 3 — hard-isolated, no cross-merge)  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐     │
│  │ Track {P,C}     │  │ Track {M,P,C}    │  │ Track {M,C}     │     │
│  │ FusionHead_PC   │  │ FusionHead_MPC   │  │ FusionHead_MC   │     │
│  │ Decoder_PC      │  │ Decoder_MPC      │  │ Decoder_MC      │     │
│  │ FusedProto_PC^c │  │ FusedProto_MPC^c │  │ FusedProto_MC^c │     │
│  └─────────────────┘  └──────────────────┘  └─────────────────┘     │
│    avg only among       avg only among        avg only among         │
│    {P,C}-subset          {M,P,C}-subset        {M,C}-subset          │
│    clients               clients               clients               │
└─────────┬──────────────────────┬──────────────────────┬──────────────┘
          │                      │                      │
  ┌───────▼────────┐   ┌─────────▼─────────┐   ┌────────▼───────┐
  │   CLIENT A      │   │    CLIENT B        │   │   CLIENT C     │
  │  Owns: P, C     │   │  Owns: M, P, C     │   │  Owns: M, C    │
  │                 │   │                    │   │                │
  │ Encoder_P (loc.)│   │ Encoder_M (loc.)   │   │ Encoder_M(loc.)│
  │ Encoder_C (loc.)│   │ Encoder_P (loc.)   │   │ Encoder_C(loc.)│
  │ FusionHead_PC   │   │ Encoder_C (loc.)   │   │ FusionHead_MC  │
  │ Decoder_PC      │   │ FusionHead_MPC     │   │ Decoder_MC     │
  │                 │   │ Decoder_MPC        │   │                │
  └─────────────────┘   └────────────────────┘   └────────────────┘
```

### 18.3 What Each Piece Actually Is (segmentation instantiation)

For a segmentation task (tumor delineation), each component maps onto a standard U-Net-style network split at the fusion point:

- **Encoder_m** — a modality-specific U-Net-style down-sampling path (with skip connections) for modality *m*. Trained on any client owning *m*, aggregated globally across all such consenting clients.
- **Proto_m^c** — the mean embedding vector for class *c* (e.g., tumor / background), computed from Encoder_m's bottleneck output, averaged (FedProto-style) across all consenting owners of modality *m*.
- **FusionHead_S** — combines the available encoder outputs (at each skip-connection resolution) for subset *S*, e.g. via concatenation + 1×1 conv, or cross-attention across modalities.
- **Decoder_S** — the up-sampling path that takes FusionHead_S's fused multi-scale features and produces the segmentation mask. Bundled with FusionHead_S as a single subset-specific unit.
- **FusedProto_S^c** — the mean fused embedding for class *c*, computed after fusion, averaged only across clients sharing subset *S*.

### 18.4 Loss Functions (per client, per round)

For a client owning subset *S* = {available modalities}:

1. **Task loss** — standard segmentation loss (Dice + cross-entropy) on the final Decoder_S output.
2. **Unimodal alignment loss** — for each owned modality *m*, a contrastive/prototype loss pulling Encoder_m's local embeddings toward Proto_m^c (same class) and pushing away from Proto_m^{c'} (different classes). *This is the fix for Gap 3 (gradient purity) that Approach 1 alone didn't solve* — encoders are only ever aligned to a same-modality prototype, never to a fused/complete one, so a PET encoder cannot be indirectly shaped by MRI-derived signal through this term.
3. **Subset-fusion alignment loss** — pulling the local fused embedding toward FusedProto_S^c, keeping intra-subset consistency without letting information leak to other subsets (since FusedProto_S^c is only ever computed and shared within subset *S*).

Total local loss: `L = L_task + λ₁ · L_unimodal_align + λ₂ · L_fusion_align`

### 18.5 Aggregation Rules

- `Encoder_m ← weighted_avg({ Encoder_m^(i) : client i owns m AND R_encoder[i][m] = 1 })`
- `Proto_m^c ← weighted_avg({ Proto_m^c,(i) : same group })`
- `FusionHead_S, Decoder_S ← weighted_avg({ FusionHead_S^(i), Decoder_S^(i) : client i's subset == S })` — **never averaged across different S**
- `FusedProto_S^c ← weighted_avg({ FusedProto_S^c,(i) : same subset group })`

`R_encoder[i][m]` is the policy gate from Part I — by default "owns modality *m* and has ethics/MoU clearance to pool it," but can be restricted further (e.g., two MRI-owning hospitals that still don't want to pool with each other would simply form two separate M-encoder groups instead of one).

### 18.6 Training Loop

```
Each round t:
  1. Server broadcasts to client i:
       - Encoder_m and Proto_m^c for each m in S_i
       - FusionHead_{S_i}, Decoder_{S_i}, FusedProto_{S_i}^c
  2. Client i trains locally for E local epochs using L = L_task + λ₁L_unimodal_align + λ₂L_fusion_align
  3. Client i uploads:
       - Updated Encoder_m, Proto_m^c for each owned m (to the Unimodal Bank)
       - Updated FusionHead_{S_i}, Decoder_{S_i}, FusedProto_{S_i}^c (to its own subset track only)
  4. Server aggregates per §18.5
  Repeat until convergence
```

### 18.7 Handling a New Client with an Unseen Subset

If a new client arrives with a modality subset that has no existing track (e.g., a hospital with only PET), cold-start its FusionHead/Decoder from the nearest existing superset track (average down from {M,P,C} or {P,C}, whichever shares the most modalities), rather than training from scratch. This is a practical patch, not a solved problem — it's flagged as an open question in Part I §10 and remains one here.

### 18.8 Reconciling with the Original Directional Policy Matrix (Part I)

The Part I architecture used a fully general, potentially asymmetric R[i][j][m] matrix (client-pair-level, not just group-level). This combined design is a **special case** of that: it assumes consent groups are symmetric within a modality (if i and j both own m and both consent, they simply average together) rather than fully directional (i receives from j but not vice versa). If you need true directional asymmetry — B accepts everyone but doesn't reciprocate — replace the plain average in §18.5 with a weighted, non-symmetric combination using the original R[i][j][m] weights per recipient. The two designs are compatible; this one is the simpler, more implementable starting point.

---

## 19. Dataset Selection

### 19.1 Setting: Segmentation, Not Classification — and Why

FedCMD's evaluation setting is **not** medical imaging at all — it's confirmed to use the UCI **Human Activity Recognition** (smartphone sensor) dataset, a classification task over accelerometer/gyroscope signals. That setting doesn't map onto the professor's MRI/PET/CT scenario at all; it was just the paper that most closely matched Approach 1's *mechanism*, not its *domain*.

FedAMM, by contrast, is confirmed evaluated on **BraTS2020** for brain tumor **segmentation**. Segmentation is also the clinically natural task for MRI/PET/CT data (tumor/lesion delineation for treatment planning), so it — not FedCMD's classification setting — is the right template to follow.

### 19.2 The Dataset Problem: BraTS Doesn't Actually Have PET or CT

This is worth flagging clearly: **BraTS is MRI-only.** Its four "modalities" (T1, T1ce, T2, FLAIR) are all MRI sequences, not separate imaging technologies. If you use BraTS, you'd be treating four MRI sequences as pseudo-modalities/pseudo-machines — a common simplification in the missing-modality literature, but not a literal match to the professor's M/P/C scenario, since there's no real PET or CT in BraTS at all.

### 19.3 Primary Recommendation: TCIA Soft-Tissue-Sarcoma (STS)

This dataset genuinely has all three: a cohort of 51 patients with soft-tissue sarcomas of the extremities, each with paired pre-treatment **MRI (T1 and T2) and FDG-PET/CT scans**, publicly available via The Cancer Imaging Archive. After typical preprocessing (cropping to the leg region), one prior study retained 39 usable patients. Tumor annotations exist for the T2 MRI and were separately delineated on PET for research use.

This is a strong match for three reasons:
- It's the only public, genuinely tri-modal (MRI + PET + CT) medical imaging dataset commonly used in this literature — not a simulated stand-in.
- Prior work on exactly this dataset (a co-segmentation paper using modality-specific encoders and decoders, supporting inference on *any subset* of the input modalities) is architecturally almost identical to what you're building — it's both a strong precedent and a ready-made baseline.
- Data comes from different sites and scanners, giving genuine (not simulated) inter-institutional heterogeneity.

**Caveat — size.** 51 (39 post-crop) patients is small for deep learning, and dangerously small once split across 3+ simulated FL clients. Two mitigations:
- Train at the **2D slice level**, not the patient/volume level — each patient contributes many axial slices, multiplying the effective sample count (this is what most STS papers already do).
- Consider the client split in §19.5 below, which avoids further shrinking an already-small cohort.

### 19.4 Secondary Recommendation: BraTS2020/2021 (for scale, and direct FedAMM comparability)

Once the architecture is validated on STS, re-run at scale on BraTS2020 (~370 patients) or BraTS2021 (~2,000), treating T1/T1ce/T2/FLAIR as pseudo-modalities. This buys two things STS can't: enough data for statistically meaningful splits across many simulated clients, and **direct comparability to FedAMM**, since it uses this exact dataset — letting you report a head-to-head number against the closest prior work.

### 19.5 Concrete Simulated Client Split (mapping the professor's exact scenario onto STS)

Given STS's small size, the cleanest first experiment is to simulate modality *ownership* as different **views of the same shared patient pool**, rather than splitting patients into disjoint groups (which would leave ~13 patients per client — too small to draw conclusions from):

| Simulated Client | Modalities Used per Patient | Analogy |
|---|---|---|
| Client A | PET + CT only (MRI discarded) | Clinic without an MRI machine |
| Client B | MRI + PET + CT | Full teaching hospital |
| Client C | MRI + CT only (PET discarded) | Clinic without a PET scanner |

All three clients train on the *same* underlying patients, just with different modality access — this isolates the variable you actually care about (modality-consent routing) from a confounding population-distribution shift, which is a separate and harder problem you don't need to solve at the same time. Once this works, a second, more realistic experiment can move to genuinely disjoint patient populations per client (accepting the smaller per-client *n*), or scale up via BraTS as in §19.4.

### 19.6 Optional Complement: HECKTOR (head & neck tumor, PET/CT)

If you want a supplementary experiment with *real* (not simulated) multi-institutional heterogeneity, HECKTOR contains 224 cases from five different real centers, all with paired PET and CT. It lacks MRI, so it only covers a 2-modality subset of the scenario — useful as a robustness check on the {P,C}-only pairing, not as the primary dataset.

---

## 20. Summary of Part III

| Question | Answer |
|---|---|
| Combined architecture | Encoders + unimodal prototypes stay global per modality (Approach 1); fusion head + decoder + fused prototypes stay hard-isolated per modality subset (Approach 3) |
| Primary dataset | TCIA Soft-Tissue-Sarcoma — genuinely tri-modal (MRI+PET+CT), small but real |
| Scale/comparison dataset | BraTS2020/2021 — matches FedAMM directly, MRI-only (pseudo-modalities) |
| Task setting | Segmentation (not FedCMD's classification setting — that domain doesn't transfer here) |
| Recommended first split | Shared patient pool, different modality access per simulated client — isolates the consent-routing variable cleanly |

