# Standards & Instructions for Visual Stepper Widgets
## CAMFS Unified Architecture (Method 2 + Patch as One)

This document establishes the mandatory standards, mathematical formatting rules, slide structures, and domain invariants for creating all visual stepper widgets in the `method 2 + patch = as one` research folder.

---

## 1. Interactive Widget Development Workflow

1. **One Widget Per Request:**  
   Generate **one standalone HTML widget file** at a time (e.g. `widget1_unimodal_alignment.html`, `widget2_freeze_transition.html`).
2. **Interactive Revision & Deepening:**  
   After generating a widget, the user will inspect the slides, ask clarifying questions on concepts/formulas, and request enhancements. The agent enriches the widget step-by-step to ensure total conceptual clarity before moving to the next widget.
3. **Standalone Portability:**  
   Each widget HTML file must be fully self-contained (HTML, CSS, JS engine, KaTeX rendering) so it can be opened directly in any web browser without external server setup.

---

## 2. KaTeX & Mathematical Formatting Standards

1. **Mandatory KaTeX LaTeX Rendering:**  
   Every mathematical formula, tensor shape, subscript, caret, or variable MUST be formatted in LaTeX math syntax (`$ ... $` for inline math, `$$ ... $$` for display equations).
2. **Strict Rule Against Raw ASCII Notations:**  
   NEVER output raw ASCII text with un-rendered underscores or carets.
   - ❌ **Incorrect:** `x_T1`, `Proto_T1^1`, `sim(z_T1(p), Proto_T1^1)`, `dL / d(theta_{Encoder_T1})`
   - ✅ **Correct:** `$x_{\text{T1}}$, $\text{Proto}_{\text{T1}}^{c=1 \text{ [NCR] Stamford}}$, $\text{sim}(z_{\text{T1}}(p), \text{Proto}_{\text{T1}}^{c=1})$, $\frac{\partial \mathcal{L}}{\partial \theta_{\text{Encoder}_{\text{T1}}}}$`
3. **Math Blocks Over Raw Code Blocks:**  
   Do NOT place math equations inside monospaced `<pre>` code blocks (which KaTeX ignores). Use styled `.math-block` containers so equations compile into crisp, publication-grade LaTeX typesetting.
4. **KaTeX Integration Template:**  
   Include the following CDN tags in the `<head>` of every widget HTML:
   ```html
   <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
   <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
   <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderLaTeX();"></script>
   ```

---

## 3. Mandatory 5-Part Slide Structure (Per-Step Data Necessity)

Every slide in every widget MUST contain the following 5 structured elements:

1. **`intro-box` (Intuition & Purpose):**  
   A prominent green-bordered card at the top answering:
   - *What are we doing in this step?*
   - *Why do we do it?*
   - *What core problem does it solve?*
2. **Metric & Symbol Introduction:**  
   Plain-English explanation of any new variables, mathematical operations, similarity functions ($\text{sim}$), temperature parameters ($\tau$), evidence weights ($w_{m,i}^c$), or drift thresholds ($\epsilon$).
3. **Explicit Ground-Truth Class Bracket Notation:**  
   Explicitly append numerical class IDs in brackets whenever classes or prototypes appear:
   - `[c=0: BG]` — Background & Healthy Brain
   - `[c=1: NCR/NET]` — Necrotic Core & Non-Enhancing Tumor
   - `[c=2: ED]` — Peritumoral Edema
   - `[c=4: ET]` — Enhancing Tumor
4. **Structured Math Block (`.math-block`):**  
   A padded container detailing concrete input/output tensor shapes, gradient norm statuses, numerical calculation steps, or vector spaces ($\mathbb{R}^{256}$).
5. **Highlight Box (`.highlight-green` / `.highlight-box` / `.highlight-warn`):**  
   A takeaway card emphasizing structural isolation, information conservation ($1 \times 240 \times 240 = 57,600 \to 256 \times 15 \times 15 = 57,600$), or SOTA benchmark standardization (e.g. matching FedAMM / FedProto 1:1).

---

## 4. CAMFS + BraTS Architectural & Methodological Invariants

When building any widget in this project, enforce these exact domain invariants:

1. **Latent Bottleneck Dimension ($D=256$):**  
   The unimodal U-Net bottleneck feature maps and prototypes MUST have 256 channels ($z_m \in \mathbb{R}^{256 \times 15 \times 15}$, $\text{Proto}_m^c \in \mathbb{R}^{256}$). This matches MICCAI SOTA baselines (FedAMM / FedProto) 1:1.
2. **The $4 \times 4 = 16$ Prototype Rule:**  
   Across the 4 modalities ($\text{T1, T1ce, T2, FLAIR}$) and 4 ground-truth dataset classes ($c \in \{0, 1, 2, 4\}$), the server maintains **$4 \times 4 = 16$ total unimodal class prototype vectors** in $\mathbb{R}^{256}$.
3. **Positive Pair Identification Rule:**  
   A Positive Pair in InfoNCE loss is formed whenever the prototype's class index $c$ matches the downsampled ground-truth mask label $c(p)$ at spatial coordinate $p$:
   $$\text{Positive Pair:} \quad \left( z_m(p),\, \text{Proto}_m^{c(p)} \right)$$
   Negative pairs are formed with the remaining 3 non-matching class prototypes.
4. **The "Shared Feature Language" Goal:**  
   Phase 1 contrastive alignment forces all 4 modality prototypes for class $c$ into the exact same region of $\mathbb{R}^{256}$:
   $$\text{Proto}_{\text{T1}}^{c=1} \approx \text{Proto}_{\text{T1ce}}^{c=1} \approx \text{Proto}_{\text{T2}}^{c=1} \approx \text{Proto}_{\text{FLAIR}}^{c=1} \in \mathbb{R}^{256}$$
   This ensures all unimodal encoders speak the exact same 256-D feature language before Phase 2 fusion begins.
5. **Clinical Evaluation Sub-Regions:**  
   Final segmentation metrics are evaluated on the 3 standard BraTS clinical sub-regions:
   - **Whole Tumor (WT):** Classes $c \in \{1, 2, 4\}$
   - **Tumor Core (TC):** Classes $c \in \{1, 4\}$
   - **Enhancing Tumor (ET):** Class $c=4$ only
   - **Primary Metrics:** Dice Similarity Coefficient (DSC %), 95th Percentile Hausdorff Distance (HD95 in mm), Compliance-Performance Gap ($\Delta_{\text{compliance}}$), and Purity Probe Accuracy ($\text{Acc}_{\text{probe}}$).
6. **Structural Isolation & Freeze Transition:**  
   Phase 1 and Phase 2 are strictly decoupled:
   - Phase 1: Trains unimodal encoders using prototype contrastive loss. No decoders or fusion heads exist in the graph.
   - Transition: Phase Controller locks encoder weights ($\theta_{E_m}^*$) and inserts a graph-level `detach()` stop-gradient wall.
   - Phase 2: Trains track fusion heads and decoders on frozen embeddings. Fusion gradients hit the `detach()` wall and CANNOT reach encoder weights:
     $$\frac{\partial \mathcal{L}_{\text{task}}}{\partial \theta_{E_m}^*} \equiv 0$$

---

## 5. Summary Checklist for Widget Authors

Before delivering any new HTML widget file, verify:
- [ ] Is KaTeX loaded in `<head>` and rendering all math properly?
- [ ] Are all raw underscores and carets eliminated in favor of LaTeX math?
- [ ] Does every slide start with an intuitive `intro-box`?
- [ ] Are all 4 ground-truth dataset classes written with explicit bracket notation `[c=0: BG, c=1: NCR, c=2: ED, c=4: ET]`?
- [ ] Are math blocks styled cleanly without monospaced code box artifacts?
- [ ] Are key takeaways highlighted in color-coded cards?
