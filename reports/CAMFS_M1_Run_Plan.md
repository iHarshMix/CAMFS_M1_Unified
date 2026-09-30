# CAMFS M1: run plan to submission

Everything left before we submit. Work top to bottom: groundwork first, then the runs, then the paper fixes.

| | |
|---|---|
| **Target** | IEEE JBHI (stretch: IEEE TMI) |
| **Setting** | Partition 1103, seeds 17, 29, 43 |
| **Compute** | ≈ 25 full-run equivalents |
| **First step** | Hours per run (item 2) |

Live checklist, where you can tick items and add results: https://claude.ai/artifact/YHHDhWcMiNm9CoVNijM83d

---

## Three things that changed the plan

- **Borrowing S4 already beats H2's private head.** S4 is H4's checkpoint, scored on the same 50 test patients: 81.07 against 78.45. Items 12–14 settle how the private head is presented.
- **The λ₁ sweep would change nothing.** Phase 1 has one loss term and AdamW cancels its scale, so λ₁ = 0.1, 0.5 and 1.0 train almost identically. A4 is redesigned (item 17).
- **Nothing in Phase 1 aligns the modalities with each other.** Each encoder is pulled toward its own modality's prototypes only, so the "shared language" claim is untested (item 5).

## Rules for every run

- Every method uses the same split files, preprocessing, 2D slice sampling, augmentation, evaluation code and seeds.
- Save a per-patient CSV for every run (item 3).
- Ablations retrain only the part they change. Tracks share no parameters and each Phase-1 encoder trains alone, so everything else is reused.
- "Compliant" means H2's T1ce is treated as missing in H2's training data. At test time every hospital uses every modality it owns.

---

## Phase 1: Groundwork

No training. These make every later number trustworthy.

### 1. Settle on one configuration
*No training*

The spec and the PDF disagree on τ (0.1 vs 0.07), the Phase-2 loss, the round schedule, which encoder checkpoint gets frozen, encoder depth, slice sampling and batch size.

- For each of these, check what the code that produced the Primary results actually does. The code is the truth.
- Write it all into one file, `configs/m1_primary.yaml`, and log its SHA-256 in the ledger.
- Update the spec and the PDF to match.

**Done when:** every run from now on loads that file, and both documents agree with it.

- [ ] Done · result:

### 2. Time one run and check determinism
*No training*

A3 changed nothing, yet it moved S1 by 0.4 points and S3's enhancing-tumour Dice by 3 points. We need to know why, and whether this plan fits the H100 time.

- Search the code for `use_deterministic_algorithms`. If it has `warn_only=True`, runs aren't bit-for-bit reproducible: bilinear upsampling has no deterministic GPU backward pass. Don't change the network; we'll state it in the paper.
- Record the wall-clock hours of one full run (Phase 1, Phase 2 and the H2 head).
- No separate repeat runs: the spread across seeds 17, 29 and 43 (item 6) is the noise level.

**Done when:** Jerry has the hours per run.

- [ ] Done · result:

### 3. Save per-patient results for every run
*No training*

One 371 mm HD95 penalty moves a 50-patient mean by 7.4 mm, and one patient with no enhancing tumour moves ET Dice by 2 points. Every number has to be traceable to patients.

- One CSV per run and track with `patient_id`, WT/TC/ET Dice, WT/TC/ET HD95, `gt_empty`, `pred_empty` and `penalty_applied`.
- Report HD95 as the median plus the number of penalty cases.
- Recompute the penalty from the real grid: 371.4 mm, not 372.1.
- Fix the test list: the PDF's ends at 367, the ablation report's at 369. Name the case that was dropped to get from 369 patients to 368.
- Count low-grade (LGG) vs high-grade (HGG) test patients from BraTS's `name_mapping.csv`.

**Done when:** every number in every table can be recomputed from the CSVs.

- [ ] Done · result:

### 4. Check how S4 was initialised in Primary
*No training*

The spec says Primary already seeds S4 from S2′ with Net2Net. If that's true, the A5 run was a repeat of Primary.

- In Primary's ledger file, find the record where track S4 was created.
- Read its `seed_mechanism` and `round` fields.

**Done when:** we know how S4 was seeded and at which round.

- [ ] Done · result:

### 5. Test the "shared language" claim
*No training · 5 min*

Phase 1 pulls each encoder toward its own modality's prototypes only. Nothing in the loss makes the T1 and T1ce prototypes of a class match.

- Load the Phase-1 prototype bank.
- For each class, compute the cosine similarity between modalities: T1 vs T1ce, T1 vs T2, T1 vs FLAIR, and so on.

**Done when:** the similarities are reported. If they aren't close, the "shared language" text in spec §5.2 and §16.3 is removed.

- [ ] Done · result:

---

## Phase 2: Main comparison

Partition 1103, seeds 17, 29 and 43, all scored on the same 50 test patients. At test time each hospital uses every modality it owns.

### 6. CAMFS Primary, seeds 29 and 43
*Full run · 2 runs*

- Run Primary with the item-1 configuration and seeds 29 and 43.
- If item 1 changed anything from the original run, rerun seed 17 as well.

**Done when:** three seeds of per-patient CSVs exist.

- [ ] Done · result:

### 7. FedAvg, compliant
*Full run · 3 runs*

- Build it on the CAMFS network with all four input slots; missing modalities are filled with zeros.
- Compliant: H2's T1ce is treated as missing in H2's training data, so nothing H2 uploads is computed from it.
- At test time H2 still feeds in its own T1ce. Nothing leaves the hospital.

**Done when:** three seeds, per-patient CSVs, parameter count noted.

- [ ] Done · result:

### 8. FedAMM, compliant
*Full run · 3 runs*

- Use the official code (`github.com/13sky/FedAMM`).
- Replace only the data loader and the evaluation with ours.
- Same compliant rule as item 7. Note the parameter count.

**Done when:** three seeds, per-patient CSVs, parameter count noted.

- [ ] Done · result:

### 9. FedMEPD (or FedMEMA), compliant
*Full run · 3 runs*

FedMEPD (Medical Image Analysis, 2025) is the closest prior architecture: modality-specific encoders with partly personalised fusion decoders.

- Check whether FedMEPD code has been released; if not, email the authors.
- If it isn't available in time, run FedMEMA instead (`github.com/QDaiing/FedMEMA`).
- Same data loader, evaluation and compliant rule as item 7.

**Done when:** three seeds, per-patient CSVs, parameter count noted.

- [ ] Done · result:

### 10. Local-only: each hospital alone
*Local only · 12 small runs*

- Each hospital trains the CAMFS network for its own modality set, end to end with Dice+CE, on its own training patients only.
- Use every modality the hospital owns; H2 includes T1ce, because local use is allowed.
- Select on the hospital's own validation patients; evaluate on the 50 test patients.
- 4 hospitals × 3 seeds.

**Done when:** there is a local-only score for every hospital.

- [ ] Done · result:

### 11. Centralised ceiling
*Full run · 3 runs*

- Pool all four hospitals' training patients.
- Train one model each for {T1, T1ce, T2, FLAIR}, {T1, T1ce, T2} and {T1, FLAIR}, selecting on the pooled validation patients. One seed each.
- Evaluate each hospital with the model for its own modality set.

**Done when:** there is a ceiling score for every hospital.

- [ ] Done · result:

---

## Phase 3: H2's alternatives

H2 withholds T1ce but may use it locally. The paper has to show how the private head compares with the obvious ways H2 could comply. One is already known: borrowing H4's S4 model scores 81.07.

### 12. Drop T1ce: score the S2′ model
*Inference only*

- Evaluate `best_track_S2.pt` on the 50 test patients using T1 and T2 only.

**Done when:** the S2′ score (macro, WT, TC, ET) is reported.

- [ ] Done · result:

### 13. Borrow S4, then fine-tune at H2
*Short local · 3 runs*

- Start from `best_track_S4.pt`.
- Fine-tune on H2's 64 training patients with T1, T1ce and T2; choose the best epoch on H2's 16 validation patients; same stopping rules as the private head.
- Nothing is sent. This assumes H1 and H4 agree to share S4 with H2.

**Done when:** we can choose the story: borrow and fine-tune when a matching track exists, the private head when none does (item 21 creates that case).

- [ ] Done · result:

### 14. FedAvg, then fine-tune at H2
*Short local · 3 runs*

This stands in for personalised FL (FedPer or FedRep).

- Take the compliant FedAvg model from item 7.
- Fine-tune it at H2 with T1ce exactly as in item 13.

**Done when:** three seeds scored on the 50 test patients.

- [ ] Done · result:

---

## Phase 4: Consent audit

The paper's central claim is that other methods leak what a site withholds and CAMFS cannot. No violation has been measured yet.

### 15. Tag every update and count violations
*2–3 rounds each · 2 short runs*

- In each method's client training step, tag every uploaded update with the modalities present in the inputs that produced it. CAMFS already records this as `image_lineage`.
- The server logs sender, round and tag set.
- V = the number of updates from a site whose tags include a modality that site doesn't send.
- Run policy-blind FedAvg and FedAMM (H2 trains with T1ce) for 2–3 rounds only; the first upload already shows the violation. Take CAMFS's V from item 6.

**Done when:** there is a table of V per method and hospital. Expected: above 0 for the policy-blind runs, 0 for CAMFS.

- [ ] Done · result:

---

## Phase 5: Ablations

Partition 1103, seeds 17, 29 and 43. Retrain only the part each ablation changes and reuse the rest.

### 16. A1: joint training (no freeze)
*Full run · 2 runs*

- Rerun with seeds 29 and 43; seed 17 exists.
- Also log its V with the item-15 tagging: joint training lets S1's T1ce signal reach the shared T1 encoder, which H3 receives.

**Done when:** three seeds, plus its V.

- [ ] Done · result:

### 17. A4: does Phase 1 matter?
*2 tracks · 3 runs*

Replaces the λ₁ sweep, which would change nothing under AdamW.

- Freeze randomly initialised encoders in place of the Phase-1 encoders.
- Retrain only S3 and S4; everything else stays.

**Done when:** S3 and S4 scores with and without Phase 1, three seeds.

- [ ] Done · result:

### 18. A5: how S4 is initialised
*1 track · 6 runs*

- After item 4, retrain S4 alone two more ways: Tier 2 (warm-start rounds with the fused-alignment loss only) and Tier 3 (fresh random initialisation).
- Tier 1, Net2Net from S2′, is the Primary run.
- 2 arms × 3 seeds.

**Done when:** S4 scores for all three tiers.

- [ ] Done · result:

### 19. A7: S3 without H1's help
*1 track · 2 runs*

- Set `R_contribute(H1, S3) = 0` and retrain S3 only, seeds 29 and 43.
- Don't report an S1 change: S1 shares nothing with S3, so any change there is noise.

**Done when:** S3 with and without H1's help, three seeds.

- [ ] Done · result:

### 20. CAMFS with no consent restriction
*1 track · 3 runs*

- Set Send(H2) = {T1, T1ce, T2}, so H2 joins S4.
- Keep Phase 1 and S4's initialisation as in Primary; retrain S4 only.

**Done when:** H2's score with and without withholding. The difference is the cost of consent.

- [ ] Done · result:

### 21. Config B: H1 withholds FLAIR
*Partial run · 3 runs*

So far one site withholds one modality, and T1ce is the most confounded choice because the enhancing-tumour labels come from T1ce. No other site owns all four, so here the private head is H1's only way to use FLAIR.

- Set Send(H1) = {T1, T1ce, T2}.
- Retrain the FLAIR encoder (now H3 only), S3, S4 and H1's private head (S4 widened with FLAIR).
- Compare H1 against dropping FLAIR and against no restriction.

**Done when:** H1's three scores, three seeds.

- [ ] Done · result:

---

## Phase 6: Statistics

Run once all of Phases 2–5 are in.

### 22. Confidence intervals and tests
*No training*

- Per-patient paired differences, averaged over seeds.
- 95% bootstrap confidence intervals, 10,000 resamples (PCG64, seed 8803).
- Paired sign-flip tests with 100,000 vectors, Holm-corrected at α = 0.05.
- Report every hospital separately. Report V as exact counts, with no interval.

**Done when:** every comparison in the paper has an interval and a corrected p-value.

- [ ] Done · result:

---

## Phase 7: Document fixes

For the spec, the PDF and the ablation report. These can run alongside everything else.

- [ ] **23. Remove the α-table.** Delete the table built from FedAMM's published numbers. Cite FedAMM as context only; item 8 carries the comparison.
- [ ] **24. Remove the receive-side material.** Gap 1, R_recv, A6 and the "bidirectional consent" novelty claim go. Restate eq. (1) as a guarantee about what each sender uploads.
- [ ] **25. Present BraTS as a testbed.** Drop the hospital descriptions ("Community Hospital", "clinics without T2 scanners") and add: *"BraTS provides all four sequences for every subject, so ownership and send policies are imposed synthetically; we use it as a controlled testbed."*
- [ ] **26. Fix citations and novelty wording.**
  - FedMEPD is Medical Image Analysis 2025, not AAAI 2024; make FedAMM's year consistent.
  - Cite and distinguish MAFS, Partial FL and MFedMC.
  - Replace "no prior work" with "to our knowledge".
- [ ] **27. Fix the PDF's technical errors.**
  - §3.4 says the server re-keys H2's uploaded gradients. That is the leak CAMFS prevents; H2 trains on {T1, T2} itself.
  - Fig. 1 labels H2's private head "S4"; give it its own name.
  - Drop "pre-registered": the plan is dated after the first results.
- [ ] **28. Fix the tone.** Remove the emojis and "100% confirmed", "catastrophic" and "breakthrough". Write differences in percentage points (pp).

---

## Held back for revision

Designed and ready, but not run unless reviewers ask:

- Partitions 2207 and 3301
- FedPer or FedRep
- Whichever of FedMEPD and FedMEMA wasn't run
- Config C: a second site withholding a modality
- A4 with a different Phase-1 pretraining (per-modality segmentation)
- A2 (late-joining H4): already run, goes to the supplement as it is

---

*Written 21 Sep 2026 from the review of the spec, the ablation report and the 17 Sep PDF.*
