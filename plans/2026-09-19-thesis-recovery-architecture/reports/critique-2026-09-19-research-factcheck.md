# Adversarial fact-check: 3 research reports (2026-09-19)

Scope: researcher-01 (arch/domain-gen), researcher-02 (eval protocol), researcher-03 (demo/edge).
Cross-checked against audit-2026-09-19-repo-capability-and-feasibility.md. WebFetch/WebSearch used to
verify every flagged citation. Read-only, no files modified.

## TL;DR for the planner

Both researcher-01 and researcher-02 have a **systematic citation-date bug**: many arXiv IDs decode
(YYMM.NNNNN) to a date one full year later than what the report's inline year says, and at least
2 citations point to completely unrelated real papers (link resolves, but content ≠ claimed
content). One arithmetic/logic error in researcher-02's Q6 defence answer is genuinely
**defence-fatal** — do not let the student say this in front of a committee. Treat every
un-sourced "+X% LCO gain" estimate in researcher-01 as a guess, not a result — none of these
techniques have been run in this repo.

---

## 1. Citation verification table

Legend: VERIFIED = link resolves, correctly attributed. MISDATED = real paper, wrong year/venue
label. MISATTRIBUTED = link resolves to a **different, unrelated** real paper than what's claimed.
FABRICATED = no such thing exists as described. NOT-CHECKED = out of budget, flag for planner.

| # | Claim / cited as | URL | Verdict | Correct info |
|---|---|---|---|---|
| 1 | Video-Oasis (2024) — R01 §5 | pith.science/paper/2603.29616 | MISDATED | pith.science is real (pre-print review site, not primary source). Underlying paper real, arXiv 2603.29616, submitted **March 2026**, accepted **ECCV 2026**, authors Lim/Park/Lee et al. "2024" label is wrong by 2 years. Content match (54% of samples solvable without video) is accurate. |
| 2 | Do Video-LLMs Actually Watch? (2025) — R01 §5 | pith.science/paper/2607.11078 | MISDATED | Real arXiv ID decodes to **July 2026**, not 2025. |
| 3 | SOAR: Scene-debiasing Open-set Action Recognition (ICCV 2023) — R01 §2 | openaccess.**thecvpr**.org/... | MISATTRIBUTED DOMAIN (confirmed dead: DNS does not resolve) | Paper is real, correctly attributed (ICCV 2023, Zhai et al., adversarial scene-reconstruction + adversarial scene classifier). Correct URL: `openaccess.thecvf.com/content/ICCV2023/papers/Zhai_SOAR_Scene-debiasing_Open-set_Action_Recognition_ICCV_2023_paper.pdf` or arXiv:2309.01265. |
| 4 | CoMix / "Contrast and Mix" (NeurIPS 2021) — R01 §2 | arxiv.org/pdf/2110.15128 | VERIFIED | Sahoo/Shah/Panda/Saenko/Das, NeurIPS 2021, does temporal contrastive + background mixing for video DA. Content claim accurate. |
| 5 | Deep NNs have inbuilt Occam's razor — R01 §3 | nature.com/articles/s41467-024-54813-x | VERIFIED (minor) | Real, Oxford (Mingard/Rees/Valle-Pérez/Louis), *Nat Commun* 16, 220. Journal-issue year is **2025**, not "2024" as labelled (DOI stamp is 2024, print/issue is 2025 — common source of this exact mislabel). Content claim accurate. |
| 6 | DEVIAS (ECCV 2024) — R01 §2 | arxiv.org/pdf/2312.00826 | VERIFIED | Bae/Ahn/Kim/Choi, ECCV 2024 oral. Content match accurate. |
| 7 | "Dey et al. 2024" unsupervised domain generalization — R02 §1.1 | arxiv.org/pdf/2606.25665 | **FABRICATED CITATION** | The URL resolves to a real but **completely different** paper: "Learning Subset-Shared Invariances for Domain Generalization with Mixture-of-Experts" (Nguyen/Tran/Nguyen/Wong, June **2026**). No author named Dey. Not about "unsupervised" domain generalization. Do not cite this for the Group-DRO/worst-group point; the actual worst-group-accuracy literature to cite is Sagawa et al. 2019 (already correctly cited elsewhere) and Zhu et al. 2019 group-accuracy work (unverified this session — check before use). |
| 8 | Focal Loss for Class Imbalance (2024) — R01 §2 | arxiv.org/html/2508.02283v1 | **MISATTRIBUTED** | Real paper is "An Enhanced Focal Loss Function to Mitigate Class Imbalance in Auto Insurance Fraud Detection" (Boabang & Gyamerah, Aug 2025) — insurance fraud, not action recognition, not 2024. If citing focal loss at all, the canonical reference is Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017 (arXiv:1708.02002). |
| 9 | Learning to Adapt Frozen CLIP for Few-Shot TTA (ICLR 2025) — R01 §4 | arxiv.org/html/2506.17307v1 | VERIFIED | Confirmed ICLR 2025 acceptance, June 2025 preprint. Content match accurate. |
| 10 | SigLIP 2 (Feb 2025) — R01 §4 | arxiv.org/html/2502.14786v1 | VERIFIED | Date and content match. |
| 11 | Unlocking VLMs for Video Anomaly Detection (Oct 2024) — R01 §4 | arxiv.org/html/2510.02155v1 | MISDATED | Real paper (ASK-Hint framework), but submitted **Oct 2025**, not Oct 2024. |
| 12 | MonitorVLM safety detection for mining (Oct 2024) — R01 §4 | (no direct link given, mentioned in prose) | MISDATED | Real paper, arXiv 2510.03666, submitted **Oct 2025**, not Oct 2024. |
| 13 | Prompt-Induced Score Variance... (Jun 2025) — R01 §4 | arxiv.org/html/2605.00326 | MISDATED (content VERIFIED) | Real paper (Weng/Li/Martin, JHU), correctly describes prompt-sensitivity finding, but submitted **May 2026**, not Jun 2025. |
| 14 | Dual-State Slot Attention (2025) — R01 §3 | arxiv.org/pdf/2606.12601 | MISDATED | Real paper, "Dual-State Slot Attention: Decoupling Appearance and Identity...", submitted **June 2026**, not 2025. |
| 15 | Scene-Dependent / "Scene Memorization in Video Understanding" — R02 sources | arxiv.org/html/2603.27259 | **MISATTRIBUTED TITLE** | Real paper title is "Seeing the Scene Matters: Revealing Forgetting in Video Understanding Models with a Scene-Aware Long-Video Benchmark" (CVPR 2026 Highlight) — about long-video context forgetting, NOT about background/appearance scene-bias shortcut learning (the concept R02 actually needs). Wrong paper for the point being made. |
| 16 | Nadeau-Bengio Corrected t-test — R02 sources | semanticscholar.org/paper/Is-the-AUC-the-.../1f89894b... | **MISATTRIBUTED (confirmed by URL slug alone)** | The slug itself reads "Is the AUC the 'fairness' measure you are looking for" — an unrelated AUC-fairness paper, not the Nadeau-Bengio corrected resampled t-test paper (Nadeau & Bengio, "Inference for the Generalization Error," Machine Learning 2003 / NeurIPS 1999). Fix citation before it goes in a bibliography. |
| 17 | YOLO26 status/claims — R01 §1 | docs.ultralytics.com/guides/end2end-detection + blog | PARTIALLY MISDATED/OVERSTATED numbers | YOLO26 is real, released **Jan 14, 2026**, NMS-free dual-head design confirmed, "43% faster CPU inference (YOLO26n vs YOLO11n)" confirmed exactly as stated. **BUT** "+1.6–2.8 AP over YOLOv11" is wrong for detection: Ultralytics' own docs report **+0.6–0.8 mAP** for detection (e.g., 40.9 vs 40.1 for the n-scale, 57.5 vs 56.9 for x-scale). The +2.5/+3.7/+7.2 AP figures researcher-01 may be echoing are for **instance segmentation / pose**, not plain object detection — wrong task conflated. |
| 18 | IEEE 1012-2024 — R02 sources | ieeexplore.ieee.org/document/11134780 | VERIFIED | Real standard, approved Nov 2024, published Aug 2025. Not actually invoked substantively in R02's body text — appears only in the source list, unclear what claim it backs. |
| 19 | Smooth ECE (ICLR 2024) — R02 §1.1/§2.3 | proceedings.iclr.cc/.../06cf4bae... | VERIFIED | Błasiok & Nakkiran, ICLR 2024, "Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing." Correct. |
| 20 | Conformal selective prediction with cost-aware deferral — R02 §2.3 | nature.com/articles/s41598-026-40637-w | VERIFIED (imprecise label) | Real, *Scientific Reports* (Nature-portfolio journal, not "Nature" itself), published ~Feb 2026, about **clinical sepsis triage**, not video safety — cited only for generic conformal-deferral methodology, which is a fair (if imprecise-journal-name) use. |
| 21 | CoMix / Group-DRO / Menon 2021 / Saerens-Latinne-Decaestecker / mid-p McNemar / BCa bootstrap / DomainBed / WILDS / Sagawa 2019 | various | VERIFIED (well-known, standard literature) | These are all genuine, correctly-named standard methods/benchmarks. No fabrication found. Group-DRO arXiv:1911.08731 (Sagawa 2019) correctly cited. Menon logit-adjustment paper is arXiv:2007.07314 (preprint July 2020, ICLR 2021 venue) — citing it as "2021" is the normal (venue-year) convention, not an error. |
| 22 | Unsafe-Net (Önal & Dandıl 2024) — R02 sources, baseline paper | link.springer.com/article/10.1007/s11042-024-19276-8 | VERIFIED (minor) | Real, *Multimedia Tools and Applications*. DOI stamped 2024 (online-first), formal issue is **2025** (vol 84). "2024" is a defensible online-first citation, not wrong. |
| 23 | iSafetyBench / MonitorVLM (topic only) | arxiv 2508.00399 / 2510.03666 | VERIFIED (real, on-topic) | Both real, both about industrial safety VLM detection. Only the MonitorVLM *date* label is wrong (see #12). |
| 24 | VLM-AutoDrive (2026), InternVideo-Next (Dec 2025) — R01 §4 | arxiv 2603.18178 / 2512.01342 | NOT independently opened this session, but arXiv-ID-to-date is internally consistent with the label given (unlike #1,2,8,11,12,13,14) | Lower priority to re-check; pattern suggests these are more likely fine. |

**Pattern observed:** every misdated citation in researcher-01 is dated **exactly one (sometimes
two) years earlier** than the real arXiv timestamp. This is not random noise — it looks like a
systematic generation artifact (e.g., the agent's date sense is anchored ~1 year behind actual
arXiv ID epochs). **Do not trust any inline year in researcher-01/02 without checking the arXiv ID
digits yourself** (YYMM prefix = year+month, trivially checkable).

---

## 2. Quantitative claims with no empirical basis (asserted, not measured)

All of the following are **speculative estimates with zero experiments run in this repo** — the
researcher reports say so themselves ("Expected Effect," "speculative"), but the numbers are
precise enough (+0.5%, +2–4%, +1–3%) that they will look like measured results if copied into a
plan or thesis without a caveat:

- R01: every row of the "Rank/Hours/Expected Gain" table (§ Recommendations Ranked) — all 12 rows,
  all unmeasured guesses.
- R01 §2: "+2–4% LCO balanced accuracy; +5–15pp Safe-class recall" for background mixing — no prior
  run of CoMix on this dataset exists; this is an analogy from UCF-to-HMDB transfer, a different
  domain-gap magnitude and different task (open-set action recognition, not binary safe/unsafe).
- R01 §4: "VideoMAE-base was already tried on 4-class task; hit same ceiling ~56%" — **stated as
  fact**, but the audit (§7.e) found **no VideoMAE extraction script anywhere in the repo**; the
  53–56% figures trace only to a README table row with no reproducible artifact (audit explicitly
  flags this as the same class of problem as its own "B1: 8-class ~78% top-1, no findable source"
  finding). Treat "already tried" as unverified folklore, not an experiment record.
- R03 §5.2 example table: "P3 Jetson Orin Nano (FP16): 54% accuracy, 1.8ms, 7W, $200" and "GPU
  A4000: 54%, 2.2ms, 50W, $2000" — R03 itself labels these "Example table," i.e., illustrative
  placeholders, but the Executive Summary above it already states P3 latency "not yet
  device-verified — defer full P3 latency claim." If the placeholder table's numbers leak into a
  plan/thesis draft as if real, that's a fabrication risk purely from copy-paste, independent of
  R03's own (correct) disclaimer.
- The "65 min" k-fold-691 timing and "116 FPS"/"8.62ms"/"2.2x slower than A4000"/"7x more energy
  efficient" figures the task brief asked to check **do not appear verbatim in any of the three
  research reports** (grepped all three files). The "65 min" figure is referenced only inside the
  audit report as something the *original task prompt* asserted and which the audit's own
  measurement (95.4 min for tier-3 LCO, 5 archs) supersedes/contradicts. If "65 min" is floating
  around in planning material, it is wrong — use the audit's measured 95m23s for LCO-691, and note
  k-fold-691's own wall-clock is **not measured anywhere in the repo** (audit §7, "not found in any
  log this session").

---

## 3. Internal contradictions

### 3a. FATAL — researcher-02 §6 Q6 answer is logically inverted

Q6's answer states: *"the 22pp gap for scene (74.6% → 52.7%) vs 14pp gap for full model (84% →
70%) is damning: scene actually generalizes *better* out-of-distribution than the full model."*

This is backwards on the report's own numbers:
- A **bigger** gap (22pp > 14pp) means the scene-only model's in-domain performance collapses
  *more* under camera shift — that is **worse** generalization, not better, by the standard
  reading of "generalization gap."
- Scene-only's absolute LCO score (52.7%) is statistically indistinguishable from chance on a
  balanced binary task (50%). "Generalizing to near-random-guessing" is not generalizing well by
  any operational definition — the model isn't preserving signal, it has nothing left to preserve.
- The correct reading: scene-only relies almost entirely on spurious camera/background identity
  that k-fold's random split lets it memorize (74.6%), and that signal is **destroyed** under LCO
  (52.7%, ≈chance) precisely because scene identity does not transfer to new cameras — i.e., scene
  is *more* shortcut-prone, not more robust. The 14pp gap for the full model is smaller **only
  because the full model retains some behavior-relevant signal beyond scene** (69.5% still well
  above chance), which is the opposite conclusion from what Q6's answer states.

**A committee member doing arithmetic in their head (22 > 14) will catch this in seconds.** Do
not let this framing anywhere near the defence. Corrected answer: *"Our scene-only control
degrades MORE under domain shift (22pp) than the full model (14pp), and collapses to
near-chance (52.7%) — proving scene identity is the more brittle, shortcut-prone signal, while
the full model retains some genuine behavior signal beyond scene, which is exactly why the
smaller relative degradation (14pp) from a much higher base (83.8%) is the real evidence of
partial generalization, not evidence that scene 'generalizes better.'"*

Also flag: R02's interpretive paragraph in §2.1 says *"driven by scene memorization (62.7% →
63.7% for scene-only baseline)"* — **this contradicts §1.1's own stated numbers for the same
scene control (74.6% k-fold → 52.7% LCO)**. Two different pairs of numbers describe the same
"scene-only baseline" in the same document. This needs to be resolved against actual
`run_binary_baseline_gates.py` / S2 output before either number goes in a thesis table — as
written, the report cannot be trusted for this specific control's numbers.

### 3b. MAJOR — researcher-02 Q4's contamination-clip count is asserted more precisely than the audit supports

R02 Q4 states: *"Three test clips duplicate training unsafe clips... the one test pair with
contradictory binary labels."* The audit's direct read of `unsafenet/dedupe.py` (§5) says the
binary-task "opposite" groups reference **5 test clips** (`0_te23, 4_te4, 4_te7, 4_te9, 5_te13`),
of which only one pair (`0_te23`≡`4_te9`) is confirmed as an in-test contradiction — the
provenance of the other 3 named test clips is unresolved ("presumably cross-split conflicts,"
per the audit's own hedge). R02's confident "three contaminated clips / one contradictory pair"
framing is plausible but **not independently re-derived this session** — verify against
`dedupe.py`'s actual keep-mask output before using these exact numbers in a defence answer.

### 3c. MINOR — researcher-03's own executive summary vs later "expected effect" table

R03's executive summary correctly hedges P3 Jetson latency as "not yet device-verified... report
as measured pipeline components," but §5.2's example table states flat "54% / 1.8ms / 7W" for P3
on Jetson as if measured. Internally consistent only if the reader keeps the "Example table"
label in mind — easy to lose in a plan document. Recommend deleting the placeholder numeric table
or marking every cell "ILLUSTRATIVE, NOT MEASURED."

---

## 4. Feasibility claims vs audit ground truth

| Claim | Report | Audit ground truth | Verdict |
|---|---|---|---|
| VideoMAE "already tried," ~1hr to reuse | R01 §4 | No VideoMAE script found anywhere in repo; README-only figures, unverified | MAJOR — treat as "build from scratch," not "rerun," if pursued. Cost estimate (~3h) is fantasy. |
| "65 min" for k-fold-691 5-arch sweep | (implicit in planning material, not literally in the 3 reports) | Not reproduced by any log; only LCO-691 (tier-3) is measured, at **95m23s**, a different protocol | MAJOR if this number is used for scheduling — tier-2 (k-fold-691) wall-clock is simply unknown, budget a dry run. |
| YOLO26 "~2 hours" swap cost (R01 §1) | R01 | Ultralytics 8.4.27 does have yolo26*.yaml locally (audit §4), so no install blocker — but token-rebuild-from-raw-video cost (needed after any detector swap) is **unmeasured** in the repo (audit §7c, "no wall-clock log... recommend a dry timing run") | MAJOR — the "~2 hours" almost certainly excludes the actually-expensive step (re-extracting relation tokens for 691 clips through a new detector), which the audit explicitly flags as unknown-cost. |
| Background Mixing "~6-8 hours" (R01 §2) | R01 | No existing background-bank/inpainting code found in repo per audit; would be new code, and interacts with the same unmeasured token-rebuild cost if done at the raw-video level | MEDIUM — cost plausible only if implemented purely as a post-hoc augmentation on cached tokens/crops, not from raw video; report doesn't specify which. |
| P3 Jetson pipeline latency (R03 §1, §7) | R03 | Audit confirms P3 has **no on-screen boxes+verdict demo path** and **no fallback recording exists** (both open per JETSON-RUNBOOK-2026-09-19.md §11) — consistent with R03's own risk section | R03 is honest and consistent with audit here; no contradiction. |
| Binary head test accuracy referenced as if known (R02's frozen results table, §7.1: "72.8%", "69.5%±4.2pp" etc.) | R02 | Audit §6 states explicitly: **"Test-set accuracy of both binary heads is UNMEASURED — val_macro (89-91%) is a training-time selection metric only; 2 test touches are pre-registered... neither has been spent"** as of 2026-09-19 | **FATAL if taken at face value** — R02's entire "Table X: Full Evaluation Summary" (§7.1) with precise LCO/k-fold percentages for the *binary* task reads as if these are already-measured results to freeze for the thesis chapter. Per the audit, the actual binary-task numbers this table would need **do not exist yet** — 0 of 2 pre-registered test touches spent. Either R02 is describing the *legacy 4-class* numbers repurposed for the binary narrative (unclear which), or the table is entirely illustrative. **The planner must not treat this table as real data. Confirm with the student which protocol/task these percentages actually came from before freezing anything.** |

---

## 5. What the planner must NOT rely on

1. Any inline year attached to an arXiv citation in researcher-01 or researcher-02 without
   checking the YYMM prefix of the actual ID yourself.
2. The "Dey et al. 2024" domain-generalization citation (R02) — fabricated attribution, does not
   support the claim it's attached to.
3. The Focal-Loss (2508.02283) and "Scene Memorization" (2603.27259) citations — both point to
   real but wrong papers.
4. The Nadeau-Bengio Semantic Scholar link — points to an unrelated AUC-fairness paper.
5. The SOAR URL as literally given (dead domain) — use thecvf.com or arXiv:2309.01265 instead.
6. Any "+X% LCO gain" number in researcher-01's ranking table — none are measured; they are all
   priors from unrelated literature/domains, useful for prioritization logic only, not for setting
   success-criteria thresholds in a plan.
7. "VideoMAE already tried" as a settled fact — script doesn't exist, cost estimate is fabricated.
8. Any specific wall-clock estimate for detector retrain, token rebuild from raw video, or
   k-fold-691 (tier-2) sweep — none measured in-repo; audit explicitly says budget a dry run first.
9. researcher-02's §6 Q6 defence answer verbatim — logically inverted, will be caught by a
   committee.
10. researcher-02's §7.1 "frozen" results table as literal binary-task ground truth — audit shows
    binary-head test accuracy is unmeasured (0/2 test touches spent) as of report date.
11. The YOLO26 "+1.6-2.8 AP" claim for detection specifically — real number is +0.6-0.8 AP;
    the larger figures are for segmentation/pose tasks.
12. R03 §5.2's illustrative Jetson/A4000 comparison table if it appears anywhere outside this
    specific "Example table" context — it is not measured data.

---

## Severity-prioritized findings

| # | Severity | Finding |
|---|---|---|
| 1 | FATAL | R02 Q6 defence answer reasoning is logically inverted (22pp>14pp read as "better") — see §3a |
| 2 | FATAL | R02 §7.1 binary-task results table presented as frozen/real data; audit shows binary head test accuracy is unmeasured (0/2 touches spent) — see §4 |
| 3 | MAJOR | "Dey et al. 2024" citation in R02 is fabricated (real link ≠ claimed paper/author/topic) |
| 4 | MAJOR | R02's two internally-inconsistent numbers for the same scene-only baseline (62.7%/63.7% vs 74.6%/52.7%) — §3a |
| 5 | MAJOR | Focal-Loss and "Scene Memorization" citations in R01/R02 point to unrelated real papers |
| 6 | MAJOR | Nadeau-Bengio citation in R02 points to an unrelated AUC-fairness paper |
| 7 | MAJOR | "VideoMAE already tried" treated as fact in R01; audit finds no script, unverified README-only claim |
| 8 | MAJOR | YOLO26 detection-AP gain overstated (+1.6-2.8 claimed vs +0.6-0.8 actual for detection; larger numbers are for segmentation/pose) |
| 9 | MAJOR | SOAR citation uses a dead/wrong domain (thecvpr.org does not resolve); paper itself correctly attributed otherwise |
| 10 | MAJOR | R02 Q4's precise contamination-clip count not fully corroborated by audit's direct dedupe.py read (5 named test clips vs "three," only 1 confirmed pair) |
| 11 | MINOR | Systematic ~1-year misdating pattern across ~6 arXiv citations in R01 (Video-Oasis, prompt-variance paper, Dual-State Slot Attention, "Do Video-LLMs Actually Watch?", Unlocking-VLMs-anomaly, MonitorVLM) — content mostly fine, dates wrong |
| 12 | MINOR | Occam's razor Nature paper labeled "2024" vs actual journal-issue year 2025 (DOI-stamp vs issue-year confusion, defensible) |
| 13 | MINOR | Unsafe-Net baseline paper labeled "2024" vs formal issue year 2025 (same DOI-stamp reason, defensible) |
| 14 | MINOR | "65 min" k-fold-691 timing figure (in planning material, not the 3 reports themselves) contradicted by audit's own measurement regime (95.4min measured for the *different* LCO-691 protocol; k-fold-691 wall-clock is simply unmeasured) |
| 15 | MINOR | All of R01's "Expected Effect: +X%" estimates are unsourced priors, not measurements — fine as prioritization heuristics, dangerous if quoted as expected results in a plan/thesis |

---

## Unresolved questions

1. Which task/protocol do R02 §7.1's frozen-table percentages (72.8%, 69.5%±4.2pp, etc.) actually
   come from — legacy 4-class collapsed to binary post-hoc, or genuinely-run binary numbers not yet
   reflected in the audit? Needs the student/lead to clarify before the table is used anywhere.
2. Is the audit's dedupe.py reading (5 test clips in binary "opposite" groups) reconcilable with
   R02's "three contaminated clips, one contradictory pair" framing — same underlying file, two
   different summaries?
3. "Zhu et al. 2019" (worst-group accuracy, cited alongside the fabricated "Dey et al. 2024" in
   R02 §1.1) was not independently checked this session — verify before citing.
4. R01's InternVideo-Next (2512.01342) and VLM-AutoDrive (2603.18178) citations were not opened
   this session (date-consistent, lower suspicion, but content not verified).
5. Where did the "65 min," "116 FPS," "8.62ms," "2.2x slower than A4000," "7x more energy
   efficient" figures the task brief flagged actually originate, if not these three reports? Worth
   asking the student/lead directly — likely a separate planning doc or brief.md not in this
   review's scope.
