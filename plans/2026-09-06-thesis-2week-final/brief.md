# Thesis 2-Week Final Push — Master Brief (binding)

**Window:** 2026-09-06 → 2026-09-19 (14 days). **Owner decisions locked below — do not re-litigate.**

Đề tài: *Phát hiện hành vi không an toàn thời gian thực trong giám sát công nghiệp dựa trên YOLO và
Transformer* (Unsafe-Net dataset, YOLO11/YOLO26 + Transformer).

---

## 0. Locked decisions (owner, 2026-09-06)

| # | Decision | Consequence |
|---|---|---|
| D1 | **Thesis language = Vietnamese**, English abstract | All `report/*.md` (currently EN) rewritten to VN per S01 Mẫu 2. EN abstract + keywords. |
| D2 | **Jetson Orin Nano: full physical access** | Headline pipeline re-exported to TensorRT and re-benchmarked. No literature-citation rows in the deployment chapter. |
| D3 | **We do NOT claim to beat the paper's headline metric.** The thesis proves effectiveness on **multiple other axes**. | Framing = "statistical tie on the paper's own noisy metric + decisive, measured wins on the axes the paper never measured". See §2. |
| D4 | **Deliverables = all four**: full VN thesis, defence slides, reproducible code+README, live demo | §6 |
| D5 | **Evaluation contract = owner's spec** (§4) | Binding. Every number in the thesis comes from this suite. |

---

## 1. State of evidence entering this plan (verified 2026-09-06)

**Three pipelines exist. They are not interchangeable.**

| Pipeline | Code | Best measured | Protocol quality |
|---|---|---|---|
| **P1** frozen-COCO YOLO11s → multi-stream Transformer (round 0/1) | root `run_*.py` | 56/64 = 87.5 % top-1, 89.28 % macro (test-selected); 53/64 blind | 64 clips, ±11 pp CI. **This is what is on Jetson today.** |
| **P2** paper-faithful rewrite (round 2) | `unsafenet/` | 50/64 = 78.12 % blind; 55/64 test-selected | Frozen val protocol, 1 test touch. Honest but weak. |
| **P3** YOLO26 entity + spatial-relation Transformer | `run_kfold_relation.py`, `run_lco_eval.py` | **k-fold 516: 80.8 ± 3.1 % top-1, 84.0 % macro · LCO 516: 78-82 % macro (range across grouping/token variants; single value NOT yet locked) · same-dist 64: 54/64 = 84.4 %, macro 88.4 %** | **Only N > 64 evidence in the project. ±3 pp CI.** |

**Paper (Önal & Dandıl 2025), verified from `paper/paper.txt`:**
*(Citation year settled 2026-09-06: use **2025**, matching the volume/pages in the full reference — *Multimedia Tools and Applications* **84**, 34967-34993. `paper/paper.txt` is body text only and carries no year or volume, so it cannot arbitrate; earlier "2024" mentions in this brief and `PROPOSAL.txt` most likely referred to an online-first date. **Verify against the publisher page before the thesis bibliography is finalised** - a wrong year in the single most-cited reference is an avoidable error.)*
- Full Unsafe-Net (YOLOv4 + ConvLSTM + frame reduction): **62/64 = 96.88 % top-1, 95.81 % macro** (Tables 6–7).
- No-reduction baseline (ConvLSTM only): **57/64 = 89.06 % top-1, 85.78 % macro** (Table 5).
- Their YOLOv4: trained on **2,262 labelled train images** + 329 test (Table 3), **nc = 4**
  (unsafe classes only, used as a frame gate), **416x416**, batch 65, 2 GPUs. The only quantified
  detector figure in the paper is *"the average mAP value during the iteration period is around 97 %"*
  (L323) - a read off the **training curve** in Fig. 8, not a test-set table.
  **RETRACTED: the "95.7 % mAP" figure appears nowhere in `paper/paper.txt` (grep-verified, zero hits).**
  It was invented downstream and propagated into `PROGRESS.md` L64 and earlier drafts of this brief.
  **No detector-mAP comparison against the paper may appear in the thesis** - there is no comparable
  published number, and the tasks differ (their nc=4 gate vs our nc=8 behaviour detector).
- Action-recognition time **0.14 s/clip**; average clip duration reduced to **1.87 s** by frame reduction.

**Decisive audit finding — the detector was never given a fair shot:**

| Fact | Source |
|---|---|
| Our bbox annotation is **already at paper scale**: 2,262 frames / 2,245 labels, 1,901 train / 343 val, 8 behaviour classes | `annotation_guide/dataset_behaviour_tight/`, `frame_manifest.csv` |
| Our behaviour detector reached only **mAP50 0.791 (tight) / 0.806 (ctx)** — reported standalone, **not** against the paper (see retraction above) | `plans/2026-09-05-*/reports/t1-objectives-audit.md` §1c |
| …because it was trained **on CPU, yolo11s, 25 epochs (early-stopped at ep15), batch 8** — a budget forced by the Ultralytics-on-MPS training bug. Note the paper also used imgsz 416, so **resolution is not the differentiator; convergence budget is** (their batch 65 × 2 GPUs × full iteration schedule) | `train_behaviour_detector.py` defaults; ~8–10 min/epoch on CPU |
| **CORRECTED:** the "emits only 5 of 8 classes" defect belongs to the *entity* detector in `kaggle-detector-retrain/`, where Safe_Walkway / Restricted_Zone have **zero training labels by design** (zones were never annotated) — structural, not a training failure. The **behaviour** detector (`dataset_behaviour_{tight,ctx}`, the A1 target) has **no zero-label class**: verified counts 170–1,634 per class | `plans/2026-09-06-*/research/r1-detector-gpu-recipe.md` §1 |
| Free GPU path packaged and unused | `kaggle-detector-retrain/`, `colab/kaggle_yolo26_onecell.py` |

→ Every prior *"the paper's mechanism does not help"* conclusion was drawn with a 0.79-mAP detector
that never converged (early-stopped at epoch 15 on CPU). **That conclusion is not yet earned.** Fixing it is hours of free GPU time and it
is what makes the reproduction claim (§2 axis 2) credible — not a bid to beat 62/64.

**Why beating 62/64 is off the table (state this in the thesis, do not hide it):** on 64 clips the
paper errs only twice, so the maximum possible discordant advantage is 2 clips. Even a flawless 64/64
gives McNemar best-case **p = 0.50** (Fisher **p = 0.496**). No accuracy whatsoever can produce a
significant win at N = 64. This is a *ceiling* property of the benchmark, not a shortfall of our
method — and it is itself a publishable methodological observation.

---

## 2. The claim structure — what the thesis actually argues

> **Thesis claim.** On the paper's own 64-clip metric our system is **statistically indistinguishable**
> from the published result — and *no system can be distinguished there*, by construction. On every
> other axis that determines whether such a system is deployable — statistical reliability,
> cross-camera generalisation, safety-oriented error structure, calibration, and measured real-time
> edge performance — this work provides evidence the original paper never produced.

| # | Axis | Our evidence | Paper |
|---|---|---|---|
| **1** | **Accuracy, paper protocol** | same-dist 64: 54/64 = 84.4 % top-1, **macro 88.4 %** (blind) → **exceeds the no-bbox baseline macro 85.78 %**; vs 57/64 top-1: Δ = −4.7 pp, z = −0.78, **p = 0.435 → statistical tie** | 62/64 / 95.81 % (full), 57/64 / 85.78 % (no-bbox) |
| **2** | **Faithful reproduction of the paper's mechanism** | Behaviour-as-object detector at paper scale (2,262 imgs) + frame-reduction gate; replicate their **0.14 s/clip** and **1.87 s** reduced-duration claims; report where they hold and where they do not | Claimed, never independently reproduced |
| **3** | **Statistical reliability** | k-fold 516 clips: macro 84.0 %, **CI ±3 pp** — ~3.5× tighter than any 64-clip point estimate; bootstrap 5,000 + Wilson/BCa everywhere. **The Nadeau–Bengio significance verdict is NOT yet stateable — see S3.** | Single 64-clip point, ±11 pp, no CI |
| **4** | **Cross-camera generalisation (the real deployment question)** | **Leave-camera-out**: **78-82 % macro** on cameras never seen in training (`sweep_results/relation_tokens/_zones_eval.log` gives 79.7 % and 82.0 % under repaired grouping; the previously-cited 78.6 % appears in no current log and is withdrawn as a point value), where camera-identity shortcuts are unavailable by construction. **The "+60.3 pp vs scene-only" framing is WITHDRAWN — see S2 below.** Rebuild the comparison against a transferable baseline before claiming any margin | Never run. Their same-dist protocol cannot distinguish behaviour from scene memorisation |
| **5** | **Safety-oriented evaluation** | Per-class **FNR/miss-rate**, cost-sensitive scoring (miss ≫ false alarm), threshold sweep, operating point chosen by safety recall | Accuracy only |
| **6** | **Calibration** | **ECE + reliability diagram** — required before any alarm threshold can be trusted | Not reported |
| **7** | **Real-time edge deployment** | Jetson Orin Nano, TensorRT FP16/INT8: latency p50/p95/p99 per frame and per clip, FPS, **multi-stream aggregate + zero-drop**, 30-min soak, **J/inference and FPS/W**, params/FLOPs/INT8 compression | No deployment metrics of any kind |
| **8** | **Methodological rigor / negative results** | Blind protocol; withdrawn "57/64 calibrated" and "+3.5 pp macro win"; val↔test independence (r ≈ −0.07…−0.25, n.s.); the Ultralytics-MPS training bug (0.0002 vs 0.940 control); box-context negative result; the N=64 ceiling proof | — |

**S2 — WITHDRAWN CLAIM: the "scene-only collapses to 16.9 %" comparison (found 2026-09-06, code-verified).**

The scene-only baseline is **not the same estimator across the two protocols**:

- `run_kfold_relation.py:52` — `vmaj.get(views[i], gmaj)`: **per-view majority class**. Informative;
  77.1 % is a real number.
- `run_lco_eval.py:66` — `Counter(yn[tr]).most_common(1)[0][0]`: the **training global-majority class**,
  i.e. **one constant prediction for every held-out clip**. A constant predictor over 4 classes has
  macro accuracy **25 % by arithmetic** (100 % recall on one class, 0 % on three); the observed 16.9 %
  is that artifact plus class-support imbalance.

So the reported "collapse from 77.1 % to 16.9 %" measures **the baseline changing definition**, not
scene shortcuts ceasing to work. **+60.3 pp is a comparison against a constant, and must not appear in
the thesis.** Anyone on the committee who opens `run_lco_eval.py` sees this in one screen.

*What genuinely survives:* under LCO the held-out camera is unseen, so camera-identity shortcuts are
structurally unavailable — and the model still reaches **78-82 % macro** vs the 25 % constant-predictor
floor. That is a real cross-camera generalisation result and it needs no inflation.

*Required fix (Lane A, before Ch3 is written):* rebuild the LCO baseline as something that can actually
transfer — global class prior, and a scene/background-appearance classifier trained on the training
cameras and applied to the held-out camera — then report the margin against **that**. Also note
(`run_lco_eval.py:66` comment aside) that appearance features are already **crop-based (30 %-context
boxes)**, not full-frame, so "scene memorisation" was always a weaker hypothesis than the earlier
framing assumed. The confound that is real is **camera-identity → label**, not raw-pixel background.

Also update `run_kfold_relation.py:83`, which prints the withdrawn 16.9 % contrast as guidance.

**S2 — CONFIRMED AND PARTLY RESOLVED (2026-09-06, later same day).**

*Independent confirmation from the log, not just the code.* `sweep_results/relation_tokens/lco_eval.log`
records per-fold scene-only accuracies of **5.9 / 0.0 / 0.0 / 18.6 / 0.0 / 57.3 / 17.2 / 0.0 %**. Three
folds at *exactly* 0.0 % is the signature of a constant predictor emitting a class the held-out camera
does not contain. No informative baseline produces exact zeros. S2 is settled beyond argument.

*First evidence from the rebuilt baseline.* The canonical 5-seed run (`lco_eval_canonical_a3b.log`),
fold 0: **transformer 60.4 % · scene-appearance 58.3 % · prior 20.8 %.** Against a *transferable* scene
baseline the margin on this fold is **+2.1 pp**, not +58.7. Expect the honest LCO scene margin to be
small. Plan the thesis narrative for that now rather than being surprised by it.

*The claim that survives and should lead instead.* Ours **75.6 %** vs a **trained whole-frame
(paper-style) classifier 52.5 %** = **+23.1 pp**, with the per-class pattern that carries the argument:
WalkwayViol 78 vs 28, UnauthorInt 35 vs 13 — the two spatially-defined classes, which is exactly what
entity geometry is for. This is a real trained transferable baseline seeing the same pixels; it was
already in the data and was merely standing behind the invalid comparison.

*Text repaired this round:* `report/abstract.md` (both the abstract proper and the "Honest headline"
paragraph — it carried the withdrawn 16.9 % claim, the most-read page in the thesis),
`report/chapter-07-…md` §7.7 table + implications. Both now also carry the 3-seed provisional banner.
Overreach removed in the same pass: the thesis no longer asserts the paper's *model* would fail under
LCO — unrunnable without their weights. It asserts the *metric* is scene-inflated, which is measured.

**S2 — CLOSED (2026-09-06). Canonical 5-seed LCO run complete; axis 4 now has an honest number.**

Log: `sweep_results/relation_tokens/canonical/lco_seeds5_yolo26L1_CANONICAL.log` (5 seeds, 60 epochs,
8 camera-groups, 516 pooled clips). Pooled arithmetic independently re-derived from the per-fold lines.

| method | LCO top-1 | LCO macro |
|---|---|---|
| **entity-relation (ours)** | **77.7 %** (401/516) | **81.2 %** |
| scene-appearance (logreg on whole-frame MobileNetV3 features, fit on training cameras only) | 53.1 % | 64.4 % |
| whole-frame trained classifier (earlier run, independent construction) | 52.5 % | 60.3 % |
| *prior / constant floor — NOT a competitor* | *16.5 %* | *10.1 %* |

**HONEST MARGIN: +16.9 pp macro, 95 % bootstrap CI [+12.5, +21.2] pp** (n_boot = 5 000). This replaces
the withdrawn +58.7/+60.3 pp everywhere. It is smaller, it is real, and the CI excludes zero
comfortably — it does not need inflating.

*The withdrawal is now proven, not merely argued.* The rebuilt `prior` baseline scores **16.5 %**, which
reproduces the old "scene-only 16.9 %" to within rounding. The old baseline **was** the constant. No
further argument is needed if this is challenged at the defence — show this line.

*Two independent transferable baselines agree to within 0.6 pp* (53.1 % vs 52.5 %, built at different
times by different means). That agreement is what makes the margin credible; cite both.

*The per-class pattern is the real argument, and it is now much stronger than the aggregate.*

| class | ours | scene-appearance | reading |
|---|---|---|---|
| OpenedPanel | 88.7 | 88.0 | tie — object is visible, scene model handles it |
| CarryOverload | 100 | 100 | tie — same |
| **WalkwayViol** | **70.5** | **17.6** | +52.9 pp — needs *where the person is* |
| **UnauthorInt** | **65.7** | **51.9** | +13.8 pp — same |

The two classes that tie are exactly the ones a scene model *should* get; the two that separate are
exactly the ones entity geometry exists for. **Lead Ch3/Ch4 with this table, not with the aggregate** —
it is a mechanism claim, and the mechanism is visible in the numbers.

*Also worth stating plainly:* per-class recall moved a great deal between the 3-seed and 5-seed runs
(UnauthorInt 35.2 -> 65.7). That is a strong caution against reading any single-seed per-class recall in
this project as stable, and retroactive justification for the >=5-seed floor (4h.5).

*Standing caveat, mandatory wherever LCO is cited:* the 8 "cameras" are a **KMeans(8) proxy** on
static-background descriptors (`tools/build_scene_grouping.py`); Unsafe-Net publishes no camera IDs.
Every LCO figure is bounded by how well that proxy tracks real cameras.

*Propagated to:* `report/abstract.md`, `report/chapter-07-…md` §7.7 + §7.8, `docs/system-architecture.md`.
Both 3-seed provisional banners removed - the numbers they guarded are now canonical.

**S3 — the k-fold Nadeau–Bengio p-value is genuinely unstable (found 2026-09-06, diagnosed).**

The p-value swings 0.0375 <-> 0.10 across runs. This is **not** a log-versioning artifact: three
different token tags (`yoloapp_`, `zones_l1_`, `yolo26L1_`) were each run **once**, at **df = 4** —
underpowered by `run_kfold_relation.py`'s own docstring. The scene-only baseline is identical across
all three, so fold assignment is not the cause. **Neither "significant" nor "n.s." may be written in
the thesis** until this is resolved.

Proper fix: `run_kfold_relation.py --tag yolo26app_ --repeats 10` (~22 h CPU, **not yet run**).
Schedule it or state the limitation explicitly — those are the only two honest options.

*Not affected:* the scorecard's "YOLO11 k-fold 419/516, pipeline not detector-limited" conclusion
traces to `kfold_relation.log`, a different artifact untouched by the D4 duplicate-file defect. That
conclusion survives.

**S3 — RESOLVED (2026-09-06). Canonical run complete; the honest answer is p = 0.0574, n.s.**

`run_kfold_relation.py --k 5 --repeats 10 --seeds 1 --epochs 60 --tag yoloapp_` (**YOLO11** track — the
tag whose single-run p = 0.0375 was the basis of the "significant" claim). Log archived in-repo at
`sweep_results/relation_tokens/canonical/kfold_repeats10_yoloapp_YOLO11.log`. **50 folds, df = 49:**

| quantity | canonical value |
|---|---|
| relation-Transformer top-1 | **80.7 ± 3.6 %** (pooled 421/516) |
| macro-recall | 85.1 % · macro-F1 83.9 % · macro-precision 83.8 % |
| Cohen κ / MCC | 0.744 / 0.749 |
| NB-corrected SE | **1.87** (naive SE 0.51 — a **3.7x** understatement if uncorrected) |
| 95 % CI (NB) | **77.0 - 84.4 %** |
| scene-only baseline | 77.1 ± 2.5 % |
| **NB relation vs scene-only** | **mean Δ = 3.6 pp, t = 1.95, df = 49, p = 0.0574 — NOT SIGNIFICANT** |

**Both prior values were wrong.** 0.0375 ("significant") and 0.10 were each single runs at df = 4. The
canonical answer sits just above α = 0.05. Write it as **p = 0.0574 (n.s.)** and never as "marginally
significant" or "trending" - those phrases are p-value laundering and a committee will name them.

*This result strengthens the thesis rather than weakening it.* Under **random** k-fold every camera
appears in training, so scene shortcuts are fully available and camera-guessing already scores 77.1 %.
That the behaviour model cannot significantly beat it **is the scene-confound**, measured directly. It
is the same finding as S2 seen from the other protocol, and it is the argument for LCO.

*Note the naive-vs-NB SE gap (0.51 -> 1.87).* Any k-fold CI in this project computed without the
Nadeau-Bengio correction is roughly **3.7x too narrow**. Check every reported k-fold CI against this.

*Log caveat.* That archived log's final NOTE still prints the withdrawn "16.9 % collapse" contrast: the
process started before `run_kfold_relation.py:99` was repaired, so Python held the pre-edit source. The
**file** is fixed; that one printed line in the log is stale. Do not cite it.

**S3 — FULLY CLOSED (2026-09-06). Both tracks canonical; both n.s.; they agree.**

`--tag yolo26app_` (the **P3 headline**), 50 folds, df = 49. Log:
`sweep_results/relation_tokens/canonical/kfold_repeats10_yolo26app_HEADLINE.log`.

| | **YOLO26 (headline)** | YOLO11 (ablation) |
|---|---|---|
| top-1 | **80.8 ± 4.0 %** (421/516) | 80.7 ± 3.6 % (421/516) |
| macro-recall | 84.2 % | 85.1 % |
| macro-F1 / κ / MCC | 83.5 % / 0.741 / 0.744 | 83.9 % / 0.744 / 0.749 |
| naive SE -> NB SE | 0.58 -> **2.12** (3.7x) | 0.51 -> **1.87** (3.7x) |
| NB 95 % CI | **76.7 - 85.0 %** | 77.0 - 84.4 % |
| scene-only baseline | 77.1 ± 2.5 % | 77.1 ± 2.5 % |
| **NB test** | **Δ = 3.7 pp, t = 1.93, p = 0.0600 (n.s.)** | Δ = 3.6 pp, t = 1.95, p = 0.0574 (n.s.) |

**Three things this settles, all worth stating in the thesis:**

1. **The p-value is n.s. on both tracks, and the two agree to 0.003.** Every single-run value was wrong:
   YOLO11 0.0375 ("significant") and YOLO26 0.134 - errors in *opposite* directions, which is exactly
   what underpowered df = 4 estimates do. Quote **p = 0.0600** for the headline. Never "marginally
   significant", never "trending".
2. **"Not detector-limited" is now properly evidenced.** 80.8 % vs 80.7 % top-1, overlapping CIs, on
   50 folds each. The earlier version of this claim rested on single runs; it happened to be right.
3. **The NB correction inflates SE by 3.7x on both tracks independently.** This is a stable property of
   this design, not a one-off. **Any k-fold CI in this repo computed without it is ~3.7x too narrow.**

*The headline result, stated honestly:* under random k-fold the model reaches 80.8 % top-1 / 84.2 %
macro but is **not** significantly better than predicting from camera identity (p = 0.0600). Under LCO,
where camera identity is structurally unavailable, it leads a transferable scene baseline by
**+16.9 pp macro, CI [+12.5, +21.2]**. That pair of results is the thesis's central evidence-integrity
finding: *the confounded protocol cannot separate the systems; the honest protocol can.*

**S4 — the ECE number, corrected.** P3 (`logits_yolo26.pt`) ECE = **0.1009**, computed on the
**ensemble** basis, i.e. `argmax(ens_softmax)` = **52/64** — *not* the 54/64 headline. There is no
hidden calibration step: `ens_softmax` (seed-averaged) and `blind_pred` (best-val single seed) are
simply **two different decision rules**, and the blind rule's own softmax was **never persisted**.
Consequence: every conformal / selective-risk / calibration figure currently computed characterises
the **52/64 ensemble rule**. Either re-persist the blind rule's softmax and recompute, or carry an
explicit basis label on every such figure. **Do not pair 54/64 with ECE 0.1009 in one row** — that is
the exact mistake the old scorecard made across pipelines, repeated within one pipeline.

> **S4 CLOSED (2026-09-06) — the first option was taken: the blind softmax was recovered, not labelled
> around.** The unlogged generating command was re-derived by pre-registered reproduction (run
> `bacckkzbm`, four candidate configs, decision rule committed as `cb4254c` *before* launching):
> `run_relation_transformer.py --tag yolo26app_ --seeds 5 --aux_w 0.5 --epochs 120`, blind seed 1.
> It reproduces the stored artifact **bit-for-bit** — `blind_pred` element-wise identical and
> `ens_softmax` max abs difference exactly **0.00e+00**. `blind_softmax`, `blind_seed` and
> `per_seed_softmax` were added to `logits_yolo26.pt` (pre-existing tensors asserted unchanged before
> and after the write; backup in scratchpad). `run_full_eval.py` now prefers `blind_softmax` when
> present and records `pred_basis` + `source_provenance` in its JSON.
>
> Blind-basis values, replacing the ensemble ones everywhere: **ROC-AUC 0.951** (was 0.950),
> **PR-AUC 0.880** (was 0.869), **ECE(15) 0.097** (was 0.101), McNemar vs scene-only **p = 0.804**
> (was 0.815). §7.5 and the abstract now describe one predictor. Note the direction: every corrected
> figure moved *slightly in the thesis's favour*, which is precisely why the reproduction was
> pre-registered — the outcome was fixed before the numbers were visible.
>
> **Two caveats kept on the record.** (i) The pre-registered rule said "exactly one candidate reproduces
> `blind_pred`"; **two** did (both select seed 1, so `blind_pred` cannot discriminate seed count). On its
> letter that is branch 3 — record all, claim none. The tie was broken by `ens_softmax`, a second tensor
> in the *same frozen artifact* whose comparison was computable before the run; the rule was
> under-specified, not the result over-fitted. (ii) Side effect worth having: P3 is demonstrably
> bit-reproducible on this stack, and `w2-number-integrity-fixes.md` unresolved #1 (provenance known
> only from file mtimes) is closed.

**S5 — no test-touch log ever existed.** Section 4h.3 requires `reports/test-touch-log.md`; it exists
nowhere in the repo, so **no artifact in this project has a verifiable pre-registration**, including
`yoloroi` (blind 55/64, higher than the headline). `yoloroi` was traced to an Aug-16 YOLO11-track
exploratory run superseded by the Aug-29 YOLO26 switch — plausibly not a cherry-pick, but
**unverifiable**. The log is created as of 2026-09-06; every entry before that date is a
**retroactive reconstruction and must be labelled as such**. Pre-registration discipline is only
claimable from this date forward, and the thesis must say exactly that.

**S6 — Ch4's k-fold confidence intervals are pseudo-replicated (~1.7x too narrow) (found 2026-09-06,
code-verified, fix landed).** Table 4.8.1 publishes `75.3 +/- 1.2 %, 95 % CI [73.1, 77.5]`. That interval
is **not** Nadeau-Bengio corrected -- an earlier note in this session claimed it was; that was wrong, and
the agreement with an NB hand-calculation (+/-2.23 pp) was a numerical coincidence. `run_kfold_eval.py`
computed it as `bootstrap_ci(np.concatenate(cat_y), np.concatenate(cat_tf))`, i.e. a percentile bootstrap
over **1548 rows = 516 clips x 3 seeds**. The same 516 clips appear three times, so the resampling unit is
the (clip, seed) row rather than the clip. Reproduced to the decimal: a bootstrap of 1548 pseudo-replicated
rows at p = 0.753 returns **[73.1, 77.5]**; the clip-level bootstrap on 516 returns **[71.7, 79.1]**.

The signature is diagnostic -- the old estimator returns +/-2.1 pp *regardless* of how much the seeds agree,
which is only correct when seeds are statistically independent:

| seed agreement | old (row bootstrap) | new (clip block bootstrap) |
|---|---|---|
| identical predictions | +/-2.10 pp | **+/-3.80 pp** |
| rho = 0.9 (realistic) | +/-2.20 pp | **+/-3.60 pp** |
| fully independent | +/-2.10 pp | +/-2.15 pp (converges, as it must) |

Real seeds share features and architecture, so they sit near the top row: the published interval is
roughly **1.7x too narrow**. FIX LANDED: `bootstrap_ci()` in `run_kfold_eval.py` now takes
`(n_seeds, n_clips)` arrays and resamples the **clip** axis, carrying all of a clip's seed predictions
with it (block/cluster bootstrap). Call site passes `np.stack(...)` instead of `np.concatenate(...)`.
Re-run launched 2026-09-06; **Table 4.8.1's CIs are provisional until it lands and must not be quoted
until replaced.** The point estimates (75.3 / 77.8 %) are unaffected -- only the intervals move.

**S8 -- Ch7 §7.5 and the abstract printed THREE metrics from the WRONG PIPELINE (found 2026-09-06,
numerically proven, fixed).** This is a strictly worse version of S4: not a two-basis mix inside P3, but
**cross-pipeline contamination in the headline chapter's headline table.**

Recomputed every §7.5 cell from `logits_yolo26.pt` and compared against every other `logits_*.pt`:

| §7.5 printed | matches `logits_yolo26.pt`? | actually reproduces | verdict |
|---|---|---|---|
| top-1 84.4, macro-recall 88.4, macro-F1 85.6, bal-acc 88.4, kappa 0.775, MCC 0.785 | **yes**, on `blind_pred` | -- | correct |
| per-class P/R/FNR line (96.2/78.1/21.9 ...) | **yes**, on `blind_pred` (exact) | -- | correct |
| ROC-AUC **0.953** | no -- yolo26 gives **0.9496** | `logits_s1a.pt` = 0.9533 | **wrong pipeline** |
| PR-AUC **0.874** | no -- yolo26 gives **0.8691** | `logits_s1a.pt` = 0.8739 | **wrong pipeline** |
| ECE 15-bin **0.104** | no -- yolo26 gives **0.1009** | `logits_s1a.pt` = 0.1041 | **wrong pipeline** |

`logits_s1a.pt` is byte-identical to `logits_yolo11_baseline.pt` (test-touch log, section A) -- i.e. the
**YOLO11 track**, not P3. All three wrong values reproduce it to four decimals; chance agreement is nil.
This is the same error the 2026-09-02 scorecard was corrected for (its "ECE 0.101" came from a P1
artifact) -- the error class survived the scorecard fix and re-appeared in Ch7 **and the abstract**.

**Corrected and applied** to `report/chapter-07-...md` §7.5 and `report/abstract.md`:
ROC-AUC **0.950**, PR-AUC **0.869**, ECE **0.101** -- all three on the **ensemble (52/64)** basis, which
is forced: the blind seed's softmax was never persisted. §7.5 now carries a per-metric `basis` column and
both files carry an explicit basis note. Also fixed: Ch7:203 stale "UnauthorInt LCO 35 %" -> 65.7 %.

**Rule this establishes:** a metric's *value* is not evidence of its *provenance*. Every probability-based
figure must be re-derived from the named artifact before it is printed, because a plausible number from
the wrong file is indistinguishable by eye from the right one.

**S7 -- the origin paper's citation was wrong in three fields; year 2025 was MY error and is reversed
(verified 2026-09-06 against Crossref).** The publisher record for DOI `10.1007/s11042-024-19276-8`
(queried via `api.crossref.org` + DOI content negotiation, APA rendering) is:

> Onal, O., & Dandil, E. (2024). Unsafe-Net: YOLO v4 and ConvLSTM based computer vision system for
> real-time detection of unsafe behaviours in workplace. *Multimedia Tools and Applications*, 84(29),
> 34967-34993.

Corrections applied to `README.md`, `report/chapter-02-related-work.md` [17], `report/abstract.md`,
`report/chapter-04-experiments-and-results.md`:
1. **First author initial: "K. B. Onal" -> "O. Onal"** (registered name Oguzhan Onal). Wrong in every
   copy of the bibliography; nobody had checked it.
2. **Year: 2025 -> 2024.** Earlier this session I normalised the repo *to* 2025 by reasoning from
   volume 84 being a 2025 issue. That reasoning was wrong: Crossref records `issued` and
   `published-online` as 2024-05-04 and renders the citation as "(2024)". The 2025 issue date is real
   but is not the registered publication year. **The normalisation is reversed.**
3. **Issue number 84(29) added** (previously omitted).

The title in the DOI-bearing Ch2 entry was already correct and is confirmed verbatim. The pending item
"verify paper title and year against the publisher record" is **CLOSED**.

**S6a -- the defect is CONTAINED to `run_kfold_eval.py` / Ch4 §4.8 (audited 2026-09-06).** Grepped every
`.md` in the repo for the four affected intervals: they appear **only** in Ch4 Table 4.8.1. The headline
k-fold track is a different script -- `run_kfold_relation.py` builds its interval from an
**NB-corrected across-fold SE** (`eval_utils.paired_tests.nadeau_bengio`, `mean +/- 1.96 * corr_se`), not
from a clip bootstrap, so `80.7 +/- 3.6 %, CI 77.0-84.4 %` and the scorecard's "+/-3 pp (516 clips)" row are
**not affected**. Nothing in the abstract, Ch7, `docs/system-architecture.md`, or the fair-head-to-head
scorecard needs revision for S6.

**S6b -- what is NOT wrong in Ch4 (checked, so nobody re-opens it).** Table 4.8.2's head-vs-probe test
(`Delta -4.65 pp, NB t = -3.15, p = 0.007`) is **sound**. It is not the S3 df = 4 defect: `run_scheme()`
accumulates `diffs` across seeds *and* folds, so n = 3 x 5 = 15 and df = 14 -- and t = 3.15 at df = 14
gives p = 0.0071, reproducing the published value exactly (at df = 4 it would read 0.0345, at df = 49
0.0028). The NB correction is applied with the correct rho = n_test/n_train = 0.25. **This finding runs
against the thesis's interest and it survives audit; report it as measured.** Its only caveat is power
(n = 15 vs the canonical 50 used for the S3 tracks), which is a consistency note, not a validity error.

**Work programme serving these axes:**

| ID | Task | Serves | Risk |
|---|---|---|---|
| **A1** | **GPU detector retrain** — yolo11s/m + yolo26s, imgsz 640–800, 150–300 epochs, full aug, `tight` + `ctx` variants, Kaggle/Colab GPU. Target **mAP50 ≥ 0.90**, all 8 classes emitted | axes 2, 1, 4 | Low — free GPU, packaged |
| **A2** | Behaviour-as-object clip decisions + frame-reduction gate with the good detector; measure **0.14 s/clip** and **1.87 s** | axis 2 | Low |
| **A3** | Rebuild P3 relation-Transformer on improved tokens; re-run **k-fold 516 + LCO 516 + same-dist 64**; add motion/dwell stream for the weak LCO class (UnauthorInt 44.4 %) | axes 3, 4 | Medium |
| **A3b** | **Rebuild the LCO baseline (S2 fix)** — replace the constant global-majority predictor with a transferable scene/appearance classifier + class-prior floor; re-measure the axis-4 margin honestly | axis 4 | Low, but **blocks Ch3** |
| **A4** | **Headline freeze**: one pipeline, one declared test touch, full §4 suite across all three protocols | all | — |
| **A5** | Safety + calibration analysis: FNR, cost matrix, threshold sweep, ECE, reliability diagram | axes 5, 6 | Low |

**Excluded by prior proof — do not spend a day on these:** further config/seed/ensemble search on the
64 clips (val↔test r ≈ −0.07…−0.25, never significant — val carries no exploitable signal); re-running
fusion to chase 55 → 56/64; the box-context ablation (settled, negative).

---

## 3. Objectives → evidence map (PROPOSAL.txt)

| Proposal objective | Delivered by | Status |
|---|---|---|
| **MT1** — YOLO–Transformer architecture: CNN speed + Transformer temporal reasoning | P3 (YOLO26/11 entity detector → spatial-relation Transformer), Ch3 | Built; A1/A3 refresh + VN write-up |
| **MT2** — Train / fine-tune / evaluate on Unsafe-Net (4 safe / 4 unsafe) | A1–A5, §4 suite, Ch4 | Core of week 1 |
| **MT3** — TensorRT + INT8, multi-stream real-time on edge, low latency, high accuracy | B1–B3 on Jetson, Ch5 | Re-run required (D2) |

---

## 4. Evaluation contract (BINDING — owner spec, 2026-09-06)

### 4a. Paper-faithful anchors
- **Macro accuracy, 4-class** = mean per-class recall. Paper mark **95.81 % (62/64)**.
- **No-bbox baseline 89.06 % (57/64) / 85.78 % macro** — the reference line to exceed where possible.
- **Action-recognition time per clip** — paper mark **0.14 s**; measure with `time.perf_counter()` per clip.
- **Reduced average video duration** — paper mark **1.87 s**; measure through the frame-gate.

### 4b. Multi-lens classification metrics
top-1 · macro-F1 · balanced accuracy · per-class precision/recall · Cohen's κ · Matthews MCC · confusion matrix.

### 4c. Safety metrics (highest priority in this domain)
Per-class **FNR / miss-rate**; **cost-sensitive** scoring, miss penalised more heavily than false alarm.
Cost matrix declared **before** results and reported alongside them.

### 4d. Operating point
ROC-AUC · PR-AUC · threshold sweep → operating point selected **by safety recall**, not by accuracy.

### 4e. Calibration
**ECE** + **reliability diagram**.

### 4f. Statistics
Bootstrap **5,000** resamples + **Wilson / BCa** CIs · **McNemar exact** (two models, same 64 clips) ·
**Nadeau–Bengio** corrected t for k-fold. Every number carries n, CI, seed count, protocol tag.

### 4g. Jetson metrics (MT3)
Latency **p50/p95/p99**, in **ms/frame and ms/clip** · FPS · **multi-stream aggregate throughput +
zero-drop** verification · model size / params / **FLOPs** · **INT8 compression ratio** · energy:
**J per inference** and **FPS/W**.

### 4j. Alternative evaluation frameworks (ADOPTED 2026-09-06)

Since a direct head-to-head win on the paper's number is closed (see the N=64 ceiling, S1), these
frameworks answer *different questions* the paper never posed. They are additions, not substitutes.

**Adopted, load-bearing:**
- **Conformal prediction / selective risk (AURC, risk-coverage curve).** The strongest addition:
  finite-sample-exact coverage guarantee that holds at n=64. Dependency **CONFIRMED AVAILABLE** —
  `sweep_results/relation_tokens/logits_*.pt` each carry `ens_softmax` + `labels` + `blind_pred` for
  6 configs, so no re-inference is needed. Caveat to state: Mondrian per-class guarantees are **not**
  attainable (smallest class n=8) — report marginal coverage only, and say why.
- **Resolving power of the benchmark.** Fisher-z 95 % CIs on the existing val/test correlations all
  span zero and are wide (e.g. r = -0.247, CI [-0.50, +0.05], N = 46). This converts "this benchmark
  cannot rank models" from an assertion into a measured quantity. Nearly free; high argumentative value.
- **Required-n analysis.** ~111 clips/arm to conclusively show non-inferiority at delta = 10 pp, ~445 at
  5 pp. **This required-n IS the reportable result** — a concrete benchmark-design recommendation.

**Adopted with strict wording limits:**
- Non-inferiority (TOST): unpaired only, since the paper's per-clip labels are unreleased. Our best
  (56/64) would need a margin >= 18.4 pp to reach p < .05 — indefensible in a safety context.
  **Report as "not achievable at this N", never as a non-inferiority claim.**
- Bayesian ROPE: P(ours > paper) = 1.5-2.7 %; P(practically equivalent, +/-10 pp) = 46-59 %.
  Supports "tied rather than badly beaten"; **must not be written as "proven equivalent"** - 46-59 %
  is not proof of anything.
- Effect size: Cohen's h = -0.37 to -0.62 (medium-large). OR and RD disagree at the margin
  (small-sample artifact) - **report both and name the disagreement**; do not pick the flattering one.

**Hard limit.** None of this reopens the 62/64 head-to-head. That comparison is closed and the thesis
says so plainly. See section C of the report for the full list of 8 things still unclaimable.

Open dependencies: per-fold per-config score matrices (blocks the correlated-t Bayesian variant and
RQ6) - existence not yet verified. `statsmodels` must be added to the pinned dependencies.

Full derivations and computed numbers: `research/r5-alternative-evaluation-frameworks.md`.

### 4i. Event-level / streaming operational metrics (ADOPTED 2026-09-06, Gate-3 conditional)

Rationale: axis 7 (edge deployment) currently proves *speed* but not *operational usefulness*. Clip-level
accuracy cannot answer "how many false alarms per hour will an operator absorb?" — the question that
decides whether such a system is deployable. The paper never measured this.

**Hard constraint:** the repo contains **no continuous/untrimmed footage** — only pre-cut 7-15 s clips.
Therefore streaming eval runs on a **synthetic stream**: existing labelled clips concatenated with >= 2 s
safe buffers, concat timestamps serving as exact segment ground truth (zero new annotation).

**Mandatory disclosure — non-negotiable.** Every table, figure, and paragraph reporting these metrics must
state that the stream is *synthetic (ghép nối clip)*, not continuous factory footage, and must name the
realism ceiling: no natural scene transitions, no ambiguous behaviour onsets, no inter-event context. These
numbers characterise the **alarm logic**, not field performance. Any wording implying real deployed
footage is a hard-rule violation.

Metrics: segment precision/recall/F1 and mAP over tIoU thresholds; false-alarm rate per hour; time-to-detect;
event miss-rate; DET curve (**full curve required** — a single operating point is a cherry-pick); temporal
flicker/flap rate before vs after debounce.

Latency must be **decomposed**, not reported as one number. Prior finding: `jetson-live-monitor.py`'s rsync
poll (`POLL_INTERVAL_SEC = 1.0`) is likely the dominant glass-to-alarm term, dwarfing model inference — if
so, report it plainly; it is a more honest and more interesting result than the inference figure alone.

Add N-of-M debounce + refractory timer to `jetson-live-monitor.py` (currently 2-tier thresholds, no
debounce); report flap-count before/after as the measured effect.

**Banned:** point-adjusted F1 (Kim et al., AAAI 2022 — a random detector scores > 0.9 under it).
**Banned:** extrapolating the 36.2-min soak pilot to any 24/7 availability claim.
Event counts will be ~50-150 → CIs will be wide; report them, do not hide them.

Budget ~14-19 h, Lane B. **Conditional on Gate 3**: if Lane A or B is behind at day 9, this axis is the
first thing cut — it is an enhancement to axis 7, not a load-bearing claim.

Full design: `research/r6-operational-safety-and-streaming-metrics.md`.

### 4h. Protocol discipline (non-negotiable)
1. Splits fixed by the paper's release: 566 train / 125 test; the 64 unsafe test clips are the test set.
   Val carved from train only, stratified on (label, view).
2. Model / epoch / seed / fusion selection **only** on val. Never on test.
3. **One declared test touch per pipeline variant**, timestamp-logged in `reports/test-touch-log.md`.
4. Headline configuration pre-registered before the touch.
5. ≥ 5 seeds for any quoted number. No single-seed claims.
6. A failed run gets a **control** before it gets an explanation.

---

## 5. Lanes, ownership, concurrency

| Lane | Scope | Writes (exclusive) | Concurrency |
|---|---|---|---|
| **A** Accuracy | A1–A5 | `sweep_results/`, `runs/`, `cache_r2/`, `train_behaviour_detector.py`, `run_kfold_relation.py`, `run_lco_eval.py`, `kaggle-detector-retrain/` | **Serialized** — one heavy job at a time |
| **B** Edge | Jetson export, benchmarks, demo | `deployment/`, `tools/jetson_benchmark.py` | Concurrent with A (separate hardware) |
| **C** Thesis (VN) | Ch1–Ch6 + abstract + assembly | `report/` | Concurrent; Ch4/Ch5 gated on A4/B2 |
| **D** Repro & docs | `docs/` (7 files), README, repro scripts, review | `docs/`, `README.md`, `scripts/` | Concurrent |
| **E** Defence | slides, Q&A brief, figures | `plans/2026-09-06-*/slides/`, `figures/` | Gated on A4/B2 |
| **F** Metrics | §4 suite as reusable code | `eval_utils/` | Concurrent, front-loaded — must land before A4 |

A and B use different hardware; C/D/E/F are writing/CPU-light. This is real parallelism — not the
previous plan's fiction of five simultaneous training workers.

---

## 6. Deliverables (definition of done)

1. **Luận văn (VN)** — Ch1 Giới thiệu · Ch2 Tổng quan · Ch3 Phương pháp · Ch4 Thực nghiệm & Kết quả ·
   Ch5 Triển khai biên · Ch6 Kết luận · English abstract · references · assembled single document.
2. **Slides bảo vệ (VN)** + Q&A brief where every anticipated question maps to a file and a number.
3. **Reproducible repo** — `docs/` populated, README rewritten, one-command repro per headline number,
   pinned environment, code review passed.
4. **Live demo** — video/camera → edge node (Jetson) → server → dashboard, real-time, runnable at defence.

---

## 7. Schedule (14 days)

| Day | Date | Lane A (accuracy) | Lane B (edge) | Lanes C/D/E/F |
|---|---|---|---|---|
| 1 | Sep 6 | Plan lock; **A1 GPU detector training launched** | Jetson env verify, JetPack/TRT inventory | F: §4 suite spec; C: Ch1/Ch2 VN outline |
| 2 | Sep 7 | A1 results: detector card, PR curves, 8/8 class coverage | B1 export path spike | F: §4 suite implemented |
| 3 | Sep 8 | A2 behaviour-clip + frame-gate; 0.14 s / 1.87 s replication | B1 ONNX → TRT FP16 | C: Ch1, Ch2 drafted |
| 4 | Sep 9 | A3 relation-Transformer retrain (seeds ≥ 5) | B2 latency p50/95/99, FLOPs, size | C: Ch3 drafted |
| 5 | Sep 10 | A3 k-fold 516 + LCO 516 | B2 INT8 calibration + parity | D: `docs/` skeleton |
| 6 | Sep 11 | **A4 headline freeze + single test touch + full §4 suite**; A5 safety/calibration | B2 multi-stream + zero-drop | E: figures regenerated |
| 7 | Sep 12 | Buffer / contingency arm | B2 soak 30 min + power (J/inf, FPS/W) | C: Ch4 drafted from A4 |
| 8 | Sep 13 | Ablation table + error analysis | B3 demo pipeline wiring | C: Ch4 finalised |
| 9 | Sep 14 | — | B3 dashboard + live run | C: Ch5 drafted from B2 |
| 10 | Sep 15 | — | B3 rehearsal + fallback recording | C: Ch6 + EN abstract |
| 11 | Sep 16 | — | — | D: README, repro scripts, `docs/` complete |
| 12 | Sep 17 | — | — | C: full VN assembly + consistency pass |
| 13 | Sep 18 | — | — | E: slides + Q&A brief; code review |
| 14 | Sep 19 | — | — | Final review, freeze, buffer |

**Critical path:** A1 → A2/A3 → A4 → Ch4 → assembly. Only a day-1/2 slip in A1 threatens the thesis.

---

## 8. Go / no-go gates

- **Gate 1 (end of day 2).** A1 detector mAP50 ≥ 0.90 **and** all 8 classes emitted → proceed to A2/A3.
  If mAP50 < 0.85 or classes still dead → stop the detector attack, keep the existing P3 k-fold/LCO
  evidence as the headline, reallocate days 3–6 to Lanes B/C/D. **Either outcome is a reportable result.**
- **Gate 2 (end of day 6).** Headline frozen, test touch spent, §4 suite complete. After this, **no
  further accuracy experiments** — a new number would land inside the existing CI and is not a result.
- **Gate 3 (end of day 10).** Demo runs end-to-end unattended for 10 min, or the fallback recording is
  cut and the live demo becomes a recorded demo.

## 9. Hard rules
1. No comparison to the paper without an interval or a test statistic in the same sentence.
2. Report what ran, including what failed. The MPS bug, the withdrawn 57/64, the withdrawn +3.5 pp
   macro win, and the negative results stay in the thesis — they are the methodological contribution.
3. No test-set peeking to select anything. Ever.
4. Every failed run gets a control before it gets an explanation.
5. The deployment chapter contains **only measured numbers from our pipeline on our hardware**.
   Literature figures, if used at all, are labelled as context and never as our result.


**S9 — an artifact was overwritten in place, silently invalidating three verified reports (found
2026-09-06 via a failing test, not via review).** `eval_utils/tests/test_real_artifacts.py::
test_reproduces_brief_kfold_516_headline` failed asserting pooled `417/516`; the file
`sweep_results/relation_tokens/kfold_oof_softmax_yolo26app_.pt` now yields **421/516 (81.59 %), macro
84.24 %**. Nothing is corrupt: the canonical `--repeats 10` run (`canonical/
kfold_repeats10_yolo26app_HEADLINE.log`, mtime Sep 6 03:22) legitimately regenerated the OOF selection
set — **under the same filename**. The superseded single-repeat run's 417/516 is simply gone.

Two distinct defects, and the second is the instructive one:

1. **Stale pins.** `qa1-adversarial-review-and-defence.md` (row 1), `t1-objectives-audit.md` (L97),
   `f1-metric-suite.md` (L189) and `w2-number-integrity-fixes.md` (L307) each record 417/516 as
   *independently verified against this exact path*. Every one of those verifications was correct when
   written and is now unreproducible from the path it cites. They are dated reports and are left as
   written; this entry is the pointer that supersedes them. **Nothing in `report/` quotes 417/516** —
   verified by grep — so the thesis text is unaffected.
2. **A quantity conflation the coincidence hid.** "80.8 % (417/516)" pairs a **fold-mean** percentage
   with a **pooled** count. These are different estimators. They agreed numerically only because the old
   pooled value (80.81 %) happened to land on the new canonical fold-mean (80.8 %). Canonically they
   separate: fold-mean **80.8 ± 4.0 %** (NB-corrected, the number with the CI) vs pooled
   **421/516 = 81.6 %**. The test now asserts the pooled value and carries a docstring naming both.

**Rule this establishes:** a result file regenerated under an existing filename destroys the evidence
base of every report that cited it, and no review catches this — only a pinned test does. Prefer
run-tagged filenames; where a path must be reused, the pinned test *is* the tripwire. Note the failure
mode is silent in the flattering direction too: 421/516 is *higher* than 417/516, so the discrepancy
would never have announced itself as a problem.

**S10 — operational note: the section sign is mangled in commit messages.** Commits `a26e69e`,
`c2538ed`, `b2f530d`, `7e7f81d` read "5.5", "5.7.1", "5.8" where they mean §7.5, §7.7.1, §4.8.
Every "§X.Y" typed into a `git commit -m` argument on this stack arrives as "5.Y" — the section
sign survives fine inside files (verified: it is stored as the correct C2 A7 bytes) but not through
the commit-message argument path. **The commit contents are correct; only the log text is wrong.**
The messages could not be corrected: `git reset --hard` and `git commit --amend` are both blocked
here as history-rewriting operations, which is the right default and was not worked around. The
commits are unpushed, so a maintainer can still fix them with an interactive rebase if desired.
**Going forward, write "sec 7.5" rather than "§7.5" in commit messages**, or pass the message with
`-F <file>`, which preserves the byte.
