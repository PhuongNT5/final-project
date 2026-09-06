# The no-pixel floor: Unsafe-Net's benchmark is camera-confounded

**Date:** 2026-09-06. Reproduce: `.venv/bin/python -m unsafenet.scene_confound`
→ `sweep_results/scene_confound.json`. Source: `unsafenet/scene_confound.py`.

## Claim

On the 4-class unsafe task, **camera identity alone predicts the label**. A classifier that sees
**no pixels at all** — input is only "which of the 8 fixed CCTV views did this clip come from" —
scores **52/64 = 81.2 %** on the paper's own test set and **77.1 % top-1 / 75.8 % macro** pooled
over all 516 unsafe clips.

This reframes every same-distribution number in the literature on this dataset, including ours.

## Evidence

### 1. View and label are near-deterministically coupled (516 pooled clips)

| statistic | value |
|---|---|
| χ² (view × label), dof 21 | 950.9, p = 8.0e-188 |
| Cramér's V | **0.784** |
| H(label) | 1.860 bits |
| I(view; label) | **1.101 bits** |
| share of label entropy given by camera id | **59.2 %** |

Class → cameras it ever appears in:

| class | n | views |
|---|---|---|
| WalkwayViol | 210 | 0,1,3,5,6 |
| UnauthorInt | 108 | 0,1,3,4,5,6 |
| OpenedPanel | 142 | 0,3,6 |
| **CarryOverload** | 56 | **2,7 only** |

Cameras **2, 4, 7 contain exactly one class each** — for those clips the camera id *is* the label.

### 2. The no-pixel floor on the paper's fixed 64-clip test

Scene prior fit on the 452 train clips only, applied blind to the 64 test clips.

| system | test | top-1 | 95 % CP CI |
|---|---|---|---|
| **camera id only (no pixels)** | 52/64 | **81.2 %** | [69.5, 89.9] |
| ours (YOLO26 + relation Transformer) | 54/64 | 84.4 % | [73.1, 92.2] |
| paper Unsafe-Net, no frame reduction | 57/64 | 89.1 % | — |
| paper Unsafe-Net, full | 62/64 | 96.9 % | — |

- Ours vs no-pixel floor: discordant 9/7, **McNemar exact p = 0.804** — our model is *statistically
  indistinguishable from knowing only the camera id* on this test set.
- The paper's headline 62/64 is **+15.6 pp over a baseline that never looks at the image**.
- 4 of the 8 test views (0, 2, 4, 7 — 17 of 64 clips) contain a single class, so the floor is
  trivially 100 % on 27 % of the test set.

### 3. Per-view breakdown on test — where the model actually adds signal

| view | n | classes present | floor | model |
|---|---|---|---|---|
| 0 | 7 | 1 | 100.0 | 100.0 |
| 1 | 27 | 2 | 88.9 | **96.3** |
| 2 | 5 | 1 | 100.0 | 100.0 |
| 3 | 5 | 3 | 20.0 | **60.0** |
| 4 | 2 | 1 | 100.0 | 100.0 |
| 5 | 8 | 2 | 62.5 | 50.0 |
| 6 | 7 | 3 | 71.4 | 57.1 |
| 7 | 3 | 1 | 100.0 | 100.0 |

The model gains only on the multi-class views 1 and 3 and loses on 5 and 6. On the 12 test clips the
camera prior gets wrong, the model recovers 9 (75.0 %); on the 52 it gets right, the model keeps 45
(86.5 %) — i.e. it **destroys 7 clips the free baseline already had**.

### 4. Cross-camera (LCO) is the only protocol that separates the two

From `plans/2026-09-02-fair-headtohead-vs-paper/reports/fair-headtohead-scorecard.md`:

**Baseline choice matters, and an earlier framing of this got it wrong.** Under LCO the *camera-id*
prior is undefined (the held-out camera was never seen), so it degenerates to a constant class-prior
predictor at 16.9 % / 10.4 %. Comparing against that constant yields a headline "+60.3 pp" which is
not an honest measure of anything — a constant predictor is not a scene model. The honest LCO
comparator is an **appearance model that actually transfers**: whole-frame features fit on the
training cameras and applied to the held-out camera.

| LCO ladder (pooled 516 clips) | top-1 | macro |
|---|---|---|
| constant class prior (trivial, *not* a scene model) | 16.9 % | 10.4 % |
| frozen CLIP probe | 52.7 % | 60.4 % |
| **whole-frame appearance (honest comparator)** | **59.9 %** | **66.2 %** |
| YOLO + relation Transformer | **77.1 %** | **78.6 %** |

**Honest margin: +17.2 pp top-1 / +12.4 pp macro** over whole-frame appearance. Not +60.3 pp.

Across protocols, against the appropriate comparator for each:

| protocol | comparator | comparator | our model | margin |
|---|---|---|---|---|
| same-distribution 64 | camera-id prior | 81.2 % | 84.4 % | **+3.1 pp** (n.s., p = 0.80) |
| random k-fold 516 | per-view majority | 77.1 % | 80.8 % | **+3.7 pp** (NB p = 0.134, n.s.) |
| **leave-one-camera-out 516** | whole-frame appearance | 59.9 % | 77.1 % | **+17.2 pp** |

The point stands, at correct magnitude: the behaviour signal is invisible under the paper's protocol
(+3.1 pp, n.s.) and clearly present cross-camera (+17.2 pp).

## What this means for the thesis

1. **Do not headline a same-distribution 64-clip number**, ours or the paper's. On that protocol our
   model is not distinguishable from a no-pixel baseline, and the metric is over half determined by
   camera identity before any model runs.
2. **Report the no-pixel floor beside every accuracy number.** It is one line of code, it is
   assumption-free, and it converts an uninterpretable 84.4 % into "+3.1 pp of behaviour signal".
   The paper reports no such control, so its 96.88 % has no floor to be read against.
3. **LCO becomes the primary endpoint**, with same-distribution demoted to a comparability footnote.
   This is defensible on measurement grounds, not on convenience: it is the only protocol in which
   the free baseline does not already win.
4. This is a **dataset-design finding, not an attack on the paper's engineering.** The confound is a
   property of how Unsafe-Net was filmed (each behaviour staged at its own station). Any method
   trained and tested on it same-distribution inherits the confound.

## Honest limits

- The no-pixel floor is a *lower* bound on scene leakage only. Background pixels also leak, so a
  whole-frame CNN can exceed the camera-id floor while still learning no behaviour.
- View labels come with the dataset; we did not verify them against the footage.
- The paper's per-clip predictions are unreleased, so we cannot run the paired McNemar of
  paper-vs-floor. The +15.6 pp figure is a difference of published point estimates, not a test.
- LCO folds are unbalanced: views 2, 4, 7 are single-class, so per-fold macro is degenerate and only
  the pooled-over-folds macro is meaningful. Confirm `run_lco_eval.py` pools rather than averages.

## Unresolved questions

1. Promote LCO to the thesis headline metric, with same-distribution kept only for paper
   comparability? (Recommended.)
2. Should the no-pixel floor be extended to a *background-only* floor (mean background frame, no
   foreground) to bound pixel-level scene leakage as well?
3. The 175 safe-class clips carry no view label. Does recovering those labels matter, given the
   4-class unsafe pool (516) is fully labelled and is the evaluated task? (Appears: no.)
4. Views 2/7 hold CarryOverload exclusively — under LCO the class is learned from exactly one other
   camera. Is a 2-camera class strong enough to claim cross-view generalisation for it?
