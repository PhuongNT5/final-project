# Phase 2 · Motion features — REJECTED (LCO did not rise; near-tie)

**Date:** 2026-09-06 · **Verdict:** REJECT (LCO macro −0.5 pp vs champion; a near-tie loss) · **Lever:** motion

## What was tested
Track-linked motion appended to the champion token set. `tools/build_motion_tokens.py` post-processes the
appearance base (`yolo26app_`, still carries xc,yc), greedily links person boxes (class 0/1) across the
T=16 frame axis by nearest centroid (gate 0.12 norm-dist), and appends **2 camera-robust, track-level dims**
(broadcast to every token of the track):
- `mot_speed`    = mean per-frame centroid displacement / bbox-height  (body-heights/frame; scale-free)
- `mot_straight` = net displacement / path length                       (1=straight line, 0=wandering)

Then `build_l1_tokens.py --src yolo26mot_ --dst yolo26motL1_` strips xc,yc but keeps the 2 motion dims at
the feature tail (74 dims: `…app_63, mot_speed, mot_straight`). Judged vs champion `yolo26L1_` under the
IDENTICAL protocol (3 seeds, repaired scene-map, 60 ep).

Coverage sanity: 70.5% of valid person tokens carry motion (1788 train / 241 test tracks); speed p50=0.045
p95=0.199 body-heights/frame; straightness centered ~0.54 — bounded, sensible, non-degenerate.

## Result (identical protocol)
| Metric | Motion (yolo26motL1_) | Champion (yolo26L1_) | Δ |
|---|---|---|---|
| **LCO macro** (headline) | **81.5%** | **82.0%** | **−0.5 pp** |
| LCO top-1 | 78.9% | 77.9% | +1.0 |
| k-fold macro (secondary) | 83.5% | 84.7% | −1.2 |
| k-fold top-1 | 79.5±2.7% | 80.6±2.8% | −1.1 |

LCO macro did not rise → fails the campaign rule (keep iff LCO↑ and no leak) → **rejected**. Both metrics
down (not the k-fold↑/LCO↓ leak signature); the −0.5 pp is within run-to-run noise, so this is a **near-tie
loss**, not a clear regression. The +1.0 LCO top-1 is real but the headline is macro (class-balanced).

## Per-class LCO recall — the real, informative effect
| | Motion | Champion | Δ |
|---|---|---|---|
| WalkwayViol (crosser) | **75.2** | 68.1 | **+7.1** |
| UnauthorInt (loiterer) | **62.0** | 70.4 | **−8.4** |
| OpenedPanel | 88.7 | 89.4 | −0.7 |
| CarryOverload | 100.0 | 100.0 | 0 |

Motion shifts EXACTLY the WalkwayViol↔UnauthorInt axis it was designed to touch, and the direction is
mechanistically correct — a moving/straight-path person reads as WalkwayViol (+7.1). But the model
over-applies it: a Person_Unauthorized who happens to move now leaks into WalkwayViol, collapsing intrusion
recall (−8.4). Net macro slips. This is the SAME class-prior-shift failure the rejected zones-detector arm
produced (WalkwayViol +7.6 / UnauthorInt −16.7) — a recurring signature: features that "explain" crossing
bias the head toward WalkwayViol at intrusion's expense.

## Why it didn't win (and what it implies)
UnauthorInt is the harder class and motion is a weak per-clip discriminator for it: an intruder standing
still and a walkway-user paused mid-frame both look low-motion, so speed/straightness cannot rescue the
loiterer while it actively promotes the crosser. Broadcasting one track-level scalar to all its tokens also
gives the attention head an easy shortcut to WalkwayViol. Motion is real signal, mis-integrated — same
lesson as zones: geometry/temporal cues help one side of the confusion but need to enter the head WITHOUT
rewriting class priors (e.g. gated/auxiliary, not a competing broadcast feature).

## Artifacts (all additive — no narrative files touched)
- Tool: `tools/build_motion_tokens.py` (KEPT — reusable; documents the linking + feature math)
- LCO log: `sweep_results/relation_tokens/_motion_lco.log` (KEPT — provenance)
- k-fold log: `sweep_results/relation_tokens/_motion_kfold.log` (KEPT — provenance)
- Tokens `tokens_yolo26mot_*` (76-dim intermediate), `tokens_yolo26motL1_*` (74-dim), OOF
  `kfold_oof_softmax_yolo26motL1_.pt`: candidates for purge (rejected set). Regenerate via the two build
  commands above. Champion `yolo26L1_` / `yolo26app_` untouched.

## Unresolved questions
1. Would motion help if fed as an AUXILIARY head (predict "is-crossing") or a GATED feature rather than a
   broadcast token dim — decoupling it from the WalkwayViol class prior? The +7.1 WalkwayViol recall says
   the signal is real; the −8.4 UnauthorInt says the integration is wrong.
2. Is per-person dwell (fraction of frames with speed≈0), rather than mean speed, a better UnauthorInt cue —
   i.e. attack the loiterer side directly instead of the crosser side?
3. Same YOLO26-vs-champion base here (no detector confound, unlike zones) — so this is a clean lever result:
   motion, as a token feature, is a wash on LCO macro.
