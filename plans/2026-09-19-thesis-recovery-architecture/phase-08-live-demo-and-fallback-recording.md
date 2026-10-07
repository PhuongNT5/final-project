---
phase: 08
title: "Live demo + mandatory fallback recording"
lane: C
effort: 8h
gate: "G7 fallback .mp4 exists"
status: pending
---
# Phase 08 — Live Demo + Mandatory Fallback Recording

## Context Links
- Master plan: [plan.md](plan.md) — lanes, calendar, gates, test-touch rules.
- Companion phase: [phase-07-jetson-measurement-runbooks.md](phase-07-jetson-measurement-runbooks.md) — shares Jetson device time for the DROPPABLE Tier-2 recording; owns all Jetson TRT-engine builds.
- Method source, **UNVERIFIED — citation-checked, do not trust its illustrative numbers**: `research/researcher-03-demo-and-edge-deployment.md` (§1 demo option matrix, §6 risk mitigation, §7 build plan — procedural framing reused; its `demo-local.py` skeleton has a real threshold bug, see Key Insight 3; its §5.2 numeric table is a confirmed placeholder, never copy).
- Fact-check: `reports/critique-2026-09-19-research-factcheck.md` — confirms researcher-03 §5.2 is illustrative, not measured.
- **Governing accuracy-lane finding:** `reports/finding-2026-09-19-lco-failure-is-camera-label-confounding.md` — E3 DeepSets (18,044 params) is the frozen deployment model; encoder search is cut project-wide. This phase wires E3 only.
- Capability ground truth: `reports/audit-2026-09-19-repo-capability-and-feasibility.md` §6 — confirms `deployment/dashboard/app.py` shows class+confidence but draws **no bounding boxes**; no `.mp4` fallback exists anywhere in the repo yet; no live boxes+verdict demo path exists for P3.
- Export provenance: `deployment/export/MANIFEST.md` — E3 checkpoint sha256, val_macro, git commit at export time.
- Code precedent reused (not duplicated): `deployment/scripts/benchmark-p3-utils.py` (`BRT` import pattern, `postprocess_frame`, `extract_crops` — exact detector-output-to-crop path used by training); `deployment/dashboard/app.py` (`T_WINDOW`/`EMIT_EVERY` sliding-window pattern, `importlib.util.spec_from_file_location` cross-file reuse pattern); `build_relation_tokens.py` (`geometry()`, `process_clip()` — the exact per-frame token-row layout: `[cls, cx, cy, w, h, conf] + geometry(4) + appearance(64) = 74-dim`).
- Depends on: phase-04 (ONNX export, done — `p3-relation-head-bin-e3-fp32.onnx` exists and is trained). Independent of phase-07's Jetson sessions for its MANDATORY deliverable (Tier-1 fallback needs no device).

## Overview
- **Priority:** P2 of 3 (accuracy is P1, edge is P2, robustness P3). The demo is a deployment-pillar deliverable, not an accuracy result — it must never be allowed to imply otherwise.
- **Status:** No demo code exists yet (`deployment/demo-local.py` confirmed absent this session). No fallback video exists anywhere in the repo. This phase creates the first-ever live P3 demo and the first-ever fallback recording.
- **Brief:** build a local Python+OpenCV live-inference demo (PRIMARY) that runs the frozen E3 pipeline end-to-end on a real video source and shows boxes, decision, confidence, and an explicit abstain band; record a screen-capture fallback `.mp4` of it running (MANDATORY, Gate G7) before the 2026-10-05 number freeze; rehearse twice; never display any number outside the frozen number sheet.

## Key Insights
1. **E3 is the only relation-head wired into the demo.** Per the governing finding, E3 (18,044 params, genuinely dynamic-batch ONNX) is the frozen deployment model. `p3-relation-head-bin-e0-fp32.onnx` and the legacy 4-class graph are **available ONNX files on disk, not wired into `demo-local.py`** — mention them in code comments as "not used here" only, never as an in-demo toggle. There is no E0-vs-E3 choice to present.
2. **The dashboard already proves boxes are not free.** `deployment/dashboard/app.py`/`pipeline-utils.py` (grepped this session) contain zero `rectangle`/`box` drawing calls — only a Streamlit sidebar + class/confidence text. All box/overlay rendering in this phase is new code, not a port.
3. **researcher-03's own demo skeleton has a live threshold bug — fix it, do not copy it.** Its pseudocode sets `CONF_THRESHOLD = 0.6` but `DECISION_ABSTAIN_ZONE = (0.45, 0.55)`, leaving `prob_unsafe` in `[0.55, 0.6]` unclassified (falls through both branches). Corrected logic for this phase: abstain if `prob_unsafe` in `(0.45, 0.55)`; otherwise split cleanly at `0.5` (`Unsafe` if `> 0.5` else `Safe`). No unclassified gap.
4. **Zone geometry is camera-specific — this bounds what "live" can honestly mean.** `build_relation_tokens.geometry()` computes IoU/foot-point features against `annotation_guide` zone polygons keyed by `view_id` (`load_zones()`/`view_map()`). A true arbitrary webcam feed has no matching `view_id`, so `geometry()` silently zero-fills those 4 of 74 feature dims (`zones` falsy → `[0,0,0,0]`) — a distribution shift the frozen model never saw in training. **Primary demo source is therefore a real `data/test/*.mp4` clip** (known view, real zones, genuinely live-computed inference on unseen-at-runtime frames) rather than an ad hoc webcam. Live webcam is an optional DROPPABLE stretch mode, shipped with an on-screen disclosure ("zone features unavailable, geometry zero-filled — demonstration only, not a validated operating condition") if attempted.
5. **The demo reads the frozen checkpoint only.** `demo-inference-pipeline.py` loads `deployment/export/p3-relation-head-bin-e3-fp32.onnx` (fp32, ONNX Runtime `CPUExecutionProvider` — 18k params needs no TRT/GPU on a laptop) read-only. No training code (`train-binary-head-checkpoint.py`, `run_kfold_relation.py`, etc.) is imported anywhere in the demo file set. No retraining, no new numbers, no gradient computation.
6. **A live-computed-vs-precomputed-token parity spot-check is not a test touch.** Step 4 below compares the demo's live per-frame token assembly against the existing `tokens_bin8_test.pt` rows for the same clip to catch encoding bugs (wrong crop padding, wrong geometry call, etc.) — this is a code-correctness check (like phase 07's ONNX parity), not a scored evaluation, and spends 0 of the 2 pre-registered test touches, mirroring phase 07's `--rule argmax` decision-agreement framing.
7. **Headline accuracy numbers may be spoken/slide context, never demo-rendered.** E3 full LCO-691 BA **69.6%**, panel-disclosed **76.6%** (both from the finding report, 0 test touches) may appear on a static title slide or in narration, always protocol-labeled, and only after the 2026-10-05 number freeze (`plans/2026-09-19-*/number-sheet.md`, Lane D, not yet written). `demo-render-overlay.py` must never hardcode either number or any other accuracy figure into the live overlay — the overlay shows only this-frame model output (probability, decision, abstain state), which is not a "number from the sheet," it is live inference.
8. **Fallback recording is MANDATORY and independent of device scheduling.** Gate G7 only requires a working `demo-local.py` run on a laptop, screen-captured — it does not need the Jetson. Do not let phase-07's device-availability risk block G7; a device-recorded Tier-2 clip is a DROPPABLE bonus, not the gate.
9. **New files must stay under 200 lines each** — split by concern (session/pipeline, rendering, entrypoint, optional calibration) rather than one large script, following the existing `benchmark-p3-onnx.py`/`benchmark-p3-utils.py` split precedent in this same directory.

## Requirements
**Functional**
- `demo-local.py` runs end-to-end on a real video source at interactive frame rate on a Mac (M4), showing: per-box overlay (box + class), rolling decision (Safe/Unsafe/Abstain), confidence readout, and a visually distinct abstain-band state.
- Fallback `.mp4` exists, plays standalone (no code/dependencies needed), shows the same demo running successfully end-to-end, ≥60s.
- Zero numbers hardcoded in overlay code beyond live model output; no accuracy/BA/F1 figure anywhere in `demo-*.py`.
- Zero training-code imports in the demo file set (`grep -L` check in step 4).

**Non-functional**
- No fabricated behavior: if a stage fails (missing ONNX file, camera not found), the demo must error/abort visibly, never silently substitute placeholder output.
- Reproducible: demo config badge (rendered or logged) states checkpoint sha256 + git commit, same convention as phase 07's measured-configuration table.
- 0 test touches spent by anything in this phase (see Key Insight 6).

**Demo-config badge (render or log on every run)**
| field | value |
|---|---|
| Checkpoint | E3 DeepSets, seed 0, val_macro 89.15% (training-selection metric, not a test-accuracy claim) |
| Checkpoint sha256 | `d1789e6a...b3177a` (full hash in `MANIFEST.md`) |
| Git commit | pin at demo-build time |
| Source | video file path (or `--webcam`, flagged OOD) |
| Test touches spent | 0 (always) |

## Architecture
```
video source (data/test/*.mp4, or --webcam [DROPPABLE, OOD]) --cv2.VideoCapture-->
  rolling frame buffer (T_WINDOW frames, adapted from dashboard/app.py's T_WINDOW=32/EMIT_EVERY=8)
    -> sample T=16 frames from buffer (build_relation_tokens.process_clip's even-spacing rule)
    -> per frame: ORT detector session -> postprocess_frame() [reused from benchmark-p3-utils.py]
       -> top N_MAX=8 boxes -> extract_crops() [reused] -> ORT appearance session -> 64-d embeddings
       -> geometry() [reused from build_relation_tokens.py] for person-class boxes
       -> row = [cls, cx, cy, w, h, conf] + geometry(4) + appearance(64)  # 74-dim, exact training layout
  -> tokens (1,16,8,74) + mask (1,16,8) -> ORT relation-head-bin-e3 session -> logits(1,2)
  -> softmax -> prob_unsafe -> decision: abstain if in (0.45,0.55), else Unsafe if >0.5 else Safe
  -> demo-render-overlay.py: draw boxes + decision + confidence bar + abstain-band cue -> cv2.imshow / video writer
```
File split (all new, kebab-case, <200 lines each):
- `deployment/demo-local.py` — argparse entrypoint, main loop, wires pipeline + render.
- `deployment/demo-inference-pipeline.py` — ORT session loading, rolling-window token assembly, decision logic.
- `deployment/demo-render-overlay.py` — pure cv2 drawing functions, no model calls.
- `deployment/demo-calibrate-threshold.py` (DROPPABLE) — val/OOF-only abstain-band sanity check.

## Related Code Files
**Modify** (Lane C scope: `deployment/**`, `tools/jetson_benchmark.py` only): none required — all deliverables are new files; no existing script needs edits for this phase.

**Create**
- `deployment/demo-local.py` — entrypoint (~120 lines).
- `deployment/demo-inference-pipeline.py` — session/window/decision logic, imports `build_relation_tokens` and `benchmark-p3-utils` functions the same way `benchmark-p3-utils.py` imports `build_relation_tokens` (`sys.path.insert(0, ROOT)`; `import build_relation_tokens as BRT`) (~180 lines).
- `deployment/demo-render-overlay.py` — cv2 drawing only (~100 lines).
- `deployment/demo-calibrate-threshold.py` (DROPPABLE) — val/OOF abstain-band check (~80 lines).
- `deployment/demo-fallback-recording.mp4` — MANDATORY Tier-1 fallback (binary asset, Gate G7).
- `deployment/demo-fallback-recording-jetson.mp4` (DROPPABLE, Tier-2) — device-recorded variant.
- `deployment/demo-rehearsal-log.md` — dated rehearsal notes (glitches, timing, fixes).

**Delete:** none.

## Implementation Steps

0. **[M4-CPU, ~5 min]** Pre-flight: confirm assets exist before writing any code.
   ```bash
   test -f deployment/export/p3-detector-fp32.onnx && \
   test -f deployment/export/p3-appearance-fp32.onnx && \
   test -f deployment/export/p3-relation-head-bin-e3-fp32.onnx && echo "all E3 graphs present"
   sha256sum deployment/export/p3-relation-head-bin-e3-fp32.onnx   # cross-check vs MANIFEST.md
   ls data/test/*/*.mp4 | head -3   # confirm a real in-distribution clip exists to demo against
   ```

1. **[M4-CPU, ~30 min]** Write `deployment/demo-inference-pipeline.py`: ORT session loader for the three E3-path graphs (`CPUExecutionProvider`, `so.log_severity_level = 3` per `dashboard/app.py`'s convention); rolling `T_WINDOW`-frame buffer + `EMIT_EVERY`-frame decision refresh (values adapted from `dashboard/app.py`'s `T_WINDOW=32`/`EMIT_EVERY=8`, sized to P3's `T=16`); per-frame token row assembly that calls `BRT.geometry()` and mirrors `build_relation_tokens.process_clip`'s exact row layout (`[cls, cx, cy, w, h, conf] + geometry(4) + appearance(64)`); corrected abstain-band decision function per Key Insight 3.

2. **[M4-CPU, ~20 min]** Write `deployment/demo-render-overlay.py`: box-draw function (from `postprocess_frame()`'s returned boxes), decision banner (color-coded: green Safe / red Unsafe / amber Abstain), confidence bar, demo-config badge (checkpoint/sha/commit/source — Requirements table above). No model/session code in this file.

3. **[M4-CPU, ~15 min]** Write `deployment/demo-local.py`: argparse (`--source <path>` default a real `data/test/*.mp4` clip, `--webcam` DROPPABLE opt-in with the OOD disclosure banner from Key Insight 4, `--record <out.mp4>` to also write the live-annotated stream to disk for reuse as backup footage); main loop wiring `demo-inference-pipeline` + `demo-render-overlay`; `cv2.imshow` window + `q`-to-quit.
   ```bash
   .venv/bin/python3 deployment/demo-local.py --source data/test/<a_real_clip>.mp4
   ```

4. **[M4-CPU, ~15 min]** Correctness spot-check (not a test touch — see Key Insight 6): compare 2-3 live-assembled token rows for a known test clip against the corresponding rows in `sweep_results/relation_tokens/tokens_bin8_test.pt`.
   ```bash
   .venv/bin/python3 -c "
   import torch
   d = torch.load('sweep_results/relation_tokens/tokens_bin8_test.pt')
   # print d['tokens'][clip_idx, frame_idx] and diff against the live pipeline's own printed row
   "
   grep -L "run_kfold_relation\|run_relation_transformer\|train-binary-head-checkpoint" deployment/demo-*.py   # confirm zero training imports
   ```
   Fix any encoding mismatch (crop padding, geometry call ordering) before proceeding — a silent skew here would make the live demo disagree with the frozen model's real behavior.

5. **[M4-CPU, ~15 min, DROPPABLE]** Write `deployment/demo-calibrate-threshold.py`: read val-only OOF probabilities (never `tokens_bin8_test.pt`), print the Safe/Unsafe/Abstain split at the current `(0.45, 0.55)` band and at 1-2 alternative bands, recommend keep-or-adjust. Val/OOF only, per plan.md's test-touch rule — this script must refuse to open any `*_test.pt` file (assert path check).

6. **[M4-CPU, ~20 min, MANDATORY — target complete by 2026-09-25 for buffer before the 10-05 freeze]** Record the Tier-1 fallback: screen-capture `demo-local.py` running end-to-end on the chosen real clip.
   ```bash
   # macOS screen recording via QuickTime Player (File > New Screen Recording) is the simplest
   # reliable path; ffmpeg avfoundation alternative if scripting the capture:
   ffmpeg -f avfoundation -i "1:none" -t 90 -y deployment/demo-fallback-recording-raw.mov
   ffmpeg -i deployment/demo-fallback-recording-raw.mov -c:v libx264 -crf 20 -pix_fmt yuv420p \
     deployment/demo-fallback-recording.mp4
   rm deployment/demo-fallback-recording-raw.mov
   ```
   This satisfies **Gate G7** on its own — no Jetson dependency.

7. **[Jetson, ~20 min, DROPPABLE — Tier-2, only if phase-07 device time allows]**
   > **NOTIFY USER BEFORE RUNNING** — ask the user to confirm the Orin Nano Super is free and to physically position it/a camera for a short recorded run, piggybacking on a phase-07 session.
   ```bash
   python3 deployment/demo-local.py --source data/test/<clip>.mp4 --record deployment/demo-fallback-recording-jetson.mp4
   ```
   Uses the same fp32 ONNX Runtime path on-device (no TRT engine required for an 18k-param head); purely a bonus device-native recording, never required for G7.

8. **[M4-CPU, ~15 min each, calendar 2026-10-06 and 2026-10-08 — "demo rehearsal ×2", no new experiments]** Run the live demo end-to-end twice in front of a practice audience (or self); log timing, any crash/glitch, and confirm the fallback `.mp4` plays cleanly as a backup in `deployment/demo-rehearsal-log.md`.

9. **[M4-CPU, ~10 min, calendar 2026-10-10 — final pre-defence check]** Re-verify: fallback `.mp4` still plays; `demo-local.py` still runs against the unchanged frozen E3 checkpoint (`sha256sum` matches `MANIFEST.md`); grep the demo file set and any prepared slides/narration notes for stray numbers not present in the frozen `plans/2026-09-19-*/number-sheet.md` (Lane D deliverable, phase 09).

## Todo List
- [ ] Step 0: pre-flight asset check
- [ ] Step 1: `demo-inference-pipeline.py` written, reuses `BRT`/`benchmark-p3-utils` functions
- [ ] Step 2: `demo-render-overlay.py` written, no model code inside
- [ ] Step 3: `demo-local.py` written, runs on a real `data/test` clip
- [ ] Step 4: token-assembly correctness spot-check passed, zero training imports confirmed
- [ ] Step 5 (droppable): `demo-calibrate-threshold.py` written, val/OOF-only guard in place
- [ ] Step 6: **MANDATORY** fallback `demo-fallback-recording.mp4` recorded (Gate G7) — target by 09-25
- [ ] Step 7 (droppable, NOTIFY): Tier-2 Jetson-recorded fallback
- [ ] Step 8: rehearsal #1 (10-06) and #2 (10-08) logged
- [ ] Step 9: final pre-defence check (10-10) — fallback plays, checkpoint unchanged, no stray numbers

## Success Criteria
- `deployment/demo-fallback-recording.mp4` exists, plays standalone, ≥60s, shows a successful end-to-end run — **Gate G7 met, independent of Jetson scheduling.**
- `demo-local.py` runs live on a real, in-distribution `data/test/*.mp4` clip and visibly renders the abstain band as a distinct state, not merged into Safe/Unsafe.
- The demo wires E3 only; E0/legacy graphs are not imported or toggled anywhere in the demo file set.
- No file under `deployment/demo-*.py` contains a hardcoded accuracy/BA/F1 number; the only numbers rendered are per-run live model output plus the demo-config badge fields.
- Zero test touches spent (`reports/test-touch-log.md` re-checked after this phase — still 2 pre-registered / 0 spent).
- All four new `.py` files are kebab-case and under 200 lines.
- Two rehearsals logged before 10-10 with no unresolved playback/crash issue on the fallback path.

## Risk Assessment
| Risk | Mitigation |
|---|---|
| Live demo crashes or hangs during the actual defence | Mandatory Tier-1 fallback `.mp4` (Gate G7), rehearsed switch-over path documented in the rehearsal log |
| Webcam/OOD mode looks broken (zero-filled zone features degrade the decision) | Default source is a real dataset clip with known zones; webcam stays DROPPABLE with an explicit on-screen OOD disclosure if used at all |
| A committee member asks for/sees a number not yet frozen | Code-level guard: no accuracy figures anywhere in demo code; narration/slides gated on phase 09's number-sheet freeze (10-05), checked again in step 9 |
| Abstain band mistuned, looks like a bug rather than a feature | DROPPABLE step 5 calibration check against val/OOF before rehearsal; if skipped, keep the as-shipped `(0.45,0.55)` band and disclose it is unassessed |
| Jetson unavailable for Tier-2 | Tier-2 is explicitly droppable; G7 is already satisfied by Tier-1 alone |
| Token-assembly bug makes live demo disagree with the real frozen model | Step 4 correctness spot-check against precomputed test tokens before any rehearsal |

## Security Considerations
- Demo runs entirely local, no network calls, no telemetry — consistent with the "zero network dependency" design goal (a procedural point, not a numeric claim, so it survives the researcher-03 fact-check downgrade).
- No credentials, SSH keys, or device IPs in any demo file; the optional Jetson step reuses phase-07's `jetson` host-alias convention.
- Video sources are existing dataset clips already used elsewhere in the pipeline (`data/test/**`) — no new data collection, no new consent/privacy surface introduced by this phase.
- `deployment/demo-fallback-recording*.mp4` are binary assets; confirm they are not swept up by any `deployment/*.onnx`/`deployment/artifacts/`-style `.gitignore` rule meant for model weights (they should NOT be ignored — the fallback video is a required deliverable, not an artifact to exclude).

## Next Steps
- Hand `deployment/demo-fallback-recording.mp4` + rehearsal log to Lane D for defence-package assembly (phase 09/10).
- Coordinate Tier-2 scheduling with phase 07's Jetson session calendar (09-22→25 or 09-30→10-03) — do not request a separate device session solely for this phase.
- Feed the final "what numbers may appear on slides" decision to Lane D once `number-sheet.md` freezes (10-05); this phase's narration notes stay in draft/bracketed form until then.

**Unresolved questions**
1. Primary demo source: a fixed pre-selected `data/test` clip (safer, reproducible) vs. a clip chosen live at defence time (more "live," more risk)? Recommend fixing the clip in advance during rehearsal and reusing the same one at defence.
2. Is live webcam mode worth building at all given the zone-feature OOD caveat (Key Insight 4), or should it be cut entirely and the phrase "live demo" scoped to mean "live inference on a real video file" throughout? Recommend cutting webcam mode unless a committee member specifically expects a physical live camera.
3. Confirm `ffmpeg`/QuickTime screen-recording permissions are already granted on the presenting Mac — untested this session, verify before 09-25 so step 6 isn't blocked by an OS permission prompt on the day.
4. Should the demo-config badge render on-screen (visible to the committee) or only log to console/file? Rendering it on-screen doubles as an anti-fabrication disclosure; logging only is less visually cluttered. Recommend on-screen, small, corner-anchored.
