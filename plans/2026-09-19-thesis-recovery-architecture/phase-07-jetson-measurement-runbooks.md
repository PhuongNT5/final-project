---
phase: 07
title: "Jetson measurement runbooks"
lane: C
effort: 10h
gate: "G6 metrics captured"
status: pending
---
# Phase 07 — Jetson Measurement Runbooks

## Context Links
- Master plan: [plan.md](plan.md) — lanes, calendar, gates, test-touch rules.
- Method source, **UNVERIFIED — citation-checked, do not trust its illustrative numbers**: `research/researcher-03-demo-and-edge-deployment.md` (§2-4, §7 build plan; procedural steps reused, but its §5.2 example latency/power/cost table is explicitly illustrative-not-measured per the fact-check below — never copy those numbers).
- Fact-check: `reports/critique-2026-09-19-research-factcheck.md` — confirms researcher-03's §5.2 table ("54%/1.8ms/7W" etc.) is a labeled placeholder, not measured data; flags researcher-01/02 citation problems (not directly used in this phase).
- **Governing accuracy-lane finding (repo-measured, 0 test touches, supersedes researcher-03 on architecture choice):** `reports/finding-2026-09-19-lco-failure-is-camera-label-confounding.md` — E3 DeepSets (18,044 params) is the **frozen deployment model**, decided on accuracy grounds (best LCO encoder, both protocols) and simultaneously the smallest/edge-friendliest. Encoder architecture search is cut project-wide.
- Capability ground truth: `reports/audit-2026-09-19-repo-capability-and-feasibility.md` (§6 deployment assets, §8 blockers, §9 test-touch ledger).
- Existing runbook (do not duplicate, extend): `deployment/JETSON-RUNBOOK-2026-09-19.md`.
- Export provenance: `deployment/export/MANIFEST.md`.
- Companion phase: [phase-08-live-demo-and-fallback-recording.md](phase-08-live-demo-and-fallback-recording.md) (shares Jetson device time).
- Depends on: phase-04 (ONNX export, done). Phase-05's G4 encoder question is **resolved before this phase runs** — E3 is frozen for both accuracy and edge reasons, no divergence risk.

## Overview
- **Priority:** P2 of 3 (accuracy is P1, edge is P2, robustness P3). Do not let this phase pull device/compute time from phase 02/05/06.
- **Status:** ONNX graphs exist (`p3-detector-fp32.onnx`, `p3-appearance-fp32.onnx`, `p3-relation-head-bin-e3-fp32.onnx` — **E3 is the sole frozen deployment target**, decided on accuracy grounds; `-bin-e0-`/legacy 4-class are characterization-only, not candidates). Zero Jetson-side artifacts dated 2026-09-19 exist yet — this session produces the first-ever on-device P3 numbers.
- **Brief:** run the existing `JETSON-RUNBOOK-2026-09-19.md` twice (Jetson #1 build+baseline, Jetson #2 confirmatory) against a physical Orin Nano Super, capture p50/p95 latency, FPS, per-stage breakdown, power/energy, 30-min thermal soak, FP32/FP16 parity, max concurrent streams — every number paired with a measured-configuration table (power mode, clock lock, precision, checkpoint arch+seed, git commit).

## Key Insights
1. **P3 has never been measured on real Jetson hardware.** Any "P1-baseline ~2.8ms/frame" or "P3 detector 10-20x slower (~27-55ms/frame)" figures floating in prior docs are Mac/ORT-CPU extrapolations from researcher-03, **[UNVERIFIED]** — the fact-check report confirms researcher-03's own §5.2 numeric table is explicitly illustrative, not measured. This phase's entire point is to produce the first real number; do not cite any pre-session P3-on-Jetson figure as fact anywhere in the results docs.
2. **E3 DeepSets is now the single frozen deployment target — architecture search is cut project-wide.** Per `reports/finding-2026-09-19-lco-failure-is-camera-label-confounding.md` (repo-measured OOF recomputation, zero test touches): E3 (18,044 params) is simultaneously the best LCO encoder under both the full and panel-disclosed protocols and the smallest, so accuracy and edge pillars agree — take the win. Do **not** run an E0-vs-E1-vs-E2-vs-E3 comparison framed as a model-selection question on this phase; that question is closed. A one-off cross-encoder latency table is allowed only as a DROPPABLE edge-characterization figure (step 3b), explicitly labeled "deployment characterization, not model selection."
3. **The E0 batch-baking defect no longer blocks the headline path.** `p3-relation-head-bin-e0-fp32.onnx` (RelationTransformer/E0) still bakes its ONNX batch dim to 1 under torch 2.11's dynamo exporter (`nn.TransformerEncoder` + `.view(B,...)` reshape) despite `dynamic_axes` being passed — reproduces on the untouched legacy 4-class graph too, confirmed pre-existing. Since E3 (genuinely dynamic-batch, verified batch=4 ORT call) is now the frozen model, this defect is a **characterized limitation of the non-frozen fallback arch**, not a blocker. `jetson-build-trt-engine.py`'s optimization profile pins `min=opt=max=1` for every dynamic dim regardless, so neither graph is ever invoked above batch=1 in practice.
4. **INT8 for P3 is out of scope by default** — no P3 calibration data exists (`deployment/artifacts/calib-jetson/` only has P1 `.npy` files). FP32 (host reference) + FP16 (device) is the measured pair for E3; INT8 is a droppable stretch (step 14) only if session #2 has slack.
5. **Zero test touches are spent by anything in this phase.** Latency uses dummy/repeated tensors or unlabeled clips; parity checks decision agreement (`--rule argmax`) between precisions, not against ground truth; multi-stream throughput reads video files but never scores against labels. Do not let any script in this phase open `tokens_bin8_test.pt` labels.
6. **Headline accuracy numbers that may appear alongside edge results, always with protocol label, never as an edge-measured claim:** E3 full LCO-691 BA **69.6%**, panel-disclosed variant **76.6%** (both from the finding report's OOF recomputation, 0 test touches). Never quote a number not yet in the frozen number sheet (phase 09) as if final.
7. **`benchmark-p3-onnx.py`/`benchmark-p3-sweep.py` (host-side, M4) are still hardcoded to the legacy 4-class `p3-relation-head-fp32.onnx`** (`benchmark-p3-onnx.py:54`) — confirmed this session by direct grep, not yet re-checked/pointed at the binary heads. Needed for the host-baseline row of the Pareto table.
8. **`jetson-full-measurement.py`/`jetson-measure-latency.py` auto-discover engines by glob** (`deployment/export/jetson/*.engine`) — no MODELS-list edit needed for P3 coverage there. The older `deployment/scripts/jetson-benchmark.py` has a hardcoded P1-only `MODELS` list and duplicates functionality already covered by `jetson-full-measurement.py`; per YAGNI, leave it untouched rather than extending a superseded script.
9. **The Jetson Orin Nano Super is a shared physical device, available repeatedly but not continuously.** Every on-device step below is a stop-and-confirm block — the agent must not run these unattended.

## Requirements
**Functional**
- Capture, per graph — **detector, appearance, relation-head bin-e3 (required, frozen target)**; relation-head bin-e0 + legacy 4-class **only if step 3b run (DROPPABLE, characterization-only)**: model size, latency p50/p95/p99, FPS.
- Capture end-to-end naive-sum latency, explicitly labeled "upper-bound proxy, not finalized pipeline order" (per runbook §4).
- Capture power (tegrastats), energy/inference (J), FPS/Watt for at least one representative stage.
- Capture 30-min thermal soak on the heaviest stage (detector), natural clock scaling, `possible_throttling` flag + window-level p50 drift.
- Capture max concurrent streams (1/2/4/8) with drop-rate, using real `data/test/*/*.mp4` sources.
- Capture FP16-vs-FP32 decision-agreement parity (`--rule argmax`) for every stage; bin-e3 is required, bin-e0 only if step 3b was run.
- Record `nvpmodel -q` + `jetson_clocks --show` before and after every session (`--power-state`).

**Non-functional**
- No fabricated numbers: every hardware-only JSON field must be a real measurement or `null` + `skipped_reason` (enforced by `tests/test_jetson_measure_dry_run_guards.py`).
- Every reported number ships with its measured-configuration row (below) — a bare number with no config context is not acceptable for the thesis.
- Reproducible: pin the git commit hash and checkpoint sha256 (from `MANIFEST.md`) alongside every result file.

**Measured-configuration table (attach to every results doc/section)**
| field | value |
|---|---|
| Date / session # | |
| Device | Orin Nano 8GB, JetPack ver, TensorRT ver |
| Power mode (`nvpmodel -q`) | |
| `jetson_clocks` state | locked / restored |
| Precision | fp32 / fp16 / int8 |
| Graph + checkpoint arch/seed | e.g. E3, seed 0, val_macro 89.15% |
| Git commit | |
| Test touches spent | 0 (always) |

## Architecture
```
video frame -> [p3-detector-fp32 TRT] -> boxes/scores
            -> [p3-appearance-fp32 TRT] (per box crop) -> 64-d embeddings
            -> tokenizer (T=16, N_MAX=8, 74-dim) -> tokens + mask
            -> [p3-relation-head-bin-e3 TRT, batch=1] -> logits(2) -> Safe/Unsafe/Abstain
```
Each stage is a separate TensorRT engine (`deployment/export/jetson/<stage>-<precision>.engine`), one optimization profile per graph with every dynamic dim pinned `min=opt=max=1`. Multi-stream mode round-robins N reader threads into one inference thread (GIL released during `execute_async`); measurement harness (`jetson-full-measurement.py`) discovers engines by filename glob, no per-model registration needed.

## Related Code Files
**Modify** (Lane C scope: `deployment/**`, `tools/jetson_benchmark.py` only)
- `deployment/scripts/benchmark-p3-onnx.py` — add `p3-relation-head-bin-e3-fp32.onnx` as the required host (M4-CPU) baseline target (currently hardcoded to the legacy 4-class head only, line 54); add `-bin-e0-`/legacy only if step 3b's DROPPABLE characterization table is attempted.
- `deployment/scripts/benchmark-p3-sweep.py` — same, if it wraps `benchmark-p3-onnx.py`'s model list.
- `deployment/JETSON-RUNBOOK-2026-09-19.md` — append a "Session log" section per run (date, operator note, deviations) rather than editing the procedure itself.

**Create**
- `deployment/jetson-session-01-results.md` — session #1 measured-config tables + result summary (links to JSON in `deployment/artifacts/jetson-measurements/`, which is gitignored).
- `deployment/jetson-session-02-results.md` — session #2 confirmatory results, diffed against session #1.
- (stretch, DROPPABLE) extend `deployment/scripts/jetson-prep-int8-calib.py` with a `p3-detector` calibration path, if step 14 is attempted.

**Delete:** none.

## Implementation Steps

0. **[M4-CPU, ~10 min]** Re-run the Mac dry-run checklist already in `JETSON-RUNBOOK-2026-09-19.md` (§"18/9 Mac dry-run checklist") if any `deployment/scripts/*.py` changed since 2026-09-19:
   ```bash
   .venv/bin/python3 deployment/scripts/jetson-full-measurement.py --dry-run --all
   .venv/bin/python3 tests/test_jetson_measure_dry_run_guards.py
   ```
   Confirm `deployment/export/MANIFEST.md` checksums still match `sha256sum deployment/export/p3-*.onnx`.

1. **[Jetson, ~10 min, calendar: 09-22→25 "Jetson #1"]**
   > **NOTIFY USER BEFORE RUNNING** — ask the user to power on the Orin Nano Super, connect it to the same network/USB as the Mac, and confirm it is free for a ~2h session.
   ```bash
   rsync -avz deployment/scripts/ jetson:~/project/deployment/scripts/
   rsync -avz deployment/export/p3-*.onnx deployment/export/MANIFEST.md jetson:~/project/deployment/export/
   ssh jetson
   source ~/.venv-jetson/bin/activate
   python3 -c "import tensorrt, pycuda.autoinit; print('ok')"
   nvpmodel -q
   sudo nvpmodel -m 0
   sudo jetson_clocks
   mkdir -p deployment/export/jetson deployment/artifacts/jetson-measurements
   python3 deployment/scripts/jetson-full-measurement.py --dry-run --all
   ```

2. **[Jetson, ~15-25 min]**
   > **NOTIFY USER BEFORE RUNNING** — confirm device is still connected before the build step (long-running, avoid mid-build disconnects).
   Skip if `deployment/export/jetson/*.engine` for P1 already exist (built 2026-06-03):
   ```bash
   for m in backbone direct4-s314 scenario-s7 pair0-s7 pair1-s7 pair2-s7 pair3-s7; do
     python3 deployment/scripts/jetson-build-trt-engine.py --model $m --mode fp16
   done
   ```

3. **[Jetson, ~15-20 min]** Build the frozen-model engines (E3 is the required target):
   > **NOTIFY USER BEFORE RUNNING**
   ```bash
   python3 deployment/scripts/jetson-build-trt-engine.py --model p3-detector --onnx deployment/export/p3-detector-fp32.onnx --mode fp16
   python3 deployment/scripts/jetson-build-trt-engine.py --model p3-appearance --onnx deployment/export/p3-appearance-fp32.onnx --mode fp16
   python3 deployment/scripts/jetson-build-trt-engine.py --model p3-relation-head-bin-e3 --onnx deployment/export/p3-relation-head-bin-e3-fp32.onnx --mode fp16
   ```
   Fallback if TRT parse fails on the dynamic-batch ONNX ops: `trtexec --onnx=... --saveEngine=... --fp16` per runbook §2.

3b. **[Jetson, ~10 min, DROPPABLE — edge-characterization only, not model selection]** Optionally build the legacy 4-class head and E0 fallback for a one-off cross-encoder latency table (explicitly labeled "deployment characterization," never framed as re-opening the E0-vs-E3 accuracy question, which is closed):
   > **NOTIFY USER BEFORE RUNNING**
   ```bash
   python3 deployment/scripts/jetson-build-trt-engine.py --model p3-relation-head --onnx deployment/export/p3-relation-head-fp32.onnx --mode fp16
   python3 deployment/scripts/jetson-build-trt-engine.py --model p3-relation-head-bin-e0 --onnx deployment/export/p3-relation-head-bin-e0-fp32.onnx --mode fp16
   ```

4. **[Jetson, instant]**
   > **NOTIFY USER BEFORE RUNNING** (device must stay on, no code change needed but confirm session continuity)
   ```bash
   python3 deployment/scripts/jetson-full-measurement.py --model-size
   ```

5. **[Jetson, ~2-5 min]** Latency, MAXN + `jetson_clocks` locked (record measured-config table row):
   > **NOTIFY USER BEFORE RUNNING**
   ```bash
   python3 deployment/scripts/jetson-full-measurement.py --latency --mode fp16 --warmup 20 --runs 200
   ```

6. **[Jetson, ~2-5 min]** FP16 vs FP32 parity, decision-agreement only (no labels read):
   > **NOTIFY USER BEFORE RUNNING**
   ```bash
   python3 deployment/scripts/jetson-full-measurement.py --parity --rule argmax
   ```
   Expect `decision_agreement_rate >= 0.98` per stage; this is a quantization-robustness check on real trained weights (E3 val_macro 89.15%, E0 90.92%), not a test-accuracy claim.

7. **[Jetson, ~10-15 min, DROPPABLE if data transfer is slow]** Multi-stream saturation (needs `data/test/*/*.mp4` on device — `du -sh data/test` on Mac first per runbook §E):
   > **NOTIFY USER BEFORE RUNNING**
   ```bash
   python3 deployment/scripts/jetson-full-measurement.py --streams 1 2 4 8 \
     --mode fp16 --sources data/test/*/*.mp4 --max-frames 300 --fps 24 \
     --class-labels Safe,Unsafe
   ```

8. **[Jetson, ~2 min/stage, DROPPABLE — cut before soak if time short]** Energy per stage:
   > **NOTIFY USER BEFORE RUNNING**
   ```bash
   python3 deployment/scripts/jetson-full-measurement.py --energy backbone --mode fp16 --energy-sec 30
   # repeat with --energy p3-detector, --energy p3-relation-head-bin-e3, etc.
   ```

9. **[Jetson, ~35 min incl. cooldown, OVERNIGHT-adjacent — run last, do not schedule right before a hard stop]** Thermal soak, natural clocks:
   > **NOTIFY USER BEFORE RUNNING** — confirm the device can sit undisturbed on natural cooling for 35 min.
   ```bash
   sudo jetson_clocks --restore
   python3 deployment/scripts/jetson-full-measurement.py --thermal-soak backbone \
     --mode fp16 --soak-min 30 --soak-window-sec 60
   ```
   If P3's heavier detector throttles well before 20 min, stop early and report the throttle window as a real finding (runbook §C), not a failure.

10. **[Jetson, instant]** Memory + power-state snapshot (before AND after step 9):
    > **NOTIFY USER BEFORE RUNNING**
    ```bash
    python3 deployment/scripts/jetson-full-measurement.py --memory --mode fp16
    python3 deployment/scripts/jetson-full-measurement.py --power-state
    ```

11. **[M4-CPU, ~15 min]** Host-baseline comparison for the Pareto table — modify `benchmark-p3-onnx.py`/`benchmark-p3-sweep.py` to add the bin-e3 target, then run:
    ```bash
    .venv/bin/python3 deployment/scripts/benchmark-p3-onnx.py --heads bin-e3
    # DROPPABLE, only if step 3b's characterization table is wanted:
    .venv/bin/python3 deployment/scripts/benchmark-p3-onnx.py --heads legacy,bin-e0
    ```

12. **[M4-CPU, ~30 min]** Write `deployment/jetson-session-01-results.md`: measured-config table + copied numbers from every JSON in `deployment/artifacts/jetson-measurements/` for this date + explicit note on the E0/E3 batch-bake defect and INT8-P3-out-of-scope.

13. **[Jetson, calendar: 09-30→10-03 "Jetson #2 confirmatory"]** Repeat steps 1, 5, 6, 9, 10 against the same frozen E3 checkpoint from session #1 (no re-export expected — E3 is frozen, not subject to further change; re-verify `MANIFEST.md` sha256 matches before assuming so). Also complete any DROPPABLE items skipped in session #1 (7, 8) if time allows.
    > **NOTIFY USER BEFORE RUNNING** for every command block, same as steps 1-10.
    Write `deployment/jetson-session-02-results.md`, diffed against session #1 (flag any >15% drift as a finding, not noise).

14. **[Jetson, DROPPABLE stretch, only if session #2 has slack]** P3 INT8: extend `jetson-prep-int8-calib.py` with a `p3-detector` calibration set (sample ~100 frames from `data/train/`), build INT8 engine, add to parity/latency runs. Cut first under any time pressure — edge is P2 of 3, INT8 is not required for G6.

## Todo List
- [ ] Step 0: Mac dry-run + checksum re-verify
- [ ] Step 1: Jetson #1 pre-flight (NOTIFY)
- [ ] Step 2: P1 engines built or confirmed present (NOTIFY)
- [ ] Step 3: P3 engines built — detector, appearance, bin-e3 (required, NOTIFY)
- [ ] Step 3b (DROPPABLE, characterization-only): legacy-4class + bin-e0 engines built
- [ ] Step 4: model-size captured (NOTIFY)
- [ ] Step 5: latency p50/p95/p99 captured, MAXN+locked config recorded (NOTIFY)
- [ ] Step 6: FP16/FP32 parity captured (NOTIFY)
- [ ] Step 7 (droppable): multi-stream 1/2/4/8 captured (NOTIFY)
- [ ] Step 8 (droppable): energy per stage captured (NOTIFY)
- [ ] Step 9: 30-min thermal soak captured, natural clocks (NOTIFY)
- [ ] Step 10: memory + power-state before/after (NOTIFY)
- [ ] Step 11: host-baseline (M4) benchmark updated + run
- [ ] Step 12: `jetson-session-01-results.md` written
- [ ] Step 13: Jetson #2 confirmatory session complete (NOTIFY), `jetson-session-02-results.md` written
- [ ] Step 14 (droppable stretch): P3 INT8 calib + engine + parity

## Success Criteria
- Every target metric in the task brief (p50/p95, FPS, per-stage breakdown, power, energy/inference, 30-min thermal soak, FP32/FP16 parity, max concurrent streams) has a real, non-fabricated value or an explicit `skipped_reason`.
- Every reported number has an accompanying measured-configuration table row (device, power mode, clock state, precision, checkpoint arch/seed, git commit).
- `nvpmodel -m 0` + `jetson_clocks` documented as the measured configuration for latency/parity; natural-clock state documented for the soak.
- E0's batch-bake defect (characterized limitation of the non-frozen fallback, does not affect E3) and the `min=opt=max=1` TRT profile limitation (affects all graphs incl. E3) are stated in `jetson-session-01-results.md`, not silently omitted.
- Zero test touches spent (confirmed by re-reading `reports/test-touch-log.md` after the session — still shows 2 pre-registered / 0 spent).
- Session #2 numbers exist and are diffed against session #1.
- Gate **G6** ("metrics captured") signed off before phase 06's frozen-suite writing needs the edge numbers (per calendar, by 10-03).

## Risk Assessment
| Risk | Mitigation |
|---|---|
| Device unavailable/borrowed elsewhere when a session is scheduled | Both sessions have calendar slack (09-22→25, 09-30→10-03); minimum-viable-set (runbook §13: pre-flight+model-size+latency+parity) is the fallback if a session gets cut to <30 min |
| TensorRT parse failure on dynamic-batch ONNX ops | `trtexec` direct fallback (step 3), documented in runbook §2 |
| Thermal soak runs long / device overheats | Stop early on sustained throttle past ~85C junction (runbook §C), report the early-throttle window as a real finding |
| Session #2 numbers drift from session #1 despite same frozen E3 checkpoint (thermal/env variance) | Explicit checkpoint/git-commit pinning + sha256 re-check in the config table; flag >15% drift as a finding, not noise |
| `data/test/*/*.mp4` transfer to device too slow, eats into session time | `du -sh data/test` on Mac before the session (already in runbook checklist); multi-stream step (7) is droppable |
| Fabricated/placeholder numbers slip into thesis by accident | `tests/test_jetson_measure_dry_run_guards.py` enforces null+skipped_reason contract; re-run it as part of step 0 |

## Security Considerations
- No SSH keys, device IPs, or credentials committed to any file created in this phase — `rsync -avz ... jetson:~/project/...` in docs uses a placeholder host alias (`jetson`), matching existing `JETSON-DEPLOYMENT.md` convention.
- Measurement JSONs and `.engine` files land under `deployment/artifacts/` and `deployment/export/jetson/`, both covered by `.gitignore` (`deployment/artifacts/`, `deployment/*.onnx` is NOT globally ignored but the jetson engine dir path is under `artifacts/`-adjacent conventions — verify `deployment/export/jetson/*.engine` is excluded before any `git add`, since `.gitignore` currently lists `deployment/*.onnx` but not explicitly `*.engine`).
- No new network services opened on the device for this phase (all measurement is local-to-device, transferred back via rsync/scp, not exposed).

## Next Steps
- Hand off `deployment/jetson-session-0{1,2}-results.md` + underlying JSON to Lane D for `plans/2026-09-19-*/number-sheet.md` (per plan.md's "C→D via measurement JSON" rule).
- Feeds phase 06's frozen-suite writing (edge numbers needed by the 10-05 freeze) and phase 09 (thesis integration).
- Shares Jetson device time with phase 08's optional Tier-2 fallback recording — coordinate scheduling, Tier-2 is droppable if it conflicts.

**Unresolved questions**
1. ~~Does the edge-freeze architecture need to match phase 05's G4 accuracy winner?~~ **Resolved** — E3 is frozen for both accuracy and edge reasons (`finding-2026-09-19-lco-failure-is-camera-label-confounding.md`); no divergence risk remains.
2. Should thermal-soak duration be reduced from 30 min for the heavier P3 detector if it throttles early, or is a shorter documented soak acceptable for the defence package? Decide live during session #1 based on the first throttle-check window.
3. `deployment/scripts/benchmark-p3-onnx.py`/`benchmark-p3-sweep.py` — confirmed this session still hardcoded to the legacy 4-class head; exact effort to add the binary-e3 head not yet scoped (assumed ~15 min in step 11, unverified until attempted).
4. Is `deployment/export/jetson/*.engine` actually covered by `.gitignore`, or does it need an explicit new ignore rule? Not verified this session — check before the first `git status` after engine builds land locally (engines won't exist on the Mac since builds run on-device, but confirm before any rsync-back-to-Mac + commit workflow).
5. Is the step-3b cross-encoder latency table worth the device time at all given E3's model-selection question is closed? Default to skipping it unless session #2 has clear slack — it is purely illustrative for a "smallest model also wins" framing line, not required by G6.
