# Jetson Measurement Session Runbook — 2026-09-19 morning

> **RESCOPED 2026-09-20 — the 09-19 session did not run (no artifact on disk is dated after
> 2026-09-19). This runbook is now the MANDATORY session under decision D4, scheduled 2026-09-22.**
> Three scope changes from the author's 2026-09-20 sign-off:
> 1. **Build and measure the E3 binary head ONLY.** D3 freezes E3; do NOT build
>    `p3-relation-head-bin-e0-fp32.onnx`, even with spare time. Spend spare time on Steps 6-7 instead.
> 2. **Step 4 (per-stage latency) is the session's headline deliverable.** It converts the
>    "P3 misses the 3.0 decisions/s cadence" conclusion from a Mac-derived projection
>    (1.13/1.64/1.71 dec/s, transfer factors 2.15x/3.11x/3.25x) into a real device measurement.
>    **DONE 2026-09-29: ~~3.02-3.04 dec/s at median crop load, 2.60-2.72 above it~~ — P3 sits *on*
>    the 3.0 dec/s boundary, not ~2x below it. The projection is refuted; see Step 4.**
>    **Figures withdrawn 2026-10-01** (engine provenance): 3.226 dec/s pinned / 2.733 DVFS at
>    median crop load. The refutation stands; the numbers are in Step 4's superseded banner.
>    If the session is cut short, Steps 1-4 plus Step 11 are the defensible minimum.
> 3. **P1 is the deployed system (D2).** Its June-03/12 numbers stand; re-verification with the
>    fixed `TRTRunner` is a nice-to-have, not a blocker. P3 is measured as the accuracy-research
>    pipeline whose cost curve is itself a reported result.

**Fixed device session — originally 2026-09-19 morning, NOT RUN; now 2026-09-22.
This runbook must match the code exactly.** Device: Orin
Nano 8GB (`phuongnguyen-desktop`), JetPack 6.0, TensorRT 10.3.0 (per
`JETSON-DEPLOYMENT.md`, measured 2026-06-03 for P1). This session adds P3
measurements (never run on Jetson before) and re-verifies P1 with the fixed
`TRTRunner` (multi-input binding bug fixed 2026-09-16, see report).

**Read this first**: P3 has ONNX exports (`deployment/export/p3-*.onnx`) but
**no TensorRT engines exist yet** — `deployment/export/jetson/` doesn't exist
on this checkout. You must build P3 engines on-device before measuring them
(Step 3). INT8 calib data exists only for P1 stages
(`deployment/artifacts/calib-jetson/backbone-batch-*.npy` etc); there is no
P3 calib data — INT8 for P3 is out of scope for 9/19, FP32/FP16 only.

**Binary relation head status (updated 2026-09-19, TRAINED checkpoints
landed)**: the OLD 4-class `p3-relation-head-fp32.onnx` (`logits [1,4]`) is
still present, untouched. Two binary graphs exist — the architecture freeze
decision (E3 vs E0) was NOT made by 18/9, so both were built (see freeze
recommendation in
`plans/2026-09-16-binary-safe-unsafe-reframe/reports/laneb-2026-09-17-encoder-impl-and-screening.md`
§6, the original random-init export report,
`plans/2026-09-16-binary-safe-unsafe-reframe/reports/laneb-2026-09-18-binary-onnx-export.md`, and
the 2026-09-19 training/re-export report,
`plans/2026-09-16-binary-safe-unsafe-reframe/reports/laneb-2026-09-19-binary-head-checkpoints.md`):

- `deployment/export/p3-relation-head-bin-e3-fp32.onnx` — E3DeepSets,
  18,044 params, **freeze-candidate (recommended)**. Genuinely dynamic-batch.
  TRAINED: 5 seeds, 60 epochs, `bin8_` TRAIN split (`drop_conflict`),
  val-selected, val_macro=89.15%.
- `deployment/export/p3-relation-head-bin-e0-fp32.onnx` — RelationTransformer,
  298,892 params, statistical-tie **fallback**. Batch=1 only (same
  pre-existing dynamo-exporter reshape limitation as the old 4-class graph —
  harmless, this session only ever builds batch=1 engines anyway). TRAINED:
  same recipe, val_macro=90.92%.

**Critical caveat — READ BEFORE QUOTING ANY NUMBER FROM THESE TWO GRAPHS**:
both checkpoints are trained (2026-09-19, TRAIN split only, zero test
touches — see the 2026-09-19 report above for the code-path proof), so
PyTorch-vs-onnxruntime parity now validates a **real, trained** graph (max
abs diff ~1-3e-6, 100% decision agreement over all 125 bin8 test clips,
batch=1 each — see `deployment/export/MANIFEST.md`). **But no test touch has
been spent on either checkpoint** — the val_macro numbers above are a
training-time selection metric, not a test-set accuracy claim. Do not report
either as a test accuracy number until one of the two pre-registered test
touches (`reports/test-touch-log.md` 2026-09-17 G0) is spent. Latency/
throughput/model-size numbers from these two graphs are real (real weights,
real graph shape); FP16/INT8 parity numbers (Step 5) are now also meaningful
decision-quality checks on a real model, not noise-on-random-logits. Pick ONE
(default to E3, the recommended freeze) for Steps 2/5/6 below; build the
other only if there's spare time and you want an early sanity check of its
engine build.

All commands below run ON THE JETSON unless marked `[MAC]`. Use the Jetson venv:
`source ~/.venv-jetson/bin/activate` (per JETSON-DEPLOYMENT.md; adjust if your
venv path differs).

---

## 18/9 Mac dry-run checklist (do this the evening before, NOT on the Jetson)

Purpose: catch CLI/schema bugs on the Mac where there's no device-time cost.
Run every one of these BEFORE packing up for the 19/9 session:

1. `git status` — confirm no other agent's in-flight edits to
   `deployment/scripts/` or `deployment/export/` (this session's file
   ownership is `deployment/` + `tests/test_jetson_*.py` only; other lanes
   own model/training/baseline code).
2. Full dry-run, must complete in <5s with zero fabricated numbers:
   ```bash
   .venv/bin/python3 deployment/scripts/jetson-full-measurement.py --dry-run --all
   ```
   Expected output: `[SAVED] deployment/artifacts/jetson-measurements/<name>-<timestamp>.json`
   (real filename pattern is `{name}-{YYYYmmddTHHMMSS}.json`, no `DRYRUN`
   marker in the filename — the dry-run-ness only shows up inside the JSON
   as `skipped_reason`, and always as `device_state.dry_run: true`) for each
   of `model-size`, `latency-fp16`, `memory-fp16`, `parity`. Verified
   2026-09-19 on Mac: `--all` covers exactly those four — it does NOT include
   `--power-state` (add the flag explicitly) nor
   `--streams`/`--energy`/`--thermal-soak` (they need explicit stage/source
   args) — dry-run those separately:
   ```bash
   .venv/bin/python3 deployment/scripts/jetson-full-measurement.py --dry-run --power-state
   .venv/bin/python3 deployment/scripts/jetson-full-measurement.py --dry-run \
     --streams 1 2 --sources dummy.mp4 --energy backbone --thermal-soak backbone
   ```
   Mac dry-run outputs from 2026-09-19 were moved to
   `deployment/artifacts/jetson-measurements/mac-dry-run-20260919/` so the
   device session's real JSONs land in a clean directory (the two older
   2026-09-16/17 Mac runs are left in place as-is).
3. Inspect every produced JSON: confirm every hardware-only field is `null`
   with a non-empty `skipped_reason` string (never a number, never an empty
   string) — this is the "no fabrication" contract
   (`tests/test_jetson_measure_dry_run_guards.py` is the automated form of
   this check; run it too: `.venv/bin/python3 tests/test_jetson_measure_dry_run_guards.py`).
4. Run the full test suite (see §7 of the review-fixes report for the exact
   command list) — all must pass before leaving for the session.
5. Confirm `deployment/export/p3-*.onnx` files present and their checksums
   match `deployment/export/MANIFEST.md` (phase-04 landed 2026-09-18 — the
   manifest now exists and covers all 5 P3 graphs: detector, appearance,
   4-class legacy head, binary-e3, binary-e0).
6. `du -sh data/test` on Mac — gauge the `--sources` transfer time for Step 6
   (Triage §E already flags this as commonly skipped in a rushed sync).

---

## 18/9-19/9 prerequisite — binary relation head export (phase-04, landed
## 2026-09-18; trained checkpoints landed 2026-09-19)

Status vs the original spec (`phase-04-pre-jetson-checkpoint-and-export.md`):

- [x] Binary relation head(s) exported — **two** graphs, not one (arch not
      frozen): `p3-relation-head-bin-e3-fp32.onnx` (E3, recommended)
      and `p3-relation-head-bin-e0-fp32.onnx` (E0, fallback). Both 2 inputs
      (`tokens`, `mask`), single `logits (batch,2)` output (main head only —
      sub-class/frame aux heads are training-only, correctly excluded), opset
      18. The old `p3-relation-head-fp32.onnx` 4-class graph was **kept, not
      replaced** (decision not made yet between E3/E0, so nothing should be
      deleted).
- [x] `deployment/export/MANIFEST.md` written and updated 2026-09-19 — every
      P3 export file + checksum + params + i/o shapes + parity numbers +
      checkpoint provenance + what's still unverified.
- [x] **Both graphs re-exported 2026-09-19 from TRAINED checkpoints**
      (`deployment/scripts/train-binary-head-checkpoint.py`, TRAIN split
      only, zero test touches, val-selected — code-path proof + val metrics
      in
      `plans/2026-09-16-binary-safe-unsafe-reframe/reports/laneb-2026-09-19-binary-head-checkpoints.md`).
      E3: 5 seeds, val_macro 89.15%. E0: 5 seeds, val_macro 90.92%.
      PyTorch-vs-onnxruntime parity of the trained graphs: max-abs diff
      1.4e-06 (E3) / 2.6e-06 (E0), 100% decision agreement, all 125 real
      bin8 test clips (batch=1 each, numerics only — no labels read — see
      MANIFEST).
- [ ] **Test-set accuracy — NOT MET, by design.** Val_macro above is a
      training-time selection metric, not a test result. Neither of the two
      pre-registered test touches (`reports/test-touch-log.md` 2026-09-17
      G0) has been spent. Do not report a test accuracy number for either
      binary head from this session.
- [ ] Phase-04's step 7 (full runbook rehearsal against Mac/ORT) — not run;
      out of scope for both the export lane and the 2026-09-19 training lane.

**Consequence for Step 2 onward**: build `p3-relation-head-bin-e3-fp32.onnx`
(the freeze-recommended arch) as the binary engine. **Do NOT build `-bin-e0-`
(RESCOPED 2026-09-20, D3)** — E3 is frozen as the headline encoder, so an E0
engine would measure a model the thesis does not report. Use `--rule argmax` for Step 5 parity
(FP16-vs-FP32 decision agreement) — this is now a meaningful decision-quality
check on a real trained model, not noise-on-random-logits as it would have
been against the 2026-09-18 placeholder graphs. `--rule threshold` is still
not used in this runbook (no calibrated operating threshold has been
selected on val/OOF yet); that remains a follow-up, not a blocker for this
session. Disclose in the thesis: Jetson latency/size numbers for the binary
head are real-weight graph-cost measurements; any test-set accuracy claim
still awaits one of the two pre-registered test touches, not this device
session.

---

## 0-. Connectivity (added 2026-09-24, after a failed session start)

The `jetson` SSH host is pinned to `192.168.100.152` in `~/.ssh/config`, but the
device takes a DHCP lease and **the IP drifts**. On 2026-09-24 the session could
not start: ARP showed `192.168.100.152 (incomplete)` and a full sweep of the /24
found only the router, the Mac, and two unrelated hosts.

Order of checks, cheapest first:

```bash
ping -c2 192.168.100.152                       # configured IP still valid?
arp -a | grep 192.168.100                      # "(incomplete)" = host not answering
for i in $(seq 1 254); do (ping -c1 -W1 192.168.100.$i >/dev/null 2>&1 \
  && echo "alive: 192.168.100.$i") & done; wait # find the drifted lease
ssh jetson 'hostname; uname -m'                # expect phuongnguyen-desktop / aarch64
```

If the IP moved, update `HostName` in `~/.ssh/config` rather than passing the raw
IP — the rsync lines below all resolve through the `jetson` alias. Consider a DHCP
reservation on the router so this does not cost session time again.

Mac-side pre-flight that does NOT need the device (do it before powering on):
all transfer-list files present, and `python3 deployment/scripts/jetson-full-measurement.py
--dry-run --all` exits 0. Verified green 2026-09-24. **Delete the dry-run JSONs it
writes into `deployment/artifacts/jetson-measurements/`** — they carry
`dry_run: true, platform: macOS-*` and must never be mistaken for device numbers.

---

## 0. Pre-flight checklist (~10 min)

- [ ] `[MAC]` Confirm no uncommitted lane conflicts: `git status` — do not
      transfer files mid-edit by another agent.
- [ ] `[MAC]` Sync code + new export/onnx to Jetson (rsync or scp), per the
      file-transfer table in `JETSON-DEPLOYMENT.md`. **The project root on the
      device is `~/my-project`, NOT `~/project`** (corrected 2026-09-29 — the
      rsync lines below used to point at a path that does not exist, which would
      have silently created a stray tree). The device has **no venv**: TensorRT
      10.3.0 and pycuda import fine from the system `python3` (3.10.12), so skip
      any `source ~/.venv-jetson/bin/activate` step — it does not exist either. New files this session
      that must be included: `deployment/scripts/trt-runner-utils.py`,
      `stream-pipeline-utils.py` (new 2026-09-17, shared by
      `jetson-multi-stream-pipeline.py` — import fails at Step 6 without it),
      `jetson-measure-utils.py`, `jetson-measure-latency.py`,
      `jetson-measure-power.py`, `jetson-full-measurement.py`,
      `deployment/export/p3-detector-fp32.onnx`,
      `deployment/export/p3-appearance-fp32.onnx`,
      `deployment/export/p3-relation-head-fp32.onnx` (old 4-class, kept),
      `deployment/export/p3-relation-head-bin-e3-fp32.onnx` (binary,
      recommended — TRAINED weights, val_macro 89.15%, see caveat above),
      `deployment/export/p3-relation-head-bin-e0-fp32.onnx` (binary
      fallback, TRAINED, val_macro 90.92%, optional if time-constrained),
      `deployment/export/MANIFEST.md`.
  ```bash
  rsync -az deployment/scripts/ jetson:~/my-project/deployment/scripts/
  rsync -az --exclude='*bin-e0*' deployment/export/p3-*.onnx deployment/export/MANIFEST.md jetson:~/my-project/deployment/export/
  ```
- [ ] On Jetson: `python3 -c "import tensorrt, pycuda.autoinit; print('ok')"`
      — must print `ok` (system python3, no venv). If it fails, see Triage §A.
- [ ] `nvpmodel -q` — note current power mode. **DO NOT run `sudo nvpmodel -m 0`
      on this device.** Corrected 2026-09-29 against the hardware: this is an Orin
      Nano **Super** on JetPack R36.5.0, whose modes are `0=15W, 1=25W,
      2=MAXN_SUPER`. Mode 0 is the *slowest*, not MAXN. The device idles in mode 2
      already, which is what we want; leave it alone and just record it. Running
      `-m 0` would have measured the headline P3 latency at 15W and made the
      cadence result look worse than the hardware actually is.
      Verify with `sudo nvpmodel -p --verbose | grep POWER_MODEL` before trusting
      any mode number written in this runbook.
- [ ] `sudo jetson_clocks` — lock clocks at max before latency/throughput
      runs (undo before the thermal soak, which specifically wants to see
      natural throttling — see Step 6).
- [ ] `mkdir -p deployment/export/jetson deployment/artifacts/jetson-measurements`
- [ ] Sanity dry-run (should complete in <5s, all `skipped_reason` fields
      populated, zero fabricated numbers — confirms the CLI itself works
      before spending device time):
  ```bash
  python3 deployment/scripts/jetson-full-measurement.py --dry-run --all
  ```

---

## 1. Build P1 engines (if not already on device) — ~15-25 min

P1 engines were built 2026-06-03; if the device still has them at
`deployment/export/jetson/*.engine`, skip to Step 2. Otherwise:

```bash
for m in backbone direct4-s314 scenario-s7 pair0-s7 pair1-s7 pair2-s7 pair3-s7; do
  python3 deployment/scripts/jetson-build-trt-engine.py --model $m --mode fp16
done
```
Good: each produces `deployment/export/jetson/$m-fp16.engine`, no errors.
~2-4 min/model on Orin Nano.

## 2. Build P3 engines (new, first time on Jetson) — ~15-20 min

`jetson-build-trt-engine.py` now takes `--model NAME --onnx PATH` for a
single non-P1 model (added 2026-09-16 specifically for this step; default
no-arg behavior still builds the full P1 `MODELS` list unchanged):
```bash
python3 deployment/scripts/jetson-build-trt-engine.py --model p3-detector --onnx deployment/export/p3-detector-fp32.onnx --mode fp16
python3 deployment/scripts/jetson-build-trt-engine.py --model p3-appearance --onnx deployment/export/p3-appearance-fp32.onnx --mode fp16
python3 deployment/scripts/jetson-build-trt-engine.py --model p3-relation-head --onnx deployment/export/p3-relation-head-fp32.onnx --mode fp16
python3 deployment/scripts/jetson-build-trt-engine.py --model p3-relation-head-bin-e3 --onnx deployment/export/p3-relation-head-bin-e3-fp32.onnx --mode fp16
# E0 build DROPPED 2026-09-20 per D3 (E3 is the frozen headline encoder). Do not run it.
```
Both binary ONNX graphs declare a dynamic `batch` symbolic dim on `tokens`/
`mask`; `build_engine()`'s profile step (name/shape-driven, not arch-aware)
sets `min=opt=max=1` for it either way, so the build call is identical
regardless of which graph is genuinely batch-capable (E3 is; E0 is
batch=1-only — see `deployment/export/MANIFEST.md`). Expect the same console
line as the other builds: `profile: tokens ['batch', 16, 8, 74] -> (1, 16,
8, 74)` (and the same for `mask`).
If this still fails (e.g. dynamic-shape ONNX ops `trt` can't parse), fall
back to `trtexec` directly:
```bash
trtexec --onnx=deployment/export/p3-detector-fp32.onnx \
  --saveEngine=deployment/export/jetson/p3-detector-fp16.engine --fp16
```
repeat per model. Good: `.engine` files appear at
`deployment/export/jetson/p3-*-fp16.engine` (**4 expected** — detector,
appearance, legacy 4-class head, binary-e3; no E0 engine per D3), no
tactic-skip failures. INT8 skipped (no
P3 calib data yet — noted above).

## 3. Model size on disk (~instant)

```bash
python3 deployment/scripts/jetson-full-measurement.py --model-size
```
Good: JSON lists every `.engine` file with size in MB; P1 heads should each
be a few MB, backbone larger. No `skipped_reason`.

## 4. Per-stage latency, p50/p95/p99 (~2-5 min)

```bash
python3 deployment/scripts/jetson-full-measurement.py --latency --mode fp16 --warmup 20 --runs 200
```
Good: every discovered engine reports non-null p50/p95/p99 in ms; p95 not
wildly above p50 (>3x gap suggests thermal/contention noise — rerun).
`end_to_end_naive_sum_ms_p50` is a naive stage-sum, **not** the true decision
latency (pipeline order for the binary+aux head design isn't finalized) —
report it as a rough upper-bound proxy only, say so explicitly in the thesis.

**What this step is for (RESCOPED 2026-09-20).** This is the session's headline
deliverable. Compare the measured decision rate against the **cadence contract:
24 fps at EMIT_EVERY=8 requires 3.0 decisions/s**. The current thesis claim
("P3 misses the cadence") rests on a Mac ORT-CPU measurement (1,825 ms/decision
= 0.55 dec/s) scaled by Mac->Jetson transfer factors 2.149/3.109/3.252, giving a
projected 1.13/1.64/1.71 dec/s. Whatever this step measures REPLACES that
projection in the thesis — including if it comes out faster than projected.
Record the raw JSON path; do not round or re-derive. If the measured rate clears
3.0 dec/s, that is a real finding and D2's framing gets revised, not the number.

**RUN 2026-09-29, 5 repeats — RESULT (revised 22:35; the first write-up of this step was wrong).**

> The first two runs (`...T214556`, `...T221408`) decoded **no** frames and produced
> **3.62 / 3.05 dec/s (p50 / cap crops) — "PASS at every percentile"**. That row is a
> **decode-free control, not a result**, and **must not be quoted**: running CPU decode in the
> same process slows the detector by a systematic **+9.4 %** (11.85 → 12.85 ms/frame), and no
> deployment decodes zero frames.

**SUPERSEDED 2026-10-01 — do not quote the three numbers below.** The 09-29 engines did not
match the device record and the run's clock state was never captured. Re-measured on rebuilt
engines in both clock conditions: **310.00 ms = 3.226 dec/s (+7.5 %) at median crop load with
`sudo jetson_clocks` applied**, versus **365.88 ms = 2.733 dec/s (−8.9 %, FAIL) under default
DVFS**; p95/p99/cap FAIL in both conditions (pinned 2.883 / 2.810 / 2.760). Authority:
`plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-01-engine-provenance-invalidates-0929-latency.md`.

> ~~**Deployment-representative (n=3, decode measured in-process;
> `decision-latency-fp16-20260929T{221836,223133,223401}.json`):**
> 331.19 / 328.88 / 329.69 ms = 3.02 / 3.04 / 3.03 dec/s at median crop load (54) — clears the
> 3.0 dec/s contract by only 0.7-1.3 % — and 2.60-2.62 dec/s at the 128-crop cap (FAIL);
> p95 crops 2.71-2.72, p99 2.64-2.66.~~ The surviving qualitative verdict: **P3 sits *on* the
> cadence boundary**, neither the "~2x miss" the projection claimed nor a comfortable pass.

Cost structure (deployment-representative): detector **62.1 %** (12.85 ms/frame x 16),
postprocess 14.3 %, decode 11.7 %, appearance 6.1 %, crop-extract 5.5 %, relation head **0.05 %**
— the head this thesis contributes is free; the frozen detector is the entire bill.
The real Mac->Jetson detector transfer factor is **9.1x**, not the 3.11x P1 proxy the projection
used, which is why the projection understated the device by ~3x. Two engine-build defects were
found and fixed before any number was taken (all engines pinned to batch 1, which would have
forced a 54-iteration appearance loop; and `throughput_decisions_per_sec_naive` = 73.051, which
is not a decision rate and may never be quoted). Full analysis:
`plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-09-29-jetson-decision-latency.md`.

## 5. FP16 vs FP32 parity (agreement rate) (~2-5 min)

```bash
python3 deployment/scripts/jetson-full-measurement.py --parity --rule argmax
```
(`run_parity()` has no `mode` param — it auto-discovers EVERY precision per
stage and compares each against the fp32 reference in one pass, so `--mode`
is a no-op here; do not pass it, it's misleading. Only `--rule`/`--threshold`
matter.) Use `--rule argmax` for BOTH binary relation-head graphs too. Both
`p3-relation-head-bin-e3-fp32.onnx` and `-bin-e0-` now carry TRAINED weights
(landed 2026-09-19, val_macro 89.15%/90.92% — see
`deployment/export/MANIFEST.md`), so `argmax` agreement is a genuine
FP16-quantization-robustness check on a real decision boundary, not noise on
random logits as it would have been against the 2026-09-18 placeholder
graphs. Still use `--rule argmax`, not `--rule threshold`: no calibrated
operating threshold (τ) has been selected on val/OOF for either binary arch
yet (that is a separate, not-yet-run step — this session doesn't block on
it). Good: `decision_agreement_rate` >= 0.98 per stage, `max_abs_diff` small
(<0.05 for softmax-ish outputs) — this validates FP16 quantization
robustness for the graph; it still is not a test-set accuracy result (no
test touch spent on either checkpoint). Lower agreement = flag for thesis
discussion, not a blocker.

**RUN 2026-09-29 22:10 — `parity-20260929T221055.json`.** FP32 engines for all four P3 stages
were built first (they did not exist; appearance with the same `1:54:128` batch profile as its
FP16 counterpart, so the comparison is like-for-like). Result: `p3-appearance` 1.000 agreement
(max abs diff 2.53e-02), `p3-relation-head` 1.000 (1.66e-03), `p3-relation-head-bin-e3` 1.000
(0.0, bit-identical). All three clear the >=0.98 bar. **The `p3-detector` row (0.633 / 794.4) is
NOT a result and must not be quoted** — `--rule argmax` is inapplicable to a stage whose output is
a `(300, 6)` `[x1,y1,x2,y2,conf,cls]` table rather than a class-score vector, and the harness feeds
it `standard_normal` noise rather than an image. A real detector parity check (real frames, boxes
compared after `BRT.CONF` + `N_MAX` selection, IoU- and class-matched) has not been run. Recorded
in `number-sheet.md` Block 3 and Unresolved 6.

## 6. Multi-stream saturation, 1/2/4/8 streams (~10-15 min)

Needs real source videos already on device (`data/test/**/*.mp4`, transfer if
missing):
```bash
python3 deployment/scripts/jetson-full-measurement.py --streams 1 2 4 8 \
  --mode fp16 --sources data/test/*/*.mp4 --max-frames 300 --fps 24 \
  --class-labels Safe,Unsafe
```
(unquoted glob — the shell expands it to a file list; `--sources` takes the
already-expanded paths, it does not glob internally. `--class-labels` is new
2026-09-17 — the old hardcoded 4-class `CLASS_LABELS` list was removed from
`jetson-multi-stream-pipeline.py` since the head is moving to binary+aux;
omit `--class-labels` and the JSONL `label` field falls back to `class_N`,
which still works but is less readable — pass `Safe,Unsafe` when the
pipeline is loading `p3-relation-head-bin-e3-fp32.onnx` or `-bin-e0-` (both
now TRAINED weights, val_macro 89-91% — but still no test-touch accuracy
number, so treat printed Safe/Unsafe labels as a real-model wiring/throughput
demo, not a validated accuracy demo), or leave `--class-labels` off entirely
if the pipeline is still loading the old 4-class graph.)
Good: `drop_rate` stays 0 up to the stream count where `aggregate_fps`
plateaus (per JETSON-DEPLOYMENT.md, P1 held 0-drop up to 4 streams @ 24fps —
use as a sanity baseline, P3 is heavier so may saturate earlier).
`aggregate_fps` here is the FIXED metric (processed frames / elapsed), not
the old bug that counted drops as processed — cross-check `total_dropped` in
the same JSON if a number looks too good.

## 7. Energy (~2 min per stage)

```bash
python3 deployment/scripts/jetson-full-measurement.py --energy backbone --mode fp16 \
  --energy-sec 30 --idle-sec 10
```
repeat swapping `backbone` for other stage names from Step 4's output. Good:
`joules_per_inference_total_board`, `fps_per_watt_total_board`,
`joules_per_inference_delta_idle`, `fps_per_watt_delta_idle` and
`idle_power_w` all non-null, no `skipped_reason`, no
`delta_idle_skipped_reason`. Needs `tegrastats` on PATH (it is, by default
on JetPack).

Two energy figures, on purpose — quote ONE and say which:
- `*_total_board` = raw `VDD_IN`, the whole module's draw. What the deployed
  box actually consumes; includes CPU/RAM/IO idle.
- `*_delta_idle` = `(P_avg - P_idle)`, the marginal cost of running the
  engine. This is the form the project spec defines
  (`research/researcher-03-demo-and-edge-deployment.md` §5-6). The run
  measures `P_idle` itself, for `--idle-sec` seconds, with the engine
  already resident and no work submitted.

`n_inferences` counts invocations of ONE engine, so `joules_per_inference`
is NOT joules-per-decision. A P3 decision is 16 detector passes + 1 batched
appearance pass + 1 head pass; per-decision energy must be composed from
per-stage runs the same way per-decision latency is (Step 5).

The measured window opens only AFTER deserialisation + 20 warm-up
inferences, and divides by the MEASURED elapsed time, not `--energy-sec`.
Before 2026-09-29 the clock and the sampler both started before the engine
was deserialised, so seconds of dead time (n=0) sat inside the integral —
any energy JSON older than that date is invalid, and none was produced.

## 8. Thermal soak, 30 min (~35 min incl. cooldown)

Run this LAST (after clocks are unlocked, i.e. no `jetson_clocks` forcing) so
it reflects real sustained-load thermal behavior:
```bash
sudo jetson_clocks --restore   # undo the pre-flight lock, if you ran it
python3 deployment/scripts/jetson-full-measurement.py --thermal-soak backbone \
  --mode fp16 --soak-min 30 --soak-window-sec 60
```
Good: `possible_throttling` false, or if true, note the window index where
p50 first exceeded 1.15x the FASTEST window (see Triage §C). The baseline is
`baseline_p50_ms` (the fastest window, index in `baseline_window_index`), not
window 0: clocks still ramp during window 0 even though the engine is warmed
beforehand, and a slow first window would hide real drift. Both drifts are
reported — `latency_drift_pct_vs_fastest` (the one the flag uses) and
`latency_drift_pct_vs_first`. This is a
heuristic (see the docstring in `jetson-measure-power.py`) — cross-check
against `tegrastats.samples[*].temps_c` (TOP-LEVEL key in the output JSON,
NOT nested per-window; `temps_c` is a plural dict keyed by sensor name, e.g.
`{"cpu": 54.5, "gpu": 58.0, ...}` — the raw sensor names tegrastats prints,
parsed by `parse_tegrastats_line()` in `jetson-measure-utils.py`. There is
no per-window `windows[i].tegrastats` or scalar `temp_c` field — `windows[i]`
only has `t_offset_sec` and `latency_ms`; correlate a window's time range to
`tegrastats.samples[*].ts` manually if you need a per-window temp reading).

## 9. Memory (~instant, run anytime after Step 1/2)

```bash
python3 deployment/scripts/jetson-full-measurement.py --memory --mode fp16
```
Good: `host_peak_rss_mb`, `device_used_mb`, `device_total_mb`,
`device_free_mb`, `engines_resident_delta_mb` and `engines_loaded` all
non-null (exact key names — no `skipped_reason` on real hardware).

`engines_resident_delta_mb` is the footprint number to quote: free-before
minus free-after deserialising every engine found for `--mode`, with
`engines_loaded` naming which ones. `device_used_mb` is NOT a model
footprint — Orin Nano memory is unified and shared with the OS, so it
measures the whole system. Before 2026-09-29 this step loaded no engine at
all and reported only the system-wide figure tagged with a mode.

## 10. Power/clock state snapshot (~instant, run once at start and end)

```bash
python3 deployment/scripts/jetson-full-measurement.py --power-state
```
Captures `nvpmodel -q` + `jetson_clocks --show` into the JSON's
`device_state` — do this before AND after the soak to show any mode drift.

---

## 11. Fallback demo video (no existing recording — must capture fresh)

The prior audit found no demo recording exists anywhere in the repo/artifacts.
Do this regardless of how the measurements above go — it's the single most
persuasive thesis-defense asset if benchmarks run short.

1. Pick 1-2 clips from `data/test/` spanning a clear Safe and a clear Unsafe
   case (check `annotation_guide` or class subfolder names for label hints).
2. On the Jetson, screen-record the live single-stream pipeline running on
   that clip, overlay visible (bounding boxes + decision label):
   ```bash
   # if pipeline has a --display or --save-video flag, use it; else:
   ffmpeg -f x11grab -video_size 1280x720 -i :0.0 -t 90 -r 30 demo-raw.mp4
   ```
   (adjust capture backend if Jetson runs headless/via SSH X-forwarding —
   simplest fallback: run `jetson-multi-stream-pipeline.py` with 1 stream and
   screen-record your monitor with a phone if no capture tool is configured).
3. Trim to <2 min, keep both a Safe and an Unsafe example if possible.
4. Save to `deployment/artifacts/demo-video-2026-09-19.mp4` (or note actual
   path in the session report) — do NOT commit large binaries to git per
   `.gitignore` convention; keep it as a local artifact.
5. If the pipeline crashes or produces no visible output, fall back to
   recording just `jetson-full-measurement.py --latency` terminal output as
   evidence of a working measurement path — better than nothing.

---

## 12. Triage

**§A — `import tensorrt, pycuda.autoinit` fails / "CUDA context changed
between createInferBuilder and buildSerializedNetwork"**: import order bug.
`pycuda.autoinit` MUST be imported before `tensorrt` anywhere in the process
(see comment in `jetson-build-trt-engine.py`). Check no script reversed this;
restart the Python process (the primary CUDA context can't be fixed
mid-process once wrong).

**§B — INT8 calibration failure**: only applies to P1 stages this session (no
P3 calib data exists). Confirm `deployment/artifacts/calib-jetson/` was
transferred and has `<model>-batch-{0..N}.npy` files matching the stage name
(`backbone`, `direct4-s314`, `scenario-s7`, `pair{0-3}-s7`). If files are
missing, re-run `jetson-prep-int8-calib.py` on Mac and re-transfer — do not
attempt INT8 without them. If it still fails, skip INT8 for the session (FP16
numbers are the priority) and note the failure in the report.

**§C — early thermal throttling** (soak `possible_throttling: true` in the
first few windows, or device feels hot fast): confirm `jetson_clocks` isn't
still forcing max clocks with no airflow — check case/fan. If it's genuine
(no forced clocks, still throttling <10 min in), that itself is a real,
reportable finding for the thesis (edge deployment constraint), not a bug —
capture the window index where it started and stop the soak early rather
than waiting the full 30 min if temps keep climbing past ~85C junction.

**§D — engine build silently produces a `.engine` file but latency/parity
show `skipped_reason: "engine not found"`**: filename mismatch. Discovery
regex is `^(.*)-(fp32|fp16|int8)\.engine$` — confirm the build step named the
file `<stage>-<precision>.engine` exactly (no extra suffix), in
`deployment/export/jetson/`.

**§E — `--streams` step has no video decode / `sources` glob matches
nothing**: confirm `data/test/*/*.mp4` transferred to the Jetson (large,
often skipped in a rushed sync) — `du -sh data/test` on Mac first to gauge
transfer time before the session.

---

## 13. Minimum viable set (if the session gets cut short)

Priority order, cheapest+highest-signal first — stop after any of these and
you still have defensible numbers. **This ordering is the one ratified as the
D4 priority table (2026-09-20); it is authoritative for the 09-22 session.**
Step 1 (P1 engines) is skippable if `deployment/export/jetson/p1-*.engine`
already exist — P1's numbers are already reported and are not what this session
is for.

1. Step 0 pre-flight + Step 3 (model size) — ~10 min total, proves engines
   exist and are reasonably sized.
2. Step 4 (per-stage latency p50/p95/p99) — ~5 min, the single most
   thesis-relevant number (was previously measured with the buggy
   single-input `TRTRunner`; this is the first CORRECT measurement).
3. Step 5 (parity) — ~5 min, answers "does FP16 change the decision."
4. Step 11 (demo video) — do this even if everything else fails; it's the
   fallback asset, not optional.

Everything else (multi-stream saturation, energy, 30-min soak) is valuable
but cuttable — defer to a follow-up session if time runs out, and say so
explicitly in the thesis rather than presenting partial/estimated numbers.
