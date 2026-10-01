# Jetson Measurement Session Runbook — 2026-10-01

**Device**: Orin Nano **Super** 8GB (`phuongnguyen-desktop`), JetPack R36.5.0, TensorRT 10.3.0.
**Subject**: the R48 deployment path — frozen CLIP ViT-B/32 tower → B1 temporal head (the headline
classifier, 84.4% balanced accuracy on LSO-1100) — plus YOLOv11s as the **localisation** stage.
**Author operates the device.** Each on-device step below is announced before it is run.

## What this session is for

The accuracy lane is closed: G14, G17 and G18 all failed, so the deployed classifier is B1
(whole-frame CLIP + temporal transformer, **no detector in the class decision**) and YOLOv11s is
retained because the system must say *where* the hazard is. The edge chapter therefore owes two
separate numbers, and must never sum them into one:

1. **class decision cost** — 16 frames → CLIP tower → B1 head.
2. **localisation cost** — YOLOv11s per frame.

The only host number on record is CPU onnxruntime (CLIP tower batch-16 = 375.0 ms, 99.9% of wall
clock). That is a CPU number with no tensor cores and is **not** a device projection.

## Pre-registered predictions — written 2026-10-01 BEFORE any device number exists

Derived from the 2026-09-29 device session (`decision-latency-fp16-20260929T221836.json`):
YOLOv11s FP16 @imgsz800 = **12.90 ms/frame** p50; 16-frame stream decode = **38.71 ms/decision**;
P3 relation head = 0.164 ms. CLIP ViT-B/32 @224 ≈ 4.4 GFLOPs/image → 70.4 GFLOPs per 16-frame
decision; YOLOv11s @800 ≈ 14.7 GFLOPs/frame. Equal-efficiency scaling gives
70.4/14.7 × 12.90 ≈ **62 ms** for the tower.

| # | prediction | falsifier |
|---|---|---|
| **P1** | Tower (batch 16, FP16) lands in **60–120 ms**. | Outside that band. A ViT that scales *worse* than FLOP-parity on Orin is the interesting case and must be reported as such. |
| **P2** | Per decision **100–160 ms** incl. decode → **6–10 dec/s**, so B1 **MEETS** the 3.0 dec/s contract that P3 fails above median crop load. | < 3.0 dec/s measured ⇒ P2 refuted, the edge chapter reports B1 as missing the cadence, and the host-CPU extrapolation is vindicated instead. |
| **P3** | Backbone is **≥ 90%** of compute (tower+head, decode excluded). | < 90%. |
| **P4** | FP16 decision agreement with the FP32 graphs is **100%** on the 24-clip fixture. | Any disagreement ⇒ the FP16 latency may **NOT** be quoted beside the 84.4% accuracy; either report FP32-engine latency or report the agreement rate alongside every FP16 number. |

**These are not renegotiated after seeing results.** Whatever the device returns is what the thesis
reports; no re-runs for a better number. Every step writes JSON with `nvpmodel`, platform and the
host commit embedded (`deployment/artifacts/host-commit.txt`, stamped by the rsync — the 09-29
session recorded `git_commit: unknown` because the device has no `.git`).

## Power mode — DO NOT CHANGE IT

Modes on this board are `0=15W, 1=25W, 2=MAXN_SUPER`. The device **idles in mode 2**, which is the
fastest. `sudo nvpmodel -m 0` would silently measure everything at 15W. Mode 2 is also what the
09-29 P3 numbers were taken at, so leaving it alone is what makes today comparable.

---

## Step 0 — host side, already done (no device needed)

- [x] `benchmark-r48-clip-pipeline.py` now has a `--backend tensorrt` path, so the device and host
      numbers are the *same composition* through two runtimes.
- [x] `verify-r48-device-parity.py --make-fixture --n 24` — 24 real clips (12 hazard / 12 normal),
      FP32 reference probabilities, and a preprocessing checksum, in
      `deployment/artifacts/r48-parity-fixture/`. Verified reproducible on the host, and verified to
      reject a wrong clip.
- [x] `LANE=r48` added to the rsync so the April P1 graph families are not pushed (≈0.5 GB instead
      of ≈2 GB).

## Step 1 — reachability  *(device)*

```bash
ssh jetson 'hostname; uname -m; nvpmodel -q; df -h / | tail -1'
# expect: phuongnguyen-desktop / aarch64 / NV Power Mode: MAXN_SUPER / >= 8 GB free
python3 -c "import tensorrt; print(tensorrt.__version__)"   # on the Jetson: expect 10.3.0
```

Free space must cover the 335 MB tower ONNX **plus** its engine. If `ssh jetson` is not configured,
`phuong@192.168.1.42` is the address on record.

## Step 2 — transfer  *(host → device, ≈0.5 GB)*

```bash
LANE=r48 JETSON_HOST=jetson bash deployment/scripts/jetson-rsync-from-mac.sh
```

Then confirm on the device that the big file arrived intact — a truncated 335 MB ONNX fails at
engine-build time with a confusing parser error:

```bash
ssh jetson 'cd ~/my-project && ls -l deployment/export/r48-*.onnx && \
  md5sum deployment/export/r48-clip-vit-b32-vision-fp32.onnx'
md5 -q deployment/export/r48-clip-vit-b32-vision-fp32.onnx    # on the Mac; must match
```

## Step 3 — build the FP16 engines  *(device, the slow step)*

The tower is 87.8M parameters; expect **several minutes**, possibly 10+. Build it first so a
failure surfaces early. `--batch_profile 1:16:16` is required: one decision is 16 frames in one
batched pass, and an engine pinned to batch 1 would force a 16-iteration loop and overstate the
cost.

```bash
cd ~/my-project
python3 deployment/scripts/jetson-build-trt-engine.py --mode fp16 \
    --model r48-clip-vit-b32-vision --batch_profile 1:16:16
python3 deployment/scripts/jetson-build-trt-engine.py --mode fp16 \
    --model r48-b1-clip-temporal-3seed
python3 deployment/scripts/jetson-build-trt-engine.py --mode fp16 --model p3-detector
ls -lh deployment/export/jetson/
```

If the tower OOMs: the builder asks for a 4 GB workspace on an 8 GB board. Close everything else
first; if it still fails, report it rather than lowering the workspace silently — the workspace
size is part of the measurement conditions.

## Step 4 — HEADLINE: per-decision latency  *(device)*

```bash
python3 deployment/scripts/benchmark-r48-clip-pipeline.py \
    --backend tensorrt --mode fp16 --runs 100 --detector --power --json
```

Produces the two numbers the edge chapter is built on (class decision cost; localisation cost
reported separately) and resolves P1, P2 and P3. `--power` samples tegrastats across the run.

## Step 5 — FP16 parity on real footage  *(device)*

```bash
python3 deployment/scripts/verify-r48-device-parity.py --mode fp16
```

Resolves P4. Exits non-zero on disagreement. It also reports end-to-end latency **including
decode**, which is the number the demo actually experiences. The preprocessing checksum is checked
first: a decode mismatch is reported as a decode mismatch, never charged to quantisation.

## Step 6 — memory, engine sizes, energy  *(device)*

```bash
python3 deployment/scripts/jetson-full-measurement.py --mode fp16 --memory --model-size
python3 deployment/scripts/jetson-full-measurement.py --mode fp16 \
    --energy r48-clip-vit-b32-vision --energy-sec 60 --idle-sec 10
```

## Step 7 — pull the artifacts back  *(host)*

```bash
rsync -avz jetson:~/my-project/deployment/artifacts/jetson-measurements/ \
    deployment/artifacts/jetson-measurements/
```

Then the numbers are adjudicated against P1–P4 above and written to the number sheet with the
device state attached.

## Minimum defensible session

Steps 1–5. Step 6 is a nice-to-have; Step 4 alone, without Step 5, yields a latency number that
cannot legitimately be printed next to the accuracy number.

---

## Outcome — session run 2026-10-01, 22:1x–22:49

Steps 1–7 all run. **P1 refuted, P2 confirmed on its falsifiable claim, P3 confirmed, P4
confirmed.** Numbers, verdicts and the reasoning are in
`plans/2026-09-19-thesis-recovery-architecture/number-sheet.md` → Block 3b and
Block 3b-adjudication. Nothing here was renegotiated after seeing results; P1's failure and the
defect in P2's wording (no stride specified) are both recorded against the predictions as written.

### The session's main lesson: a shared device needs a guard, not a check

Two sessions were measuring at once. Step 4 and Step 5 were first run while a concurrent 60 s
energy sweep saturated the GPU, and re-run while that session had pinned all clocks — a **1.79×
spread** on the tower with **byte-identical provenance** in all three artifacts. A point-in-time
idle check before launching is **not** sufficient: exactly that check passed at 22:11, and the
competing sweep started afterwards.

What worked is a device-side guard that **polls until the clocks are restored and no measurement
process is running, settles 90 s, measures, then re-reads the state** — and aborts instead of
measuring in a dirty window. Kept at `/tmp/r48-clean-window.sh` on the device; the pattern is worth
re-creating rather than reusing a stale copy.

Provenance is now self-describing: `deployment/scripts/jetson-clock-state-utils.py` records
per-policy CPU/GPU min/cur/max, `clocks_pinned`, `competing_processes` and `quiet` into every
artifact via the shared `device_state()`, and warns on stderr at measurement time. **Before quoting
any device latency, check its artifact's `quiet` and `clocks_pinned` fields.** Artifacts written
before 2026-10-01 22:38 do not carry them and cannot be cleared from the record alone.

### Answers to the three unresolved questions above

1. **Does the tower hit FP16 tensor cores?** Evidently yes, and better than predicted — 23.386 ms
   for 70.4 GFLOPs is ~3.0 TFLOP/s effective, and the `trtexec --verbose` layer-precision dump was
   never needed. There is no LayerNormalization fallback problem to chase.
2. **INT8 for the tower?** Moot. FP16 clears the cadence contract by 14× at the demo's stride, so
   INT8 buys nothing the thesis needs and would cost a CLIP calibration set that does not exist.
3. **imgsz 640 vs 800.** The localisation figures above are **imgsz 800**, batch 1, stated as such
   in every row — the same configuration as the 09-29 detector number, so the two are comparable.
   The features were built at 640; that mismatch belongs to the concurrent imgsz-scaling work
   (`finding-2026-10-01-cadence-levers.md`) and is not resolved here.

### Still open after this session

- Device memory for the B1 engine set is **provisional**: the artifact self-labels `quiet: false`
  (`memory-fp16-20261001T224929.json`, 92.24 MB resident delta). Re-run
  `--memory --engine-dir deployment/export/jetson-r48` in a quiet window.
- No **composed** live-pipeline measurement exists for B1 — only per-stage numbers and a stage-sum
  duty budget (Block 3b-note 4), which must not be presented as a measured cadence.
- The idle power baseline disagrees between sessions (4.842 W here, 7.563 W on 09-29). A concurrent
  session measured a 60 s settled baseline at ~22:54 which should supersede both.

## Unresolved questions

1. Does the exported tower graph hit Orin's FP16 tensor cores at all, or does opset-18
   `LayerNormalization` fall back? If the measured tower lands far above the 120 ms band, this is
   the first thing to check (`trtexec --verbose` layer precision dump).
2. INT8 for the tower is out of scope today — no CLIP calibration data exists. If FP16 misses the
   cadence, INT8 is the next lever, not a same-day fix.
3. The 09-29 detector number is imgsz **800**; the features were built at imgsz **640**. The
   localisation cost quoted in the thesis must state which imgsz it is, and the concurrent
   imgsz-scaling work (`finding-2026-10-01-cadence-levers.md`) is the place that reconciles them.
