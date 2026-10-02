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
P3 relation head = 0.164 ms. *(Inputs superseded 2026-10-01. Taking the median-total
run in each clock condition: detector **11.724** ms/frame pinned vs **14.246** DVFS, decode
**37.61** vs **43.41** ms/decision, head **0.158** ms. The projection below is left at its
pre-registered values — it was written before any device number existed and is not renegotiated;
on the pinned detector the tower estimate would be 70.4/14.7 × 11.724 ≈ **56 ms** rather than
~62 ms.)* CLIP ViT-B/32 @224 ≈ 4.4 GFLOPs/image → 70.4 GFLOPs per 16-frame
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

- ~~Device memory for the B1 engine set is **provisional**.~~ **RESOLVED 2026-10-01 23:28:** three
  repeats in a guarded quiet window give **90.01 MB** median (91.36 / 90.01 / 89.61;
  `memory-fp16-20261001T{232747,232808,232830}.json`), spread **1.94 %**, which passes the rule
  pre-registered before the run (`max − min ≤ 0.10 × median`). The withdrawn `quiet: false` reading
  was 92.24 MB, only 2.5 % off, so the guard protected against a risk that did not materialise.
  Surviving caveat: 90.01 MB is **less than half** the 197.4 MB engine-file total, so the field is a
  **system-wide free-memory delta**, not "weights resident", and must be worded that way.
- ~~No **composed** live-pipeline measurement exists for B1.~~ **RESOLVED 2026-10-01 23:35, and it
  refuted the stage-sum budget.** Three repeats at fixed seed 0 (24 clips, 201 decisions, 5559
  frames, `--window 48 --stride 24`, all `quiet: true`, clocks unpinned):
  **289.672 ms/decision = 3.452 decisions/s**, p95 322.708 ms, duty **76.1 %**, spread 1.21 %
  (`r48-composed-cadence-fp16-20261001T{233159,233324,233449}.json`). The stage sum understates that
  by **3.97x**. Four consequences, all in Block 3b-note 9 and
  `plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-01-stage-sum-is-not-a-cadence.md`:
  **preprocessing** (160.765 ms, 73.4 % of compute) was absent from every budget and is now the top
  optimisation target; the **tower costs 2.51x more in the loop**, adjudicated by a pre-registered
  probe to **GPU clock droop over the idle gap** (idle alone reproduces 94.2 %), not contention, so
  pinning clocks is worth at most **+12.5 %**; the **"1.05 dec/s contract" is withdrawn** (it assumed
  25.1 fps, which 1.0 % of the corpus has — median is 29.97 and 19.8 % of clips are 60 fps, where the
  margin is **1.38x**); and **decode cost is resolution-dependent** (2.048 ms/frame on this corpus
  against 5.047 at 1080p), so a 1080p 60 fps source would leave only ~1.10x — **unmeasured, and the
  most likely way the deployment misses real time.**
- **New, opened by the above:** none on a **60 fps 1080p** source, no **soak**; and two of the four
  items opened here are now closed or decided.
  - ~~preprocessing has not been profiled internally~~ **PROFILED 2026-10-02, on host, and the cause
    is a defect in our own code.** The BGR→RGB swap ran **before** the resize on a negative-stride
    view, forcing OpenCV to copy each frame at **source** resolution: 90.3 of 101.6 ms at 1080p, 5.5
    of 15.5 ms at 480×270. Swapping after the resize is **bit-identical** and 7.46× / 1.36× faster.
    Applied to `verify-r48-device-parity.preprocess` and `build-isafetybench-clip-features.to_batch`;
    gated by 11 tests against the hard-coded pre-change formula. **The device re-measurement has not
    run** — see the queued step below — so no Block 3b row and no table in §5.6.9 has been changed.
  - ~~no composed measurement with **clocks pinned**~~ **DECIDED: will not be measured.** The cause
    is already adjudicated, the arithmetic ceiling is **+12.5 %**, and pinned clocks are not the
    state the device boots or deploys in. Block 3b-note 11 records the decision and its reason; the
    whole series stays unpinned.

## Step 7b — DO THIS FIRST when the board answers: pull six missing artifacts  *(host)*

Found 2026-10-03 by the citation-vs-index sweep. Six result files measured on the device
on 2026-10-01 were never rsync'd back and exist **nowhere** on the Mac:

```
r48-composed-cadence-fp16-20261001T233159.json   # Bảng 5.5O
r48-composed-cadence-fp16-20261001T233324.json
r48-composed-cadence-fp16-20261001T233449.json
r48-idle-droop-probe-fp16-20261001T234145.json   # Bảng 5.5P
r48-idle-droop-probe-fp16-20261001T234242.json
r48-idle-droop-probe-fp16-20261001T234322.json
```

They carry the headline **3,452 decisions/s**, p95 322,708 ms, the 2,76×/1,38× margins,
the 76,1 % duty factor and the entire pre-registered H-sụt-nhịp adjudication. Until they
are here, those two tables are quoted but not re-readable — the one standard the 02-10/03-10
audits hold every other number to. Runs **before** Step 8 because Step 8 overwrites the
device's measurement directory state and because a re-measure cannot restore 01-10 evidence.

```bash
ssh jetson 'ls -la ~/r48/out/ | grep -E "composed-cadence|idle-droop"'
rsync -av jetson:'~/r48/out/r48-composed-cadence-fp16-20261001T23*.json' \
          jetson:'~/r48/out/r48-idle-droop-probe-fp16-20261001T23*.json' \
          deployment/artifacts/jetson-measurements/
# then: confirm each carries quiet=true, clocks_pinned.any=false, competing_processes=[]
#       and that the quoted p50s match Bảng 5.5O/5.5P before committing
git add -f deployment/artifacts/jetson-measurements/r48-composed-cadence-*.json \
           deployment/artifacts/jetson-measurements/r48-idle-droop-probe-*.json
```

**If they are gone from the device:** both tables must be re-measured under the same
protocol, or marked unverifiable. Do not re-quote them from the chapter text.

## Step 8 — QUEUED: re-measure after the preprocessing fix  *(device, blocked)*

**Blocked since 2026-10-02 ~21:00: the board is unreachable — `ping` 100 % loss and ARP incomplete,
i.e. powered off or unplugged, not an ssh problem.** Run this in order the moment it answers. The
predictions each step adjudicates are registered, with falsifiers, in
`plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-02-preprocessing-cost-is-a-channel-swap.md`
and are not renegotiated after seeing the output.

```bash
# host: ship the four changed files (engines and artifacts stay where they are)
rsync -av deployment/scripts/{verify-r48-device-parity.py,benchmark-r48-composed-cadence.py,\
profile-r48-preprocess.py,preprocess-variants-utils.py} jetson:my-project/deployment/scripts/

# device, in a quiet window (check load, temp, clocks_pinned first, as every step above does)
cd ~/my-project
# 1. the gate itself, on THIS machine's OpenCV build -- adjudicates P-PRE1's premise
python3 deployment/scripts/profile-r48-preprocess.py --data-root data/isafetybench --json
python3 deployment/scripts/profile-r48-preprocess.py --data-root data/test --json     # 1080p
# 2. pixel checksums vs a fixture built by the PRE-change host code -- adjudicates P-PRE1
python3 deployment/scripts/verify-r48-device-parity.py --mode fp16
# 3. same 24 clips, same seed, 3 repeats -- adjudicates P-PRE2 and P-PRE3
for i in 1 2 3; do python3 deployment/scripts/benchmark-r48-composed-cadence.py \
  --engine-dir deployment/export/jetson-r48 --mode fp16 --clips 24 --seed 0 --json; done
# 4. the 1080p risk, which no run has ever closed -- adjudicates P-PRE4
for i in 1 2 3; do python3 deployment/scripts/benchmark-r48-composed-cadence.py \
  --engine-dir deployment/export/jetson-r48 --mode fp16 --data-root data/test \
  --clips 24 --seed 0 --json; done
```

Check before step 4 that `data/test` is present on the device; if not, rsync it (125 clips, all
1920×1080). **If step 2 reports any pixel mismatch, stop: revert the fix rather than patch it, and
withdraw the 02-10 finding** — that is P-PRE1's falsifier and it is not negotiable.
- ~~The idle power baseline disagrees between sessions (4.842 W here, 7.563 W on 09-29).~~
  **RESOLVED 2026-10-01 22:55:** one settled 60 s baseline was measured after a 300 s settle —
  **4.691 W** (n=118, `quiet: true`, clocks not pinned, load 0.08, 50 °C,
  `energy-p3-relation-head-bin-e3-fp16-20261001T225523.json`). It supersedes both earlier figures.
  Every `*_delta_idle` field in the stage artifacts is computed against that stage's own 10 s
  window and is **not quotable**; recompute marginal energy against 4.691 W instead.
  Consequence for the R48 row, recomputed: **total board 0.4811 J per decision is unchanged** --
  it is `P_avg / fps` and never touches the baseline -- while the marginal figure moves from
  0.36522 to **0.36885 J** per tower invocation, a 1.0 % shift. The settled baseline was taken with
  only the 0.190 MB head engine resident, so it is a lower bound on engine-resident idle and that
  marginal figure is correspondingly a mild upper bound. The B1 head's own energy was never measured
  separately; it is bounded by its 1.7 % share of decision wall clock.

## Unresolved questions

1. Does the exported tower graph hit Orin's FP16 tensor cores at all, or does opset-18
   `LayerNormalization` fall back? If the measured tower lands far above the 120 ms band, this is
   the first thing to check (`trtexec --verbose` layer precision dump).
2. INT8 for the tower is out of scope today — no CLIP calibration data exists. If FP16 misses the
   cadence, INT8 is the next lever, not a same-day fix.
3. The 09-29 detector number is imgsz **800**; the features were built at imgsz **640**. The
   localisation cost quoted in the thesis must state which imgsz it is, and the concurrent
   imgsz-scaling work (`finding-2026-10-01-cadence-levers.md`) is the place that reconciles them.
