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

## Step 7b — ✅ DONE 2026-10-03 16:0x: six missing artifacts recovered  *(host)*

Found 2026-10-03 by the citation-vs-index sweep. Six result files measured on the device
on 2026-10-01 were never rsync'd back and existed **nowhere** on the Mac:

```
r48-composed-cadence-fp16-20261001T233159.json   # Bảng 5.5O
r48-composed-cadence-fp16-20261001T233324.json
r48-composed-cadence-fp16-20261001T233449.json
r48-idle-droop-probe-fp16-20261001T234145.json   # Bảng 5.5P
r48-idle-droop-probe-fp16-20261001T234242.json
r48-idle-droop-probe-fp16-20261001T234322.json
```

They carry the then-headline **3,452 decisions/s**, p95 322,708 ms, the 2,76×/1,38× margins,
the 76,1 % duty factor and the entire pre-registered H-sụt-nhịp adjudication. Ran **before**
Step 8 because a re-measure cannot restore 01-10 evidence.

**⚠ The path in this step was wrong.** `~/r48/` does not exist on the device; the files were in
the device repo's own measurement directory all along. Corrected commands:

```bash
ssh jetson 'find ~ -name "r48-*.json" | head -40'   # this is what located them
D=~/my-project/deployment/artifacts/jetson-measurements
rsync -av jetson:"$D/r48-composed-cadence-fp16-20261001T23*.json" \
          deployment/artifacts/jetson-measurements/
rsync -av jetson:"$D/r48-idle-droop-probe-fp16-20261001T23*.json" \
          deployment/artifacts/jetson-measurements/
# NOTE macOS rsync 2.6.9 rejects two remote SRC specs in one invocation -- split them, as above.
git add -f deployment/artifacts/jetson-measurements/r48-composed-cadence-*.json \
           deployment/artifacts/jetson-measurements/r48-idle-droop-probe-*.json
```

**Outcome.** All six recovered and staged. Every run carries `quiet=true`,
`clocks_pinned.any=false`, `competing_processes=[]`. **Every figure Bảng 5.5O/5.5P quote was
re-derived from the artifacts and matched exactly**: 289,672 ms · 3,452 dec/s · p95 322,708 ms ·
duty 0,7615 · droop ratio 2,088 (also 2,152 / 2,051) · 94,2 % reproduction · 3,399 ms residual ·
band membership. `git merge-base --is-ancestor` proves commit `9c1e219` predates the fix `cc27d82`,
so these are the **pre-fix arm of the Step 8 A/B**, not stale numbers.

## Step 8 — ✅ DONE 2026-10-03 15:53–16:05: re-measured after the preprocessing fix  *(device)*

Was blocked 2026-10-02 ~21:00 → 2026-10-03 (board powered off). Ran in the order below. The
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

### ⚠ Provenance trap found while running this — read before any future device session

**The device has no `.git`.** `jetson-measure-utils.py:38-48` `git_commit()` tries
`git rev-parse --short HEAD` and, on failure, falls back to `deployment/artifacts/host-commit.txt`.
That file still read `9c1e219` — the **pre-fix** commit. Without repairing it first, every Step 8
artifact would have been stamped with the commit of the code it was measuring the replacement of,
and the A/B would have been unauditable. **Always rsync a fresh `host-commit.txt` before measuring.**
It was set to `6319b21-dirty`; the dirty delta was proven docstring-only by comparing ASTs with
docstrings stripped (identical), not by assertion.

### Outcome — all four predictions adjudicated

| | prediction | measured | verdict |
|---|---|---|---|
| P-PRE1 | parity exact on 24 clips | **0** mismatches, agreement 1,0 | **PASS** |
| P-PRE2 | prep p50 factor in [1,20 ; 1,55] | **3,160×** | **FALSIFIED** (upward) |
| P-PRE3 | steady state in [230 ; 275] ms | **167,291 ms** | **FALSIFIED** (upward) |
| P-PRE4 | 1080p *stage* gain ≥ 3× | **8,70×** | **PASS** |

Headline, protocol-identical A/B (same 24 clips / 201 decisions / 5559 frames / window 48 /
stride 24): **289,672 → 167,291 ms = 3,452 → 5,978 dec/s (+73,2 %)**; margin over the **1,249 dec/s**
corpus-median contract **2,76× → 4,79×** (the 3,29× → 5,69× pair against the withdrawn 1,05 dec/s
contract is retired). At 1080p: **227,214 ms = 4,401 dec/s**, margin **3,52×** (hợp đồng trung vị 1,249; 1,76× ở p95 2,500), absolute only
(no pre-fix 1080p baseline exists — P-PRE4 forbids a before/after claim there).

Both falsifier-assigned causes were **refuted by measurement**, and the real cause was a
mis-specified band (read off a 480×270 clip that is 1 of 24 and has 7,1× fewer pixels than the
corpus median). Full account, with the accounting identity and the unresolved questions:
`plans/2026-09-19-thesis-recovery-architecture/reports/finding-2026-10-03-preprocessing-fix-adjudicated-on-device.md`.

Artifacts: `r48-preprocess-profile-20261003T{155346,155531}.json`,
`r48-parity-fp16-20261003T155636.json`,
`r48-composed-cadence-fp16-20261003T{155728,155806,155843}.json` (iSafetyBench),
`r48-composed-cadence-fp16-20261003T{160245,160341,160439}.json` (1080p). All `is_device_number:
true`, `quiet=true`, `clocks_pinned.any=false`, `competing_processes=[]`.

## Step 9 — ✅ DONE 2026-10-03 16:07–16:36: pinned-clock thermal soak  *(device)*

Closes the last open item on the P3 lane: the **+7,5 %** pinned-clock margin was never shown to
survive 30 minutes. Clocks verified pinned before launch (CPU min=cur=max=1.728.000 kHz;
GPU min=cur=max=1.020 MHz; `clocks_pinned.any=true`; `quiet=true`; `competing_processes=[]`).

```bash
ssh jetson 'cd ~/my-project && sudo jetson_clocks && \
  nohup setsid python3 deployment/scripts/jetson-full-measurement.py \
    --thermal-soak p3-detector --mode fp16 --soak-min 30 > soak-pinned.log 2>&1 < /dev/null &'
# the launching ssh will NOT return -- background it and verify the pid in a second session.
# soak-pinned.log stays EMPTY until exit: python3 buffers stdout when redirected. Verify liveness
# with `ps -o etime,time,pcpu -p <pid>` and /sys/devices/platform/gpu.0/load instead of the log.
```

**MANDATORY afterwards: `sudo jetson_clocks --restore`.** Pinned state is not comparable with the
unpinned runs that every other number in this runbook uses, and leaving it pinned would silently
contaminate the next session.

### ⚠ `--restore` does not work if you did not `--store` first

`jetson_clocks` writes a baseline file **only when `--store` is passed**. The launch command above
runs bare `sudo jetson_clocks`, so **no baseline file was ever written** and `--restore` had nothing
to restore. Verified: `find / -maxdepth 3 -name l4t_dfs.conf` returned nothing.

**Path correction (2026-10-03, verified by running it).** On this image (JetPack R36.5.0) the store
target is **`/root/.jetsonclocks_conf.txt`**, not `${HOME}/l4t_dfs.conf` — the earlier version of
this note named the wrong file, which is why the `find` above searched for a name that this
JetPack never writes. `sudo jetson_clocks --store` printed
`Storing system configuration in /root/.jetsonclocks_conf.txt` and succeeded; it was run on the
**unpinned** state immediately before pinning, so `--restore` is now usable for the first time in
this campaign. Run `--store` while the board is in its default state — storing a pinned state makes
`--restore` a no-op. The three knobs `jetson_clocks` actually pins (read off `/usr/bin/jetson_clocks`) had to be
released by hand:

```bash
# CPU: release the min-freq floor on all 6 cores (governor was already schedutil)
for d in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do
  echo 115200  | sudo tee $d/scaling_min_freq >/dev/null   # cpuinfo_min_freq
  echo 1728000 | sudo tee $d/scaling_max_freq >/dev/null   # cpuinfo_max_freq
done
# GPU: release the devfreq floor (hw range 306 MHz .. 1.020 MHz)
G=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu
echo 306000000  | sudo tee $G/min_freq >/dev/null
echo 1020000000 | sudo tee $G/max_freq >/dev/null
# EMC: drop the rate override (this is the one that is easy to forget)
echo 0 | sudo tee /sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked >/dev/null
```

`nvpmodel` is **not** touched — MAXN_SUPER (2) is the standing power mode, not a pin.

**Do not accept the knob values as proof of unpinning.** Proof is that frequencies *fall below* the
pinned values at idle. Measured after restore: `cpu4 = 115.200` kHz, `cpu0 = 1.113.600` kHz,
`gpu = 306 MHz`, `emc mrq_rate_locked = 0` — all impossible while pinned. **For the next session:
pass `--store` before `sudo jetson_clocks`,** or skip `jetson_clocks` and set the floors directly so
the inverse is known.

### Outcome — pinning is worth more in predictability than in speed

`thermal-soak-p3-detector-fp16-20261003T163622.json`, `clocks_pinned.any = **true**` (cpu true, gpu
true) — the clock state is now *recorded in an artifact*, which was the whole reason for the run.
Counts derived by **summing** the 30 windows.

| metric | unpinned 10-01 | **pinned 10-03** | Δ |
|---|---:|---:|---|
| duration / windows | 30 min / 30 | 30 min / 30 | identical protocol |
| `clocks_pinned.any` | false | **true** | the point of the run |
| `possible_throttling` | false | **false** | neither throttled |
| inferences (summed) | 140.001 | **152.721** | **+9,09 %** |
| per-window n | 4.651–4.689 | 5.072–5.105 | |
| p50 median (ms) | 12,796 | **11,684** | **−8,69 %** |
| p50 range (ms) | 12,734–12,866 | 11,670–11,706 | |
| **p50 spread (%)** | 1,03 | **0,31** | **3,3× tighter** |
| drift vs fastest (%) | 0,7 | **0,13** | |
| drift vs first (%) | 0,6 | **−0,06** | |
| tj max (°C) | 69,9 | **70,9** | +1,0 K |
| power mean (W) | 16,98 | **17,93** | +5,55 % |
| power peak (W) | 18,02 | **19,07** | +5,87 % |
| J/inference marginal | 0,1573 | **0,1547** | **−1,69 %** |
| fps/W marginal | 6,357 | **6,466** | **+1,71 %** |

Four things this settles:

1. **The pinned *detector stage* holds for 30 minutes.** No throttling, tj 70,9 °C max — far from
   the throttle point — and drift 0,13 %. ~~The +7,5 % pinned margin survives 30 minutes.~~
   **Rewritten 2026-10-03:** that margin came from a stage sum and is withdrawn (see the
   composition note below and Step 10). This soak ingests **only the detector stage**, so it
   evidences sustained *detector* throughput under a pinned clock — nothing about a decision
   cadence. **No composed-decision soak exists.**
2. **Thermal steady state was reached, so 30 min was long enough.** tj over the last third of
   samples changes by **−0,1 K** (unpinned: +0,1 K); the soak measures sustained operation, not a
   transient. A longer soak would add nothing.
3. **Pinning is not a power-for-speed trade here — it is net positive on energy.** +9,09 % work for
   +5,55 % mean power ⇒ marginal energy per inference **falls 1,69 %** and marginal fps/W **rises
   1,71 %** (against the settled 4,691 W idle). The usual objection to pinning does not apply.
4. **The real argument for pinning is predictability, not throughput.** Latency spread tightens
   **3,3×** (1,03 % → 0,31 %) and drift collapses from 0,7 % to 0,13 %. For a safety system a tight
   tail matters more than a 9 % mean.

**Composition note — the coherence check is WITHDRAWN, and composition is now MEASURED
(2026-10-03):** the old form read "pinning buys +18,0 % on the P3 decision stage sum
(365,88 → 310,00 ms) but only +9,5 % on the detector stage alone (12,796 → 11,684 ms)" as
corroboration of the bus-contention mechanism. **That inference is unavailable:** the P3 decision
figure is a sum of stages each timed **alone**
(`reports/finding-2026-10-03-p3-decision-latency-is-a-stage-sum.md`), so it contains no concurrency
and cannot evidence contention. The **composition** explanation — that the sum carries CPU-bound
stages which `scaling_min_freq` accelerates while the soak is GPU-bound and thermally saturated —
was speculation then and is **measured now** (Step 10). The pinning effect scales with how
composed the measured quantity is:

| quantity | DVFS | pinned | pinning effect |
|---|---:|---:|---:|
| detector stage alone (30 min soak) | 12,796 ms | 11,684 ms | **+9,5 %** |
| stage sum, thermally gated pair | 421,52 ms | 310,37 ms | **+26,4 % (1,26×)** |
| **composed cadence, measured** | **1,441 qd/s** | **3,479 qd/s** | **+141,4 % (2,41×)** |

Mechanism, machine-recorded in the composed artifacts: under `schedutil`, **`policy4` (cores 4-5)
sat at the 115200 kHz floor in every DVFS sample**, before and after every run, while `policy0`
wandered 192000–1728000; pinned, both clusters read `cur = min = 1728000`. The composed loop is
CPU-bound on preprocessing — exactly the work the stage sum omits — so the more of that work the
measured quantity contains, the larger the clock effect. The two runs still use different
protocols, so this **corroborates** the mechanism rather than testing it.

Compare against the clean unpinned soak (2026-10-01 23:25): 30 × 60 s windows, **140.001**
inferences, p50 12,734–12,866 ms, spread 1,03 %, drift 0,7 % vs fastest / 0,6 % vs first, tj
69,9 °C max, 16,98 W mean / 18,02 W peak, `possible_throttling` false. Derive per-window counts by
**summing** the windows, not by multiplying a rate.
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

## Step 10 — ✅ DONE 2026-10-03 16:54–17:47: the composed P3 cadence, matched pair  *(device)*

Steps 4 and 8 produce a **stage sum**, which Step 9's composition note shows cannot adjudicate a
cadence. This step replaces it with a measurement of the whole loop under one wall clock. Two
measurements were taken, in this order:

**10a — thermally gated stage-sum pair (16:54–17:26)**, to separate the thermal confound from the
governor: 3 runs per arm, each gated on entry (tj ≤ 56 °C, load ≤ 0,60, after a 180 s settle).
Pinned 310,37 ms median, DVFS 421,52 ms median ⇒ **+26,4 %**, not the ungated +18,0 %.
**The thermal-confound hypothesis is REJECTED:** gated and ungated DVFS medians are 421,52 and
422,82 ms — **0,31 % apart**. The spread is the governor's in-run ramp, not entry temperature.
Across all 11 stage-sum runs over two days the two arms **do not overlap** (pinned 309,08–311,78 ms,
spread 0,49 %; DVFS 365,88–440,01 ms, spread 20,3 %; max pinned < min DVFS). **DVFS is not a
reproducible measurement condition on this device** — a method result in its own right, and the
reason every artifact must record its clock state.
Artifacts: `decision-latency-fp16-20261003T{170736,171000,171223}.json` (pinned),
`decision-latency-fp16-20261003T{172128,172406,172644}.json` (DVFS).

**10b — composed cadence (17:33–17:47)**, the headline. `jetson-p3-multi-stream.py` runs file
decode → 800×800 letterbox preprocess → detector → select + zone geometry → 224² context crops →
appearance encoder → 74-d relation-token assembly → binary E3 head, all in one wall clock. **All
six omissions of the stage sum are closed.** Same gate as 10a, N = 1, 480 frames, 3 test clips,
FP16, MAXN_SUPER. **Timing only — no labels are read, so this consumes no test touch.**

```bash
# per arm (pinned: `sudo jetson_clocks --store && sudo jetson_clocks` first; see Step 9)
for CLIP in 1_te10 3_te1 4_te10; do
  python3 deployment/scripts/jetson-p3-multi-stream.py --streams 1 \
    --source data/test/*/${CLIP}.mp4 --max-frames 480 --fps 1000 --mode fp16 \
    --engine-dir deployment/export/jetson-p3 --detector-stage p3-detector \
    --head-stage p3-relation-head-bin-e3 \
    --out deployment/artifacts/jetson-measurements/p3-composed-cadence-fp16-<arm>-free-${CLIP}.json
done
# --fps 1000 is deliberate: it means "unpaced". See reading rule 1 below -- `--fps 24` does NOT
# measure the contract.
```

| clip (view) | PINNED dec/s | ms/frame | DVFS dec/s | ms/frame |
|---|---:|---:|---:|---:|
| 1_te10 (3) | **3,479 MEETS** | 35,32 | 1,441 FAILS | 85,32 |
| 3_te1 (7) | **3,965 MEETS** | 31,01 | 1,511 FAILS | 81,37 |
| 4_te10 (1) | 2,493 FAILS (1,20×) | 49,31 | 1,141 FAILS | 107,76 |
| median | **3,479** | | **1,441** | |

Pinned **meets on 2 of 3 clips**, DVFS on **none** (short by 1,99×–2,63×). The arms are
**disjoint** (max DVFS 1,511 < min pinned 2,493). **Both earlier claims are dead:** the Step 4
"+7,5 % margin, PASS" rested on the stage sum, and its first correction — "P3 misses in both clock
conditions" — rested on arithmetic over that same sum and is **false in the pinned arm**. The
measurement was necessary, not merely more rigorous. Artifacts:
`p3-composed-cadence-fp16-{pinned,dvfs}-{free-{1_te10,3_te1,4_te10},contract24}-20261003T17*.json`.

### ⚠ Three reading rules for the composed artifacts — read before quoting any of them

1. **`--fps F` is an additive pacer, not a deadline — this is a real defect.**
   `stream-pipeline-utils.py:84` sleeps `period` *after* all per-frame work, so the frame interval
   is `work + 1/F` and the achievable rate is `1/(1/F + work) < F`. At `--fps 24` the ceiling is
   3,000 dec/s only if work = 0; at the pinned 20,27 ms/frame it is ≈ 2,0. Proven internally: the
   pinned aggregate is **30,17 fps at N = 2 but 17,89 fps at N = 1** — N = 1 is pacing-limited, not
   pipeline-limited. **The `--fps 24` rows are NOT a contract verdict; only `--fps 1000` is.**
2. **`drop_rate = 0` is not evidence of real-time operation at N ≤ 2.** The reader is a coroutine on
   the consumer's event loop with `preprocess_fn` inline, so it is back-pressured into lockstep
   instead of dropping. The signature is exact in all five N ≤ 2 configs: **`read_fps ==
   aggregate_fps` to the cent**, including the run that requested 1000 fps and delivered 11,72.
   Only N = 3 decouples (18,93 read vs 13,39 processed, 29,24 % dropped). Cite `read_fps` against
   `target_fps`, never `drop_rate`.
3. **`clock_state_stable` means the pin *mode* and contention did not change, NOT that frequencies
   held.** It compares `clocks_pinned` and `quiet` before/after
   (`stream-pipeline-utils.py:140`), so it reads `true` for every DVFS run while `policy0 cur_khz`
   moved 1574400 → 192000 inside one run. **The code matches its docstring — the field *name* is
   the hazard.** For a frequency claim, read `device_state.cpu_clocks.*.cur_khz` before and after.

**Multi-stream, for context only (not a contract verdict — see rule 1).** Pinned aggregate
17,89 / 30,17 / 27,83 fps at N = 1/2/3 ⇒ worst-stream 2,199 / 1,854 / 1,140 dec/s; DVFS
12,02 / 14,56 / 13,39 fps ⇒ 1,478 / 0,895 / 0,486–0,565. The aggregate ceiling is roughly **flat in
N** (≈28–30 fps pinned, ≈13–15 fps DVFS): concurrency redistributes rather than adds, and **no
N ≥ 2 config reaches 3,0 dec/s per stream**. Drops appear only at DVFS N = 3 (29,24 %); pinned
N = 3 drops nothing.

**Harness arithmetic verified before use:** `n_decisions == floor((processed − 16)/8) + 1` holds
exactly in all 12 stream records, so the emission stride really is 8 and the contract really is
24 fps ÷ 8 = 3,0 dec/s. The stage sum, by contrast, multiplies the detector p50 by `T_FRAMES = 16`
and never uses `DECISION_EMIT_EVERY = 8`, defined on the next line
(`jetson-measure-latency.py:105-106`) — a double-charge of 8 detector passes per decision.

**A pilot run on `0_te10` reported 1,061 dec/s but wrote no artifact — it is NOT citable.**

**Device left clean:** `sudo jetson_clocks --restore` (after a `--store`, per Step 9), verified by
**observed frequency**, not config readback: cpu0 min 115200, cpu4 min 115200, GPU min 306 MHz.

### Still missing after Step 10

- **No composed-decision thermal soak.** Both soaks ingest only the detector stage. A 30-min
  composed soak is the one device measurement that would close sustained-cadence behaviour.
- **No composed per-decision distribution.** The harness reports a per-run cadence and per-frame
  percentiles, not a per-decision p95. Any p95-vs-contract statement is still stage-sum arithmetic.
- The cadence artifact records no `seed`, `data_root`, or per-clip width/height. Deferred: it is
  measurement-code churn this close to the number freeze.
- Numerical parity of the 74-d token assembly against the training-time pipeline is still
  unverified **on device**, and training samples 16 frames across a clip while deployment slides a
  16-frame consecutive window — a train/deploy temporal-footprint gap, recorded but not measured.

## Unresolved questions

1. Does the exported tower graph hit Orin's FP16 tensor cores at all, or does opset-18
   `LayerNormalization` fall back? If the measured tower lands far above the 120 ms band, this is
   the first thing to check (`trtexec --verbose` layer precision dump).
2. INT8 for the tower is out of scope today — no CLIP calibration data exists. If FP16 misses the
   cadence, INT8 is the next lever, not a same-day fix.
3. The 09-29 detector number is imgsz **800**; the features were built at imgsz **640**. The
   localisation cost quoted in the thesis must state which imgsz it is, and the concurrent
   imgsz-scaling work (`finding-2026-10-01-cadence-levers.md`) is the place that reconciles them.
