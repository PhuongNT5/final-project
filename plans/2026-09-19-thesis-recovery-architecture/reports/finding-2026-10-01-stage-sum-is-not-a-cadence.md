# The stage-sum duty budget for the B1 path is refuted by the composed measurement

Status: **sections 1-3 written before the discriminator probe ran; section 4 after.** Date 2026-10-01.
Owner: this session. Not yet in the chapter; §5.6.9 draft must be rewritten against this.

## 1. What was claimed

Block 3b-note 4 of the number sheet, and the first draft of §5.6.9, presented a *stage sum* as the
deployed cadence of the R48 / B1 decision path: take the isolated per-stage benchmarks (decode
p50 per frame x stride, plus tower p50 + head p50 from `decision-latency-*.json`), add them, and
read off a duty cycle. The draft's words were "Giai ma chiem **83,6 %** ngan sach trung vi va mo
hinh chiem 16,4 %" with a "margin khoang **6,6x**" over the 1,05 decisions/s contract.

The sum it was built on: 24 x 2,048 ms decode + 23,796 ms model = **72,9 ms/decision**.

## 2. Why a sum cannot be a cadence

Three costs are structurally invisible to a sum of isolated stage benchmarks:

1. **Preprocessing.** Resize + normalise of 16 frames happens between decode and the tower. No
   isolated benchmark measures it, so it was never in the budget -- not underestimated, *absent*.
2. **State.** An isolated benchmark runs the engine back to back, dozens of times, with one fixed
   input buffer. A deployment runs it once every few hundred milliseconds. Clock state and cache
   state at the moment of the call are therefore different by construction.
3. **Contention.** Decode, preprocessing and the host-to-device copy share one memory bus with the
   GPU on an integrated SoC. A sum measures each alone.

## 3. The composed measurement, and the pre-registered discriminator

`deployment/scripts/benchmark-r48-composed-cadence.py` runs the real loop: sequential decode into a
ring buffer of `--window 48`, one decision every `--stride 24` frames sampled by the same
`np.linspace` rule the evaluation and the demo use, whole-frame CLIP -> B1, **no detector**.
Single-threaded on purpose, so its throughput is a LOWER bound and its duty an UPPER bound.

Three repeats at fixed `--seed 0` (same 24 clips every time, so the spread measures the device, not
clip-set variation), all self-labelling `quiet: true`, `clocks_pinned.any: false`, zero competing
processes.

The composed numbers refute the sum (section 4.1). That leaves one open question, and it is the
reason this finding has a probe: **the tower runs 58,6 ms in the loop against 23,386 ms isolated**.
Two causes predict that, and they have different consequences for the thesis:

- **H-droop** -- GPU clock droop. Clocks are unpinned; between decisions the GPU is idle for
  ~210 ms while the CPU decodes and preprocesses, so the governor drops the SM clock and each
  tower call pays the ramp. The GPU range on this board is 306 MHz to 1020 MHz, a 3,33x span, so a
  2,5x inflation is inside what droop alone can produce. Consequence: the inflation is a *device
  power-state* effect, removable by pinning clocks, and the isolated number is the real compute cost.
- **H-contend** -- memory-bus / CPU contention from the interleaved decode and preprocessing.
  Consequence: the inflation is intrinsic to the composed pipeline and is *not* removable by
  pinning clocks; the isolated number is then simply the wrong number to quote.

**Probe** (`--probe-idle` in the same script, no sudo, same unpinned state, 60 runs per arm):
arm A runs the tower back to back; arm B runs the identical call with an injected idle gap of
`--idle-gap-ms 210` before each call, and *nothing else happening* -- no decode, no preprocessing,
so contention is excluded by construction and only the idle-induced clock state differs.

**Pre-registered adjudication rule, fixed before the probe ran and not renegotiable:**

- **H-droop confirmed** iff arm B p50 >= 1,8 x arm A p50 **and** arm B p50 is within +-20 % of
  58,630 ms (the 3-repeat median in-loop tower p50), i.e. in [46,90 ; 70,36] ms.
- **H-droop refuted** iff arm B p50 is within +-20 % of arm A p50. H-contend then stands as the
  only remaining candidate in this design -- *stands*, not proven: this probe can refute droop, it
  cannot positively demonstrate contention.
- **Mixed** in every other case (arm B above 1,2x arm A but outside the 58,630 +-20 % band): both
  contribute, and no single cause may be attributed in the chapter.

Whatever the verdict, it does not change the cadence of section 4.1: that is a measurement of the
deployed loop as deployed, with clocks in their default unpinned state, which is how the demo runs.

## 4. Results

### 4.1 The composed cadence (3 repeats, fixed seed 0, 24 clips, 201 decisions, 5559 frames each)

All three artifacts self-label `quiet: true`, `clocks_pinned.any: false`, zero competing processes.
`r48-composed-cadence-fp16-20261001T{233159,233324,233449}.json`.

| quantity | min | **median** | max | spread |
|---|---|---|---|---|
| steady-state ms/decision (p50) | 289,339 | **289,672** | 292,834 | 1,21 % |
| steady-state ms/decision (p95) | 321,764 | **322,708** | 322,909 | 0,35 % |
| incl. window fill, ms/decision | 259,704 | **260,315** | 260,825 | 0,43 % |
| preprocess p50, ms | 160,629 | **160,765** | 160,926 | 0,18 % |
| tower p50, ms | 58,002 | **58,630** | 58,883 | 1,50 % |
| head p50, ms | 0,790 | **0,798** | 0,806 | 2,01 % |
| compute p50, ms/decision | 218,750 | **219,153** | 222,310 | 1,62 % |
| decode p50, ms/frame | 2,048 | **2,055** | 2,104 | 2,73 % |
| duty cycle (compute/wall) | 0,758 | **0,761** | 0,761 | 0,41 % |

Every row passes the 10 % reproducibility rule already used for the memory field. The deployed
cadence is **289,672 ms/decision = 3,452 decisions/s**, p95 322,708 ms.

**The sum is refuted on three counts.**

1. **Preprocessing was never in the budget.** 160,765 ms = **73,4 % of compute** per decision. It is
   CPU resize + normalise of 16 frames, and it is the single largest cost in the whole pipeline --
   larger than the ViT, larger than decode. It appears in no isolated stage benchmark.
2. **The tower costs 58,630 ms in the loop against 23,386 ms isolated -- 2,51x.** Cause adjudicated
   in 4.2.
3. **The sum understates the measured cadence by ~4x.** Recomputed honestly with the in-loop decode
   figure, 24 x 2,048 + 23,796 = 72,948 ms against a measured 289,672 ms: a factor of **3,97**.

Corrected shares, replacing "decode 83,6 % / model 16,4 %":

| | share of compute | share of the 289,672 ms interval |
|---|---|---|
| preprocess | 73,4 % | 55,5 % |
| CLIP tower | 26,5 % | 20,2 % |
| B1 head | 0,4 % | 0,3 % |
| decode (24 frames x 2,048 ms) | -- | 17,0 % |
| unattributed (loop overhead, `cap.read()` outside the timed span, ring-buffer indexing) | -- | 7,4 % |

The 1,05 decisions/s contract is still met, with **3,29x** margin -- not 6,6x. **[Retired
2026-10-03: the 1,05 contract was itself withdrawn later the same day (it comes from 25,1 fps,
11 of 1.100 clips). Against the corpus-median 1,249 dec/s contract this margin is 2,76x. The
3,29x above is kept as the record of what was reported on 10-01, not as a quotable figure.]**
Single-threaded, so
3,452 decisions/s is a LOWER bound and 76,1 % duty an UPPER bound.

### 4.2 Verdict on the probe: H-droop CONFIRMED, H-contend not needed

Three probe runs, all `quiet: true`, `clocks_pinned.any: false`
(`r48-idle-droop-probe-fp16-20261001T{234145,234242,234322}.json`):

| arm | p50 (min / **median** / max), ms | spread |
|---|---|---|
| back-to-back | 25,944 / **26,448** / 26,653 | 2,7 % |
| + 210 ms idle gap, nothing else running | 54,675 / **55,231** / 55,832 | 2,1 % |
| ratio idle / back-to-back | 2,051 / **2,088** / 2,152 | 4,8 % |

Against the rule fixed in section 3: ratio 2,088 >= 1,8 **and** 55,231 ms inside [46,90 ; 70,36].
**H-droop is confirmed, in all three repeats independently.** Idle alone -- no decode, no
preprocessing, no contention possible -- reproduces **94,2 %** of the in-loop tower cost
(55,231 / 58,630). H-contend is not required to explain anything; the residual 3,4 ms (5,8 %) is
left unattributed rather than assigned to it.

So the inflation is a **GPU power-state** effect: with clocks unpinned, ~210 ms of GPU idle per
decision lets the governor drop the SM clock, and every tower call pays the ramp. The 306-1020 MHz
range on this board is a 3,33x span, which comfortably contains a 2,09x effect.

Two consequences, and they pull in opposite directions -- both must be stated:

- The isolated 23,386 ms is **not wrong**, it is a *back-to-back* number, and the chapter must label
  it as such. It is the cost when the GPU is already at its operating point.
- It is **not the deployed cost** either, because the deployment is duty-cycled by construction: one
  decision every 24 frames, with the GPU idle in between. A number measured back-to-back cannot be
  carried into a duty-cycled pipeline.

**Arithmetic bound on what pinning clocks could buy** (a bound, not a measurement): if pinning took
the tower from 58,630 to 26,448 ms and changed nothing else, the interval would fall by at most
32,18 ms to 257,5 ms = 3,88 decisions/s, **+12,5 %**. The ceiling is low because preprocessing, a
CPU cost on a continuously busy CPU, dominates. Optimising preprocessing is worth several times more
than pinning clocks, and unlike a clock pin it is a portable software change.

### 4.3 What the probe also showed, and should be reported

The back-to-back arm reads **26,448 ms** where `decision-latency-fp16-*.json` reads 23,386 ms --
**13,1 % higher** for what is nominally the same measurement. Different harness, different input
provenance (a real decoded clip here, a synthetic buffer there), same engine and same device state.
That bounds the precision of any isolated tower figure at roughly +-13 %, and no isolated latency
number in the chapter should be quoted to more than 3 significant figures because of it.

### 4.4 A third defect, found while checking the margin: the real-time contract was built on an fps that 1 % of the corpus has

The draft derived the cadence contract as "at 25,1 fps, stride 24 gives **1,05 decisions/s**". The
corpus does not run at 25,1 fps. Metadata over all **1100** iSafetyBench clips:

| | value |
|---|---|
| median fps | **29,970** |
| mean fps | 34,245 |
| p95 fps = max | **60,000** |
| min fps | 6,000 |
| 30,0 fps | 653 clips (59,4 %) |
| **60,0 fps** | **218 clips (19,8 %)** |
| 25,0 fps | 92 clips |
| 15,0 fps | 53 clips |
| **25,1 fps** | **11 clips (1,0 %)** |

Stride 24 turns source frame rate into the contract: a source at *f* fps requires *f*/24 decisions/s
to keep up. So the contract is a distribution, not a number:

| source | contract (dec/s) | margin of the measured 3,452 dec/s |
|---|---:|---:|
| median clip (29,97 fps) | 1,249 | **2,76x** |
| mean fps (34,25) | 1,427 | 2,42x |
| **p95 = max (60,0 fps), 19,8 % of clips** | **2,500** | **1,38x** |

The contract is still met everywhere in the corpus, 60 fps clips included. But the honest headline
margin is **2,76x at the median and 1,38x at 60 fps**, from a single-threaded lower bound -- not a
single comfortable multiple. Every "real time" claim in the chapter must carry the frame rate it
applies to.

Provenance note: 25,1 fps appears **only in this session's own §5.6.9 draft**. Neither
`report/chapter-05-edge-deployment.md` nor the number sheet contains it, so nothing published is
affected and nothing in a shared file needed correcting for it.

### 4.5 The two decode figures, reconciled

5,047 ms/frame (isolated) against 2,048 ms/frame (in loop) is a 2,5x disagreement with two causes,
and the larger one is **resolution**, verified not assumed:

- `data/test`, which the isolated figure was measured on, is **125 clips, all 1920x1080** (checked
  with `CAP_PROP_FRAME_WIDTH/HEIGHT` over every file).
- the composed loop ran on `data/isafetybench` itself, whose sampled resolutions span **320x240 to
  1280x720** — mostly at or below 720p, i.e. roughly a quarter to a seventh of 1080p's pixel count.
- secondarily, the isolated figure times **8 consecutive frames** per measurement (`emit_every: 8`),
  so per-burst startup is amortised over 8 frames; the composed loop decodes **whole clips
  continuously** (5559 frames / 24 clips ~ 232 frames/clip) and amortises it over hundreds.

**2,048 ms/frame is the deployed figure on this corpus, and that qualifier is load-bearing**: decode
cost is resolution-dependent, so it is not a device constant. **Risk that must be stated, not
buried:** a 1080p camera would pay roughly 5 ms/frame, which by arithmetic (24 x 5,047 - 24 x 2,048
= +72 ms) pushes the interval to ~362 ms = 2,76 decisions/s, leaving about **1,10x** over a 60 fps
1080p contract of 2,500 decisions/s. That configuration is **not measured**. It is the single most
likely way the deployment misses real time, and the mitigation -- NVDEC, which this OpenCV build
cannot reach because it was compiled without GStreamer -- is also unmeasured.

## 5. What this changes in the write-up

- §5.6.9 draft: the Bang 5.5O stage-sum subsection is **false and must be deleted**, not softened.
  Replace with 4.1's table and the margin against the corpus-median 1,249 dec/s contract (2,76x; **not** the 3,29x figure, retired — see above). The "83,6 % / 16,4 %" split is replaced by the
  corrected-shares table.
- Number sheet Block 3b-note 4: the duty budget must be marked **superseded by measurement**, with
  the composed rows added.
- `deployment/JETSON-RUNBOOK-2026-10-01.md` open item "No composed live-pipeline measurement exists
  for B1" is **resolved**.
- Any chapter sentence quoting an isolated tower latency must say "back-to-back" and must not be
  used to derive a pipeline cadence.
- Any "real time" claim must name the frame rate it holds at (4.4). The bare "1,05 decisions/s
  contract" must not be used again.
- §5.6.9's DVFS subsection heading "DVFS makes tails, it does not shift the median" is **retracted
  for the deployed mode** and rewritten: back-to-back it makes tails; duty-cycled it shifts the
  tower median by 2,09x, the largest DVFS effect measured anywhere in the chapter.
- §5.6.9's Bang 5.5N must be labelled "back-to-back", and its conclusion "all future latency work
  belongs to the backbone" narrowed -- true among model stages, false for the pipeline, where
  preprocessing dominates.

## Unresolved questions

1. Preprocessing at 160,8 ms for 16 frames is ~10 ms/frame of CPU resize+normalise. Not profiled
   inside; the split between `cv2.resize`, the float conversion, and the normalise is unknown. A
   single `cv2.resize` to 224x224 should not cost 10 ms, so there is likely a cheap win here, but it
   is unmeasured and must not be claimed in the chapter.
2. No composed measurement with clocks pinned exists. Section 4.2's +12,5 % is arithmetic, not
   measured. Running it requires `jetson_clocks` (sudo) and would break the all-unpinned series.
3. The 7,4 % unattributed share of the interval is not broken down.
4. `--probe-runs 60` per arm; whether the droop depends on the length of the idle gap (it was fixed
   at 210 ms) is untested, so the effect is characterised at the deployed duty cycle only.
5. The composed measurement used 24 clips whose fps median is 30,0. It was not run on a 60 fps
   subset, so the 1,38x margin in 4.4 is arithmetic from a corpus-wide fps distribution against a
   cadence measured on a mixed-fps sample -- the cadence itself is fps-independent by construction
   (it is ms/decision, not frames/s), but that has not been verified by a 60 fps-only run.
6. 2,048 ms/frame is a median over mixed resolutions up to 1280x720. The decode cost per frame
   almost certainly scales with resolution; no per-resolution breakdown was computed.
