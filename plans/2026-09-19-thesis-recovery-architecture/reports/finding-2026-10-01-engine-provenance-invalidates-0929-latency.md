# Every engine built on/before 2026-09-29 carries a stale device record — the 09-29 latency numbers are not attributable

> **⚠ WITHDRAWN IN PART — 2026-10-03, REPLACED BY MEASUREMENT THE SAME DAY.** Every P3
> "per-decision" figure below (310.00 ms / 3.226 dec/s pinned, 365.88 / 2.733 DVFS, and the
> p95/p99/cap rows) is a **sum of separately-timed stages**, not a composed loop:
> `jetson-measure-latency.py:211` adds five stage p50s, each measured with only that stage
> running, on synthetic fixed tensors. **The +7.5 % contract margin is withdrawn.**
> **An intermediate correction is ALSO withdrawn:** the morning's arithmetic ("six costs are
> omitted, every omission can only add latency, so 3.226 dec/s is an upper bound; corrected pinned
> >= 360.87 ms = <= 2.771 dec/s, therefore P3 misses the contract in both clock conditions") is
> **false in the pinned arm**, and the sum is not a one-sided bound at all — it also
> **double-charges the detector** (16 passes per decision where the loop's stride of 8 amortises
> 8), so its bias **changes sign with the clock state**: the published sum is 7.98 % too HIGH when
> pinned and 39.3 % too LOW under DVFS.
> **The measured composed cadence (2026-10-03 17:33–17:47, both arms thermally gated, N=1,
> 480 frames, 3 test clips, FP16):** pinned **2.493–3.965 dec/s (median 3.479), MEETS on 2 of 3
> clips**; DVFS **1.141–1.511 (median 1.441), MEETS on none**. The arms are **disjoint**
> (max DVFS < min pinned), so pinning is worth **+141.4 % (2.41x)** composed against only
> **+26.4 % (1.26x)** on the stage sum. **P3's cadence is clip-dependent and holds only with the
> clocks pinned.** Quote `p3-composed-cadence-fp16-*` artifacts, never a stage sum, for any
> contract statement. Adjudication:
> `reports/finding-2026-10-03-p3-decision-latency-is-a-stage-sum.md` (§11).


Date 2026-10-01. Device: Jetson Orin Nano Super (`Orin`, cc 8.7, 8 SMs), TensorRT 10.3.0,
JetPack 6, power mode MAXN_SUPER (2). **0 test touches** — this touches no dataset.

## The warning

Loading any engine from `deployment/export/jetson/` prints:

```
[TRT] [W] Using an engine plan file across different models of devices is not recommended
          and is likely to affect performance or even cause errors.
```

TensorRT emits this when the device properties recorded in the serialized plan do not match the
device the plan is deserialized on. Its own wording — "likely to affect performance" — is the
problem: it says the kernel tactics in the plan were selected for a configuration that is not the
one being measured.

## Scope: all 30 engines, not 2

An earlier probe reported 2 warnings in a run that loaded many engines, and a follow-up probe
reported all of them. The discrepancy was logger deduplication inside a shared `trt.Runtime`.
Resolved by deserializing **one engine per OS process** (`/tmp/probe_one.py`), which cannot dedup:

| engines probed | warn | clean |
|---|---|---|
| 30 (all of `deployment/export/jetson/*.engine`, fp32 + fp16 + int8) | **30** | 0 |

So it is a property of the whole engine set, P1 and P3 alike.

## Cause: build-time device state, not the device and not the build script

Four hypotheses were tested and three were killed by evidence.

1. **Engines copied from another machine** — dead. `build-appearance-dyn.log` (2026-09-29 21:44)
   matches `p3-appearance-fp16.engine` mtime 21:44:10; the engines were built on this device.
2. **Different power mode at build vs run** — dead. Every artifact in
   `deployment/artifacts/jetson-measurements/` records `nvpmodel_q = "NV Power Mode: MAXN_SUPER / 2"`,
   including the 2026-09-29 21:27 latency run. Build and measurement were both at mode 2.
   (`/var/lib/nvpmodel/status` mtime 2026-09-30 03:27 dates an *apply*, i.e. a boot, not a change.)
3. **The build path — `pycuda.autoinit` before `tensorrt` in `jetson-build-trt-engine.py`
   vs `trtexec`** — dead. See the A/B below: the project script produces a clean engine today.
4. **Build-time device state on 2026-09-29 differed from today** — the only hypothesis left standing.

### The controlled A/B

Same ONNX (`p3-relation-head-bin-e3-fp32.onnx`), same build script
(`deployment/scripts/jetson-build-trt-engine.py --mode fp16 --model p3-relation-head-bin-e3`),
same device, same power mode, same TensorRT version. Only the build date differs:

| engine | built | on load |
|---|---|---|
| `~/engines-backup-20260929/p3-relation-head-bin-e3-fp16.engine` | 2026-09-29 21:26 | **WARNS** |
| `deployment/export/jetson/p3-relation-head-bin-e3-fp16.engine` | 2026-10-01 21:19 | **clean** |

A `trtexec`-built engine from today is clean too, so the warning is not an artifact of the Python
builder. The variable is *when*, not *how* or *where*.

Verbose load on today's device reports `Application Compute Clock Rate: 1.02 GHz` (the MAXN_SUPER
ceiling). The plan's recorded value is not exposed by the TRT 10 Python API, so the precise field
that differs is **not identified**. Most plausible: on 2026-09-29 MAXN_SUPER had been selected but
the GPU clock ceiling had not yet taken effect (no reboot), so the builder recorded the pre-switch
ceiling; the 2026-09-30 03:27 boot made it effective. Unproven, and not worth more time —
rebuilding costs 5.6 s per head.

## Consequence for the thesis

Every number below was taken with stale-provenance engines and is **not attributable to a known,
reproducible engine/device configuration**:

| artifact | what it feeds |
|---|---|
| `latency-fp16-20260929T212712.json` | §6 per-stage latency |
| `decision-latency-fp16-20260929T{2145,2214,2218,2231,2234}*.json` | §5 per-decision latency, the cadence verdict |
| `parity-20260929T221055.json` | fp16/fp32 parity |
| `memory-fp16-20261001T210938.json` | §9 memory (also used the wrong engine dir — see below) |

This is a provenance defect, not a demonstrated error: the direction and size of the effect are
unknown. It does **not** mean the cadence verdict flips — the measured compute path had
~5.4x headroom over the 3.0 decisions/s contract — but a committee question of the form
"which engine produced this number, and was it built for this device state?" currently has no
clean answer. Re-measuring on rebuilt engines replaces "unknown provenance" with one sentence of
provenance, at a cost of ~20 minutes.

## Second defect found while chasing this: wrong engine dir in the §9 memory run

`deployment/export/jetson-p3/` already exists and holds **symlinks** to the 8 P3 engines — the
per-pipeline engine directory that was listed as an open design decision. The 2026-09-29 latency
run correctly used `--engine-dir deployment/export/jetson-p3`. The 2026-10-01 memory run used the
script default `deployment/export/jetson`, so it loaded **11 fp16 engines (P1 + P3 mixed)** and its
`engines_resident_delta_mb = 417.84` is an aggregate over both pipelines, not a P3 footprint.
No code change needed: pass `--engine-dir deployment/export/jetson-p3`.

## Actions taken

- Backed up the 9 pre-existing P3 engines to `~/engines-backup-20260929/` before any rebuild.
- Rebuilt the P3 fp16 set on 2026-10-01 at MAXN_SUPER with the project script:
  `p3-relation-head-bin-e3`, `p3-relation-head`, `p3-appearance` (`--batch_profile 1:54:128`,
  the p50/cap crop counts), `p3-detector`.
- Quarantined the 5 invalid pre-fix power/memory/soak artifacts under
  `deployment/artifacts/invalid-pre-fix-20260929/` with `WHY-INVALID.md` (separate defect set:
  dead time in the energy integral, no idle baseline, `run_memory` loading no engine, soak
  baseline taken from the cold window).

## Unresolved questions as of the first pass (22:15) — all three now closed; the authoritative list is at the end

1. Which recorded device field actually differs between a 09-29 plan and a 10-01 plan. Not
   exposed by the TRT 10 Python API; would need plan-header inspection or `--profilingVerbosity`
   at build time. Low value now that rebuilding is known to be cheap.
2. Whether the fp32 and int8 engines (used only for parity and the quantisation comparison) need
   the same rebuild before their numbers are quoted. They do if quoted; fp32 parity is cheap to
   redo, int8 needs the calibration dir.
3. Whether the 09-29 numbers should be reported alongside the rebuilt ones as a stability check,
   or simply superseded. Reporting both costs nothing and pre-empts the provenance question.

   **Closed.** (1) carried forward as item 4 of the final list and downgraded. (2) carried
   forward as item 5. (3) decided: the 09-29 rows are kept in place, marked SUPERSEDED with the
   reason, rather than deleted — which is what pre-empts the provenance question.

---

# Measured impact (added 2026-10-01 22:15): the headline cadence verdict flips to FAIL — *scoped, 23:00, to the default-DVFS condition; with clocks pinned it PASSES at p50*

The rebuild was not cosmetic. Rebuilt engines are 21.54 / 2.73 / 0.99 / 0.19 MB against the
09-29 set's 21.46 / 2.86 / 0.996 / 0.19 MB — TensorRT selected **different tactics**, which is
precisely what the warning said would happen.

## Re-measured on the rebuilt set, same protocol, same `--engine-dir deployment/export/jetson-p3`

Device settled before each batch (load 0.07-0.08, 50 C). Headline = the **with-decode** condition
mandated by Unresolved 7 (`compute_p50 + decode.stream_frames_per_decision.p50`). The decode-free
rows are the labelled non-deployment control and remain under the standing quote ban.

**Table — P3 decision cadence, 3 runs per session, 24 fps / `EMIT_EVERY=8` ⇒ 3.0 dec/s required**

| Crop load | 09-29 stale engines, dec/s | verdict | 10-01 rebuilt engines, dec/s | verdict |
|---|---|---|---|---|
| p50 = 54 | 3.019 / 3.041 / 3.033 | PASS +1.1 % | **2.778 / 2.690 / 2.733** | **FAIL -8.9 %** |
| p95 = 106 | 2.708 / 2.723 / 2.718 | FAIL -9.4 % | 2.521 / 2.438 / 2.481 | FAIL -17.3 % |
| p99 = 119 | 2.642 / 2.662 / 2.650 | FAIL -11.7 % | 2.475 / 2.384 / 2.421 | FAIL -19.3 % |
| cap = 128 | 2.596 / 2.616 / 2.606 | FAIL -13.1 % | 2.435 / 2.354 / 2.380 | FAIL -20.7 % |

Artifacts: `decision-latency-fp16-20261001T{220144,220543,220925}.json` (with decode, headline)
and `decision-latency-fp16-20261001T{214412,215118,215236,215354}.json` (compute-only control,
4 runs, spread 2.6-3.4 %).

**`report/chapter-05-edge-deployment.md` Table 5.5G must change.** Its p50 row reads
`331.19 ms / 3.02 dec/s / PASS, margin 0.7 %`. On engines that match the device it is
`~366 ms / 2.73 dec/s / FAIL, -8.9 %`. The chapter's surrounding prose already refuses the
real-time claim ("on the boundary, not meets cadence"; "a 0.7 % margin is not engineering
headroom"), so the argument does not collapse — but the table, the PASS label and the
`number-sheet.md` HEADLINE row are now wrong.

## The delta is NOT purely the rebuild — and the verdict survives that confound

Both the engine provenance *and* the device state differ between 2026-09-29 and 2026-10-01, so the
session-to-session delta cannot be attributed to the rebuild alone. The pure-CPU terms, which touch
no engine, bound the environmental component:

| term | engine involved? | 09-29 | 10-01 | change |
|---|---|---|---|---|
| `postprocess_ms_per_decision` | CPU + light GPU | 47.252 | 47.843 | +1.3 % |
| `crop_extract_ms` (cv2) | **no** | 18.074 | 18.677 | +3.3 % |
| `decode.stream_ms_per_frame` (OpenCV) | **no** | 4.839 | 5.239 | **+8.3 %** |
| `detector_ms_per_frame` | yes | 12.852 | 14.246 | +10.9 % |
| `appearance_ms` @ 54 crops | yes | 20.274 | 26.638 | **+31 %** |
| `head_ms_per_decision` | yes | 0.169 | 0.282 | **+67 %** |

Engine-bearing terms moved 2-8x more than the worst pure-CPU term, so a real tactic effect is
present. But the honest reading is that **the 10-01 session is the one with clean provenance and is
therefore the one to quote**, not that "the rebuild cost 8.9 %".

**Robustness check.** Scale the 10-01 p50 median *up* by the largest pure-CPU inflation observed
(+8.3 %, i.e. assume the entire environmental penalty away): 2.733 x 1.083 = **2.96 dec/s — still
below 3.0**. The FAIL verdict is robust to the most generous possible environmental correction.

## Clocking provenance gap (new, affects both sessions)

`device_state.jetson_clocks_show` is `null` in **every** artifact — `run_cmd(['jetson_clocks',
'--show'])` fails without sudo, so the suite has never recorded the clock-pinning state. Checked
directly: `jetson_clocks` is **not** applied (CPU governor `schedutil`; GPU devfreq 306 MHz idle
against a 1020 MHz ceiling). Both sessions therefore ran under **dynamic DVFS**, which is a
legitimate deployment condition but leaves run-to-run DVFS ramp as an uncontrolled variable on a
verdict that sits within 10 % of the threshold.

## Composition at p50 (10-01, compute-only 318.1 ms) — the architectural contribution is free

| Component | ms | Share |
|---|---:|---:|
| Detector x 16 frames | 218.7 | **70.1 %** |
| Postprocess (select + geometry) | 47.8 | 15.3 % |
| Appearance (one batch of 54 crops) | 26.6 | 8.5 % |
| Crop extract (cv2, CPU) | 18.7 | 6.0 % |
| **E3 binary relation head (18,044 params)** | **0.28** | **0.09 %** |

The thesis contribution costs 0.09 % of the decision budget. The cadence miss is owned by the
detector (70 %) and the CPU postprocess (15 %), neither of which is a contribution of this work.

## Consequence for decision D2

D2 ("the deployed system is P1; stop forcing P3 to real-time") was recorded with its premise
unverified, and a pending item in this plan read *"surface the D2 framing consequence: P3 meets
cadence on the compute path, so D2's premise is measured false."* **That pending item is wrong and
is withdrawn.** On engines that match the device, P3 misses the contract at every crop load
*under the default dynamic-DVFS governor*, including the median; with clocks pinned it clears the
contract at the median only. D2's premise is **supported by measurement** under the condition a
deployment actually ships in (governor left alone). See "Clock-pinned condition" below, which was
measured after this section was first written and narrows the claim from "every crop load" to
"every crop load under DVFS".

Related correction: an earlier claim in this session that the compute path had "~5.4x headroom"
came from `throughput_decisions_per_sec_naive = 73.051`, which the suite's own docstring and
`number-sheet.md` Note 3 both flag as a per-stage figure that is *not* a decision rate. The real
per-decision margin is -8.9 % to -20.7 %.

## Section 9 memory, corrected

Re-run with `--engine-dir deployment/export/jetson-p3`:
`memory-fp16-20261001T214413.json` — `n_engines_resident = 4`,
`engines_resident_delta_mb = **251.1**`, `engines_loaded = [p3-appearance, p3-detector,
p3-relation-head, p3-relation-head-bin-e3]`. This supersedes the 417.84 MB figure, which was an
11-engine P1+P3 aggregate from the default directory. `device_used_mb` (3523.38) remains
system-wide unified memory and is not a model footprint.

## Clock-pinned condition — the rebuild did NOT cause the regression

The sections above attributed the p50 PASS -> FAIL flip to the engine rebuild. **That attribution
was wrong.** A controlled clock-pinning experiment separates the two candidate causes (engine
provenance vs DVFS state) and shows the deficit is owned by the governor.

Method: store `jetson_clocks` state to `/tmp/jc_before.conf`, apply `sudo jetson_clocks`
(CPU min=cur=max=1728000 kHz on both policies; GPU min=cur=max=1020 MHz), settle 240 s, run the
with-decode decision condition 3x on the rebuilt engines, restore, verify restore. Same engines,
same script, same crop loads, same `--clips-dir data/test` decode source as the DVFS runs.

| Crop load | DVFS dec/s (3 runs) | Pinned dec/s (3 runs) | Pinned verdict |
|---|---|---|---|
| p50 = 54 | 2.778 / 2.690 / 2.733 | **3.207 / 3.245 / 3.226** | **PASS, +7.5 %** |
| p95 = 106 | 2.521 / 2.438 / 2.481 | 2.869 / 2.899 / 2.883 | FAIL, -3.9 % |
| p99 = 119 | 2.475 / 2.384 / 2.421 | 2.796 / 2.824 / 2.810 | FAIL, -6.3 % |
| cap = 128 | 2.435 / 2.354 / 2.380 | 2.746 / 2.775 / 2.760 | FAIL, -8.0 % |

Artifacts: pinned `decision-latency-fp16-20261001T{222854,223216,223538}.json`; DVFS
`...T{220144,220543,220925}.json`. All mirrored into the repo.

### Why this clears the rebuild

Per-term medians, with the 09-29 stale-engine session for reference:

| Term | 09-29 (stale engines) | 10-01 DVFS | 10-01 pinned |
|---|---:|---:|---:|
| Detector, per frame | 12.852 | 14.246 | **11.779** |
| Postprocess | 47.252 | 47.672 | 47.352 |
| Crop extract (CPU) | n/a | 18.801 | 17.953 |
| Appearance @ 54 crops | 20.274 | 27.779 | **19.343** |
| Relation head | 0.169 | 0.284 | **0.155** |
| Decode per decision | 38.71 | 43.409 | 37.606 |

The rebuilt engines at pinned clocks are **faster than the stale engines on every GPU term**. A
rebuild that produced slower engines cannot produce this table. The DVFS column is therefore a
clocking result, not an engine result, and the earlier inference ("rebuild costs ~9 %") is
withdrawn.

Corollary about 09-29: its per-term values sit between the two 10-01 conditions and close to the
pinned one, so that session was *probably* running at or near maximum clocks. It cannot be
assigned to either condition, because `jetson_clocks_show` was null in every 09-29 artifact — the
provenance gap this report opened with. **This stays an inference, not a measurement**; the 09-29
rows are superseded rather than reinterpreted.

### Postprocess is clock-insensitive and is now the binding term

Postprocess moves 47.672 -> 47.352 ms (-0.7 %) across a clock change that moves the detector
-17.3 % and appearance -30.4 %. It is not GPU-bound and not CPU-frequency-bound at this operating
point; it is the Shapely zone geometry plus selection, single-threaded Python. At the pinned p95
row the gap to 3.0 dec/s is 13.5 ms, and postprocess is 47.4 ms — so postprocess is the only term
whose optimisation could plausibly bring p95 inside the contract. This is a concrete, scoped
future-work item, and it is **not** in the relational head.

### Measurement provenance for this condition

- Clock state is recorded in the artifact for **run 3 only** (`clocks_pinned.cpu/gpu = true`,
  CPU 1728000 fixed, GPU 1020 MHz fixed). Runs 1-2 predate the instrumentation landing (below) and
  their pinned state is attested by the session shell transcript (store -> apply -> verify ->
  restore -> verify) rather than by the artifact. The 3 pinned dec/s values agree within 1.2 %, so
  run 3 — the one with recorded state — is representative of all three.
- Contention: run 3's `competing_processes` lists exactly one entry, pid 12791, whose cmdline is
  this session's own orchestrating `bash -c` wrapper. That is a **false positive** of the
  name-based marker match (the wrapper's cmdline contains the string it is about to run), not a
  GPU consumer. No `benchmark-r48-clip-pipeline` or second `jetson-full-measurement` was resident.
- Load average at run 3: 3.49 / 2.44 / 1.84 (1/5/15 min), a rising profile consistent with this
  session's own back-to-back runs. `device_state` is captured at JSON-write time, i.e. after the
  run, so this is the measurement's own load, not an external spike.

### Two sessions shared the device tonight — contamination direction

A second session was benchmarking the R48/CLIP path on the same Orin concurrently. Timeline by
artifact write time: this session's DVFS runs 22:01-22:09, energy sweep 22:16-22:20, clock pin
applied ~22:24, pinned runs 22:28-22:35, restore ~22:36. The other session wrote
`r48-pipeline-20261001T221822` and `r48-parity-fp16-20261001T222019` **inside this session's 60 s
energy sweep**, and `r48-pipeline-20261001T{222505,222511}` **inside this session's clock pin**.

So the contamination ran this session -> the other session, not the reverse: no R48 artifact was
written inside the 22:28-22:35 pinned measurement window, and the nearest ones bracket it
(22:25 before, 22:38 after). The pinned condition stands.

Consequence for the other session's numbers: those four R48 artifacts are not a device baseline —
two were taken under GPU saturation, two under a clock pin that their own provenance block could
not see. Checked: **none of the four is cited in `report/` or in this plan directory**, so nothing
contaminated has entered the thesis. They must not be quoted without re-measurement.

### Instrumentation now in place

`deployment/scripts/jetson-clock-state-utils.py` (145 lines, sysfs-only, no privileges) landed
mid-session and is wired into `jetson-measure-utils.py:85`, so every artifact written from now on
carries `cpu_clocks`, `gpu_clocks`, `clocks_pinned`, `load_average`, `competing_processes`, and
prints a `[WARN]` at measurement time when a run is contended or pinned. Verified byte-identical
on device and in the repo. This closes the provenance gap that opened this report: the failure mode
was not that the device lied, it was that two different device states were filed as one condition.

### What the thesis should now claim

Two conditions, both labelled:

- **Default DVFS** (what a deployment ships): P3 misses 3.0 dec/s at **every** crop load,
  p50 included (2.73 dec/s, -8.9 %).
- **Clocks pinned** (`jetson_clocks`, a declarable operator action): P3 meets the contract at
  **median crop load only** (3.23 dec/s, +7.5 %) and misses at p95 / p99 / cap (-3.9 / -6.3 /
  -8.0 %).

This is a stronger result than the single-condition version it replaces, because it names the
mechanism (DVFS ramp on a 10 %-margin verdict), quantifies the operator action that recovers the
median row, and localises the residual miss to one non-contribution term (postprocess).

## Thermal soak re-run clean — verdict holds, and the DVFS penalty splits in two

The 09-29 soak §5.6.8 rested on had eight contaminated windows and ran on the engines this report
retires, so it was re-run: `thermal-soak-p3-detector-fp16-20261001T232524.json`, 23:25–23:55.

**Condition, chosen deliberately: default DVFS, not pinned.** A concurrent R48 session held the
device, and an earlier 12-minute pin of mine had already corrupted four of its artifacts; a
35-minute pin would have done it again. DVFS is also the condition a deployment ships in. The cost
is explicit: **the +7.5 % pinned margin is still not shown to survive 30 minutes.** That needs an
exclusive device window.

**No thermal throttling.** 30 windows of 60 s, 4651-4689 inferences each (140,001 total):

| quantity | value |
|---|---|
| detector p50 across windows | **12.734 – 12.866 ms**, spread **1.04 %** |
| drift vs first window / vs fastest | **0.6 %** / 0.7 % |
| junction temperature | 54.0 °C → plateau, **max 69.9 °C**, mean 68.8 |
| sustained board power | **16.98 W** mean, 18.02 W peak |
| GPU utilisation | mean 80 %, min 69 % |
| system RAM | 2,932 → 2,962 MB (**+30 MB**, no leak) |
| `possible_throttling` | **false** |

This is the first soak whose artifact records its own measurement conditions: `clocks_pinned.any`
false, `competing_processes` empty, `quiet` true. The suite now also writes two drift fields
(`latency_drift_pct_vs_first`, `..._vs_fastest`) in place of the single first-minus-last field whose
09-29 value read −7.86 % and was unquotable.

### The DVFS penalty is substantially a CPU-clock penalty

The soak separates something the two decision-latency conditions could not. Under sustained
saturation, DVFS **does** drive the GPU to its ceiling by itself — `cur_hz` 1,020,000,000 =
`max_hz` — but it leaves the CPU short of its ceiling: policy0 `cur_khz` 1,113,600 against a
`max_khz` of 1,728,000. The detector's per-frame time lands accordingly:

| regime, same rebuilt engine | detector p50 ms/frame |
|---|---:|
| in-pipeline, DVFS (median-total run) | 14.246 |
| saturated soak, DVFS | **12.796** |
| in-pipeline, clocks pinned (median-total run) | 11.779 |

Saturation alone recovers **59 %** of the DVFS-to-pinned gap; pinning recovers the remaining
**41 %**, and since the GPU was already at maximum, that residue sits on the CPU side — launch and
synchronisation overhead, which is also where the clock-insensitive postprocess term lives.

**This is a decomposition sketch, not an experiment.** The soak times the detector in a tight loop;
the pipeline times it interleaved with CPU postprocess and frame decode. The access patterns differ,
so the 59/41 split is indicative. What it does cleanly explain is why the 09-29 soak looked *faster*
(11.665–11.725 ms) than tonight's in-pipeline DVFS figures: a saturated soak is close to a pinned
condition for the GPU, so soak latencies were never comparable with pipeline latencies in the first
place.


## Unresolved questions

1. **Pinned-clock thermal cost is still unmeasured — the soak that ran was DVFS.** `jetson_clocks`
   raises idle draw and removes the race-to-idle behaviour §5.6.4 reports, so the +7.5 % margin may
   not survive 30 minutes. The DVFS soak above shows the *device* is thermally stable under
   sustained GPU saturation — 69.9 °C, 16.98 W, no throttling — which bounds the risk without
   closing it: pinning adds CPU clock on top of a GPU already at maximum. **Needs an exclusive
   device window**; not taken tonight because a concurrent session was measuring.
2. Runs 1-2 of the pinned condition have no artifact-recorded clock state. Cheap to close: one
   more pinned repeat under the new instrumentation.
3. The stale engines were never measured under pinned clocks, so "rebuilt is faster than stale"
   rests on a cross-session comparison (09-29 stale vs 10-01 pinned), not a same-condition A/B.
   The backup at `~/engines-backup-20260929/` makes the clean A/B a ~15-minute job if the claim
   needs to be load-bearing. It currently is not: the 09-29 rows are superseded either way.
4. Which exact plan field differs between the 09-29 and 10-01 engines is still unidentified (TRT 10
   Python API does not expose it). Now lower priority, since the rebuild is exonerated.
5. fp32/int8 engines have not been rebuilt and so carry the same device-record warning. Their
   numbers must not be quoted until rebuilt (fp32 cheap; int8 needs the calibration directory).
6. Postprocess has not been profiled internally (Shapely vs selection vs Python overhead), so the
   "optimise postprocess to reach p95" claim is a direction, not a costed plan.
