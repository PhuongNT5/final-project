# Finding — two sessions on one Jetson produced a 1.8× latency spread with identical provenance

**Date** 2026-10-01, ~22:16–22:42 (device clock, +07).
**Device** Orin Nano Super 8GB `phuongnguyen-desktop`, JetPack R36.5.0, TensorRT 10.3.0, nvpmodel 2
(MAXN_SUPER) throughout.
**Severity** every R48 device latency number taken before 22:38 is withdrawn. No accuracy result and
no parity verdict is affected.
**Status** cause identified, provenance gap closed in code, measurement re-taken clean.

## What happened

Two Claude sessions were measuring on the same device at the same time. Neither knew about the
other, and nothing in the measurement harness could tell them apart afterwards.

| my run | device condition at the time | tower batch-16 p50 | YOLOv11s p50 |
|---|---|---:|---:|
| `r48-pipeline-20261001T221822` | contended: a 60 s energy sweep saturating the GPU | 41.8 ms | 21.1 ms |
| `r48-pipeline-20261001T222505` / `...222511` | idle GPU, but all clocks pinned by `jetson_clocks` | 23.5 / 23.2 ms | 11.8 / 11.7 ms |
| `r48-pipeline-20261001T223826` / `...223853` / `...223920` | **quiet, clocks restored, 51 °C, load 0.24** | **23.4 / 23.3 / 23.6 ms** | **12.6 / 13.0 / 12.7 ms** |

**1.79× on the tower between the first and last row, on the same engine, same board, same power
mode, same host commit.**

### Evidence for the contention (22:16–22:21)

The other session's own artifacts, pulled back by my rsync, date themselves:

- `energy-p3-appearance-fp16-20261001T221834.json` — `duration_sec_measured: 60.013`,
  `n_inferences: 2481`, `fps: 41.341`, `avg_power_w: 15.01`. Writing at 22:18:34 after 60 s of
  continuous inference means it started ≈22:17:34 and ran **through my entire Step 4 window
  (22:18:12–22:18:22)**.
- `energy-p3-relation-head-bin-e3-fp16-20261001T222030.json` — same shape, overlapping my Step 5
  parity run (22:19:23–22:20:19).

I had checked for busy processes at 22:11 and the device was idle (load 0.77). The sweep began after
that check, so a point-in-time check before launching is not sufficient protection.

### Evidence for the clock pin (22:22–22:36)

At 22:26 the device read `cpu0 min == cur == max == 1728000` kHz and
`gpu min == cur == max == 1020000000` Hz, with `/tmp/jc_before.conf` written by root at 22:22. The
other session's wrapper was visible in `ps`: `sudo jetson_clocks` pin, `sleep 240` settle, three
`jetson-full-measurement --decision` repeats, then `jetson_clocks --restore`. My 22:25 pair landed
inside its 240 s settle — hence an *idle* GPU at *pinned* clocks.

## Why the artifacts could not distinguish these conditions

`device_state` captured `nvpmodel -q` and `jetson_clocks --show`, and that is not enough:

1. **`nvpmodel` is blind to a clock pin.** `jetson_clocks` raises each `scaling_min_freq` to
   `scaling_max_freq` without touching the power model, so `nvpmodel -q` still reports
   `MAXN_SUPER / 2` under a full pin. Both the 41.8 ms and the 23.5 ms run recorded exactly that
   string.
2. **`jetson_clocks --show` needs root**, so `run_cmd` returned None and the field was serialised as
   `"jetson_clocks_show": null` — a silent failure that looked like ordinary off-device degradation.
3. **Contention was not recorded at all.** Orin's integrated GPU exposes no per-process accounting,
   so nothing in the JSON hinted that another process was mid-sweep.

Net effect: three runs spanning 1.8× carried byte-identical `device_state`. That is the same class of
defect as `finding-2026-10-01-engine-provenance-invalidates-0929-latency.md` — a number that cannot
be tied to a reproducible device configuration — reached by a different route.

## Fix

`deployment/scripts/jetson-clock-state-utils.py` (new, 143 lines) reads the state from sysfs, which
needs no privileges and cannot fail quietly into a plausible-looking null:

- `cpu_clocks()` / `gpu_clocks()` — per-policy and per-devfreq-node min/cur/max and governor.
- `clocks_pinned()` — `min == max` is the `jetson_clocks` signature. Returns **None** when there was
  nothing to read, never `False`; conflating "not measured" with "not pinned" is what hid this.
- `competing_gpu_processes()` — scans `/proc` for the suite's own measurement entry points.
  Name-based by necessity, and therefore a warning signal rather than a guarantee.
- `warn_if_unreliable()` — prints `[WARN] ... CONTENDED` / `[WARN] clocks are PINNED` to stderr **at
  measurement time**, when it can still change what the operator does.

Folded into the shared `device_state()` in `jetson-measure-utils.py`, so all six measurement scripts
inherit it from one edit. Covered by `tests/test-jetson-clock-state.py` (12 tests) against real
sysfs- and `/proc`-shaped files on disk, not patched readers.

Verified live: `r48-pipeline-20261001T223826.json` now carries
`quiet: true`, `clocks_pinned: {cpu: false, gpu: false, any: false}`, `competing_processes: []`,
`cpu policy0 {min 729600, cur 729600, max 1728000}`, `gpu {min 306000000, cur 918000000,
max 1020000000}`, `load_average [0.36, 1.41, 1.53]`.

## Quarantined artifacts

Withdrawn for timing, with the reason attached:

| file | withdrawn because | what still stands |
|---|---|---|
| `r48-pipeline-20261001T221822.json` | GPU contended by a concurrent 60 s energy sweep | nothing |
| `r48-pipeline-20261001T222505.json`, `...222511.json` | clocks pinned by another session, non-default state | usable only as a *declared* pinned-clock condition |
| `r48-parity-fp16-20261001T222019.json` | end-to-end latency contended | **the correctness verdict: 100 % decision agreement, 0 preprocessing mismatches, max abs delta 2.66e-03 — load-independent, and reproduced in the clean window at 22:40** |

## Consequence for earlier sessions

No device latency artifact written before this commit carries `quiet` or `clocks_pinned`, so for none
of them can contention or pinning be **excluded** from the record alone. Note 8 of the number sheet
already documents that 2026-09-29 was also a multi-session day, so this is a live concern there and
not a hypothetical.

One positive cross-check exists: the 09-29 detector figure (12.903 ms, imgsz 800, batch 1) sits
inside today's clean spread of **12.6–13.0 ms** on the same graph and configuration. That is evidence
the 09-29 detector measurement was taken quiet and unpinned. It is evidence for that one row, not a
clearance for the rest of the 09-29 set.

The 21.1 ms detector reading from 22:18, which looked like a 1.64× regression against 09-29 and was
briefly treated as a real discrepancy to explain, was contention. There is no regression.

## Operating rule for a shared device

A point-in-time idle check before launching is **not** sufficient — that is exactly the check that
passed at 22:11. What worked was a device-side guard that polls until the clocks are restored *and*
no measurement process is running, settles 90 s, measures, and re-reads the state afterwards; it
aborts rather than measuring in a dirty window. Kept at `/tmp/r48-clean-window.sh` on the device.

## Unresolved questions

1. Which condition should the thesis headline — **default DVFS** (comparable with 09-29, reflects
   the device as deployed) or **pinned clocks** (lower variance, conventional for latency reporting)?
   Default is used below pending the author's call, with the pinned pair reported as a labelled
   second condition. Note the variance difference is real and one-sided: under default DVFS the
   tower's p95 is 31–35 ms against a 23.4 ms p50, while pinned it is 24–25 ms against 23.2 ms.
2. `clocks_pinned` is inferred from `min == max`. A device that natively boots with a single
   available operating point would read as pinned. Not the case on this board (CPU 729600–1728000,
   GPU 306–1020 MHz when restored), but the inference is not universal.
