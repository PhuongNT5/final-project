# Finding — `engines_resident_delta_mb` survives its reproducibility test; the 251,1 MB in §5.6.8 does not

**Date:** 2026-10-01. Sections 1–3 written **before** the R48 re-measurement reported; section 4
written after, against the rule fixed in section 3.
**Status:** **adjudicated — rule PASSED.** The field is reproducible under quiet conditions. One
cited number (251,1 MB) is refuted; one caveat about what the field means survives.
**Affects:** §5.6.8 of `report/chapter-05-edge-deployment.md` (quotes **251,1 MB** for the 4-engine
P3 set), and any §5.6.9 memory row for the B1 path.

## How the number is produced

`deployment/scripts/jetson-measure-power.py:200-215`:

```python
free0, total = cuda.mem_get_info()
...  # construct a TRTRunner per engine, held resident
free1, _ = cuda.mem_get_info()
result['engines_resident_delta_mb'] = round((free0 - free1) / mb, 2)
```

`cuda.mem_get_info()` on this board reports **system-wide** free unified memory — the same pool the
OS, the page cache and the desktop session use. The artifact already says so for the sibling field
(`device_used_note`: "system-wide (unified memory shared with OS), not a model footprint"), but the
*delta* inherits exactly the same contamination: anything else that allocates or frees between the
two reads is charged to, or credited against, the engines.

## The evidence that raised the defect (as read before the re-measurement)

| artifact | engines | `n_engines_resident` | `engines_resident_delta_mb` | `quiet` |
|---|---|---:|---:|---|
| `memory-fp16-20261001T214413.json` | the 4 P3 engines | 4 | **251,10** | not recorded |
| `memory-fp16-20261001T224837.json` | the **same** 4 P3 engines | 4 | **38,05** | `true` |
| `memory-fp16-20261001T224030.json` | 13 engines (P1+P3+R48) | 13 | **184,62** | `true` |
| `memory-fp16-20261001T210938.json` | 11 engines | 11 | 417,84 | not recorded |
| `memory-fp16-20261001T224929.json` | the 3 R48-path engines | 3 | 92,24 | `false` |

Two apparent failures:

1. **Same engine set, 6,6× apart** — 251,10 MB and 38,05 MB for an identical 4-engine load. A
   footprint cannot depend on the hour it was read.
2. **Not monotone in engine count** — 13 engines (184,62 MB) read *lower* than 4 engines
   (251,10 MB). A resident footprint must grow when engines are added.

Note the direction of the 22:48 reading: the quiet, state-labelled run is the **small** one. So the
251,1 MB already in the thesis is not merely uncertain, it is the figure the quiet re-read
contradicts by 213 MB.

Section 4 shows that **failure 2 is a consequence of failure 1**, not a second independent defect:
once the one un-labelled reading is set aside, the quiet readings are ordered consistently.

## Pre-registered adjudication (written before the 3 repeats report)

The memory guard now running measures the R48 engine set **3×** in one quiet, unpinned window
(`--memory --engine-dir deployment/export/jetson-r48`). Decision rule, fixed now:

- **Reproducible** iff `(max − min) ≤ 0,10 × median` across the 3 repeats. Then the median is
  quoted as the engine-resident device-memory delta of the B1 decision path, with the caveat that
  it is a system-wide free-memory delta and not a per-engine footprint.
- **Not reproducible** otherwise. Then `engines_resident_delta_mb` is declared **not a measurement**
  for *any* pipeline: no value derived from it is quoted anywhere in the thesis — the 251,1 MB in
  §5.6.8 included — and the memory claim is reduced to what is exact.

Passing the 3-repeat test would show only that the field is *stable under quiet conditions*. It
would not repair failure 2 above, which is a validity failure rather than a noise failure, so the
non-monotonicity is reported either way.

## What is exact regardless of the outcome

Engine **file sizes** are deterministic and reader-checkable: B1 class path = tower 174,022 MB +
head 1,874 MB = **175,9 MB**; adding the YOLOv11s localisation engine (21,543 MB) gives 197,4 MB.
Host peak RSS is also a real process measurement (31,4–33,9 MB across all five artifacts above, a
3-run spread far tighter than the device delta) and is quotable.

---

## 4. Adjudication (written after the run, against the rule fixed in section 3)

Three repeats of the R48 engine set in one quiet, unpinned window — all three
`quiet = true`, `competing_processes` empty, `clocks_pinned.any = false`, load 0,07–0,25,
52–53 °C, MAXN_SUPER:

| artifact | `engines_resident_delta_mb` | host peak RSS |
|---|---:|---:|
| `memory-fp16-20261001T232747.json` | 91,36 | 33,445 |
| `memory-fp16-20261001T232808.json` | 90,01 | 31,453 |
| `memory-fp16-20261001T232830.json` | 89,61 | 33,445 |

median **90,01 MB**; max − min = **1,75 MB** = **1,94 %** of the median.
Rule was `(max − min) ≤ 0,10 × median` → `1,75 ≤ 9,00` → **PASS**.

**Consequence, per the rule: the median 90,01 MB is quotable** as the engine-resident device-memory
delta of the B1 decision path (tower + head + the YOLOv11s localisation engine), measured as a
system-wide free-memory delta.

### The provisional figure was not biased by its contention

The run this replaced (`memory-fp16-20261001T224929.json`, `quiet = false`) read **92,24 MB** —
2,5 % above the clean median, inside the noise band. So the contention flag correctly marked it as
*uncitable*, but it turns out not to have been *wrong*. Recorded because the honest version of this
finding is that the guard protected against a risk that did not materialise on this particular run.

### Failure 2 retracted: the quiet readings are ordered consistently

Lining up only the readings that carry a state block against the engine **file** totals:

| engines resident | file total | `engines_resident_delta_mb` (quiet) |
|---|---:|---:|
| 4 P3 engines | 24,6 MB | 38,05 |
| 3 R48-path engines | 197,4 MB | **90,01** |
| all 13 engines | > 197,4 MB | 184,62 |

Monotone, and in the right order. The apparent non-monotonicity in section 2 was produced entirely
by the single un-labelled 251,10 MB reading. **Failure 2 is withdrawn**; failure 1 stands, and its
cause is contamination of that one reading, not an invalid metric.

### What does NOT survive

**`251,1 MB` in §5.6.8 must be replaced by `38,05 MB`.** The identical 4-engine P3 set, re-read in a
quiet state-labelled window, is 38,05 MB (`memory-fp16-20261001T224837.json`). The 251,1 MB source
(`memory-fp16-20261001T214413.json`) carries no `quiet`, no `clocks_pinned` and no
`competing_processes`, so it cannot be cleared, and it is 6,6× the quiet re-read.

### The caveat that survives the PASS

The resident delta (90,01 MB) is **less than half** the engine file total it corresponds to
(197,4 MB, dominated by the 174 MB CLIP tower plan). So the field is reproducible but it is **not**
"weights resident in device memory" and must never be labelled that way: a deserialised TensorRT
engine does not hold its plan file's byte count in device memory in a form `cuda.mem_get_info()`
attributes to the calling process. The defensible wording is *"system-wide free-memory delta
attributable to making these engines resident"*, with the engine file sizes quoted separately as the
exact, deterministic figures.

## Unresolved questions

- Is there an unprivileged per-process device-memory accounting path on L4T 5.15 that would give a
  true footprint? `cuda.mem_get_info()` demonstrably cannot.
- §5.6.8's 251,1 MB is in the other session's section. Replace it there, or raise and leave?
