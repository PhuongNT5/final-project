# Finding — the preprocessing fix, adjudicated on device: 2 of 4 predictions falsified, both by my own mis-specification

**Date:** 2026-10-03 · **Device:** Jetson Orin Nano Super 8GB, `phuongnguyen-desktop`, JetPack R36.5.0,
TensorRT 10.3.0, MAXN_SUPER (2), `schedutil`, clocks **not** pinned, `quiet: true`,
`competing_processes: []` on all six cadence runs.
**Code measured:** `6319b21-dirty` — dirty is docstring-only, verified by AST comparison after
stripping docstrings (identical). Pre-fix baseline: `9c1e219`, confirmed an ancestor of `cc27d82`
(the fix) by `git merge-base --is-ancestor`.

## 1. Verdict

Predictions and falsifiers registered 2026-10-02 in
`finding-2026-10-02-preprocessing-cost-is-a-channel-swap.md` §9, before any device run.

| | prediction | measured | verdict |
|---|---|---|---|
| **P-PRE1** | parity exact on 24 clips | **0** preprocessing mismatches, decision agreement **100,0 %** | **PASS** |
| **P-PRE2** | prep p50 drops by a factor in **[1,20 ; 1,55]** | **3,160×** | **FALSIFIED** |
| **P-PRE3** | steady state in **[230 ; 275] ms** | **167,291 ms** | **FALSIFIED** |
| **P-PRE4** | 1080p *stage* gain ≥ **3×** | **8,70×** | **PASS** |

Both falsifications are in the direction of a **larger** win than predicted. That is not an excuse:
a band missed upward is a band missed. What follows is the diagnosis, and it is not the diagnosis the
falsifiers assigned.

## 2. The falsifiers assigned causes. Both assigned causes are refuted by the same run.

**P-PRE2's falsifier said:** *outside the band ⇒ host profiling does not transfer to the device's
memory system, and §5.6.9 must say the split was established on host, not on device.*

Refuted. At matched resolution the host predicted the device closely:

| resolution | host factor | device factor | gap |
|---|---|---|---|
| 480×270 | 1,362× | **1,380×** | 1,3 % |
| 1920×1080 | 7,625× | **8,700×** | 14,1 % |

Host profiling transferred. §5.6.9 does **not** need that caveat.

**P-PRE3's falsifier said:** *outside the band ⇒ some other stage moved, and the attribution in
§5.6.9 is wrong.*

Refuted by an accounting identity over the 122,381 ms interval change:

| term | Δ | share of the change |
|---|---|---|
| preprocess | **+109,898 ms** | **+89,8 %** |
| un-summed residual | +14,903 ms | +12,2 % |
| tower | +0,578 ms | +0,5 % |
| head | −0,044 ms | −0,0 % |
| decode (16 fr) | −2,032 ms | −1,7 % |
| **sum** | **+123,303 ms** | closes to **0,922 ms** (0,75 %) of the target |

No stage other than preprocessing contributes more than 1,7 %. The attribution in §5.6.9 stands.

**Honesty note.** decode (+6,2 %) and head (+5,5 %) each shifted by more than their within-session
run-to-run spread. I do not claim "only preprocessing moved". Within-session spread is the wrong
noise yardstick for runs taken two days apart on a DVFS-governed board. The claim made is the weaker
and sufficient one: those two terms are +0,13 ms/frame and +0,044 ms, together −1,7 % of a 122 ms
change, and cannot carry the result either way.

## 3. The actual cause: I priced a mixed-resolution corpus with a single-clip measurement

The profiler samples **one** clip via `pick_clips(data_root, 1, 0)`. For `data/isafetybench` that is
`R4pN50yYI5k_trim_5.mp4` at 480×270 = 129.600 px. The cadence loop samples **24** clips at seed 0.
Measured mix of those 24:

| resolution | px | n | share |
|---|---|---|---|
| 1280×720 | 921.600 | 12 | 50,0 % |
| 360×640 | 230.400 | 5 | 20,8 % |
| 640×360 | 230.400 | 2 | 8,3 % |
| 720×1280 | 921.600 | 1 | 4,2 % |
| 626×360 | 225.360 | 1 | 4,2 % |
| 450×360 | 162.000 | 1 | 4,2 % |
| **480×270** | **129.600** | **1** | **4,2 %** ← the clip the band came from |
| 320×240 | 76.800 | 1 | 4,2 % |

Mix re-derived **on the host** (03/10/2026) from the 24 clip names the cadence artifact records, and
`pick_clips('data/isafetybench', 24, 0)` reproduces that list **exactly** — so this table does not
depend on the device session that produced it. The profiler's single clip is
`data/isafetybench/hazard/R4pN50yYI5k_trim_5.mp4`, confirmed 480×270.

**54,2 % of the corpus sits at 921.600 px — 7,1× the pixels of the clip the band was read off.** The
band was mis-specified by construction, and the sentence in P-PRE2 that called 480×270 "this
resolution class" was simply false about the corpus.

The cost is linear in frame area, as a memcpy must be — two independent device points 16× apart in
pixels:

| resolution | px/frame | flip sub-step, 16 fr | per frame per pixel |
|---|---|---|---|
| 480×270 | 129.600 | 16,745 ms | **8,075e-6 ms/px** |
| 1920×1080 | 2.073.600 | 283,969 ms | **8,559e-6 ms/px** |

Same slope within **6,0 %** across a 16× range in frame area. The measured in-loop factor **3,160×** therefore lies where it must: between
the 480×270 factor (1,380×) and the 1080p factor (8,700×).

**Device amplifies the defect relative to host.** The swap is 94,5 % of the whole preprocessing
function at 1080p on device (283,969 of 300,506 ms) against 90,0 % on host (91,408 of 101,614 ms) —
consistent with a shared memory bus on an integrated SoC, which is the mechanism §5.6.9 already names.

## 4. What the fix does to the deployed decision path

iSafetyBench corpus, 24 clips seed 0. Protocol identity verified before comparing: `n_clips` 24,
`n_decisions` 201, `frames_decoded` 5559, `window` 48, `stride` 24, `overlapped` false,
`detector_in_loop` false — **identical** pre and post. The only change is the preprocessing code.

| | pre-fix (10-01) | post-fix (10-03) | change |
|---|---|---|---|
| decode | 2,055 ms/frame | 2,182 | +6,2 % |
| **preprocess** | **160,765 ms** | **50,867** | **−68,4 % (3,160×)** |
| tower | 58,630 ms | 58,052 | −1,0 % |
| head | 0,798 ms | 0,842 | +5,5 % |
| compute/decision | 219,153 ms | 109,114 | −50,2 % |
| **steady state p50** | **289,672 ms** | **167,291** | **−42,2 %** |
| steady state p95 | 322,708 ms | 198,224 | −38,6 % |
| duty cycle | 0,7615 | 0,6311 | — |
| **cadence** | **3,452 dec/s** | **5,978 dec/s** | **+73,2 %** |
| margin vs **1,249 dec/s** corpus-median contract | 2,76× | **4,79×** | — |

**1080p (`data/test`, 125 clips all 1920×1080, 24 sampled, 209 decisions, 5836 frames):** steady state
**227,214 ms p50**, 266,300 p95 → **4,401 dec/s**, margin **3,52×** vs the 1,249 dec/s
corpus-median contract (3,08× vs the 1,427 mean, **1,76×** vs the 2,500 p95/60 fps), duty 37,7 %. Per P-PRE4 **no
before/after loop claim is made at 1080p** — no pre-fix baseline exists there. This is the absolute
post-fix cadence only.

Use of `data/test` here is **timing-only** and does **not** consume a test touch. Touches remain **0 of 2**.

## 5. Secondary finding — part of the "stage sum is not a cadence" gap was the defect itself

`steady_state / stage_sum` fell **1,078 → 1,034**; the absolute un-summed residual fell
**20,4 → 5,5 ms**, i.e. 12,2 % of the whole interval improvement came from the residual, not from the
stage. The full-frame memcpy was itself generating memory-bus pressure that the isolated stage
benchmarks could not see.

The 10-01 finding's **direction survives** — a sum still understates the cadence, 1,034 > 1 — but its
**magnitude was inflated by the defect**. At 1080p, where frames are larger, the ratio is **1,116**
(residual 23,7 ms), so the structural gap grows with frame size. §5.6.9 must state the ratio per
resolution, not as one number.

## 6. Variant gate, re-run on device

Pre-registered rule unchanged: bit-identical **and** p50 ≥ 10 % below current.

| variant | 480×270 | 1920×1080 | bit-identical | verdict |
|---|---|---|---|---|
| current | 41,321 ms | 300,506 ms | — | baseline |
| **flip_after_resize** | **29,928 (1,38×)** | **34,535 (8,70×)** | **yes** | **ADOPTED** |
| nhwc_normalise | 31,869 (1,30×) | 36,574 (8,22×) | yes | adoptable, not adopted |
| area_interp | 40,694 (1,01×) | 188,134 (1,60×) | **no** (diff 0,435 / 1,981) | **REJECTED** as declared |

`area_interp` was declared in `EXPECT_DIFF` before the first run so it could not be adopted by
accident. It was not.

Engine parity on the same run: `decision_agreement_rate` 1,0, `max_abs_prob_diff` 2,661e-3 against
the ONNX FP32 reference. That residual is FP16 engine arithmetic, not preprocessing — the
preprocessing leg is exactly 0 mismatches.

## 7. Thesis consequences

1. **§5.6.9 headline cadence changes: 3,452 → 5,978 dec/s**, p95 198,224 ms, duty 63,1 %, margin **4,79×** vs the 1,249 dec/s corpus-median contract.
   Bảng 5.5O must be reissued; the 10-01 table becomes the pre-fix row of a before/after pair.
2. Add the 1080p row: 4,401 dec/s, margin 3,52× (median contract), flagged as absolute-only.
3. §5.6.9 keeps its attribution to preprocessing (89,8 % of the change) and **drops** nothing: the
   "established on host" caveat P-PRE2's falsifier demanded is not warranted, and the reason is
   recorded here with the matched-resolution numbers.
4. State the `steady/sum` ratio per resolution (1,034 low-res, 1,116 at 1080p), not as one number.
5. Report P-PRE2 and P-PRE3 as **falsified**, with §3's cause. A pre-registration whose misses are
   reported is worth more than one that is quietly re-banded; the band is **not** renegotiated.
6. Abstract: the B1 lane's margin over its contract moves 2,76× → **4,79×** (corpus-median
   1,249 dec/s). **Margins against the 1,05 dec/s contract — the 3,29× → 5,69× pair earlier
   drafts of this report carried — are retired**, because that contract was derived from 25,1 fps,
   which occurs in 11 of 1.100 clips (1,0 %), and is withdrawn. The withdrawn 6,6× is still
   withdrawn and this is not it.

## 8. Artifacts (all on this machine, `git add -f`'d)

```
r48-preprocess-profile-20261003T155346.json   device, 480x270,   is_device_number: true
r48-preprocess-profile-20261003T155531.json   device, 1920x1080, is_device_number: true
r48-parity-fp16-20261003T155636.json          P-PRE1, 24 clips, 0 mismatches
r48-composed-cadence-fp16-20261003T155728/155806/155843.json   post-fix, iSafetyBench
r48-composed-cadence-fp16-20261003T160245/160341/160439.json   post-fix, 1080p
r48-composed-cadence-fp16-20261001T233159/233324/233449.json   pre-fix baseline (recovered 10-03)
r48-idle-droop-probe-fp16-20261001T234145/234242/234322.json   H-droop (recovered 10-03)
```

Parity fixture `fixture.npz` mtime **2026-10-01 21:26**, fix commit `cc27d82` **2026-10-02 21:48** —
the fixture genuinely predates the fix, so P-PRE1 compared against pre-change pixels.

## 9. Unresolved questions

1. In-loop preprocessing at 1080p (37,293 ms) is **lower** than on the 720p-median corpus
   (50,867 ms), despite 2,25× the pixels. The duty cycles differ (37,7 % vs 63,1 %), so contention is
   the obvious candidate — but this is **not a controlled test** and no contention claim is made.
2. decode +6,2 % and head +5,5 % between sessions are unattributed. A same-session A/B would settle
   whether this is DVFS drift or something real. Not load-bearing for any claim.
3. `nhwc_normalise` clears the gate at both resolutions and was not adopted. It is a second ~6 %
   stage saving left on the table; out of scope before the 10-05 freeze.
4. The 1080p `steady/sum` = 1,116 means the un-summed residual grows with frame size. Whether it
   scales with pixels, with duty, or with both is unmeasured.
5. The cadence artifact records neither `seed` nor `data_root`, and `per_clip` records no
   resolution — so §3's table is not re-readable from the artifact alone. **Mitigated, not fixed:**
   the artifact does record all 24 clip names, and `pick_clips('data/isafetybench', 24, 0)`
   reproduces that list exactly from the host, so the selection is recoverable *given* the seed and
   root from the runbook. Writing `seed`, `data_root` and per-clip width/height into the artifact
   would close it properly. Applies to every cadence artifact, not just these six.
