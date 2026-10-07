# Finding 2026-10-01 — Recovering the P3 decision cadence

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


**Problem.** §5.6.8's headline is the weakest measured result in the thesis. *(Figures in this
paragraph restated 2026-10-01 from the re-measurement below; the 3.02 dec/s / 0.7 %-margin version
it originally carried is withdrawn with the rest of Table 5.5H.)* With clocks pinned P3 reaches
**3.226 dec/s** at the median crop load against a 3.0 dec/s contract — a **+7.5 % margin** — but
still **FAILS at p95 (2.883), p99 (2.810) and the crop cap (2.760)**; under default DVFS it fails
at **every** crop load, the median included (2.733). A result that passes only at the median, and
only with the clocks pinned, is not a deployable claim. This report measures levers to fix the
number rather than restate it.

Budget being attacked (Table 5.5H — **WITHDRAWN**, p50 = 331.19 ms/decision): detector 206.45 (62.3 %),
postprocess 47.29 (14.3 %), decode 38.71 (11.7 %), appearance 20.41 (6.2 %), crop 18.17 (5.5 %),
relation head 0.164 (0.05 %).

**The budget above is withdrawn; the base this report projects from is now the 2026-10-01 device
record.** That record is **not mine and is not restated here** — it belongs to two sibling findings,
which are the authority:

- `finding-2026-10-01-engine-provenance-invalidates-0929-latency.md` — the 09-29 engines carried a
  stale device record; rebuilt engines, 3 runs × 2 clock conditions, with-decode as the headline.
- `finding-2026-10-01-concurrent-session-contamination.md` — a concurrent session on the same device
  produced a 1.8× spread under byte-identical `device_state`, and set the clock pin (window
  22:22–22:36) that separates the two conditions.

I verified the six artifacts by opening them (`decision-latency-fp16-20261001T22{0144,0543,0925}`
= DVFS, `…{2854,3216,3538}` = pinned). Only the four terms the levers act on are carried here:

| term, per decision | 10-01 DVFS | 10-01 pinned | Table 5.5H (withdrawn) |
|---|---:|---:|---:|
| detector (×16 frames) | 223.0–230.3 | 187.6–188.6 | 206.45 |
| postprocess | 47.60–48.61 | 47.31–47.59 | 47.29 |
| live-stream decode (×8) | 41.9–45.9 | 34.5–38.4 | 38.71 |
| **p50 dec/s, with decode** | **2.69–2.78 ✗** | **3.21–3.25 ✓** | 3.02 ⚠ |

Three things the levers must be sized against. (a) **p95, p99 and the crop cap FAIL in all six
runs** — the loaded-percentile failure is not a measurement artefact and no lever here is optional.
(b) Detector spans 187.6–230.3 ms, **straddling** the 206.45 ms Lever 1 assumes, so that projection
is mid-range rather than conservative or optimistic. (c) Postprocess is unmoved by clock condition
(47.3–48.6 ms) — it is CPU-bound Shapely, which is exactly why Lever 2 exists and why its gain, not
being clock-sensitive, is the one number here that transfers cleanly between the two conditions.

Crop loads are 54/106/119/128 throughout; the adopted `bin8i640` tag is slightly lighter
(52/103/114/126), so these runs are marginally pessimistic.

---

## Lever 1 — detector input size

**Scaling law, measured.** Four ONNX exports of the same detector, host CPU, 12 runs each:

| imgsz | p50 ms (host) | ratio vs 800 | pixel ratio (s/800)² |
|---:|---:|---:|---:|
| 800 | 103.92 | 1.000 | 1.000 |
| 640 | 66.21 | 0.637 | 0.640 |
| 512 | 42.01 | 0.404 | 0.410 |
| 416 | 29.39 | 0.283 | 0.270 |

Latency tracks pixel count to within 1-5 %. That the **ratio** transfers to Jetson, while the
absolute ms do not, is not an assumption — it is checkable, and checked. The detector is fully
convolutional, so its FLOPs scale exactly with input pixels, and measured latency should follow
FLOPs wherever the kernel is compute-bound. Both sides measured on `yolo26_best.pt`, the
checkpoint actually in the P3 pipeline and in `p3-detector-fp32.onnx`:

| imgsz | GFLOPs | FLOPs ratio | pixel ratio | measured latency ratio | latency vs FLOPs |
|---:|---:|---:|---:|---:|---:|
| 800 | 35.20 | 1.0000 | 1.0000 | 1.0000 | — |
| 640 | 22.53 | 0.6400 | 0.6400 | 0.6370 | **−0.47 %** |
| 512 | 14.42 | 0.4096 | 0.4096 | 0.4040 | **−1.37 %** |
| 416 | 9.52 | 0.2704 | 0.2704 | 0.2830 | **+4.66 %** |

FLOPs ratio equals pixel ratio to four decimals, and measured latency follows FLOPs to within
1.4 % at 640 and 512. At 416 measured latency runs 4.7 % *above* the FLOPs prediction — the
signature of leaving the compute-bound regime, where fixed per-call costs (NMS, memory traffic,
kernel launch) stop being negligible. So the projection is most trustworthy exactly where it is
being used (640) and would over-promise at 416. (Host ms are an Apple-silicon laptop figure and
are not quotable as device numbers.)

**Projected cadence**, applying the ratio to the measured 206.45 ms Jetson detector cost:

| imgsz | detector ms | p50 total | p50 dec/s | p95 | p99 | cap |
|---:|---:|---:|---:|---:|---:|---:|
| 800 (current) | 206.5 | 331.2 | 3.02 ⚠ | 2.71 ✗ | 2.64 ✗ | 2.60 ✗ |
| **640** | 131.5 | 256.3 | **3.90** | **3.40** ✓ | **3.29** ✓ | **3.22** ✓ |
| 512 | 83.4 | 208.2 | 4.80 | 4.06 ✓ | 3.95 ✓ | 3.87 ✓ |

640 px converts "passes at the median only, 0.7 % margin" into "passes at every crop load with
≥ 7 % margin". **Projection, not measurement** — requires device confirmation.

This table is the Table 5.5H base (206.45 ms detector, decode excluded) and is kept for continuity.
Both counts are now superseded by the six-run re-measurement above: detector spans
187.6–230.3 ms/decision depending on clock condition, and decode belongs **in** the budget. Quote
the live-decode, per-condition version in *Combining levers* instead.

**Cost side, step 1 — entity-set agreement. WITHDRAWN, re-measurement queued.** What the relation
head consumes is the top-N_BOX=8 detections per frame, so the right cost metric is how that token
set changes, not COCO mAP. The first pass of `tools/detector-imgsz-tradeoff.py` measured that, but
on the **wrong detector and the wrong threshold**: it read `unsafenet/config.py`, which names
`detector_best.pt` (YOLO11) at conf 0.20, whereas the P3 relation-token pipeline uses
`yolo26_best.pt` at conf 0.25 (`build_relation_tokens.py:32,36`; confirmed by the `bin8_` build log).
Two different checkpoints — 19.2 MB vs 20.3 MB, distinct md5s. Its numbers (recall 0.955 at 640)
are therefore **not evidence about this pipeline** and are withdrawn rather than quoted.

The script now takes the pipeline pair explicitly and the measurement is re-queued; it is deferred
only to avoid MPS contention with the 640 token build. The direction of the earlier result is
plausible but has no standing until re-run.

*Secondary finding:* `unsafenet/config.py` disagrees with `build_relation_tokens.py` about both the
detector and the confidence threshold. Blast radius is small — the only other consumer is
`unsafenet/entities.py`, a different lane (the `cache_r2/entities` cache) — but anyone reading
`config.DETECTOR` would reasonably assume it names *the* detector. Logged as a hazard, not
silently unified: the entity-cache lane may legitimately want the other pair.

**Cost side, step 2 — end-to-end decision accuracy. MEASURED.** Entity agreement is necessary, not
sufficient: the E3 head is a DeepSets pooler over T×N tokens, so per-frame differences may wash
out — or may not. Tokens were rebuilt at 640 for all 691 clips (`--imgsz 640 --out_tag bin8i640_`,
same YOLO26 detector, same `yolo` appearance backbone, **691-entry** view map). The two arms are
paired clip-for-clip: `clip_ids`, `labels`, `views` and `feat_layout` are identical element for
element in both splits, and neither arm has a single `views = None`. Detection volume barely moves
— train 31,873 → 31,611 boxes (**−0.82 %**), test 6,774 → 6,576 (**−2.92 %**).

**Baseline, read from its artifact before the 640 run finished** (so it cannot be chosen after the
fact). `tier2_binary_logs/E3.log`, 2026-09-17: pooled 687 clips after `drop_conflict` dropped 4,
folds 85.5 / 77.5 / 81.8 / 86.1 / 79.6, **fold-mean top-1 82.1 %**, NB-corrected SE 2.49
(95 % CI 77.2-87.0), balanced accuracy 82.5 %, scene-only baseline 76.4 %, Δ = +5.7 pp (p = 0.0487).
Re-run today under tag `bin8ref_` (token files symlinked so no cited artifact is overwritten): the
new OOF is **bit-identical** to the stored one — `max |softmax diff| = 0.0`, every fold to the
decimal. So the runner is deterministic *and* has not drifted since the baseline was produced,
which is what licenses comparing a 2026-10-01 run against a 2026-09-17 one at all.

### Pre-registered decision rule (written before the 640 result existed)

Absence of a significant difference is not evidence of equivalence, so the claim is stated as
non-inferiority with a margin fixed in advance:

- **Primary statistic:** paired Δ balanced accuracy (640 − 800) on the same 687 clips, same folds,
  cluster bootstrap over the 8 scene clusters (`tools/compare-oof-runs.py`), plus exact McNemar.
- **Margin: 3 pp.** Chosen because it is smaller than the protocol's own NB-corrected half-width
  (±4.9 pp), so the claim stays inside what this protocol can resolve.
- **ADOPT 640 px** if the lower bound of the cluster-bootstrap CI on Δ is **above −3 pp**. The
  cadence result then changes from a one-sided marginal PASS to a PASS at every crop load, and the
  thesis recommends 640 px as the deployment operating point.
- **REJECT** if the lower bound falls below −3 pp. 640 px is then a trade-off, not a free win; the
  thesis publishes the accuracy-versus-cadence curve and keeps 800 px as the headline, and the
  cadence fix falls back to the geometry lever plus NVDEC.
- **Either way the 800 px cadence numbers stay in the thesis.** They are correctly measured; a
  better operating point is an addition to the record, never a replacement for it.

### Result — the rule fires ADOPT

Same runner, same flags, same corpus; only `--tag` differs (`bin8_` → `bin8i640_`). Both runs print
the same header: `dropped 4 of 691`, `pooled 687 clips (unsafe 512)`, `feat=74`, `params=18,044`,
`coord=T1`. The scene-only control is **identical to the decimal in all five folds**
(78.3/76.1/75.9/76.6/75.2, mean 76.4±1.0) — it does not read relation tokens, so an identical
control is the check that nothing but the detector input changed.

| | fold 1 | 2 | 3 | 4 | 5 | fold-mean top-1 | balanced acc | NB Δ vs scene-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 800 px | 85.5 | 77.5 | 81.8 | 86.1 | 79.6 | 82.1 % (NB SE 2.49) | 82.5 % | +5.7 pp, p = 0.0487 |
| **640 px** | 87.0 | 77.5 | 85.4 | 84.7 | 80.3 | **83.0 %** (NB SE 2.63) | **83.1 %** | **+6.6 pp, p = 0.0414** |

Paired test (`tools/compare-oof-runs.py`, 687 clips, 8 scene clusters, 5000/5000 valid draws):

- **Δ balanced accuracy (640 − 800) = +0.6 pp**, cluster-bootstrap 95 % CI **[−0.2, +1.4] pp**.
- Exact McNemar: b = 9, c = 15, 24 discordant, **p = 0.307** — 640 px is *not* significantly better.
- Clip-level CI [−1.0, +2.2] pp, reported as disclosure only; the cluster is the correct unit.

**Pre-registered rule applied verbatim: the CI lower bound on Δ is −0.2 pp, which is above the
−3 pp margin ⇒ ADOPT 640 px.** Note what is and is not claimed. Non-inferiority at a 3 pp margin
is established, and tightly — the whole CI sits inside ±1.5 pp. Superiority is **not** claimed:
McNemar p = 0.307, and the +0.6 pp point estimate is well inside noise. The honest sentence is
"640 px costs no measurable accuracy", not "640 px is better".

*Why it does not cost accuracy — hypothesis, not a finding.* Lower input resolution removes mostly
low-confidence detections, and the token set keeps only the top 8 per frame by confidence, so the
tokens the head actually consumes are largely the same objects. Consistent with the −0.82 % box
count, but not instrumented; no mechanism is claimed.

**This decision was made on screening data only.** The pooled k-fold binary OOF is classified as
screening by the pre-registered log, not a test touch; the 125-clip blind split was not opened.
Test-touch budget remains **0 of 2 spent**.

**Tooling defect found while running this.** The comparison plan called for
`--ids_from <…_oof.json>`, which crashes with `KeyError: '_row'`: `load()` only attaches `_row` to
`.pt` donors, so `_attach_ids` cannot order a JSON donor's rows. The help text says "tensor", so
JSON was never supported — the plan was wrong, not the tool. Correct donor is
`lco_oof_softmax_bin8_binary_E3_drop_conflict.pt` (n = 687, carries both `clip_ids` and `groups`).
The row-alignment guard did its job: label equality is asserted row by row before the ids are
trusted.

---

## Lever 2 — zone geometry (postprocess)

`geometry()` built a union geometry per (box, polygon) pair and recomputed polygon centroids for
every box, although zones are **static per view**. Four exact-output-preserving changes:

1. `|A ∪ B| = |A| + |B| − |A ∩ B|` — removes one constructed geometry per pair.
2. Hoist polygon area and centroid into a per-view precompute.
3. `shapely.prepared.prep()` for the foot-point containment test.
4. Bounding-box rejection before any intersection.

**Applied** (`build_relation_tokens.py:114-162`) and guarded by a regression bench that fails the
build if equivalence ever breaks. Measured, all 7 views (`geometry-postprocess-bench.json`):

- **max |reference − shipped| = 1.665e-16** over **924** box/view pairs (7 views × 132 boxes,
  including 4 degenerate/out-of-frame cases: zero-area, full-frame, and two outside the frame).
  Double-precision rounding. Not an approximation.
- **Directly measured at both loads**, no longer extrapolated per-box. Ranges span **two host runs**
  of the same bench; the artifact holds the later one:

| load | boxes w/ geometry | reference | shipped | saved | speedup |
|---|---:|---:|---:|---:|---:|
| **median decision** | **41** | **3.52–3.69 ms** | **1.79–1.81 ms** | **1.73–1.88 ms** | **1.97–2.04×** |
| worst case (all slots) | 132 | 12.46–12.49 ms | 6.52–6.78 ms | 5.68–5.97 ms | 1.84–1.92× |

Stated as ranges deliberately. The two runs differ by up to 8 % on the same load — an unloaded
MacBook Air, not a quiet measurement rig — so the third decimal of any single run is noise, and
quoting one would invite the same drift that already put this table out of step with its own
artifact once.

The median load is not a guess: the baseline token tensors carry **mean 56.3 boxes, median 53,
p95 105**, and geometry runs for classes 0/1 only — **72.5 %** of them — so 41 boxes. The **ratio
(≈ 1.8–2.0, stable across load and across runs) is the transferable quantity; the ms are host-side**
and not quotable as device figures.

**Supersedes my own two earlier write-ups of this lever.** The first quoted the 128-box bench
(17.09 → 8.94 ms) as if it were a typical decision — wrong by ~3× in load. The second corrected the
load but *scaled* the per-box cost to it (5.45 → 2.85 ms, saved 2.60). Both are now replaced by
direct measurement at the median load, which the scaling **overstated by 38–50 %**: 1.73–1.88 ms,
not 2.60. Scaling a per-box cost to a smaller load is what failed — the per-call overhead it
assumed away does not shrink with the box count.

Device carry-over, with its assumption stated. The stage benchmark behind host postprocess
19.53 ms and device 47.29 ms runs 16 × 8 = 128 boxes of which ~64 are geometry-eligible (class
drawn uniform over 4), so geometry is ≈ 5.5–6.1 ms = **28–31 %** of the host stage, and the lever
returns ≈ 2.6–2.9 ms = 13–15 % of it. At the same share on the device: **≈ 6–7 ms/decision**. The
assumption is that the Jetson's Shapely-vs-numpy cost split matches the Mac's — *unverified*; only
the stage total is measured on device. Note the device stage figure is itself a full-occupancy
number, so it already sits above a typical decision.

Worth noting where the lever lands: geometry cost scales with box count, so it pays most at the
loaded percentiles — roughly 2× more at p95 than at p50. p95 is exactly the row that currently
fails.

---

---

## Side check — do byte-identical duplicate clips leak across k-fold folds?

Raised by an anomaly while shortlisting demo clips: `0_tr7.mp4` and `1_tr29.mp4` agreed to three
decimals on five independent statistics. Independently re-hashed all 691 source `.mp4` files:
**32 byte-identical groups, 65 clips, 33 redundant.** This is **already a known and documented
finding**, not a new one — `plans/2026-09-05-…/reports/footage-verification-of-label-contradictions.md`
(2026-09-06) and `plans/2026-09-06-…/reports/laneb-2026-09-16-binary-dedupe-and-ceiling.md` record
the same 32 groups, and my independent count reproduces their taxonomy exactly: **22 benign** (same
split, same binary label), **6 leakage** (cross-split, same binary label — "flagged, not dropped by
default"), **4 poisoning** (conflicting binary label). I also confirmed the mechanism of
`drop_conflict`: the 4 clips it drops are exactly `4_te4`, `5_te13`, `4_te7`, `4_te9` — the test-side
members of the binary-conflicting groups. It dedupes on *label conflict*, not on *duplication*, so
the 6 same-label cross-split pairs survive by design.

**What is new is the fold-level check, and it comes out clean.** Using each clip's `fold_id` from the
protocol-of-record OOF: of the 29 duplicate groups that survive `drop_conflict`, **0 are split across
folds** — all 29 sit entirely inside one fold. So the pooled k-fold never trains on a clip and scores
its byte-identical twin, and the 82.1 % / 83.0 % numbers carry **no duplicate-induced cross-fold
leakage**. This is not luck: every duplicate pair shares a camera view, and folds are grouped by
scene cluster, so duplicates are structurally forced into the same fold.

Consequence for the **blind** split is unchanged from the existing audit and stays a disclosure, not
a re-run: 6 of the 125 test clips are byte-identical to a train clip with the same binary label
(`0_te7/10/11`, `1_te4/5`, `6_te11`). Post-hoc exclusion is banned by R8, so the correct treatment is
the stated bound — the blind number is an upper bound by at most 6/125 = **4.8 pp** in the worst
case where all six are correct only through memorisation. No test touch was spent establishing
this: group membership comes from file hashes and the view map, never from predictions on test.

---

## Not yet attempted

- **NVDEC.** OpenCV on the device was built without GStreamer, so hardware decode is unmeasured.
  Live-stream decode is **34.5–38.4 ms/decision** pinned, **41.9–45.9** under default DVFS — after
  the detector, the **largest remaining item**, and under DVFS the single term that turns p50 from
  3.07–3.14 (pass) into 2.69–2.78 (fail). Any gain figure would be invented; it has to be measured.
- **INT8 detector.** Parity harness cannot currently express detector output parity (§5.6.8).
- **512 px.** The projection says 4.80 dec/s, but its accuracy cost is **unmeasured** — only 640 px
  was taken through the pre-registered rule. 512 px must not be quoted as an operating point on the
  strength of the 640 px result; non-inferiority at one resolution is not non-inferiority at all of
  them. If the device measurement later shows 640 px still short of the contract, 512 px is the next
  arm to build, under the same rule and a rule written before the run.
- **Combining levers — redone on both n=3 bases, with decode counted.** The old line here
  ("640 px −75 ms + geometry −6 ms ≈ 250 ms ⇒ ~4.0 dec/s") excluded decode. Applying the measured
  0.637 detector ratio per run (saving 68.1–68.5 ms pinned, 80.9–83.6 ms under DVFS) plus geometry
  (−6.5 ms) to each run's live-decode total:

  | crop load | 640 px + geometry, **default DVFS** | 640 px + geometry, **pinned** |
  |---|---:|---:|
  | p50 | 3.55–3.67 ✓ | 4.22–4.29 ✓ |
  | p95 | 3.12–3.23 ✓ | 3.65–3.70 ✓ |
  | p99 | 3.04–3.16 ✓ | 3.54–3.58 ✓ |
  | cap | **2.99–3.09 ⚠** | 3.46–3.50 ✓ |

  Both levers together clear the contract at every crop load **under pinned clocks** (margin
  15–43 %) and at p50/p95/p99 **under default DVFS**, where the crop cap lands on the line — 2.99 at
  the slowest run, a 0.3 % miss. So the crop-cap row is the one that depends on the clock decision,
  and it is the row the thesis would most like to claim. The detector does ~91 % of the lever work;
  geometry adds ~2 %. Projection, not measurement — see Unresolved 1.

## Unresolved

1. **The one thing still missing is a 640 px device measurement.** 640 px is adopted on the accuracy
   side with a measured, pre-registered result, and 800 px is now measured n=3, but the cadence gain
   640 px buys remains a projection. The device is up and a pinned 800 px sweep ran tonight, but
   **no 640 px engine has been built** — `deployment/export/jetson-p3/` holds no `*imgsz640*`. The
   build + measure script is written (`p3-detector-imgsz640-fp16.engine`, 240 s settle, 3 repeats,
   `P3_TOKEN_TAG=bin8i640`) and **has not run — the launch was refused by the permission
   classifier, awaiting the user.**
   Until it runs, the thesis states the 640 px operating point as *adopted on accuracy grounds with
   a projected cadence*, and keeps the measured 800 px cadence as the record: with live decode
   counted, **p95/p99/cap fail under both clock conditions (2.35–2.90), and p50 passes only
   pinned (3.21–3.25 vs 2.69–2.78)**.
2. ~~If 640 px costs real binary accuracy…~~ **RESOLVED.** It does not: Δ = +0.6 pp, CI
   [−0.2, +1.4] pp, rule fires ADOPT. There is no trade-off curve to publish — the thesis reports a
   single non-inferiority result at a 3 pp margin and the cadence benefit that motivated it.
3. ~~Whether to apply the geometry optimisation to `build_relation_tokens.py` at all.~~
   **RESOLVED — applied.** The condition I set for myself (equivalence check over all seven views
   before touching the file) is met and now runs as a regression guard: 1.665e-16 over 924 pairs,
   `SystemExit` above 1e-12. Being exact-equivalent it needed no token rebuild and invalidated no
   artifact. It remains a ~2 % lever — kept for the loaded percentiles, where the cap sits on the
   line, not because it rescues anything on its own.
4. ~~`prep_zones` cannot be cached by mutating the zone dict…~~ **SATISFIED.** The shipped
   precompute is a module-level `_ZONE_PRE` keyed by `id(zones)`
   (`build_relation_tokens.py:118-127`); the zone dict is untouched, so the demo overlay at
   `jetson-p3-demo-video.py:103` still iterates only polygons.
5. **Default DVFS or pinned clocks for the headline?** Not mine to decide and **not a new question**
   — it is Unresolved 1 of `finding-2026-10-01-concurrent-session-contamination.md`, raised there
   from the same device evidence. Recorded here because of what it costs *this* report: it decides
   whether 640 px + geometry clears the crop cap with 15 % margin or misses it by 0.3 %. My
   recommendation is default DVFS for the headline with pinned as a labelled second condition, which
   is also the direction that cannot be accused of cherry-picking.
6. **Does the crop cap clear?** It is the only row whose verdict flips with the clock decision
   (2.99–3.09 DVFS vs 3.46–3.50 pinned), and it is the row that distinguishes "meets the contract at
   every crop load" from "meets it at the median". It cannot be settled by projection: it needs the
   640 px engine of item 1, measured under whichever condition item 5 picks.
7. **`clocks_pinned['any']` conflates "not pinned" with "not measured", in the one field its own
   docstring says to filter runs on.** `_all_pinned()` is careful — it returns `None` when there was
   nothing to read, and the sibling finding states plainly that conflating the two "is what hid
   this". But `clocks_pinned()` then aggregates with `'any': bool(c) or bool(g)`
   (`jetson-clock-state-utils.py:84`), so `cpu=None, gpu=None` → `any=False`, i.e. an unreadable
   device reports the same flag value as a verifiably-unpinned one. Reproduced on the Mac:
   `{'cpu': None, 'gpu': None, 'any': False}`. Filtering a set of artifacts on `any == False` would
   therefore admit exactly the unlabelled runs the guard exists to exclude. The file belongs to the
   concurrent session and is already committed (`2f57477`), so I have not edited it; my own sweep
   writer compares the whole `clocks_pinned` dict rather than `any`, so it is not exposed. Suggested
   fix there: `'any': None if c is None and g is None else bool(c) or bool(g)`.
