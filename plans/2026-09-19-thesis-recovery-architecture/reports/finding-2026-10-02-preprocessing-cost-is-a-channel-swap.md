# Finding — 90 % of the dominant stage was a colour-channel swap done at the wrong resolution

**Date:** 2026-10-02. **Severity:** high, and *recoverable* — the fix is bit-identical, so it costs
no accuracy to take. **Status: fixed in code, gated by tests, NOT yet re-measured on device**
(the Jetson has been unreachable since 2026-10-02 ~21:00: `ping` 100 % loss, ARP incomplete, i.e.
powered off, not a login problem).

## 1. What the composed measurement left open

`finding-2026-10-01-stage-sum-is-not-a-cadence.md` established that preprocessing is **55,5 % of the
decision interval** (160,8 ms per 16 frames ≈ 10 ms/frame) and that no budget in the thesis ever
contained it. That finding deliberately refused to claim a win from it:

> "A single `cv2.resize` to 224x224 should not cost 10 ms, so there is likely a cheap win here, but
> it is unmeasured and must not be claimed in the chapter."

This finding measures it.

## 2. The rule, written before the first run

From `deployment/scripts/profile-r48-preprocess.py` (committed with the rule in its docstring):

> A variant replaces the current preprocessing **ONLY IF** both hold.
> (1) **BIT-IDENTICAL** output on every frame tested — exact equality of the whole tensor, not a
> tolerance. The deployed 84,2 % was measured through the current pixels; a variant that moves any
> pixel is a different system and its latency is not comparable.
> (2) p50 at least **10 %** below the current p50, over `--repeats` independent repeats.
> Faster but not bit-identical ⇒ REJECTED, and reported as rejected with its diff.

`area_interp` is in the harness **only** to price the filter the system did not choose; it is
declared in `EXPECT_DIFF` before the run so it cannot be adopted by accident.

## 3. The cause

```
ims = [cv2.resize(f[:, :, ::-1], (224, 224)).astype('float32') / 255.0 for f in frames]
```

`f[:, :, ::-1]` is the BGR→RGB swap. It is a **negative-stride view**, so OpenCV cannot read it and
materialises a contiguous copy of the **whole frame** — 6,2 MB at 1080p — *before* downscaling to
224×224. The swap is then paid on 2 073 600 pixels instead of on 50 176.

Sub-step split of the current function, host (M3, 8 cv2 threads), 16 frames, p50:

| sub-step | 1920×1080 | 480×270 |
|---|---:|---:|
| **a. channel flip at full resolution** | **90,3 ms** | **5,5 ms** |
| b. resize (from the flipped copy) | 2,6 ms | 1,3 ms |
| c. resize **then** flip (the replacement) | 2,5 ms | 1,3 ms |
| d. `astype(float32)/255` per frame | 3,2 ms | 3,2 ms |
| e. stack + transpose | 0,5 ms | 0,5 ms |
| f. normalise `(x-mean)/std` | 7,2 ms | 7,2 ms |
| g. + the caller's redundant `.astype('float32')` | 7,4 ms | 7,4 ms |

Step **a is 89 % of the 1080p function and 35 % of the 480×270 one**, and it is the only step whose
cost scales with the *source* resolution: 15,4× the pixels, 16,5× the time.

## 4. Variants measured, both legs of the rule applied

Host, 16 frames, median of 3 repeats × 30 runs. `is_device_number: false` on these artifacts.
**The artifacts for these two runs were never saved** — see section 9; the values below were re-measured on 2026-10-03 and reproduce within noise.

| variant | 1920×1080 | speedup | 480×270 | speedup | bit-identical |
|---|---:|---:|---:|---:|---|
| `current` (baseline) | 101,6 ms | 1,00× | 15,5 ms | 1,00× | — |
| **`flip_after_resize`** | **13,6 ms** | **7,46×** | **11,4 ms** | **1,36×** | **yes — adopted** |
| `nhwc_normalise` | 15,4 ms | 6,59× | 13,5 ms | 1,15× | yes — passes the gate but is slower than the adopted variant, so not adopted |
| `area_interp` | 180,9 ms | 0,55× | 26,8 ms | 0,58× | **no**, max abs diff 1,98 — rejected by rule (1), as declared |

`nhwc_normalise` also normalises on the contiguous NHWC block instead of through the transposed
view. It is kept in the harness because the ranking is a *memory-system* property and the Jetson's
may differ from the host's; the script prints which variant it would adopt, and on device that
decision is re-taken from device numbers.

## 5. Why the replacement cannot move a pixel

`cv2.resize` forms each output pixel as a weighted sum of input pixels **within one channel**; it
never mixes channels. A channel permutation therefore commutes with it exactly. That argument is
not trusted on its own:

- `tests/test-r48-preprocess-variants.py` (11 tests) asserts `np.array_equal` — exact, no tolerance
  — between the production `preprocess` and the hard-coded **pre-change** formula at five shapes,
  including 1080×1920 and the odd 137×251.
- The profiler re-checks the same equality on **real decoded frames** at every run, on whichever
  machine it runs, so an ARM OpenCV build that disagreed with the host's would be caught on device
  rather than assumed away.

## 6. What changed in code

| file | change |
|---|---|
| `deployment/scripts/verify-r48-device-parity.py` | `preprocess`: swap channels **after** the resize. This is the function the parity gate, the composed-cadence loop and the demo all share, so one edit moves all three. |
| `build-isafetybench-clip-features.py` | `to_batch`: same change, so the host feature builder and the device path stay the same formula. Features already built are unaffected — the pixels are identical. |
| `deployment/scripts/benchmark-r48-composed-cadence.py` | `.astype('float32')` → `.astype('float32', copy=False)` at both call sites: `preprocess` already returns float32, so the copy was 9,6 MB of pure waste per decision (≈0,15 ms host). A test pins the dtype so the elision cannot silently become a conversion. |
| `deployment/scripts/profile-r48-preprocess.py`, `preprocess-variants-utils.py` | new: the harness and the candidate formulas. |
| `tests/test-r48-preprocess-variants.py` | new: the bit-exactness gate as a test. |

## 7. Predictions, registered now, before the device is re-measured

Not renegotiable after seeing the numbers.

- **P-PRE1 (parity).** The device parity run re-preprocesses the fixture clips with the NEW code and
  compares against a pixel checksum computed by the OLD code on the host. Prediction: **exact
  match** on all 24 clips. *Falsifier:* any `pixels_match` failure ⇒ the fix is reverted, not
  patched, and this finding is withdrawn.
- **P-PRE2 (iSafetyBench cadence).** Same 24 clips, `--seed 0`, 3 repeats. Preprocessing p50 drops
  by a factor in **[1,20 ; 1,55]** (host said 1,36× at this resolution class). *Falsifier:* outside
  that band ⇒ host profiling does not transfer to the device's memory system, and §5.6.9 must say
  that the split was established on host, not on device.
- **P-PRE3 (steady-state).** With preprocessing at 55,5 % of a 289,672 ms interval, a 1,36× stage
  gain predicts **247 ms ≈ 4,05 dec/s** (+17 %). Prediction: measured steady-state lands in
  **[230 ; 275] ms**. *Falsifier:* outside ⇒ some other stage moved, and the attribution in §5.6.9
  is wrong.
- **P-PRE4 (1080p).** On `data/test` (125 clips, all 1920×1080) the *stage* gain is ≥ 3×. The
  composed cadence there has no pre-fix baseline, so **no before/after claim will be made for the
  loop at 1080p** — only the stage comparison the profiler measures in one run, plus the absolute
  post-fix cadence against the 60 fps contract.

## 8. What this does NOT yet license

- **No device latency number in the thesis may be changed on the strength of this finding.** Every
  figure in §5.6.9 Bảng 5.5N/5.5O/5.5P describes the **pre-fix** system and stays as measured until
  the device re-run exists. The chapter's "chưa được đo" paragraph has been corrected to say the
  split is now profiled **on host**, the cause found, and the fix applied but not yet re-measured.
- The 1080p real-time risk (≈1,10× margin at 60 fps) is **not** closed. It is *expected* to improve
  most, since the defect scales with source resolution — which is precisely why it must be measured
  and not asserted.

## 9. Provenance repair, 2026-10-03

The 10-02 write-up quoted its numbers from console output, not from a saved artifact. Found by a
citation-vs-artifact sweep: the only `r48-preprocess-profile-*` file on the machine was
`…20261002T214004.json`, and it does **not** back this finding's tables — it is 2 repeats × 20 runs
(not 3 × 30), 1920×1080 only, and carries no 480×270 column at all. The two runs section 4 reports
left no evidence behind.

Re-measured at the protocol as stated, same host, same script, same seeded clip pick:
`r48-preprocess-profile-20261003T153439.json` (1920×1080, `2_te12.mp4`) and
`r48-preprocess-profile-20261003T153500.json` (480×270, `R4pN50yYI5k_trim_5.mp4`).

| quantity | quoted 10-02 | re-measured 10-03 | delta |
|---|---:|---:|---:|
| 1080p `current` | 101,6 ms | **101,614 ms** | exact |
| 1080p channel flip (sub-step a) | 90,3 ms | **91,408 ms** | +1,2 % |
| 1080p `flip_after_resize` | 13,6 ms | **13,326 ms** | −2,0 % |
| 1080p speedup | 7,46× | **7,62×** | +2,1 % |
| 480×270 `current` | 15,5 ms | **15,766 ms** | +1,7 % |
| 480×270 channel flip (sub-step a) | 5,5 ms | **5,495 ms** | exact |
| 480×270 `flip_after_resize` | 11,4 ms | **11,573 ms** | +1,5 % |
| 480×270 speedup | 1,36× | **1,36×** | exact |

Both legs of the pre-registered rule hold again on both artifacts: `flip_after_resize` is
`bit_identical: true` with `max_abs_diff: 0.0`, and clears the 10 % margin at both resolutions.
`area_interp` is still rejected by rule (1), with a resolution-dependent diff (1,981 at 1080p,
0,435 at 480×270) — which is itself a reason it could never have been adopted quietly.

**The 10-02 figures were correct; only their evidence was missing.** Nothing in sections 1-8 changes
as a result, and P-PRE2's band [1,20 ; 1,55] — registered before any device run and not renegotiable
— now has an artifact behind the 1,36× that motivated it, rather than a remembered console line.

**Process note.** This is the second instance in three days of a number that was true but
unverifiable (the first: six `r48-composed-cadence` / `r48-idle-droop-probe` artifacts still absent,
audit item 5). Both share one cause — a measurement read off the terminal while the artifact was
left on the device or never written. The standing rule this session adopts: **a number enters a
report only with the artifact filename beside it, checked to exist on this machine.**

## Addendum 2026-10-03 — adjudicated on device

The P-PRE block above was run on the Jetson. Verdicts: **P-PRE1 PASS**, **P-PRE2 FALSIFIED**
(3,160× against [1,20 ; 1,55]), **P-PRE3 FALSIFIED** (167,291 ms against [230 ; 275]), **P-PRE4
PASS** (8,70× against ≥ 3×). Both misses are upward. The band is **not** renegotiated.

Both falsifier-assigned causes are refuted by the same run: host profiling **did** transfer
(1,362 → 1,380× at 480×270, 7,625 → 8,700× at 1080p), and **no other stage moved** (preprocessing is
89,8 % of the 122,381 ms interval change; every other stage ≤ 1,7 %; identity closes to 0,922 ms).

**The real defect was mine, in §9's wording.** P-PRE2 called 480×270 "this resolution class" and
derived the band from it. That clip is **1 of 24 (4,2 %)** of the seeded corpus and has **7,1× fewer
pixels than the corpus median**; 54,2 % of the 24 clips are 921.600 px. A band for a mixed-resolution
corpus was read off its smallest clip. Cost is linear in frame area (8,075e-6 vs 8,559e-6 ms per
frame per pixel, 6,0 % apart over a 16× range), so the in-loop factor had to land between the two
single-clip factors, and it did.

Headline effect, protocol-identical A/B: **3,452 → 5,978 dec/s (+73,2 %)**, margin over the
**1,249 dec/s** corpus-median contract **2,76× → 4,79×** (margins against the withdrawn
1,05 dec/s contract, 3,29× → 5,69×, are retired). Full adjudication:
`finding-2026-10-03-preprocessing-fix-adjudicated-on-device.md`.

## Unresolved questions

- ~~Does the Jetson's OpenCV build agree bit-for-bit with the host's on `resize`?~~ **ANSWERED
  2026-10-03: yes.** Both `flip_after_resize` and `nhwc_normalise` are bit-identical on device at
  both resolutions (exact `np.array_equal`), and the 24-clip parity run shows **0** preprocessing
  mismatches with decision agreement 1,0.
- ~~Is `f_normalise` worth attacking, and does the NHWC trade go the other way on the Jetson?~~
  **ANSWERED 2026-10-03: no, it does not.** On device `nhwc_normalise` is 31,869 / 36,574 ms against
  `flip_after_resize` at 29,928 / 34,535 ms — NHWC loses on the Jetson too, by the same contiguity
  copy. It still clears the adoption gate (1,30× / 8,22×, bit-identical) and was **not** adopted.
- **New, from the device split:** `g_redundant_astype_float32` costs **16,142 ms** (480×270) and
  **14,346 ms** (1080p) — resolution-independent, as expected for work on the 224×224 batch, and now
  comparable to the **whole** post-fix function (34,535 ms at 1080p). `f_normalise` is similarly
  13-15 ms on device against 7,2 ms on host. Whether the deployed path actually executes a redundant
  `astype` is **not** established here — the profiler prices the sub-step in isolation. Checking that
  is the obvious next lever and is deliberately out of scope before the 10-05 freeze.
- Decode (17,0 % of the interval) is still CPU-only: OpenCV on this board has no GStreamer, so NVDEC
  remains unmeasured. After this fix, decode becomes the largest remaining non-model stage.
- The composed loop is single-threaded. Decode and preprocessing are now both CPU and both
  ~proportional to resolution; overlapping them with the GPU stage is the obvious next lever and is
  not in scope for the one-day box.
