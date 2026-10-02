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

## Unresolved questions

- Does the Jetson's OpenCV build agree bit-for-bit with the host's on `resize`? Argued and tested on
  host; the device test has not run.
- Is `f_normalise` at 7,2 ms for 2,4 M floats also a strided-access defect worth attacking? The
  NHWC variant that fixes it loses on host to the extra contiguity copy; on the Jetson the trade may
  go the other way.
- Decode (17,0 % of the interval) is still CPU-only: OpenCV on this board has no GStreamer, so NVDEC
  remains unmeasured. After this fix, decode becomes the largest remaining non-model stage.
- The composed loop is single-threaded. Decode and preprocessing are now both CPU and both
  ~proportional to resolution; overlapping them with the GPU stage is the obvious next lever and is
  not in scope for the one-day box.
