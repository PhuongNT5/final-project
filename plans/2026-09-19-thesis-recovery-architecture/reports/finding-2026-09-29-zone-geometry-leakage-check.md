# Finding 2026-09-29 — Zone-geometry features are NOT a label proxy (checked, clean)

Raised while preparing the Jetson demo video, not by a reviewer. Both checks below are negative
results — nothing needs fixing in the measured numbers — but the first one is a question an
examiner can be expected to ask, and the answer should be in the thesis rather than improvised.

## 1. The suspicion

`build_relation_tokens.build_subset` resolves zones per clip as `zones.get(vmap.get(basename))`.
If a clip's basename is absent from the view map, `zones` is `None`, and `geometry()` returns
`[0, 0, 0, 0]` for every box in that clip. Feature indices 6-9 (`iou_walkway`, `iou_restricted`,
`foot_in_walkway`, `norm_dist`) would then be identically zero.

The module-level default map makes this concrete:

```
VIEW_MAP = sweep_results/zone_prior/clip_view_map.json
```

That file has **516 entries and covers only the unsafe classes**. Cross-tabulated against the
clip tree, map membership is a *perfect* predictor of the binary label:

| | zoned | mapped, no zone file | unmapped |
|---|---:|---:|---:|
| train UNSAFE | 404 | 48 | 0 |
| train SAFE | 0 | 0 | 114 |
| test UNSAFE | 56 | 8 | 0 |
| test SAFE | 0 | 0 | 61 |

Had the binary tokens been built with this default, the four geometry features would be exactly
zero for every Safe clip and generally non-zero for Unsafe ones — a shortcut worth ~100%
separation that has nothing to do with behaviour.

## 2. What was actually used — the check

The binary task's tokens are `tokens_bin8_{train,test}.pt`. Its build log
(`sweep_results/relation_tokens/build_bin8_2026-09-16.log`, line 2) records:

```
zones for views: ['0','1','3','4','5','6','na'] | view-map entries: 691
(source=sweep_results/zone_prior/clip_view_map_691.json)
```

691 entries — the complete corpus, passed explicitly via `--view_map`. Confirmed in the artifacts:
**no clip in either split has `views = None`.**

Cross-tab of "clip has any non-zero geometry value on a real (masked-in) box" against the binary
label, read directly from the token tensors:

| split | UNSAFE non-zero / zero | SAFE non-zero / zero |
|---|---:|---:|
| train (566) | 401 / 51 (88.7%) | 90 / 24 (78.9%) |
| test (125) | 56 / 8 (87.5%) | 53 / 8 (86.9%) |

On the test split the rates are **87.5% vs 86.9%** — a 0.6 pp difference. Geometry coverage is not
label-determined. **No leakage.** The train split differs by 9.8 pp, which is a real but modest
imbalance and not a shortcut; it is worth one sentence in the limitations, not a re-run.

## 3. Residual risk (this is the actionable part)

The leaking map is still the **default** in `build_relation_tokens.py:33`. Any future rebuild that
forgets `--view_map sweep_results/zone_prior/clip_view_map_691.json` silently reintroduces the
shortcut, and the symptom — accuracy going *up* — is one nobody investigates. The docstring at
line 238 already warns ("516 unsafe clips only"), which is not the same as preventing it.
Recommended: point `VIEW_MAP` at the 691 map, or assert coverage of the clip list before building.

## 4. Second check — crop-count percentiles are computed on the wrong corpus (immaterial)

`benchmark-p3-utils.crop_count_percentiles()` reads `tokens_yolo26app_{train,test}.pt`, which hold
**516 unsafe-only clips**, not the 691-clip binary corpus. These percentiles set the crop
scenarios for every Jetson cadence measurement.

| corpus | n | p50 | p95 | p99 | cap | mean |
|---|---:|---:|---:|---:|---:|---:|
| yolo26app (used) | 516 | 54 | 106 | 119 | 128 | 57.8 |
| bin8 (correct) | 691 | **53** | **104** | 119 | 128 | 55.9 |

At ~0.7 ms per crop (appearance + extraction), the p50 row moves by ~0.7 ms. Against the current
2026-10-01 pinned base that is 310.00 -> ~309.3 ms, 3.226 -> 3.233 dec/s; against the DVFS base
365.88 -> ~365.2 ms, 2.733 -> 2.738 dec/s. (The arithmetic originally ran on the retired 331.19 ms
base and gave 3.02 -> 3.03; the magnitude of the correction is what matters and it is unchanged.)
The p99 and cap rows do not move at all. **The cadence conclusion is unchanged under either clock
condition**; disclose the corpus and move on. Do not re-run the latency suite for this.

## Unresolved

1. Should `VIEW_MAP`'s default be changed now (a one-line edit that touches a file used by
   completed experiments) or left with a louder assertion? Author's call — the measured numbers
   are unaffected either way.
2. The 9.8 pp train-split imbalance in geometry coverage (88.7% unsafe vs 78.9% safe) is
   unexplained. Most likely more Safe clips sit in views with sparser polygons, but this was not
   traced to a cause.
