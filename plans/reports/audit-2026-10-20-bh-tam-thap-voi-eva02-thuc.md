# Audit/fix — Q11: EVA02 "never loaded" text vs real BH numbers, 2026-10-20

Scope: `analyze_round3_tower_bh.py` + its output `sweep_results/posthoc_sensitivity/round3_tower_bh.json`.
Read-only on all `*_oof.json`. 0 training, 0 refit, 0 feature extraction, `report/vn/` untouched.

## Step 1 — reproduction (control), BEFORE any edit

Ran `.venv/bin/python3 analyze_round3_tower_bh.py` on the untouched script. Diffed the freshly
written JSON against the tracked copy: **byte-identical, 0 differences.** All 8 towers' BA, margin,
CI, p, bh_rejected, survives_preregistered_rule reproduced bit-for-bit, both families (primary
transformer-head, secondary meanpool-head).

This reproduction already contained real EVA02 numbers (ba=83.65 primary, 84.45 secondary, p=0.9336
/0.7072) — NOT p=1. Reason: `evaluate()`'s fallback (`if not os.path.exists(path): ... p=1.0`, lines
109-112) is never triggered for EVA02, because
`sweep_results/posthoc_sensitivity/round2/towerEVA02_{control_round2,arch_meanpool}_oof.json` exist
on disk (1,100 rows each, confirmed). **The computation was never broken.** This matches the prior
audit `plans/reports/audit-2026-10-19-trang-thai-chua-hop-nhat.md` item 4, independently re-verified
here with a fresh run.

## Step 2 — method used for EVA02 (same as the other 7, nothing new)

- `paired_draws()`, lines 69-82: cluster (source-video) percentile bootstrap, `n_boot=5000`,
  `np.random.default_rng(seed=0)`, draws that lose a class discarded.
- `asl()`, lines 85-91: two-sided achieved significance level,
  `p = 2*min(P(d*<=0), P(d*>=0))`, floored at `1/B`.
- Both called once per tower from `evaluate()`, lines 106-146, against the same control
  (`B1_published_config_oof.json`, ba=83.25%) and the same `groups` (source-video ids, from
  `fold_id` in the published OOF file — misleadingly named but verified string source-ids, not CV
  folds, per the prior audit).

EVA02 real values (already in the file pre-edit, reproduced here, unchanged by this fix):
- PRIMARY (transformer head, vs `control_round2`): ba=83.65, margin=+0.41pp, CI=[-4.33,+6.99],
  p=0.9336, draws=5000/5000, inside drift band (1.46pp), CI does not exclude 0.
- SECONDARY (meanpool head, vs `arch_meanpool`): ba=84.45, margin=+1.20pp, CI=[-3.54,+7.78],
  p=0.7072, draws=5000/5000, inside drift band, CI does not exclude 0.

## Step 3 — BH re-run over all 8 real p-values (already run; re-verified, not re-invented)

Threshold at rank i, m=8, q=0.05: `0.05*i/8`. Rank-1 = **0.00625**.

PRIMARY (transformer head):
| rank | tower | margin pp | p | BH threshold | reject? |
|---|---|---|---|---|---|
| 1 | metaclipB16 | +3.09 | 0.1280 | 0.00625 | no |
| 2 | siglip2B16 | +3.14 | 0.2220 | 0.01250 | no |
| 3 | clipB16 | +2.58 | 0.3196 | 0.01875 | no |
| 4 | openclipB16_l2b | +1.12 | 0.5664 | 0.02500 | no |
| 5 | openclipB16_dc1b | +1.05 | 0.5676 | 0.03125 | no |
| 6 | mobileclip_s1 | +0.47 | 0.9292 | 0.03750 | no |
| 7 | **eva02B16** | **+0.41** | **0.9336** | 0.04375 | no |
| 8 | dfn2bB16 | +0.43 | 0.9524 | 0.05000 | no |

SECONDARY (meanpool head):
| rank | tower | margin pp | p | BH threshold | reject? |
|---|---|---|---|---|---|
| 1 | siglip2B16 | +4.00 | 0.0928 | 0.00625 | no |
| 2 | openclipB16_l2b | +2.52 | 0.1496 | 0.01250 | no |
| 3 | clipB16 | +2.64 | 0.2712 | 0.01875 | no |
| 4 | metaclipB16 | +1.52 | 0.3508 | 0.02500 | no |
| 5 | dfn2bB16 | +2.25 | 0.4424 | 0.03125 | no |
| 6 | **eva02B16** | **+1.20** | **0.7072** | 0.03750 | no |
| 7 | openclipB16_dc1b | -0.38 | 0.8416 | 0.04375 | no |
| 8 | mobileclip_s1 | +0.53 | 0.9192 | 0.05000 | no |

**0 of 16 rows (8 primary + 8 secondary) reject.** EVA02's real p is nowhere near either family's
minimum (rank 7/8 primary, 6/8 secondary) so it never sets or threatens the rank-1 threshold.

## Step 4 — script edited in place, JSON regenerated

4 stale "never loaded"/"NOT MEASURED" strings found and removed:
1. Docstring point 1, old lines 14-18 ("`eva02B16` never loaded, so only seven p-values exist; the
   missing one enters BH as p = 1").
2. Comment above `FAMILY`, old line 47 ("`eva02B16` has no row (it never loaded) and is carried as
   p = 1").
3. `FAMILY['towerEVA02']` role string, old line 57 (`'NOT MEASURED: open_clip could not load it'`
   -> `'EVA-02, merged-2B data'`, matching the pretrained tag in
   `kaggle/kaggle-extract-tower-features.py` line 57: `'eva02B16': ('open_clip',
   'EVA02-B-16|merged2b_s8b_b131k')`).
4. `multiple_comparisons.note` in the written JSON, old line 167 (`'eva02B16 never loaded; carried
   as p=1'` -> `'all eight towers measured; no placeholder p-values'`).

One remaining match for "NOT MEASURED" (current line 138): a `print()` guarded by
`if not r['measured']`. It never executes today — all 8 towers have on-disk OOF files — and makes
no claim specific to EVA02; left in place as defensive handling for a future tower that genuinely
fails to load. Not part of the 4 stale strings (none of them claimed EVA02-specific non-measurement
any more after the edit above).

File is **175 lines** (cap: 200).

Regenerated `round3_tower_bh.json`. Diff, before (pre-edit, step-1 reproduction) vs after:
```
22c22
<   "note": "eva02B16 never loaded; carried as p=1"
---
>   "note": "all eight towers measured; no placeholder p-values"
146c146 (primary_transformer_head.towerEVA02.role)
<    "role": "NOT MEASURED: open_clip could not load it",
---
>    "role": "EVA-02, merged-2B data",
284c284 (secondary_meanpool_head.towerEVA02.role)
<    "role": "NOT MEASURED: open_clip could not load it",
---
>    "role": "EVA-02, merged-2B data",
```
**3 lines changed, all text/metadata. 0 numeric fields changed** (ba, margin_pp_vs_control,
cluster_ci_pp, p_bootstrap, draws, clears_drift_band, ci_excludes_zero, bh_rejected,
survives_preregistered_rule — identical for all 16 rows, before and after).

## Step 5 — does the published conclusion change?

**No.** The published conclusion — "no tower clears the BH rank-1 threshold of 0.05/8=0.00625;
smallest p=0.093 (siglip2B16, meanpool, +4.00pp)" — was already computed from a BH pass that
included EVA02's real p (0.9336/0.7072), not a fabricated p=1, because the on-disk JSON already had
the real numbers before this session touched anything (step 1). EVA02 ranks 7/8 (primary) and 6/8
(secondary) by p-value, nowhere near the minimum, so it never affects: the rank-1 threshold
(0.00625, unchanged), the smallest p (0.0928 -> "0.093", unchanged), the tower/head that sets it
(siglip2B16, meanpool, unchanged), or its margin (+4.00pp, unchanged). Every number in the published
conclusion holds exactly as stated.

## Before/after diff of every number that changed

None changed. Only the 3 text fields quoted in step 4 changed (2 role strings + 1 note string).

## Test-touch budget

0 consumed by this task. The script reads only `B1_published_config_oof.json`, the published OOF
file, and `round2/*_oof.json` — grepped for any frozen-test-set reference
(`test[-_]?64`, `test[-_]?125`, `np-?125`, "frozen test"): 0 matches, confirming (again,
independently of the 2026-10-19 audit) that this script never touches the held-out test split.
Budget unchanged: 2 pre-registered, 1 consumed (elsewhere, unrelated), 1 remaining.

## report/vn/

Not touched.

## Files changed this session

- `analyze_round3_tower_bh.py` — 4 string edits, no logic/computation change, 175 lines.
- `sweep_results/posthoc_sensitivity/round3_tower_bh.json` — regenerated by re-running the (now
  text-fixed) script; 3 text-only diffs, 0 numeric diffs, vs the pre-edit reproduction.

## Unresolved questions

1. The role phrase chosen for `towerEVA02` ("EVA-02, merged-2B data") is inferred from
   `kaggle/kaggle-extract-tower-features.py`'s pretrained tag and the pre-registration doc's "B/16,
   better data" grouping — nobody specified exact wording. Confirm before next commit if a more
   precise phrase is wanted (e.g. naming EVA-02's MIM pretraining explicitly).
2. The same stale "never loaded" / "chưa xác nhận tải được" language still sits in
   `plans/trang-thai-chuong-trinh.md` (line 62) and
   `plans/261013-2307-bien-luan-lua-chon-va-xuat-xu-theo-gvhd/chien-luoc-thuc-nghiem-vong-3.md`
   (section 7). Out of this task's scope (script + its JSON only) — flagging for `docs-manager` /
   whoever reconciles `plans/trang-thai-chuong-trinh.md`'s Q11 row, per the 2026-10-19 audit's own
   recommendation on section 7.
3. `round3_tower_bh.json` is still excluded from git by the blanket `sweep_results/**/*.json`
   `.gitignore` rule (no negation line) — noted previously, not fixed here, out of scope.
