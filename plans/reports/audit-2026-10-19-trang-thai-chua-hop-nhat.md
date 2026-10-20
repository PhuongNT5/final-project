# Audit — unmerged/uncommitted state, 2026-10-19 (S3)

Read-only audit. No file touched except this report. Repo is a MOVING TARGET: `git status` grew
from the task's stated 28 paths to **38** during this audit (S1's Jetson runbook/scripts/2 ONNX
exports landed mid-session) and `plans/trang-thai-chuong-trinh.md` (the master tracker that
spawned this task) was itself edited live by other parallel streams while this ran. All findings
below are against the repo state as of this report's write time. Numbers quoted elsewhere in this
report that were independently re-derived from files on disk are marked `[ĐO]`-style only where
useful; otherwise plain prose, per instructions.

## DECISION TABLE (38 paths)

| Path | Rec | Why (one line) |
|---|---|---|
| `deployment/scripts/benchmark-r48-clip-pipeline.py` (M) | COMMIT | adds `--tower`, default unchanged, reproduces published 23.386 ms |
| `deployment/scripts/benchmark-r48-composed-cadence.py` (M) | COMMIT | adds `--tower`/`--head`, default unchanged, reproduces published 289.672 ms |
| `sweep_results/posthoc_sensitivity/asymmetric_measures.json` (M) | COMMIT | pure addition, 16 rows, 0 removed/changed lines |
| `sweep_results/posthoc_sensitivity/paired_vs_control.json` (M) | COMMIT | 16 rows added + 1 benign count-string edit (38→54), no measured value changed |
| `analyze_round3_tower_bh.py` | COMMIT-AFTER-FIX | code and output are correct; its own docstring/role-strings for EVA02 are false (see item 4) — fix text before commit, not the numbers |
| `demo/out/` (4 files, ~8.2MB) | INVESTIGATE | no gitignore rule covers it, no evidentiary role found; ask author if this is a keeper demo reel or scratch output |
| `deployment/JETSON-RUNBOOK-2026-10-19.md` | COMMIT | pre-registration doc, predictions written before measurement, outcome section deliberately empty |
| `deployment/export/p3-appearance-fp32.onnx` (4.0MB) | COMMIT | well under 100MB |
| `deployment/export/p3-detector-fp32.onnx` (36.4MB) | COMMIT | well under 100MB |
| `deployment/export/p3-relation-head-bin-e0-fp32.onnx` (1.4MB) | COMMIT | well under 100MB |
| `deployment/export/p3-relation-head-bin-e3-fp32.onnx` (101KB) | COMMIT | well under 100MB |
| `deployment/export/p3-relation-head-fp32.onnx` (1.4MB) | COMMIT | well under 100MB |
| `deployment/export/r48-b1-head-cost768-3seed-fp32.onnx` (2.7MB) | COMMIT | small; its own export script already discloses weights are random/cost-only, no accuracy claim attached |
| `deployment/export/r48-tower-clipB16-fp32.onnx` (345MB) | **BLOCKED — do not git add** | exceeds GitHub's 100MB hard limit; no `.gitignore` rule covers `deployment/export/*` subpaths yet; repo has no LFS |
| `deployment/export/r48-tower-siglip2B16-fp32.onnx` (372MB) | **BLOCKED — do not git add** | same as above |
| `deployment/scripts/export-b1-head-cost-only.py` | COMMIT | self-documenting, guards its own misuse (param-count check) |
| `deployment/scripts/export-round3-towers-onnx.py` | COMMIT | self-documenting parity-gate rationale |
| `deployment/scripts/jetson-clean-window-guard.py` | COMMIT | encodes a real lesson from a 2026-10-01 contamination incident |
| `kaggle/` (7 files; `__pycache__/*.pyc` already gitignored by root rule, no action needed) | INVESTIGATE, lean COMMIT | master tracker's own Q5 asks whether to commit since round-3 ultimately ran locally, not on Kaggle; code is still the real feature-extraction pipeline that produced the round-2/3 OOF files — recommend commit as provenance regardless of where the final run executed |
| `plans/261013-2307-bien-luan-lua-chon-va-xuat-xu-theo-gvhd/chien-luoc-thuc-nghiem-vong-3.md` | COMMIT-AFTER-FIX | core pre-registration doc, section 8 verified correct; section 7 (lines 85-88) has a stale sub-note superseded by section 8 — update before/with commit |
| `plans/reports/` (this dir; holds S1/S2/S4 outputs + this file) | defer to program lead | not S3's call alone — commit once all four parallel streams close |
| `plans/trang-thai-chuong-trinh.md` | COMMIT | master tracker, normal under `plans/`; commit once this round of S1-S4 closes |
| 16× `sweep_results/posthoc_sensitivity/round2/tower{B16,DC1B,DFN2B,EVA02,L2B,META,MOBS1,SIG2}_{arch_meanpool,control_round2}_oof.json` | COMMIT | intentionally negated by `.gitignore`'s `!sweep_results/**/*_oof.json`; these are the evidentiary OOF files behind every round-3 number |

**New hard blocker not in the task's framing:** the two new tower ONNX exports total 717MB, with
two individual files (345MB, 372MB) over GitHub's 100MB per-file cap. `deployment/export/*.onnx`
(gitignore line 34-35, no `**`) does NOT match the `export/` subdirectory, so nothing currently
hides these from `git add -A`. A rule analogous to line 132-133 (for the existing 351MB
`r48-clip-vit-b32-vision-fp32.onnx`) must be added to `.gitignore` for these two specific files
**before** any blanket `git add`, or the push will fail outright. This matches
`plans/trang-thai-chuong-trinh.md` §2 Q3's own "Chặn cứng mới" note, confirmed independently here.

**Item 1 size correction to the task's framing:** the task assumed "5 ONNX exports (~36MB each)".
Actual: 5 files, total 45.36MB — `p3-detector-fp32.onnx`=36.4MB (the only one near 36MB),
`p3-appearance-fp32.onnx`=4.0MB, `p3-relation-head-fp32.onnx`=1.4MB,
`p3-relation-head-bin-e0-fp32.onnx`=1.4MB, `p3-relation-head-bin-e3-fp32.onnx`=0.1MB. None of
these 5 need `.gitignore` changes — all under 100MB, no existing rule blocks them, they are
untracked simply because no one ran `git add`. `demo/out/` similarly untracked by omission, not
by rule. The real `.gitignore` gap is the two round-3 tower ONNX exports above, not the 5 P3 files.

## Item 2 — diff of the two modified tracked JSONs

`asymmetric_measures.json`: `grep -n '^-[^-]' <diff>` → 0 removed/changed lines. 16 new
`"config"` entries added (8 towers × `_arch_meanpool`/`_control_round2`). Purely additive.

`paired_vs_control.json`: 16 rows added (same 8×2 towers), plus exactly one changed existing
line:
```
- "multiple_comparisons": "38 rows compared to one control; no correction applied, ..."
+ "multiple_comparisons": "54 rows compared to one control; no correction applied, ..."
```
38+16=54, arithmetically consistent — a row-count metadata string, not a measured value. No
previously-published row or value was altered. **No reproducibility problem.**

## Item 3 — round-3 artifact completeness

`sweep_results/posthoc_sensitivity/round2/` has 16 OOF files: all 8 pre-registered towers
(B16, DC1B, DFN2B, EVA02, L2B, META, MOBS1, SIG2) × both `arch_meanpool` and `control_round2`.
`round3_tower_bh.json` exists (6713 bytes) and contains both families complete, 8/8 each.

**Why `round3_tower_bh.json` never showed in `git status`:** it is NOT absent/uncommitted by
oversight — it is silently caught by `.gitignore`'s blanket `sweep_results/**/*.json` rule (line
12), which is negated only for `paired_vs_control.json`, `asymmetric_measures.json`, and
`*_oof.json`. `round3_tower_bh.json` matches none of the negations, confirmed via
`git check-ignore -v`. **This is a finding in itself: the primary BH-analysis output artifact for
round 3 is currently unreachable by git and will stay that way until a negation rule is added.**
Recommend adding `!sweep_results/posthoc_sensitivity/round3_tower_bh.json` to `.gitignore`.

Section 8.1's table (strategy doc, lines 98-107) cross-checked value-by-value against
`round3_tower_bh.json`: all 8 towers, BA/margin/CI/p/verdict match exactly after rounding. Section
8.2 (lines 111-125): denominator=8 confirmed correct (not 7, a stated miscount); smallest p=0.093
vs BH threshold 0.05/8=0.00625 — nothing survives, matches `bh_rejected: false` for all 16 rows in
the JSON. Section 6 line 74 ("8/8 tháp của họ đã đo ở máy") is consistent with the full-measured
JSON but directly contradicts the EVA02 "never loaded" language still sitting in the script and
JSON's own metadata (see item 4). Section 7 (lines 85-88) has a stale sub-note claiming
`dfn2bB16, eva02B16, mobileclip_s1` "vẫn chưa xác nhận tải được" — superseded by section 8, not
removed; recommend fixing on commit.

Section 8.4's R@FAR5 claim (control-row cluster CI = [3.74, 80.93], "không đo được") is
**corroborated by two independent documents** (the strategy doc itself and
`plans/trang-thai-chuong-trinh.md` §3's "Đã xong" table, which restates the identical CI) but the
raw bootstrap artifact that produced that specific CI was not located in this audit — R@FAR5 is
not a field in `round3_tower_bh.json` (BA-only). **Cannot independently verify the number from a
primary artifact; can only report it is consistently repeated across two secondary documents.**
clipL14 88.58% figure and the AUROC/AP table (section 8.3) were not independently checked against
a raw artifact in the time available — flagged below as unresolved.

## Item 4 — `analyze_round3_tower_bh.py`

Verified against code (full file read):
- BH denominator m=8 (`len(FAMILY)` = 8 entries) — matches "corrected from 7" claim. Confirmed in
  code at line 126 (`m = len(FAMILY)`), not a hardcoded 7 anywhere.
- `asl()` (lines 84-90): `p = 2*min((boots<=0).sum(), (boots>=0).sum())/b`, floored at `1/b` via
  `max(min(p,1.0), 1.0/b)` — matches two-sided ASL formula exactly, floored at 1/B as specified.
- `paired_draws()` (lines 68-81): `np.random.default_rng(seed=0)`, `n_boot=5000`, discards a draw
  when `len(np.unique(y[idx])) < 2` (the "drop draws that lose a class" rule) — matches.
- Clustering: `groups` built from `fold_id` (line 150-152) — `fold_id` is a misleading field name;
  verified by direct inspection of `enc_isbclip_B1_lso-1100_headline-none_oof.json` that its values
  are STRING source-video ids (151 unique), not CV-fold indices. Clustering is correctly by
  SOURCE despite the name. No bug.
- Reads only OOF artifacts (`CONTROL`, `PUBLISHED`, `round2/{tag}_{suffix}_oof.json`) — confirmed
  no reference anywhere in the file to a 64/125-clip frozen test-set path. **Never touches the
  test set.**

**Discrepancy found and reportable:** the script's module docstring (lines 14-18), the
`FAMILY` dict's role-string for `towerEVA02` (line 57: `'NOT MEASURED: open_clip could not load
it'`), and the output JSON's own `multiple_comparisons.note` field (`"eva02B16 never loaded;
carried as p=1"`) are all **stale/false**. Direct inspection of
`round3_tower_bh.json` shows `primary_transformer_head.towerEVA02` and
`secondary_meanpool_head.towerEVA02` both `"measured": true` with real values (ba=83.65/84.45,
p_bootstrap=0.9336/0.7072) — the real-data code path, not the `p=1` placeholder path at line 110.
The actual OOF files for EVA02 (`round2/towerEVA02_{arch_meanpool,control_round2}_oof.json`) exist
and were confirmed to contain 1,100 real prediction rows each. **The computed numbers are real and
correct — only the self-documentation text (docstring + one dict value + one JSON metadata
string) is wrong and should be corrected before this script/output is committed.** This is a
narrative-staleness bug, not a computational one: the "NOT MEASURED" story predates whatever later
got EVA02 to load, and nobody updated the three text spots that still tell the old story.

## Item 5 — chapter-05 reconciliation

Current state of `report/vn/chuong-05-trien-khai-bien.txt` (1,353 lines, post-bf9a627):
- Fix (a) line 1234: `"...tháp CLIP là hạng mục lớn nhất của phần tính toán (53.2 %); ở 1080p,"`
  — **OUTSTANDING**, still reads 53.2%, not 52.6%. (An unrelated correct 52.6% exists at line 397
  in a different sentence — not the fix target, do not confuse.)
- Fixes (b) and (c), lines 552-554: `"...trên vòng lặp tổ hợp: ở xung nhịp ghim, độ trễ mỗi quyết
  định nằm trong 309.08–311.78 ms — **phân tán 0.49%**; dưới DVFS mặc định..."` — **BOTH
  OUTSTANDING**: "trên vòng lặp tổ hợp" not changed to "trên tổng giai đoạn"; "0.49%" not changed
  to "0.87%".

`git show --stat bf9a627` confirms the commit touched only this one chapter file (156 insertions,
142 deletions), not the archive file; its own message states "No number moves" — consistent with
all three fixes remaining unapplied, exactly as
`finding-2026-10-13-kiem-toan-so-lieu-jetson-muc-tieu-nhi-phan.md` described.

Archive file `report/vn/chuong-05-trien-khai-bien-ghim-xung-nhip.txt` (1,166 lines, untouched by
bf9a627): line 22 has the same uncorrected 53.2% (fix-a outstanding there too). Searched for the
sentence containing fix (b)/(c) — zero matches for its key phrase; **that sentence does not exist
in the archive file at all**, so fixes (b)/(c) are N/A there, not outstanding.

**Cross-confirmed by `plans/trang-thai-chuong-trinh.md` §2 Q2** (live, independent of this audit):
same three fixes listed, same "CHỜ NGƯỜI" (pending author) status. Two independent checks agree:
all three fixes are real, unapplied, and waiting on author sign-off.

## Item 6 — `report/vn/` convention compliance

Checked every file for (i) >100-character lines (Python `len()` on decoded str, not byte length)
and (ii) lines starting with `^[0-9]\.[0-9]` that are not valid headings
(`^[0-9]+(\.[0-9]+)*\.\s+\S`).

| File | Lines | Words | Long-line count | Bad-head count |
|---|---|---|---|---|
| 00-tom-tat.txt | 229 | 3983 | 0 | 0 |
| 00-trang-thai-bien-soan.txt | 303 | 3769 | 1 | 0 |
| bao-cao-tien-do-2026-09-17.txt | 315 | 4097 | 31 | 0 |
| chuong-01-mo-dau.txt | 520 | 7075 | 0 | 0 |
| chuong-02-tong-quan.txt | 674 | 8403 | 0 | 0 |
| chuong-03-phuong-phap.txt | 1527 | 19405 | 0 | 0 |
| chuong-04-chi-tiet-thuc-nghiem.txt | 1279 | 15001 | 4 | 0 |
| chuong-05-trien-khai-bien-ghim-xung-nhip.txt | 1166 | 15370 | 0 | 6 |
| chuong-05-trien-khai-bien.txt | 1353 | 18016 | 0 | 0 |
| chuong-06-danh-gia.txt | 2787 | 34816 | 121 | 0 |
| chuong-07-phan-tich-that-bai.txt | 1083 | 13642 | 0 | 0 |
| chuong-08-ket-luan.txt | 518 | 7334 | 0 | 0 |
| phu-luc-a-bang-nhat-ky.txt | 534 | 5238 | 0 | 0 |
| phu-luc-b-cau-hinh-fold.txt | 435 | 3182 | 0 | 0 |
| phu-luc-c-danh-muc-hinh.txt | 169 | 1512 | 0 | 0 |

**Long-line detail, with exemption assessment:**
- `00-trang-thai-bien-soan.txt` L43 (110 chars): a file-path + reference-doc-path sentence —
  EXEMPT (unbreakable paths).
- `bao-cao-tien-do-2026-09-17.txt`: 31 lines. Lines 103-110 (103 chars each) are pipe-delimited
  results-table rows — EXEMPT (wide data table). Lines 42, 61, 78, 94, 142, 144, 158, and the
  remaining ~16 not shown above are plain Vietnamese prose 101-106 chars — **NOT exempt, genuine
  violations.** Examples: L42 "- Gom cụm nền (k=8) trên cả 691 clip, gán view theo view chiếm đa
  số của clip không an toàn cùng cụm." (106 chars); L94 "Đã thay bằng McNemar chính xác (khi có
  dự đoán từng clip) hoặc chặn McNemar (khi chỉ có số tổng)." (102 chars).
- `chuong-04-chi-tiet-thuc-nghiem.txt`: 4 lines. L893 (113 chars) and L1014 (110 chars) are
  file-path sentences — EXEMPT. L894 (102 chars) names a `.pt` filename plus parenthetical — EXEMPT
  (field/filename). L1145 (101 chars) is prose naming two backtick field names
  (`ms_per_decision_p50_with_stream_decode`, `decisions_per_sec_p50_with_stream_decode`) —
  borderline, treat as EXEMPT (unbreakable field names per the task's own exemption wording).
- `chuong-06-danh-gia.txt`: all 121 are pipe-delimited wide comparison-table rows (sample: L225-241
  shown, each a `|`-separated row of a long reconciliation table) — **EXEMPT (wide data-table
  rows)**, confirmed by sampling; did not find a non-table long line in this file.

**Bad-heading-start detail (`chuong-05-trien-khai-bien-ghim-xung-nhip.txt`, 6 lines):** L486
("2.500 quyết định/giây..."), L527 ("3.33 lần, nên nó chứa..."), L686 ("0.158 ms trong một quyết
định..."), L921 ("4.691 W đổi con số biên..."), L1007 ("2.09 lần ở tháp CLIP..."), L1157 ("3.0
quyết định/giây trên nguồn..."). All six are **false-positive-shaped but literal violations**: each
is a wrapped continuation of a bolded mid-paragraph figure (e.g. "**2.500** quyết định/giây"),
not an actual section heading — the digit-dot-digit pattern at line-start is an artifact of text
reflow, not a numbering mistake. Per the rule as literally stated ("no line may BEGIN with..."),
these ARE violations; flagging per task instruction, but noting none is an actual stray heading
number — this file is archive-only (not submitted) so lower priority to fix.

**Cross-check against `00-trang-thai-bien-soan.txt`:** read in full (first ~40 lines shown here,
rest consistent). It documents a freeze-date (05/10), two logged post-freeze exceptions (G20 on
10/10, G21/G25/G23 on 11/10) and a third exception (NP-125-D re-expression, zero touches). It
states the gate table is "hai đạt, sáu trượt" / "tám cổng" (2 pass, 6 fail, of 8 pre-registered
thesis-wide gates — **this is a different "8" from round-3's 8-tower family**, do not conflate the
two "8"s/"two-pass-six-fail" phrasings that appear in different documents). It does not itself
enumerate per-file line/word counts, so it cannot be mechanically cross-checked against the table
above beyond the freeze-date and submitted-vs-archive distinction, which is consistent with what
was found: the archive file (`-ghim-xung-nhip` suffix) is explicitly the non-submitted twin of the
main chapter-05 file, matching its "archive-only" status used throughout this audit.

## Item 7 — test-touch ledger

`reports/test-touch-log.md` (916 lines, read in full): most recent/final budget line (dated
2026-10-11): **"Budget unchanged: 2 pre-registered, 1 consumed, 1 remaining."** The one consumed
touch is the batched G21 (ConvLSTM device control, REFUTED) + G25 (binary NP-125, FAILS) run. All
later entries in the log (G23 frame-reduction, NP-125-D re-expression, its CI-unit-fix follow-up)
explicitly state "ZERO TOUCHES CONSUMED." No entry postdates 2026-10-11 — round-2 (~10-14/15) and
round-3 (10-16) tower work is absent from this log entirely.

**Corroboration found this pass:** `plans/trang-thai-chuong-trinh.md` §4 (constraint #2) restates
the identical figure live today — "2 đăng ký trước, 1 đã dùng, 1 còn lại" — and
`report/vn/00-trang-thai-bien-soan.txt`'s own exception log (read above) independently states the
same "đúng một lần gộp cho G21 và G25... ngân sách vẫn còn một lần" and "G23 tiêu không lần nào."
Three independent documents agree on 2/1/1.

**Did round-2/3 silently consume a touch?** Grepped `analyze_round3_tower_bh.py`,
`export-round3-towers-onnx.py`, `export-b1-head-cost-only.py`, `jetson-clean-window-guard.py`,
both benchmark scripts, and all of `kaggle/*.py` for any reference to a frozen-test-set path
(`test[-_]?64`, `test[-_]?125`, `np-?125`, "frozen test"): **zero matches across every file.**
All round-2/3 tower work reads only `*_oof.json` (LSO-1100 out-of-fold) artifacts. Combined with
the Jetson runbook's own outcome section being explicitly empty ("filled in only after the
measurements; empty on purpose until then") — the device measurements have not even run yet, so
they cannot have touched anything. **Conclusion: no evidence of an unlogged touch. Confidence is
higher than last pass (now grep-confirmed across every new script, not just inferred), but still
not a 100% proof of absence** — only an absence-of-reference across the files that exist today.

## Unresolved questions

1. Section 8.3's clipL14 88.58% figure and the AUROC/AP table were not independently verified
   against a raw artifact in this pass (ran out of scope/time) — someone should re-check these
   two specific numbers against whatever artifact produced them.
2. The R@FAR5 control-row CI [3.74, 80.93] is repeated identically in two documents but its
   producing artifact was not located — is there a raw bootstrap output file for this number, or
   is it only ever hand-copied between docs?
3. `round3_tower_bh.json` is invisible to git by a blanket `.gitignore` rule with no negation —
   was this deliberate (post-hoc, not meant to be committed) or an oversight? If it should ship,
   add the negation line before committing anything else in `sweep_results/posthoc_sensitivity/`.
4. The two round-3 tower ONNX exports (clipB16 345MB, siglip2B16 372MB) cannot be committed as-is
   (over GitHub's 100MB cap, no LFS in this repo). Fix `.gitignore` or set up LFS before the next
   `git add`, or these will sit untracked indefinitely or break a push.
5. `kaggle/` commit question is still open per the master tracker's own Q5: commit as provenance,
   or drop since round-3's real run was local, not on Kaggle? This audit leans COMMIT but the call
   belongs to the program lead, not S3 alone.
6. `demo/out/` has no stated purpose found in any artifact or doc touched during this audit —
   needs a person to say whether it's a keeper or scratch.
7. The five already-known unresolved questions in
   `finding-2026-10-13-kiem-toan-so-lieu-jetson-muc-tieu-nhi-phan.md` §end remain open and are not
   re-litigated here; see that file directly.
8. `plans/trang-thai-chuong-trinh.md` and the repo's untracked-file count both changed under this
   audit while it was running (28→38 paths; S1/S2/S4 flipped to XONG mid-session). This report is
   accurate as of its write time only — re-run `git status --porcelain` before acting on the
   decision table if any time has passed.
