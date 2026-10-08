# Phase 08 — Vietnamese Thesis (.txt)

## Context Links

- Source material (read-only): `report/*.md` — 34,920 words, 100% English.
- `docs/thesis-number-sheet.md` (phase 07) — **the only permitted source of numbers.**
- `plans/2026-09-06-thesis-2week-final/reports/audit-2026-09-14-thesis-document-readiness.md`
- `plans/2026-09-06-thesis-2week-final/reports/review-2026-09-13-thesis-alignment.md`

## Overview

- **Priority:** P1 (largest single effort). **Status:** pending. **Effort:** 30h.
- Vietnamese thesis, **plain `.txt`, one file per chapter**. The English thesis is **dropped entirely** as a
  deliverable — no EN/VN consistency work, no docx, no template, no ToC automation, no styling.
  The user formats to the school template personally.

## Key Insights

- The reframe is **not a translation job**: the objective changes from 8-way/4-class to **binary**, so the
  problem statement, objectives, experiments and conclusions chapters are rewritten, not rendered.
- Existing 4-class results survive as **component/sub-class evidence** (C5) — they move from "the result"
  to "diagnostic evidence supporting the binary result".
- The EN source contains claims that are **withdrawn**; translating faithfully would resurrect them (R14).
  The strike-list below is mandatory.
- `report/thesis-reframe-deployment-led.md` (1348 w) is **superseded** — do not translate it.
- There is no `thesis-assembled.md`; chapters are standalone.

## Requirements

Output: `report/vn/` — `tom-tat.txt`, `chuong-01-gioi-thieu.txt`, `chuong-02-tong-quan.txt`,
`chuong-03-phuong-phap.txt`, `chuong-04-thuc-nghiem.txt`, `chuong-05-ket-qua-va-danh-gia.txt`,
`chuong-06-ket-luan.txt`, `tai-lieu-tham-khao.txt`, `phu-luc.txt`.
Plain UTF-8 text, no Markdown syntax, no tables-as-pipes where a plain list reads better.

## Architecture — chapter mapping

| VN file | Source | Rewrite depth |
|---|---|---|
| `tom-tat.txt` | `abstract.md` | full rewrite (binary objective; **fix the unmeasured YOLO26 INT8 claim at `abstract.md:36`**) |
| `chuong-01` | `chapter-01` | full rewrite (binary problem statement) |
| `chuong-02` | `chapter-02` | translate + add binary-task related work |
| `chuong-03` | `chapter-03` + encoder redesign | substantial new content (phase 02) |
| `chuong-04` | `chapter-04`, `-04-section-kfold`, `chapter-07` | rewrite around the phase-06 protocol suite |
| `chuong-05` | `chapter-05`, `chapter-06` | rewrite: binary headline + Jetson metrics |
| `chuong-06` | `chapter-06`/`-07` | full rewrite (conclusions + **limitations**) |

## Mandatory strike-list (do NOT carry into VN)

1. "8-class ~78%" — **unsourced**; appears at `chapter-01.md:105` and `chapter-06.md:227-228`; probably a
   miscopy of `chapter-04.md:1113` ("78% of clips collapse onto CarryOverload"). Delete.
2. `abstract.md:36` — claims a YOLO26 Jetson TRT-INT8 characterisation that was **never measured**. Replace
   with the phase-05 measured numbers or delete.
3. Bare `56/64` at `chapter-05.md:36, 205/208, 381, 700-701` — must be hedged (blind = 54/64; the 56/64 and
   "57/64 calibrated" variants are withdrawn/superseded).
4. "+60.3pp" macro gain (S2) → the honest figure is **+16.9pp, CI [+12.5,+21.2]**, and the LCO margin is a
   **range +13.3 to +16.9 pp** (S19).
5. p=0.0600 must be stated as **not significant** — never "marginally significant" (S3).
6. **Every claim of statistical significance against the paper's baselines must be removed.** The old
   two-proportion z-test assumed independent samples but was applied to two classifiers scored on the
   **same clips**. Under the corrected McNemar bound, **neither previously-"significant" result survives**:
   54/64 vs full-bbox 62/64 was p=0.015 → bound **[0.008, 0.382]**; hybrid 51/64 vs 62/64 was p=0.003 →
   bound **[0.001, 0.207]**. Replace each with the bound interval **plus a plain sentence explaining why
   only a bound is available** (we do not have the paper's per-clip outputs). Never quote a bare p-value.
7. Do **not** claim the smaller encoder fixes the edge-latency failure — that cost is dominated by the
   detector and appearance extraction, not the relation head.

## Structural narrative point (chương 4/5)

Comparisons against the published Unsafe-Net aggregates **can never be tested for paired significance**,
because their per-clip outputs are unavailable. The thesis must therefore **not rest its central claim on
beating the paper's number**. The defensible claims are the ones where we control both sides — encoder
variants (E0/E1/E2/E3), the detector-rule gate (S1), the scene-appearance control (S2) — which is exactly
what the binary experiment suite is built around. State this as a deliberate methodological choice, not an
apology.

## Favourable argument for the reframe (chương 1/4) — state with exact counts

The old 4-class audit found all **32** duplicate groups label-contradictory, of which **26/32** were the
pair `0↔1` (walkway violation ↔ unauthorized intervention). Both are UNSAFE, so under the binary label
those contradictions **disappear**: the binary framing removes roughly **28 of 32** label contradictions,
leaving 4. This is a real argument in favour of the reframe. Report the counts; do not overstate it.

## Mandatory pre-registered limitations (chương 6)

- **L1 Panel-pair confound.** The opened/closed-panel sub-pair's safe/unsafe split correlates with background
  appearance (shared-cluster coverage **0.12/1.00**); its contribution to binary accuracy is partly
  attributable to scene identity. State that the descriptor is a temporal median of the raw frame and panel
  state is itself a large static visual change, so camera identity and panel state **cannot be fully
  disentangled** by this method. *Pre-registered 2026-09-16 — may not be dropped.*
- **L2 CarryOverload single-rule artifact** — its 100% recall is achievable by a one-line detector rule and
  gets *stronger* under binary collapse.
- **L3 OpenedPanel degeneracy** (S12) — one constant box localises 100% of val instances; ~0.125 of any
  mAP50 is free.
- **L4 Cross-clip box repetition** (S15) — 1790/6933 boxes repeat verbatim across clips; 114/566 clips linked.
- **L5 Scope** — the thesis objective is binary; the 8 sub-classes are auxiliary evidence only.
- **L6 Edge budget** — the P3 configuration does not meet 3.0 dec/s at imgsz 800; report the measured curve.
- **L7 Test-set contamination and ceilings.** 691 clips → 658 unique hashes, 32 duplicate groups.
  `0_te23.mp4` ≡ `4_te9.mp4` are byte-identical, both in test, **opposite binary labels** → **hard ceiling
  124/125 (99.2%)**. Three further test clips (`4_te4`, `4_te7`, `5_te13`) are byte-identical to unsafe
  train clips → **practical ceiling ≈121/125 (96.8%)**. This contamination is **binary-specific** — it did
  not exist in the 4-class protocol, because the safe clips were not in the dataset. State the declared
  dedupe policy and show the contaminated number beside the deduped one.
- **L8 No paired significance test against the published baselines** — see the structural narrative point
  above; only a bound is available.

## Related Code Files

Create: the nine `report/vn/*.txt` files. Modify: none. Delete: none — `report/*.md` stays read-only.

## Implementation Steps

1. Create `report/vn/`; draft a VN terminology glossary first (an inconsistent Vietnamese technical
   vocabulary is the main quality risk) — keep it in `report/vn/thuat-ngu.txt`.
2. Write in defence-criticality order: **chương 3 → 4 → 5 → 1 → 6 → 2 → tóm tắt → phụ lục → tài liệu**.
3. Every numeral is copied from the number sheet **by ID**, never retyped from the English prose.
4. Apply the strike-list while writing, not in review.
5. Write chương 6 limitations from L1-L6 verbatim-in-substance.

## Todo List

- [ ] `report/vn/` created + glossary
- [ ] chương 3 (phương pháp, encoder redesign)
- [ ] chương 4 (thực nghiệm, protocol suite)
- [ ] chương 5 (kết quả + Jetson)
- [ ] chương 1 (giới thiệu, binary problem statement)
- [ ] chương 6 (kết luận + L1-L6)
- [ ] chương 2 (tổng quan)
- [ ] tóm tắt, phụ lục, tài liệu tham khảo
- [ ] Strike-list verified absent (incl. **all z-test significance claims**)
- [ ] L1-L8 present in chương 6
- [ ] Ceilings printed beside every accuracy figure
- [ ] Favourable 28-of-32 reframe argument stated with exact counts

## Success Criteria

Nine `.txt` files exist; every number traces to a sheet ID (checked in phase 10); no strike-list item
present; L1-L6 present; binary framing consistent in every chapter's objective statement.

## Risk Assessment

R13 (silent numeric edits in translation) → copy by ID. R14 (withdrawn claims resurface) → strike-list.
R15 (slip) → chapter order is by defence criticality, so a slip costs chương 2/phụ lục, not the core.

## Security Considerations

No footage stills with identifiable faces in the appendix without blurring.

## Next Steps

Phase 10 verifies. Phase 11 (slides) stays deferred.
