---
title: "Binary Safe/Unsafe Reframe — Master Implementation Plan"
description: "Reframe the thesis objective to binary Safe-vs-Unsafe on all 691 Unsafe-Net clips, redesign the encoder, measure the full Jetson metric set on 2026-09-19, and deliver a Vietnamese thesis as .txt."
status: pending
priority: P1
effort: 104h
branch: next-round
tags: [thesis, binary-classification, unsafe-net, yolo26, transformer, jetson, vietnamese]
created: 2026-09-16
---

# Binary Safe/Unsafe Reframe — Master Plan

**Objective (binding, 2026-09-16):** the task is **BINARY: Safe (4-7) vs Unsafe (0-3)** over all **691 clips**
(566 train = 452 U / 114 S; 125 test = 64 U / 61 S). The 8 sub-classes are **auxiliary supervision and
diagnostic evidence only** — never the headline objective.

**Deliverable:** Vietnamese thesis as plain `.txt`, one file per chapter, in `report/vn/`. No Word/docx, no
template work (user formats personally). English `report/*.md` is a **source to rewrite from**.

## Binding constraints

| # | Constraint |
|---|---|
| C1 | Binary objective propagates into every objective, experiment and chapter. |
| C2 | Encoder **redesigned** (E1 F-STE), not re-headed. 8-way sub-class aux confirmed. |
| C3 | **Jetson access = 2026-09-19 only.** User operates the device; notify before each on-device step. |
| C4 | Eval contract §4h: selection on **val/OOF only**; **two** declared test touches (E0-binary + winner); **≥5 seeds**. |
| C5 | Existing 4-class results are **historical context only — never a baseline row** in a binary table (equal-recipe rule). |
| C6/C7 | Deadline ~1 week past 19/9 (only the Jetson date is hard). Slides + Q&A **deferred** (phase-11). |

## Phases (listed in execution order)

| # | Phase | Lane | Effort | Gate |
|---|---|---|---|---|
| 01 | [Binary data prep + rule baseline](phase-01-binary-data-prep-and-rule-baseline.md) | B | 4h | G1 tokens verified |
| 12 | [Duplicate audit, dedupe policy, ceilings](phase-12-duplicate-audit-and-ceilings.md) | B | 3h | G0 policy fixed pre-touch |
| 02 | [Binary encoder redesign E0/E1/E2/E3](phase-02-binary-encoder-redesign.md) | B | 12h | G2 arch frozen 18/9 |
| 03 | [Code-defect fixes](phase-03-code-defect-fixes.md) | B | 8h | G3 TRT 2-input fixed |
| 04 | [Pre-Jetson checkpoint + export](phase-04-pre-jetson-checkpoint-and-export.md) | B | 6h | G4 parity gate |
| 05 | [**Jetson session 2026-09-19**](phase-05-jetson-measurement-session.md) | B | 8h | G5 metric set captured |
| 06 | [Binary experiment suite](phase-06-binary-experiment-suite.md) | B | 16h | **GATE S1/S2 first**; G6 headline frozen |
| 07 | [Number-sheet freeze](phase-07-number-sheet-freeze.md) | D | 5h | G7 single source |
| 08 | [Vietnamese thesis (.txt)](phase-08-vn-thesis.md) | C | 30h | — |
| 09 | [README + docs/](phase-09-repo-docs-and-readme.md) | D | 6h | — |
| 10 | [Final consistency review](phase-10-final-consistency-review.md) | D | 5h | G8 trace-to-sheet |
| 11 | [Slides + Q&A — DEFERRED](phase-11-slides-and-qa-deferred.md) | — | — | chưa kích hoạt |

## Critical path

`01 → 12 → 02 → 04 → **05 (19/9 FIXED)**` then `06 → 07 → 08 → 10`.
Phase 03's TRT fix (D2) is a hard prerequisite of 05; D6/D7/D8 gate the validity of 06.

**Two de-risking facts:** (1) **the Jetson session consumes zero test touches** — latency/energy/thermal use
dummy or train-split tensors and parity is *output agreement vs FP32*, so 19/9 needs only a frozen graph +
a representative val-selected checkpoint; (2) **architecture freeze, not weight finality, is the 18/9 gate**
— cost metrics are graph-dependent, so the final identical-graph checkpoint inherits latency/energy exactly.
**Gates-before-training:** phase 06's S1 (detector-rule) and S2 (scene-appearance) run **before any encoder training** — S1 may reframe the whole accuracy claim.

## Dependencies (do not duplicate)

- `.../laneb-2026-09-16-binary-data-prep.md` — 691 tokens, scene grouping, views. Landed (1-2), 3-5 pending.
- `.../design-2026-09-16-binary-encoder-redesign.md` — **landed**; E0/E1/E2/E3, decision rule, cost table.
- `.../laneb-2026-09-16-paired-significance-fix.md` — **landed**; McNemar replaces the z-test.
- `.../laneb-2026-09-16-binary-dedupe-and-ceiling.md` — **incoming**; binary/691 dedupe policy.
- `brief.md` — D1-D5, eval contract §4h, S1-S20. Audits (3) + reviews (4).

## Gates

- **G0** dedupe policy declared and dated **before** any test touch.
- **G1** 691 tokens verified (shape, layout, no NaN, safe-clip geometry non-zero) → else no training.
- **G2/G3/G4 (all 18/9)** architecture frozen + tests pass · TRT multi-input fix landed · ONNX parity ≤1e-5.
  Miss G2 → phase 05 benchmarks E0 only; miss G3 → no binary head on device; miss G4 → no latency number.
- **G5 (19/9)** mandatory Jetson metric set captured, or fallback recording cut. **No second chance.**
- **G6** winner decided by the **pre-registered decision rule**; two touches logged → then **no new accuracy experiments**.
- **G7** number sheet frozen; every comparison row carries b, c + exact-p-or-bound + paired CI.
- **G8** every number in `report/vn/*.txt` traces to the sheet; no z-test p-value survives anywhere.

See [risks.md](risks.md) for the risk register.
