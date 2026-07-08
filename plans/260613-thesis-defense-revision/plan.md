# Thesis Defense Revision — Overview Plan

**Goal:** Close every gap a defense committee will exploit, before the report. Make the
proposal→product deviation, the statistical limits, and the factual contradictions
*self-disclosed and defended* rather than discovered by the panel.

**Source of truth:** Numbers verified against committed artifacts (`deployment/artifacts/*.json`,
`sweep_results/ablations/*.json`, `run_resolution_ablation.py`). See `revision-checklist.md`.

**User decisions (2026-06-13):** (1) Run blind held-out experiment, adopt honest headline.
(2) Fix factual errors + update writing this session. (3) Compute limited — one re-run only.

---

## Phases

| # | Phase | Status | File | Risk if skipped |
|--|-------|--------|------|-----------------|
| 1 | **Blind held-out experiment** (mandatory, Phản biện 1a) | TODO (user-run; `run_blind_eval.py` spec ready) | `phase-01-blind-eval-experiment.md` | Headline 56/64 is test-leaked → invalidated at defense |
| 2 | **Factual contradiction fixes** (objective bugs) | ✅ DONE (E1–E4) | `revision-checklist.md §E` | Panel finds chapters contradict each other |
| 3 | **Narrative reframing** (scope/novelty front-loading) | PARTIAL — A1 ✅, D1 ✅; A2/A3 pending (A2 needs blind #) | `revision-checklist.md §A,§D` | "Đề cương lệch sản phẩm" read as failure not finding |
| 4 | **Statistical honesty reframing** | TODO — C1 needs blind #; C2/C3 text pending | `revision-checklist.md §C` | Lucky-seed + non-significant ensemble exposed |
| 5 | **Overclaim softening** | PARTIAL — F1 ✅, F2 ✅; F3 pending | `revision-checklist.md §F` | V10/security/demo over-read |
| 6 | **Cross-chapter consistency sweep + Q&A brief** | Brief ✅; final sweep after blind # | `rebuttal-brief.md` | Number drift; unprepared rebuttals |

### FINAL RESULT LOCKED (2026-06-17) — see reports/final-result-decision.md
- **Best config = hybrid (frozen ⊕ bbox-finetuned) + frame-gate.** Blind **53/64 (85.4 % macro)** —
  macro parity with no-bbox baseline (85.78 %) EVEN BLIND; best-case 86.9–87.7 % macro (> baseline).
- **Framing = positive Pareto** (beat/parity no-bbox baseline on macro + edge + lower annotation cost).
- Corrected-mechanism investigation done: bbox-finetune regresses (over-specialises); frame-gate +1
  on frozen (noise); **hybrid recovers + is best** (small novel finding). Ceiling ~54/64 holds across
  12+ tracks incl. paper's exact mechanism @1200 → gap is **data-volume-bound** (needs ≥2262 bbox).
- NO 2262 gamble, NO ConvLSTM rebuild (cheap levers exhausted; data trends against more-data gamble).
- Written into Ch4 §4.6.2 (full table + 3 findings), Ch6 Finding 2 (positive macro-parity framing).

### DECISIONS LOCKED (2026-06-13)
- **Spine = C** (deployment-led empirical study): lead with the delivered edge system; accuracy is
  evidence for the data-bound negative finding, not a competitive claim.
- **Headline accuracy = blind held-out ≈ 52/64** (≈ 81 % top-1, macro ≈ 84 %), 3-number ladder
  (best-case 56 → typical 53 → blind ≈ 52). Deployed artifact keeps best-case 56/64 checkpoint;
  scientific claim = blind. **Macro "+3.5 pp win" withdrawn** (did not survive blind).
- **Cascade demoted** — blind config-selection picks single direct-4 every seed; reported as
  tried-but-unhelpful, not a contribution.
- **MT1 = substantially met** (Transformer temporal stage built); RT-DETR dropped on principle, NOT
  re-run. **Blind experiment DONE** (`run_blind_eval.py`, §4.7.11, two protocols, robust ~52).

### Spine-C rewrite applied this session
- Ch1: §1.3.1 (MT1-substantially-met + blind ladder), §1.4 fully reordered deployment-led, §1.5.
- Ch3: §3 intro + §3.4.1 + §3.5.2/3 caveated to best-case + blind.
- Ch4: NEW §4.7.11 (blind held-out experiment, Table 4.24); §4.0/§4.1/§4.2.2/§4.3/§4.7.6/§4.7.7/§4.7.9/§4.9
  all caveated/withdrawn macro-win; Table 4.3 adds blind row.
- Ch5: §5.6 intro + §5.6.7 Table 5.5A withdraw macro-win, add blind row.
- Ch6: §6.1 reordered (deployment Finding 1, blind Finding 2, cascade-rejected Finding 4), §6.2 ladder
  table, §6.3(a) leakage, §6.5 closing. README headline + contributions reframed.

### Done this session
- Re-ran M4 benchmark (`benchmarks.json`, 100 runs) → authoritative **CPU-FP32 26.8/28.8 ms/37.3 FPS**;
  the stale 28.8/34.7 prose corrected across Ch1, Ch2, Ch5 (Table 5.2 + prose), Ch6, README.
- Resolution lie fixed (§5.7: 224² both train+deploy, no 640×384).
- UCF-Crime Ch6 synced to the real 267-clip/3-class probe (was "25 Burglary, blocked").
- N=1 async-overhead explanation added (§5.5.2 M4, §5.6.3 Jetson).
- V10 "only route" softened; security T2–T4 reframed as operational not novel.
- Ch1 §1.3.1 "Scope evolution from proposal" table (RT-DETR drop, >95.8% miss, 8→4) added.
- §4.7.8 temporal-reasoning naming honesty added.

### Remaining (mostly blocked on the blind number from Phase 1)
- C1 + A2: finalize headline as blind mean±seed once `run_blind_eval.py` lands (demote 56/64 to upper bound).
- C2 (name the proven contribution in §4.7.9/§6.1), C3 (macro-vs-top1 one-liners), F3 (demo = PoC wording),
  A3 (8→4 line in abstract) — number-independent, can do next session.
- G1/G2 final consistency grep after numbers settle.

---

## Priority order (compute-limited)

1. **P1 — Blind eval** (Phase 1): the ONE re-run. Changes the headline number; everything
   downstream cites it. Do first. Fallback = disclosure-only if it cannot complete.
2. **P2 — Factual fixes** (Phase 2): zero-compute, surgical edits, highest credibility-per-effort.
3. **P3 — Narrative + statistics + overclaim** (Phases 3–5): writing only.
4. **P4 — Consistency sweep + rebuttal brief** (Phase 6): last, after numbers settle.

## Three deviations to front-load in Chapter 1 (non-negotiable)

- **RT-DETR dropped**: proposal core novelty (`PROPOSAL.txt:27,44,104,138`) never built.
  Re-cast as "frozen YOLOv11s spatial encoder + lightweight Transformer heads; contribution is
  the empirical study + edge artifact, not a new mechanism."
- **>95.8% target → 87.5% (now blind ~53–54)**: reframe as falsified hypothesis + data-bound finding.
- **8-class → 4-class headline**: declare scope narrowing in abstract/scope, not buried in §6.4.

## Key dependencies

- Phase 1 output (blind number) feeds Phases 3–6 and README/abstract. Do NOT finalize headline
  prose until Phase 1 lands.
- All Phase 2 numbers are already verified — can proceed in parallel with Phase 1.

## Done = 

Every committee point in `rebuttal-brief.md` has either (a) a fix applied, or (b) a written,
defensible answer. No chapter contradicts another. Every headline number traces to a committed artifact.
