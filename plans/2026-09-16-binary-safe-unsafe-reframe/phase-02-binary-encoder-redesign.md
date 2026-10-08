# Phase 02 — Binary Encoder Redesign (E0/E1/E2/E3)

## Context Links

- **Design (landed):** `plans/2026-09-06-thesis-2week-final/reports/design-2026-09-16-binary-encoder-redesign.md`
  — this phase implements §2, §3, §7 of that document. Do not re-derive it.
- `model_relation_head.py`, `run_relation_transformer.py`, `build_relation_tokens.py`.
- Phase 01 (tokens), phase 12 (contamination ceilings), phase 06 (the comparison that judges this work).

## Overview

- **Priority:** P1. **Status:** design landed, implementation pending. **Effort:** 10h.
- Implement four encoders under **one identical binary recipe**, freeze the architecture **18/9**.

## Key Insights

- The current encoder is over-parameterised by orders of magnitude: **297,860 params / 464 fit clips =
  642 params per training clip.** The measured symptoms are already in the record — 11.0 pp same-distribution
  per-seed spread, 15.3-21.1 pp LCO per-class recall spreads.
- **The positional table is the clearest defect.** `pos` index = `1 + t*8 + j` where `j` is the
  **confidence rank** (`np.argsort(-conf)[:N_MAX]`), not an entity identity. Slot *j* is a different object
  from frame to frame, so 16,512 params (29 per training clip) are spent on unstable semantics. The model
  cannot express "the same entity moved", but can express "position 37 is special" — memorisation.
- **54.3% of token slots are padding** (mean 3.66 of 8 valid boxes/frame) → masked pooling is the natural
  reduction, and sizing a positional table for the padded worst case is waste.
- **Tracking is deliberately NOT added** (YAGNI): no tracker exists in the pipeline and adding one would
  confound the encoder comparison. This supersedes the IoU-association idea in the earlier draft of this
  phase — permutation invariance replaces it.
- **`cls_embed` may already solve the task.** The detector's 8 classes are safe/unsafe *counterpart pairs*
  (Person_Unauthorized/Authorized, Panel_Opened/Closed, Forklift_Overload/Safe_Load), so under the binary
  label the class id alone may be near-sufficient — the CarryOverload pathology generalised. **This is
  measured before the redesign is credited with anything** (phase 06 gate S1 + the class-id-blinded ablation).

## Requirements

Functional: four encoder classes selectable by `--arch {E0,E1,E2,E3}`; `--binary`; 2-logit softmax;
three heads; token variants T1/T2/T3. Non-functional: E0 stays byte-compatible for the legacy 4-class path;
ONNX-exportable (2 inputs) for phases 04/05; deterministic per seed.

## Architecture

| | params | pos params | attn pairs | params/clip | role |
|---|---:|---:|---:|---:|---|
| **E0** flat (today), retrained binary | 297,860 | 16,512 | 16,641 | 642 | **null hypothesis — mandatory, not optional** |
| **E1** F-STE d=64, T3 | **94,380** | 1,088 | 1,313 | 203 | **recommended proposal** |
| E1 d=96 | 197,292 | 1,632 | 1,313 | 425 | capacity check |
| **E2** flat + frame-only pos | 283,524 | 2,176 | 16,641 | 611 | attribution control |
| **E3** pooled DeepSets MLP, no attention | **18,044** | 0 | 0 | 39 | anti-overfitting floor |

**E1 "factorized set-then-time" (F-STE)** — per frame: `[cls_embed(16) ‖ num_proj(73→d-16)]` → LayerNorm →
**no positional embedding** (entities in a frame are an unordered *set*) → 1 encoder layer (d=64, h=4,
ffn=2d, p=0.2) → masked mean‖max over valid boxes → `Linear(2d→d)` → frame embedding `f_t`
(empty frame → learned `empty_frame`). Then temporal: `[CLS] f_0..f_15` + `temporal_pos[0..16]`
(the model's **only** positional table, 1,088 params) → 1 encoder layer → pool `CLS ‖ mean ‖ max` (3d) → heads.

**E2 is the scientific control that makes E1 interpretable:** E0→E2 isolates "was it the positional prior?",
E2→E1 isolates "was it the flat attention?". Without E2, an E1 win is an unattributable bundle of three
changes — exactly what a committee pushes back on.

**E3 matters most for honesty:** if E3 ≈ E1 within CI, the defensible sentence is *"a permutation-invariant
pooled encoder matches a Transformer on this dataset; the extra 5.2× parameters are unjustified at n=691"* —
a stronger contribution than a 1-pp win with overlapping CIs.

**Heads and loss.** 2-logit softmax (NOT BCE — `eval_utils/suite.py` and `unsafenet/metrics.py` derive
`n_cls` from `probs.shape[1]`, so a 2-column softmax reuses the **entire** metric suite with zero new code).
`L = CE_bin(weighted) + 0.3·CE_sub(8-way) + 0.5·CE_frame`, label smoothing 0.05 on all three.
Convention repo-wide: `class_names=['Safe','Unsafe']`, **index 1 = Unsafe = positive**.

**Class weights ON.** `class_weights(y, k=2)` mean-normalises → `w_unsafe=0.403`, `w_safe=1.597` (ratio 3.96).
Quote the normalised values, not Agent A's 0.626/2.48 (same ratio, different scale; scale interacts with LR).
Rationale is **prior shift**: train is 3.96:1 unsafe-heavy, test is near-balanced (64/61), so
inverse-frequency weighting matches the evaluation prior. This **contradicts the k-fold script's current
`use_cw=False`** — the change is scheduled in phase 03 and must be stated in the thesis as a deliberate,
prior-matched decision, not slipped in. Headline metric is therefore **balanced accuracy / macro-F1, not top-1**.

**Token variants** (all derived from existing tensors, no re-tokenisation): T1 = full feat-74 absolute coords
(shortcut-prone, upper-bound reference); T2 = feat-72 `yolo26L1_` (xc,yc dropped, matches canonical LCO);
**T3 = frame-relative coords** (`xc − mean_valid(xc)` per frame, zeroed on padded slots) — **E1's default**,
translation-invariant w.r.t. scene layout while preserving entity-to-entity geometry.

## Related Code Files

Modify: `model_relation_head.py` (add E1/E2/E3; **leave `RelationTransformer` untouched** — E0 = pass
`n_cls=2`; add the 8-way aux head), `run_relation_transformer.py` (`NAMES`, `class_weights(y,k=2)`, two
`torch.zeros(...,4)`, `range(4)` loops, the `two_proportion_z(...,64,57,64)` hardcodes, `--arch`/`--binary`
flags, T3 transform). Create: `tests/test_relation_encoder.py`.

## Implementation Steps

1. Add the 8-way aux head + 2-logit binary head; wire `L = CE_bin + 0.3·CE_sub + 0.5·CE_frame`.
2. Implement E1 (d=64 and d=96), E2, E3 as separate classes behind `--arch`.
3. Implement the T3 frame-relative coordinate transform (applied at load time, not at token-build time).
4. `class_weights(y, k=2)`; remove the 4-class hardcodes listed above.
5. Unit tests: **permutation invariance of E1's intra-frame stage** (shuffling valid slots cannot change the
   output); masked padding cannot influence logits; empty-frame path; shapes for all four archs; fixed-seed
   determinism; ONNX export of the 2-input E1 graph.
6. **10-minute calibration run** (E0-binary, 1 seed, 5 epochs, machine otherwise idle) — the two logged
   throughput anchors in this repo disagree **6×** (8.9 s/epoch vs 1.6 s/epoch, cause = machine contention).
   Every downstream wall-clock estimate is a range until this lands. **Run one heavy job at a time.**
7. Freeze the candidate list: **E0 / E1(d=64,d=96) / E2 / E3 + the four ablations. No new architectures.**

## Todo List

- [ ] Heads + loss implemented (2-logit softmax, λ_sub=0.3, λ_frame=0.5)
- [ ] E1 (d=64, d=96) implemented
- [ ] E2 implemented (frame-only pos, nothing else changed)
- [ ] E3 implemented (no attention)
- [ ] T3 frame-relative transform
- [ ] Class weights ON; 4-class hardcodes removed
- [ ] Permutation-invariance + masking tests green
- [ ] **Calibration run done → timing range collapsed**
- [ ] **G2: architecture frozen 18/9** — no graph change afterwards

## Success Criteria

All four archs train end-to-end under the identical recipe; permutation-invariance test passes; ONNX export
succeeds; param counts match the table (±1%); architecture frozen with its hash in the number sheet.

## Risk Assessment

R1 retired (design landed). **R5** — λ_sub/λ_frame are the *only* tunable hyper-parameters, chosen on
val/OOF, on E1 only, then **frozen and applied identically to E0/E2/E3**; any further tuning breaks the
equal-recipe rule. **R2** — E1 may not beat E0; the negative-result wording is pre-committed in phase 06.
**R8** — do **not** present the smaller encoder as the fix for the 1825 ms/decision edge failure: that cost
is dominated by detector + appearance extraction, not the relation head.

## Security Considerations

New checkpoints use new tags; never overwrite an existing artifact (S9 precedent).

## Next Steps

Phase 06 judges these encoders (gates S1/S2 run **before** any encoder training). Phase 04 exports **both
E0 and the winner** for the 19/9 device session.
