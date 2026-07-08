# Phase 1 — Blind Held-Out Evaluation (MANDATORY)

**Priority:** P1 (the one re-run). **Addresses:** Phản biện 1a (test-set leakage at epoch selection),
1b (lucky-seed). **Owner:** main. **Status:** TODO.

## Why this is mandatory

`run_pairwise_contrastive.py:230` — `train_classifier(model, ds, test_x, test_y, ...)` selects the
saved checkpoint (`best_st`) by **correct-count on the 64-clip test set** at line 255
(`pnp = model(test_x).argmax(1)`; `if c > best_c: best_st = deepcopy(...)`). Every head's reported
epoch is chosen on the test labels. So 56/64 inherits model-selection-on-test leakage. §4.1 admits it
but no clean held-out number was ever measured. Phản biện 1 will (correctly) demand it.

## Exact experiment

Create `run_blind_eval.py` (a thin variant; do NOT edit the original — reproducibility):

1. Load `all8_train.pt` / `all8_test.pt` via the existing `extract_all_features()`; build 4-class
   stats exactly as `main()` does (`tr4_stats`, `te4_stats`).
2. **Stratified split** `tr4_stats` → `tr_fit` (80%) / `tr_val` (20%), seeded. Keep the 64-clip
   `te4_stats` **completely untouched** until the final single evaluation.
3. Call `train_classifier(model, ds=tr_fit, test_x=tr_val, test_y=tr_val_labels, ...)` — i.e. early
   stopping / `best_st` now selected on `tr_val`, NOT on the 64 test clips.
4. After training stops, evaluate the selected checkpoint **once** on `te4_stats` → blind top-1, macro.
5. Repeat for: (a) the headline seed 314, (b) the standard seed pool used in Table 4.16, (c) ≥5 seeds
   total if compute allows (user said "limited — a few runs": minimum = seed 314 + 2 standard seeds).
6. Report **blind mean ± std** and the **per-seed** values. Also re-evaluate the cascade heads under
   the same blind protocol so the ensemble blind number is honest.

Expected outcome (from Table 4.16 retrains, which already avoid the lucky seed): **~53–54/64 ± ~0.5**.

## Reporting changes once the number lands

- **Headline becomes:** "Blind held-out top-1 = X/64 (mean ± std over K seeds); best-case (test-selected,
  lucky-seed `direct4_s314`) = 56/64 — reported as an upper bound, not the primary metric."
- Update: `README.md:14-33,142-153`, `chapter-01` headline numbers, `chapter-04 §4.2.2` Table 4.2,
  `chapter-04 §4.7.7/§4.7.8`, `chapter-05 §5.1,§5.4.2` (the deployment accuracy is still measured on
  the test-selected checkpoint — clarify that the *deployed artifact* uses the best-case checkpoint,
  while the *scientific claim* uses the blind number), `chapter-06 §6.1 Finding 1, §6.2`.
- Add a new row to Table 4.13a / a new Table 4.x: "Blind held-out (val-selected epoch)".

## Fallback (if the run cannot complete before defense)

Disclosure-only: state in §4.1 + §6.2 that the blind number was not measured and that Table 4.16's
standard-seed retrains (53.0–53.67, selected the same leaky way) **upper-bound** the blind number, so
the true blind figure is ≤53.67. Label 56/64 explicitly "test-selected, lucky-seed — upper bound."
This is weaker; prefer running the experiment.

## Acceptance

☐ `run_blind_eval.py` runs end-to-end, prints per-seed blind top-1/macro + the single final test eval.
☐ Blind number + std recorded in this folder's `reports/blind-eval-result.md`.
☐ All headline-number locations updated to lead with the blind figure, 56/64 demoted to upper bound.
