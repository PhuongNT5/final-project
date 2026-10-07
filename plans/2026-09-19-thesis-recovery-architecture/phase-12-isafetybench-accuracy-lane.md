# Phase 12 — iSafetyBench accuracy lane (external dataset)

**Lane** A/B · **Effort** 10h · **Gate G10** — encoder must beat the pre-registered R42 count-rule
floor under leave-source-out, on identical folds, or the lane is written up as a second negative
result and the thesis stays on the Unsafe-Net failure-analysis spine.

## Context links

- `reports/finding-2026-09-19-isafetybench-feasibility.md` — dataset facts, R42 gate PASS (63.3% BA
  LSO / 67.0% random / 50% majority), §6 reassignment recommendation, 4 unresolved questions.
- `reports/finding-2026-09-19-walkway-geometry-is-unsalvageable.md` — why phase-03 died and why an
  external dataset was opened at all.
- `reports/researcher-2026-09-19-alternative-safety-datasets.md` — survey (read the ORCHESTRATOR
  CORRECTIONS banner first; three claims in it are wrong).
- `risks.md` R41 (same confounding, diluted), R42 (count-rule shortcut gate), R43 (switch timing).

## Overview

Priority **P1** — after phase-03's death this is the only place a positive accuracy result is still
possible, and the user's binding objective order puts accuracy first.

Status: **lane measured out.** G10 passed, G11 **failed**, G12 **passed**, G13 not a clean pass.
No surviving positive accuracy claim on this corpus; the surviving contribution is methodological
(protocol inflation + non-discriminativeness) — see
`reports/finding-2026-09-21-protocol-inflation-and-the-capacity-contrast.md` §10.

Scope is deliberately narrow: reuse every existing artefact (encoders E0–E3, `train_one`,
`evaluate_suite`, the grouped-CV driver) and change only the data. No new architecture work — the
architecture-invariance finding (plan §Day-0 #2) makes encoder search unfalsifiable here too.

**Superseded 2026-09-21 (R47).** That premise was wrong on this corpus. Capacity is decisive here
(E0 − E3 = +5.1 pp, p=7.74e-06) and irrelevant on Unsafe-Net (−2.8 pp, CI includes 0). Invariance
was a symptom of an unlearnable task, not a property of the encoder family, and it holds only where
there is no extractable signal.

## Key insights

1. **The corpus is 1,100 clips / 150 source videos, and 0/150 sources carry both classes.** The
   Unsafe-Net pathology is *diluted, not cured* (R41). Grouped CV is mandatory; random splits are
   meaningless and must never be reported as the headline.
2. **Geometry is unavailable** — no zone polygons ship with the dataset, so token columns 6–9 are
   exactly zero. This is acceptable only because geometry was already proven dead on Unsafe-Net. It
   does change the narrative: "relation" here means entity-to-entity, not entity-to-zone.
3. **The detector must be COCO-pretrained** (`yolo11s.pt`, 80 classes). The project's 8-class
   Unsafe-Net detector is meaningless on this corpus. COCO's 80 classes are folded into 8 coarse
   entity groups so `nn.Embedding(n_entity=8, …)` and every existing encoder run unchanged.
4. **The shortcut bar is already measured.** persons/frame 1.32 (hazard) vs 2.08 (normal) — a
   3-feature count rule reaches 63.3% BA under leave-source-out. Any encoder number on this corpus
   is meaningless unless printed beside that floor on the same folds.

## Architecture

```
data/isafetybench/{hazard,normal}/*.mp4          (1,100 clips, gitignored, 1.2 GB)
  └─ build_isafetybench_tokens.py                reuses build_relation_tokens.process_clip
       detector = yolo11s.pt (COCO-80) → COCO_GROUPS → 8 coarse entity ids
       zones    = None                  → geometry columns exactly zero
       views    = YouTube source id parsed from filename
  └─ sweep_results/relation_tokens/tokens_isb_all.pt   (n,16,8,74) — byte-compatible with bin8_
       └─ isafetybench_dataset.py       load / folds (StratifiedGroupKFold) / count-rule baseline
            └─ run_kfold_relation.py --dataset isafetybench --lco --binary
                 grouped folds, pooled OOF, report_binary(sub_names=None)
                 + R42 count-rule floor recomputed on the SAME folds
```

`--folds 10` packs the 150 single-class sources into 10 grouped folds. 150 literal leave-one-source-out
folds are both prohibitively slow and degenerate per fold; packing preserves the property that matters
— no source ever spans the train/held-out boundary.

## Related code files

Created: `build_isafetybench_tokens.py` (124), `isafetybench_dataset.py` (71),
`tests/test-isafetybench-dataset.py` (129).
Modified: `run_kfold_relation.py` (`--dataset`/`--folds`, `_grouped_folds` dispatch, optional
`sub_names`), `run_relation_transformer.py` (`train_one(..., n_sub=)`), `.gitignore`.
Not touched: every deployment/ONNX/Jetson artefact — edge metrics are label-independent and stay on
Unsafe-Net, so the switch costs zero rework there.

## Implementation steps

1. ✅ Build tokens for all 1,100 clips (`--appearance --imgsz 640`, MPS inference).
2. ✅ Dataset module + grouped-fold generator + count-rule baseline on identical folds.
3. ✅ Driver dispatch + regression test proving the Unsafe-Net LCO partition is unchanged.
4. Run E3 (18k params) LSO first — cheapest, and the frozen Unsafe-Net headline arch.
5. Run E0 (299k params) LSO as the architecture-invariance cross-check on a *second* corpus. If the
   two agree here as they did on Unsafe-Net, that invariance claim generalises and is worth stating.
6. Ablations, cheapest-first: appearance-free (columns 10–73 zeroed), class-id-blinded (entity group
   channel constant). These localise where the signal lives — the same decomposition that turned
   Unsafe-Net's result into a mechanism-level finding.
7. Cross-dataset contamination check (feasibility Q4) before any transfer claim.
8. Number sheet rows + thesis section.

## Todo

- [x] `build_isafetybench_tokens.py` + smoke test (8 clips, 74 feats, geometry zero)
- [x] `isafetybench_dataset.py` + `tests/test-isafetybench-dataset.py`
- [x] `run_kfold_relation.py --dataset` dispatch + Unsafe-Net partition regression test
- [x] Full 1,100-clip token build verified against the test suite (52,997 detections, 151 sources)
- [x] E3 LSO run vs the floor on identical folds — **75.2% vs 67.2%, G10 PASSES**
      (`reports/finding-2026-09-20-isafetybench-g10-first-positive-accuracy-result.md`)
- [x] `feature_ablation.py` + `--ablate` wiring + `tests/test-feature-ablation.py` (10/10 pass)
- [x] Seven-condition ablation sweep — **`appearance_only` matches the full model (75.6% vs 75.2%,
      p=0.551): all ten FEAT columns are dispensable, the signal is crop appearance.** Kills the
      spatial-relational framing on this corpus; see the G10 finding report §5(b).
- [x] Contamination check vs Unsafe-Net — **no contamination**, min Hamming 13/64 (R46)
- [x] Number-sheet rows (`number-sheet.md` Block 1, all measured rows carry `log_path:line`)
- [x] **G11 whole-frame falsification baseline — FAILED.** Best detector-free probe (linear-mean)
      **74.1%** vs E3 75.2%: margin **+1.1 pp**, cluster CI **[-4.8, +6.9] includes 0**, McNemar
      **p=0.625**. Both pre-registered criteria fail. **The thesis may not claim the detector +
      relation head add accuracy on this corpus** — iSafetyBench becomes a parity/negative result,
      accuracy reverts to a limitation, edge efficiency carries the contribution. G10 finding §9.
- [x] **E0 LSO cross-check — 80.3% BA, and it falsifies architecture invariance (R47).**
      E0−E3 = +5.1 pp, CI [+2.0, +8.3], p=7.74e-06. E0−whole-frame = +6.2 pp, CI [+2.1, +10.5],
      p=1.59e-05 — clears G11's arithmetic but **G11 named E3 and remains FAILED**; E0 was planned
      as a cross-check, not a headline candidate, so promoting it is undeclared selection. Both
      facts reported together, failure first. G10 finding §10.
- [x] **Scope the invariance claim (R47)** in `phase-04` L23, `phase-09` step 5 chapter-07
      subsection, `phase-11` narrative template, `phase-12` L26: invariance held on Unsafe-Net,
      not on iSafetyBench — it was a symptom of an unlearnable task, not a property of encoders.
- [x] Fold-6-at-chance failure analysis — one source (`0zQ9fI30v1E`, 31 crowded normal clips,
      8.0 obj/frame) at 0% correct; density is not a corpus-wide driver (r +0.058 excluding it) but
      FP rate is U-shaped in density (41.9% / 12-18% / 52.4%) — narrow operating envelope, goes in
      limitations.
- [x] **G12 protocol-inflation measurement — PASSES.** Same tokens/encoders/folds/epochs, only fold
      construction changes. Δ_inflation E3 **+15.4 pp** (cluster CI [+10.6, +21.5], McNemar
      p=3.58e-45), E0 **+10.4 pp** ([+6.3, +15.3], p=1.3e-24). Control: the 3-feature count rule,
      which cannot memorise source identity, gains only **+1.0 pp** — the inflation is identity
      memorisation, not split arithmetic. Every random-split number is a diagnostic artefact and
      never a headline. `reports/finding-2026-09-21-protocol-inflation-and-the-capacity-contrast.md`
- [x] **G13 capacity ladder — does NOT pass cleanly.** LSO: E3 75.2 / E1d64 78.1 / E1d96 **77.0** /
      E0 80.3. Spread 5.1 pp clears the flatness clause, but E1 d96 sits 1.1 pp BELOW E1 d64 at
      double the parameters, so the ordered-trend clause fires. Pre-registered wording was ambiguous;
      that is my drafting error and is not renegotiated. Permitted: "capacity class matters where
      signal exists". Forbidden: "accuracy increases with parameter count".
- [x] **Headline that outranks both gates: a leaking protocol is non-discriminative.** E0 − E3 (17x
      capacity) = **+5.1 pp, p=7.74e-06** under LSO-1100 but **+0.1 pp, CI [−1.2, +1.5], p=0.665**
      under RAND-1100; the widest pair anywhere on the random ladder is +1.4 pp (p=0.105). Such a
      protocol cannot support ANY architecture comparison — it invalidates the *comparisons* in the
      baseline literature, not merely its absolute numbers.
- [x] **Baseline-paper gap explained from the papers.** iSafetyBench reports no high numbers at all
      (16-option MCQ, zero-shot, GPT-4o 40.3/37.3, chance 6.25%); Unsafe-Net's 96.88% sits one clip
      below a provably corrupted 63/64 ceiling on a 64-clip test set, random-split over 2 cameras.
      Finding report §9.
- [x] Number-sheet Blocks 1 and 2 filled with the full 2x2 (Block 2 was all placeholders).
- [ ] **Reframe consequence:** phase-12 §6's lane assignment (accuracy -> iSafetyBench) no longer
      holds. Accuracy pillar has no surviving positive claim on either corpus; the thesis argument
      must be rebuilt around efficiency + protocol critique + the deployment measurement. Author
      decision, not this lane's.

### Floor recomputation changed the bar (2026-09-20)

The gate's 63.3% was a 300-clip T=8 probe. Recomputed on all 1,100 clips at T=16 on the encoder's own
folds it is **67.2%** — the harder bar, and the one G10's wording ("identical folds") mandates. The
dominant count-rule feature turned out to be `mean_conf` (0.518 hazard / 0.628 normal), NOT person
count: hazard footage is simply harder to detect in. Hence the conf-zeroed ablation is mandatory, not
optional, and it is applied to the floor as well as the encoder (`--ablate` feeds both).

## Success criteria

**G10 (pre-registered, declared before the run):** E3 under leave-source-out must beat the count-rule
floor **recomputed on the same 1,100 clips and the same folds** by a margin whose paired bootstrap CI
excludes 0. A margin under ~3 pp is declared *not* a result regardless of sign — the same bar the R42
gate used.

If G10 fails: the lane is written up as a negative result (external dataset, same relational
component, same weakest-link conclusion), which strengthens rather than weakens the mechanism claim.
No third dataset. No retry with a different encoder — architecture invariance forbids it.

## Risk assessment

- **R41** residual scene confounding (150 groups, top-5 sources hold 24% of clips). Mitigated by
  StratifiedGroupKFold and by reporting grouped-vs-random gap, never random alone.
- **R42** the count-rule shortcut. Mitigated by printing the floor beside every headline number.
- **R43** timing — token build + runs land inside the 09-22→25 window freed by phase-03's death, so
  this does not displace the Jetson or writing lanes.
- **New:** class-imbalance 420/680 is mild and handled by the existing `use_cw=True` path.
- **New:** COCO detector quality on this footage is unmeasured (no ground-truth boxes ship with the
  dataset). Must be stated as a limitation — unlike Unsafe-Net, there is no mAP number to quote.

## Security considerations

Dataset is gitignored (1.2 GB). No credentials, no external calls at run time.

## Next steps

Feeds phase-06 (frozen metric suite) and phase-09 (number sheet). Does **not** feed phase-07/08 —
edge and demo stay on Unsafe-Net.

## Unresolved questions

1. Headline vs transfer framing — is iSafetyBench the accuracy dataset, or a transfer experiment
   beside the Unsafe-Net negative result? Proceeding on the feasibility report's §6 recommendation
   (accuracy → iSafetyBench, edge → Unsafe-Net, Unsafe-Net → failure-analysis chapter) since it
   follows the user's binding accuracy-first order; reversible until the number sheet freezes.
2. Does the thesis keep a per-entity *relation* framing with geometry unavailable? Tokens carry class
   + box + appearance only, so "relation" means entity-to-entity here. Narrative change, not just an
   experiment.
3. Do the two test touches remain pre-registered against the Unsafe-Net blind split, or does a
   headline switch re-point T1 at a held-out iSafetyBench source set? Currently unchanged: 0 spent,
   both still pointed at Unsafe-Net.
4. Contamination between the two corpora — unchecked, near-zero prior, must be stated as checked
   before any transfer claim.
