# Phase 02 — LCO Confound Decomposition + Reporting Variant

## Context Links
- [plan.md](plan.md) · [risks.md](risks.md) · [phase-01](phase-01-detector-evidence-package.md)
- `reports/finding-2026-09-19-lco-failure-is-camera-label-confounding.md` — **the source finding**
- `reports/critique-2026-09-19-research-factcheck.md` — researcher reports are UNVERIFIED
- Artifacts: `sweep_results/relation_tokens/enc_*_lco-691_headline-drop_conflict_oof.json` (6 encoders),
  `sweep_results/zone_prior/clip_view_map_691.json`, `sweep_results/zone_prior/clip_scene_map_691.json`

## Overview
- **Priority** P1 — **this is the thesis headline**, not a supporting analysis.
- **Status** finding landed ad-hoc; this phase makes it reproducible, versioned and frozen.
- **Effort** 5h · **Lane B** · **Gate G2** · **0 test touches, 0 GPU hours**
- Turn a one-off recomputation into a scripted, re-runnable evidence chain: per-sub-class LCO recall
  for all six encoders, sub-class × camera-group cross-tab, the four-way sub-task decomposition, the
  declared panel-disclosed protocol variant, and the carry-lookup triviality disclosure.

## Key Insights
1. Pooled LCO safe recall 42-51% is **not** a uniform inability to recognise Safe. Two sub-classes
   (`walkway_ok` n=75 at 15-36%, `panel_closed` n=32 at 0-12.5%) carry the entire failure; the other
   six sit at 74-100%.
2. **All 32 `panel_closed` clips are in one camera group**; `panel_open` is 88% in a different one.
   Held-out fold ⇒ zero training examples ⇒ 0.0% recall is arithmetic. LCO cannot evaluate the panel
   sub-task (174 clips, 25% of the pool) at all.
3. The failure is **architecture-invariant** over 18k→300k params: same two failures, same four
   successes, same difficulty order. Every remaining encoder experiment is therefore uninformative.
4. Carry sub-task is 100%/100% but is a **detector class lookup** (`Forklift_Overload` R=1.000,
   `Forklift_Safe_Load` R=1.000). Reporting it as a model result without disclosure is R13.
5. Sub-class is recoverable **for free** from `clip_id` (leading digit), and `fold_id` is already in
   every OOF row — so the whole decomposition is a pure post-hoc read of artifacts on disk.
6. Two of eight folds contain **no safe clip at all**; views 0 and 5 are 100% unsafe; 43% of safe
   clips sit in view 6. State this as a dataset property in the thesis, with numbers.

## Requirements
**Functional** — one script reproduces every table in the finding from artifacts on disk; the
panel-disclosed variant is computed as a *declared* variant reported alongside the full number; the
carry-lookup baseline is quantified, not asserted.
**Non-functional** — deterministic, <2 min runtime, no GPU, no test access, output as both markdown
and JSON so Lane D can ingest it without re-deriving anything.

## Architecture
```
enc_*_lco-691_*_oof.json  ──┐
clip_view_map_691.json    ──┼─> tools/eval-subtask-decomposition.py ─┬─> sub-task-tables.md   (Lane B)
clip_scene_map_691.json   ──┘                                        └─> sub-task-tables.json ─> number-sheet (Lane D)
                                                                      │
detector per-class recall  ───────────────────────────────────────────┴─> carry-lookup disclosure
```
Sub-class label = `int(clip_id[0])` (source folder class). Binary label = `to_binary()` in
`binary_gate_lib.py`. Fold id = OOF row field. No re-training anywhere in this phase.

## Related Code Files
**Create**
- `tools/eval-subtask-decomposition.py` (<180 lines) — per-sub-class recall × encoder; sub-class ×
  fold cross-tab; sub-task rollup; panel-disclosed variant; fold-vs-view agreement check. Emits
  markdown + JSON.
- `tools/eval-carry-lookup-baseline.py` (<80 lines) — trivial rule baseline: predict Unsafe iff a
  `Forklift_Overload` box is present in the clip's tokens; score it under LCO on the carry sub-task.
**Modify** — `reports/test-touch-log.md`: record "phase 02 = 0 touches, recomputation only".
**Delete** — none.

## Implementation Steps
1. **[M4-CPU, 25 min] Grouping-separation rule (Q2 RESOLVED — enforce, do not re-litigate).**
   Measured 2026-09-19: fold ids and view ids agree only **81.7%**; fold 1 mixes views 0/3/6 at 50%
   purity; folds 6 and 7 are compositionally identical. They are **not interchangeable**. All
   LCO/confounding claims are **fold-based** (the grouping LCO actually used) and stand unchanged.
   This phase's job is to enforce the wording constraint: re-run the cross-tab to regenerate the
   artifact, then (a) re-phrase every view-phrased sentence in fold terms, (b) attach the
   `run_kfold_relation.py:106` `[LIMITATION]` banner — *KMeans(8) background-descriptor proxy, not
   verified camera metadata* — to every LCO number, and (c) add an assertion that no emitted table
   mixes the two groupings.
   ```bash
   cd /Users/nguyenthiphuong/Learning/Proposal/project/my-project
   .venv/bin/python tools/eval-subtask-decomposition.py --check-grouping \
     --oof sweep_results/relation_tokens/enc_E3_lco-691_headline-drop_conflict_oof.json \
     --view-map sweep_results/zone_prior/clip_view_map_691.json \
     --scene-map sweep_results/zone_prior/clip_scene_map_691.json
   ```
2. **[M4-CPU, 60 min] Per-sub-class recall table**, all six OOF dumps, n and recall per sub-class,
   plus Wilson CI per cell (n=30-207, so CIs are wide and must be shown — a 32-clip cell without a CI
   is indefensible).
   ```bash
   .venv/bin/python tools/eval-subtask-decomposition.py \
     --oof-glob 'sweep_results/relation_tokens/enc_*_lco-691_headline-drop_conflict_oof.json' \
     --out reports/sub-task-tables
   ```
3. **[M4-CPU, 30 min] Sub-class × fold cross-tab** with `#groups` and `max share` columns; flag any
   sub-class with `#groups == 1` as structurally unanswerable under LCO. Make the rule explicit and
   general, not hand-picked: *a class whose clips occupy a single camera group cannot be evaluated by
   leave-camera-out.* `panel_closed` is the only one that trips it — say so.
4. **[M4-CPU, 30 min] Sub-task rollup** (zone 145 / carry 86 / walkway 282 / panel 174) with both-side
   recall, and the verdict column: generalises / trivial / open failure / unanswerable.
5. **[M4-CPU, 45 min] Panel-disclosed variant.** Recompute BA and safe recall excluding the panel
   sub-task, for all six encoders. **Reported alongside the full number, never replacing it.** Write
   the justification paragraph in the same artifact so the two can never be separated: declared in
   advance, structurally motivated, computed on the 691 pool, 0 test touches, R8-compliant because it
   is not a post-hoc *selection* of favourable clips but the removal of a sub-task LCO cannot score.
6. **[M4-CPU, 45 min] Carry-lookup disclosure.** Run `tools/eval-carry-lookup-baseline.py`. If the
   trivial detector rule matches the model on the carry sub-task, the thesis must say the carry result
   is a detector result, not a relation-model result. Pre-register: *if the lookup baseline scores
   ≥95% of the model's carry recall, carry is disclosed as trivial in every table footnote.*
7. **[15 min] `N_MAX` fix-list (Q4 RESOLVED).** Measured: headline `tokens_bin8_{train,test}.pt` are
   `(566|125, 16, 8, 74)` — 16 frames × **8** entity slots × 74 features. The "≤16 entity slots"
   figure is **wrong**. Grep the report tree and the plan tree for it, list every occurrence, and hand
   the list to Lane D (do not edit `report/**` from Lane B). No capacity or complexity statement may
   ship with 16 slots.
8. **[M4-CPU, 15 min] Detector-class-blinded ablation — how much is relational reasoning?** Blind the
   class-id channel and re-run E3 under LCO (≈51s per the measured anchor); compare per sub-task.
   This separates "relational reasoning" from "authorized-person lookup" and **bounds how strongly the
   zone positive result may be worded**. A committee will ask exactly this question.
   ```bash
   .venv/bin/python tools/eval-feature-ablation.py --tag bin8_ --arch E3 --lco \
     --variants full,blind_cls --per-subtask \
     --scene_map sweep_results/zone_prior/clip_scene_map_691.json
   ```
   Pre-registered wording rule: if zone-pair recall collapses under class blinding, the zone result is
   described as *class-plus-geometry separation*, never as relational reasoning.
9. **[15 min] Freeze.** Tag the JSON with git SHA + date; from here the tables are inputs to phase 09
   and change only if an upstream artifact changes.

## Todo List
- [ ] `tools/eval-subtask-decomposition.py` created and deterministic
- [ ] Grouping rule enforced: fold-based wording everywhere, `[LIMITATION]` banner attached, mixing assertion in code
- [ ] Per-sub-class recall × 6 encoders, with Wilson CIs
- [ ] Sub-class × fold cross-tab with the single-group rule stated generally
- [ ] Four-way sub-task rollup with verdicts
- [ ] Panel-disclosed variant computed + justification paragraph bound to the numbers
- [ ] Carry-lookup baseline run; disclosure decision applied per the pre-registered rule
- [ ] Class-blinded E3 LCO ablation run per sub-task; zone-result wording rule applied
- [ ] `N_MAX = 8` fix-list produced and handed to Lane D
- [ ] `reports/test-touch-log.md` updated: 0 touches
- [ ] **G2 declared**: tables frozen, architecture search formally closed

## Success Criteria
| Check | Pass condition |
|---|---|
| Reproducibility | one command regenerates every table in the finding from disk artifacts |
| Uncertainty | every recall cell carries n and a Wilson CI |
| Variant integrity | panel-disclosed number never appears without the full number in the same table |
| Disclosure | carry triviality decided by the pre-registered ≥95% rule, not by taste |
| Cost | 0 test touches, 0 GPU hours, <2 min runtime |

## Risk Assessment
| Risk | L | Impact | Mitigation |
|---|---|---|---|
| Fold/view conflation in prose (they agree only 81.7%) | **Confirmed possible** | High | Fold-based wording is mandatory; `[LIMITATION]` banner rides with every LCO number; emitted tables assert non-mixing; phase 10 greps for view-phrased claims |
| Committee reads panel-disclosure as cherry-picking | **High** | **High** | Always paired with the full number; justification paragraph is part of the artifact; R8 argument written out; declared before any further experiment |
| Wilson CIs on n=32 swallow the effect | High | Med | Expected — the claim is structural (single camera group), not a small-effect claim; lead with the cross-tab, not the recall |
| Sub-class from `clip_id[0]` mis-parses | Low | High | Assert parsed distribution equals the known 207/107/142/56/75/38/32/30 before emitting anything |
| Recomputation silently disagrees with the finding | Med | Med | Script asserts the finding's published numbers as a regression test; any delta is investigated, not overwritten |

## Security Considerations
- Read-only over local artifacts; no footage decoded, no network. Nothing leaves the machine.
- The cross-tab exposes camera/site structure — keep view ids abstract (`f0..f7`) in any public
  artifact, as the finding already does.

## Next Steps
- **Gates phase 03**: L1's falsification bar (76.6% panel-disclosed, E3) is only meaningful once this
  phase has frozen how that number is computed.
- **Gates phase 04**: if the walkway failure is structural, an appearance upgrade cannot be the fix —
  step 4's rollup decides whether phase 04 is worth its 6h at all.
- Hands Lane D the chapter section content for phase 09 and the `N_MAX` fix-list.

**Unresolved:** whether `walkway_ok`'s 3-group spread is enough for LCO to score it fairly, or whether
it is a weaker version of the same confound. (Fold-vs-view agreement and `N_MAX` are resolved —
81.7% and 8 respectively; both are now enforcement tasks, not questions.)
