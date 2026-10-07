# Phase 01 — Detector Evidence Package (LANDED, bank it)

## Context Links
- [plan.md](plan.md) · [risks.md](risks.md)
- `reports/finding-2026-09-19-detector-starvation-root-cause.md` — the falsification + the mAP table
- `reports/audit-2026-09-19-repo-capability-and-feasibility.md` — capability audit
- `reports/finding-2026-09-19-lco-failure-is-camera-label-confounding.md` — why classes 2/3 matter downstream
- `annotation_guide/dataset/data.yaml` — `nc: 8`, classes 2/3 have zero instances
- `sweep_results/entity_detector/train_run.log` — the abandoned MPS run (stage-2 val mAP identically 0)

## Overview
- **Priority** P1 (blocks nothing, but is the cheapest thesis deliverable in the plan)
- **Status** ~80% landed. Measurement done 2026-09-19; only packaging + two re-runs remain.
- **Effort** 2h · **Lane A** · **Gate G1**
- Turn the already-measured detector numbers into a defence-ready evidence package, and **close the
  retrain lever permanently**. No training of any kind happens in this phase.

## Key Insights
1. The "undertrained detector" premise is **false**. `yolo26_best.pt` (the checkpoint that built the
   headline tokens, per `sweep_results/relation_tokens/build_bin8_2026-09-16.log` line 1) measures
   **mAP50 0.9658 / mAP50-95 0.8720 / P 0.9196 / R 0.9448**. `best.pt` = `detector_best.pt` (YOLO11s,
   md5-identical to two other filenames) measures **0.9752 / 0.8883 / 0.9418 / 0.9560**. Both are
   100-epoch GPU runs (embedded `train_args`: Colab and Kaggle respectively).
2. `stage1_frozen/weights/best.pt` (0.4620) is a 3-epoch / 10%-data CPU smoke test whose weights
   appear **nowhere downstream**. It is the origin of the false premise; say so explicitly.
3. Minority detection survived the 11.8:1 imbalance — Person_Authorized R=0.899. Do **not** build a
   narrative on imbalance-induced safe-recall collapse.
4. The detector mAP had never been recorded anywhere (both runs lived on ephemeral Colab/Kaggle
   filesystems). "What is your detector's mAP?" was previously unanswerable at defence. It now is.
5. `train_run.log` is the *evidence* for the MPS non-convergence rule, not a defect in the pipeline:
   stage-1 val `all 343 988 0.000483 0.00421 0.000248 9.85e-05`, every stage-2 epoch `0 0 0 0`.

## Requirements
**Functional**
- A thesis-ready detector table (overall + per-class) with provenance for every row.
- Two mandatory caveats attached to the table wherever it appears.
- A provenance appendix: md5s, embedded `train_args`, which checkpoint built which token tag.
**Non-functional**
- Zero test-touch cost. Zero GPU hours. Everything re-runnable in <10 min on M4.

## Architecture
```
annotation_guide/dataset/data.yaml ──┐
                                     ├─ yolo val ──> results (P/R/mAP50/mAP50-95, per-class)
sweep_results/entity_detector/*.pt ──┘                    │
                                                          v
                          reports/detector-evidence-2026-09-19.md  (Lane A)
                                                          │
                                        (read-only) ──> number-sheet.md (Lane D, phase 09)
```
Detector is **frozen** for the remainder of the project. Its only remaining degree of freedom is the
A/B checkpoint choice, evaluated downstream in phase 03 (L5), never by retraining.

## Related Code Files
**Create**
- `plans/2026-09-19-thesis-recovery-architecture/reports/detector-evidence-2026-09-19.md` — the package
- `tools/detector-eval-table.py` (<120 lines) — runs `yolo val` over the 3 checkpoints at both
  imgsz 640 and 800, dumps one markdown table + one JSON. Idempotent, no training.
**Modify** — none outside Lane A.
**Delete** — none. `stage1_frozen/` is kept as evidence.

## Implementation Steps
1. **[M4-MPS, 4 min]** Re-run val at the deployment resolution. The landed table was measured at
   imgsz 640 but tokens are built at **imgsz 800** (`build_relation_tokens.py:IMGSZ`). Mismatch is a
   defence liability.
   ```bash
   cd /Users/nguyenthiphuong/Learning/Proposal/project/my-project
   for ck in sweep_results/entity_detector/yolo26_best.pt \
             sweep_results/entity_detector/detector_best.pt; do
     .venv/bin/yolo val model=$ck data=annotation_guide/dataset/data.yaml \
       imgsz=800 split=val device=mps project=sweep_results/entity_detector/val800 \
       name=$(basename $ck .pt) 2>&1 | tee -a sweep_results/entity_detector/val800.log
   done
   ```
2. **[M4-CPU, 20 min]** Write `tools/detector-eval-table.py` wrapping step 1 so the table regenerates
   from one command; emit `sweep_results/entity_detector/detector-eval.json`.
3. **[30 min]** Write `reports/detector-evidence-2026-09-19.md` containing: overall table (3 rows ×
   both resolutions), per-class table for `yolo26_best.pt`, provenance table (filename → md5 →
   `train_args` epochs/imgsz/device/fraction → date → which token tag it built).
4. **[15 min]** Write the two caveats verbatim, to be pasted under every appearance of the table:
   > *(i) This validation split is frame-level and drawn from the **train** partition, so it is an
   > in-domain detector score. It does not establish cross-camera detector quality and is not
   > comparable to a clip-level held-out number. (ii) `data.yaml` declares `nc: 8`, but classes 2
   > (Safe_Walkway) and 3 (Restricted_Zone) have zero instances; the mAP is therefore averaged over
   > the 6 populated classes.*
5. **[10 min]** Add the baseline-paper comparison note: the reference paper reports "around 97%" read
   off a **training curve** for a `nc: 4` gate at 416×416 with no published test-set mAP
   (`paper/paper.txt`). Our number is a val-set mAP at 800 px over 6 populated classes — state that
   the comparison is indicative, not like-for-like.
6. **[10 min]** Record the MPS rule with its evidence pointer, so no future agent re-litigates it:
   *Ultralytics detection training must run on Kaggle/Colab GPU or CPU, never MPS — evidence
   `sweep_results/entity_detector/train_run.log`.*
7. **[DROPPABLE, M4-MPS, 40 min]** Cross-camera detector eval: hold out the frames belonging to one
   background cluster, val on it, report the in-domain-minus-OOD detector gap. Only do this if
   phase 02/03 produce a claim that leans on detection quality.

## Todo List
- [ ] Re-run `yolo val` at imgsz 800 for both production checkpoints
- [ ] `tools/detector-eval-table.py` created, emits markdown + JSON
- [ ] `reports/detector-evidence-2026-09-19.md` written (overall, per-class, provenance)
- [ ] Two caveats written verbatim and marked "paste under every table appearance"
- [ ] Baseline-paper comparison note added with the not-like-for-like disclaimer
- [ ] MPS training rule recorded with log pointer
- [ ] **G1 declared**: detector frozen, retrain lever closed, Kaggle hours released
- [ ] [DROPPABLE] cross-camera detector eval

## Success Criteria
| Check | Pass condition |
|---|---|
| Table completeness | 3 checkpoints × {P, R, mAP50, mAP50-95} at imgsz 800 + per-class for the production checkpoint |
| Provenance | every row traces filename → md5 → `train_args` → token tag it produced |
| Caveats | both appear verbatim in the report and in the number sheet entry |
| Reproducibility | one command regenerates the table from the checkpoints |
| Cost | 0 test touches, 0 GPU hours |

## Risk Assessment
| Risk | L | Impact | Mitigation |
|---|---|---|---|
| imgsz-800 re-run scores materially lower than the 640 run | Low | Table changes | Report the 800 number as primary (matches tokens); keep 640 in the appendix; either way the lever stays closed |
| Committee reads 0.966 as an end-to-end claim | Med | Credibility | Caveat (i) is mandatory everywhere; number sheet stores it as a field, not a footnote |
| Someone reopens the retrain | Med | Burns 10+ GPU h | This phase is the written kill record; risks.md carries it |
| `nc: 8` with dead classes read as sloppiness | Med | Credibility | Caveat (ii) + unresolved question 3 in plan.md (annotated-and-lost vs never-annotated) |

## Security Considerations
- Checkpoints and dataset stay local; nothing uploaded. `yolo val` is offline — confirm no telemetry
  (`YOLO_OFFLINE=1`, `settings sync=False`) before running so no frames leave the machine.
- Footage contains workplace video of identifiable people: figures in the thesis must use the same
  face-handling as existing chapters; no raw frames in the public repo.

## Next Steps
- Unblocks nothing — phases 02-05 do not depend on G1. Run it first anyway because it is 2h for a
  deliverable that is otherwise missing at defence.
- Hands Lane D (phase 09) a finished table + caveats on day 1 for the placeholder draft.
- Hands phase 03 the L5 A/B framing: YOLO11s outscores YOLO26 in-domain (0.9752 vs 0.9658) yet YOLO26
  built the headline tokens and that choice was never justified.

**Unresolved:** (1) cross-camera detector quality unmeasured; (2) were classes 2/3 annotated and lost,
or never annotated — relabel to `nc: 6` vs genuine data gap.
