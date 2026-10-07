# Phase 03 — Walkway Geometry Repowering (the one accuracy bet)

## Context Links
- [plan.md](plan.md) · [risks.md](risks.md) · [phase-02](phase-02-lco-confound-decomposition-and-reporting.md)
- `reports/finding-2026-09-19-lco-failure-is-camera-label-confounding.md` §4, §6
- `reports/finding-2026-09-19-detector-starvation-root-cause.md` — why no retrain happens here
- Code: `build_relation_tokens.py` (`geometry()`, `process_clip()`, `FEAT=10`, `N_MAX=8`, `IMGSZ=800`),
  `annotation_guide/zones/view_*.json`, `tools/check_zone_separation.py`, `tools/extract_zones_from_coco.py`

## Overview
- **Priority** P1 — the **only** mechanism-driven accuracy lever left after the detector and
  architecture levers died.
- **Status** not started. Gated on G2.
- **Effort** 6h · **Lane A** (token rebuilds) with Lane B running the evals · **Gate G3**
- **Scope redirected 2026-09-19** (`reports/finding-2026-09-19-what-each-subtask-actually-rests-on.md`):
  zone-**entity injection** is CUT (insight 9) and polygon repair for `view_2`/`view_7` is CUT
  (insight 7). What remains is repowering the walkway geometry channels that already exist. No
  hand-drawing is asked of the user; the kill gate has been run and L1 survives it.
- **0 test touches.** All evaluation is pooled LCO-691 / k-fold-691 over the 691 pool.

## Key Insights
1. The walkway sub-task (282 clips, the largest) is the real open failure: ~90% unsafe recall vs
   15-36% safe. The model learned *"walkway scene ⇒ Unsafe"*, not *"is this person on the walkway"*.
2. `Safe_Walkway` (detector class 2) and `Restricted_Zone` (class 3) have **zero instances** in the
   label set. The walkway is therefore representable **only** through hand-drawn polygons — while the
   zone sub-task, whose geometry channels are populated, is exactly the one that generalises. That is
   a coherent, falsifiable mechanism, and it is the bet this phase makes.
3. Zones **are** wired in already (token dims 6-9: `iou_walkway`, `iou_restricted`, `foot_in_walkway`,
   `norm_dist`). Never write "zones are unused" — the accurate statement is that they are (a) absent
   from the detector label space, (b) computed for person classes 0/1 only, (c) zero for `view_2`/`view_7`, which is **correct behaviour**, not a gap (insight 7).
4. `iou_walkway` is likely **degenerate by construction**: IoU of a person box against a large
   walkway polygon is near-zero for everyone, on or off the walkway. `norm_dist` — the channel the
   working zone sub-task leans on — is scale-free. Replacing box-polygon IoU with a
   containment fraction is a cheap, mechanism-targeted fix, not a tuning knob.
5. No detector retraining anywhere: zone entities are injected as **synthetic tokens at build time**
   from the polygons (class id 2/3, polygon bbox, conf 1.0). Relabelling the dataset and retraining
   is explicitly out of scope for 3 weeks.
7. **The 86 zero-geometry clips need no repair.** They are *exactly* 8-class labels 3 and 7
   (`carry_overload` 56 + `carry_safe` 30) — the carry sub-task — and all six encoders get
   **86/86 of them correct** under LCO. Zero headroom. Forklift load-state does not need walkway
   polygons, and the detector solves it outright. This **strengthens R13**: the carry sub-task is
   solved with geometry identically zero, which is direct evidence it is a detector class lookup and
   not relational reasoning. That sentence belongs in the thesis.
8. **Sub-task success tracks detector-class separability, not model quality.** carry and panel are
   separated by **detector class alone** (saturated / unanswerable); zone by **class + geometry**
   (76-90% both sides); walkway by **geometry alone** (17-36%). The relational component is the
   weakest link — which is why the only honest lever left is geometry, and why fixing it is worth
   more to the thesis than any accuracy point it buys.
9. **Zone-as-entity-token is a trap, not a lever.** (a) The person-zone relation is *already* encoded
   in dims 6-9 — that is precisely why the zone sub-task works; classes 2/3 would be a second
   encoding of the same information, not missing information. (b) A polygon is **static per camera**,
   so a zone token is a per-view **constant** and carries camera identity — the exact shortcut LCO
   exists to delete. Expected effect: neutral-to-harmful under LCO while in-distribution k-fold
   *rises*. That failure mode **looks like success on the wrong protocol** (R39).
10. **The measured defect is channel power, not channel absence.** `foot_in_walkway` averages 0.229 on
   `walkway_ok` where a person genuinely on the path should approach 1.0 — only ~23% of person-frames
   register. `iou_walkway` is useless by construction (0.014-0.020: a person bbox against a large
   polygon cannot produce meaningful IoU). `norm_dist` carries no signal for this pair (0.223 vs
   0.228). Classes 2/3 were **never annotated** (0 instances in 6,933 labels) — a genuine data gap —
   but the polygons themselves are rich (7 view files, 1-3 walkway + 1-5 restricted each, full
   vertices at 1920×1080), so every candidate below is instantiable today.
11. `yolo26_best.pt` built the headline tokens although `best.pt` (YOLO11s) scores higher in-domain
   (0.9752 vs 0.9658). That choice was never justified — resolve it once, by A/B, and never revisit.

## Requirements
**Functional** — (a) classes 2/3 provenance settled before any rebuild; (b) zone entities occupy token
slots; (c) containment-based geometry channels available behind a flag; (d) one A/B on detector
checkpoint; (e) every variant evaluated with E3 under LCO-691 and k-fold-691 against the baseline.
**Non-functional** — every new token set written under a distinct `--out_tag`; baseline tokens never
overwritten; each build logged with its full argv; new scripts <200 lines, kebab-case.

## Architecture
```
annotation_guide/zones/view_{0,1,3,4,5,6}.json   (view_2/view_7 absent BY DESIGN - carry sub-task)
        │
        v
build_relation_tokens.py
   ├─ geometry()          dims 6-9, person classes only   [+ containment variant, flagged]
   ├─ zone-entity inject  emit class-2/3 tokens per frame [new, flagged]
   └─ --detector {yolo26_best.pt | detector_best.pt}      [A/B]
        │
        v   tokens_<tag>{train,test}.pt   (Lane A owns)
        │
        v   run_kfold_relation.py --binary --arch E3 --lco   (Lane B, 51s/run)
        │
        v   tools/eval-subtask-decomposition.py  -> panel-disclosed BA vs the 76.6% bar
```
Token width grows from 74 to 74 (+0) for the A/B, to 74 for zone injection (slots consumed, not
features), and to 76 for the containment variant (2 extra channels). `N_MAX = 8` is a hard budget —
injected zone tokens **compete with entity detections for slots**; this is the main design risk.

## Related Code Files
**Modify** (Lane A only)
- `build_relation_tokens.py` — add `--geom_variant {iou,signed_dist,foot_fix,temporal}`;
  keep defaults identical to today so the baseline tag is bit-reproducible.
**Create**
- `tools/zones-coverage-audit.py` (<120 lines) — per-view polygon presence, per-clip geometry
  non-zero rate, cross-tab against sub-task and fold id.
- `tools/tokens-build-matrix.sh` (<60 lines) — the exact build commands for every tag in this phase.
**Delete** — none. Baseline tokens are immutable evidence.

## Implementation Steps
Cheapest-first; each step is separately evaluable, and the phase stops as soon as the bar is cleared
or the candidates are exhausted.
1. **[M4-CPU, 45 min] Per-view walkway polygon placement audit.** Before changing any feature, check
   whether the polygons are in the right place: if a walkway polygon is mis-drawn for a view, every
   downstream channel is measuring the wrong region and no feature engineering can rescue it.
   ```bash
   cd /Users/nguyenthiphuong/Learning/Proposal/project/my-project
   .venv/bin/python tools/check_zone_separation.py
   .venv/bin/python tools/zones-coverage-audit.py --per-view-foot-rate \
     --tokens sweep_results/relation_tokens/tokens_bin8_train.pt \
     --view-map sweep_results/zone_prior/clip_view_map_691.json \
     --subtask-json reports/sub-task-tables.json
   ```
   Report per view: fraction of `walkway_ok` person-frames with `foot_in_walkway == 1`. A view far
   below the 0.229 pool average is a placement bug, not a modelling problem — and fixing one polygon
   is cheaper than every other candidate combined.
2. **[M4-MPS, 30 min build] Signed foot-to-boundary distance replacing IoU.** Drop `iou_walkway`
   (useless by construction) in favour of a signed distance from the foot point to the walkway
   boundary, normalised by person height so it is scale-free — the same property that makes
   `iou_restricted`/`norm_dist` work for the zone sub-task.
   ```bash
   .venv/bin/python build_relation_tokens.py --appearance --appearance_backbone yolo \
     --detector sweep_results/entity_detector/yolo26_best.pt --device mps \
     --classes 0,1,2,3,4,5,6,7 --geom_variant signed_dist --out_tag geomsd_ \
     2>&1 | tee sweep_results/relation_tokens/build_geomsd.log
   ```
3. **[M4-MPS, 30 min build] Foot-point estimation fix.** `geometry()` takes the foot as the bbox
   bottom-centre, which is wrong under occlusion, truncation and oblique views — a plausible cause of
   the 23% registration rate. Use a robust foot estimate (bottom-centre clamped to the visible box,
   with a tolerance band around the polygon edge) and rebuild as `--out_tag geomfoot_`.
4. **[M4-MPS, 30 min build + 5 min eval] Temporal aggregation of foot-in over T=16.** A single frame's
   foot-in is a noisy bit; the clip-level question is whether the person is *on the path*. Add a
   per-entity temporal summary (fraction of valid frames with foot-in, and its longest run) so the
   head does not have to learn the aggregation from 16 noisy bits. Rebuild as `--out_tag geomtemp_`.
5. **[M4-MPS, 30 min build] L4 detector A/B** — identical settings, YOLO11s instead of YOLO26.
   ```bash
   .venv/bin/python build_relation_tokens.py --appearance --appearance_backbone yolo \
     --detector sweep_results/entity_detector/detector_best.pt --device mps \
     --classes 0,1,2,3,4,5,6,7 --out_tag y11s_ \
     2>&1 | tee sweep_results/relation_tokens/build_y11s.log
   ```
6. **[M4-CPU, ~10 min total] Evaluate every tag with the frozen encoder only.** E3, 5-fold LCO ≈51s
   per run; k-fold for the in-domain column. **No architecture variation** — that lever is closed.
   ```bash
   for tag in bin8_ geomsd_ geomfoot_ geomtemp_ y11s_; do
     .venv/bin/python run_kfold_relation.py --binary --arch E3 --tag $tag --lco \
       --dedupe drop_conflict --dedupe_task binary --seeds 3 \
       --scene_map sweep_results/zone_prior/clip_scene_map_691.json \
       2>&1 | tee -a sweep_results/relation_tokens/phase03_lco.log
     .venv/bin/python run_kfold_relation.py --binary --arch E3 --tag $tag --k 5 --repeats 3 \
       --dedupe drop_conflict --dedupe_task binary --seeds 3 \
       2>&1 | tee -a sweep_results/relation_tokens/phase03_kfold.log
   done
   ```
7. **[M4-CPU, 30 min] Score against the bar.** Run `tools/eval-subtask-decomposition.py` on each new
   OOF dump; compare panel-disclosed LCO BA to **76.6% (E3 baseline)**; McNemar exact on paired
   clip-level predictions (same pool, same folds — **never** a two-proportion z-test); BCa bootstrap
   CI on the BA difference; per-sub-class deltas so a walkway-specific effect is visible even if the
   pooled number does not move.
8. **[15 min] Verdict, written before looking at any further result.**

### Pre-registered decision rules (write the verdict, do not negotiate it)
| Variant | ADOPT if | ABANDON if |
|---|---|---|
| Any geometry variant (L1) | panel-disclosed LCO BA > **76.6%** *and* the paired BCa CI on ΔBA excludes 0 *and* `walkway_ok` recall rises | BA ≤ 76.6% or CI spans 0 → **written up as a negative result in the thesis**, mechanism stated, not buried |
| Protocol trap | — | **any variant that improves k-fold while flat/worse on LCO is rejected, not reported as a win** — that is the camera-identity signature (R39) |
| `geomsd_` (signed distance) | same bar | same → report the degenerate-IoU hypothesis as tested and rejected |
| `geomfoot_` / `geomtemp_` | same bar | same → report the foot-registration hypothesis as tested and rejected |
| `y11s_` (L4) | — | never abandoned, never repeated: whichever wins becomes the reported token set, with the A/B table published either way |
| Polygon placement (step 1) | a mis-placed view is found and corrected | placement is sound everywhere → the defect is the feature, not the annotation; proceed to step 2 |

## Todo List
- [x] Provenance kill gate — classes 2/3 **never annotated** (0/6,933); polygons rich; L1 survives, redirected
- [ ] Per-view walkway polygon placement audit; any mis-placed view corrected first
- [ ] `build_relation_tokens.py` extended with `--geom_variant {iou,signed_dist,foot_fix,temporal}`, defaults unchanged
- [ ] Baseline tag re-built once with defaults to prove bit-reproducibility before trusting any delta
- [ ] `geomsd_`, `geomfoot_`, `geomtemp_`, `y11s_` token sets built and logged with full argv
- [ ] E3 LCO + k-fold run for all four tags
- [ ] McNemar exact + BCa CI computed on paired predictions
- [ ] Verdicts written per the table above, including any negative result
- [ ] **G3 declared**; `reports/test-touch-log.md` updated (0 touches)

## Success Criteria
| Check | Pass condition |
|---|---|
| Separability | each geometry candidate evaluated independently, never bundled |
| Reproducibility | baseline rebuild reproduces the existing tokens bit-for-bit (or the diff is explained before proceeding) |
| Statistics | every own-vs-own comparison uses McNemar exact + BCa; no unpaired test appears anywhere |
| Honesty | a failed L1 produces a written negative result, not a quiet drop |
| Cost | ≤2.5h of builds, ≤15 min of training, 0 test touches |

## Risk Assessment
| Risk | L | Impact | Mitigation |
|---|---|---|---|
| L1 fails the 76.6% bar | **High** | Headline accuracy gain disappears | Pre-committed: it becomes a negative result supporting the confounding claim (phase 11); the thesis does not depend on it |
| Injected zone tokens starve entity slots (`N_MAX=8`) | Med | Unsafe recall drops | Cap at 2 injected tokens/frame; explicit revert rule; report the slot budget as a finding |
| Walkway failure is label ambiguity, not geometry | Med | L1 dies | Four independent candidates; if all four fail, the negative result is strong evidence the walkway pair is not separable by geometry at n=282, which is itself reportable |
| Rebuild does not reproduce the baseline | Med | All deltas untrustworthy | Made an explicit gate in the todo list before any comparison |
| A/B winner differs between LCO and k-fold | Med | Ambiguous choice | Pre-commit: **LCO decides**, in-domain reported alongside |
| Six comparisons invite multiplicity | Med | False positive | All four tags declared in advance; Holm correction across the four primary BA comparisons; stated in the results table |

## Security Considerations
- Polygon drawing displays raw workplace footage containing identifiable people — local only, no
  frames into the repo, no cloud tooling for annotation.
- Token builds run locally on MPS (inference only, allowed); nothing uploaded to Kaggle/Colab.
- Zone polygons encode site layout; treat `annotation_guide/zones/*.json` as sensitive and keep view
  ids abstract in published figures.

## Next Steps
- Feeds phase 05 (calibration on whichever token set wins) and phase 06 (frozen suite).
- Feeds phase 07/08: the winning tag defines the checkpoint that gets exported to the Jetson.
- If L1 lands, phase 09 gains an accuracy section; if it dies, phase 11's wording is invoked on 10-04.

**Unresolved:** whether the walkway failure is polygon placement, foot-point estimation, temporal
aggregation, label ambiguity, or genuinely unlearnable at n=282 — the four steps separate the first
three; whether `walkway_ok`'s 3-group spread is enough for LCO to score it fairly. (Classes 2/3
provenance is CLOSED: never annotated, 0 of 6,933 — a genuine data gap.)
