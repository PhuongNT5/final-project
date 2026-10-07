# Phase 04 — Appearance Probe + Upgrade (DROPPABLE)

## Context Links
- [plan.md](plan.md) · [phase-02](phase-02-lco-confound-decomposition-and-reporting.md) · [phase-03](phase-03-walkway-geometry-repowering.md) · [risks.md](risks.md)
- Code: `build_relation_tokens.py` (`get_appearance()`, `embed_boxes()`, `APP_DIM = 64`)
- `reports/finding-2026-09-19-lco-failure-is-camera-label-confounding.md` (why this is demoted)

## Overview
- **Priority** P3 — **DROPPABLE in full.** Cut it on any slip; it is not on the critical path.
- **Status** not started. Gated on G2; only runs if the phase-02 rollup leaves appearance plausible.
- **Effort** 6h · Probe = Lane B (existing tokens), rebuild = Lane A · **0 test touches**
- Two questions only: does the appearance block carry any cross-camera signal, and does the 1.3×
  context crop leak background into it?

## Key Insights
1. Appearance is **not noise**: `get_appearance()` runs a real pretrained/fine-tuned backbone (YOLO
   backbone GAP 512-d, or MobileNetV3-small 576-d). Only the 512→64 compression is a fixed seed-0
   Gaussian — a Johnson-Lindenstrauss projection of real features. **Do not overstate this as
   "random features"** in the thesis; it is a compression, and JL preserves distances.
2. `embed_boxes()` crops each box with **30% context padding** (`w/2*1.3`, `h/2*1.3`), so the
   appearance channel demonstrably contains background — the exact substrate for a scene shortcut,
   and the walkway failure is a scene-shortcut failure.
3. **R47 scope (2026-09-21):** architecture invariance holds **on Unsafe-Net only**. On iSafetyBench capacity is decisive (E0 - E3 = +5.1 pp, cluster CI [+2.0, +8.3], McNemar p=7.74e-06). Invariance is a symptom of an unlearnable, camera-confounded task - not a property of the encoder family. Every sentence below that says "architecture-invariant" means "architecture-invariant on this corpus" and must be written that way in the thesis.
   The architecture-invariance result (18k→300k params, identical per-sub-class failures) means a
   richer *encoder* cannot help. A different *representation* is not ruled out by that argument — but
   the prior is still low, which is why this phase is droppable rather than prioritised.
4. The cheap probe comes first and can kill the expensive rebuild: ablate the 64 appearance dims at
   load time; no rebuild, seconds per run.

## Requirements
**Functional** — appearance-free / geometry-free ablations under LCO-691 with E3 only; if and only if
appearance survives, one rebuild with zero-padding crops and one with a PCA-fitted projection.
**Non-functional** — PCA fitted on **training folds only**, never on the pool, never on test; each
rebuild under its own `--out_tag`; <200-line scripts.

## Architecture
```
tokens_bin8_*.pt ──> tools/eval-feature-ablation.py ──> LCO BA(E3) for
                       {full, appearance-free, geometry-free, class-id-blinded}
                                   │  gate
                                   v
build_relation_tokens.py --crop_pad 1.0        -> tokens_tightcrop_*.pt   (Lane A)
build_relation_tokens.py --app_proj pca        -> tokens_apppca_*.pt      (Lane A)
```

## Related Code Files
**Create** — `tools/eval-feature-ablation.py` (<120 lines) — masks feature blocks at load time, runs
E3 LCO, emits a markdown table.
**Modify** (only if the gate passes) — `build_relation_tokens.py`: `--crop_pad` (default 1.3, keeps
current behaviour) and `--app_proj {gauss,pca}` (default `gauss`).
**Delete** — none.

## Implementation Steps
1. **[M4-CPU, 20 min] Ablation probe.** E3, LCO-691, four variants: full / appearance-free (drop dims
   10-73) / geometry-free (zero dims 6-9) / class-id-blinded.
   ```bash
   cd /Users/nguyenthiphuong/Learning/Proposal/project/my-project
   .venv/bin/python tools/eval-feature-ablation.py --tag bin8_ --arch E3 --lco \
     --variants full,no_app,no_geom,blind_cls \
     --scene_map sweep_results/zone_prior/clip_scene_map_691.json \
     2>&1 | tee sweep_results/relation_tokens/phase04_ablation.log
   ```
2. **[10 min] Apply the gate.** *If appearance-free BA ≥ full BA (CI overlapping), the appearance
   block carries no cross-camera signal → **stop here**, report the ablation as a finding, and skip
   steps 3-5.* This result is publishable either way: it tells the reader what the model uses.
3. **[M4-MPS, 30 min, conditional] Tight-crop rebuild** (`--crop_pad 1.0`) to test the context-leak
   hypothesis directly. Same detector, same everything else.
4. **[M4-MPS, 30 min + 20 min fit, conditional] PCA projection rebuild** replacing the seed-0 Gaussian,
   **fitted on training folds only**. Under LCO this must be re-fitted per fold or it leaks camera
   information across folds — if that is too invasive to implement cleanly in the time available,
   **drop step 4 rather than fudge it**.
5. **[M4-CPU, 5 min] Evaluate** both tags with E3 under LCO + k-fold; McNemar exact + BCa vs baseline.

### Pre-registered decision rules
| Variant | ADOPT if | ABANDON if |
|---|---|---|
| Any appearance change | panel-disclosed LCO BA beats the phase-03 incumbent with BCa CI excluding 0 | otherwise → reported as a tested, rejected hypothesis |
| `no_app` ablation | — | not a lever; it is a *diagnostic* and gets reported regardless of sign |
| PCA per-fold refit | implementable cleanly in ≤2h | otherwise dropped; a pool-fitted PCA is leakage and is never shipped |

## Todo List
- [ ] `tools/eval-feature-ablation.py` written and run
- [ ] Gate applied and recorded; phase either continues or is closed with a finding
- [ ] [conditional] tight-crop tokens built and evaluated
- [ ] [conditional] per-fold PCA projection built and evaluated, or explicitly dropped for leakage risk
- [ ] Ablation table handed to Lane D
- [ ] `reports/test-touch-log.md` updated (0 touches)

## Success Criteria
| Check | Pass condition |
|---|---|
| Diagnostic value | the ablation table ships regardless of whether any lever adopts |
| Leakage | no projection or statistic is fitted on the pool or on test |
| Cost discipline | phase stops at step 2 unless the gate passes |
| Honesty | "random projection" is never claimed; "fixed seed-0 JL compression of pretrained features" is |

## Risk Assessment
| Risk | L | Impact | Mitigation |
|---|---|---|---|
| PCA fitted on the pool leaks camera identity | Med | **Invalidates result** | Per-fold refit or drop; explicit in the decision rules |
| Phase consumes days on a low-prior lever | Med | Schedule | Hard DROPPABLE marking; gate at step 2; cut on any slip |
| Tight crops lose person context and hurt everything | Med | Low | Expected outcome is informative either way; baseline is untouched |
| Ablation misread as a causal claim | Med | Med | Report as "what the model uses", not "what matters" |

## Security Considerations
- Rebuilds decode raw footage locally; MPS inference only; nothing uploaded.
- Appearance crops are person imagery — embeddings stay local, no third-party feature services.

## Next Steps
- Feeds phase 06 as an ablation table whatever the outcome.
- Cut this phase first if 03 overruns; the thesis does not depend on it.

**Unresolved:** whether the 1.3× context crop carries the scene leak (only a tight-crop rebuild
answers it, and only if the step-2 gate passes).
