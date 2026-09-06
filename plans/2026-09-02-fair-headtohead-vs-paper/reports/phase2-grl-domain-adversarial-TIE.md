# Phase 2 · GRL / domain-adversarial (DANN) — TIE (champion retained on parsimony)

**Date:** 2026-09-06 · **Verdict:** TIE (LCO macro +0.2 pp; within noise) → champion retained (YAGNI/KISS) ·
**Lever:** objective-axis (not features)

## Why this lever
Every FEATURE lever failed the same way — scene-coupled features leak camera identity (k-fold↑/LCO↓) or
shift WalkwayViol priors (zones-detector −2.3, L2 homography −5.6, L5 proximity, motion −0.5). GRL attacks
the LCO gap from the OTHER side: it adds no features, it changes the objective so the shared representation
is camera-invariant. A domain (camera-group) classifier is attached to the CLS representation through a
gradient-reversal layer (Ganin & Lempitsky 2015); the reversal makes the backbone UNlearn camera identity
while keeping behaviour signal. Domain labels exist only for the N−1 training cameras (held-out camera
unseen) → trained strictly within the LCO protocol, no test leakage.

Implementation (additive): `model_relation_head.py` gained an optional `return_repr` flag (default off →
champion path byte-identical); `run_lco_grl.py` is self-contained (GradReverse, domain head, DANN λ-ramp,
per-fold val-selected λ). `run_lco_eval.py` (the champion's eval) untouched.

## Result — two stages
### Stage 1: λ=1.0 pre-registered (DANN default) — REJECTED (−16.0)
| LCO recall | GRL λ=1.0 | Champion | Δ |
|---|---|---|---|
| WalkwayViol | 74.3 | 68.1 | **+6.2** |
| UnauthorInt | 72.2 | 70.4 | **+1.8** |
| OpenedPanel | **17.6** | 89.4 | **−71.8** |
| CarryOverload | 100.0 | 100.0 | 0 |

LCO macro 66.0% (−16.0). **First and only lever to lift BOTH sides of the core WalkwayViol↔UnauthorInt
confusion.** The −16.0 is entirely OpenedPanel collapsing (115/142 panels → UnauthorInt): λ=1.0
over-regularized, scrubbing the scene context OpenedPanel depends on. Textbook too-strong-λ signature →
motivates a val-selected λ sweep (λ is a hyperparameter; selecting on train-view val, never test, is
protocol-legal).

### Stage 2: λ swept {0.05,0.1,0.3,1.0}, per-fold val-selected — TIE (+0.2)
| Metric | GRL (val-sel λ) | Champion | Δ |
|---|---|---|---|
| **LCO macro** (headline) | **82.2%** | **82.0%** | **+0.2 pp** |
| LCO top-1 | 78.1% | 78.9% | −0.8 |

| LCO recall | GRL (val-sel) | Champion | Δ |
|---|---|---|---|
| WalkwayViol | 68.6 | 68.1 | +0.5 |
| UnauthorInt | **72.2** | 70.4 | **+1.8** |
| OpenedPanel | 88.0 | 89.4 | −1.4 |
| CarryOverload | 100.0 | 100.0 | 0 |

Most folds selected weak λ (0.05–0.1) on val — confirming λ=1.0 over-regularized. Val-selection recovered
OpenedPanel (17.6→88.0) and retained a small UnauthorInt gain (+1.8, the harder class). Net LCO macro +0.2
= a statistical tie (within seed/fold noise).

## Decision — TIE, champion retained
+0.2 pp is not a meaningful improvement; it is indistinguishable from the champion. Adopting GRL would add a
domain adversary + a per-fold λ-tuning step for no measurable headline gain → rejected on parsimony
(YAGNI/KISS). GRL is nonetheless the campaign's strongest and only non-harmful lever, and the ONLY one that
improves the harder core-confusion class (UnauthorInt) without collateral damage — worth reporting in the
thesis discussion as evidence the WalkwayViol↔UnauthorInt residual is partly a camera-invariance problem,
not purely a feature-availability one.

## Leak check (reasoned, one item unmeasured)
Not a leak signature: GRL REMOVES camera-identifiable structure by construction, so it can only lower — never
raise — the scene-confounded k-fold. A +0.2 LCO with (expected) flat-or-lower k-fold is a clean tie, the
opposite of the k-fold↑/LCO↓ leak fingerprint. NOT re-measured: the GRL k-fold number (would need k-fold
domain-label wiring); since a tie is not adopted regardless, it does not change the verdict. Flagged as the
one unmeasured check.

## Artifacts (all additive — no narrative files touched)
- `run_lco_grl.py` (KEPT — reusable; GRL + val-selected λ sweep)
- `model_relation_head.py` `return_repr` flag (KEPT — optional, default-off, champion path unchanged)
- Logs: `sweep_results/relation_tokens/_grl_lco.log` (λ=1.0), `_grl_lco_sweep.log` (val-selected) — KEPT
- No token sets produced (GRL is a training-objective change, not a feature) → nothing to purge.

## Unresolved questions
1. GRL k-fold macro (leak double-check) — unmeasured; expected flat/lower, not a leak. Wire k-fold domain
   labels if a reviewer wants the number.
2. Would per-class-weighted domain loss (protect OpenedPanel explicitly) push GRL from tie to a small win?
   Plausible but risks overfitting the λ/weight to this scene grouping — the subtle leakage the campaign
   guards against. NOT recommended without a second independent scene grouping to validate against.
3. Campaign status: feature axis AND objective axis now both exhausted; champion 82.0 sits at the research
   reports' honest ceiling (LCO 82–84). Recommend freeze → run the untouched blind test set ONCE → write up.
