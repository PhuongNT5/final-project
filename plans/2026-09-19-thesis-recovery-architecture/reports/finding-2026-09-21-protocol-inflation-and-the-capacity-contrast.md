# Finding 2026-09-21 — Protocol inflation, and what it does to the capacity ladder

**Gates adjudicated:** G12 **PASS**. G13 **NOT a clean pass** (conservative reading, §8).
**Test touches spent: 0 of 2.** No number here touches the frozen blind split.

Supersedes nothing. Sits beside `finding-2026-09-20-isafetybench-g10-first-positive-accuracy-result.md`
(read its §9/§10 first — G11 failed there, and nothing in this report revives it).

## 1. What this settles

Two questions, one design.

**(a) Why do the baseline papers report numbers we cannot reach?** Answered in §9: one paper does not
in fact report high numbers (different task, 16-option MCQ, GPT-4o at 40.3%); the other reports
96.88% one clip below a *provably corrupted* ceiling on a 64-clip test set under a random split over
2 cameras. Neither is a number our protocol could or should produce.

**(b) How much of the gap is protocol rather than model?** Measured, not argued: **+15.4 pp** on
iSafetyBench for the small encoder, **+10.4 pp** for the large one, from changing *only* how folds are
drawn. Same tokens, same encoders, same 10 folds, same epochs, same seed.

The result that matters most was not the one predicted. Leakage does not merely *inflate* accuracy —
it **destroys the protocol's power to tell models apart**. A 17x parameter difference that is worth
+5.1 pp (p=7.7e-06) under the identity-respecting protocol is worth **+0.1 pp (p=0.665)** under the
leaking one. A leaking protocol cannot support *any* architecture claim, including a favourable one.

## 2. Design (pre-registered 2026-09-20, `risks.md` G12/G13)

Both gates were written into `risks.md` **before `--split random` existed in the code** and before any
random-split number had been produced. Neither wording was edited after results were seen.

2 corpora x 2 fold-construction protocols x a capacity ladder. The only thing that varies within a
corpus is fold construction:

| corpus | identity-respecting | identity-leaking | what the identity is |
|---|---|---|---|
| iSafetyBench | **LSO-1100** `StratifiedGroupKFold(10)` | **RAND-1100** `StratifiedKFold(10)` | YouTube source video id (151) |
| Unsafe-Net | **LCO-691** leave-camera-out (8 scene groups) | **k-fold-691** `StratifiedKFold` | scene / camera group (8) |

Why leakage is total on iSafetyBench, not partial: **0 of 151 sources carry both classes** (R41). A
source is therefore a *perfect* label proxy. Put clips from one source on both sides of the boundary
and the task degenerates into source recognition.

Resampling unit is the cluster (source / camera group), never the clip. Every own-vs-own comparison
is paired: cluster bootstrap + exact McNemar. Clip-level CIs are printed for disclosure and are the
wrong unit.

## 3. iSafetyBench — the 2x2

`tools/protocol-inflation-table.py`, log `sweep_results/relation_tokens/logs/g12-protocol-inflation-table.log`.
Balanced accuracy recomputed from the saved per-clip OOF tensors, **not** parsed from training logs,
so a disagreement with a log would be a real one. There was none.

| encoder | params | LSO-1100 (valid) | RAND-1100 (leaking) | Δ_inflation |
|---|---:|---:|---:|---:|
| E3 DeepSets | 17,270 | 75.2% | 90.6% | **+15.4 pp** |
| E1 d64 | 93,222 | 78.1% | 91.3% | +13.2 pp |
| E1 d96 | 195,558 | 77.0% | 91.9% | +14.9 pp |
| E0 RelationTransformer | 298,118 | **80.3%** | 90.7% | **+10.4 pp** |
| **spread across a 17x ladder** | | **5.1 pp** | **1.4 pp** | |

Paired Δ_inflation, as G12 required (`logs/g12-inflation-paired.log`):

| encoder | Δ | cluster CI over 151 sources | McNemar exact |
|---|---:|---|---|
| E3 | +15.4 pp | [+10.6, +21.5] **excludes 0** | b=186 c=8, p=3.58e-45 |
| E0 | +10.4 pp | [+6.3, +15.3] **excludes 0** | b=150 c=22, p=1.3e-24 |

Direction of the discordance is itself evidence: for E3, 186 clips flip wrong→right when leakage is
introduced and only 8 flip right→wrong. This is not noise around a shared decision boundary.

**Weaker models inflate more** (E3 +15.4 vs E0 +10.4). Mechanism: the leaking protocol has a ceiling
near 90.6% that even a 17k-parameter head reaches, so inflation = ceiling − true skill. I predicted
the opposite to the author earlier in the session and was wrong.

## 4. Unsafe-Net — the same 2x2

Recomputed from pre-existing OOF tensors; **no Unsafe-Net model was re-trained for this report**.

| encoder | params | LCO-691 (valid) | k-fold-691 (leaking) | Δ_inflation |
|---|---:|---:|---:|---:|
| E3 DeepSets | 18,044 | **69.6%** | 82.5% | +13.0 pp |
| E1 d64 | 94,380 | 62.3% | 84.1% | +21.8 pp |
| E1 d96 | 197,292 | 63.9% | 84.5% | +20.5 pp |
| E2 frame-positional | 284,556 | 66.7% | 84.6% | +17.9 pp |
| E0 RelationTransformer | 298,892 | 66.7% | 83.6% | +16.9 pp |
| **spread across a 17x ladder** | | **7.3 pp** | **2.1 pp** | |

Param counts differ by <1k from the iSafetyBench table because Unsafe-Net carries an 8-way auxiliary
sub-class head. Same encoders otherwise.

Under the valid protocol the **smallest** encoder is nominally best. Under the leaking one the
ordering is essentially flat and the small model's advantage disappears.

## 5. The control that rules out an arithmetic explanation

A sceptic's first move: *maybe random folds are just easier*. The R42 count-rule floor answers it.
Same three hand-made features (persons/frame, objects/frame, mean confidence), same two protocols,
same folds as the encoder:

| | LSO-1100 | RAND-1100 | Δ |
|---|---:|---:|---:|
| count-rule floor (3 features) | 67.2% | 68.2% | **+1.0 pp** |
| E3 encoder | 75.2% | 90.6% | +15.4 pp |

A 3-feature rule has **no capacity to memorise source identity**, and it gains +1.0 pp. The encoder,
which does, gains +15.4 pp. The inflation is memorisation of identity, not a property of the split
arithmetic. This control is the load-bearing piece of the protocol critique and should appear in the
thesis next to the headline table.

(The floor figure also records a fixed defect: `run_kfold_relation.py` previously rebuilt the floor's
folds and ignored `--split`, scoring the floor on leave-source-out folds while the encoder ran on
random ones — and still printing "identical folds". Fixed by deriving the floor's splits as the set
complement of the encoder's own held-out indices; regression in `tests/test-split-protocol-modes.py`;
the two affected logs were regenerated. `risks.md` carries the full record.)

## 6. Headline — leakage destroys discriminative power

Largest encoder (E0, ~298k) minus smallest (E3, ~18k). Positive = capacity helps.
`logs/g13-capacity-contrast.log`.

| corpus | protocol | Δ (E0 − E3) | cluster CI | McNemar |
|---|---|---:|---|---|
| iSafetyBench | LSO-1100 **valid** | **+5.1 pp** | [+2.0, +8.3] **excludes 0** | p=**7.74e-06** |
| iSafetyBench | RAND-1100 leaking | **+0.1 pp** | [−1.2, +1.5] includes 0 | p=**0.665** |
| Unsafe-Net | LCO-691 **valid** | −2.8 pp | [−7.1, +0.1] includes 0 | p=0.606 |
| Unsafe-Net | k-fold-691 leaking | +1.1 pp | [−1.6, +5.2] includes 0 | p=0.0271 † |

† the only cell where McNemar and the cluster CI disagree. With **8** clusters the bootstrap is thin;
the conservative CI reading is the one adopted. Disclosed, not resolved.

Nothing on the random iSafetyBench ladder is significant: the widest pair there (E1 d96 − E3, the
1.4 pp spread) is +1.4 pp, CI [−0.4, +3.0], McNemar p=0.105.

Two statements follow, and the second is the more useful one:

1. A leaking protocol reports ~90–92% where the honest protocol reports 75–80%.
2. **A leaking protocol reports ~90–92% for everything.** It cannot distinguish a 17k-parameter
   DeepSets head from a 298k-parameter relation transformer. Any ablation, any architecture
   comparison, any "our module helps" claim measured under it is measuring nothing. This is a
   stronger and more transferable criticism than "the number is too high", because it invalidates the
   *comparisons* in such papers, not only their absolute figures.

Second-order observation, stated with its limits: under the valid protocol the capacity ranking
**reverses between corpora** — Unsafe-Net nominally prefers the smallest encoder, iSafetyBench
significantly prefers the largest. Only the iSafetyBench direction is statistically supported; the
Unsafe-Net direction is a point estimate whose CI includes 0. See §8.

## 7. G12 adjudication — **PASS**

Pre-registration, verbatim: *"Δ_inflation > 0 and materially large (>= 5 pp) for at least one
encoder"*; falsified *"if Δ_inflation < 3 pp for BOTH E3 and E0 with CIs including 0"*.

Measured: E3 **+15.4 pp** and E0 **+10.4 pp**, both CIs excluding 0. Prediction met with margin;
falsification condition not remotely approached. **G12 passes.**

Two disclosures:

- G12 fixed *"encoders E3 + E0 only"*. E1 d64 and E1 d96 were also run under the random split, to
  complete G13's contrast. **G12 is adjudicated on E3 and E0 as written**; the two E1 rows are
  disclosure, and they could only have weakened the verdict (they did not — +13.2 and +14.9 pp).
- The random-split numbers are **diagnostic artefacts**. Per pre-registration they may never appear
  as this thesis's accuracy in any table, abstract or slide, and every appearance carries the label
  "source-leaking protocol, shown for comparison". That constraint is inherited by every document
  downstream of this report.

## 8. G13 adjudication — **NOT a clean pass**

Pre-registration: accuracy increases with capacity on iSafetyBench while Unsafe-Net stays flat — *the
claim is the CONTRAST between corpora*. Falsified if the ladder is flat (max−min < 3 pp) **or** if
*"the E0−E3 gap is not reproduced by the intermediate encoders showing an ordered trend"*.

- Flatness clause: **does not fire.** Spread 5.1 pp > 3 pp.
- Ordered-trend clause: **fires.** E1 d96 (195,558 params, 77.0%) sits **1.1 pp below** E1 d64
  (93,222 params, 78.1%). The ladder is not monotone.

My pre-registered wording "ordered trend" was ambiguous — it did not state whether monotonicity or
a positive association was required. That ambiguity is a drafting error of mine, and the rule is that
pre-registration is not renegotiated after seeing results. Resolving an ambiguity in the direction
that favours our own claim is exactly the move the rule exists to prevent. **Therefore the
conservative reading is adopted: G13 does not pass cleanly.**

Consequences, precisely:

- **Permitted:** "capacity class matters where extractable signal exists, and does not where it does
  not" — supported by +5.1 pp (p=7.7e-06) on iSafetyBench against −2.8 pp (CI includes 0) on
  Unsafe-Net.
- **Forbidden:** "balanced accuracy increases with parameter count". The E1 d96 inversion falsifies
  it, and it must not be smoothed over by plotting only E3 and E0.
- **Forbidden:** using any of this to revive G11. G11 named E3 and stays failed.
- R47 (invariance falsified as a *general* claim) survives in its already-scoped form: invariance was
  a symptom of an unlearnable task on Unsafe-Net, not a property of the encoder family. The present
  data supports that scoping and does not license a stronger version.

## 9. Why the baseline papers report what they report

The author's question. Answered from the papers, not from inference.

**iSafetyBench (arXiv 2508.00399, ICCV-W 2025) — the premise is wrong: the paper does not report
high numbers.** Its task is a **16-option multiple-choice** question (1 correct, 15 distractors) over
165 action categories, **zero-shot**, with all 1,100 clips used as the evaluation set. Table 2
single-label accuracy (normal / hazard): Ovis2-8B 47.3/40.3 · InternVL2.5-8B-MPO 42.2/38.3 ·
Qwen2.5-VL-7B 46.9/33.6 · VideoLLaMA3-7B 38.9/32.7 · VideoChat-Flash-7B 31.0/26.8 · Oryx-7B 25.0/21.5
· Valley-Eagle-7B 48.8/35.9 · **GPT-4o 40.3/37.3**. The paper's own conclusion is that models
struggle. Chance is 6.25% there and 50% in our binary task; the numbers are not comparable in either
direction, and our 80.3% is not "beating GPT-4o".

**Unsafe-Net (Önal & Dandıl) — the number is real but the measurement is not sound.** 62/64 = 96.88%
top-1 (95.81% macro) on a **64-clip** test set. That test set contains byte-identical duplicate clips
carrying contradictory labels (`0_te23` ≡ `4_te9`; `0_te8` ≡ `1_te1`), so a perfect classifier cannot
exceed **63/64 = 98.44%**. 96.88% is one clip below a provably corrupted ceiling. 12.5% of test clips
have broken labels. The split is random over footage from 2 IP cameras, with no scene-only control —
precisely the protocol this report measures at +13 to +22 pp of inflation on the same corpus.

So the honest comparison is not "96.88% vs our 69.6%". It is: under that paper's protocol our
encoders score **82.5–84.6%** (§4), and under a protocol that does not leak camera identity they
score **62.3–69.6%**. The remaining distance to 96.88% is a 64-clip test set with a corrupted ceiling.

**This is the thesis's most defensible contribution and it should be stated as such.** Not "we did
worse", but: the field's headline numbers on this task are produced by protocols that inflate by
10–22 pp *and* cannot distinguish a 17k-parameter model from a 298k-parameter one — demonstrated on
two corpora, with a 3-feature control proving the mechanism is identity memorisation.

## 10. What the thesis may and may not claim after this report

**May claim** (each with the evidence above):

1. Protocol inflation is 10–22 pp across two corpora, five encoders, ten cells — measured, paired,
   cluster-bootstrapped.
2. The mechanism is identity memorisation — the 3-feature floor gains only +1.0 pp.
3. Leaking protocols are non-discriminative: 17x capacity = +0.1 pp (p=0.665) under leakage vs
   +5.1 pp (p=7.7e-06) without it. **This is the headline.**
4. Capacity class matters where signal exists and not where it does not (the corpus contrast, §8).
5. Published baselines on these corpora inherit (1)–(3); the Unsafe-Net baseline additionally sits
   one clip below a corrupted ceiling on 64 clips.

**May not claim:**

- Accuracy increases with parameter count (E1 d96 inversion, §8).
- That the detector + relation head add accuracy on iSafetyBench (G11 failed; the detector-free
  whole-frame probe is statistically indistinguishable *and* structurally cheaper — G10 finding §9).
- Any random-split number as an accuracy result, anywhere.
- Anything about generalisation from the Unsafe-Net capacity ordering alone (CI includes 0).

The accuracy pillar still has no surviving positive claim. What this report adds is not an accuracy
result — it is a **methodological** result, and on the evidence it is stronger than the accuracy claim
the plan originally went looking for. Whether the thesis reframes around it is the author's call.

## 11. Traceability

| artefact | path |
|---|---|
| 2x2 tables, recomputed from tensors | `tools/protocol-inflation-table.py` → `logs/g12-protocol-inflation-table.log` |
| paired Δ_inflation (G12) | `logs/g12-inflation-paired.log` |
| capacity contrast, 4 cells (G13) | `logs/g13-capacity-contrast.log` |
| per-run training logs | `logs/{lso,g12-rand}-isb-{E3,E1-d64,E1-d96,E0}.log` |
| protocol-mode regression | `tests/test-split-protocol-modes.py` (PASS) |
| dataset invariants | `tests/test-isafetybench-dataset.py` (ALL PASS) |

Method notes, disclosed:

- The Unsafe-Net random k-fold tensors were saved without clip ids or group labels. They are
  row-aligned with the leave-camera-out tensors of the same corpus; `compare-oof-runs.py --ids_from`
  supplies the identity columns and **asserts label equality row by row** before trusting that
  alignment, refusing otherwise. Recomputed balanced accuracies match the originals exactly.
- Unsafe-Net cluster bootstraps use **8** clusters. That is thin. Treat every Unsafe-Net CI here as
  indicative; the iSafetyBench CIs (151 clusters) carry the weight.
- No model was re-trained for this report except the four iSafetyBench random-split runs that G12
  required.

## 12. Unresolved questions

1. Does the thesis reframe around the protocol critique as its primary contribution? Author's
   decision. This report supplies the evidence; it does not make the call.
2. The Unsafe-Net capacity inversion (smallest encoder nominally best, CI includes 0) — report as a
   null, or investigate? Investigating costs runs that the 2–3 week deadline may not have.
3. Should G13 be **re-run** with an unambiguous pre-registration (monotonicity vs positive
   association, stated in advance) rather than left as a non-clean pass? Cleaner, but it is a new
   gate, not a rescue of the old one, and it must be written before the runs.
4. E1 d96 < E1 d64 at double the parameters — optimisation artefact (60 epochs, fixed LR, no tuning
   per capacity) or real? Untested. If optimisation, the ladder is uninformative at the top end and
   the honest statement shrinks to "E0 > E3".
5. Whether the two pre-registered test touches stay pointed at the Unsafe-Net blind split. Still
   0 of 2 spent, still pointed there, unchanged by this report.
