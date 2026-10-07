# G10 PASSES — first positive accuracy result of the thesis (iSafetyBench, leave-source-out)

Date 2026-09-20. Lane A/B. Phase 12. Supersedes nothing; extends
`finding-2026-09-19-isafetybench-feasibility.md` (R42 gate) with the encoder result it gated.

**Result: E3 DeepSets reaches 75.2% balanced accuracy under leave-source-out on 1,100 clips /
151 source videos, +8.0 pp over the detector-count shortkey floor recomputed on identical clips and
identical folds (95% CI [+1.9, +13.5] pp, cluster bootstrap over sources; McNemar exact
p = 1.02e-05).** G10 was written into `risks.md` before the run. This is the first accuracy claim in
the project that survives its own pre-registered falsification rule.

**But G11 FAILS (§9), and G11 is the one the committee will ask about.** A whole-frame CNN probe with
no detector at all reaches 74.1% on the same clips and the same folds. E3's margin over it is
**+1.1 pp, cluster CI [-4.8, +6.9], McNemar p = 0.625.** Under the terms fixed before the number
existed, **the thesis may not claim the detector + relation head add accuracy on this corpus.** Read
§9 before quoting the headline: G10 is a true statement about the count-rule shortcut and nothing
more. **§10 then reports E0 at 80.3%, which beats the whole-frame baseline significantly — and why
that still does not make G11 pass.**

## 1. The floor moved UP before the encoder ran — use 67.2%, not 63.3%

The R42 gate probe used 300 clips at T=8. G10 says the floor is "recomputed on the same clips and the
same folds", so it was recomputed on all 1,100 at T=16, on the exact `StratifiedGroupKFold(10)`
partition the encoder used.

| count-rule (persons/frame, objects/frame, mean conf), logistic regression, class-weighted | BA |
|---|--:|
| **full corpus 1,100 @ T=16, leave-source-out, pooled OOF** | **67.2%** |
| gate probe, 300 clips @ T=8, GroupKFold(10) | 63.3% |
| same full-corpus rule, RANDOM stratified 10-fold | 68.2% |
| majority class | 50.0% |

Two readings, both load-bearing:

- **grouped − random = −1.0 pp.** The shortcut is barely scene-memorising. On Unsafe-Net the
  equivalent contrast was catastrophic. This is independent evidence that R41's residual source
  confounding is genuinely diluted, not merely argued away by counting groups.
- **The dominant separator is NOT person count.** `mean_conf` splits 0.518 (hazard) vs 0.628
  (normal); persons/frame is 1.453 vs 1.679 and objects/frame 2.949 vs 3.050 — far weaker than the
  probe's 1.32/2.08 and 2.51/4.48 suggested at T=8/n=300. **Hazard footage is simply harder to detect
  in.** That is a shortcut of a different kind — image quality and occlusion, not safety semantics —
  and it forces a conf-channel ablation before any "relational reasoning" wording. Registered below.

## 2. E3 under leave-source-out

`run_kfold_relation.py --binary --lco --dataset isafetybench --tag isb_ --arch E3 --folds 10
--seeds 3 --epochs 60`. 17,270 params (n_sub=2 shrinks the aux head vs Unsafe-Net's 18,044).

| metric | value |
|---|--:|
| balanced accuracy | **75.2%** |
| macro F1 | 73.9% |
| hazard recall | 78.3% |
| hazard precision | 63.4% |
| normal recall | 72.1% |
| ROC-AUC | 0.815 |
| PR-AUC (hazard) | 0.692 |

Per held-out fold, **top-1** (what the log prints): 87.5 / 73.8 / 74.8 / 66.0 / 62.3 / **50.9** /
82.1 / 86.8 / 84.0 / 71.7. The same folds in **balanced accuracy**
(`tools/analyse-isb-fold-failures.py`): 91.2 / 73.0 / 72.5 / 66.6 / 63.0 / **57.3** / 81.9 / 87.0 /
84.7 / 71.7. Quote whichever, but never the log's per-fold number as a BA — they are different
metrics and fold 6 differs by 6.4 pp between them.

**Fold 6 is near chance**, and the cause is a single source. Worst-group performance must be reported
beside the mean; a 30-point spread across held-out source packets is itself a finding about scene
dependence.

| fold 6 source | n | label | acc | objects/frame |
|---|--:|---|--:|--:|
| `0zQ9fI30v1E` | 31 | normal | **0.0%** | **8.00** |
| `0a-WuC1Xdl4` | 9 | normal | 0.0% | 3.38 |
| remaining 13 sources | 66 | mixed | 66.7–100% | 0.75–6.16 |

Fold 6's mean object density is 4.85/frame against a corpus mean of 3.01, almost entirely from
`0zQ9fI30v1E` — 31 clips of a **crowded normal scene, every one of them called hazard**. The failure
mode is therefore nameable: *the model over-predicts hazard on unusually crowded normal footage.*
Note this coexists with the ablation result that counts alone are not what it uses (§5) — the density
prior is real but is not the mechanism; it is what the appearance features correlate with here.
Because every source is single-class (R41), one bad source costs the whole fold, which is exactly why
grouped CV is the only honest protocol on this corpus.

**Is crowd density a corpus-wide false-positive driver, or just that source?** Quintiles of
objects/frame over all 1,100 clips:

| objects/frame | n | normals | false-positive rate | hazards | false-negative rate |
|---|--:|--:|--:|--:|--:|
| 0.00–1.38 | 205 | 93 | **41.9%** | 112 | 16.1% |
| 1.38–2.06 | 220 | 145 | 12.4% | 75 | 33.3% |
| 2.06–2.88 | 233 | 171 | 17.5% | 62 | 29.0% |
| 2.88–4.44 | 219 | 147 | 25.9% | 72 | 15.3% |
| 4.44–8.00 | 223 | 124 | **52.4%** | 99 | 19.2% |

The density–false-positive correlation over normals is r = +0.245 (p = 8.8e-11) — but **r = +0.058
(p = 0.14) once `0zQ9fI30v1E` alone is excluded.** So the *crowded* half of the story is one source,
not a corpus-wide law, and must be written as such.

What is *not* explained away is the **U-shape**: the sparsest quintile also misfires badly (41.9% FP
on 93 normals, a different set of sources entirely). The model is unreliable on atypical scene
density in **both** directions — near-empty and crowded normal footage both get called hazard — while
mid-density footage sits at 12–18% FP. That is an honest, reportable operating-envelope limitation
and belongs in the limitations section, not buried.

## 3. G10 adjudication

| G10 clause (pre-registered 2026-09-20, before the run) | outcome |
|---|---|
| beat the count-rule floor recomputed on same clips + same folds | **+8.0 pp** (75.2 vs 67.2) |
| paired bootstrap CI excludes 0 | **[+1.9, +13.5] pp** — excludes 0 |
| margin under ~3 pp is "not a result" regardless of sign | margin 8.0 pp — clears it |

McNemar exact on the paired per-clip correctness: b = 211, c = 129, 340 discordant,
**p = 1.02e-05**, paired top-1 difference +7.45 pp.

**Honest qualification: the CI's lower bound (+1.9 pp) sits BELOW the 3 pp triviality threshold.**
The point estimate clears the bar comfortably and the test is significant, but the data are also
consistent with a margin small enough that G10 would have called it not-a-result. Say "+8.0 pp
(95% CI +1.9 to +13.5)" and never "+8 pp" alone.

### Why a cluster bootstrap, and what it costs

Clips inside one source video share a scene, and the top 5 sources hold ~24% of the corpus, so
clip-level resampling treats near-duplicates as independent evidence. Resampling the 151 **sources**
instead gives [+1.9, +13.5] pp; clip-level resampling gives [+4.6, +11.4] pp — **~1.9× narrower, and
wrong.** New module `eval_utils/cluster_bootstrap.py` (49 lines) + 6 tests, one of which asserts the
cluster interval is wider than the clip interval on constructed within-group dependence, so a silent
regression to clip-level resampling fails the suite.

## 4. Defect found and fixed in passing

`eval_utils.paired_tests.mcnemar_exact` returned a hardcoded `'note': 'N=64: few discordant cells…'`.
It printed **"N=64"** beside an n=1,100 comparison with 340 discordant cells — the exact opposite of
what the data showed. Now derived from the input (`N={len(a)}, {n_disc} discordant cells`, with the
underpowered wording only when `n_disc < 10`) and `n_clips` added to the returned dict. Full
`eval_utils` suite still passes.

## 5. Ablations — what the +8.0 pp is actually made of

Every ablation is applied to **both** the encoder input and the R42 floor (`--ablate` feeds both), so
the two are always denied the same cue. `feature_ablation.py` reads column indices from the token
file's own `feat_layout` and raises if the layout ever moves; `tests/test-feature-ablation.py`
(12 tests, real tokens) asserts each mode zeroes exactly the intended columns and nothing else.

| condition | what survives | encoder BA | R42 floor BA | margin | margin 95% CI (cluster) |
|---|---|--:|--:|--:|---|
| full tokens | everything | 75.2% | 67.2% | +8.0 pp | [+1.9, +13.5] |
| conf zeroed | all but conf | 76.1% | **53.0%** | +23.2 pp | [+15.3, +31.4] |
| appearance zeroed | the 10 FEAT cols | 73.8% | 67.2% | +6.5 pp | [+1.8, +11.7] |
| class blinded | all but class id | 76.6% | 68.1% | +8.5 pp | [+2.2, +14.0] |
| no_boxes | all but xc/yc/w/h | 76.2% | 67.2% | +9.0 pp | [+2.4, +15.0] |
| boxes_only | xc/yc/w/h only | **64.2%** | 47.4% | +16.8 pp | [+7.3, +26.6] |
| **appearance_only** | **the 64 appearance dims only** | **75.6%** | 47.4% | +28.3 pp | [+19.5, +37.0] |

And the encoder against **itself**, paired on the same 1,100 clips (this is the table that matters):

| full vs … | Δ BA | 95% CI (cluster) | McNemar p | read |
|---|--:|---|--:|---|
| conf zeroed | −0.9 pp | [−2.2, +0.3] | 0.043 | within noise |
| appearance zeroed | +1.4 pp | [−4.3, +6.1] | 0.254 | within noise |
| class blinded | −1.4 pp | [−3.8, +1.2] | 0.097 | within noise |
| no_boxes | −1.0 pp | [−2.5, +0.5] | 0.034 | within noise |
| **appearance_only** | **−0.4 pp** | **[−2.9, +2.4]** | **0.551** | **within noise** |
| boxes_only | +11.0 pp | [+3.4, +18.8] | 2.6e-13 | **real drop** |

### Two conclusions, one good and one that costs the framing

**(a) The encoder is not riding the shortcut.** Removing detection confidence destroys the shortcut
(67.2% → 53.0%, essentially chance) while the encoder is *unaffected* (75.2% → 76.1%, CI on the
paired difference spans 0). The image-quality cue identified in §1 is therefore **not** what the
encoder is using. That is a far stronger statement than the headline margin alone, and it is the
answer to unresolved question 1 of the first draft of this report: **the +8.0 pp does not live in
`mean_conf`.**

**(b) All ten FEAT columns are dispensable; the signal is the crop appearance vectors.**
`appearance_only` — every one of class, xc, yc, w, h, conf and the four zero geometry columns
neutralised, leaving nothing but the 64 appearance dims — scores **75.6%, statistically
indistinguishable from the full model** (−0.4 pp, CI [−2.9, +2.4], McNemar p = 0.551). The converse
`boxes_only` is the only ablation that genuinely hurts (−11.0 pp, CI [+3.4, +18.8], p = 2.6e-13),
and even it stays 16.8 pp above its own floor.

So no single channel is *necessary* (conf, appearance, class and box geometry are each individually
removable within noise) but the appearance block is *sufficient* on its own. **This kills the
spatial-relational reading of the result on this corpus.** The honest description of what E3 does
here is: it pools ImageNet-derived appearance descriptors of detected entity crops over 16 frames.
That is a real and non-trivial model — it beats the detector-count shortcut by 8 pp under
leave-source-out, and 28.3 pp when the shortcut is denied its confidence cue — but it is not
evidence of reasoning over entity geometry, and the thesis must not say it is.

It also raises the decisive follow-up, now pre-registered as **G11** before its features finished
extracting: if crop appearance is the whole signal, does the *detector* buy anything over feeding the
whole frame to the same backbone? Section 8.

A caveat that must travel with the `blind_class` row: only **47.1%** of valid slots changed under it,
because the remaining 52.9% of detections are already `person` (group id 0). Class blinding is
therefore a weaker intervention on this corpus than it looks, and "class identity does not matter"
should be stated as "does not matter beyond person-vs-not, which is itself half the detections".

The floor moving under ablation is expected and is itself a check: `no_appearance` leaves the floor
at exactly 67.2% (the count rule never read appearance), while `blind_class` lifts it to 68.1%
because collapsing every class to one value makes persons/frame and objects/frame the same feature.
Both behaved as predicted before the runs.

## 6. Cross-dataset contamination — ruled out

`tools/check-cross-dataset-contamination.py`: 64-bit dHash on 3 frames per clip, 691 Unsafe-Net clips
(2,059 hashes) against 1,100 iSafetyBench clips (3,300 hashes), exact-intersection plus chunked
minimum cross-corpus Hamming distance.

- **exact frame-hash collisions: 0**
- **minimum cross-corpus Hamming distance: 13/64** (near-duplicate threshold 6), nearest pair
  `data/train/3_carrying_overload_with_forklift/3_tr3.mp4` vs
  `data/isafetybench/normal/MEEGwePfDgM_trim_49.mp4`

**VERDICT: no contamination.** Feasibility Q4 closed; R46 in `risks.md`. The two corpora are disjoint,
so nothing in the Unsafe-Net chapters leaks into this result and vice versa.

## 7. What this does and does not license

**Does:** a positive, pre-registered, grouped-CV accuracy result exists. The thesis no longer rests
solely on negative findings. `phase-12` §6's reassignment (accuracy → iSafetyBench, edge →
Unsafe-Net, Unsafe-Net → failure-analysis chapter) is now supported by evidence, not just by the
absence of alternatives.

**Does not:**
- It does **not** show relational reasoning — §5(b) settles this against us. `appearance_only`
  matches the full model, so the model is pooling appearance descriptors of entity crops, not
  reasoning over entity geometry. Geometry is identically zero on this corpus anyway (R45). Any
  sentence in the thesis using "spatial relation" for this result is false and must be cut.
- It does **not** show the *detector* is needed at all. That was G11, and **G11 failed** (§9): a
  detector-free whole-frame probe is within noise of the full pipeline. This is now settled against
  us, not pending.
- It does **not** transfer to Unsafe-Net. Different detector (COCO-80 folded to 8 groups), different
  label semantics, zero geometry. The corpora are at least verified disjoint (§6).
- It does **not** consume a test touch. Pooled grouped-OOF, 0 spent, both still pointed at the
  Unsafe-Net blind split.

## 8. Still running / next

- **G11 — closed, failed.** See §9.
- **E0 (299k params) leave-source-out** — architecture invariance. Folds 1-4 came in at
  92.4 / 81.3 / 77.6 / 76.4 vs E3's 92.4-equivalent folds 87.5 / 73.8 / 74.8 / 66.0 (top-1, same
  held-out sources); 10 folds needed before that means anything. **E0 cannot rescue G11** — the
  pre-registration forbids retrying the falsification with a different encoder, and E0 was already
  running when G11 was adjudicated. Whatever E0 lands at, it is an architecture-invariance row.

## 9. G11 — the whole-frame falsification baseline. **FAILED.**

The decisive question after §5: if no single token channel is necessary, does the detector buy
anything at all? G11 answers no.

Same 1,100 clips, same 151 sources, same 10 StratifiedGroupKFold leave-source-out folds, same
MobileNetV3-small ImageNet backbone and the same fixed 64-d projection — the only difference is that
the whole-frame probe never sees a detection. Three probes, and the **best of the three** is the
baseline (choosing the weakest would be grading our own homework):

| probe (no detector) | balanced acc | macro-F1 | ROC-AUC |
|---|--:|--:|--:|
| **linear-mean** | **74.1%** | 73.0% | 0.826 |
| linear-meanstd | 72.7% | 71.5% | 0.810 |
| mlp-meanstd | 73.6% | 73.2% | 0.808 |

**E3 (full entity pipeline) 75.2% vs best whole-frame 74.1%.**

| paired on the same clips | value |
|---|---|
| point margin | **+1.1 pp** |
| cluster bootstrap 95% CI over 151 sources | **[-4.8, +6.9] — includes 0** |
| clip-level CI (WRONG UNIT, disclosure only) | [-2.0, +4.2] |
| McNemar exact | b=138, c=129, n_disc=267, **p = 0.625** |

Both pre-registered criteria fail: the point margin is under the 3 pp triviality threshold *and* the
CI includes 0. McNemar is not marginal — 267 discordant clips split 138/129, which is as close to a
coin flip as this corpus can produce. The two models disagree on a quarter of the corpus and are
equally right when they do.

**Consequence, applied as written and not renegotiated:**
- The thesis **may not** claim the detector + relation head add accuracy on iSafetyBench.
- iSafetyBench is written up as a **parity / negative result** against a whole-frame CNN baseline.
- The accuracy pillar reverts to a **stated limitation** on this corpus; **edge efficiency carries
  the contribution** (the user's declared pillar order was accuracy > edge > robustness, so this is
  a demotion of the first pillar, not a cosmetic change).
- G10's +8.0 pp survives **only** as a statement about the count-rule shortcut. It is not a statement
  about whole-frame CNNs and must never be quoted as one.
- No retry with a different encoder, backbone, pooling, or threshold. Fixed as written.

This is the coherent ending of §5, not a surprise. §5 showed `appearance_only` matched the full model
— the detector's 10 geometry/class/conf columns carried nothing. §9 closes the loop: the per-entity
*crops* also carry nothing over the whole frame. What is working on this corpus is ImageNet
appearance statistics over 16 frames, with or without a detector in front of them.

**And the efficiency escape hatch does not exist either — the direction is the wrong way round.**
The tempting move is "equal accuracy at 17,270 parameters, so the contribution is efficiency". That
is false and must not be written. 17k counts the DeepSets head alone. To produce its input, E3 runs a
YOLOv11s detection pass *plus* up to N_MAX=8 MobileNetV3-small crop passes on every one of 16 frames.
The whole-frame probe runs **one** MobileNetV3-small pass per frame and no detector at all. The
baseline that matches our accuracy is strictly **cheaper**, by roughly a detector plus an order of
magnitude in backbone invocations. On this corpus the entity pipeline is dominated on both axes at
once. Exact FLOPs/latency belong to the edge lane; the ordering does not depend on that measurement.

What G11 leaves intact is narrower and must be stated as such: the pipeline produces *per-entity*
outputs (which worker, which object, where in frame) that a whole-frame classifier structurally
cannot, and that a deployed safety system arguably needs for an actionable alert. That is a utility
argument about output granularity, unvalidated here, and it is not an accuracy or efficiency claim.
Unresolved Q5 is rewritten accordingly.

## 10. E0 lands at 80.3% — and this is the most dangerous number in the report

E0 RelationTransformer (298,118 params), same 1,100 clips, same 10 leave-source-out folds:
**balanced acc 80.3%**, macro-F1 79.0%, hazard recall 84.0%, precision 68.9%, PR-AUC 0.747,
ROC-AUC 0.857. Floor on identical folds 67.2% -> **+13.1 pp**.

Paired, cluster bootstrap over the 151 sources:

| comparison | margin | cluster CI | McNemar |
|---|--:|---|---|
| E0 - whole-frame baseline | **+6.2 pp** | **[+2.1, +10.5] excludes 0** | b=140 c=76, **p=1.59e-05** |
| E0 - E3 | **+5.1 pp** | **[+2.0, +8.3] excludes 0** | b=102 c=47, **p=7.74e-06** |

### 10a. This does NOT make G11 pass. Read this before using the number.

The numbers clear G11's arithmetic — margin over 3 pp, CI excluding 0. **G11 still fails**, because
G11 named E3, and the pre-registration says in as many words: *"No retry with a different encoder, a
different backbone, a different pooling, or a tuned threshold."* Swapping the encoder after watching
the designated one fail is precisely the move that rule exists to stop. A falsification test that is
re-run with a new model until one passes is not a falsification test.

What is verifiable in our favour, and what is not:

- **Verifiable:** E0 was not launched in response to G11 failing. `lso-isb-E0.log` was created
  **19:08:43**; the whole-frame features only began extracting at **19:42:28** and the first
  whole-frame number appeared at **20:07:54**. E0 predates any knowledge of the baseline by ~1 h.
- **Not in our favour, and decisive:** E0 was planned as the *architecture-invariance cross-check*,
  explicitly not as a headline candidate. Phase-12 step 5 says so. The headline encoder was
  designated E3 and frozen. So the family here is not "two pre-planned headline candidates" that a
  Bonferroni correction would clean up (for the record, 1.59e-05 x 2 = 3.2e-05, still significant —
  but multiplicity is not the problem). The problem is that the *decision rule* "run both, report
  whichever beats the baseline" was never declared, and cannot be declared now.

### 10b. The one principled argument for E0, stated so the committee can weigh it

E3's designation as frozen headline rested on the **architecture-invariance finding** — six encoders,
18k-300k params, identical pass/fail pattern on Unsafe-Net — which licensed "take the cheapest,
architecture does not matter". **That premise is now falsified** (E0 - E3 = +5.1 pp, p=7.7e-06). The
reason E3 was the designated encoder is no longer true.

That is a real argument and not a rationalisation — but it does not retroactively make E0 the
pre-registered choice, and it cannot be settled with more analysis of this corpus, because nothing on
this corpus is untouched by the selection. It is an author-and-committee decision about how to
present two facts that are both true:
1. The pre-registered test (E3) failed.
2. A larger encoder, run as a cross-check, beats the detector-free baseline significantly.

**They must be presented together, in that order.** Reporting (2) without (1) would be misconduct.

### 10c. What the invariance falsification actually buys — a better thesis point

Do not write "architecture invariance was wrong". Write the scoped version, which fits both corpora:

> Encoder capacity was irrelevant on Unsafe-Net (18k-300k params, identical failure pattern) and
> decisive on iSafetyBench (+5.1 pp, p=7.7e-06). Invariance was never a property of the
> architecture — it was a symptom of a task with no extractable signal. Where signal exists,
> capacity buys accuracy; where the labels are camera-confounded, no encoder can help.

This is sharper than the original claim, it explains both datasets with one mechanism, and it turns
the Unsafe-Net negative result into evidence rather than an embarrassment. It also means every
document asserting invariance as a general property must be scoped to Unsafe-Net: `phase-04` §23,
`phase-09` step 5 (chapter-07 subsection), `phase-11`'s narrative template, `phase-12` §26.

### 10d. Consequence for the frozen deployment artifact

E3 is frozen for deployment and every ONNX/Jetson artefact is built from it. If E0 becomes the
accuracy story, the edge lane's measurements describe a model the thesis no longer headlines — E0 is
17x the parameters and the exported `p3-relation-head-bin-e0-fp32.onnx` already exists, but no Jetson
numbers exist for it. Choosing E0 therefore costs a re-measurement on device. That is a real,
schedulable cost and it belongs in the decision, not after it.

## Unresolved questions

1. ~~Does the +8.0 pp survive zeroing `mean_conf`?~~ **Answered: yes, fully** — the encoder is
   unchanged (75.2% -> 76.1%) while the shortcut collapses to 53.0%. §5(a).
2. ~~Fold 6 at chance — which sources?~~ **Answered:** one source (`0zQ9fI30v1E`, 31 crowded normal
   clips, 8.0 objects/frame vs corpus 3.01) predicted hazard throughout. §2. Open sub-question: is
   crowd density a systematic false-positive driver corpus-wide? **Also answered: no** — r drops
   from +0.245 to +0.058 (p=0.14) when that one source is removed. But the density-stratified
   breakdown surfaced a *new* limitation: FP rate is U-shaped in density (41.9% sparsest / 52.4%
   densest / 12–18% middle), so the operating envelope is genuinely narrow. §2.
3. Hazard precision 63.4% vs recall 78.3% — the operating point is untuned. Threshold/temperature on
   OOF only (phase-05 rules) could trade these; not done, and must not be selected after seeing any
   test number.
4. **G11 failed — so what is the thesis's accuracy contribution?** (no longer hypothetical) The pre-registration says the
   pillar reverts to a limitation and edge efficiency carries the work. That is a defensible thesis
   (a negative accuracy result + a real deployment measurement + a rigorous protocol critique), but
   it is a different thesis from the one the current chapter drafts assume, and the decision about
   how to frame it belongs to the author, not to this lane.
5. **Rewritten after G11.** The "turn §5(b) into an efficiency argument" escape is closed — the
   detector-free baseline is both equally accurate *and* structurally cheaper (§9). The surviving
   question is narrower and harder: does per-entity output granularity (which worker, which object,
   where) justify a pipeline that buys no accuracy and costs more compute? That needs an argument
   from deployment utility, and ideally a measurement the thesis does not currently have (e.g. does
   a localised alert change operator response?). If the answer is no, the honest thesis is a
   methodological/negative one. Author's call.
6. E0 result pending; if E0 substantially beats E3 the "architecture is irrelevant" claim weakens and
   the frozen-arch decision for deployment needs revisiting.
