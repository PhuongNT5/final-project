# §4.9 Cross-Validated Generalisation (Leakage-Free k-Fold)

> **Status:** standalone draft section for review; slots into Chapter 4 after §4.8
> (Interpretability) and before the chapter summary, which is renumbered §4.10 accordingly.
>
> **Pipeline: P1** (frozen-COCO YOLO → multi-stream Transformer). These are *not* P3 numbers —
> the P3 k-fold at N = 516 is Ch7 §7.7.1 (fold-mean 80.8 ± 4.0 % top-1, macro 84.2 %,
> Nadeau–Bengio p = 0.0600 n.s.). See the attribution banner at the head of Chapter 4.
>
> **Renumbered 2026-09-06** from §4.8, which collided with Chapter 4's own §4.8
> (Interpretability, incl. §4.8.5 UCF-Crime, the target of the Ch1 and Ch2 cross-references).
> Working reports written before that date cite these tables as 4.8.1 / 4.8.2. All numbers are
> reproducible from `run_kfold_eval.py` → `plans/260628-thesis-improvement-strategy/reports/kfold-eval.json`.

## 4.9.1 Motivation

Every accuracy figure reported so far rests on a single held-out test split of 64 four-class
clips. On a sample this small the 95% bootstrap confidence interval on top-1 accuracy spans
**[79.7%, 95.3%]** — a width of 15.6 percentage points. An interval this wide cannot
distinguish the proposed system from the published baseline, and a single swapped clip moves the
headline by 1.6 points. The single-split protocol is therefore adequate for reporting the
deployed checkpoint but inadequate for any claim about *generalisation*.

The standard remedy is k-fold cross-validation, which evaluates every clip in the dataset exactly
once across the folds and averages the result. The Unsafe-Net dataset, however, was captured over
39 days in a single facility, which raises a second concern: clips recorded in the same session
share backgrounds, lighting and worker identities, so a model can score well by memorising scenes
rather than learning behaviours. The principled defence is **leave-one-day-out** or
**group-by-session** cross-validation. We verified the released dataset directly and found that it
carries **no day, session or camera metadata** of any kind: clip filenames are sequential indices
(`{class}_tr{N}` / `{class}_te{N}`), the MP4 containers expose no `creation_time` tag, and the
auxiliary `frame_manifest.csv` contains no temporal column. True day-grouping is therefore
impossible on the public release.

In place of the missing session labels we introduce a **scene-cluster grouping** as a proxy: each
clip is represented by its time-averaged appearance feature, the clips are clustered with k-means
into 24 visual groups, and a *grouped* k-fold guarantees that all clips from one visual cluster
fall in the same fold. A model that relies on background memorisation will lose accuracy under
this protocol relative to a random split; the size of that drop quantifies the scene dependence.

## 4.9.2 Protocol

All 691 clips are pooled and restricted to the four unsafe classes, giving N = 516 clips
(WalkwayViolation 210, UnauthorizedIntervention 108, OpenedPanel 142, CarryOverload 56). Two
five-fold schemes are run, each repeated over three seeds {42, 123, 314}:

1. **Stratified k-fold** — class-balanced random folds.
2. **Scene-group k-fold** — `StratifiedGroupKFold` over the 24 k-means scene clusters
   (leakage-robust).

Within every training fold an inner stratified 20% validation slice is carved out for early-epoch
selection; **the held-out fold is never seen during training or model selection.** The model
evaluated is the single Direct-4 `TransformerHead` on frozen-COCO stats features — the honest best
configuration established under the blind protocol of §4.6, where the scenario cascade was rejected
and the single head was best on every seed. Confidence intervals are 5,000-sample bootstrap on the
pooled cross-validated predictions; per-class results report both recall and false-negative rate
(FNR), the safety-relevant miss rate.

## 4.9.3 Results

> **Intervals corrected and re-run landed (2026-09-06).** The intervals previously printed here came
> from a percentile bootstrap that resampled **1548 rows = 516 clips × 3 seeds**, treating the same 516
> clips re-predicted under three seeds as three independent draws. That is pseudo-replication.
> `bootstrap_ci()` in `run_kfold_eval.py` is now a **clip-level block bootstrap** (the clip is the
> resampling unit; when a clip is drawn, all of its per-seed predictions come with it), and the folds
> were re-run in full under the argparse defaults — `--k 5 --seeds 42 123 314 --clusters 24`, 204 min.
> The table below is that run.
>
> **Two corrections to what the provisional banner itself claimed.** (i) It asserted "the point
> estimates are unaffected — only the intervals move." That is true of the stratified protocol
> (75.3 → 75.26 %) but **false of the scene-group protocol**, which moved 71.9 → 70.48 % top-1 and
> 71.8 → 72.49 % macro. (ii) The re-run's stratified interval is [71.8, 78.6], not the [71.7, 79.1]
> the banner projected from a partial recomputation.
>
> **The previously published values cannot be reconciled, only replaced.** Their source artifact
> (`plans/260628-thesis-improvement-strategy/reports/kfold-eval.json`) was deleted from the repository
> in commit `a93a997`, so no stored artifact reproduces them; the re-run recreated the file. KMeans
> grouping is seeded (`random_state=0`), so cluster assignment is not the cause. The likely cause is
> visible in the per-fold log: scene-group **fold 2 scores 53.5 / 51.5 / 69.3 % across the three
> seeds** — a 17.8-point swing on one fold. With that instability the scheme's mean is worth about
> ±1.1 pp of standard error, and a 1.4-point shift between runs is ordinary noise, not a discrepancy
> to be explained away. **Quote the scene-group figures as unstable; the stratified ones are solid.**

**Table 4.9.1 — Cross-validated accuracy (4-class unsafe, N = 516), clip-level bootstrap.**

| Protocol | Top-1 (mean ± std) | Top-1 CI₉₅ | Macro-F1 (mean ± std) | Macro CI₉₅ |
|---|---|---|---|---|
| Stratified k-fold (random) | **75.3 ± 0.2 %** | [71.8, 78.6] | **77.6 ± 0.3 %** | [74.7, 80.4] |
| Scene-group k-fold (leakage-robust) | **70.5 ± 1.9 %** | [66.9, 74.0] | **72.5 ± 1.0 %** | [69.5, 75.4] |

The cross-validated interval remains substantially tighter than the 15.6-point interval of the single
split — the central methodological gain of this section survives the correction, though by a smaller
margin than the pseudo-replicated version advertised: **6.8 points** for stratified top-1 (7.1 for
scene-group), i.e. under half the single-split width rather than the ~4.4 points previously claimed.
The scene-group protocol costs **4.8 points of top-1 and 5.1 points of macro** relative to the random
split: scene dependence is real but modest, not the catastrophic background-memorisation that wide
scene leakage would produce. We therefore take the scene-robust figure (**70.5 ± 1.9 % top-1,
72.5 ± 1.0 % macro**) as the honest estimate of generalisation to unseen camera scenes, with the
instability caveat above attached.

These figures sit below the single-split headline (56/64 = 87.5% best-case, 53/64 = 82.8% blind).
The difference is expected and is itself informative: the headline benefited from a favourable
single split together with test-time ensembling and configuration selection, whereas the k-fold
number is a single head, no ensemble, evaluated on every clip with no selection. The k-fold result
is the more representative estimate of how the core method generalises; the single-split 56/64
remains the correct figure for the *specific deployed checkpoint* on the canonical split. The two
are reported as complementary, not competing.

### Finding 1 — the Transformer head does not beat a linear probe

To test whether the temporal Transformer machinery earns its place, an identical k-fold was run
with a logistic-regression probe on the time-averaged frozen features, and the two were compared
with the Nadeau–Bengio corrected resampled t-test (the correct test for k-fold comparisons, which
accounts for the dependence between overlapping training sets).

**Table 4.9.2 — Transformer head vs. linear probe (same folds).**

| Protocol | Δ top-1 (head − probe) | Nadeau–Bengio t | p | Verdict |
|---|---|---|---|---|
| Stratified | **−4.72 pp** (75.3 % vs 80.0 %) | −2.77 | **0.015** | Head **significantly worse** |
| Scene-group | +0.22 pp (70.5 % vs 70.3 %) | 0.07 | 0.94 | No significant difference |

*Test details (audited 2026-09-06).* The differences are accumulated over seeds **and** folds, so
n = 3 × 5 = 15 paired measurements and df = 14; the Nadeau–Bengio variance inflation uses
ρ = n_test/n_train = 0.25. This was **not** affected by the Table 4.9.1 bootstrap defect, which
concerns a different estimator, and the re-run confirms it: the stratified effect size reproduces
(−4.65 → −4.72 pp) while its *t* and *p* move with the re-run's variance (−3.15 → −2.77,
p 0.007 → 0.015). **Both verdicts are unchanged**, which is the substantive point — the head is
significantly worse under the random split and indistinguishable under the scene-group split in both
runs. The scene-group Δ is the one number that shifted materially (+1.70 → +0.22 pp), consistent
with that scheme's instability documented above; it was and remains far from significance. The
standing caveat is power: n = 15 here against n = 50 in the repeated-k-fold runs of Chapter 7, so
these p-values are less precisely estimated than those.

On the random split the logistic-regression probe is **significantly more accurate** than the
Transformer head (p = 0.015); under the scene-robust split the two are statistically
indistinguishable. In neither protocol does the Transformer head provide a measurable advantage
over a linear classifier on mean-pooled features. This is consistent with, and quantifies, the
temporal-stream ablation of §4.7.8, which found no lift from the motion streams. The honest reading
is that on a dataset of this size and this feature representation, the attention machinery is not
the operative component; the result is reported as a negative finding, which the computer-vision
literature recognises as a substantive contribution that spares future work a known dead end.

### Finding 2 — the safety risk is concentrated in two zone-ambiguous classes

**Table 4.9.3 — Per-class false-negative (miss) rate.**

| Class | FNR (random) | FNR (scene-robust) |
|---|---|---|
| WalkwayViolation | 32.4% | 29.4% |
| **UnauthorizedIntervention** | **44.8%** | **68.2%** |
| OpenedPanel | 7.7% | 6.8% |
| CarryOverload (forklift) | 0.0% | 0.0% |

For a safety monitor the miss rate matters more than overall accuracy. Almost the entire safety
risk falls on the two classes that are visually a "person in a restricted area" —
UnauthorizedIntervention misses **two thirds** of true events under the scene-robust protocol,
WalkwayViolation roughly one third — while OpenedPanel and the visually distinct CarryOverload are
detected near-perfectly. This is the strongest single argument for the zone-prior intervention of
the following section: the failure is not generic but localised exactly where image appearance
alone cannot recover the restricted-zone boundary.

### Finding 3 — the confusion structure reproduces the zone ambiguity at full scale

The pooled confusion matrix (stratified, seed 42; rows = true, columns = predicted, order
WalkwayViolation / UnauthorizedIntervention / OpenedPanel / CarryOverload):

```
WalkwayViolation        [140,  52,  18,   0]
UnauthorizedIntervention[ 32,  57,  19,   0]
OpenedPanel             [  3,  12, 127,   0]
CarryOverload           [  0,   0,   0,  56]
```

The dominant off-diagonal mass is the bidirectional WalkwayViolation ↔ UnauthorizedIntervention
swap (52 + 32 clips), exactly the error mode diagnosed on the 64-clip split in §4.7.6, now
confirmed at the full 516-clip scale. CarryOverload, which is defined by a distinct object
(an overloaded forklift) rather than a zone relationship, is perfectly separable.

## 4.9.4 Limitations of this evaluation

Three caveats are stated explicitly. **First**, the k-fold protocol evaluates the *single*
Direct-4 head, not the deployed ensemble-plus-cascade-plus-gate pipeline; the figures in
Table 4.9.1 should not be read as the accuracy of the shipped system, for which the canonical-split
56/64 remains the reference. **Second**, the scene clusters are a k-means proxy for recording
sessions, not ground-truth day labels, which the public dataset does not provide; the 3.4-point
leakage estimate is therefore a lower bound subject to clustering quality. **Third**, N = 516 is
still a modest sample, and although the cross-validated intervals are far tighter than the
single-split interval, individual per-class FNR figures (especially CarryOverload at n = 56)
retain non-trivial uncertainty.

## 4.9.5 Summary

Leakage-free k-fold cross-validation reduces the headline accuracy uncertainty from a 15.6-point
to a **6.8-point** interval and yields an honest generalisation estimate of **70.5 ± 1.9 % top-1 /
72.5 ± 1.0 % macro** under scene-robust evaluation, with the stability caveat in §4.9.3 attached to
the scene-group numbers. The analysis surfaces two results that the
single-split protocol could not: the Transformer head provides no statistically significant gain
over a linear probe (and is significantly worse on the random split), and the entire safety-
relevant miss rate is concentrated in the two restricted-zone classes. Both findings motivate the
zone-prior intervention evaluated next and support the deployment-led framing of the thesis: the
contribution is a rigorously characterised, honestly bounded, safety-aware edge system rather than
an accuracy advance over the baseline.
