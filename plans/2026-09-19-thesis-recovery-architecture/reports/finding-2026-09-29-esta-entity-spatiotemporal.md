# Finding 2026-09-29 — ESTA: entity spatio-temporal attention over YOLOv11 detections

Status: **complete. G14 FAILED.** The entity decomposition does not beat a whole-frame
representation at matched backbone. R48's underlying hypothesis — that the ceiling was the
appearance representation — is supported in direction but not certified at cluster resolution.

## Why this lane exists

R48. The seven-condition ablation on iSafetyBench showed the relational head contributes nothing:
`appearance_only` 75.6% vs full 75.2% balanced accuracy, p=0.551. The suspected cause is not the
head but the appearance representation the head is fed — MobileNetV3-small crushed through a
**fixed random projection 576 -> 64**. So the hypothesis under test is: *the ceiling is the
representation, and an entity-level representation with a real vision backbone lifts it.*

## What was built

```
video --16 frames--> YOLOv11s (conf 0.25) --top-8 boxes, 30% context crop--> frozen CLIP ViT-B/32
      -> 512-d per entity  -> stage 1: attention over entities within a frame  -> frame state
                            -> stage 2: attention over 16 frame states         -> clip state -> binary head
```

`EntitySpatioTemporal` (ESTA) is factorised ViViT-style space-time attention with the spatial axis
redefined as *detected entities* rather than image patches. Entity tokens carry CLIP appearance +
COCO-group embedding + box geometry (xc, yc, w, h, conf).

## The models, and why each one is there

| model | params | what it is | which critique it closes |
|---|---:|---|---|
| **ESTA** | 352,258 | the architecture under test | — |
| **B1** | 201,090 | ESTA with the entity axis deleted; whole-frame CLIP straight into the same temporal block | "you changed the backbone, not the architecture" |
| **B1w** | 333,570 | B1 + a second temporal layer, capacity-matched to ESTA within 5.3% | "you simply gave ESTA more parameters" |
| **B2** | 233,858 | same head, frozen **Kinetics-400 fine-tuned** VideoMAE-base (768-d) | "you beat a weak opponent" |
| **E0@64d** | 298,118 | the existing 64-d relation encoder | isolates the representation change (G15) |

B1 is not a re-implementation of a baseline: it is ESTA with one axis removed, sharing the same
`TemporalTransformer` object, the same width, depth, dropout, optimiser and schedule. Every model
in the table is trained by the same `esta_training.train_one`, deliberately mirroring
`run_relation_transformer.train_one`, so optimisation is not a free variable.

**The models are also guaranteed to have seen the same frames**: both feature arms are extracted in
one pass in `build-isafetybench-clip-features.py`, and the VideoMAE builder shares the same
`frame_sampling.decode`. The entity and whole-frame arms come from the same decoded frames and the
same detector pass. Detection count (52,997) is byte-identical to the old token build, which is the
evidence that the only changed variable is the appearance representation.

## Protocol (pre-registered, `risks.md`, unchanged)

- **LSO-1100**: `StratifiedGroupKFold(10)`, seed 0, grouped on source YouTube video id. 151 sources,
  **0 carry both classes (R41)** — so no source spans a fold boundary.
- 3 seeds per fold, 60 epochs, best-val-macro checkpoint on a 15% stratified split inside each
  training pool (`random_state=0`), AdamW lr 1e-3 / wd 0.01, cosine schedule, inverse-frequency class
  weights, label smoothing 0.05.
- **No threshold tuning. No per-fold model selection on held-out data. No re-runs for a better
  number. 0 test touches** — LSO-1100 is cross-validation over the development corpus.
- Every own-vs-own comparison is **paired**: cluster (grouped) bootstrap over the 151 sources +
  **exact McNemar**. `two_proportion_z` on paired predictions is banned.
- Feature alignment is *asserted, not assumed*: `run-esta-lso.load_aligned` permutes the CLIP tensor
  onto `isafetybench_dataset.load`'s clip order and aborts on any id or label disagreement, so the
  OOF files join to the existing 64-d runs on `clip_id` with no re-derivation.

## Gates (verbatim thresholds, written before any of these models were trained)

- **G14 (the gate).** ESTA must beat `max(B1, B1w)` by **>= 3.0 pp** balanced accuracy, with a
  cluster-bootstrap 95% CI excluding 0 **and** McNemar p < 0.05. Best-of-two is pre-registered so
  the baseline cannot be chosen after the fact.
  *Pre-registered consequence of failure:* the entity decomposition adds nothing over a whole-frame
  representation at matched backbone, and that is final.
- **G15 (component ablation, not the contribution).** ESTA vs E0@64d, same margin and CI rule.
  Isolates the representation bottleneck. Reported as an ablation, never as the headline claim.
- **G16 (honest strong opponent).** ESTA vs B2. If **B2 beats ESTA** with a CI excluding 0, the
  entity pipeline is not justified on accuracy grounds *even if G14 passed*, and that must be
  reported.
- **G17 (detector generation).** YOLO26s (10,009,784 params, `end2end=True`, NMS-free) is run as an
  ablation only. **The headline detector stays YOLOv11s whatever YOLO26 shows** — the deployment
  artifact and every edge number are frozen on it, and selecting a detector on CV is forbidden.

## Integrity disclosures

- A **2-epoch / 1-seed plumbing check** printed 82.9% balanced accuracy before the headline run
  started. Nothing was changed in response. **That number is not evidence and is not quotable as a
  result**; it is recorded here and in `risks.md` only so it cannot later look concealed.
- The ESTA run is **sharded across 4 processes** (`--shard i --nshards 4`). Folds are independent
  and the slice is `folds[i::4]`, so this is a scheduling change: same folds, same clips, same
  seeds, same epochs, same arithmetic. `tools/merge-esta-shards.py` refuses to merge unless every
  clip is covered exactly once, and delegates scoring to the same `finalise` the unsharded path
  calls. Guards in `tests/test-merge-shards-coverage.py` (5/5 pass).
- Conceded **GAPs** (recorded, not run): COCO's 80 classes exclude helmets and harnesses, so the
  entity vocabulary cannot name PPE; and there is no end-to-end fine-tuning of the backbone, which
  is a resource bound, not a design claim.

## Results

All four models, LSO-1100, 10 folds, 3 seeds, 60 epochs, identical recipe, identical frames.

| model | backbone | entity axis | params | balanced acc | macro F1 | unsafe recall | PR-AUC | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **B1w** | CLIP ViT-B/32 whole-frame | no | 333,570 | **84.4%** | 83.6% | 84.8% | 0.801 | 0.884 |
| **B1** | CLIP ViT-B/32 whole-frame | no | 201,090 | **84.2%** | 83.5% | 84.5% | 0.792 | 0.897 |
| **ESTA** | CLIP ViT-B/32 per entity | yes | 352,258 | 83.8% | 83.1% | 84.3% | 0.760 | 0.895 |
| **B2** | VideoMAE-base Kinetics-400 | no | 233,858 | 81.5% | 80.7% | 82.1% | 0.763 | 0.875 |
| E0@64d | MobileNetV3-s -> random 64-d | yes | 298,118 | 80.3% | — | — | — | — |

### Gate verdicts (`tools/adjudicate-r48-gates.py`, log: `logs/r48-adjudication.log`)

**G14 — FAIL.** ESTA 83.8% vs B1w 84.4% = **-0.5 pp**. Cluster bootstrap over 151 sources
95% CI [-5.5, +5.0] includes 0; McNemar exact b=72 c=78 **p=0.683**.
ESTA is not merely short of the 3.0 pp margin — it is *behind*, and the discordance is symmetric.
The pre-registered consequence applies and is not renegotiated: **the entity decomposition adds
nothing over a whole-frame representation at matched backbone.**

**G15 — FAIL (on the CI rule).** ESTA 83.8% vs E0@64d 80.3% = +3.5 pp. McNemar b=129 c=82
**p=0.00148**, but cluster CI [-2.3, +10.0] **includes 0**. The margin clears 3.0 pp and the paired
test is strongly significant, yet the pre-registered rule requires *both*, so this is a FAIL. The
disagreement between McNemar and the source-clustered CI is the finding, not an inconvenience: at
151 sources the corpus cannot certify a 3-4 pp effect at cluster resolution.

**G16 — no separation.** ESTA 83.8% vs B2 81.5% = +2.3 pp, CI [-2.3, +7.5] includes 0, p=0.083.
B2 does not beat ESTA, so the G16 failure clause does not fire; neither does ESTA separate from it.

### Descriptive paired contrasts (NOT gates — read for direction, not certification)

| contrast | diff | cluster 95% CI | McNemar p |
|---|---:|---|---:|
| B1w - E0@64d | +4.0 pp | [-0.9, +8.8] includes 0 | 0.000301 |
| B1 - E0@64d | +3.8 pp | [-1.0, +8.6] includes 0 | 0.000451 |
| B1w - B2 | +2.9 pp | [-1.8, +7.0] includes 0 | 0.0178 |
| B2 - E0@64d | +1.2 pp | [-2.9, +5.3] includes 0 | 0.184 |
| ESTA@yolo11s - ESTA@yolo26s | +2.2 pp | [-1.7, +6.0] includes 0 | 0.00412 |

### G17 — detector generation (ablation only; the headline detector was fixed in advance)

ESTA re-run end-to-end on YOLO26s entity features: **81.6%**, i.e. **-2.2 pp** against YOLOv11s
(McNemar b=75 c=43 p=0.00412; cluster CI [-1.7, +6.0] includes 0). The newer generation makes this
task worse. At conf 0.25 YOLO26s emits 45,512 detections over 87.9% of frames against YOLOv11s's
52,997 over 90.2% — a sparser scene for the entity arm to attend over.

This is the answer to **"was your detector the problem?"** It was not. A newer, larger, NMS-free
detector moves the entity arm *further* from the whole-frame baseline (B1w - ESTA@yolo26s = +2.7 pp,
p=0.0044), not closer. G14's failure is a property of the entity decomposition on this corpus, not
of YOLOv11s.

The whole-frame arm of the YOLO26 feature file is byte-identical to the YOLOv11 one (max |diff| =
0.0) because no detector touches it, so B1/B1w need no re-run and the detector is the only changed
variable between the two ESTA runs. YOLO26s remains in the thesis as the **edge** subject
(`end2end=True`, NMS-free), never as an accuracy claim.

## What this actually establishes

1. **R48 was pointing the right way.** Replacing the fixed random 576->64 projection with a frozen
   CLIP tower moves balanced accuracy 80.3% -> 84.4% with McNemar p=3.0e-04 on 209 discordant clips.
   The earlier `appearance_only` ablation (75.6% vs 75.2%, p=0.551) was not saying "nothing works" —
   it was saying the appearance channel was the binding constraint. It was.
2. **The entity decomposition is not the mechanism.** Deleting the entity axis entirely costs
   **-0.2 to +0.6 pp** and is statistically indistinguishable (p=0.683). Detections, box geometry
   and per-entity attention buy nothing for the clip-level binary decision on this corpus.
3. **A video-pretrained backbone is not the mechanism either.** Kinetics-400 VideoMAE (81.5%) sits
   *below* frozen image CLIP (84.4%), so the gain is not temporal modelling capacity.
4. **The corpus, not the model, is now the binding constraint on certification.** Every cluster CI
   in this report includes 0 while the clip-level CIs exclude it. With 151 sources and 0 sources
   carrying both classes (R41), between-source variance dominates. **No margin of this size can be
   certified on iSafetyBench at cluster resolution**, and reporting the clip-level CI as the headline
   would be the same protocol inflation this project already documented.

## Consequence for the thesis

The accuracy lane has a real result — 80.3% -> 84.4%, a +4.0 pp lift from one representation change
— but it does **not** support "entity-aware spatio-temporal attention". The defensible claim is
narrower and points the opposite way from the architecture that was built:

> On iSafetyBench, clip-level unsafe-behaviour recognition is carried by whole-frame appearance.
> A frozen image-language backbone plus a single temporal attention layer (201k trainable
> parameters) matches or beats entity-decomposed attention, a capacity-matched variant, and a
> Kinetics-pretrained video transformer.

**Open tension, for the author to decide (see Unresolved questions).** B1 does not use the detector
at all, so a thesis headlined on B1 has no YOLOv11/26 in its accuracy path. The detector's remaining
justification is **localisation and actionability** — the system must say *where* the hazard is, not
only that a clip contains one — plus the edge chapter, where YOLO26s's `end2end=True` NMS-free graph
is the measurement subject. That is an honest role, but it is a different role from the one G14 was
written to test, and it must be stated as such rather than quietly retrofitted.

## Unresolved questions

1. **Does the thesis headline on B1 (no detector in the accuracy path) or keep the entity pipeline
   for localisation?** The numbers force this choice; the framing decision is the author's.
2. ~~G17 pending.~~ **Answered: no.** YOLO26s costs the entity arm 2.2 pp (p=0.00412). The failure
   is not detector-limited.
3. **B1 vs B1w for deployment.** They are indistinguishable (84.2% vs 84.4%, 0.2 pp) and B1 is 40%
   smaller. Choosing B1 is an *edge* decision on the pre-registered smallest-model criterion, not an
   accuracy selection, and must be reported with B1's own OOF number, never with `max(B1, B1w)`.
4. Jetson measurement is unscheduled until the deployment artifact is named; the device is operated
   by the author, so each on-device step needs an explicit hand-off.
