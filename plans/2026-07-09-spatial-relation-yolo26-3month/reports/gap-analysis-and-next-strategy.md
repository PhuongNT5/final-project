# Gap Analysis vs Paper + Next Strategy

**Date:** 2026-08-10. Back-check of our pipeline against Önal & Dandıl (Unsafe-Net) to explain the
accuracy gap and set the next strategy. Paper lines cited from `paper/paper.txt`.

## 1. The paper's method (exact, verified)
- **YOLO v4 = per-frame BEHAVIOUR detector.** Trained on **2,262 frames labelled with the 4 behaviour
  classes** (behaviour-as-object), 95.66 % mAP, imgsz 416, "number of classes was 4" (L205, L323).
- **Frame-reduction + single-action separation:** YOLO "select[s] the flowing images and exclud[es] all
  other frames" and, critically, **"ensur[es] there is only one action in the same video"** — splitting
  each raw clip into clean single-behaviour segments handed to ConvLSTM (L48, L323–325).
- **ConvLSTM:** trains on **452 clips / 64 test** — *identical to our split* — but on the YOLO-cleaned
  single-action segments (L327).
- **Decisive numbers (Tables 5–7):**
  - ConvLSTM **without** YOLO reduction: **57/64 (89.06 %) top-1, 85.78 % macro** (per-class 31/32, 9/11, 10/13, 7/8).
  - Unsafe-Net **with** reduction: **62/64 (96.88 %), 95.81 % macro** (32/32, 10/11, 12/13, 8/8).
  - ⇒ the **entire +5 clips / +10 pp is the YOLO-v4 behaviour-frame-reduction**.

## 2. Our pipeline vs the paper
| Stage | Paper | Ours |
|---|---|---|
| 2,262-frame annotation | **behaviour** bbox (the answer) | **entity** bbox (intermediate) |
| Per-frame detector | YOLO v4 = behaviour detector (95.66 mAP) → frame-select + single-action split | YOLO11 = entity detector (0.969 mAP), **no behaviour selection** |
| Temporal classifier | ConvLSTM on 452 clips, **YOLO-cleaned single-action** input | Transformer on 452 clips, **unfiltered uniform T=16** input |
| Classification supervision | ~2,262 **frame-level** behaviour signals | 452 **clip-level** signals |

## 3. Root cause of the gap (3 causes)
Our best (v2b) = **52/64, 84.2 % macro**; v1 = 86.1 % macro. **Macro matches the paper's no-reduction
baseline (85.78 %)** → we reproduced that regime; the gap to 95.81 % is the reduction step we lack.
1. **Supervision placement (primary).** Same annotation budget; paper puts it on the *behaviour* at the
   frame level (direct, 2,262 signals), we put it on *entities* and defer discrimination to a 452-clip
   Transformer (indirect, ~5× less signal).
2. **No behaviour-frame-reduction / single-action separation** — the mechanism worth the +10 pp. Our
   entity detector cannot select behaviour-frames.
3. **Lossy entity tokenization** — boxes+geometry discard scene appearance raw-frame ConvLSTM keeps;
   explains top-1 52 < 57 even vs the no-reduction baseline. (v2b appearance crops recovered +1.)
4. *(secondary)* weak per-view zone geometry (foot_in_walkway ~0.12 for both confused classes).

**Scientific finding:** for small-N industrial behaviour recognition, **front-loading discriminative
supervision at the frame level is far more sample-efficient than deferring it to a relational temporal
model over intermediate entity labels** — same budget, +10 pp. This sharpens the thesis's "data-bound"
diagnosis into a **supervision-placement** diagnosis.

## 4. Next strategies (ranked)
We already have the 2,262 frames + clip labels — several strategies need **no new annotation**.

**S1 — Per-frame behaviour signal (paper-faithful, highest value/effort). [recommended first]**
Propagate each clip's behaviour label to its sampled frames → train a **per-frame behaviour recognizer**
(the paper's YOLO-v4 role). Two sub-steps, both reuse existing data:
- **S1a — auxiliary per-frame loss** on the existing relation-Transformer: each frame's tokens → behaviour,
  supervised by the propagated label. Upgrades supervision 452 → ~452×16. *Cheap (loss change), ~1 h.*
- **S1b — behaviour-frame-gate + single-action aggregation:** use the per-frame behaviour confidence to
  drop irrelevant frames and aggregate (mean/majority) — replicates the +10 pp mechanism.
Expected: toward 57–62/64.

**S2 — Whole-frame appearance stream.** Add a per-frame global scene embedding (recover the lossy
tokenization; this is what raw ConvLSTM / the thesis frozen-features used). Expected: lift top-1 toward 57.

**S3 — Literal replication: YOLO26 behaviour detector.** Train YOLO26 on behaviour bboxes (or clip-label
per frame) → aggregate. Strongest paper reproduction; sets an upper bound and a clean ablation
(entity-ontology vs behaviour-detector). Abandons the ontology elegance, so run as a *comparison*.

**S4 — Data lever.** More/better zone or behaviour annotation (paper-scale). Ceiling-breaker, big effort.

**S5 — YOLO26 + edge deployment (plan phases 02/05).** Orthogonal; strengthens the thesis's *primary*
contribution (measured edge system). Do regardless of accuracy.

## 5. Recommendation
1. **S1a** now (cheap, directly attacks the supervision-placement cause; ~1 h). If it lifts top-1, add **S1b** (behaviour-gate).
2. **S3** as a controlled comparison (paper-faithful upper bound) — likely the strongest accuracy and a great thesis ablation.
3. **S5** in parallel for the deployment story (the thesis's headline contribution).
4. Frame the whole result honestly: we reproduce the no-reduction regime; the +10 pp is a
   supervision-placement effect, now diagnosed and testable via S1/S3.
