# Gate 1 — behaviour-detector GPU retrain: PRE-REGISTERED PREDICTION

**Written 2026-09-06, BEFORE the run. Nothing below the "Outcome" line existed when the job was launched.**

Purpose: the thesis claims pre-registration discipline from 2026-09-06 onward
(`reports/test-touch-log.md`). A prediction written after seeing results is worthless; this file exists so
the retrain can *falsify* something. It is not a test-set touch — the behaviour detector is scored on its
own 343-frame **val** split, never on the 64-clip benchmark.

## The run being predicted

```
python build-behaviour-labels.py --variant tight     # and again with --variant ctx
python train-behaviour-detector-gpu.py --data .../dataset_behaviour_{tight,ctx}/data.yaml
```
`yolo11s.pt`, imgsz 800, epochs 200, batch 16, SGD lr0 0.01, cos_lr, patience 30, close_mosaic 15,
`hsv_h=0.0`, `fliplr=0.5`, `mixup=0.1`, seed 0. Dataset 1901 train / 343 val frames, 6122 boxes, 8 classes.

Baseline to beat: CPU run at imgsz 416, 25 epochs, patience 10 (early-stopped at epoch 15 while mAP50 was
still rising) — **mAP50 0.791 tight / 0.806 ctx**.

## Headline predictions

| quantity | point | 80% interval | gate |
|---|---|---|---|
| mAP50, `tight` | **0.87** | 0.84 – 0.90 | ≥0.90 **predicted FAIL** (~30% pass) |
| mAP50, `ctx` | **0.89** | 0.86 – 0.92 | ≥0.90 **predicted borderline** (~45% pass) |
| mAP50-95, `ctx` | 0.60 | 0.54 – 0.66 | — |
| all 8 classes recall > 0 | **PASS** | ~85% confidence | — |
| `ctx` − `tight` | **+2.0 pp** | +0.5 – +4.0 | — |
| wall-clock per arm (T4) | 1.5 h | 1.0 – 2.5 h | fits the 9 h session |
| epoch of early stop | ~150 | 110 – 200 | — |

**Reasoning.** Three levers, in the order R1 ranks them. (1) imgsz 416→800: the median WalkwayViol box is
0.003 of frame area ≈ 23 px linear at 416, ≈ 44 px at 800 — this is the dominant lever and is worth most of
the gain. (2) epochs 15→200 with cos_lr: the CPU run never converged, so this is recovering training that
never happened, not squeezing a converged model. (3) fliplr 0.0→0.5 on a 1901-image set: small but free.
Together I expect +6 to +9 points over the CPU baseline, which lands just *under* the 0.90 gate rather than
over it — hence the FAIL prediction for `tight`.

## Per-class AP50 predictions (`ctx` arm)

| class | train boxes | predicted AP50 | note |
|---|---|---|---|
| OpenedPanel | 525 | **0.98 – 1.00** | *degenerate — see below.* Free mass. |
| ClosedPanel | 1181 | 0.92 – 0.98 | many labels, near-constant box size |
| CarryOverload | 195 | 0.85 – 0.95 | forklifts are large and distinctive |
| AuthorInt | 390 | 0.80 – 0.92 | vest-hue class; `hsv_h=0` protects it |
| SafeWalkway | 397 | 0.75 – 0.90 | |
| WalkwayViol | 1634 | 0.75 – 0.88 | context-dependent, see falsifier F2 |
| UnauthorInt | 1630 | 0.75 – 0.88 | context-dependent, see falsifier F2 |
| SafeCarry | 170 | **0.70 – 0.90** | widest interval — smallest class, 170 labels |

## Ceilings and free mass — why the mAP will be partly fake

1. **OpenedPanel is degenerate.** All 525 boxes sit at essentially one position and size
   (sd(x)=0.004, sd(y)=0.003, sd(w)=0.008, sd(h)=0.007). A *single constant box*, the mean of the train
   boxes, localises **100% of val OpenedPanel instances at IoU 0.5** (96.2% at IoU 0.75). Since mAP50 is a
   mean over 8 classes, ~**0.125 of the headline is free** and measures nothing. A reported 0.90 means the
   other seven classes average 0.886. This must be disclosed wherever the number appears.
   *(Checked and NOT confirmed for ClosedPanel: near-constant size but sd(y)=0.13, constant-box recall 0.0%.)*
2. **The val split is scene-confounded.** All 8 camera views appear on both sides. The mAP measures
   in-domain detection, not cross-camera generalisation, and must never be cited as the latter.
3. **The val split is contaminated.** 12 of 343 val frames (3.5%) are byte-identical to a train frame, and
   60 of the 62 duplicate frame groups carry *contradictory* behaviour labels. This caps achievable AP50 on
   classes 0/1 and makes the val number both slightly leaked and slightly pessimistic at the same time.

## Falsifiers — what would prove this prediction wrong, and what it would mean

- **F1 — mAP50 > 0.95.** Do not celebrate. Check per-class first: at 6122 boxes over 8 classes with two
  semantically ambiguous majority classes, >0.95 is more likely contamination or degenerate classes carrying
  the mean than genuine detection quality.
- **F2 — WalkwayViol or UnauthorInt AP50 > 0.90 on the `tight` arm.** This is the sharpest falsifier in this
  file. At `tight` those two classes are *the same object* — a person box — labelled by which clip it came
  from. Nothing inside the box distinguishes them. High AP50 there is therefore positive evidence that the
  detector is reading **camera/scene identity** through its receptive field, not behaviour. If this happens,
  it is a finding about the confound, not a detector success, and belongs in the scene-confound chapter.
- **F3 — mAP50 < 0.82 on either arm.** The recipe is wrong or the data.yaml is pointing at the wrong dataset
  (the classic failure: training the *entity* detector by mistake). Check `names:` in the yaml first.
- **F4 — any class with recall exactly 0.** Would contradict the verified label counts (no class has zero
  labels). Would indicate a threshold or stride-assignment problem, not a data problem.

## What Gate 1 does NOT buy, even if it passes

The behaviour detector is **not in the P3 pipeline**. P3 (the 54/64 headline) uses the **entity** detector
plus the relation-Transformer. A successful behaviour-detector retrain strengthens the *paper-faithful
reproduction* axis, and nothing else — it will not move 54/64, the k-fold numbers, or the LCO numbers.
Expect no headline change from this run. If the thesis later implies otherwise, that is an error.

## Outcome

*(to be filled in AFTER the run — never edit anything above this line after seeing results)*

| arm | mAP50 | mAP50-95 | dead classes | gate ≥0.90 | prediction correct? |
|---|---|---|---|---|---|
| tight | | | | | |
| ctx | | | | | |
