# Project Overview & PDR

## What this is

Master's thesis capstone: real-time unsafe-behaviour detection in industrial video surveillance
using a YOLO + Transformer architecture, evaluated on the **Unsafe-Net** dataset (Önal & Dandıl,
paper year cited inconsistently across repo docs as 2024 vs 2025 — see Unresolved). Thesis title
(VN): *Phát hiện hành vi không an toàn thời gian thực trong giám sát công nghiệp dựa trên YOLO và
Transformer*.

Deployment-led framing (see `README.md`): the delivered contribution is a measured, working
edge-deployment system plus a statistically rigorous re-evaluation of the paper's claim — not a
claim of beating the paper's headline number. See `system-architecture.md` for why three pipeline
variants (P1/P2/P3) exist and which one backs which claim.

## Research objectives (PROPOSAL.txt, MT1-MT3)

| ID | Vietnamese objective (PROPOSAL.txt) | Plain-English | Status |
|---|---|---|---|
| **MT1** | Kết hợp CNN (YOLO) tốc độ/thông lượng với Transformer cho temporal reasoning, thành kiến trúc YOLO-Transformer | Combine YOLO's speed with a Transformer's temporal/sequence reasoning for industrial safety monitoring | Built — P3 (YOLO entity detector -> spatial-relation Transformer) is the current architecture instantiating this objective |
| **MT2** | Huấn luyện, tinh chỉnh, đánh giá YOLO-Transformer trên Unsafe-Net (4 an toàn / 4 nguy hiểm) | Train/fine-tune/evaluate on Unsafe-Net's 8-class (4 safe + 4 unsafe) video clips | In progress — evaluation contract defined in `plans/2026-09-06-thesis-2week-final/brief.md` §4; canonical numbers in `codebase-summary.md` |
| **MT3** | TensorRT/INT8, đa luồng thời gian thực trên biên, độ trễ thấp, độ chính xác cao | Deploy to Jetson edge hardware with TensorRT + INT8, multi-stream, low latency, measured accuracy | Partially done — see `deployment-guide.md`; **all measured Jetson numbers today are P1, not P3** (re-benchmark of P3 on Jetson is scheduled, not yet complete as of this writing) |

## Relationship to the origin paper

Baseline: Önal & Dandıl, *Unsafe-Net: A real-time safety analysis system for unsafe behaviors in
industrial facilities using deep learning*, verified against `paper/paper.txt` in this repo.

Paper's method: YOLOv4 per-frame behaviour detector (nc=4, frame-reduction gate) + ConvLSTM
temporal classifier, trained on 2,262 labelled train images + 329 test images (paper Table 3).

Paper's headline numbers (verified by grep against `paper/paper.txt`, Tables 5-7):

| Config | Top-1 (64-clip 4-class subset) | Macro (mean per-class recall) |
|---|---|---|
| Full pipeline (YOLOv4 + ConvLSTM + frame reduction) | 62/64 = 96.88% | 95.81% |
| No-reduction baseline (ConvLSTM only, no bbox) | 57/64 = 89.06% | 85.78% |

Paper also reports action-recognition time **0.14 s/clip** and average clip duration reduced to
**1.87 s** by frame reduction (both from paper text, not yet independently reproduced end-to-end
in this repo — see `phase-01` in the active plan).

**Retracted claim — do not resurrect:** an earlier draft of this project's docs cited a "95.7%
mAP" detector figure attributed to the paper. Exhaustive grep of `paper/paper.txt` returns zero
hits for that figure. The paper's only quantified detector metric is a training-curve read
("average mAP ~97% during the iteration period," a read off Fig. 8, not a test-set number, for a
nc=4 frame-gate detector at 416x416). No detector-mAP comparison against the paper is defensible
and none should be written anywhere in this repo's docs or thesis.

## Statistical ceiling on the 64-clip benchmark

The paper's own test set is 64 clips, and the paper itself is wrong on only 2 of them. This bounds
what any comparison can show: even a hypothetical flawless 64/64 result gives McNemar best-case
**p = 0.50** (Fisher exact p = 0.496) against the paper's 62/64 — independently re-derived and
verified in `plans/2026-09-06-thesis-2week-final/reports/qa1-adversarial-review-and-defence.md`
item 14. No system can be shown to significantly beat the paper's number at N = 64; this is a
property of the benchmark's size, not a shortfall of any particular method.

## Claim structure (what this thesis argues)

Per the binding brief (`plans/2026-09-06-thesis-2week-final/brief.md` §2): on the paper's own
64-clip metric, this work is statistically indistinguishable from the published result — and by
the ceiling argument above, no system can be distinguished there. The thesis instead argues
effectiveness on axes the paper never measured: statistical reliability at larger N (k-fold over
516 pooled clips), cross-camera generalisation (leave-camera-out), safety-oriented error structure
(per-class miss rate, cost-sensitive scoring), calibration (ECE), and measured real-time edge
performance. See `system-architecture.md` for which pipeline (P1/P2/P3) backs each of these axes.

## Unresolved questions

1. Paper publication year is cited inconsistently across this repo: `brief.md` and `PROPOSAL.txt`
   say 2024; `README.md`'s citation block says 2025 (*Multimedia Tools and Applications*, 84,
   34967-34993). Not resolved here — whoever finalises the thesis references section should pick
   one and verify against the actual publication record.
2. The LCO (leave-camera-out) cross-camera generalisation number is explicitly in flight as of
   this writing: the model reaches a 78-82% macro range across grouping/token variants, but no
   single canonical value is locked yet, and the baseline it should be compared against is being
   rebuilt (task A3b in `brief.md`). Do not treat any single point value as final.
