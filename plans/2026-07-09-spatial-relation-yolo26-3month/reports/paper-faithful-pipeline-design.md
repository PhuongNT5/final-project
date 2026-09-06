# Paper-Faithful Re-Implementation — Modern Pipeline Design

**Decision (2026-07-11):** Re-implement the Unsafe-Net paper pipeline **from scratch**, keeping its
three-stage structure, but swapping the components:
- **YOLOv4 → YOLOv11s / YOLO26** — frame-selection gate **+** multi-entity detector
- **ConvLSTM → Transformer encoder** — temporal **+** spatial-relation classifier
- **Monitoring & Warning System → unchanged** (already built/deployed, Ch5)

## Paper pipeline (from Fig 1 + Fig 6)
- **Stage A — Selection of frames:** YOLOv4 flags "unsafe behaviour in object?" per frame; Yes→keep,
  No→exclude → *reduced video stream* (paper: 105 frames/3.5 s → 40 frames/1.33 s).
- **Stage B — ConvLSTM:** 4×[ConvLSTM(filters 4,8,14,16)+MaxPool+Dropout] → Flatten → Dense → unsafe class.
- **Stage C — Monitoring & Warning:** Control / Warning / Reporting / GUI.

## Modern re-implementation (component map)
| Paper stage | Paper component | Our replacement | Role |
|---|---|---|---|
| A Frame selection | YOLOv4 (behaviour-as-object bbox, used as frame filter) | **YOLO26** entity detector | (1) detect multi-entity boxes; (2) frame-gate: keep behaviour-relevant frames, drop rest → reduced stream |
| B Temporal classifier | ConvLSTM (CNN + LSTM) | **Transformer encoder** over a box-token sequence | classify unsafe behaviour from entity **geometry** across frames |
| C Warning | Control/Warning/Reporting/GUI | same (deployed, Ch5) | audible/visual alert + logging + GUI |

## Data flow — the key change
Per selected frame, YOLO26 emits entity detections. Each detection → one token (this is the user's
"Coordinate Embedding" spec, extended with the research-backed relative-geometry dims):
```
token_i = [ ROI appearance feature  (YOLO box feature vector) ]   # per-entity appearance
        ⊕ [ normalized box coords   x_c, y_c, w, h ]             # the coordinate embedding
        ⊕ [ entity class embedding  (8 ontology classes) ]        # Person_Auth/Unauth, Zone, Panel, Forklift...
        ⊕ [ relative geometry       IoU(person,Safe_Walkway), IoU(person,Restricted_Zone),
                                     foot_in_walkway(bin+cont), norm_dist_to_zone ]   # helps at small N
        ⊕ [ track-id embedding ]                                  # temporal consistency
```
Sequence `[CLS] + {token_i}` over T frames → **Transformer encoder**. Self-attention learns:
- `Person_Unauthorized` × near `Restricted_Zone` → **Unauthorized Intervention**
- person foot-point **outside** `Safe_Walkway` → **Walkway Violation**

→ CLS → MLP → 4-class unsafe behaviour. This replaces ConvLSTM's implicit spatiotemporal recurrence
with **explicit entity-relation attention** — the mechanism that breaks the WalkwayViol↔UnauthorInt
confusion (cosine 0.834 in pure appearance).

## Zone handling (resolves the open question)
Zones (`Safe_Walkway`, `Restricted_Zone`) are **annotated as training labels** (polygon per view) and
**detected as entities**; the Transformer **learns** the person×zone relation via attention. We do
**not** hardcode a runtime foot-in-polygon rule engine — consistent with "let the Transformer reason."
The geometric scalars are *derived token features*, not a decision rule. (A rule-engine fallback may be
added later for interpretability, but the learned path is primary.)

## Faithful to the paper (keep)
- 3-stage architecture + the frame-reduction idea → measure the same **"reduced avg duration 1.87 s"**
  and **"0.14 s/clip"** metrics (§Phase 04).
- 4-class unsafe evaluation on the 64-clip test set; macro = mean per-class recall.
- Targets: beat no-bbox **85.78 % / 89.06 %**; approach full-bbox **95.81 %**.
- Monitoring & Warning System (Stage C).

## Differs (modernization)
- Detector: anchor-free **YOLO26** (NMS-free/DFL-free, INT8-stable, native ByteTrack) vs YOLOv4.
- Classifier: **Transformer self-attention** vs ConvLSTM recurrence.
- Supervision: **multi-entity spatial ontology** vs paper's behaviour-as-object bbox.
- Rigor: blind held-out + k-fold + BCa CIs (`eval_utils/`, Phase 00) — beyond the paper.

## Where each stage lives in the plan
- **Stage A** (YOLO26 detector + frame-gate) → [phase-02](../phase-02-yolo26-and-detector.md)
- **Stage B** (box-token Transformer, token spec above) → [phase-03](../phase-03-spatial-relation-transformer.md)
- **Stage C** (warning + paper timing metrics) → [phase-05](../phase-05-jetson-and-thesis-integration.md)
- **Evaluation** (paper metrics + multi-lens) → [phase-04](../phase-04-evaluation-rigor.md)

## Two detector options for Stage A (decide in Phase 02)
1. **YOLOv11s** — matches the current thesis backbone; lowest migration risk; the frozen-feature path already works.
2. **YOLO26** — newer, faster INT8 on Jetson (3.8 ms), native tracking; backbone identical to v11 so migration is a hook-remap + re-extract.
Recommendation: build/validate on **YOLOv11s first** (known-good), then port to **YOLO26** for the edge/deployment benchmark — report both (a clean ablation the thesis can use).
