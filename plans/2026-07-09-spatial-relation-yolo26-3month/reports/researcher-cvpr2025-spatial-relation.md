# Spatial-Relation Transformer for Box-Token Safety Classifier — Research Report
**Date:** 2026-07-09  
**Context:** Unsafe-Net video dataset, 8-class industrial safety, fix for Safe-Walkway-Violation vs Unauthorized-Intervention confusion (cosine 0.834 with frozen COCO features).  
**Goal:** Ground a "box-token + spatial-relation transformer" design in 2024–2025 literature.

---

## Theme 1 — Encoding Box Coordinates as Tokens (Canonical + 2024–2025)

### Canonical References

**P1. DETR** (Carion et al., ECCV 2020) — `arXiv:2005.12872`  
Introduced learnable object queries + sinusoidal 2D positional encoding added to encoder keys/queries. Boxes predicted from CLS-token output; coordinates never directly tokenized, but the spatial prior is injected via PE. Foundation for everything below.

**P2. DAB-DETR** (Liu et al., ICLR 2022) — `arXiv:2201.12329`  
**Direct design template.** Each query IS a 4D anchor box `(x, y, w, h)`. Coordinates are passed through sinusoidal embedding and used as positional query. WH-modulated attention scales the contour by box size. Decoder updates anchor coordinates layer-by-layer. Key insight: treating coordinates as queries accelerates convergence and makes spatial priors explicit — exactly the "box as token" idea needed here.

**P3. DN-DETR** (Li et al., CVPR 2022) — `arXiv:2203.01305`  
Adds denoising training: noisy box coordinates are injected as queries to stabilize anchor learning. Shows box-coordinate queries benefit from curriculum noise. Relevant: even with noisy/imprecise zones (painted walkway polygon), denoising training helps.

### 2024–2025 Improvements

**P4. Relation DETR** (Xiuqhou et al., ECCV 2024 Oral) — `arXiv:2407.11699` | [GitHub](https://github.com/xiuqhou/Relation-DETR)  
Adds a **position-relation encoder**: pairwise box IoU, distance, and overlap ratios are computed between all query boxes and injected as **attention bias** (additive to attention logits). Achieves +2.0 AP over DINO on COCO; faster convergence from scratch. **Most directly applicable** — this is the "IoU person∩zone → attention bias" mechanism requested.

**P5. LP-DETR** (Kang et al., Feb 2025) — `arXiv:2502.05147`  
Layer-wise progressive spatial relations in self-attention: local/medium/global relation scales across decoder layers. Achieves 52.3 AP with R50. Extends Relation DETR idea to multi-scale. Confirms pairwise spatial relation bias in attention is now standard practice in 2025 DETR variants.

**P6. GTA: Geometry-Aware Attention** (Miyato et al., ICLR 2024) — `arXiv:2310.10375` | [Project](https://takerum.github.io/gta/)  
Transforms key/value vectors by a **relative geometric transformation** (e.g., relative camera pose, relative bounding box transform) so that attention operates in a unified coordinate frame. Zero extra learned parameters, minor compute overhead. Relevant for normalizing person/zone spatial relation into a canonical frame.

**P7. BROS** (Hong et al., AAAI 2022, widely used 2024) — [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/bros)  
BERT-style encoder that takes `(token, bounding_box)` pairs; encodes **relative spatial position** instead of absolute. Designed for documents but demonstrates that relative box-coordinate encoding generalizes better on small datasets than absolute encoding. Cheap win: encode person-to-zone distance/IoU as relative, not raw pixel coordinates.

---

## Theme 2 — Human-Object / Human-Scene Interaction (HOI) for Behavior Recognition

**P8. Action Genome** (Ji et al., CVPR 2020) — `arXiv:1912.06992`  
Decomposes actions into spatio-temporal scene graphs: object bounding boxes + pairwise relations (attention/spatial/contact) per frame. Shows that "who is where relative to what" is necessary and sufficient for action recognition. Direct conceptual model for the Safe-Walkway-Violation task — person's spatial relation to walkway zone IS the action.

**P9. Disentangled Pre-training for HOI** (Li et al., CVPR 2024) — `arXiv:2404.01725` | [GitHub](https://github.com/xingaoli/DP-HOI)  
Separates detection decoder (person/object localization) from interaction decoder (verb/relation classification). Pre-trains each on domain-specific data. Key result: disentangling spatial localization from relation reasoning significantly boosts rare/long-tail categories — highly relevant since Safe-Walkway-Violation is a minority class.

**P10. VOST-SGG** (Dec 2024) — `arXiv:2512.05524`  
One-stage spatio-temporal scene graph generation; disentangles "what" vs "where" queries; integrates visual + textual + spatial cues via VLM. Shows spatial cues alone (box positions) are insufficient — semantic class labels of the boxes must be included. Directly validates "embed each box's class + coords as token."

**P11. RegFormer** (CVPR 2026 accepted, arXiv Apr 2025) — `arXiv:2604.00507`  
Weakly-supervised HOI detection; uses spatially grounded signals from off-the-shelf detector as query constraints. Locality-aware: human-object bounding box pairs constrain attention radius. Competitive with fully supervised methods. Confirms that for small/weak supervision, explicitly injecting spatial priors via bbox constraints is essential.

---

## Theme 3 — Video Anomaly / Safety / Zone-Intrusion Detection (2024–2025)

**P12. Safe-Construct** (Chharia et al., CVPR 2025 / IEEE 2025) — `arXiv:2504.10880` | [Project](https://safe-construct.github.io/Safe-Construct/)  
**Closest domain match.** Reframes construction safety violation as a **3D multi-view engagement task**: encodes scene-level worker-object context, spatial relationships between workers and objects/zones. Uses synthetic data (SICSG) to overcome small-dataset problem. Key lesson: 2D appearance features fail; explicit spatial worker-object encoding is the fix. Their 3D approach is overkill for this thesis (no stereo), but the "encode worker-zone spatial relation explicitly" principle directly validates the proposed design.

**P13. Action Recognition-based Industrial Safety Violation Detection** (Dec 2024) — `arXiv:2412.05531`  
Two-stage: activity recognition first, then PPE/rule check. Achieves +23% F1 over PPE-only approach. Shows action context is necessary for correct rule application — pure appearance (PPE class) ≠ violation. Validates the claim that frozen COCO appearance features are insufficient.

**P14. Automated non-PPE detection with YOLOv10 + Transformer** (Nature Sci Reports, 2025) — [Link](https://www.nature.com/articles/s41598-025-12468-8)  
Construction site; detects helmet/vest non-compliance with YOLO detections fed to transformer. No explicit zone geometry — but confirms YOLO → transformer pipeline is the de-facto industrial safety stack in 2025.

---

## Theme 4 — Absolute vs Relative Geometry for Small Datasets

**Evidence synthesis from P4, P7, P8, P9, RegFormer:**

| Encoding | Pros | Cons | Recommendation |
|---|---|---|---|
| **Absolute [x,y,w,h] / img_size** | Simple, direct | Brittle to camera change; wastes capacity on position memorization | Use only as secondary feature |
| **Relative (IoU, containment ratio, centroid distance / zone_size)** | Invariant to camera zoom/crop; generalises across scenes; small dataset friendly | Requires zone polygon at inference | **Primary encoding** |
| **Sinusoidal embedding of absolute coords** (DAB-DETR style) | Richer representation; enables anchor modulation | Needs many samples to learn well | Use for within-box queries if data >5k frames |

**Literature consensus:** For small N (<5,000 frames), **relative geometric features** (IoU person∩zone, containment ratio, centroid offset normalised by zone width) outperform absolute coordinates because they encode the task-relevant invariant directly and leave less for the transformer to learn. This is supported by BROS (relative spatial encoding generalises better), Action Genome (pairwise relations as features), and Relation DETR (pairwise IoU/distance as attention bias).

**Recommended input token for this thesis:**  
```
token_i = [sin_embed(x_c, y_c, w, h) ||  # 4×64-d sinusoidal
            class_embed(label_i)          # learned, 64-d
            IoU(person_i, zone_j)         # scalar, for each zone type
            containment(foot_y, zone_y_range)  # scalar: foot in zone?
           ]   → MLP project → 128-d token
```
Feed N such tokens (one per detected box) as sequence to a small Transformer encoder (2–4 layers, 4 heads, d=128).

---

## Theme 5 — Track-Based Temporal Consistency

**P15. ODTrack** (Jan 2024) — `arXiv:2401.01686`  
Online dense temporal token learning: propagates temporal token auto-regressively across frames. Each frame produces a temporal token encoding target trajectory. Most relevant mechanism for feeding stable box sequences to transformer.

**ByteTrack** (Zhang et al., ECCV 2022) — widely used in 2024–2025 production  
Associates both high- and low-confidence detections; strongest production MOT baseline. Use for generating track-IDs before temporal transformer.

**Practical approach for this thesis (KISS):**  
ByteTrack → assign stable track_id per person per clip → use track_id as a learnable embedding token concatenated to box token → transformer attends across time steps T for the same track. Occlusion interpolation: linearly interpolate box coords for frames where track is lost (standard in ByteTrack). Do NOT use transformer-based trackers (MOTR, TrackFormer) — research-grade, unnecessary complexity.

---

## Recommended Concrete Design

### Architecture (Box-Token Spatial Relation Transformer)

```
Input: YOLO detections per frame (T frames, up to N_max=8 boxes/frame)
       → Filter: keep Person_Auth, Person_Unauth, Zone_Walkway, Zone_Restricted, Forklift

Per-box token:
  t_i = LayerNorm(
    Linear(
      [sin_embed(x,y,w,h, d=64)         # spatial PE
       || class_onehot_embed(d=32)       # entity class
       || [IoU_walkway, IoU_restricted,  # geometric relations (scalars)
           foot_in_walkway, foot_in_restricted,
           norm_dist_to_walkway_centroid]
      ], d_model=128)
  )

Sequence: [CLS] + {t_i for all boxes, all T frames}  →  (T×N_max + 1) × 128
  → Transformer Encoder (2 layers, 4 heads, d=128, FFN=256)
  → CLS token → Linear(8) → softmax
```

**Key design choices:**
- `d_model=128`, 2 layers, 4 heads: minimal capacity to avoid overfitting on ~3,000 frames
- Foot-point containment as explicit scalar: `foot_y_min` relative to walkway polygon Y-range is the primary discriminator for Safe-Walkway-Violation
- Class token encodes entity TYPE (Person_Auth vs Person_Unauth is the primary signal for Unauthorized-Intervention)
- Track-ID token (ByteTrack): add learnable `track_embed(track_id % 16)` to handle temporal consistency without complexity

### Overfitting Mitigations (Literature-backed)

| Technique | Source | Cost |
|---|---|---|
| Drop path rate 0.1–0.2 | Stochastic Depth | Free |
| Dropout 0.3 on FFN | Standard | Free |
| CutMix/MixUp on frame-level inputs | Selective Volume Mixup (arXiv:2309.09534) | Low |
| Label smoothing 0.1 | Standard | Free |
| Weight decay 0.01 | Standard | Free |
| Pre-compute relative features (IoU, containment) — reduces learned parameters | BROS, Relation DETR | Free |
| Freeze YOLO backbone, train only transformer head | This thesis baseline | Free |

**Do NOT use:** full scene graph generation (VOST-SGG), 3D multi-view (Safe-Construct), VLM integration (GridVAD), MOTR tracking — all overkill for N=3,000.

---

## Paper Inventory (8–15 Most Relevant)

| # | Title | Venue | Takeaway |
|---|---|---|---|
| P2 | DAB-DETR: Dynamic Anchor Boxes are Better Queries | ICLR 2022 | Box (x,y,w,h) directly as query token; WH-modulated attention = canonical box-token design |
| P4 | Relation DETR: Explicit Position Relation Prior | ECCV 2024 Oral | Pairwise IoU/distance between boxes → additive attention bias; fastest convergence, +2 AP |
| P5 | LP-DETR: Layer-wise Progressive Relations | arXiv Feb 2025 | Multi-scale spatial relation bias across decoder layers; validates Relation DETR in 2025 |
| P6 | GTA: Geometry-Aware Attention (Multi-View) | ICLR 2024 | Relative geometric transform as key/value transformation; zero extra params |
| P7 | BROS: BERT Relying On Spatiality | AAAI 2022 | Relative box encoding >> absolute for small datasets; directly validates thesis approach |
| P8 | Action Genome: Actions as ST Scene Graphs | CVPR 2020 | "Who-is-where-relative-to-what" as primary action descriptor; spatial graph IS the label |
| P9 | Disentangled Pre-training for HOI | CVPR 2024 | Separate spatial localization from relation reasoning; boosts rare classes (= minority violation classes) |
| P10 | VOST-SGG: VLM-Aided ST Scene Graph Gen | arXiv Dec 2024 | Class + spatial cues must both be in token; spatial alone insufficient |
| P11 | RegFormer: Relational Grounding Weakly-Supervised HOI | CVPR 2026 | Spatial grounding from detector boxes as query constraints; competitive even weakly supervised |
| P12 | Safe-Construct: Construction Safety as 3D Engagement | CVPR 2025 / IEEE 2025 | 2D appearance fails; worker-zone spatial encoding is the fix; validates entire thesis hypothesis |
| P13 | Action Recognition-based Industrial Safety Violation | arXiv Dec 2024 | Action context > appearance for violation detection (+23% F1); frozen appearance features insufficient |
| P14 | YOLOv10 + Transformer for PPE Detection | Nature Sci Rep 2025 | YOLO→transformer is 2025 production stack for industrial safety |
| P15 | ODTrack: Online Dense Temporal Token Learning | arXiv Jan 2024 | Temporal token propagation for tracking; design pattern for track-id token in temporal transformer |

---

## Overkill vs Cheap Wins

**Cheap wins (implement these):**
- DAB-DETR-style sinusoidal embedding of box (x,y,w,h) — 5 lines of code
- Precomputed IoU(person∩zone) and foot-point containment as explicit input scalars — purely geometric, no learning needed
- Class embedding per entity type — 1 lookup table
- BROS-style relative encoding for inter-box pairs (person-to-zone) — 10 lines
- ByteTrack + track-ID token — standard off-the-shelf
- Drop path + dropout 0.3 — 2 lines

**Overkill for N=3,000:**
- Relation DETR full encoder (designed for COCO-scale)
- 3D multi-view (Safe-Construct) — no stereo camera
- VLM-based reasoning (GridVAD) — needs LLM
- Full scene graph generation (VOST-SGG, Action Genome pipeline)
- Transformer-based tracker (MOTR, TrackFormer)
- GTA geometric transform attention — multi-view specific

---

## CVPR 2025 Specific Notes

The CVPR 2025 Open Access page (June 13 session) was not crawlable (403). However:
- **Safe-Construct** (arXiv 2504.10880) appears in IEEE Xplore as CVPR 2025
- **LP-DETR** (Feb 2025) is a strong 2025 DETR-family paper
- **RegFormer** (arXiv 2604.00507) is confirmed CVPR 2026
- GridVAD (arXiv 2603.25467) is referenced in a CVPR 2026 challenge workshop
- The main CVPR 2025 safety/spatial work is Safe-Construct; no direct walkway-zone paper found

---

## Unresolved Questions

1. Does Relation DETR's pairwise relation bias apply without modification to a non-detection classifier (no set-matching loss)? May need simplification to just additive attention bias without bipartite matching.
2. Is foot-point y_min (bottom of person bbox) a reliable proxy for foot position when person is partially occluded by zone marking? Need to validate on Unsafe-Net frames.
3. What is the actual annotation budget for bbox annotation? The thesis notes 400 frames (V10 pilot) → 1,200 (paper-scale). At what N does the spatial-relation transformer head add measurable lift over the frozen backbone alone?
4. Does the walkway polygon need to be manually annotated per video, or can it be detected (e.g., lane-marking detection)? If auto-detected, zone-IoU feature will have noise.
5. Track-ID token across clips: if a clip is 32 frames, ByteTrack IDs reset between clips — is clip-level track-ID sufficient or does cross-clip consistency matter?

---

## Key URLs
- DAB-DETR: https://arxiv.org/abs/2201.12329
- Relation DETR (ECCV 2024): https://arxiv.org/abs/2407.11699
- LP-DETR (2025): https://arxiv.org/abs/2502.05147
- GTA ICLR 2024: https://arxiv.org/abs/2310.10375
- BROS: https://huggingface.co/docs/transformers/en/model_doc/bros
- Action Genome: https://arxiv.org/abs/1912.06992
- Disentangled HOI Pre-training (CVPR 2024): https://arxiv.org/abs/2404.01725
- VOST-SGG: https://arxiv.org/abs/2512.05524
- RegFormer: https://arxiv.org/abs/2604.00507
- Safe-Construct (CVPR 2025): https://arxiv.org/abs/2504.10880
- Industrial Safety Violation Detection: https://arxiv.org/abs/2412.05531
- YOLOv10+Transformer PPE: https://www.nature.com/articles/s41598-025-12468-8
- ODTrack: https://arxiv.org/abs/2401.01686
