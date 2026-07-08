# Research: Closing the Accuracy Gap (cited, 2024–2026)

**Premise:** gap to paper's 95.81% is DATA-bound (paper used 2,262 per-frame bbox encoding zone boundaries; thesis uses frozen COCO YOLOv11s, no zone supervision). Root failure = WalkwayViolation ↔ UnauthorizedIntervention confusion (both "person in restricted area").

## Ranked highest-ROI moves
| Rank | Technique | Effort | Expected gain | Verdict |
|---|---|---|---|---|
| 1 | **Virtual fencing** — 1 polygon/camera, person∩zone geometric prior (non-learned) | 1–2 hrs | +6–12% on zone-confused classes | PURSUE |
| 2 | Grounding DINO + SAM 2 auto-annotation + active-learning review (~200–400 frames) | 2–4 days | unlocks spatial supervision, fixes confusion | PURSUE |
| 3 | VideoMAE ViT-B (frozen early layers) + video augmentation (SV-Mix) | 1–2 days | +3–6% on motion classes | PURSUE |
| 4 | RT-DETRv3 backbone swap (NMS-free, crowded scenes) | 3–5 days | +1–3%, only if #1/#2 done | MAYBE |
| — | InternVideo2 (6B), full SSOD loop, synthetic video, TimeSformer | high | marginal/uncertain at 566 clips | SKIP |

**Combined realistic ceiling (ranks 1–3): ~91–93% macro at near-zero annotation cost** — publishable as a Pareto trade vs paper's 190-hr bbox effort.

## Key reasoning
- **Virtual fencing** = industry-standard ("geofencing"/"no-access zone"). Encodes zone boundary as static geometry, 0 learned params, directly disambiguates the two confused classes. Cheapest root-cause fix.
- **Auto-annotation**: Grounding DINO (open-source) / YOLO-World (already integrated; re-parameterizable, 0 deploy cost) + SAM2 mask propagation pseudo-label "person"/PPE; active learning annotates only ~200–300 low-confidence frames (vs 2,262). ~94% annotation-time reduction reported.
- **Temporal**: handcrafted derivative streams failed because problem is spatial-zone, not temporal. VideoMAE is explicitly data-efficient (designed for HMDB-51 scale; 566 clips < that). Replaces dead streams; won't fix zone confusion.
- **RT-DETR**: orthogonal to data gap; helps only as feature-quality bump after spatial supervision exists. Use to answer "why was RT-DETR dropped".

## Sources
- Grounding DINO 1.5 — arXiv:2405.10300 · YOLO-World (CVPR 2024) — arXiv:2401.17270 · OWLv2
- SAM 2 — arXiv:2408.00714 · Grounded-SAM-2 (github IDEA-Research) · CVAT SAM2 integration
- SSOD survey (PMC12788260) · PseCo · DeLR active learning — arXiv:2312.16931
- Virtual fencing — arXiv:2503.20237; tensorvok / overdriveiot industrial geofencing
- RT-DETR — arXiv:2304.08069 · v2 2407.17140 · v3 (WACV2025) 2409.08475 · v4 2510.25257
- VideoMAE — arXiv:2203.12602 · VideoMAEv2 2303.16727 · InternVideo2 (ECCV2024) · FROSTER (ICLR2024) · EZ-CLIP 2312.08010
- Low-label transformers — arXiv:2209.07474 · SV-Mix (IJCV2025) 2309.09534 · ReMA 2601.00311
- iSafetyBench (ICCVW2025) — arXiv:2508.00399 · SIAVC — arXiv:2405.14506
- Unsafe-Net — Springer s11042-024-19276-8 / PMC11367630
