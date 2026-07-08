# Phase 2 — MT1 hybrid + paper-scale bbox annotation (staged)

**Decision (2026-06-13):** timeline 2–4 weeks; self-annotate ~15–20 h; **staged — pilot 1000 frames
first**, decision gate, then push to 2000+ only if the trend turns.

**Goal:** Execute the thesis's own Priority-1 — scale behaviour-class bbox annotation from 400 (V10,
regressed to 54/64) toward paper scale (~2000–2600), and replicate the **paper's actual mechanism**
(below), evaluating **blind**. Tests directly whether the gap is data-bound.

**Outcome reframes the thesis:** success (→ ~58–62/64) flips back to a positive-result spine (MT1+MT2
delivered as proposed); failure strengthens spine C. Either way the experiment is scientifically valid.

---

## ⭐ KEY INSIGHT FROM THE PAPER (Önal & Dandıl 2025) — corrects V10's mechanism

The paper's full text (read 2026-06-13) reveals YOLO v4 is a **FRAME SELECTOR / FILTER**, *not* a
feature extractor:
- YOLO v4 (trained on **2262** bbox images, **4 unsafe classes**, 416×416, mAP 95.66%) runs per frame
  and **keeps only frames where an unsafe-behaviour object is detected, discarding all others** →
  a *reduced* video → ConvLSTM classifies the reduced sequence.
- **Paper Table 5** — ConvLSTM **WITHOUT** YOLO frame-reduction = **85.78 % macro / 57·64 top-1**.
- **Paper Table 7** — full Unsafe-Net (YOLO filter **+** ConvLSTM) = **95.81 % macro / 62·64**.
- ⇒ **The entire +10 pp gain IS the frame-filtering**, enabled by the 2262 bbox annotations — not
  richer per-frame features.

**Consequences for this thesis (major):**
1. Our frozen-COCO pipeline (all frames, no filter) ≈ the paper's **85.78 % row**; our blind ≈ 52/64
   sits *below* even that — because we feed every frame, including the ambiguous ones the paper drops.
2. **V10 replicated the WRONG mechanism** — it used the bbox-finetuned YOLO as a *backbone feature
   extractor*, not as a *frame filter*. This plausibly explains the V10 regression and is itself a
   finding worth stating.
3. The correct MT1 experiment is therefore the **frame-selection architecture**, not feature swap:

   ```
   clip → [YOLOv11 detector, fine-tuned on ~2000 bbox, 4 unsafe classes]
        → keep frames with a confident unsafe detection (drop the rest)   ← THE mechanism
        → reduced frame sequence → temporal head (existing Transformer head OR ConvLSTM)
        → 4-class prediction
   ```
   This is a smaller change than a re-architecture: add a **YOLO frame-gate** in front of the existing
   temporal pipeline. The hypothesis under test becomes precise: *does YOLO-based frame selection,
   trained on paper-scale bbox, close the gap?* — the actual paper claim, never tested in V1–V10.

---

## Existing assets (reuse, don't rebuild)
- `run_finetune_bbox.py` — full pipeline: prep dataset → fine-tune YOLOv11s (imgsz 640) → extract
  behaviour-aware features → train Transformer head → ensemble → eval. Ran at 400 (= V10).
- `auto_annotate.py` — YOLO-World open-vocab first-pass → review in CVAT.
- `annotation_guide/annotations/class_{0..7}/` — **401 existing V10 labels** (`{cls}_tr{clip}_{frame}.txt`).
  ⚠ Source frames are gone — must re-extract from `data/train/` videos.
- 566 train videos, 8 classes, **severe imbalance**: cls0=178 … cls6=19.

## Four V10 flaws to fix (Phản biện 2d) — non-negotiable

1. **Representative sampling, not "easy" frames.** Stratify across classes AND temporal positions;
   oversample the confusion pair (WalkwayViol cls0 ↔ UnauthorInt cls1) and minority classes.
2. **Inter-annotator κ.** A second annotator labels a ≥100-frame subset; report Cohen's κ. Closes the
   "single annotator, no agreement" critique.
3. **Freeze schedule to preserve COCO prior.** `run_finetune_bbox.py` currently full-unfreezes at
   lr0=0.001 (likely why V10 lost the prior). Add: freeze backbone first ~30 epochs, then unfreeze only
   the deepest 2 C2f blocks at low lr (`freeze=` arg + two-stage schedule).
4. **Blind evaluation from the start.** Phase-4 head training must select epoch on a train-carved
   validation slice (reuse `run_blind_eval.py` logic), not the 64 test clips. The new number is honest
   on day one — no repeat of the §4.7.11 leakage.

## Annotation protocol (matches the paper, corrected from V10)
- **4 unsafe classes only** for the YOLO detector (paper trained YOLO on 4 classes: WalkwayViol,
  UnauthorInt, OpenedPanel, CarryOverload). Safe-class frames serve as **negatives/background**
  (no boxes) so the detector learns to *not* fire on them — this is what enables frame-filtering.
- **Box = the object/region exhibiting the behaviour**, labeled with the behaviour class (the OHS
  expert's judgment of "violation vs safe" is encoded in the class label, exactly as the paper did
  with LabelImg). Worker for cls0/1; open-panel region for cls2; forklift+load for cls3. Not a generic
  zone box — the *class label* carries the context, the box localizes the object.
- Paper used **2262 train + 329 test** boxes at 416×416. Target ~2000+ on the 4 unsafe classes.
- The existing extraction's unsafe-class frames (cls0-3 = 600 at the 1000 pilot) are the priority to
  annotate; cls4-7 frames are kept as background negatives (annotate as empty / no unsafe object).

---

## Staged execution

### Stage 0 — setup
- [x] `extract_frames_for_annotation.py` — **DONE**: 1000 frames extracted to
      `annotation_guide/frames_to_annotate/class_{0..7}/` (400 existing-label frames re-extracted at
      native res + 600 new, stratified; cls0/cls1 oversampled to 175 each). Manifest:
      `annotation_guide/frame_manifest.csv`. 0 failures.
- [x] `run_finetune_bbox.py` **freeze schedule** — DONE: `freeze=10` (backbone frozen, COCO prior
      preserved), lr0 0.001→0.0005. Env override `FT_FREEZE`. This is the #1 V10 fix.
- [ ] `run_finetune_bbox.py` **blind eval** — TODO before the pilot run: head training must select
      epoch on a train-carved val slice (port `run_blind_eval.py` logic), not the 64 test clips.
- [x] **YOLO frame-gate `run_yolo_framegate.py`** — WRITTEN (the paper's mechanism). Per clip: run
      fine-tuned YOLO on 32 cached frames → keep conf ≥ τ → resample to T=32 → gated stats from the
      ALREADY-CACHED frozen features (no re-extraction) → blind direct-4 head, τ + epoch selected on
      val. τ=0 = no-gate baseline. ⚠ untestable until YOLO weights exist; verify channel-order +
      row-alignment on first run (flagged in-file). Reuses `run_blind_eval` + `run_pairwise_contrastive`.
- [x] **`annotation_guide/LABELING-GUIDE.md`** — WRITTEN: 4-unsafe-class protocol, box=object /
      label=judgment, safe=background, κ subset, confusion-pair care. Read before CVAT.
- [x] **Edge-contribution sharpened** (§5.6.7): paper runs on 400 W workstation + Arduino alarm, no
      edge optimization → our 7 W Jetson contribution is orthogonal and uncontested.

### Stage 1 — pilot 1000 (you annotate ~8–10 h, then me)
- [ ] auto_annotate the +600 → export to CVAT → **you review/correct** (4 unsafe classes; safe = bg).
- [ ] κ subset (≥100 frames, 2nd annotator).
- [ ] Fine-tune YOLO (freeze=10, 4 classes) → build **frame-gate** → blind-eval the gated pipeline
      vs the no-gate baseline. Report YOLO mAP too (gate quality).

### 🚦 DECISION GATE at 1000 frames
- Two baselines to beat: frozen no-gate blind ≈ 52/64; paper's no-reduction row 57/64.
- **Go** if the **frame-gated** blind clearly exceeds the no-gate blind (the mechanism is working) and
  trends toward the paper's 85.78%-row → proceed to 2000+ to chase the full filter gain.
- **Stop** if the gate gives no lift even with clean 1000-frame detection → strong evidence the gain
  needs paper-scale gate quality (mAP), reported as a sharpened negative result; spine C stands.
- Either branch is publishable: we will have isolated *frame-selection* as the operative mechanism.

### Stage 2 — scale to 2000+ (only if Go; you annotate ~8–10 h more)
- [ ] Annotate +1000 → 2000+. Re-run blind. Compare 400 → 1000 → 2000 curve.

### Stage 3 — rewrite (1 week)
- [ ] If success: re-pivot Ch1/§1.3.1 (MT1+MT2 delivered), new results chapter section, update spine.
- [ ] If not: fold the 400→1000→2000 curve into §4.6 as a stronger negative result; spine C stands.

---

## Timeline (2–4 weeks)
| Week | Work |
|---|---|
| 1 | Stage 0 setup + Stage 1 annotate 1000 + pilot run + **decision gate** |
| 2 | (if Go) Stage 2 annotate to 2000 + run |
| 3 | Stage 2 finish + blind eval + analysis |
| 4 | Stage 3 chapter rewrite + consistency + rebuttal update |

## Parallel cheap win
- [ ] **Email Önal & Dandıl again** for the original Unsafe-Net bboxes (2591). If they release, skip
      all annotation labor + get a perfect replication. Chase in parallel.

## Success criteria
- Honest **blind** test number on the bbox-finetuned hybrid, with the 400→1000(→2000) curve.
- κ reported. Freeze-schedule preserves COCO prior (detection mAP doesn't collapse).
- A clear, defensible answer to "does paper-scale bbox close the gap?" — yes (positive pivot) or no
  (strengthened negative result).

## Risks
- Annotation quality variance → mitigated by auto-annotate first-pass + κ + freeze schedule.
- Time overrun → mitigated by the 1000-frame decision gate (don't sink 2000 worth of labor blindly).
- Still regresses at 2000 → acceptable; spine C is the fallback and is strengthened.

## Unresolved
- Exact zone-boundary annotation protocol (how to box "restricted zone") — needs a 1-page labeling
  guide before Stage 1. Draft next.
- Whether to fine-tune at imgsz 640 (detection) but keep 224 for the temporal-feature extraction —
  confirm the two-resolution split is intended (it is, per §4.7.10 + §5.7).
