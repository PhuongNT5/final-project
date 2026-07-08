---
priority: parallel-track
effort: 4–6 person-days post-download (download runs in background)
depends_on: []
branches: A=zero-shot anomaly/normal binary, B=light-finetune (10% labels) on curated subset
---

> **PARALLEL TRACK — runs alongside defense critical path.**
> Not on the 5-week defense critical path (Phases 01–03, 06–08). Pre-download starts
> NOW (background) so data is ready when spare cycles open. If defense materials slip,
> pause this phase. Output lands in §4.8 if completed; otherwise becomes a Future Work
> bullet in Ch 7. UCF-Crime targeted instead of HMDB51 (committee-level surveillance
> dataset, larger, well-known anomaly benchmark).

# Phase 05 — Cross-Dataset Generalization (UCF-Crime)

## Context Links

- `report/chapter-04-experiments-and-results.md` — add §4.8 (new section, 1 table + 1 paragraph)
- `report/chapter-02-related-work.md` §2.3 — industrial / surveillance video anomaly survey; cite UCF-Crime here
- `PROGRESS.md` — all experiments are Unsafe-Net-only; this is a gap committee will probe
- `data/ucf_crime/` — dataset directory (gitignored; not committed)
- `run_cross_ensemble.py` (untracked, in repo root) — currently does cross-ensemble within Unsafe-Net; reuse the inference loop, swap loaders

## Overview

All 9 experiment tracks in Chapter 4 use only Unsafe-Net (single factory, 2 cameras,
Turkey). The committee will ask whether the architecture generalizes. This phase tests
the frozen-backbone + best-head ensemble on UCF-Crime — the standard real-world
surveillance anomaly benchmark (Sultani et al., CVPR 2018; 1900 untrimmed clips,
13 anomaly classes + Normal). Two branches:

- **Branch A (zero-shot):** apply Unsafe-Net-trained Direct4 head to UCF-Crime test
  clips with a class-mapping heuristic (Unsafe-Net's 4 unsafe behaviors → UCF-Crime
  semantically-closest anomaly classes; binary anomaly-vs-normal evaluation).
- **Branch B (light-finetune):** freeze COCO YOLOv11s backbone + transformer
  encoder; replace 4-class head with binary {anomaly, normal} head; train on 10 % of
  UCF-Crime training labels; evaluate on UCF-Crime test.

Goal: establish non-triviality (features transfer across datasets), not SOTA. Minimum
viable test.

## Key Insights

- UCF-Crime has 13 anomaly classes; we curate **5 classes most semantically aligned
  with industrial unsafe behaviors** to keep download/compute tractable on the M4
  (~30 GB curated subset vs ~128 GB full):
  - **Fighting** ↔ Unsafe-Net `UnauthorIntervention` (interpersonal aggression)
  - **Vandalism** ↔ `UnauthorIntervention` (object destruction in restricted area)
  - **Stealing** ↔ `SafeWalkwayViolation` (person in unauthorized zone)
  - **Burglary** ↔ `OpenedPanel` (breaching/opening enclosures)
  - **RoadAccidents** ↔ `CarryOverload` (loose proxy: collision/load events; weakest mapping, document this)
  - **Normal_Videos_event** — sample ≈ 200 clips from the ~800 normal training clips
- Frozen COCO YOLOv11s backbone is dataset-agnostic (COCO covers `person`, `vehicle`,
  generic indoor/outdoor scenes). Spatial features should transfer.
- Zero-shot Direct4 head will likely underperform random on multi-class — label
  spaces are non-overlapping. Reframe Branch A as **binary anomaly vs normal**:
  collapse all 4 Unsafe-Net unsafe-classes into "anomaly", `SafeWalkway` into "normal";
  evaluate on UCF-Crime binary task. This is fairer.
- Branch B (light-finetune) is the **stronger generalization claim**: shows the
  COCO+transformer feature stack admits new classification heads with minimal data.
- **Disk constraint: 41 GB free.** Full UCF-Crime is ~128 GB. We must download a
  curated subset (~30 GB) — the 5 anomaly classes above + a Normal sample. Strategy
  documented in download script.
- Pre-download runs in background (Phase 05 doesn't block defense critical path); this
  is why we kick it off NOW even though active work starts post-defense.

## Requirements

**Functional**
- Download curated UCF-Crime subset (5 anomaly classes + Normal sample) → ~30 GB.
- Branch A: zero-shot binary {anomaly, normal} evaluation of Direct4-s314 head on
  UCF-Crime test split.
- Branch B: light-finetune binary head on 10 % of UCF-Crime train; 5-seed sweep;
  report mean ± std macro accuracy on UCF-Crime test.
- Comparison table for §4.8 (≥ 4 rows: zero-shot, light-finetune, random, Unsafe-Net reference).

**Non-functional**
- Download ≤ 30 GB (curated; full would exceed disk).
- License: research use; cite Sultani et al. (CVPR 2018). Do not commit data to git.
- Compute: ≤ 12 h total (download in background; finetune ≈ 4 h).

**Acceptance thresholds**
- Branch B macro accuracy > random (≥ 55 % on binary).
- Table in §4.8 has ≥ 4 rows.
- §4.8 ≤ 1.5 pages with 1 table + 1 paragraph.
- Disk usage stays under 35 GB at peak.

## Architecture / Approach

### Dataset: UCF-Crime Curated Subset

**Source options surveyed (2026-05-01):**

| Mirror | Size | Format | Verdict |
|---|---|---|---|
| `odins0n/ucf-crime-dataset` | 11.8 GB | **64×64 PNG frames** | rejected — too low-res for 224×224 backbone |
| `mission-ai/crimeucfdataset` | 35 GB | mp4 nested in `Anomaly_Dataset/.../Part-N/` | fallback for Normal videos |
| `alirakhmaev/ucf-crime-full` | **24.4 GB** | mp4 320×240 h264 30fps; clean `<Class>/<Clip>.mp4` | **CHOSEN** — anomaly classes |
| `minhajuddinmeraj/anomalydetectiondatasetucf` | 41 GB | pre-extracted feature txt | rejected — features not videos |
| HF `jinmang2/ucf_crime` | partial | LFS-hosted | unverified fallback |

**Chosen: `alirakhmaev/ucf-crime-full`** for the 5 anomaly classes
- **320×240 mp4 clips** at 30 fps, ~90 sec average — original UCF-Crime resolution.
- Clean flat layout: `Fighting/Fighting002_x264.mp4`, `Vandalism/Vandalism001_x264.mp4`, …
- All 5 target classes confirmed present (some clip-001 missing; clip-002+ exist).
- **Anomaly-only mirror** — no Normal_Videos folder.
- Download: 24.4 GB zip → selectively extract 5 target classes (~10 GB) → delete zip
  → final disk usage ~10 GB. Peak during extract ≈ 35 GB used.

**Normal-videos source: deferred** (anomaly download in progress at time of writing).
Probe `mission-ai/crimeucfdataset` once anomaly download lands; it has nested
`Anomaly_Dataset/...` paths and likely a sibling `Normal_Videos/` somewhere.
HuggingFace `jinmang2/ucf_crime` is the second fallback. **If Normal cannot be
sourced cleanly, fall back to "5-way anomaly classification" framing instead of
binary** (acceptable degraded scope; light-finetune still demonstrates feature transfer).

**On framing — clip-level vs MIL:**

UCF-Crime original protocol is weakly-supervised MIL on full untrimmed videos.
We use the simpler clip-level protocol matching Unsafe-Net (T=32 uniform-sampled
frames per clip, one label per clip). Justification: this phase is a generalization
non-triviality check, not a UCF-Crime SOTA attempt. State this explicitly in §4.8.

Curated subset (downloaded files):
```
data/ucf_crime/
├── Anomaly/
│   ├── Fighting/        # ~50 clips
│   ├── Vandalism/       # ~50 clips
│   ├── Stealing/        # ~100 clips
│   ├── Burglary/        # ~100 clips
│   └── RoadAccidents/   # ~150 clips
├── Normal/
│   └── Normal_Videos_event/  # ~200 clips sampled from ~800
└── splits/
    ├── train.txt
    └── test.txt
```

### Branch A — Zero-Shot Binary

- Load `direct4-s314` (best Unsafe-Net 4-class head).
- For each UCF-Crime test clip: forward through frozen-COCO + Direct4; argmax → 4-class label.
- Map: classes {0,1,2,3} = unsafe → "anomaly"; class 4 (Safe Walkway) → "normal".
- Compute binary macro accuracy on UCF-Crime test (50/50 anomaly/normal balanced subset).
- Expected: 55–70 % (better than chance because COCO features detect persons/vehicles which correlate with both unsafe and anomaly).

### Branch B — Light-Finetune Binary

- Freeze COCO YOLOv11s backbone + transformer encoder.
- Replace Direct4 head with `BinaryHead` (Linear: `512 → 2`).
- Train: 10 % of UCF-Crime train labels (≈ 65 clips total: ~13 per class avg + 13 normal).
- Hyperparameters: same as V3 (Adam, lr=1e-3, dropout 0.5, 50 epochs, early stop on val acc).
- 5-seed sweep (seeds 7, 314, 1729, 2024, 2027) for variance reporting.
- Test on held-out UCF-Crime test (≈ 130 clips: balanced anomaly/normal).

### Table Structure (§4.8)

| Method | Train clips | Test clips | Macro Acc (mean ± std) |
|--------|-------------|------------|------------------------|
| Frozen Unsafe-Net Direct4 head (zero-shot) | 0 | ~130 | X.X % |
| Light-finetune (10 % UCF-Crime labels), 5 seeds | ~65 | ~130 | X.X ± Y.Y % |
| Random baseline (binary) | — | ~130 | 50.0 % |
| Reference: best ensemble on Unsafe-Net | 452 | 64 | 89.28 % |

## Related Code Files

| Action | Path |
|--------|------|
| Create | `data/ucf_crime/download_ucf_crime.sh` — multi-method download with curated subset |
| Create | `run_cross_dataset.py` — Branches A + B; reuses VideoDataset + inference loop from `run_cross_ensemble.py` |
| Read/reuse | `run_cross_ensemble.py` — inference loop, ensemble logic |
| Read/reuse | `run_pairwise_contrastive.py` — VideoDataset class (T=32, 224×224) |
| Modify | `.gitignore` — add `data/ucf_crime/` |
| Modify | `report/chapter-04-experiments-and-results.md` — add §4.8 |
| Modify | `report/chapter-02-related-work.md` §2.3 — cite Sultani et al. (UCF-Crime) |

## Implementation Steps

1. **(NOW) Configure Kaggle API token** (one-time, ~3 min):
   - Visit `https://www.kaggle.com/settings` → scroll to "API" → click **Create New Token**.
   - Browser downloads `kaggle.json` (contains `username` + `key`).
   - Move and lock it down:
     ```
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
     chmod 600 ~/.kaggle/kaggle.json
     ```
   - Install the CLI: `pip install kaggle`
   - Sanity check: `kaggle datasets list -s ucf-crime` should print result rows.
2. **(NOW) Pre-download** (background, resumable, ~30 GB):
   ```
   nohup bash data/ucf_crime/download_ucf_crime.sh > data/ucf_crime/download.log 2>&1 &
   tail -f data/ucf_crime/download.log   # monitor; Ctrl-C to stop tailing (download keeps running)
   ```
2. **(NOW) `.gitignore`**: append `data/ucf_crime/` to `.gitignore`.
3. **Verify download**: count clips per class; confirm 5 anomaly + Normal subdir each
   has ≥ 50 clips; spot-check 5 clips decode in ffmpeg.
4. **Adapt VideoDataset** for UCF-Crime: clips are mp4/avi at varying FPS; uniform
   sample T=32 frames per clip; resize to 224×224. Test on 5 clips end-to-end.
5. **Build train/test split** (80/20 stratified by class; save to `splits/`).
6. **Branch A — Zero-shot inference** (`run_cross_dataset.py --branch A`):
   load direct4-s314; forward all test clips; map argmax → binary; compute accuracy.
7. **Branch B — Light-finetune** (`run_cross_dataset.py --branch B --seed N`):
   subsample 10 % train; freeze backbone+encoder; train binary head; save best-val checkpoint per seed.
8. **Run 5 seeds**; collect test accuracies; compute mean ± std (bootstrap CI optional).
9. **Build §4.8 table** in `report/chapter-04-experiments-and-results.md`.
10. **Write §4.8 narrative** (≤ 1.5 pages): dataset description, mapping rationale,
    Branch A interpretation (label-space mismatch dampens zero-shot), Branch B result,
    1-paragraph conclusion ("COCO+transformer features admit new heads with 65 clips;
    confirms architecture is not Unsafe-Net-specific").
11. **Add citation** in Ch 2 §2.3: Sultani, Chen & Shah (CVPR 2018), "Real-world Anomaly Detection in Surveillance Videos".

## Todo List

- [ ] 1. Install kaggle CLI and configure ~/.kaggle/kaggle.json
- [ ] 2. Append `data/ucf_crime/` to .gitignore
- [ ] 3. Run `data/ucf_crime/download_ucf_crime.sh` (background)
- [ ] 4. Verify download: clip counts per class, ffmpeg decode sanity
- [ ] 5. Adapt VideoDataset for UCF-Crime (T=32, 224, binary labels)
- [ ] 6. Build stratified 80/20 split files
- [ ] 7. Branch A: zero-shot inference with direct4-s314 → binary mapping → accuracy
- [ ] 8. Branch B: light-finetune binary head, 5 seeds × 50 epochs
- [ ] 9. Aggregate results, compute mean ± std
- [ ] 10. Write §4.8 (1 table + 1 paragraph)
- [ ] 11. Add Sultani et al. (2018) citation to Ch 2 §2.3

## Success Criteria

- §4.8 exists with ≥ 4-row comparison table and 1-paragraph narrative.
- Branch B macro accuracy > 55 % (binary, > random).
- Section ≤ 1.5 pages; non-triviality established without overclaiming.
- `data/ucf_crime/` is gitignored.
- Disk peak ≤ 35 GB.

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Kaggle dataset removed / requires acceptance | Low | Fallback: rclone Dropbox mirror; or HuggingFace partial |
| Download exceeds 35 GB | Medium | Curated subset only (5 classes + Normal sample); validate per-class size before full download |
| UCF-Crime clips at varied FPS break VideoDataset | Low | Uniform-frame sampler in loader; ffmpeg pre-normalize fallback |
| Zero-shot Branch A near random | High | Expected; report and frame as "label-space mismatch motivates Branch B" |
| 65-clip light-finetune overfits | Medium | Dropout 0.5, early stopping on val acc, 5-seed sweep |
| Defense materials slip → pause Phase 05 | Medium | Phase 05 explicitly off the critical path; download completes in background regardless |
| Class-mapping heuristic disputed by committee | Medium | Document mapping rationale + state RoadAccidents↔CarryOverload is weakest; primary claim = Branch B (no mapping needed) |

## Security / Ethics / Privacy

- UCF-Crime is research-use-only (Sultani et al., CC BY-NC). Cite properly. Never commit data.
- Contains real-world surveillance footage of crime; treat as sensitive. No redistribution. Do not include frames in thesis figures unless faces are blurred or only schematic illustrations are used.
- No PII collected by us; original dataset's IRB applies.

## Dependencies + Next Steps

- No dependency on other phases (parallel track).
- Output consumed by: **Phase 07** (§4.8 new section; §1.4 Contribution 5 references generalization).
- Landing chapter: `report/chapter-04-experiments-and-results.md` §4.8 (new); 1 sentence in Ch 1 §1.4 Contributions.

**Resolved questions (2026-05-01)**
- **Kaggle token**: configured (`access_token`-style, kagglehub format).
- **Mirror format**: `alirakhmaev/ucf-crime-full` chosen — 320×240 mp4, all 5 target anomaly classes confirmed. `odins0n` rejected (64×64 PNGs).
- **Clip-level vs MIL**: clip-level chosen; T=32 uniform sampling, one label per video.
- **`run_cross_ensemble.py` binary logic**: hardcoded 4-class only — binary collapse goes in `run_cross_dataset.py`.

**Unresolved questions**
- **Normal-videos source**: alirakhmaev anomaly-only. Need to probe `mission-ai/crimeucfdataset` for `Normal_Videos*` folder (rate-limited at time of writing) OR fall back to 5-way anomaly classification. Decide post-anomaly-download.
- **Per-class clip counts in alirakhmaev**: ~50/class expected (matching UCF-Crime spec). Confirm via `bash data/ucf_crime/download_ucf_crime.sh status` after extraction.
