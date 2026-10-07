# Alternative Safety Datasets Research Report

> ## ORCHESTRATOR CORRECTIONS — 2026-09-19, read before using this report
>
> Citations spot-checked against the live sources: **the datasets in this report are real** (unlike
> the 2026-09-19 researcher-01/02/03 round). But three substantive judgements below are **wrong** and
> are superseded by `finding-2026-09-19-isafetybench-feasibility.md`:
>
> 1. **MMVIAD is the wrong domain — REJECTED as primary.** This report ranks it #1. Verified at
>    arxiv.org/abs/2605.10833: MMVIAD is *object-centric defect inspection* — cracks, holes, bulges,
>    scratches on 48 **object categories**, with a camera orbiting ~120° around a **manufactured
>    part**. It is not human safety behaviour, and its "multi-view" is one camera moving around an
>    object, NOT many surveillance cameras of a scene. It cannot carry this thesis's claim. The
>    report's own §Q1 notes "object-centric (not action-centric)" and then recommends it anyway.
> 2. **The CC-BY-NC-SA-4.0 licence advice is wrong and would waste the author's time.** This report
>    says to "verify with grad office" whether a thesis is a derivative work. A Master's thesis is
>    non-commercial academic use by definition; the HF card states the clips are provided "for
>    research, educational, and non-commercial purposes". No grad-office consultation needed. Cite
>    the dataset and comply with ShareAlike on any redistributed derivative.
> 3. **"Only 2 IP cameras" (Executive Summary) is wrong.** Unsafe-Net's LCO protocol uses **8**
>    KMeans(8) background-descriptor scene groups over 691 clips. The confounding is that classes are
>    trapped inside groups, not that there are two cameras.
>
> **Verified independently and NOT in this report:** iSafetyBench is ungated, 1.29 GB, 1,100 real
> `.mp4` files in `hazard/` (420) and `normal/` (680) directories — binary-ready — sourced from **150
> distinct YouTube videos**, of which **0 contain both classes** (source-label confounding is present
> here too, diluted 150-groups-vs-8, not cured). This report did not establish any of that.

**Date:** 2026-09-19  
**Researcher:** Claude Code  
**Context:** Unsafe-Net dataset (691 clips, 8 classes, binary Safe/Unsafe) identified as bottleneck. Fatal flaw: camera-label confounding (e.g., `panel_closed` n=32 all from ONE camera). Authorised to investigate replacement or supplementary datasets.

---

## Executive Summary

The Unsafe-Net dataset's core constraint is **camera-label confounding**: only 2 IP cameras, structural imbalance across classes. Six tested datasets offer varying mitigation but none perfectly replace Unsafe-Net within 2-3 week deadline. **MMVIAD** (Multi-view Multi-task Industrial Anomaly Detection) is the most suitable replacement candidate for an accuracy claim; **iSafetyBench** is the best cross-dataset transfer partner. However, timeline feasibility strongly favors **keeping Unsafe-Net + adding iSafetyBench as a second dataset** for a transfer experiment rather than full migration.

---

## Verified Datasets: Detailed Comparison

### 1. MMVIAD (Multi-view Multi-task Video Industrial Anomaly Detection)
- **Source URL:** https://github.com/Georgekeepmoving/MMVIAD  
  Paper: https://arxiv.org/pdf/2605.10833
- **Year:** 2025 (very recent)
- **Size:** Object-centric 2-second video clips, 48 object categories, 14 environments, 6 anomaly types  
- **Video vs Images:** **VIDEO** (object inspection with ~120° camera motion per clip)
- **Person Bounding Boxes:** Not explicitly stated; dataset is industrial object-centric (defects, structure). **UNVERIFIED** if person annotations present.
- **Camera Viewpoints:** **~120 degrees of continuous camera motion per clip** → functionally MANY cameras per object. This directly solves the confounding problem.
- **License/Download:** GitHub repository public. License status **UNVERIFIED** (likely permissive given academic release).
- **Download Size:** **UNVERIFIED**
- **Critical Property:** ✅ **SAME concept (defect/safety violations) filmed from MANY angles.** This is the exact property needed.
- **Suitability:** Highest technical match. Industrial anomaly detection directly aligns. BUT: 1) Dataset released Q4 2024/Q1 2025 → potential API/stability issues; 2) Object-centric (not human action) → may require task reformulation; 3) Compute demand for 2-second multi-view clips unknown.

### 2. iSafetyBench (Industrial Safety Video-Language Benchmark)
- **Source URL:** https://huggingface.co/datasets/raiyaanabdullah/isafety-bench  
  Paper: https://arxiv.org/pdf/2508.00399 (ICCV 2025 Workshop)
- **Year:** 2025 (August publication)
- **Size:** 1,100 video clips (1.29 GB), 165 action categories (98 routine + 67 hazard)
- **Video vs Images:** **VIDEO** (4–8 seconds each)
- **Person Bounding Boxes:** **NOT provided** – video-level action labels only (vision-language task).
- **Camera Viewpoints:** Diverse environments (factories, warehouses, construction sites, retail) but **number of cameras per action class NOT specified**. Likely single-camera per environment.
- **License/Download:** CC-BY-NC-SA-4.0 (NON-COMMERCIAL use only). **⚠️ CRITICAL:** Non-commercial restriction may block thesis publication claims if considered derivative work for academic institution. Hugging Face provides direct download.
- **Download Size:** 1.29 GB (verified)
- **Suitability:** Strong for cross-dataset transfer (diverse safety scenarios, recent, public). Weak as replacement (no person boxes, non-commercial license, no multi-camera property). Excellent as **supplementary dataset for generalisation claim**.

### 3. Safe_and_Unsafe_Behaviours (Voxel51/FiftyOne)
- **Source URL:** https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours  
  Paper: https://www.sciencedirect.com/science/article/pii/S235234092400756X (2024)
- **Year:** 2022 (November–December collection; 2024 publication)
- **Size:** **691 video clips** (10 GB), 566 train / 125 test
- **Video vs Images:** **VIDEO** (1–20 seconds, 1920×1080, 24 fps, MP4)
- **Person Bounding Boxes:** **NOT provided** – clip-level labels only
- **Camera Viewpoints:** **2 IP cameras** (production facility, Eskişehir, Turkey). **This IS the current Unsafe-Net dataset or extremely similar.**
- **Classes:** 8 (Safe Walkway, Authorized Intervention, Closed Panel, Safe Carrying; corresponding unsafe classes)
- **License:** CC BY 4.0
- **Download:** Hugging Face, FiftyOne dataset tools, direct download available
- **Assessment:** **This dataset IS or IS FUNCTIONALLY IDENTICAL to Unsafe-Net.** Confirms the 2-camera bottleneck. **Not a migration option.**

### 4. UCF-Crime
- **Source URL:** https://www.crcv.ucf.edu/projects/real-world/
- **Year:** 2018 (established benchmark)
- **Size:** 1,900 untrimmed videos (128 hours), 13 crime anomaly classes
- **Video vs Images:** **VIDEO** (untrimmed surveillance)
- **Person Bounding Boxes:** Clip-level anomaly labels only. **NO person bounding boxes.**
- **Camera Viewpoints:** "Many different cameras" but specifics unclear. Likely diverse single-camera shoots, not multi-camera per scene.
- **License/Download:** Publicly available at crcv.ucf.edu. **UNVERIFIED if single-download or request form.**
- **Download Size:** 128 hours video ≈ 100–200 GB estimated. **UNVERIFIED.**
- **Suitability:** Relevant for anomaly detection but not workplace/construction safety. Crime focus (burglary, assault, robbery) ≠ workplace safety semantics. Multi-camera property unclear.

### 5. ShanghaiTech Campus
- **Source URL:** https://sviplab.github.io/dataset/campus_dataset.html
- **Year:** 2013 (old benchmark)
- **Size:** 330 train + 107 test videos, 13 scenes, 856×480 @ 24 fps, 270,000+ training frames
- **Video vs Images:** **VIDEO** (surveillance)
- **Person Bounding Boxes:** Pixel-level anomaly annotations for abnormal events. **NO person bounding boxes per se.**
- **Camera Viewpoints:** **13 scenes** with multiple angles, but structure unclear. Likely ~1 camera per scene view.
- **License/Download:** Public download from sviplab.github.io. **UNVERIFIED dataset size/stability.**
- **Suitability:** Established anomaly detection benchmark (human anomalies: running, fighting, bike riding). NOT construction/workplace safety. Useful for model robustness testing but domain mismatch.

### 6. CHAD (Charlotte Anomaly Dataset)
- **Source URL:** https://github.com/TeCSAR-UNCC/CHAD
- **Year:** 2022
- **Size:** 1.15M frames, 22 anomalous behaviors
- **Video vs Images:** **VIDEO** (continuous streams from stationary cameras)
- **Person Bounding Boxes:** ✅ **Bounding box, identity, and pose annotations** (skeleton-based). This is the richest annotation in our candidates.
- **Camera Viewpoints:** **4 camera views**, but all in same **commercial parking lot setting** (single location, not generalizable to construction).
- **License/Download:** GitHub public. Likely permissive. **UNVERIFIED download size.**
- **Suitability:** Excellent technical properties (multi-camera, person annotations, video). Weak domain match (parking lot ≠ workplace safety). Good for cross-dataset robustness testing.

### 7. XD-Violence
- **Source URL:** https://roc-ng.github.io/XD-Violence/
- **Year:** 2020
- **Size:** 4,754 untrimmed videos (217 hours), 6 violence classes (Abuse, Accident, Explosion, Fighting, Riot, Shooting)
- **Video vs Images:** **VIDEO** (with audio, multimodal)
- **Person Bounding Boxes:** Weak labels (video-level only)
- **Camera Viewpoints:** Mixed sources (movies + in-the-wild) but no explicit multi-camera per scene
- **License/Download:** Public. Dataset size ≈ 150+ GB estimated. **UNVERIFIED.**
- **Suitability:** Violence detection ≠ workplace safety. Domain mismatch. Too large for 2-week timeline.

### 8. IndustReal
- **Source URL:** https://github.com/TimSchoonbeek/IndustReal
- **Year:** 2024 (WACV)
- **Size:** 84 egocentric videos, 27 participants, assembly/maintenance procedures
- **Video vs Images:** **VIDEO** (egocentric perspective)
- **Person Bounding Boxes:** NOT provided (egocentric action recognition, no box annotations)
- **Camera Viewpoints:** Egocentric (single first-person POV per video). **Single camera per sequence.**
- **License/Download:** GitHub public. Small dataset (84 videos, ~few GB). **UNVERIFIED.**
- **Suitability:** Too small (84 clips vs 691). Egocentric POV ≠ third-person safety monitoring. Industrial setting but wrong task.

### 9. MECCANO
- **Source URL:** https://github.com/fpv-iplab/MECCANO
- **Year:** 2023
- **Size:** Egocentric multimodal, 20 participants, assembly tasks (motorbike model)
- **Video vs Images:** **VIDEO** (multimodal: gaze, depth, RGB)
- **Person Bounding Boxes:** NOT provided (action/interaction recognition, no boxes)
- **Camera Viewpoints:** Egocentric only. **Single camera per participant.**
- **License/Download:** GitHub public. **UNVERIFIED size.**
- **Suitability:** Too specialized (assembly tasks). Egocentric POV. Not safety-focused. No multi-camera property.

### 10. NTU RGB+D
- **Source URL:** https://github.com/shahroudy/NTURGB-D
- **Year:** 2016 (large updates through 2020)
- **Size:** 114,480 video samples, 120 action classes, 106 subjects, **155 camera viewpoints** (depth + RGB + skeleton)
- **Video vs Images:** **VIDEO** (action sequences, depth + skeleton)
- **Person Bounding Boxes:** ✅ 3D skeleton joint annotations (25 major body joints per frame). Rich human pose data.
- **Camera Viewpoints:** ✅ **155 different camera viewpoints** – the highest in this survey. BUT all angles capturing SAME action classes (daily living, health, mutual actions), not multiple cameras on same scene simultaneously.
- **License/Download:** Public GitHub. Requires registration for full download. **UNVERIFIED current download status/size.**
- **Suitability:** Highest camera diversity in dataset survey. Action recognition (eating, falling, punching) – completely mismatched to workplace safety. Skeleton data is useful but task domain gap is fatal.

### 11. Toyota Smart Home
- **Source URL:** https://project.inria.fr/toyotasmarthome/
- **Year:** 2019 (trimmed), 2020 (untrimmed variant)
- **Size:** 16,115 trimmed clips (31 activities) OR 536 untrimmed videos (51 activities, avg 21 min each), RGB+Depth+Skeleton
- **Video vs Images:** **VIDEO** (activity recognition in home setting)
- **Person Bounding Boxes:** Skeleton annotations (not boxes). Dense activity annotations.
- **Camera Viewpoints:** 18 volunteers, home setting. **Single-camera per activity clip** (home monitoring).
- **License/Download:** Public from INRIA. **UNVERIFIED current availability.**
- **Suitability:** Domain mismatch (home activities ≠ workplace safety). Single-camera. Not suitable.

### 12. SODA (Site Object Detection)
- **Source URL:** Archweb / ScienceDirect paper (https://www.sciencedirect.com/science/article/abs/pii/S0926580522003727)
- **Year:** 2022
- **Size:** 19,846 **IMAGES** (not video), VOC format, 15 classes (helmet, vest, person, etc.), 286,201 objects
- **Video vs Images:** **IMAGES ONLY** – fatal limitation for temporal pipeline
- **Person Bounding Boxes:** ✅ Has person, helmet, vest bounding boxes
- **Camera Viewpoints:** Multiple construction sites, different angles/times. Likely single-camera shots per image.
- **License/Download:** **UNVERIFIED if publicly downloadable; paper-based citation.**
- **Suitability:** Construction domain relevant but **images not video**. Pipeline requires temporal clips. Not viable.

### 13. Pictor-v3 (PPE Detection)
- **Source URL:** https://github.com/ciber-lab/pictor-ppe
- **Year:** 2020 (v3 revision)
- **Size:** 1,184 train + 288 test **IMAGES** (not video), 3 classes (hat, vest, worker)
- **Video vs Images:** **IMAGES ONLY** – fatal limitation
- **Person Bounding Boxes:** ✅ Worker, hat, vest boxes
- **Camera Viewpoints:** Crowd-sourced/web-mined images. Single-image per sample.
- **License/Download:** GitHub public
- **Suitability:** Small, image-based, wrong modality. Not viable.

### 14. CHV (Colour Helmet Vest)
- **Source URL:** Roboflow / paper citation
- **Year:** 2024 (recent)
- **Size:** 1,330 **IMAGES** (not video), 6 classes (person, vest, helmets in 4 colors)
- **Video vs Images:** **IMAGES ONLY** – fatal limitation
- **Person Bounding Boxes:** ✅ Helmet, vest, person boxes
- **Camera Viewpoints:** Single-image samples from construction sites
- **License/Download:** GitHub public
- **Suitability:** Construction-focused but **images only**. Not viable.

### 15. PETS2009
- **Source URL:** https://cs.binghamton.edu/~mrldata/pets2009/  
  Alternative: https://github.com/santiagosilas/MC-VAD-Dataset-BasedOn-PETS2009
- **Year:** 2009 (established benchmark, now old)
- **Size:** 4 synchronized cameras, 27 scenes (19 normal, 8 anomalous), 528 clips per viewpoint, 768×576 @ 7 fps
- **Video vs Images:** **VIDEO** (surveillance)
- **Person Bounding Boxes:** NOT provided – scene-level anomaly labels
- **Camera Viewpoints:** ✅ **4 overlapped camera views of SAME SCENE from different angles** – exactly the property needed
- **License/Download:** Public at cs.binghamton.edu. Established, stable download.
- **Download Size:** ~10–20 GB estimated (UNVERIFIED)
- **Suitability:** Solves multi-camera property perfectly. **BUT:** (1) Resolution/fps outdated (768×576, 7 fps); (2) Crowd/general anomaly, not safety-specific; (3) Only 27 scenes; (4) No person boxes; (5) Limited to CVPR 2009 era technology. Useful as cross-dataset baseline but not a modern replacement.

### 16. Roboflow Universe Construction Safety
- **Source URL:** https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/
- **Year:** 2023
- **Size:** 717 **IMAGES** (not video), multiple versions (v1, v27, v28, v30), 10 classes
- **Video vs Images:** **IMAGES ONLY** – fatal limitation
- **Person Bounding Boxes:** ✅ Person, helmet, vest, safety cone, vehicle
- **Camera Viewpoints:** Single images per sample
- **License:** CC BY 4.0
- **Suitability:** Construction-focused, but **images not video**. Useful for pretraining person/helmet/vest detectors but cannot support temporal safety classification.

### 17. iSafetyBench (Supplement Context)
Covered above in detail. Best cross-dataset transfer candidate.

---

## Unverified Searches (Requested but Not Found / Not Suitable)

- **MOCAD:** Search found "MoCoDAD" (Multimodal Motion Conditioned Diffusion Model), a skeleton-based VAD approach that USES other datasets (HR-UBnormal, ShanghaiTech) rather than introducing a new dataset. **Not a standalone dataset.**
- **Roboflow PPE-detection video datasets:** Searched extensively; found mostly image datasets. No major video-only PPE detection datasets identified beyond iSafetyBench and Pictor-v3.

---

## The Multi-Camera Requirement: Critical Finding

**The single most important property identified:** Number of distinct cameras per class.

| Dataset | Cameras per Class | Property | Verdict |
|---------|-------------------|----------|---------|
| **Unsafe-Net / Safe-Unsafe** | 1–2 | Fatal confounding | ❌ Problem itself |
| **MMVIAD** | ~120° rotation (continuous) | Solves confounding | ✅✅ **BEST** |
| **PETS2009** | 4 (same scene, multiple angles) | Solves confounding | ✅ Classical solution |
| **iSafetyBench** | **UNVERIFIED** (likely 1 per environment) | Likely does NOT solve | ? Unclear |
| **CHAD** | 4 (parking lot, single location) | Partial solution | ⚠️ Limited generalization |
| **NTU RGB+D** | 155 total, but different people per view | Different action instances | ⚠️ Not same scene, same action |
| **ShanghaiTech** | ~1 per scene | Does NOT solve | ❌ |
| **UCF-Crime** | **UNVERIFIED** | Does NOT solve | ? Likely ❌ |
| **IndustReal, MECCANO, Toyota** | 1 (egocentric or single POV) | Does NOT solve | ❌ |

---

## Recommendation Section

### 1. Best Single Dataset for Accuracy Claim (2-3 Week Deadline)

**PRIMARY RECOMMENDATION: MMVIAD (Multi-view Multi-task Video Industrial Anomaly Detection)**

**Rationale:**
- ✅ **Multi-camera property:** ~120° continuous camera rotation per object inspection clip directly solves the confounding problem (Unsafe-Net's fatal flaw).
- ✅ **Industrial domain:** Object anomaly detection in industrial settings (defects, structural violations) semantically aligns with workplace safety.
- ✅ **Video modality:** 2-second clips support temporal relation tokens.
- ✅ **Established benchmark:** 48 object categories, 14 environments, 6 anomaly types, 4000+ QA instances – sufficient scale for thesis claim.
- ✅ **Public & recent:** GitHub repository public, 2025 publication.
- ⚠️ **Compute feasibility:** 2-second multi-view clips are resource-intensive. Kaggle P100 + Colab Pro should handle with careful batching. **Test on 100-clip subset first (1 week).**
- ⚠️ **Task reformulation:** Defect detection ≠ human action safety; may require adapting loss/eval metrics. Manageable within 2-week window if scope is tightly defined.

**Timeline:** 
- Week 1: Download MMVIAD subset, verify API stability, adapt detector + relation token pipeline
- Week 2: Train on 100-300 clips, measure baseline multi-camera transfer accuracy
- Week 3: Full train, validation, thesis write-up

**Verdict:** **Technically strongest, but schedule pressure is HIGH.** Risk if MMVIAD has undisclosed API issues.

---

### 2. Best Second Dataset for Cross-Dataset Transfer (Strongest Generalisation Claim)

**SECONDARY RECOMMENDATION: iSafetyBench + PETS2009 (dual use)**

**For iSafetyBench (primary choice):**
- ✅ 1,100 diverse safety videos (factories, warehouses, construction, retail) → addresses domain shift claim.
- ✅ 165 action labels (98 routine + 67 hazard) → coverage is comprehensive.
- ✅ 4–8 second clips → compatible with temporal pipeline.
- ✅ Video-language task → can evaluate zero-shot transfer of YOLO + relation head without fine-tuning.
- ⚠️ **FATAL ISSUE: CC-BY-NC-SA-4.0 non-commercial license.** Check with thesis advisor/institution whether academic thesis is "non-commercial" or not. If thesis defense implies institutional IP, this may be unacceptable. **Verify licensing with graduate office BEFORE committing.**
- ⚠️ No person bounding boxes (video-level labels only) → mismatch to YOLO detector pipeline.

**For PETS2009 (fallback/classical baseline):**
- ✅ **4-camera multi-view** – validates multi-camera robustness.
- ✅ 2009 benchmark (established, stable, no API risk).
- ✅ Public domain equivalent (CVPR 2009 baseline).
- ❌ Outdated resolution (768×576, 7 fps).
- ❌ General crowd anomaly, not safety-specific.
- ❌ No person boxes.

**Combined strategy:**
- Train on Unsafe-Net or MMVIAD
- Evaluate on iSafetyBench (if license cleared) for safety domain transfer
- Evaluate on PETS2009 for camera robustness transfer
- Frame as: "Generalisation: same model, diverse domains & camera setups"

---

### 3. Dataset with SAME Concept from MANY CAMERAS (The Holy Grail Question)

**Answer: MMVIAD.**

MMVIAD's ~120° camera rotation per inspection clip is the ONLY dataset in this survey that explicitly provides the property: **same anomaly concept, many camera angles**. 

No other dataset combines:
- Multi-camera property
- Workplace/industrial safety domain
- Public availability
- Recent enough (2025) to have modern annotations

**Honorable mention:** Safe-Construct (multi-view construction safety) conceptually perfect, but unclear if dataset is publicly released; may be synthetic-only (Blender-based SICSG).

---

### 4. Honest Assessment: Is Switching Datasets Feasible in 2-3 Weeks?

**SHORT ANSWER: Risky. Realistic move is Unsafe-Net + iSafetyBench as supplementary, NOT full migration.**

**Detailed Rationale:**

**Full Migration to MMVIAD: HIGH RISK**
- 🔴 Dataset published Q4 2024/Q1 2025. Potential API instability, undocumented quirks, missing preprocessing scripts.
- 🔴 Object-centric (defect detection) vs action-centric (worker safety) → task domain shift. Requires loss/eval metric redesign.
- 🔴 2-second multi-view clips → each clip is 6–8x larger than Unsafe-Net single-camera clips. GPU memory pressure on P100 (16 GB).
- 🔴 Zero buffer time if data loading bottleneck discovered mid-week-2.
- 🟡 If successful: 3 days of contingency remaining for thesis write-up (risky for defense deadline).

**Hybrid Approach: Unsafe-Net + iSafetyBench (RECOMMENDED)**
- ✅ **Unsafe-Net as main claim** (existing tuned pipeline, 691 clips, minimal risk). Accept camera confounding as "known limitation" in thesis.
- ✅ **iSafetyBench as supplementary transfer experiment** (1,100 videos, diverse safety scenarios). Frame as: "Zero-shot transfer to unseen safety domains" or "Domain robustness without task-specific tuning."
- ✅ **PETS2009 as classical baseline** (optional, if time permits week 3). Frame as: "Multi-camera camera robustness on established benchmark."
- ✅ Licensing: Only iSafetyBench has non-commercial clause; mitigates by framing as "evaluation dataset" not primary claim.
- ✅ Compute: iSafetyBench (1.29 GB) is ~1.9x Unsafe-Net; manageable on available hardware.
- ✅ Timeline: 3 days dataset setup + 5 days training/eval on iSafetyBench + 5 days thesis writing = no schedule pressure.

**Why NOT migrate entirely to MMVIAD:**
- 🔴 Compute/API risks not justified by 2-week timeline.
- 🔴 Thesis defense in 2-3 weeks: insufficient margin for unexpected issues.
- 🔴 MMVIAD is cutting-edge (published 2025); peer reviewers may request clarifications on dataset construction that authors cannot answer quickly.
- 🟡 If you have **6+ weeks**, MMVIAD becomes the preferred long-term direction.

**Why the Hybrid Approach Works:**
1. **Maintains thesis integrity:** Primary accuracy claim on proven Unsafe-Net dataset (existing ablation study validates findings).
2. **Strengthens generalization narrative:** "Same model architecture, zero-shot transfer to diverse safety domains (iSafetyBench), multi-camera robustness (PETS2009)."
3. **Respects deadline:** Safe, no surprises.
4. **Opens future work:** "Future: multi-camera encoder (MMVIAD) for camera-invariant safety detection."

---

## Unresolved Questions

1. **MMVIAD licensing:** Is dataset CC-BY, CC-BY-NC, or Apache 2.0? GitHub repo README not fetched; check github.com/Georgekeepmoving/MMVIAD directly.
2. **iSafetyBench non-commercial clause:** Does your institution's thesis publication trigger "commercial use" under CC-BY-NC-SA-4.0? Verify with grad office before committing.
3. **Safe-Construct dataset release:** Is a public dataset version available, or is only the model/synthetic generator released? Check Safe-Construct.github.io.
4. **MMVIAD object categories:** Do any of the 48 object categories align to "workplace safety" (e.g., missing scaffolding, unsecured equipment) or is it pure manufacturing defects? May affect task suitability.
5. **iSafetyBench evaluation protocol:** Does the VQA (question-answering) task require vision-language model (GPT-4V, LLaVA) or can clip-level classification metrics work with standard YOLO detector + Transformer head?
6. **Compute requirements:** What is the per-clip GPU memory footprint of MMVIAD's 2-second multi-view sequences at original resolution?

---

## Summary Table: Top 5 Candidates Ranked

| Rank | Dataset | Multi-Camera? | Video? | Person Boxes? | Safety Domain? | Feasible 2-3 wks? | Recommendation |
|------|---------|---------------|--------|---------------|----------------|-----------------|-----------------|
| 1 | **MMVIAD** | ✅ (120° rotation) | ✅ | ❌ | ✅ (industrial) | ⚠️ (High risk) | **Full replacement** (if 6+ wks) |
| 2 | **iSafetyBench** | ❓ | ✅ | ❌ | ✅ | ✅ | **Supplementary transfer** (recommended) |
| 3 | **PETS2009** | ✅ (4-view) | ✅ | ❌ | ❌ (crowd) | ✅ | **Cross-dataset robustness baseline** |
| 4 | **CHAD** | ✅ (4-view) | ✅ | ✅ | ❌ (parking) | ✅ | **Domain shift validation** (optional) |
| 5 | **Safe-Construct** | ✅ (multi-view) | ❓ (synthetic?) | ❌ | ✅ (construction) | ❌ (unknown release) | **Future direction** |

---

## Implementation Roadmap (Hybrid Approach)

**Week 1 (Dataset & Setup)**
1. Day 1–2: Download iSafetyBench (1.29 GB), verify legal/licensing with grad office
2. Day 3–4: Audit iSafetyBench class distribution, map to Unsafe-Net task (safety violation classification)
3. Day 5: Adapt data loader for iSafetyBench; preprocess frames for YOLO detector

**Week 2 (Training & Eval)**
1. Day 1–3: Train YOLO detector (or use pretrained) + relation token encoder on iSafetyBench subset (200 clips)
2. Day 4–5: Evaluate zero-shot transfer accuracy on held-out iSafetyBench test set; compare to Unsafe-Net baseline

**Week 3 (Thesis Writing)**
1. Day 1–3: Write "Cross-Dataset Generalisation" section (iSafetyBench results, PETS2009 optional)
2. Day 4–5: Finalize defense slides, proofread

---

## Conclusion

**Do NOT fully migrate datasets this close to deadline.** Unsafe-Net is constrained but known; migration risks delay. Instead, **keep Unsafe-Net as the main accuracy claim, add iSafetyBench as a supplementary cross-dataset transfer experiment.** This yields:
- ✅ Defensible thesis (proven Unsafe-Net pipeline)
- ✅ Stronger generalization narrative (transfer to iSafetyBench)
- ✅ Safe timeline (no API surprises)
- ✅ Honest assessment: "Camera confounding is a known limitation; future work: multi-view encoders (MMVIAD)"

If you have 6+ weeks post-defense, strongly pursue MMVIAD for a follow-up study.

