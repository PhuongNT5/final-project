# Codebase Summary

Map of entry points and what artifact each produces. See `system-architecture.md` first for the
P1/P2/P3 pipeline distinction — most confusion in this repo traces to not knowing which pipeline a
given script belongs to.

## Root `run_*.py` scripts

Root scripts split across P1 (frozen-COCO, many files, oldest) and P3 (relation-Transformer,
newest, current headline). P2 lives entirely in `unsafenet/`, invoked via `run_unsafenet_r2.py`.

### P3 — current accuracy headline

| Script | Produces |
|---|---|
| `build_relation_tokens.py` | `sweep_results/relation_tokens/tokens_{train,test}.pt` — per-clip entity+geometry token sequences from the entity detector |
| `run_relation_transformer.py` | Trains `RelationTransformer` (`model_relation_head.py`); blind protocol (val-selected seed touches test once); writes `logits_*.pt` (blind_pred + ens_softmax + labels) |
| `run_kfold_relation.py` | Stratified 5-fold + Nadeau-Bengio significance vs scene-only baseline, pooled 516 clips; writes `kfold_oof_softmax_*.pt` and `*_relation.log` |
| `run_lco_eval.py` | Leave-camera-out evaluation, pooled 516 clips grouped by camera view; 3 baselines (constant-prior floor, honest scene-appearance classifier, per-view-majority); writes `_zones_eval.log` |
| `run_full_eval.py` | Full multi-lens report (accuracy battery, safety, calibration, statistics) for a given `logits_*.pt`; writes `full_eval*.json` + confusion/reliability PNGs |
| `augment_relation_tokens.py`, `ensemble_relation.py` | Token augmentation and multi-seed ensembling helpers for the P3 line |

### P1 — frozen-COCO, currently deployed on Jetson

| Script | Produces |
|---|---|
| `run_safe_unsafe.py` | Binary safe/unsafe classifier sweep (5 archs x seeds) |
| `run_temporal_fusion.py` | Multi-temporal-stream ablation on frozen features |
| `run_scenario_cascade.py` | Scenario-aware cascade pipeline (pairwise + direct heads) |
| `run_pairwise_contrastive.py` | 8-class contrastive + cascade + direct ensemble — the 56/64 best-top-1 P1 result |
| `run_yoloworld_push.py` | YOLO-World open-vocabulary backbone variant |
| `run_finetune_e2e.py` | End-to-end fine-tuning (partial backbone unfreeze) |
| `run_cross_ensemble.py` | Cross-ensemble of COCO + YOLO-World backbones |
| `run_finetune_bbox.py` | YOLOv11s fine-tune on behaviour-class bboxes (V10 negative-result track) |
| `run_blind_eval.py`, `run_kappa_prepare.py`, `run_kappa_compute.py` | Blind-protocol evaluation and inter-rater-style kappa checks on P1 outputs |
| `run_ablation_*.py` (bootstrap, cascade, calibration, inference, retrain, significance) | P1 ablation suite: bootstrap CIs, cascade variants, calibration, significance tests |

### Detector training (feeds both P2 and P3)

| Script | Produces |
|---|---|
| `train_behaviour_detector.py` | Trains the paper's YOLO-v4-equivalent stage: a per-frame behaviour detector (nc=8) on `annotation_guide/dataset_behaviour_{tight,ctx}`. Defaults to `--device cpu --amp False` — MPS detection training does not converge in this stack (see Unresolved). |
| `run_train_entity_detector.py` | Trains the entity/zone detector consumed by P3's `build_relation_tokens.py` |

### Paper-baseline replication (G21 and G25 pre-registered scoring)

| Script | Produces |
|---|---|
| `run_convlstm_paper_baseline.py` | Scores the paper-protocol ConvLSTM baseline on two non-overlapping tasks: G21 (4-class unsafe subset, 64 test clips) and G25 (binary Safe/Unsafe, 125-clip frozen test set). Reads artifacts only, trains nothing. Writes `ket-qua-g21-g25.json` to plan folder. Derives test-policy bounds (`np_bounds()`) from actual test labels and duplicate-audit groups rather than hardcoding. |
| `run_g23_framegate_val.py` | Validation-only scorer for the frame-reduction mechanism question ([17]'s +10 pp claim). Answers whether frame reduction adds value over plain detector aggregation using the 83-clip val slice, with no test touch. Saves fused test posteriors without scoring them, allowing downstream tests to decide how to use them. |

### Other

| Script | Produces |
|---|---|
| `run_unsafenet_r2.py` | Drives the P2 pipeline (`unsafenet/` package) end to end |
| `run_representation_ladder.py` | Cross-round accuracy-progression comparison table |
| `run_duplicate_audit.py` | Detects byte-identical `logits_*.pt` artifacts mistakenly treated as independent runs (see w2-number-integrity-fixes report, Defect 4) |
| `run_operating_point.py` | Threshold sweep + safety-recall operating-point selection, standalone entry point (also wrapped by `eval_utils.suite`) |
| `run_export_perclip_csv.py`, `auto_annotate.py`, `extract_frames_for_annotation.py` | Annotation and export tooling |

## `unsafenet/` — P2 package (paper-faithful rewrite)

| Module | Role |
|---|---|
| `config.py` | Central config: class names, detector/zone paths, paper's reference numbers |
| `frame_cache.py` | T=32 frame extraction/caching (`cache_r2/`, gitignored) |
| `entities.py`, `geometry.py` | Entity detection + ORB/RANSAC zone-registration geometry |
| `behaviour_dataset.py`, `train_frame.py` | Per-frame behaviour-CNN dataset + training |
| `behaviour_clip.py` | Clip decision from the behaviour detector (the paper's detector -> temporal path): `det-mean`/`-max`/`-pair` aggregation plus `det-gateQ`, the frame-reduction step. `detector_gate_status()` stamps each member with its detector's peak mAP50 and its standing against gate G22 |
| `convlstm.py`, `temporal.py`, `train_temporal.py` | ConvLSTM / temporal head training (22 variants: mean, BiLSTM, Transformer). `convlstm.py` renormalizes the ensemble posterior by its row sums after slicing, so `--task 8` (an 8-way softmax read on 4 classes) still satisfies the row-stochasticity precondition of `metrics.report()`. |
| `aggregate.py`, `fusion.py` | Training-free aggregation members + bagged Caruana fusion |
| `streams.py`, `clip_stream.py`, `scene_confound.py` | Motion/appearance stream construction, scene-confound diagnostics |
| `metrics.py`, `metrics_stats.py` | P2's own metrics implementation — overlaps with `eval_utils/` (see Unresolved) |
| `evaluate.py`, `data_index.py` | Evaluation driver + dataset indexing |

## `eval_utils/` — the binding metric suite (§4 of the evaluation contract)

Single reusable entry point: **`evaluate_suite()`** in `eval_utils/suite.py`. Signature:
`evaluate_suite(y_true, y_prob, class_names, protocol_tag, n_seeds=1, y_pred=None,
cost_matrix=None, unsafe_classes=None, target_safety_recall=0.95, n_boot=5000, seed=0,
compare_to=None, selection_y_true=None, selection_y_prob=None, n_bins=15)` — returns one
JSON-serialisable dict covering multi-lens classification metrics, safety/cost metrics, operating
point, calibration, and the bootstrap/Wilson leg of the statistics section. Built on top of
existing code rather than reimplementing: calls `unsafenet.metrics.report()` for the core
classification battery and composes the modules below for everything else.

| Module | Role |
|---|---|
| `suite.py` | `evaluate_suite()` — the single entry point |
| `metrics.py` | Top-1, macro accuracy, per-class recall/FNR, confusion matrix |
| `bootstrap_ci.py` | 5,000-resample bootstrap, Wilson / BCa confidence intervals |
| `paired_tests.py` | McNemar exact, two-proportion z-test, Nadeau-Bengio corrected t-test (k-fold) |
| `calibration.py` | ECE (binned) + reliability-diagram data |
| `cost_matrix.py` | Cost-sensitive scoring (miss penalised more than false alarm) |
| `operating_point.py` | Confidence-threshold sweep + safety-recall-driven operating-point selection |
| `conformal.py`, `selective_risk.py` | Conformal prediction / AURC / risk-coverage curve |
| `power_analysis.py` | Required-n analysis for a target effect size |
| `report_schema.py` | Canonical record schema + markdown-table rendering |
| `timing.py` | Per-clip timing helpers (paper-anchor reproduction: 0.14s/clip) |
| `verify_against_csv.py` | Cross-checks computed metrics against a reference CSV |
| `tests/` | Unit tests (`test_calibration.py`, `test_classification_metrics.py`, `test_conformal.py`, `test_cost_and_operating_point.py`, `test_power_analysis.py`, `test_selective_risk.py`, `test_statistics.py`, `test_suite_integration.py`, `test_timing.py`, `test_edge_cases.py`, `test_real_artifacts.py`, driven by `run_all.py`) |

Out of scope for `eval_utils`: Jetson metrics (§4g of the contract) — `deployment/` owns those.

## `deployment/` — edge export, benchmark, dashboard

| Path | Role |
|---|---|
| `JETSON-DEPLOYMENT.md`, `JETSON_QUICKSTART.md` | Jetson setup + TensorRT build instructions (see `deployment-guide.md` for the summary and pipeline-attribution caveat) |
| `export/` | ONNX exports (FP32 + INT8) of the P1 backbone and Transformer heads |
| `scripts/export-backbone.py`, `export-transformer-heads.py` | ONNX export scripts |
| `scripts/quantize-int8.py`, `verify-onnx-parity.py` | INT8 quantization + FP32/INT8 drift check |
| `scripts/benchmark-harness.py` | M3 (CPU/CoreML) latency benchmark -> `artifacts/benchmarks.json` |
| `scripts/jetson-benchmark.py`, `jetson-build-trt-engine.py` | Jetson-side TensorRT engine build + latency/FPS benchmark -> `artifacts/benchmarks-jetson-{fp16,int8}.json` |
| `scripts/jetson-multi-stream-pipeline.py` | Multi-stream throughput + zero-drop verification -> `artifacts/multi-stream-jetson-fp16-realtime.json` |
| `scripts/jetson-power-sweep.sh` | Power/energy sweep -> `artifacts/power-sweep-summary.json` |
| `dashboard/app.py` | Streamlit live dashboard |
| `dashboard/jetson-live-monitor.py` | Camera -> edge -> alert loop; currently 2-tier thresholds, no debounce (see plan phase-03 for the N-of-M debounce addition) |
| `artifacts/` | All measured JSON/PNG outputs — see `deployment-guide.md` for which pipeline (P1) they belong to |

## `annotation_guide/` — manual bbox annotation

| Path | Role |
|---|---|
| `frame_manifest.csv` | 2,262 annotated frames, manifest row per frame |
| `dataset_behaviour_tight/`, `dataset_behaviour_ctx/` | Two box-context variants (literal paper boxes vs person-boxes widened) for `train_behaviour_detector.py`; 1,901 train / 343 val each |
| `zones/` | Per-camera-view zone polygons (walkway / restricted-zone boundaries), consumed by `build_relation_tokens.py` and `run_lco_eval.py` |
| `annotation-sop.md`, `ontology-multi-entity.md` | Annotation standard-operating-procedure and class ontology docs |
| `cvat-labels.json` | CVAT label-config for the annotation tool |

## `kaggle-detector-retrain/` — free-GPU detector retrain package

Self-contained package to retrain the multi-entity YOLO detector on Kaggle's free GPU tier.
`train-detector-gpu.py` runs a 2-stage COCO transfer (freeze -> finetune); `build-zone-labels.py`
converts zone polygons into YOLO-format boxes for the "emit zone classes" experimental arm;
`retrain-detector.ipynb` is the notebook used on Kaggle; `prepare-upload.sh` packages the local
dataset for upload. Motivated by a distinct detector (not the behaviour detector) having two
classes (`Safe_Walkway`, `Restricted_Zone`) with zero training labels by design.

## `sweep_results/` — experiment outputs

| Path | Role |
|---|---|
| `relation_tokens/` | P3 artifacts: `tokens_*.pt`, `logits_*.pt`, `kfold_oof_softmax_*.pt`, `full_eval*.json`, `*.log` |
| `entity_detector/` | Trained entity detector checkpoint (`detector_best.pt`) consumed by P3 |
| `zone_prior/` | `clip_view_map.json`, `clip_scene_map.json` — camera-view/scene grouping used by k-fold and LCO |
| `r2/` | P2 (`unsafenet/`) training outputs, incl. behaviour-detector `results.csv` (mAP curves) |
| `pairwise/` | P1's best ensemble checkpoints (`run_pairwise_contrastive.py` output) |
| `finetune_bbox/`, `ablations/`, `representation_ladder.json`, `scene_confound.json`, `tta_results.json` | P1 ablation and fine-tuning tracks |

## Other top-level directories

| Path | Role |
|---|---|
| `paper/` | Source paper text (`paper.txt`) — grep target for every paper-attributed claim |
| `figures/` | Generated thesis figures (confusion matrices, reliability diagrams, etc.) |
| `report/` | Vietnamese thesis chapter drafts (`chapter-01-introduction.md` through `chapter-07-...md`, `abstract.md`) |
| `colab/` | Colab/Kaggle notebooks and templates, parallel free-GPU path to `kaggle-detector-retrain/` |
| `tools/` | Utility scripts: token builders (`build_l1_tokens.py`, `build_l2_tokens.py`, `build_motion_tokens.py`), `build_scene_grouping.py` (classifier-independent camera grouping, fixes the old circular-KMeans leak), `jetson_benchmark.py`, frame extraction/annotation prep scripts |
| `data/` | Unsafe-Net video dataset (8 classes, 691 clips, gitignored per repo convention — verify with `git status` before assuming it's tracked) |

## Unresolved questions

1. `eval_utils/` and `unsafenet/metrics.py` + `unsafenet/metrics_stats.py` are two overlapping,
   partially-duplicated implementations of the same metric contract (noted as a DRY violation in
   `plans/2026-09-06-thesis-2week-final/plan.md`). `eval_utils.suite.evaluate_suite()` calls into
   `unsafenet.metrics.report()` for its classification battery, so the two are not fully
   independent, but the boundary between what lives in which module is not documented anywhere
   else. Consolidation is tracked as a Lane F task, not yet complete as of this writing.
2. `train_behaviour_detector.py` and detection-training on Apple's MPS backend: confirmed broken
   in this stack (torch 2.11.0 / ultralytics 8.4.27) — mAP50 collapses to ~0.0002-0.0144 vs a
   working CPU run reaching 0.94+. Defaulted to `--device cpu`, which is why detector training is
   CPU-bound and slow in this project; not something a doc fix can resolve, flagging for anyone
   extending detector training.
