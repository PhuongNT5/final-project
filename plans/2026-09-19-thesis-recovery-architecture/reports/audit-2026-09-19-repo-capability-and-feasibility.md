# Audit: repo capability + feasibility, 2026-09-19

Read-only. No training/edits run. Ground truth for planning. Numbers uncertain are flagged
`[UNVERIFIED]`. Two prior audits (`plans/2026-09-06-thesis-2week-final/reports/audit-2026-09-14-*.md`)
cover pre-binary-reframe state in depth and are cited, not re-derived, where still current.

## 1. Data inventory

**Raw video**: `data/{train,test}/<8 class folders>/*.mp4(+.mp4.npy)`. 1382 files, 12GB
`[measured, du -sh]`. Folder names: `0_safe_walkway_violation` … `7_safe_carrying` (README's
8-class taxonomy; folder-name "safe"/"unsafe" prefixes are historical, do not match the binary
Safe/Unsafe split — binary mapping is source-class 0-3→Unsafe, 4-7→Safe, per
`binary_gate_lib.py:32-34`). 691 total clips (566 train/125 test), matches paper split.

**Token caches** (`sweep_results/relation_tokens/`, 91 items):
- `tokens_bin8_{train,test}.pt` — binary-task canonical, 74-dim/token, consumed by
  `binary_gate_lib.py:load_bin8()` and all E0-E3 binary runs.
- `tokens_wf_bin8_{train,test}.pt` — whole-frame (no-detector) S2 baseline features,
  built by `build_wholeframe_features.py`, loaded by `binary_gate_lib.py:load_wf()`.
- `tokens_{yolo26app_,yoloapp_,yolo26L1_}{train,test}.pt` and `_drop_all` variants — legacy
  4-class-task tags (`yolo26app_`=YOLO26+appearance 74d headline, `yoloapp_`=YOLO11+appearance,
  `yolo26L1_`=T2 coord variant, 72d). Corrupted-eval variants:
  `tokens_yoloapp_{flip,noise,blur,jpeg,dark}_test.pt`.
- `canonical/` subdir: `p3_blind_relation_head.pt` (4-class blind checkpoint),
  `bin8_head_e{0,3}_checkpoint.pt` (binary trained checkpoints, 2026-09-19, see §6),
  `kfold_repeats10_*.log`, `lco_seeds5_*.log` (headline logs cited by report/ch07 and the
  2026-09-14 audit, verified byte-match there).
- `lco_cluster_sweep/` — `--cluster_seed` sweep artifacts (ARI check, ties to test-touch-log
  2026-09-07 entry).
- `tier2_binary_logs/`, `tier3_lco_logs/` (2026-09-17/18) — per-arch binary k-fold-691/LCO-691
  logs + `master.log` timing wrappers (see §2/§7).
- Many `logits_*.pt` / `kfold_oof_softmax_*.pt` / `lco_oof_softmax_*.pt` / `enc_E*_lco-691_*.json`
  — per-run OOF/logit dumps, provenance tracked in `reports/test-touch-log.md` (§9).

**Regeneration cost**: see §7. Token rebuild from raw video is the expensive step (detector +
crop-embedding pass over 691 clips); re-running the small-head training on existing tokens is
cheap (§2 timings).

**`cache_r2/`**: `frames/` 2.8GB, `entities/` 2.9MB, `raw_{train,test}_112px_t16.npy` (325MB/72MB) —
P2 (`unsafenet/`) rewrite's own frame/entity cache, separate pipeline from P3's relation tokens.
Not required for P3 (relation-transformer) work.

**`features/`**: does **not exist** on disk `[verified: ls features -> No such file or directory]`.
README's mention of "pre-extracted YOLOv11s features (T=16/32/48)" is stale or refers to
`run_ucf_crime_zeroshot.py`'s in-memory extraction (below) or `cache_r2/entities/` — no directory
literally named `features/` is present.

**`annotation_guide/`** (bbox annotation set): `frame_manifest.csv` (2,262 frames, per the
2026-09-14 audit's verified count — all 2,262 rows' `clip_path` are under `/train/`, **zero**
`/test/` rows, confirming detector training never saw test-partition frames). Contents:
`dataset/`, `dataset_behaviour/`, `dataset_behaviour_tight/`, `dataset_behaviour_ctx/` (each with
`images/{train,val}`, `labels/{train,val}`, `data.yaml` — YOLO-format detection sets, entity vs
behaviour-labelled variants), `zones/view_{0,1,3,4,5,6,na}.json` (no view_2/view_7 — the
2026-09-14 audit's "H2" finding: 56/516 clips, all class-3 CarryOverload, get an all-zero zone
geometry, a second independent detector-class leak), `annotated/` (11 CVAT export jobs as
`job_N_annotations_*_{coco,ultralytics}.zip` + `_extracted/`), `annotation-sop.md`,
`ontology-multi-entity.md`, `cvat-labels.json`. The **400-frame behaviour set** is
`dataset_behaviour{,_tight,_ctx}` — exact frame count not independently re-verified this session
`[UNVERIFIED — task prompt's own figure, not confirmed by a fresh count of labels/*/*.txt]`.

**External dataset wiring**: `run_ucf_crime_zeroshot.py` (211 lines) — F4 cross-dataset zero-shot
eval. Applies the **legacy 4-class** `direct4_s314` + scenario-cascade ensemble (loaded from
`sweep_results/pairwise/`) to UCF-Crime anomaly clips at `data/ucf_crime/Anomaly/` (path expected,
not confirmed present this session), extracting frozen YOLOv11s layer-22 features live
(`extract_features()`, lines 44-60ish) — T=32, 16 grid cells, 512-dim, ImageNet-normalized 224²
crops. Output: `sweep_results/ablations/ucf_crime_zeroshot.{json,md}`. Reports prediction
distribution / mean confidence / UnauthorInt-routing fraction — **not an accuracy eval** (no
UCF-Crime labels map to the 4-class taxonomy), purely descriptive. This script targets the
old 4-class ensemble, not the binary E0-E3 heads — would need adaptation to reuse for the binary
task.

## 2. Model + training entry points

| Script | Trains | Protocol | Token tag | Key CLI flags | Timing (measured) |
|---|---|---|---|---|---|
| `run_relation_transformer.py` | `RelationTransformer` (4-class or `--binary`) | blind (val-macro-selected, `train_one()`) | any (`--tag`) | `--tag --binary --appearance --coord_variant --seeds --epochs --aux_w --dedupe` | not isolated this session; DEVICE hardcoded `'cpu'` (line ~30, comment: MPS Adam-update bug on macOS 14) |
| `run_kfold_relation.py` | same model, pooled | k-fold-691 / legacy-516 | via `--tag` | `--k --repeats --seeds --binary --dedupe` | k-fold-691, 5 archs (see below) = ~65 min total per task prompt `[not independently re-derived — no isolated per-arch kfold-691 timestamp file found this session, only combined E0 log below]` |
| `run_lco_eval.py` | `RelationTransformer`, pooled, legacy 4-class | LCO-691(-predecessor)/516 | via `--tag` | `--seeds --epochs --tag --aux_w --scene_map --dedupe` | not read this session (deferred — time-boxed); referenced by 2026-09-14 audit as the LCO-516 canonical-log producer |
| `run_lco_grl.py` | GRL/DANN camera-invariance variant | LCO | — | not read this session | not measured |
| `run_binary_baseline_gates.py` + `binary_gate_lib.py` | S1 (detector-rule logreg/tree) + S2 (scene-appearance logreg) — sklearn, no NN | k-fold-691 + LCO-691 | `tokens_bin8_*`, `tokens_wf_bin8_*` | `--k --repeats` | seconds (sklearn LogReg/tree on 687-691 rows × 8-64 dims) — cheap |
| `model_binary_encoders.py` | defines E1FSTE/E2FrameOnlyPos/E3DeepSets + `ARCHES`/`build_encoder()` | n/a (library) | n/a | n/a | n/a |
| `model_relation_head.py` | defines `RelationTransformer` | n/a | n/a | n/a | n/a |
| `build_relation_tokens.py` | n/a (token builder) | n/a | `--out_tag` | `--limit --device --subsets --appearance --appearance_backbone --corrupt --detector --classes --view_map` | see §7 |
| `build_wholeframe_features.py` | n/a (S2 input builder) | n/a | `tokens_wf_{tag}{split}.pt` | — | ~97-line script, cheap (MobileNetV3-small whole-frame, no detector) |
| `tools/build_scene_grouping.py` | n/a (LCO group builder) | n/a | `clip_scene_map{,_691}.json` | `--cluster_seed` | cheap (KMeans(8) on background descriptors) |
| `run_full_eval.py` | nothing (eval-only) | reads `logits_yolo26.pt` default | — | `--logits` | fast (eval_utils suite on precomputed logits) |
| `run_ablation_significance.py` | nothing (eval-only, F1/F2 stats) | reads `sweep_results/pairwise/features/all8_test.pt` | — | — | fast |
| `ensemble_relation.py` | nothing (50/50 ensemble of 2 logit files) | — | — | — | fast |

**Measured binary tier-2/tier-3 timings** (from `sweep_results/relation_tokens/tier2_binary_logs/`
and `tier3_lco_logs/master.log`, both dated 2026-09-17/18):
- **E0 kfold-691** (`tier2_binary_logs/E0.log`): 5-fold × 1 repeat on 687 pooled clips (`drop_conflict`),
  684K-param model wait — actually 298,892 params (n_num=73). Log has no per-epoch wall-clock lines
  (grepped, zero matches for time-keywords) — only `master.log` start marker exists for tier2
  (`[Thu Sep 17 22:38:36] START E0`, no matching DONE line captured this session — file may be
  truncated/still-appending or the DONE line wasn't in the tail read).
- **Tier-3 LCO-691, all 5 archs, wall-clock from `master.log` timestamps**:
  E0 23:53:24→00:27:03 = **33m39s**; E1_d64 00:27:03→00:39:00 = **11m57s**; E3 00:39:00→00:39:51 =
  **51s**; E1_d96 00:39:51→00:55:19 = **15m28s**; E2 00:55:19→01:28:47 = **33m28s**. **Total tier-3
  (5 archs) = 95m23s** on whatever device ran it (CPU, per repo-wide `DEVICE='cpu'` convention;
  not independently confirmed this run used CPU). This is a **measured number**, supersedes the task
  prompt's own "k-fold-691 5 archs = 65 min" figure — that 65-min figure is not reproduced by any
  log found this session; the 95m23s figure is for **LCO-691** (tier-3), not k-fold-691 (tier-2).
  Tier-2 (kfold-691) total wall-clock is **not found in any log this session** `[UNVERIFIED]`.
- E0's per-epoch/per-fold cost implied by params (298,892) vs E3 (18,044, `deployment/export/MANIFEST.md`)
  roughly tracks the task prompt's own E0>E1>E2>E3 cost ordering (E0 slowest, E3 fastest — E3 tier-3
  run took 51s total for 5-fold LCO vs E0's 33m39s, ~40x).

## 3. Token representation

`build_relation_tokens.py` (259 lines). Constants: `T=16` frames/clip, `N_MAX=8` box slots/frame,
`CONF=0.25` detector threshold, `IMGSZ=800`, `FEAT=10` base geometry dims, `APP_DIM=64` appearance
dims.

**Base 10-dim geometry token** (per box slot): `[class_id, xc, yc, w, h, conf, iou_walkway,
iou_restricted, foot_in_walkway, norm_dist]`. `class_id, xc, yc, w, h, conf` come straight from the
YOLO detector; `geometry()` (in-file) computes the last 4 from zone polygons
(`annotation_guide/zones/view_*.json`) — **person-classes only** get non-zero geometry features
(other classes e.g. Forklift get zeros here), which is the "zone-all-zero ⟺ CarryOverload" second
leak flagged by the 2026-09-14 audit (§1 above).

**+64-dim appearance** (flag `--appearance`, headline tag `yolo26app_`): crop embeddings via
`get_appearance()`/`embed_boxes()` — MobileNetV3-small (default) or a fine-tuned YOLO backbone
(`--appearance_backbone`), random-projected to 64 dims. Crops include ~30% padding around the box
(per prior review, not re-verified this session) — **this padding means appearance embeddings can
encode background/scene context, not just the entity**, a plausible secondary camera-leak channel
beyond raw xy (not row-level provable without a targeted ablation; flagged as suspicion, not fact).

**Total 74 dims** = 10 base + 64 appearance (73 numeric feature dims + 1 class-id categorical slot,
matches `model_relation_head.py`'s `n_num=73` for the appearance config).

**Padding/masking**: clips with <16 valid frames or <8 boxes/frame are zero-padded; a parallel
`mask` tensor (`(N,T,8)` bool/float) marks valid slots — consumed everywhere as `tok, mask` pairs
(`binary_gate_lib.py:s1_descriptor()` explicitly zeroes out padded slots via `-1` sentinel before
max-reduction).

**Camera/scene-identity leak, exact features**: **absolute `xc, yc`** (base-token dims 1-2) under
the default identity coordinate handling (T1, see below) directly encode camera framing/position —
the single most direct leak. Appearance-crop embeddings are a secondary, less certain leak channel
(background bleed via padding, see above). `class_id`/`conf` are NOT direct leaks per se, but
combined with the zone-derived geometry (which is keyed to a specific camera's zone polygon,
`view_map()` looks up per-clip camera id to select the right `zones/view_N.json`) they are
implicitly camera-conditioned even when scaled/relative.

**`apply_coord_variant` (`run_relation_transformer.py`, lines ~84-104)**:
- **T1** (default): identity — absolute xc/yc kept as-is. "Shortcut-prone" per its own docstring.
- **T2**: drops xc, yc columns entirely (74→72 numeric dims, matches tag `yolo26L1_`). Simplest
  camera-invariance control.
- **T3**: per-frame-relative coords — subtracts the per-frame mean xc/yc over currently-valid boxes
  in that frame, zeroed on padded slots. This is E1's default coordinate handling and the
  project's primary defence against the T1 shortcut while keeping xy's relative-geometry signal.

## 4. Detector status

**Checkpoints** (`sweep_results/entity_detector/`, `ls -la` this session):
- `best.pt` (19,182,810 bytes, mtime Aug 20) and `detector_best.pt` (byte-identical size,
  mtime Aug 9) — **same size, different mtimes**; exact provenance relationship (copy vs
  re-export vs re-train) not resolved this session; no docstring/README in that dir clarifies
  which is canonical `[UNVERIFIED]`.
- `yolo26_best.pt`/`yolo26_best.zip` (20,318,213 bytes, Aug 28) — separate, larger checkpoint,
  consistent with a YOLO26-architecture model (vs the 19.18MB YOLO11s-family checkpoints above).
- `yolo26n.zip` (5.5MB), `yolo26s.zip` (20.4MB) — likely pretrained YOLO26 nano/small weights
  (untrained-on-this-data baseline checkpoints), not confirmed by opening.
- `stage1_frozen/weights/{best,last}.pt` (19,154,010 bytes each, Aug 9) — a **3-epoch** run per
  `stage1_frozen/args.yaml` (`epochs: 3, batch: 4, imgsz: 640, device: cpu, model: yolo11s.pt`).
  `stage1_frozen/results.csv` (3 rows, one per epoch) shows **mAP50 climbing 0.308→0.435→0.462,
  mAP50-95 0.214→0.321→0.354 over just 3 epochs** — this is very likely a **smoke-test/sanity run**
  (`quick_check.log` name and 3-epoch budget both suggest this), NOT the final detector training
  run. No `results.csv` with more epochs was found anywhere in `sweep_results/entity_detector/`
  `[searched: only stage1_frozen/results.csv found repo-wide outside runs/detect/sweep_results/r2/*
  which are P2/behaviour-detector runs, not this entity detector]`. **The real/final mAP for
  whichever checkpoint (`best.pt`/`detector_best.pt`/`yolo26_best.pt`) is actually used in
  production token-building is not found in this repo as a clean results table** `[UNVERIFIED —
  candidate location: train_run.log (2.9MB) contains per-epoch progress bars but tail-grepping
  for "all"/"mAP50" summary rows returned only in-progress bar fragments, not a clean final-epoch
  summary row, in the slice read this session]`.
- `sweep_results/finetune_bbox/results.json` — NOT detector mAP; this is **downstream 4-class
  clip-classification accuracy** for multiple architectures/seeds (`direct4_s*`, `eight_s*`,
  `cascade_s*`, `pairwise_s*`), e.g. `direct4_s42: correct=52, macro=86.15`. Only one file in that
  directory (`find sweep_results/finetune_bbox -type f` returns exactly this one JSON, no weights).

**`kaggle-detector-retrain/`**: `retrain-detector.ipynb`, `train-detector-gpu.py`,
`train-behaviour-detector-gpu.py`, `build-behaviour-labels.py`, `build-zone-labels.py`,
`prepare-upload.sh`, `upload/unsafenet-detector.zip`, `README.md`. Notebook content not opened
this session (time-boxed) — **runnability as-is not verified**; the presence of a companion
`prepare-upload.sh` + pre-built `upload/unsafenet-detector.zip` suggests the notebook expects a
specific uploaded-dataset structure on Kaggle, which is a manual step, not push-button
`[UNVERIFIED, deferred]`.

**YOLO26 grep** (~35 hits across `.py`/`.md`/`.ipynb`): confirms YOLO26 work touches
`run_wholeframe_eval.py`, `run_lco_eval.py`, `run_lco_grl.py`, `tools/build_{l5,l1,l2,motion}_tokens.py`,
`tools/jetson_benchmark.py`, plus plan reports under `plans/2026-07-09-spatial-relation-yolo26-3month/`
and `plans/2026-09-16-binary-safe-unsafe-reframe/`. None of the plan-report content was opened this
session beyond the two 2026-09-14 audits already cited.

**Ultralytics/torch versions**: `ultralytics==8.4.27` (confirmed, `pip show`), `torch==2.11.0`,
`torchvision==0.26.0`, `onnx==1.21.0`, `onnxruntime==1.24.4`, `numpy==2.4.3`,
`opencv-python==4.13.0.92`. **YOLO26 architecture IS present in this ultralytics version**
(`.venv/lib/python3.13/site-packages/ultralytics/cfg/models/26/yolo26{,-cls,-obb,-p6,-p2,-pose,-seg}.yaml`
all found) — no version-compatibility gap for YOLO26 itself.

## 5. Eval + statistics stack

`eval_utils/` (14 modules, 1212 lines total): `suite.py` (122L, orchestrator), `bootstrap_ci.py`
(56L, BCa), `calibration.py` (69L, ECE/MCE/reliability bins), `conformal.py` (114L, split-conformal,
read in full — see below), `cost_matrix.py` (77L), `metrics.py` (53L), `operating_point.py` (92L),
`paired_tests.py` (195L, McNemar/Nadeau-Bengio/discordance-bound), `power_analysis.py` (69L),
`report_schema.py` (56L), `safety_metrics.py` (70L), `selective_risk.py` (93L),
`timing.py` (60L), `verify_against_csv.py` (76L).

**`eval_utils/conformal.py`** (read in full): split-conformal prediction, marginal coverage only
(Vovk/Gammerman/Shafer 2005). `nonconformity_score()` = 1-P(true label) (LAC score).
`conformal_quantile()` = ceil((n_cal+1)(1-α))/n_cal empirical quantile, clipped to 1.0 for small
n_cal. `split_conformal_repeated()` runs 1000 random cal/test splits at n=64 to report a
coverage/set-size **distribution**, not one noisy point estimate. `class_conditional_conformal()`
is **deliberately unimplemented** — raises `NotImplementedError` by design, because the smallest
class (n=8) can't support a meaningful per-class quantile (~11pp granularity) — an explicit
"honest refusal" rather than a fake per-class guarantee.

**`unsafenet/dedupe.py`** (164 lines, read in full): drop-list for 32 md5-identical-but-
contradictory-label clip groups (`run_duplicate_audit.py`/`run_duplicate_audit_binary.py` outputs).
Two task framings: `task='unsafe4'` (train policies `drop_all`/`keep_one`/`none`; test policies
`keep`/`drop_contradictory`) and `task='binary'` (train `none`/`drop_conflict`; test
`keep`/`drop_conflict`). Binary "opposite" groups = 4 (the ones cited throughout: `0_tr123, 1_tr55,
0_tr136, 0_tr138` train + `0_te23, 4_te4, 4_te7, 4_te9, 5_te13` test — note 5 test clips in the
opposite groups per this file's own comment, though the hard-ceiling discussion elsewhere cites
only the `0_te23`≡`4_te9` pair as the single in-test conflict; the other 3 named test clips are
presumably cross-split conflicts against dropped train members). Never deletes footage — pure
keep-mask filtering at load time, `policy='none'` always reproduces the raw set.

**`unsafenet/metrics.py`, `scene_confound.py`** not opened this session (time-boxed; contents
inferred from call sites — `metrics.py:report()` underlies `evaluate_suite()`, `balanced_accuracy`
confirmed to be a literal alias of `macro_accuracy` per `run_binary_baseline_gates.py:38` comment).

**`evaluate_suite()` return keys** (from `eval_utils/suite.py`, read in full, prior session):
`top1, macro_accuracy, macro_f1, balanced_accuracy, cohen_kappa, matthews_mcc, per_class{precision,
recall,f1,fnr_miss_rate,n}, confusion_matrix(+normalized), safety_cost, roc_auc_ovr, pr_auc_ovr,
threshold_sweep, operating_point_safety_recall, calibration{ece,mce,reliability_bins}, brier, nll`,
optional `comparison{mcnemar_exact, mcnemar_discordance_bound, two_proportion_z}`
(`two_proportion_z` explicitly kept only as a labeled-invalid diagnostic for same-64/125-clip
comparisons, per `review-2026-09-13-eval-statistics.md` finding #1 — fix status: **the 3 flagged
scripts still show as `M` (modified) in git status**, consistent with either the fix having landed
uncommitted or being still in progress; not diffed against the review's line numbers this session,
same open question the 2026-09-14 audit also left unresolved).

## 6. Deployment/demo assets

**`deployment/export/`** (du -sh, this session): P1 graphs (`backbone{,-ft}-fp32/int8`,
`direct4-s314`, `eight-s7`, `scenario-s7`, `pair{0-3}-s7`, fp32+int8 each, 33-46MB fp32 / 8.7-12MB
int8) unchanged since Apr 19 per prior audit. P3 graphs:
- `p3-detector-fp32{,-imgsz{640,512,416}}.onnx` (36MB each) — YOLO26 NMS-free detector, opset 18.
- `p3-appearance-fp32.onnx` (4.0MB) — MobileNetV3-small + fixed random projection.
- `p3-relation-head-fp32.onnx` (1.4MB) — **legacy 4-class** head, kept untouched per
  `deployment/export/MANIFEST.md`'s own stated constraint.
- `p3-relation-head-bin-e0-fp32.onnx` (1.4MB) and `p3-relation-head-bin-e3-fp32.onnx` (104KB) —
  **new binary heads, TRAINED weights as of 2026-09-19** (not random-init anymore — superseded a
  2026-09-18 random-init placeholder disclosed in
  `plans/2026-09-16-binary-safe-unsafe-reframe/reports/laneb-2026-09-18-binary-onnx-export.md`).

**Provenance + parity (from `MANIFEST.md`, read in full)**:
- **E3DeepSets** (18,044 params, recommended freeze-candidate): trained via
  `deployment/scripts/train-binary-head-checkpoint.py --arch E3 --seeds 5 --epochs 60
  --dedupe drop_conflict` on the 562-clip TRAIN split only (**zero test touches**, code-path
  proven in `laneb-2026-09-19-binary-head-checkpoints.md`). Blind seed=0, val_macro=89.15%.
  Parity PyTorch-vs-ONNXRuntime: max abs logit diff **1.431e-06**, decision agreement **100%**
  (125/125 bin8-test clips, numerics-only check, **labels never read** for this check). Genuinely
  dynamic-batch (verified batch=4 ORT call succeeds).
- **E0/RelationTransformer** (298,892 params, "statistical-tie fallback"): same training recipe,
  `--arch E0`. Blind seed=2, val_macro=90.92%. Parity: max abs diff **2.623e-06**, 100% decision
  agreement. **NOT genuinely dynamic-batch** — batch=4 ORT call fails
  (`Reshape node 'gemm_input_reshape'` shape mismatch), root cause: `nn.TransformerEncoder`'s
  batch dim gets baked during torch 2.11 dynamo export despite `dynamic_axes` being passed.
  **Same pre-existing defect** reproduces on the untouched legacy 4-class
  `p3-relation-head-fp32.onnx` — **not introduced by this task, confirmed still present**.
  **Harmless for deployment**: `jetson-build-trt-engine.py`'s optimization profile always sets
  `min=opt=max=1` for every dynamic dim, so TensorRT never invokes this graph above batch=1 anyway.
- **Test-set accuracy of both binary heads is UNMEASURED** — val_macro (89-91%) is a training-time
  selection metric only; 2 test touches are pre-registered (G0, §9) and neither has been spent on
  these specific checkpoints as of the manifest's writing.

**TRT multi-input defect — FIXED.** `deployment/scripts/trt-runner-utils.py`'s `TRTRunner` class
(read in full this session) now binds every I/O tensor into a `dict[name] -> {host,dev,shape}`
(lines 34-59), resolves dynamic dims via the profile's opt-shape before allocating buffers, and
`run()` accepts positional-or-keyword binding by name (line 61+). This supersedes the 2026-09-14
audit's finding of a single `host_in/dev_in` pair silently dropping the second input — **confirmed
fixed** (matches `MANIFEST.md`'s claim "multi-input fix committed 2026-09-16"). `jetson-benchmark.py`
line 35-38 comment explicitly documents this fix and its purpose (P3 relation head has 2 inputs).
Caveat: `jetson-benchmark.py`'s own `MODELS` list (line 32-33) is **still P1-only**
(`backbone, direct4-s314, scenario-s7, pair{0-3}-s7`) — the fix exists in the shared utility but
this particular script hasn't been pointed at P3 graphs; `jetson-build-trt-engine.py` and
`train-binary-head-checkpoint.py` DO reference `p3-*` graph names (grepped), so P3 engine-build
support exists somewhere in the newer scripts, consistent with `JETSON-RUNBOOK-2026-09-19.md`'s
§2 "Build P3 engines (new, first time on Jetson)".

**`deployment/JETSON-RUNBOOK-2026-09-19.md`** (24KB, not fully read — headers only): sections for
pre-flight, build P1 engines, **build P3 engines (new)**, model size, per-stage latency p50/p95/p99,
FP16/FP32 parity, multi-stream saturation (1/2/4/8), energy, 30-min thermal soak, memory,
power/clock snapshot, **fallback demo video capture (§11, "no existing recording — must capture
fresh")**, triage, minimum-viable-set fallback. This is a **runbook for a session that has not yet
happened** (dated today, 2026-09-19) — it describes what to measure, not measured results; no
Jetson-side artifacts dated 2026-09-19 were found this session.

**Live-defence-demo gap** (synthesizing the 2026-09-14 `audit-2026-09-14-edge-and-demo-readiness.md`,
re-confirmed still current for the parts checked this session): the demo-ready pipeline is **P1**
(`deployment/dashboard/app.py`, Streamlit, single/multi-video upload → sliding-window inference →
class+confidence, no live boxes-on-screen overlay — reads `deployment/scripts/pipeline-utils.py`).
`jetson-live-monitor.py` is the actual "video→Jetson→dashboard" live architecture but is wired
**only to P1's MODELS list**. **P3 (the accuracy-headline, YOLO26-based pipeline) has no on-screen
boxes+verdict demo path** — its own harness (`benchmark-p3-onnx.py`) is a latency benchmark, not a
visual demo. **No `.mp4`/fallback recording exists anywhere in the repo** (checked, zero hits) —
`JETSON-RUNBOOK-2026-09-19.md` §11 flags this as still-open ("must capture fresh"). For a live
"video in → boxes + verdict + confidence on screen" demo: P1 has the closest existing plumbing
(`app.py` shows class+confidence, but not literal bounding boxes drawn on frames — not confirmed
whether `pipeline-utils.py` draws boxes; not opened this session) `[UNVERIFIED]`; P3 would need new
integration work (detector boxes are computed but no script visualizes them + the relation-head
verdict together in one live view, as far as greped this session).

## 7. Compute feasibility table

| Task | Estimate | Basis | Confidence |
|---|---|---|---|
| (a) Full LCO-691, 5-seed, per arch | E0 33.7min, E1_d64 12.0min, E3 0.85min, E1_d96 15.5min, E2 33.5min (measured, `tier3_lco_logs/master.log`, this session) — **total 95.4min for all 5 archs** at whatever seed-count that run used | measured wall-clock deltas between START/DONE markers | high (direct log timestamps) — but seed-count per run not reconfirmed (assume 5-seed per pre-registration convention) |
| (b) YOLO detector retrain, Kaggle GPU | Not measured this session. `stage1_frozen` 3-epoch CPU run took ~365s total (`results.csv` epoch-3 cumulative `time` column) = ~122s/epoch on CPU, `imgsz=640, batch=4`. On a Kaggle T4/P100 GPU, per-epoch cost for YOLO11s/YOLO26 at this data volume (566 train clips × ~4 frames/clip ≈ 2262 annotated frames) is typically 10-30s/epoch GPU-side for this scale — **rough order-of-magnitude only, not measured on Kaggle in this repo** | extrapolated from `stage1_frozen/results.csv` + `args.yaml`, not a Kaggle log | low — `[UNVERIFIED]`, no Kaggle-side log found in repo |
| (c) Rebuild relation tokens from raw video after detector change | Not measured this session; `build_relation_tokens.py` runs YOLO inference (imgsz=800) + optional appearance-embedding pass over all 16 sampled frames × up to 8 boxes, for 691 clips. No wall-clock log for a full 691-clip build found this session (only per-training-run logs) `[UNVERIFIED — recommend a dry timing run on a small `--limit` before committing to a full estimate]` | — | low |
| (d) Rebuild whole-frame features | `build_wholeframe_features.py` is a lighter pipeline (MobileNetV3-small only, no detector, no NMS) — should be substantially cheaper than (c); no measured number found this session | — | low |
| (e) CLIP/VideoMAE feature extraction, all 691 clips, M4 | **No CLIP/VideoMAE extraction script found in this repo** (`grep -rli videomae/clip` across all `.py` returned zero hits outside the README's historical mention of a "V5"/"VideoMAE" experiment, README.md lines 155/158). The V5 (multi-scale frozen backbone, 53/64) and VideoMAE-base (55/64) rows in README's accuracy table are apparently from a P1-track experiment whose script is **not present/committed in this repo** — cost estimate not derivable from repo contents `[UNVERIFIED — treat as "needs a from-scratch build" if this axis is planned, not a "rerun existing script"]` | README.md:155,158 | none — script absent |

**Overnight vs minutes**: (a) LCO-691 5-arch full sweep is a **~95min job**, not overnight.
Individual small-head training (E3, ~1min for 5-fold LCO) is minutes. (b)/(c) are the only
candidates that plausibly need an overnight/multi-hour budget, but neither has a measured number
in this repo — both should be treated as **unknown-cost, budget a dry-run first** rather than
assumed-overnight or assumed-quick.

## 8. Blockers and defects

| Defect | Status as of 2026-09-19 | Source |
|---|---|---|
| TRT multi-input single-buffer bug (`TRTRunner`) | **FIXED** — dict-based per-name buffer binding confirmed in current `deployment/scripts/trt-runner-utils.py` | this session, direct code read |
| `benchmark-p3-onnx.py` hardcoding legacy 4-class graph | **Not directly re-checked this session against the binary heads** — `benchmark-p3-onnx.py`/`benchmark-p3-sweep.py` both still grep-match on `p3-relation-head` graph names generically; whether they've been pointed at the new `-bin-e{0,3}-` graphs or still default to the legacy 4-class one is `[UNVERIFIED]`, not opened this session | `deployment/scripts/benchmark-p3-onnx.py` (grep hit only) |
| ONNX batch-baking on E0 (RelationTransformer, torch 2.11 dynamo export) | **STILL PRESENT, confirmed on both the legacy 4-class graph and the new binary E0 graph** — architectural (nn.TransformerEncoder + dynamo), not a checkpoint issue. **Deemed harmless** because the TRT builder only ever targets batch=1. Not fixed, not blocking. | `deployment/export/MANIFEST.md` §2, this session |
| aggregate_fps counts dropped frames (multi-stream pipeline) | Confirmed present as of 2026-09-14 audit; **not re-checked this session** `[UNVERIFIED current status]` | `plans/2026-09-06-thesis-2week-final/reports/audit-2026-09-14-edge-and-demo-readiness.md` §2 |
| `benchmark-p3-onnx.py` excludes non-graph stages (NMS/crop/decode), uses mean crop count (58) not p95/cap | Confirmed present as of 2026-09-14; **not re-checked this session** | same source |
| two_proportion_z misuse (3 scripts) | **Unresolved as of both the 2026-09-14 audit and this session** — files still show `M` in git status, not diffed against the specific review line numbers either time | `run_full_eval.py`, `ensemble_relation.py`, `run_ablation_significance.py` (all reference "review-2026-09-13-eval-statistics.md finding #1" in comments) |
| B1: "8-class ~78% top-1" ch01/ch06 citation with no findable source | **Unresolved as of 2026-09-14; not re-investigated this session** (out of this audit's declared scope) | `audit-2026-09-14-experiment-inventory-and-feasibility.md` §3 |
| Detector-leakage in pooled k-fold-691/LCO-691 (train-partition clips seen by detector training; test-partition clips not) | Structural, **no leakage-free eval code exists**; unresolved | `frame_manifest.csv` (2262 rows, 100% train-partition), `audit-2026-09-14-experiment-inventory-and-feasibility.md` candidate (d) |
| CarryOverload / zone-all-zero detector-class shortcuts (100% recall via trivial rule) | Structural, disclosed everywhere it's reported, not "fixed" (it's a property of the data/labels, not a code bug) — binary reframe's per-sub-class recall tables (§ tier2 E0 log) still show CarryOverload recall 100.0% | `sweep_results/relation_tokens/tier2_binary_logs/E0.log` line 16 |
| No fallback demo recording | **Still open as of `JETSON-RUNBOOK-2026-09-19.md` §11** ("no existing recording — must capture fresh") | `deployment/JETSON-RUNBOOK-2026-09-19.md` |
| No live boxes+verdict demo path for P3 | Open, not scheduled to close per 2026-09-14 audit's recommendation (present P1 as deployed, P3 as accuracy-only study) | `audit-2026-09-14-edge-and-demo-readiness.md` §4 |

## 9. Test-touch ledger

`reports/test-touch-log.md` (266 lines, read in full). Log started 2026-09-06; entries before that
date are **retroactive reconstructions**, explicitly disclosed as such, not pre-registrations.

**Section A (retroactive)**: 6 logit artifacts reconstructed after the fact —
`logits_v1nocw.pt` (51/49 split — likely a typo/label-count artifact, reproduced as read),
`logits_v2bnocw.pt` (52/52), `logits_s1a.pt` (50/54, duplicate of `logits_yolo11_baseline.pt`),
`logits_yolo11_baseline.pt` (50/54), `logits_yoloroi.pt` (55/54 — higher than headline,
**unverifiable, not promoted**), `logits_yolo26.pt` (54/52, **current P3 4-class headline**).

**Section B (pre-registered, counted as real test touches)**:
1. 2026-09-06 provenance-recovery run (bit-exact reproduction of the P3 headline command).
2. 2026-09-06 (later) weight persistence + ONNX parity check (bit-exact, parity 2.980e-07).
3. 2026-09-07 LCO cluster_seed sweep integrity check (`--seeds 1 --epochs 1`, fold-shape only).
4. 2026-09-09 P3 edge latency measurement — **zero test touches** (dummy tensors + re-scoring
   already-logged softmax).
5. 2026-09-17 binary G0 pre-registration — declares exactly **2 test touches** for the entire
   binary task: (i) E0 retrained on the binary recipe, (ii) the winner of the pre-registered
   decision rule (E1 vs E0/S1/S2). **Neither has been spent as of `MANIFEST.md`'s writing**
   (2026-09-19) — the E0/E3 binary checkpoints trained 2026-09-19 for ONNX export were trained
   on **TRAIN split only, zero test touches**, per the manifest's explicit code-path proof.
6. 2026-09-17 (late) addendum — reports S1/S2 gate LCO-691 results (pooled k-fold/LCO OOF, not a
   test-set touch — train+test are pooled for these protocols by design, counted as "0 test
   touches" per the project's own convention, phase-06/2026-09-17 pre-registration).

**Exact count**: **2 pre-registered binary test touches declared, 0 spent** (as of the latest
dated artifact found, `deployment/export/MANIFEST.md`, 2026-09-19). For the legacy 4-class task,
Section B lists 3 pre-registered touches (entries 1-3 above; #4 explicitly consumed zero), all
**spent** (results already exist and are cited as the P3 4-class headline chain). No global numeric
"budget" (e.g. "10-15 touches total") is declared anywhere in `test-touch-log.md` itself — a figure
like "10-15 honest limit" appears only in a separate researcher agent's advisory report (not this
project's own pre-registration), so it should **not** be treated as this project's committed policy
without checking that report's authority `[UNVERIFIED — the ~10-15 figure is external advice, not
a repo-committed rule]`.

## Unresolved questions

1. `best.pt` vs `detector_best.pt` (`sweep_results/entity_detector/`) — same byte size, different
   mtimes (Aug 20 vs Aug 9); exact provenance/relationship not resolved.
2. Real/final detector mAP for the checkpoint actually used in production token-building —
   `stage1_frozen/results.csv` only covers a 3-epoch smoke test; no full-training results table
   found anywhere in the repo this session.
3. `kaggle-detector-retrain/retrain-detector.ipynb` runnability as-is — not opened this session.
4. Tier-2 (k-fold-691) total wall-clock across all 5 archs — no `master.log`-equivalent DONE
   timestamps found this session (only tier-3/LCO-691's master.log was complete); the task
   prompt's own "65 min total" figure is not reproduced by any log found.
5. Token-rebuild-from-raw-video wall-clock (deliverable 7c) and detector-retrain-on-Kaggle
   wall-clock (7b) — no measured numbers exist in-repo; both need a dry-run before planning
   overnight-vs-quick.
6. CLIP/VideoMAE extraction script — referenced only as a README results-table row (V5, VideoMAE),
   no script found in the current repo; unclear if it was ever committed, or run externally/deleted.
7. `benchmark-p3-onnx.py`/`benchmark-p3-sweep.py` — do they currently default to the legacy
   4-class `p3-relation-head-fp32.onnx` or has anyone pointed them at the new binary
   `-bin-e{0,3}-` graphs? Not opened this session.
8. `run_lco_eval.py`, `run_lco_grl.py` source — not read this session (deferred for time); GRL/DANN
   mechanism details (loss weighting, gradient-reversal schedule) are unconfirmed beyond the
   `return_repr` hook mentioned in `model_relation_head.py`'s `forward()` signature.
9. Whether the two_proportion_z fix (review-eval-statistics finding #1) has actually landed —
   still unresolved from the 2026-09-14 audit, not diffed against specific line numbers either time.
10. Exact 400-frame behaviour annotation count — not independently re-verified this session
    (only the 2,262-frame entity set's count was reconfirmed via `frame_manifest.csv`).
11. Does `deployment/dashboard/app.py`'s pipeline (`pipeline-utils.py`) draw bounding boxes on
    frames, or only report class+confidence? Determines exactly how close P1's existing demo is to
    the "boxes + verdict + confidence on screen" requirement.
12. The "10-15 test touch" budget figure — confirmed to originate from an external researcher
    advisory doc seen in this session's context, not from `reports/test-touch-log.md` itself; is
    there a separate repo-committed ceiling elsewhere (e.g. a brief.md or design doc) not searched
    this session?
