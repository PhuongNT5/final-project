# deployment/export/ — P3 pipeline ONNX manifest

Scope: the P3 (YOLO26 entity + appearance + relation-head) edge graphs only. P1
(backbone/pair/scenario/direct4) graphs are tracked separately in
`deployment/artifacts/trt-compat-report.md`.

Produced by: `deployment/scripts/export-p3-graphs.py` (detector, appearance, legacy 4-class head),
`deployment/scripts/train-binary-head-checkpoint.py` (new 2026-09-19, trains the binary checkpoints,
TRAIN split only, zero test touches — see report), and
`deployment/scripts/export-p3-relation-head-binary.py --head_ckpt ... --force` (BINARY Safe/Unsafe
heads; 2026-09-18 random-init placeholders, **re-exported 2026-09-19 from trained checkpoints**).
Git commit at export/train time: `43657eafc5c8264f9ab5232f458e511f65e7d555` (working tree had
unrelated uncommitted changes in other files — see `git status`; none touched `deployment/` or the
exported graphs' source files, `model_relation_head.py` / `model_binary_encoders.py`, which were
clean). Opset 18 throughout (torch 2.11's opset-17 downgrade pass fails on these graphs — see
export-p3-graphs.py docstring).

## G4 gate status: MET for the binary heads (2026-09-19, trained checkpoints)

The original phase-04 plan's G4 ("parity ≤1e-5 max-abs vs PyTorch on 50 val clips, trained
checkpoint") assumed a trained binary checkpoint would exist by 18/9. It did not on 18/9 (see
`plans/2026-09-16-binary-safe-unsafe-reframe/reports/laneb-2026-09-18-binary-onnx-export.md` for
that lane's random-init disclosure). **2026-09-19: both graphs now carry TRAINED weights** — trained
on the 566-clip `bin8_` TRAIN split (`--dedupe drop_conflict`), val-selected, **zero test touches**
(see `plans/2026-09-16-binary-safe-unsafe-reframe/reports/laneb-2026-09-19-binary-head-checkpoints.md`
for the exact code-path proof). Both the export pipeline (parity ~1-3e-6, decision-agreement 100% on
all 125 bin8 test clips, PyTorch vs onnxruntime, **numerics-only, no labels read**) and the weights
themselves are now real. **Test-set accuracy is still unmeasured** — only 2 test touches are
pre-registered (`reports/test-touch-log.md` 2026-09-17 G0) and neither has been spent on these
checkpoints; do not report test accuracy for either graph without spending one of those touches.

---

## 1. `p3-relation-head-bin-e3-fp32.onnx` — E3DeepSets, freeze-candidate (recommended)

| field | value |
|---|---|
| arch | `E3DeepSets` (`model_binary_encoders.py`) — masked-mean box pooling, no attention |
| params | 18,044 |
| checkpoint provenance | **TRAINED, 2026-09-19.** `deployment/scripts/train-binary-head-checkpoint.py --arch E3 --seeds 5 --epochs 60 --dedupe drop_conflict` on the 562-clip `bin8_` TRAIN split (566 minus 4 `drop_conflict` drops), internal (label,view)-stratified val carve (fit=460/val=102), val macro-accuracy selected, class weights on, label smoothing 0.05, sub-class(λ=0.3)+per-frame(λ=0.5) aux losses as implemented. **Zero test touches** — `tokens_bin8_test.pt` is never opened by the training script (see `plans/2026-09-16-binary-safe-unsafe-reframe/reports/laneb-2026-09-19-binary-head-checkpoints.md` for the code-path proof). 5 seeds; blind seed=**0**, val_macro=**89.15%**, val_top1=91.18% @epoch49 (of 60). Checkpoint: `sweep_results/relation_tokens/canonical/bin8_head_e3_checkpoint.pt` (sha256 `d1789e6ae4b44d77b56a36e341cd555487c08b4f9fa3eebe63863dde00b3177a`). Git commit at train time: `43657eafc5c8264f9ab5232f458e511f65e7d555`. |
| input | `tokens` float32 `(batch,16,8,74)`, `mask` float32 `(batch,16,8)` |
| output | `logits` float32 `(batch,2)` — index 0=Safe, 1=Unsafe (`NAMES_BIN` convention, repo-wide) |
| deployed decision path exported | cls_logits only (default `model(tok,mask)` call, `return_sub=False, return_frame=False`) — the 8-way sub-class head and per-frame aux head are training-time-only auxiliary losses (design §3.2), never called by any inference/runner script, correctly excluded |
| dynamic batch | genuinely dynamic — `tokens`/`mask`/`logits` all declare symbolic dim `batch`; verified with a batch=4 onnxruntime call (output shape `(4,2)`, no error). No `nn.TransformerEncoder` in this arch, so the dynamo exporter does not bake the batch dim. |
| parity (PyTorch vs onnxruntime, batch=1 per clip, all 125 bin8 test clips, **numerics only — labels never read**) | max abs logit diff **1.431e-06**, decision agreement **100.0%** (125/125) |
| checksum (sha256) | `6dbfd47a1701e2e1e4304985015e8db9b93e8ffb86332012269fbb0b589d3477` |
| size | 103,606 bytes |

## 2. `p3-relation-head-bin-e0-fp32.onnx` — RelationTransformer, statistical-tie fallback

| field | value |
|---|---|
| arch | `RelationTransformer` (`model_relation_head.py`), used via `n_cls=2, n_sub=8` — same class as the legacy 4-class head |
| params | 298,892 |
| checkpoint provenance | **TRAINED, 2026-09-19.** Same recipe as E3, `--arch E0`. **Zero test touches**, same code-path proof. 5 seeds; blind seed=**2**, val_macro=**90.92%**, val_top1=91.18% @epoch42 (of 60) — per-seed val_macro range 87.2-90.9% across the 5 seeds. Checkpoint: `sweep_results/relation_tokens/canonical/bin8_head_e0_checkpoint.pt` (sha256 `5ad24d40397aed9b5935126d877b84e9c209e45b9e8f48f05a96d26f8725640d`). Git commit at train time: `43657eafc5c8264f9ab5232f458e511f65e7d555`. |
| input | `tokens` float32 `(batch,16,8,74)`, `mask` float32 `(batch,16,8)` |
| output | `logits` float32 `(batch,2)` (nominally; see dynamic-batch caveat below — only `batch=1` is verified end-to-end) |
| deployed decision path exported | cls_logits only, same reasoning as E3 |
| dynamic batch | **input dims declare symbolic `batch`, but the graph is NOT genuinely dynamic-batch-capable**: a batch=4 onnxruntime call fails (`Reshape node 'gemm_input_reshape' ... Input shape:{129,4,128}, requested shape:{129,128}`). Root cause: `RelationTransformer.forward()`'s `x.view(B, T*N, -1)` feeding `nn.TransformerEncoder` gets its batch dim baked to a concrete `1` somewhere in torch 2.11's dynamo-based `torch.onnx.export` trace, despite `dynamic_axes={'tokens': {0:'batch'}, ...}` being passed. **This is not a defect introduced by this task** — the pre-existing `p3-relation-head-fp32.onnx` (4-class, exported by `export-p3-graphs.py`, predates this session) has the identical failure at batch=5, confirmed by re-running it here (still reproduces on the retrained graph). Harmless for actual deployment: `jetson-build-trt-engine.py`'s optimization profile always sets `min=opt=max=1` for every dynamic dim (`DEFAULT_BATCH=1`), i.e. TensorRT never builds nor calls this graph at batch>1. |
| parity (PyTorch vs onnxruntime, batch=1 per clip, all 125 bin8 test clips, **numerics only — labels never read**) | max abs logit diff **2.623e-06**, decision agreement **100.0%** (125/125) |
| checksum (sha256) | `307884c01fe2d650e03e6b362274b8014e4b81e73e013f30e04ce77cdfeb2956` |
| size | 1,427,196 bytes |

## 3. Legacy / unchanged graphs (present, untouched by this task)

| file | arch | i/o | checksum (sha256, first 16) | notes |
|---|---|---|---|---|
| `p3-relation-head-fp32.onnx` | `RelationTransformer` (4-class) | tokens/mask -> `logits (1,4)` | `582733bfac4f340b` | **NOT deleted or overwritten** per task constraint. Same batch-baked-to-1 defect as E0 above (pre-existing, confirmed by this task, not caused by it). Trained checkpoint: `sweep_results/relation_tokens/canonical/p3_blind_relation_head.pt` (see `export-p3-graphs.py` `DEFAULT_HEAD`). |
| `p3-appearance-fp32.onnx` | MobileNetV3-Small + fixed random projection | crops -> `appearance (K,64)` | `f47f76a338846e3a` | unchanged this session |
| `p3-detector-fp32.onnx` (+ `-imgsz{416,512,640}` cost probes) | YOLO26 (ultralytics exporter) | — | `4781deec84b3ff7d` | unchanged this session |

## What is NOT verified (device time required)

- **TensorRT engine build.** Not run — no TensorRT/CUDA on this Mac. Both new graphs were checked by
  reading `deployment/scripts/jetson-build-trt-engine.py`'s `build_engine()`: it iterates
  `network.num_inputs`, and for any tensor whose parsed shape contains a `-1` dim (i.e. the ONNX
  `batch` symbolic dim on `tokens`/`mask` for both E3 and E0), it calls
  `profile.set_shape(name, (1,...), (1,...), (1,...))` and adds one optimization profile covering
  both named inputs — this is shape/name-driven, not arch-specific, so it should build both graphs
  identically to how it already builds the legacy 4-class head. **Not run on-device**, so "builds
  without a tactic-skip failure" is unverified.
- **`trt-runner-utils.py`'s `TRTRunner` binding both inputs by name.** Read, not executed (no `trt`
  module here). Its constructor resolves any dynamic input shape via
  `engine.get_tensor_profile_shape(name, 0)[1]` (the profile's opt-shape, `(1,16,8,74)` /
  `(1,16,8)` for `tokens`/`mask`) before allocating buffers, then binds every I/O tensor by name in a
  `{name: {host,dev,shape}}` dict (the multi-input fix committed 2026-09-16, see the module
  docstring). Since both new graphs use the same two input names (`tokens`, `mask`) as the already-
  verified legacy 4-class graph, and the same profile-resolution code path, this should work
  unchanged — but has only been exercised against the 4-class graph on real hardware, never these two.
- **Test-set accuracy of either binary graph.** Both now carry TRAINED weights (2026-09-19), but
  **no test touch has been spent on either checkpoint** — val_macro (89-91%, see provenance rows
  above) is a training-time selection metric, not a test-set accuracy claim. Whoever spends one of
  the two pre-registered test touches (`reports/test-touch-log.md` 2026-09-17 G0) should re-run
  `run_relation_transformer.py --binary --arch {E3,E0} --save_logits <path>` (or equivalent) to get
  a reportable test number; this checkpoint/export pair is unaffected by that and does not need
  re-exporting once that happens, unless the test-touch run retrains with different hyperparameters.
- **Batch>1 on real TensorRT** (as opposed to ORT CPU) for E3. ORT confirms the ONNX graph itself
  tolerates batch=4; TensorRT engine behavior at batch>1 was not built/tested (moot for this session
  since the builder only ever targets batch=1).

## Reproduce

```bash
# Train (TRAIN split only, zero test touches, ~10s for E3 / ~5.5min for E0 on M-series CPU, 5 seeds):
.venv/bin/python3 deployment/scripts/train-binary-head-checkpoint.py --arch E3 --seeds 5 --epochs 60 \
  --dedupe drop_conflict --out sweep_results/relation_tokens/canonical/bin8_head_e3_checkpoint.pt
.venv/bin/python3 deployment/scripts/train-binary-head-checkpoint.py --arch E0 --seeds 5 --epochs 60 \
  --dedupe drop_conflict --out sweep_results/relation_tokens/canonical/bin8_head_e0_checkpoint.pt

# Export (one invocation per checkpoint -- the other arch is skipped via ctor.arch mismatch):
.venv/bin/python3 deployment/scripts/export-p3-relation-head-binary.py --force \
  --head_ckpt sweep_results/relation_tokens/canonical/bin8_head_e3_checkpoint.pt
.venv/bin/python3 deployment/scripts/export-p3-relation-head-binary.py --force \
  --head_ckpt sweep_results/relation_tokens/canonical/bin8_head_e0_checkpoint.pt
```

---

## R48 lane — CLIP representation graphs (added 2026-09-29)

Different lane from the P3 graphs above: these serve the **iSafetyBench** binary task, not
Unsafe-Net, and the classifier in this lane has **no detector in its accuracy path**. Exported by
`deployment/scripts/export-b1-clip-temporal-head.py` and
`deployment/scripts/export-clip-vision-tower.py`. Opset 18.

### `r48-b1-clip-temporal-3seed-fp32.onnx`

| field | value |
|---|---|
| arch | B1 temporal head (`SeedEnsemble`: mean softmax over the 3 training seeds — the same ensemble the OOF numbers were computed with) |
| params | 603,270 total (201,090/seed) |
| inputs | `frames` (1, 16, 512) CLIP embeddings, `valid` (1, 16) frame mask |
| size | 2.56 MB |
| parity | max abs(torch − onnxruntime) = **1.19e-07**, PASS at TOL 1e-5; checked over all-valid, all-invalid and random validity patterns |
| accuracy | 84.2 % balanced accuracy, LSO-1100 OOF, 3-seed ensemble. **Zero test touches.** |
| tracked in git | yes |

### `r48-clip-vit-b32-vision-fp32.onnx`

| field | value |
|---|---|
| arch | frozen CLIP ViT-B/32 vision tower (the head's input producer) |
| params | 87,849,216 |
| size | 351.6 MB (+ external `.data`) |
| parity | max abs = **1.57e-05**, PASS at TOL 1e-4 (a 12-layer ViT in fp32 accumulates more than the 1e-5 a 1-layer head does) |
| tracked in git | **no — gitignored.** 351 MB exceeds GitHub's 100 MB hard per-file limit; a push carrying it is rejected outright. Rebuild with `export-clip-vision-tower.py`; weights come from the HF cache, not this repo. |

**Edge cost, and it is the point of this section.** `benchmark-r48-clip-pipeline.py` on host arm64
CPU (onnxruntime, 20 runs, 16 frames/clip): tower **375.0 ms**, head **0.3 ms**, end-to-end
**375.2 ms/clip** = 2.67 clips/s. The backbone is **99.9 %** of wall clock and costs **18.3×** the
MobileNetV3-small appearance graph it replaced. The head got cheaper and the backbone got ~35×
larger — report both or neither. **These are host numbers; no R48 graph has been measured on the
Jetson.**
