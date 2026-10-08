# Phase 04 — binary relation-head ONNX export + parity (2026-09-18)

Scope: export both E3/E0 binary graphs, verify export-pipeline parity, check TRT-compat metadata,
write MANIFEST.md, update runbook placeholder. Zero test touches (numerical parity only, not accuracy).

## 1. Checkpoint search — none found, exported as random-init placeholders (disclosed)

Searched `sweep_results/relation_tokens/*bin8_binary*` and `run_kfold_relation.py`/
`run_relation_transformer.py` save paths. Finding: k-fold-691/LCO-691 screening (which produced the
freeze decision) only ever `torch.save`s pooled OOF **softmax**
(`{kfold,lco}_oof_softmax_bin8_binary_{ARCH}_drop_conflict.pt`) — never a `state_dict`. The only code
path that saves binary weights is `run_relation_transformer.py --binary --save_model`
(`main_binary()`), which scores **every seed against the 125-clip test set inside the training loop**
— that would spend one of the two test touches pre-registered in `reports/test-touch-log.md`
(2026-09-17 G0), reserved for (1) E0 retrain and (2) the pre-registered rule's winner. Not this lane's
budget to spend, and my task explicitly says "zero test touches."

**Decision: exported both graphs with randomly-initialised weights** (`torch.manual_seed(0)`),
disclosed loudly in code docstring, MANIFEST.md, and the runbook. Per the task's own instruction, this
is legitimate for graph-shape/latency/TensorRT-buildability validation — never for an accuracy claim.

## 2. Exports

New file `deployment/scripts/export-p3-relation-head-binary.py` (190 lines, own file — kept
`export-p3-graphs.py` at 221 lines untouched, not required to fix it for this task):

| graph | arch | params | file | sha256 (short) |
|---|---|---|---|---|
| `p3-relation-head-bin-e3-fp32.onnx` | E3DeepSets (recommended freeze) | 18,044 | 103,606 B | `92888b16...` |
| `p3-relation-head-bin-e0-fp32.onnx` | RelationTransformer (fallback) | 298,892 | 1,415,344 B | `c021f14c...` |

Both: opset 18, inputs `tokens (batch,16,8,74)` + `mask (batch,16,8)`, single output `logits
(batch,2)`. **Only the main cls_logits head exported** — sub-class (8-way) and per-frame aux heads are
training-time-only losses (design §3.2), never called by any inference script (checked
`jetson-multi-stream-pipeline.py`, `trt-runner-utils.py`) — correctly excluded from the deployed graph.
4-class legacy `p3-relation-head-fp32.onnx` untouched, not deleted, per constraint.

## 3. Parity (task 2 — numerical only, no accuracy)

PyTorch vs onnxruntime CPU, all 125 clips of `tokens_bin8_test.pt`, batch=1 per clip (see §4 for why):

| graph | max abs logit diff | decision agreement (torch vs onnx) |
|---|---|---|
| E3 | 9.686e-08 | 100.0% (125/125) |
| E0 | 2.310e-07 | 100.0% (125/125) |

Both far under the 1e-4 tolerance. This validates the **export pipeline** is lossless, not that the
model is accurate — logits are from random init. No label was read or compared; "agreement" = torch
vs onnx argmax agreement, never vs ground truth.

## 4. Dynamic batch / TRT compat (task 3)

- E3: genuinely dynamic-batch-capable — confirmed with a direct batch=4 onnxruntime call (`(4,2)`
  output, no error). No `nn.TransformerEncoder`, so nothing bakes the batch dim.
- E0: input declares symbolic `batch` on both tensors, **but the graph is batch=1-only in practice** —
  a batch=4 (also tried batch=5) onnxruntime call fails at a `Reshape` node
  (`gemm_input_reshape`, "Input shape:{129,4,128}, requested shape:{129,128}"). Root-caused to
  `RelationTransformer.forward()`'s `x.view(B, T*N, -1)` feeding `TransformerEncoder`: torch 2.11's
  dynamo-based `torch.onnx.export` bakes `B` to a concrete `1` here despite `dynamic_axes` being
  passed (a known dynamo/dynamic_axes interaction gap — the exporter itself warns "dynamic_axes is not
  recommended when dynamo=True"). **Confirmed this is pre-existing, not introduced by this task**: the
  already-deployed 4-class `p3-relation-head-fp32.onnx` (predates this session) has the byte-identical
  failure at batch=5.
- Read `jetson-build-trt-engine.py::build_engine()`: its optimization-profile step is name/shape-driven
  (iterates `network.num_inputs`, sets `min=opt=max=1` for any dim `<0`), not arch-aware — it will
  treat both new graphs exactly like the already-working legacy graph. Since deployment is always
  batch=1 (`DEFAULT_BATCH=1`), E0's batch-lock is harmless in practice, but it means E0 could never be
  used for a multi-clip-batch TRT engine even if someone wanted one later; E3 could.
- Read `trt-runner-utils.py::TRTRunner`: resolves dynamic dims via
  `engine.get_tensor_profile_shape(name,0)[1]` before allocating buffers, binds every tensor by
  **name** (the 2026-09-16 multi-input fix) — both graphs use the same `tokens`/`mask` names as the
  already-verified legacy graph, so this should work unchanged.
- **Not verified** (no TensorRT/CUDA on this Mac): actual engine build, actual `TRTRunner` execution.
  Stated plainly in MANIFEST.md and the runbook.

## 5. MANIFEST.md (task 4)

Written at `deployment/export/MANIFEST.md` (96 lines) — scoped to the P3 pipeline (P1
backbone/pair/scenario graphs already tracked in `deployment/artifacts/trt-compat-report.md`). Covers
all 5 P3 graphs (detector, appearance, legacy 4-class, bin-e3, bin-e0): arch, params, checkpoint
provenance (explicit "NONE — random init" for both binary heads), i/o names+shapes, parity numbers,
checksums, and an explicit "what is NOT verified" section (TRT build, TRTRunner execution, batch>1 on
real hardware, any accuracy claim).

## 6. Runbook update (task 5)

`deployment/JETSON-RUNBOOK-2026-09-19.md`: replaced the "18/9 prerequisite — placeholder" block with
landed status (checkboxes filled/unfilled per what's actually true), updated the top status paragraph,
Step 0 file-transfer list (now includes both binary graphs + MANIFEST.md), Step 2 build commands (both
`--model p3-relation-head-bin-e3`/`-e0` build calls, noted E0 optional/time-permitting), Step 5 parity
guidance (`--rule argmax` not `--rule threshold` for either binary graph — threshold has no meaning on
random logits), Step 6 `--class-labels` guidance (same caveat). No stale `p3-relation-head-bin-fp32.onnx`
(singular placeholder name) references remain (grepped clean).

## Constraints check

- File ownership: only touched `deployment/` (new script, MANIFEST.md, runbook) + this report. Did not
  touch `export-p3-graphs.py` or any model/training file.
- File sizes: `export-p3-relation-head-binary.py` 190 lines (<200). Did not need to split further.
- Naming: kebab-case.
- Did not commit.
- Ran the script twice (both before and after a cosmetic import cleanup) — identical checksums both
  times, confirming determinism.

## Unresolved questions

1. **Which arch should Step 2/5/6 of tomorrow's session actually spend device time on?** I defaulted
   the runbook to "E3 primary, E0 optional/time-permitting" per the freeze recommendation, but the
   user has not chosen — if E0 is picked instead, swap the emphasis, both graphs already exist.
2. **When should the real checkpoint get trained?** `export-p3-relation-head-binary.py --head_ckpt
   PATH --force` is ready to consume a `run_relation_transformer.py --binary --save_model` checkpoint
   the moment one of the two pre-registered test touches is spent on it — not scheduled by me, since
   that's a test-touch decision outside this lane's authority.
3. `benchmark-p3-onnx.py` (a different script, not touched) still hardcodes `p3-relation-head-fp32.onnx`
   (the legacy 4-class graph) as its cost-probe target — out of scope here, flagging in case whoever
   runs P3 cost-curve sweeps wants it pointed at a binary graph too.
