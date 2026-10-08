# Lane B — P3 edge cost accounting + ONNX export path

Status: **partial**. Export path built and parity-checked for stage A2; stage B blocked on the
weight regeneration now running; no Jetson measurement yet.

## 1. What P3 actually runs at the edge (this was not written down anywhere)

Chapter 5 benchmarks **P1** (frozen-COCO YOLOv11s backbone + pairwise-cascade heads). The accuracy
headline is **P3**. Their edge graphs are not the same shape, and the difference had never been
enumerated. Confirmed from `tokens_yolo26app_*.pt` `feat_layout`, not from the docstring:

```
feat_layout = [class, xc, yc, w, h, conf, iou_walkway, iou_restricted,
               foot_in_walkway, norm_dist] + app_0 .. app_63          # 74 dims
```

`build_relation_tokens.py`'s module docstring still advertises `feat[10]`, and
`model_relation_head.py` says "tokens (T, N_MAX, 10)". **Both describe a variant that is not the
headline.** The headline tag `yolo26app_` runs at feat=74; the ablation tag `yolo26L1_` at 72
(it drops `xc, yc`). So P3 has a third stage with no P1 analogue:

| stage | what | per decision |
|---|---|---|
| A1 | YOLO26 entity detector @ 800x800 | 16 frames |
| A2 | MobileNetV3-Small on context crops -> 576-d -> fixed random projection -> 64-d | **57.8 crops (mean)**, p95 106, max 128 |
| B  | RelationTransformer, 0.298 M params | 1 pass over (16,8,74) |

A2 crop counts are measured over all 516 clips from the stored `mask` tensors, not assumed:
mean 57.8 boxes/clip (3.66 boxes/frame), 45.7 % of the 16x8 slot cap filled, 0.3 % empty frames.

## 2. Cost accounting (FLOPs — deterministic, hardware-independent)

All figures **fused** (what actually runs at inference; ultralytics reports fused, and unfused
overstates yolo26 by ~10 %: 35.20 vs 32.09 GFLOPs @800).

| | GFLOPs/frame | per decision | GFLOPs/decision |
|---|---:|---:|---:|
| P1 backbone (yolo11s L0-22 @224) | 2.24 | 32 frames | 71.6 |
| P3 A1 detector (yolo26s @800)    | 32.09 | 16 frames | 513.4 |
| P3 A2 appearance (mnv3-small @224) | 0.1217 | 57.8 crops | 7.0 |
| P3 B  relation head | — | 1 | negligible (0.3 M params) |

**Two findings, both of which cut against my prior assumption:**

1. **The appearance branch is not the problem.** A2 is **1.4 %** of A1's arithmetic (7.0 vs 513.4),
   and even at the 128-crop worst case only 3.0 %. I had flagged A2 as the thing that "makes P3's
   edge cost materially heavier"; on arithmetic that is **wrong**. What A2 does add is a *per-box
   CPU preprocessing path* — `tf(Image.fromarray(crop_bgr[:, :, ::-1]))`, i.e. a numpy copy, a PIL
   construction, resize-256 + center-crop-224, tensor + normalise, **per box** — which FLOPs do not
   capture and which lands on the Jetson's weak CPU cores. That cost is unmeasured and is the real
   open question for A2, not the MobileNet forward.
2. **P3 costs ~7.3x P1 per decision** (520.5 vs 71.6 GFLOPs), and the driver is resolution:
   detecting at 800² is **14.3x** a 224² backbone frame, only partly offset by P3 needing 16 frames
   where P1 needs 32.

## 3. Latency: the projection cannot settle feasibility (so the Jetson run is required)

First, the unit. `benchmarks-jetson-fp16.json` reports e2e p50 8.62 ms -> **116 FPS**, computed as
`t_backbone(1 frame) + t_direct4(1 clip) + t_cascade(1 clip)`. That sums a **per-frame** and a
**per-clip** term, so it is only meaningful under a stated cadence: it is the *stride-1* figure (one
decision per incoming frame). `jetson-multi-stream-pipeline.py` uses a **different** cadence —
`T_WINDOW = 32, EMIT_EVERY = 8`, one decision per 8 frames. Neither number is wrong and 116 is the
conservative one, but the two harnesses answer different questions and the thesis never says which.
**Any P3-vs-P1 comparison must fix the cadence first**, or it compares units.

Fixing the unit as **ms per decision**, the measured P1 cost is
`32 x 2.76 + 1.41 + 4.44 = 94.2 ms` -> **10.6 decisions/s**. The requirement at 24 fps with
`EMIT_EVERY = 8` is **3.0 decisions/s**.

Projecting P3 requires an assumption about achieved throughput, and that assumption dominates the
answer. The measured P1 backbone achieves only **0.81 TFLOP/s** (2.24 GFLOPs in 2.76 ms) — far below
the Orin Nano's FP16 capability, i.e. that measurement is launch/memory-overhead-bound, not
compute-bound. A large 800² conv graph should utilise the device considerably better:

| assumption | effective TFLOP/s | P3 ms/decision | decisions/s |
|---|---:|---:|---:|
| A — scale the P1 backbone linearly (same poor utilisation) | 0.81 | 643 | **1.6** |
| B — 3 TFLOP/s (plausible for a large conv graph) | 3.0 | 176 | **5.7** |
| C — 6 TFLOP/s (optimistic FP16) | 6.0 | 89 | **11.3** |

**The band 1.6–11.3 straddles the 3.0 requirement.** So the projection does not answer whether P3
runs in real time on an Orin Nano — it only establishes that the answer is not obvious and that
assumption A (the naive one) puts P3 *below* the line. No feasibility claim for P3 may be made from
arithmetic; the device measurement is the only thing that settles it.

**Correction to an earlier version of this report.** It stated "~68 ms/decision, ~15 decisions/s,
~5x headroom". That was a unit error: it scaled the *stride-1 e2e* (8.62 ms) by 7.3x instead of
scaling the *per-decision* cost (94.2 ms). The corrected band is above; the comfortable-headroom
conclusion does not survive.

**Superseded by §6.** The band above is pure arithmetic. §6 replaces it with a measurement plus an
*empirically calibrated* transfer factor, and the answer lands below the whole optimistic half of
this band.

## 4. Export path built

`deployment/scripts/export-p3-graphs.py` (new). Exports the two P3 graphs that ultralytics does not
cover; stage A1 exports through ultralytics as usual.

- **A2 `p3-appearance-fp32.onnx`** — DONE. 4.2 MB, self-contained, `(K,3,224,224) -> (K,64)` with K
  dynamic. The 576->64 random projection (`default_rng(0) / sqrt(64)`) is folded into the graph as a
  constant `Linear`, so the ONNX output *is* `app_0..63` and cannot drift from the projection that
  produced the training tokens. **torch-vs-onnxruntime max abs diff 4.649e-06.**
- **B `p3-relation-head-fp32.onnx`** — blocked, see §5.

Two things the export surfaced:

- `OPSET = 17` is **not reachable**. torch 2.11's exporter emits opset 18 for these graphs and its
  downgrade pass fails inside onnx's `version_converter` (`No initializer or constant input to node
  found`). torch swallows that error, so an unfixed script would silently ship an opset it did not
  request. Set to 18 explicitly with the reason recorded; TRT >= 8.6 parses it. The P1 graphs stay
  at 17.
- The first `torch.onnx.export` writes an `.onnx.data` sidecar; `_save_inline` then re-saves inline
  but the orphaned sidecar survived. Every P1 export in `deployment/export/` has this pair on disk.
  Harmless locally, but rsync'ing a stale `.data` to the Jetson next to an inline model invites the
  belief that the graph still depends on it. `_save_inline` now deletes it.

## 5. Blockers

1. **Stage B weights did not exist.** The blind headline model was never persisted — only its
   softmax. `run_relation_transformer.py --save_model` added; regeneration running now under the
   recovered provenance command (`--tag yolo26app_ --seeds 5 --aux_w 0.5 --epochs 120`). seed0 51/64,
   seed1 **54/64** so far — seed 1 is the recorded blind seed, so it is tracking. The exported head
   must be checked against `logits_yolo26.pt`'s `blind_softmax` before any latency number is quoted.
2. **`jetson-benchmark.py` cannot run the relation head.** `TRTRunner.__init__` loops over IO
   tensors but keeps a *single* `self.host_in / self.dev_in` — the last input wins. The relation head
   takes **two** inputs (`tokens`, `mask`), so the harness would bind only one and run garbage. This
   is a structural limit, not a config change: multi-input support has to be added, or P3 gets its
   own harness. Not yet done.
3. **No P3 accuracy path in the harness.** `benchmark()` computes 4-class accuracy through
   `stats_pool_np` + cascade, which is P1-specific. P3 accuracy on-device needs the token tensors
   shipped, not `all8_test_onnx.pt`.

## 6. Measured: P3 does NOT meet the cadence at its current operating point

Ran `benchmark-p3-onnx.py --parity --bench --sweep` on an idle Apple M3 under ORT-CPU
(`deployment/artifacts/benchmarks-p3-onnx-cpu.json`). Parity gate passed first (max abs softmax diff
2.98e-07, argmax 64/64, ONNX top-1 54/64), so these timings belong to a graph that provably
reproduces the headline predictor.

**Measured, per decision (16 frames @800 + 58 crops + 1 head pass):**

| stage | p50 | share |
|---|---:|---:|
| A1 detector | 109.06 ms/frame x16 = **1745.0 ms** | **95.6 %** |
| A2 appearance | 79.6 ms (one batched call, 58 crops) | 4.4 % |
| B relation head | 0.46 ms | 0.03 % |
| **total** | **1825.0 ms = 0.55 decisions/s** | |

This confirms §2's FLOP accounting on the point that matters: **the detector is the entire cost**,
the appearance branch is a rounding error, and the Transformer is free. The earlier claim that the
64-d appearance branch makes P3 "materially heavier" was wrong in FLOPs and is now wrong in wall
clock too.

**Mac -> Jetson transfer, calibrated rather than assumed.** P1 is the only pipeline measured on both
machines: e2e p50 26.80 ms on this Mac under ORT-CPU vs 12.47 / 8.62 / 8.24 ms on the Orin Nano under
TensorRT fp32 / fp16 / int8 — speedups of **2.15x / 3.11x / 3.25x**. Applying them:

| precision | projected P3 ms/decision | decisions/s | vs 3.0 required |
|---|---:|---:|---|
| fp32 | 883 | **1.13** | **FAIL** |
| fp16 | 587 | **1.64** | **FAIL** |
| int8 | 562 | **1.71** | **FAIL** |

**Verdict: at its current operating point P3 misses the 24 fps / `EMIT_EVERY=8` cadence on an Orin
Nano by roughly 2x, at every precision.** §3 left this open because arithmetic could not decide it;
the measurement decides it, and it decides it against P3. Note this is *below* even assumption A of
§3's band — the optimistic B/C rows assumed a device utilisation the calibrated factor does not
support.

The transfer factor is a proxy and must be labelled one: P1 is a 224-px backbone with tiny heads,
P3 is dominated by an 800-px detector, and a GPU usually utilises the larger convolution *better*,
so 2.15–3.25x is if anything conservative. Even so, closing a 2x gap by that route alone is not
credible. **This does not become a feasibility claim until it runs on the device** — but the
direction is now measured, not guessed, and the honest Ch5 sentence is that P3 buys its accuracy
with an edge cost P1 does not pay.

### 6.1 What operating point *would* clear the bar (cost curve, accuracy unmeasured)

Since the detector is 95.6 % of the cost, the only two levers are frames-per-decision (T) and imgsz.
Measured on the same machine by exporting cost-probe detectors at 640/512/416
(`export-p3-graphs.py --cost_imgsz`), projected with the fp16 factor:

| imgsz | T | ms/decision (Mac) | Jetson~fp16 dec/s | |
|---|---:|---:|---:|---|
| **800** | **16** | **1897** | **1.64** | the measured configuration, 54/64 |
| 800 | 8 | 989 | 3.14 | marginal |
| 640 | 16 | 1238 | 2.51 | still short |
| 640 | 8 | 659 | 4.72 | |
| **512** | **16** | **824** | **3.77** | **least invasive pass** |
| 416 | 16 | 632 | 4.92 | |
| 416 | 4 | 218 | 14.27 | |

**Every row except the first is COST ONLY — its accuracy is unmeasured.** No tokens were ever built
at those sizes, so none of them carries the 54/64.

One asymmetry makes the choice non-arbitrary. T is part of the head's input shape `(1,16,8,74)`, so
changing it changes the *graph* and forces a re-architecture plus retrain. imgsz does **not** touch
the head at all — it only changes the token *values*. So **imgsz 800 -> 512 at T=16 is the cheapest
credible route to the cadence** (projected 3.77 dec/s fp16, 3.95 int8): rebuild tokens at 512, retrain
the head, re-measure accuracy. That is one token build plus one retrain, and it is a falsifiable
experiment rather than a claim.

Recommended thesis framing: report the measured 800x16 cost and the FAIL honestly, present the cost
curve as the engineering answer, and state the 512 route as future work with its accuracy explicitly
untested. This is stronger than a hedge — it quantifies the accuracy/latency trade P3 sits on.

## Unresolved

- Which cadence should Chapter 5 standardise on for *both* pipelines? Recommend `EMIT_EVERY = 8`
  (matches the multi-stream harness and a plausible deployment), with the stride-1 number kept as
  the conservative bound.
- Does A2 run per-box or batched on device? `embed_boxes` batches all boxes of one frame
  (`torch.stack`), so ~3.66 crops/call — small batches, launch-overhead dominated. Batching all 57.8
  crops of a clip in one call is an easy win but changes nothing numerically; worth doing before
  measuring.
- Multi-input `TRTRunner` vs a separate `jetson-benchmark-p3.py`: not decided. The P1 harness is
  already at the file-size limit and its accuracy path is P1-shaped, which argues for a separate file.
