# MPS Training Divergence Bug — Research Report
**Date:** 2026-07-08
**Context:** YOLOv11 (Ultralytics 8.4.27), torch 2.11.0, Apple M3, Python 3.13. box_loss fine at epoch 1 (~1.9), explodes at epoch 2+ (box→3.9, cls 7.5→10), mAP stays ~0. Reproducible regardless of optimizer/lr/imgsz/mosaic/amp. CPU/CUDA trains fine.

---

## 1. Is This a Known Issue?

**YES — multiple confirmed, overlapping MPS bugs cause exactly this symptom.**

### Bug A: Adam/AdamW addcmul_ + addcdiv_ silent failure on non-contiguous tensors (CRITICAL)
- PyTorch MPS backend: `addcmul_` and `addcdiv_` — the exact in-place ops used by Adam/AdamW for exp_avg and exp_avg_sq state updates — **silently fail to write back** when the target tensor is non-contiguous.
- Result: optimizer state stays frozen/zero → effective learning rate shoots toward infinity → loss explodes after the first optimizer step. Epoch 1 trains on pretrained weights; epoch 2 is where the corrupted optimizer state first manifests.
- Root cause: MPS "Placeholder" abstraction (temp buffer for non-contiguous memory) computed correctly but forgot to copy result back to original tensor.
- Fix: **macOS 15.0+** fixed the non-contiguous tensor path for these ops. On macOS < 15 the bug persists even on torch 2.10+.
- References: [Elana Simon's detailed post](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/), [PyTorch #165257](https://github.com/pytorch/pytorch/issues/165257), [blockchain.news writeup](https://blockchain.news/ainews/pytorch-mps-backend-bug-debugging-non-contiguous-tensor-failures-in-ai-model-training)

### Bug B: SGD also affected — MPS produces divergent/poor training vs CPU/CUDA baseline
- [PyTorch #137964](https://github.com/pytorch/pytorch/issues/137964): MPS on PyTorch 2.4.1, M3, macOS 15.0.1 — loss explodes to 1.9 billion with ReLU; CPU/CUDA converge normally. The bug is not optimizer-specific.
- [PyTorch #92615](https://github.com/pytorch/pytorch/issues/92615): Confirmed on PyTorch 1.13–2.0, MPS trains 4x worse perplexity vs CPU. Still labeled "module: correctness (silent)" and unresolved.

### Bug C: AMP / GradScaler broken on MPS
- MPS doesn't support float64; GradScaler internally calls `.double().reciprocal()` → TypeError crash or silent NaN.
- autocast fp16 on MPS added in PyTorch 2.5.0 but GradScaler not fully working ([PyTorch #142397](https://github.com/pytorch/pytorch/issues/142397), now closed).
- Ultralytics `amp=True` with MPS can silently produce garbage gradients even without a visible crash.

### Bug D: Ultralytics-specific MPS loss anomalies (confirmed 2025, unfixed)
- [ultralytics/hub#1199](https://github.com/ultralytics/hub/issues/1199): M3 Max, macOS 24.3, PyTorch 2.8.0, Ultralytics 8.3.202: `box_loss=0, dfl_loss=0, cls_loss=120`. Closed **"not planned"** — no fix.
- [ultralytics/ultralytics#18413](https://github.com/ultralytics/ultralytics/issues/18413): MPS pose training fails during backward pass, closed "not planned".
- [ultralytics/ultralytics#22174](https://github.com/ultralytics/ultralytics/issues/22174): MPS classification IndexError on M3, fixed in #22181 (tensor device mismatch).

**Summary:** The divergence pattern (epoch 1 fine → epoch 2+ explodes) is the classic signature of the Adam non-contiguous bug (Bug A). Bug B confirms SGD also produces incorrect MPS gradients. These bugs coexist and reinforce each other.

---

## 2. Is torch 2.11.0 a Stable Release? Any MPS Regressions?

- **torch 2.11.0 is a stable release** (GA announced March 23, 2026, [PyTorch 2.11 Release Blog](https://pytorch.org/blog/pytorch-2-11-release-blog/), [dev-discuss GA](https://dev-discuss.pytorch.org/t/pytorch-2-11-0-general-availability/3328)).
- 2.11 MPS changes: "Comprehensive Operator Expansion" — new distributions, `erfcx`, `grid_sampler_2d`, `baddbmm/addbmm` for int/complex, asynchronous error reporting for OOB indexing. **No mention of fixing the non-contiguous in-place op bugs.**
- No community reports of "MPS now trains correctly in 2.11" for YOLO-class models. The correctness bugs appear to persist.
- Reported stable versions for MPS from community: torch 2.4.1 and 2.5.x are the most-cited "worked for me" versions, but even those have the non-contiguous issue on macOS < 15.

**Recommended stable pin for MPS (if using MPS at all):** torch 2.5.1 on macOS 15.0+. But correctness is still not guaranteed.

---

## 3. Workarounds (Ranked by Effectiveness)

### Rank 1 — Use CPU (device='cpu') [MOST RELIABLE]
```bash
yolo train device=cpu ...
```
- Eliminates all MPS bugs. CPU on M3 is slower but correct.
- For 1900 images × ~100 epochs: expect 2–6 hours on M3 depending on imgsz and model size (yolo11n/s).
- **This is what the Ultralytics community converges on for M-series Mac training.**
- References: [ultralytics#2599](https://github.com/ultralytics/ultralytics/issues/2599), multiple community posts.

### Rank 2 — Kaggle free GPU (T4 x2 or P100) [BEST FOR QUALITY + SPEED]
- 30 hours/week free, no credit card needed.
- T4: ~16GB VRAM. P100: 16GB VRAM. Both are CUDA → no MPS bugs.
- For 1900 images, yolo11s, 100 epochs at imgsz=640: ~30–60 min on T4.
- Upload dataset via Kaggle API or dataset panel; set `device=0` or `device=cuda`.
- Ultralytics has an official Kaggle integration guide for YOLO26 ([docs.ultralytics.com/integrations/kaggle](https://docs.ultralytics.com/integrations/kaggle)).
- Kaggle is more stable than Colab for long jobs (no "are you still there" popups, 9-hour session max).

### Rank 3 — PYTORCH_ENABLE_MPS_FALLBACK=1 [PARTIAL FIX ONLY]
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 yolo train device=mps ...
```
- Falls back unsupported ops to CPU; does NOT fix the silent non-contiguous Adam bug (those ops ARE implemented on MPS, just incorrectly).
- Some users report it helps with crashes but loss divergence persists.
- References: [PyTorch #86195](https://github.com/pytorch/pytorch/issues/86195), [PyTorch #134416](https://github.com/pytorch/pytorch/issues/134416).

### Rank 4 — Force SGD + amp=False + lr0 reduction [REDUCES DIVERGENCE RISK, NOT GUARANTEED]
```yaml
optimizer: SGD
lr0: 0.001       # 10x lower than default
amp: False       # disables broken GradScaler path
```
- Avoids the Adam addcmul_/addcdiv_ bug. AMP=False removes GradScaler.
- SGD on MPS still has correctness issues (Bug B) but is less catastrophic than Adam.
- Some users report partial convergence with this combo; not reliable for thesis-quality results.

### Rank 5 — Upgrade macOS to 15.0+ [FIXES BUG A, NOT BUG B/C]
- macOS 15.0 fixed the non-contiguous tensor write-back path.
- If on macOS 14.x (Sonoma), upgrading to macOS 15 (Sequoia) is worth trying.
- Does not fix all MPS correctness issues — Bug B (general MPS gradient errors) persists.

### Rank 6 — PYTORCH_MPS_HIGH_WATERMARK_RATIO [MEMORY, NOT CORRECTNESS]
```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```
- Controls memory allocation ceiling, not correctness. Not relevant to this divergence.

---

## 4. Does Ultralytics Recommend MPS for Training?

Ultralytics official docs ([docs.ultralytics.com/modes/train](https://docs.ultralytics.com/modes/train)) state MPS is **supported** and list `device='mps'` as a valid option. However:
- Multiple MPS issues closed "not planned" — Ultralytics does not actively maintain MPS training correctness.
- No official guidance distinguishing MPS inference (works fine) from MPS training (correctness bugs).
- The practical signal: community threads consistently end with "switch to CPU or cloud GPU."

**MPS for inference: fine. MPS for training: unsafe for production-quality results.**

---

## 5. Practical Recommendation for Thesis (16GB M3, ~1900 images)

### Recommended path (in order of preference):

**Option A — Kaggle (best quality + speed for thesis)**
1. Upload dataset to Kaggle (free, up to 20GB).
2. Create notebook, enable GPU (T4 or P100).
3. `pip install ultralytics` → `yolo train device=0 data=your.yaml model=yolo11s.pt epochs=100 imgsz=640`
4. Expected: 30–60 min per 100 epochs. 30 hrs/week free quota is ample for full sweep.
5. Download `best.pt` when done.

**Option B — CPU on M3 (self-contained, slower)**
1. `yolo train device=cpu data=your.yaml model=yolo11s.pt epochs=100 imgsz=640 batch=8`
2. Use yolo11n or yolo11s (not m/l) to keep time manageable.
3. Overnight run (~4–8 hrs for 100 epochs). Correct results guaranteed.

**Option C — MPS with mitigations (risky, thesis results may not be reproducible)**
- Only if macOS 15.0+, torch 2.5.1, `amp=False`, `optimizer=SGD`, `lr0=0.001`.
- Still not recommended for thesis — results may not match CUDA baseline cited in papers.

### Bottom line
The divergence at epoch 2 is almost certainly the Adam non-contiguous MPS bug (Bug A) combined with broken AMP GradScaler (Bug C). No torch version pin fully resolves this. **Use Kaggle T4 for all training runs in the thesis; reserve the M3 for inference testing and Jetson benchmarks.**

---

## Unresolved Questions
1. Whether torch 2.11 + macOS 15 fully resolves Bug B (general MPS gradient incorrectness vs CUDA). No community confirmation found.
2. Whether Ultralytics 8.4.27 includes any MPS-specific fixes beyond #22181 (classification IndexError). Release notes not reviewed line-by-line.
3. macOS version of the affected machine not stated — if already on macOS 15+, the addcmul_ bug (Bug A) may be fixed, pointing to Bug B or Bug C as the remaining culprit.

---

## Sources
- [Elana Simon — MPS addcmul_ bug post](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/)
- [PyTorch #137964 — MPS silent training failure M3](https://github.com/pytorch/pytorch/issues/137964)
- [PyTorch #165257 — MPS random in-place ops fail macOS<15](https://github.com/pytorch/pytorch/issues/165257)
- [PyTorch #92615 — MPS bad training results vs CPU/CUDA](https://github.com/pytorch/pytorch/issues/92615)
- [PyTorch #142397 — GradScaler MPS float64 crash](https://github.com/pytorch/pytorch/issues/142397)
- [ultralytics/hub#1199 — box_loss=0, cls_loss=120 on M3](https://github.com/ultralytics/hub/issues/1199)
- [ultralytics#18413 — MPS pose training backward pass fail](https://github.com/ultralytics/ultralytics/issues/18413)
- [ultralytics#22174 — MPS classification IndexError M3](https://github.com/ultralytics/ultralytics/issues/22174)
- [PyTorch 2.11 Release Blog](https://pytorch.org/blog/pytorch-2-11-release-blog/)
- [PyTorch 2.11 GA Announcement](https://dev-discuss.pytorch.org/t/pytorch-2-11-0-general-availability/3328)
- [Ultralytics Kaggle integration guide](https://docs.ultralytics.com/integrations/kaggle)
- [Ultralytics train docs](https://docs.ultralytics.com/modes/train)
