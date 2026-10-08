# Phase 04 — Pre-Jetson Checkpoint + Export

## Context Links

- `deployment/scripts/export-p3-graphs.py`, `deployment/export/p3-{detector,appearance,relation-head}-fp32.onnx`
- `plans/2026-09-06-thesis-2week-final/reports/laneb-p3-edge-cost-and-export.md`
- `reports/test-touch-log.md` — 2026-09-09 precedent: latency work consumes **zero** test touches.

## Overview

- **Priority:** P1 — last step before the fixed device date. **Status:** pending. **Effort:** 6h. **Due 18/9 EOD.**
- Produce the exact artifacts the 19/9 session consumes: a frozen graph, a representative val-selected
  checkpoint, ONNX exports, and INT8 calibration data built from the **train split only**.

## Key Insights

- **The 19/9 session does not need the final headline model.** Latency, throughput, energy, thermal and
  memory depend on the **graph**, not the weights; FP16/INT8 parity is measured as *output agreement vs
  FP32* on train/val clips. So a representative checkpoint suffices, and the single test touch is spent
  calmly later (phase 06). This is what makes the fixed date survivable.
- Consequence: **architecture freeze (G2), not weight finality, is the hard 18/9 gate.** When the final
  headline checkpoint lands with an identical graph, latency/energy numbers transfer **exactly**; only
  accuracy-dependent numbers are re-verified (on CPU/ORT) and the substitution is disclosed in the thesis.
- The existing `p3-relation-head-fp32.onnx` outputs `logits [1,4]` — a **binary reframe invalidates it**.
  Re-export is mandatory, not optional.
- INT8 calibration data must come from the **train split**; calibrating on test would be a test touch.

## Requirements

Functional: frozen `.pt` checkpoint + `relation-head-bin-fp32.onnx` (2 inputs, binary+aux outputs) +
calibration cache + a `p3-detector` export at imgsz ∈ {800, 512, 416}.
Non-functional: ONNX opset 18; parity ≤1e-5 max-abs vs PyTorch on 50 val clips; artifacts checksummed.

## Architecture

train-split tokens → train binary encoder (seeds 0-4, select on **val**) → pick the median-val seed as the
*representative* checkpoint → `torch.onnx.export` (dynamic batch) → ORT CPU parity check → package
`deployment/export/` + `calib/` for transfer to the device.

## Related Code Files

Modify: `deployment/scripts/export-p3-graphs.py` (binary head, 3 detector image sizes),
`deployment/scripts/jetson-prep-int8-calib.py` (train-split source). Create:
`deployment/export/p3-relation-head-bin-fp32.onnx`, `deployment/export/MANIFEST.md`.

## Implementation Steps

1. Train the binary encoder on train, select on val, **5 seeds** (`--seeds 0,1,2,3,4`), ≈40 min.
   **0 test touches** (no test data loaded). Supports: "representative deployed configuration".
2. Choose the **median-val** seed (not the best — avoids optimistic selection); record seed + checksum.
3. Re-export: relation head (2 inputs), appearance backbone, detector at imgsz 800/512/416.
4. Parity: ORT CPU vs PyTorch on 50 **val** clips, max-abs ≤1e-5 → **G4**.
5. Build the INT8 calibration set: 200 **train** clips, cached tensors, checksummed.
6. Write `MANIFEST.md`: every file, checksum, graph hash, seed, and the sentence "no test clip was used".
7. Rehearse: run the full phase-05 runbook against the Mac/ORT path end-to-end to catch missing flags.

## Todo List

- [ ] 5-seed binary training complete (val-selected)
- [ ] Representative checkpoint chosen (median val) + graph hash recorded
- [ ] Binary relation head re-exported with 2 inputs
- [ ] Detector exported at 800/512/416
- [ ] **G4 parity ≤1e-5**
- [ ] INT8 calibration set built from train split
- [ ] MANIFEST.md written
- [ ] Runbook dry-run passes on the Mac

## Success Criteria

G4 met; all artifacts present with checksums; the dry-run executes every runbook command without an unknown
flag or missing file.

## Risk Assessment

R10 (export drift) — mitigated by the parity gate; no latency number may be published for a graph that has
not passed G4. If G2 slips, export the *fallback* graph (PE change only) so the device session still
measures a binary 2-input head.

## Security Considerations

Calibration tensors derive from factory footage — keep them local; do not commit `.pt`/`.onnx` blobs
larger than the repo policy allows. Never calibrate or select on test (contract §4h).

## Next Steps

Hands off to phase 05 on 19/9. Phase 06 reuses the same training command with the full protocol suite.
