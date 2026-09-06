# T4 - Training & experiments

**Priority:** P0. **Status:** running (behaviour_clip inference on final detectors). **Owns:**
`runs/`, `sweep_results/`, `cache_r2/`, all logs.

## Overview
Execute the experiment queue. Hardware-bound: one heavy job at a time. This track is a *queue*, not a
parallel worker.

## Hardware facts (measured, plan against these)
- Ultralytics YOLO **training** on MPS is broken in this stack -> `device=cpu, amp=False` always.
  CPU cost: ~8.25-10 min/epoch at 416 px / batch 8 / 1901 images. 25 epochs ~= 3.5 h.
- Ultralytics **inference** on MPS is fine (0.9752 mAP validation) - use MPS for all predict passes.
- Pure-PyTorch training on MPS is fine (frame CNNs trained there: val macro 83.7 / 84.7).
- `behaviour_clip` full inference: ~14 min per detector over 691 clips.

## Queue (in order; each writes a log, nothing runs unattended without a completion hook)
1. **[running]** behaviour_clip on beh_tight + beh_ctx -> `det{T,C}-*` members.
2. Fusion + eval including det members -> blind and test-selected numbers. *First real answer on
   whether the paper's own stage helps us.*
3. ConvLSTM baseline (T3 step 2) - target the paper's 57/64 no-reduction row.
4. Paper decomposition, leaky vs leak-free (T3 step 3).
5. Detector-conditioned temporal heads (T3 step 1), seeds >= 5 - single seeds are noise (88.2 -> 85.2
   when a "best" config was rerun at 8 seeds).
6. T2 upgrades, if any.

## Rules
- Every run: fixed seed, logged config, results.csv retained.
- >= 5 seeds for any number that will be quoted. No single-seed claims.
- A failed run gets a **control** before it gets an explanation.
- Chain jobs in one script so the machine never idles waiting for a check-in.

## Todo
- [ ] det members (running)
- [ ] fusion + eval with det members
- [ ] ConvLSTM baseline
- [ ] decomposition leaky vs leak-free
- [ ] detector-conditioned heads, 5 seeds
- [ ] T2 upgrades

## Success criteria
Every T1-required number exists with a log behind it and a seed count attached.

## Risks
- Compute: items 3-5 are ~6-12 h total on this machine. Mitigation: chain overnight, single script.
- Overfitting the test set by re-evaluating. Mitigation: T1's frozen protocol; test touches counted
  and reported.
