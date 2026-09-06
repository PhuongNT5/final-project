# T3 - Build model code

**Priority:** P0. **Status:** blocked on T1 claim decision (partial work can start). **Owns:**
`unsafenet/*.py`, `train_*.py`, `run_unsafenet_r2.py`.

## Overview
Implement the paper's full path now that its detector stage finally trains, plus the <=2 upgrades T2
recommends. No new "enhanced" files - edit existing modules (project rule).

## Related code files
- Modify: `unsafenet/behaviour_clip.py` (detector -> clip decision; already runs)
- Modify: `unsafenet/train_temporal.py` (accept detector confidences as a stream)
- Modify: `unsafenet/streams.py` (add behaviour-confidence channels to the per-frame vector)
- Modify: `unsafenet/fusion.py`, `unsafenet/aggregate.py` (member pool includes det-* members)
- Create: `unsafenet/convlstm.py` - the paper's actual temporal model, which we have never built.
  We used mean/BiLSTM/Transformer heads instead. A ConvLSTM over raw frames is the paper's model and
  its absence is a live "you did not reproduce it" question at the defence.

## Implementation steps
1. **Detector-conditioned streams.** Feed `behaviour_conf_*.npy` (N,32,8) into `train_temporal` as an
   extra stream; retrain the temporal heads on [entity geometry + CNN logits + behaviour conf].
2. **ConvLSTM baseline** (`convlstm.py`): raw-frame ConvLSTM on 452 clips, matching the paper's
   no-reduction row (their 57/64). Reproducing their *baseline* is the credibility anchor - if we hit
   ~57/64 there, our 55/64 stops looking like underperformance.
3. **Paper's decomposition, implemented honestly.** Apply YOLO frame-reduction + single-action
   segmentation exactly as described, and *also* a leak-free variant where segment class is never used
   as the answer. Report both. This is the head-to-head the thesis needs.
4. **Upgrades from T2** (<=2), each behind a flag so ablations are one command.
5. Compile check + `code-reviewer` pass after each module.

## Todo
- [ ] behaviour-conf stream wired into train_temporal
- [ ] convlstm.py implemented + compiles
- [ ] paper decomposition (leaky + leak-free variants)
- [ ] T2 upgrade #1
- [ ] T2 upgrade #2 (if justified)
- [ ] code-reviewer clean

## Success criteria
Every configuration in T4's queue is runnable by one command with a flag; ConvLSTM baseline trains
without error; no file over 200 lines.

## Risks
- ConvLSTM on raw frames at T=32 may not fit memory on MPS. Mitigation: downscale to 112x112, T=16,
  and say so.
- Scope creep into a fourth round of head variants. Mitigation: the member pool is already 34 configs
  and val cannot rank them (r = -0.066). Adding heads is known-useless; only add what T2 justifies.
