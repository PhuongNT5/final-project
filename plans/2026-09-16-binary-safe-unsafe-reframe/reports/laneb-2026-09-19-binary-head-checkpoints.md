# Binary relation-head TRAINED checkpoints + re-export (2026-09-19, pre-Jetson-session)

Scope: train E3/E0 binary checkpoints (TRAIN split only, zero test touches), re-export both ONNX
graphs, re-run numerics-only parity, update MANIFEST.md + runbook. Supersedes the 2026-09-18
random-init disclosure in `laneb-2026-09-18-binary-onnx-export.md` (that report is left unedited as
historical record; this report + MANIFEST.md are now the current-state sources).

## 1. Why the previous lane's "training = test touch" belief was wrong

18/9's lane read `run_relation_transformer.py --binary --save_model` (`main_binary()`) as the only
save path and correctly saw it evaluates the 125-clip test set inside its own flow — so it stopped,
believing ANY training run would cost a touch. That conflates "the existing CLI entry point touches
test" with "training touches test." Training itself never reads test data; only *scoring* on it does.
The fix is a dedicated entry point that trains + selects on val + saves, and exits before touching test
— not a flag bolted onto `main_binary` (safer: a new file can't accidentally inherit a later edit to
`main_binary` that adds a test read).

## 2. Path taken: new dedicated script, not a `--no_test_eval` flag on `main_binary`

New file `deployment/scripts/train-binary-head-checkpoint.py` (97 lines). It imports only
`load_binary` and `train_one` from `run_relation_transformer.py` — never `main_binary`, `predict`,
or `evaluate_suite`. The **only** `load_binary(...)` call in the file:

```python
Xtr, Mtr, ytr, ysub_tr, views_tr, _ = load_binary('train', args.tag, args.dedupe)
```

`load_binary(sp, tag, dedupe_policy)` builds the token path as `tokens_{tag}{sp}.pt` — with `sp`
hardcoded to the literal `'train'` there is no code path in this file that can construct
`tokens_bin8_test.pt`. Grepped to confirm exactly one call site (`grep -n "load_binary(" ...` → 1
match, see script docstring for the same grep instruction so it self-documents). No
`evaluate_suite`/`mcnemar_exact`/`bootstrap_ci`/test-metric call anywhere in the file. Val selection
copies `main_binary()`'s own (label,view)-stratified carve verbatim (same `random_state=0`,
`test_size=0.18`) so the split matches what the screening runs used.

## 3. Training run

Recipe: `--dedupe drop_conflict --dedupe_task binary` implied by `load_binary`'s dedupe arg (binary
audit forced since `load_binary` always calls `dedupe.keep_mask(..., task='binary')`), 60 epochs,
class weights on (`use_cw=True` in `train_one`, `class_weights(yf, k=2)`), label smoothing 0.05
(hardcoded in `train_one`'s `CrossEntropyLoss`), sub-class aux λ=0.3 + per-frame aux λ=0.5 (both
always on for `binary=True` in `run_epoch`, no flag to disable). 562 train clips after
`drop_conflict` drops 4 (`0_tr123, 0_tr136, 0_tr138, 1_tr55`), fit=460/val=102. 5 seeds each (time
budget allowed — total wall time well under the 20-min cap, see §5, so no reason to drop to 1 seed).

| arch | seeds | epochs | blind seed | val_macro | val_top1 | @epoch | wall time |
|---|---|---|---|---|---|---|---|
| E3DeepSets | 5 | 60 | 0 | **89.15%** | 91.18% | 49 | 9.2s |
| RelationTransformer (E0) | 5 | 60 | 2 | **90.92%** | 91.18% | 42 | 320.6s |

E0's 5-seed val_macro spread: 87.2–90.9% (seeds 0,1,3,4,2 = 88.45/90.21/87.83/87.21/90.92). E3's:
84.3–89.2%. Both consistent with the val-gate numbers already reported in
`laneb-2026-09-17-binary-baseline-gates.md`. Checkpoints:
`sweep_results/relation_tokens/canonical/bin8_head_{e3,e0}_checkpoint.pt` (gitignored by
`sweep_results/**/*.pt`, not committed), each stores `state_dict`, `ctor` (arch/n_num/d),
`blind_seed`, `val_macro`, `val_top1`, `epoch`, `git_commit` (`43657eafc5c8`), `argv`, dedupe/coord
metadata. **No test-set field exists in either checkpoint** — the save call has no `labels`/`softmax`
key for test, unlike `main_binary`'s `--save_model` output which stores `blind_softmax` computed
against `Xte`.

## 4. Re-export + parity (numerics only, no labels)

Ran the existing `deployment/scripts/export-p3-relation-head-binary.py --head_ckpt <ckpt> --force`
once per checkpoint (its `ctor.arch` mismatch `continue`s past the other arch each time, so the two
invocations don't clobber each other — this was already how the script was designed 2026-09-18, no
script-logic change needed).

| graph | provenance | max abs logit diff | decision agreement | dynamic batch | checksum (sha256, first 16) |
|---|---|---|---|---|---|
| `p3-relation-head-bin-e3-fp32.onnx` | TRAINED (val_macro 89.15%) | 1.431e-06 | 100.0% (125/125) | genuinely dynamic (batch=4 OK) | `6dbfd47a1701e2e1` |
| `p3-relation-head-bin-e0-fp32.onnx` | TRAINED (val_macro 90.92%) | 2.623e-06 | 100.0% (125/125) | batch=1 only (pre-existing, confirmed unchanged) | `307884c01fe2d650` |
| `p3-relation-head-fp32.onnx` (legacy 4-class) | untouched | n/a | n/a | n/a | `582733bfac4f340b` (byte-identical to before) |

Parity harness (`check_parity_real_tokens` in the export script, unmodified) loads
`tokens_bin8_test.pt` **only** to get the input tensors (`tok`, `mask`) for a forward pass through
both PyTorch and ONNX Runtime, then compares `want` vs `got` — it never reads or touches the file's
`labels`/`sub_labels` field, and nothing in the diff/agreement computation involves ground truth
(`want.argmax(1).item() == got.argmax(1).item()`, torch-vs-onnx only). This is the same numerics-only
harness the 18/9 lane wrote and used; re-running it against trained weights doesn't change what it
computes, only that the compared model is no longer random. Still zero test touches: no metric
against `labels` was computed, printed, or saved.

E0's batch=4 ORT failure reproduces byte-identically (`Reshape node 'gemm_input_reshape' ...
{129,4,128}->{129,128}`) on the retrained graph — confirms the dynamo-exporter batch-baking is a
model-architecture property, not an artifact of the random-init weights.

## 5. Time budget

Total wall time: E3 train 9.2s + E0 train 320.6s + 2 exports (~15s each incl. parity over 125 clips)
≈ **6 min**, comfortably under the ~20-min cap. 5 seeds used for both archs (not reduced to 1) —
recording this since the task allowed dropping to 1 seed under time pressure; it wasn't needed.

## 6. MANIFEST.md / runbook / export-script docstring updates

- `deployment/export/MANIFEST.md`: G4 gate flipped NOT-MET → MET (trained weights); both graph
  sections rewritten with trained-checkpoint provenance, new checksums/sizes, new parity numbers;
  "What is NOT verified" bullet on accuracy rewritten to say test-set accuracy (not "any accuracy")
  is what's outstanding; "Reproduce" section now documents train-then-export as two steps.
- `deployment/JETSON-RUNBOOK-2026-09-19.md`: top status block, 18/9 prerequisite checklist (checkbox
  flipped for trained-checkpoint export, new checkbox added for test-touch-still-pending), Step 0
  file-transfer list, Step 5 FP16/FP32 parity guidance (now says `argmax` agreement is a real
  decision-quality check, not noise-on-random-logits; `--rule threshold` still unused pending a
  calibrated τ, unrelated to training), Step 6 `--class-labels` note. Grepped for stale
  "random"/"placeholder"/"no trained" — remaining hits are correctly historical (referencing the
  2026-09-18 state) or unrelated (`p3-appearance-fp32.onnx`'s *architecture* uses a fixed random
  projection layer, unrelated to checkpoint training).
- `deployment/scripts/export-p3-relation-head-binary.py` docstring: added a 2026-09-19 update note
  (no logic change) so the file's own header doesn't contradict MANIFEST.md; file still 189 lines.

## 7. Constraints check

- File ownership: `deployment/` (new script, MANIFEST.md, runbook, export-script docstring) +
  `sweep_results/relation_tokens/canonical/*.pt` (gitignored) + this report. Did not touch
  `model_relation_head.py`, `model_binary_encoders.py`, `run_relation_transformer.py` (import-only).
- File sizes: `train-binary-head-checkpoint.py` 97 lines, `export-p3-relation-head-binary.py` 189
  lines (unchanged count, docstring edit was a net-neutral swap). Both <200.
- Naming: kebab-case, matches `deployment/scripts/` convention (`docs/code-standards.md` §File
  naming).
- Compile check: `python3 -m py_compile` both changed/added scripts — pass.
- Ran existing `tests/test_binary_encoder_forward.py` + `test_binary_encoder_wiring.py` (unaffected
  by this lane's changes, sanity-checked anyway since checkpoints depend on that model code) — all
  pass (9/9).
- Did not commit (per instructions).
- Did not run `main_binary`, `predict()`, or any test-labeled evaluation at any point in this lane.

## Unresolved questions

1. **Which arch does the Jetson session build/measure primarily?** Runbook still defaults to
   "E3 primary, E0 optional" per the freeze recommendation — E0 now has the higher val_macro
   (90.92% vs 89.15%) on this single 5-seed run, but that's a training-time metric on 102 val clips,
   not the LCO-691/statistical-tie evidence that drove the freeze call in
   `laneb-2026-09-17-encoder-impl-and-screening.md` §6. Not overriding that decision here — flagging
   the new number in case whoever picks the final arch wants it.
2. **When to spend a test touch on these checkpoints?** Both are now real, trained, deployment-ready
   artifacts with zero test-set accuracy number. The two touches pre-registered in
   `reports/test-touch-log.md` (2026-09-17 G0) are (1) E0 retrain, (2) the pre-registered rule's
   winner — this lane's E0 checkpoint could arguably serve as "the E0 retrain" touch's model, but
   spending the touch itself is a thesis-author decision, not this lane's to make.
3. `benchmark-p3-onnx.py` still hardcodes the legacy 4-class graph as its cost-probe target (flagged
   by the 18/9 report too, still unaddressed, still out of scope here).
