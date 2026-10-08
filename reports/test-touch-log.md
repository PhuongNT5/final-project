# Test-touch log

Required by `plans/2026-09-06-thesis-2week-final/brief.md` section 4h.3: *"One declared test touch per
pipeline variant, timestamp-logged."* and 4h.4: *"Headline configuration pre-registered before the touch."*

> ## Honesty statement — read before citing anything below
>
> **This log was created on 2026-09-06. It did not exist before that date.**
>
> Every entry dated before 2026-09-06 is a **retroactive reconstruction** from file mtimes and run logs,
> **not** a pre-registration. Those touches were not declared in advance and their pre-registration
> **cannot be verified**. They are recorded here for provenance and transparency only.
>
> The thesis must state this plainly. Pre-registration discipline is claimable **from 2026-09-06 onward
> only** — claiming it for earlier work would be false. A reconstructed provenance trail is worth
> having; passing it off as a pre-registration is not.

---

## Section A — retroactive reconstruction (NOT pre-registered)

Basis: file mtime plus surrounding run logs. The 64 unsafe test clips are the test set throughout.
`blind_pred` = best-val single seed. `argmax(ens_softmax)` = seed-averaged ensemble. These are two
different decision rules (see brief.md S4); both are listed because they differ.

| Date (mtime) | Artifact | blind_pred | ensemble argmax | Notes |
|---|---|---|---|---|
| 2026-08-10 00:27 | `logits_v1nocw.pt` | 51/64 | 49/64 | early relation-token variant, no class weighting |
| 2026-08-10 00:27 | `logits_v2bnocw.pt` | 52/64 | 52/64 | ditto, v2b tokens |
| 2026-08-13 21:38 | `logits_s1a.pt` | 50/64 | 54/64 | **byte-identical to `logits_yolo11_baseline.pt`** (SHA-1 of `ens_softmax` matches; `torch.equal` on all three stored tensors). One of the two names is mislabeled — same run, not two configs. Never cite both as independent evidence. |
| 2026-08-16 15:04 | `logits_yolo11_baseline.pt` | 50/64 | 54/64 | see above — duplicate of `s1a` |
| 2026-08-16 17:07 | `logits_yoloroi.pt` | **55/64** | 54/64 | **Higher than the current headline.** Traced to a same-day YOLO11-track exploratory run, superseded by the 2026-08-29 switch to the YOLO26 pipeline. Plausibly not a cherry-pick — but **unverifiable**, because no pre-registration exists for it or for anything else in this section. Not promoted to headline. |
| 2026-08-29 00:28 | `logits_yolo26.pt` | **54/64** | 52/64 | **current P3 same-dist headline.** Exact generating `--tag` is inferred from mtime only; no training log was persisted. |

**Unresolved for section A:** the number of test evaluations actually performed before 2026-09-06 is
**unknown and unknowable from the artifacts** — only the surviving files are visible, not overwritten or
deleted ones. The one-touch-per-variant rule therefore cannot be demonstrated for any pre-2026-09-06
result. State this as a limitation; do not assert compliance.

---

## Section B — pre-registered touches (2026-09-06 onward)

Rules for this section, per brief 4h:
1. Write the entry **before** running anything against the 64-clip test set.
2. Record: date, pipeline variant, exact command, the declared headline configuration, the cost matrix,
   and the metric set expected — all fixed in advance.
3. One touch per pipeline variant. A second touch requires a new dated entry with a stated reason.
4. Model / epoch / seed / fusion selection happens on val only, never here.

### Declared cost matrix (fixed 2026-09-06, before any touch)

Safety domain: a missed unsafe behaviour is worse than a false alarm. Fixed in advance so it cannot be
tuned to flatter a result.

| | predicted safe | predicted unsafe |
|---|---|---|
| **actually unsafe** | cost 10 (miss) | cost 0 |
| **actually safe** | cost 0 | cost 1 (false alarm) |

Ratio 10:1. Any deviation from this ratio in a reported result requires an entry here saying so and why.

### Entries

### 2026-09-06 — P3 (YOLO26 entity + relation-Transformer) — REPRODUCTION / PROVENANCE-RECOVERY touch

**Written before the run. Nothing below the "Result" line existed when the command was launched.**

- **Purpose (not a headline selection).** Two open defects need the same artifact:
  (a) **brief.md S4 / DEFECT 5** — `logits_yolo26.pt` stores only `ens_softmax` (the seed-ENSEMBLE rule,
      52/64). The blind rule's own softmax was never persisted, so *every* calibration / ECE / conformal /
      selective-risk / ROC / PR figure in the thesis describes a predictor that is **not** the 54/64
      headline. `run_relation_transformer.py` has been patched to persist `blind_softmax`,
      `blind_seed`, and `per_seed_softmax`; recovering them for P3 requires re-running the training.
  (b) **w2-number-integrity-fixes.md unresolved #1** — the exact command that produced
      `logits_yolo26.pt` was never logged; it is currently *inferred from file mtimes only*.
- **Declared headline config (fixed, unchanged from the stored artifact — nothing is being selected):**
  `--tag yolo26app_`, `RelationTransformer` defaults, `--epochs 120`, seeds `0..N-1` via `set_seed(s)`,
  class weighting on, epoch chosen by **val** macro inside `train_one`, blind seed chosen by **val**
  macro. No test quantity enters any choice.
- **Command (one candidate config per line; `--seeds` / `--aux_w` are the only unknowns being recovered,
  because the original invocation was not logged):**
  ```
  .venv/bin/python run_relation_transformer.py --tag yolo26app_ --seeds 7            --save_logits <scratch>/repro_s7_a0.pt
  .venv/bin/python run_relation_transformer.py --tag yolo26app_ --seeds 7 --aux_w 0.5 --save_logits <scratch>/repro_s7_a05.pt
  .venv/bin/python run_relation_transformer.py --tag yolo26app_ --seeds 5            --save_logits <scratch>/repro_s5_a0.pt
  .venv/bin/python run_relation_transformer.py --tag yolo26app_ --seeds 5 --aux_w 0.5 --save_logits <scratch>/repro_s5_a05.pt
  ```
  Outputs go to the scratchpad. **`sweep_results/relation_tokens/logits_yolo26.pt` is not overwritten.**
- **Protocol:** same-dist 64. **Metric set:** brief 4b/4c/4d/4e/4f. **Cost matrix:** as declared above.
- **Reason this touch is permitted:** it is a *reproduction* of an existing recorded result, not a new
  configuration search. The stored `blind_pred` (54/64) already exists on disk and is unchanged by this
  run; the run can only confirm or fail to confirm it.
- **Decision rule, fixed in advance (this is the part that must not be decided after seeing the output):**
  1. If exactly one candidate reproduces the stored `blind_pred` **element-wise on all 64 clips**
     (`torch.equal`), that command is the recovered provenance and its `blind_softmax` is adopted for
     all calibration/conformal figures. Chance agreement on 64 4-way predictions is negligible.
  2. If **no** candidate reproduces it, the conclusion recorded is **"P3 is not bit-reproducible on this
     stack; provenance cannot be recovered"** — and the calibration figures stay on the ensemble basis
     with the S4 label. The re-run's own `blind_top1`, whatever it is, is **NOT** promoted to headline
     and **NOT** substituted for 54/64; it is reported only as a reproducibility measurement.
  3. If more than one candidate matches, all matching commands are recorded and none is claimed as *the*
     provenance.
  4. The stored 54/64 remains the reported same-dist number in every branch.
- **Result (2026-09-06, run `bacckkzbm`; all four candidates completed):**

  | candidate | `--seeds` | `--aux_w` | blind_top1 | `blind_pred` == stored | `ens_softmax` == stored | max abs diff |
  |---|---|---|---|---|---|---|
  | s7_a0  | 7 | 0.0 | 51 | ✗ | ✗ | 1.29e-01 |
  | s7_a05 | 7 | 0.5 | **54** | **✓** | ✗ | 5.61e-02 |
  | s5_a0  | 5 | 0.0 | 50 | ✗ | ✗ | 1.23e-01 |
  | s5_a05 | 5 | 0.5 | **54** | **✓** | **✓** | **0.00e+00** |

  `labels` identical to the stored artifact in all four.

  **Branch taken — and a disclosure about the rule itself.** Two candidates reproduce `blind_pred`
  element-wise, so on the *letter* of the pre-registration this is **branch 3** (more than one match),
  which would forbid naming a provenance. I am not silently promoting it to branch 1. What resolves it
  is that the stored artifact contains a **second** tensor the rule did not think to name: `ens_softmax`.
  Only `s5_a05` reproduces it — bit-for-bit, max abs difference exactly `0.00e+00` over all 64x4 entries.
  `s7_a05` matches `blind_pred` only because both configs select seed 1 as the val-best blind seed
  (deterministic given identical seeding), while averaging 7 vs 5 seeds yields a different ensemble.

  This is a *stronger* test on the *same* frozen artifact, not a metric chosen after seeing results: the
  comparison was computable before the run and its outcome could equally have excluded every candidate.
  The pre-registered rule was under-specified — it should have said "reproduces the stored artifact"
  rather than "reproduces the stored `blind_pred`". Recorded here so the weakness is on the record and
  not discovered by an examiner.

  **Recovered provenance:**
  `run_relation_transformer.py --tag yolo26app_ --seeds 5 --aux_w 0.5 --epochs 120` (blind seed = 1).
  Closes `w2-number-integrity-fixes.md` unresolved #1 (previously inferred from file mtimes only).
  Confirms `--tag yolo26app_` and shows P3 **is** bit-reproducible on this stack.

  **Consequences (branch-1 action, on the strengthened match):** `s5_a05`'s `blind_softmax` is adopted
  for all calibration / operating-point / conformal figures, resolving S4/DEFECT 5. Per rule 4, the
  stored **54/64** remains the reported same-dist number — unchanged, and independently re-derived here.


### 2026-09-06 (later same day) — P3 — WEIGHT PERSISTENCE + ONNX PARITY (continuation of the entry above)

**Not a new configuration.** Same pre-registered command, same recovered provenance (`s5_a05`), run
again for one reason: the entry above recovered the *softmax* but the model **weights** were still
never on disk, so no P3 graph could be exported for deployment. `run_relation_transformer.py` gained
`--save_model` (persists the val-selected blind seed's `state_dict` + ctor config + `argv`).

- **Command (unchanged apart from the two output paths):**
  `run_relation_transformer.py --tag yolo26app_ --seeds 5 --aux_w 0.5 --epochs 120
   --save_model sweep_results/relation_tokens/canonical/p3_blind_relation_head.pt --save_logits <scratch>/p3_repro_for_weights.pt`
- **Why this touch is permitted:** reproduction of an existing recorded result. It can only confirm
  or fail to confirm 54/64; nothing is selected. `logits_yolo26.pt` was not overwritten.
- **Result — bit-exact, third independent reproduction:**

  | tensor | `torch.equal` vs stored | max abs diff |
  |---|---|---|
  | `blind_softmax` | **True** | 0.000e+00 |
  | `ens_softmax` | **True** | 0.000e+00 |
  | `blind_pred` | **True** | 0.000e+00 |
  | `labels` | **True** | 0.000e+00 |

  Blind seed = 1, blind_top1 = **54/64**, unchanged. P3 is bit-reproducible on this stack.

- **Per-seed test spread, now fully visible (all 5 seeds):** 51, **54**, 51, 52, 47 correct
  = 79.7 / **84.4** / 79.7 / 81.2 / 73.4 % top-1. Mean 79.6 %, **spread 11.0 pp**.
  The blind seed is chosen on **val** macro (seed 1 has the highest, 87.5), so the selection is
  legitimate and no test quantity entered it. But the reported 54/64 is the **top** of the seed
  range and 4.8 pp above the seed mean. **Report as "val-selected single seed 54/64; the five seeds
  span 47–54".** Quoting 54/64 without the spread implies a stability the model does not have.

- **ONNX parity gate (new, deployment-side).** The exported
  `deployment/export/p3-relation-head-fp32.onnx` was run over the 64 test clips and compared to the
  stored `blind_softmax`: **max abs softmax diff 2.980e-07, argmax agreement 64/64, ONNX top-1
  54/64**. The deployable graph is the headline predictor, not an approximation of it. This is a
  parity check against an already-computed artifact — it selects nothing — and
  `benchmark-p3-onnx.py` refuses to report latency unless it passes.


<!-- TEMPLATE — copy, fill in, and commit BEFORE running:
### YYYY-MM-DD HH:MM — <pipeline variant>
- Declared headline config: <model, tokens, seeds, epochs, fusion — all fixed on val>
- Command: <exact command line>
- Protocol: <same-dist 64 | k-fold 516 | LCO 516>
- Metric set: brief.md section 4b/4c/4d/4e/4f
- Cost matrix: as declared above (or state the deviation and the reason)
- Reason this touch is permitted: <first touch for this variant | new variant | stated exception>
- Result: <filled in AFTER the run — never edit the fields above after seeing it>
-->

### 2026-09-07 — LCO cluster_seed sweep: integrity check on the cs=2 leg

Test touches: **zero** (all on the pooled 516-clip set).

The cs=2 leg returned numbers byte-identical to cs=0 on every field, including the per-seed arrays.
`docs/code-standards.md` documents a silent-map-fallback failure mode with exactly this signature, so
the leg was audited before being written up rather than reported as a replicate.

| check | result |
|---|---|
| ARI(cs0, cs2) | 0.9952 (`identical_partition = False`) — genuinely different partitions |
| map actually loaded | yes — `[scene_map] overriding views from ...k8cs2.json \| mapped 516/516` |
| folds actually built from it | yes — cs=2 fold sizes 90/154/35/139/21/18/48/11 vs cs=0's 48/139/39/11/90/17/154/18 |
| clips whose fold membership changes | 56 |
| their true labels | **all 56 are class 3 (CarryOverload)** |
| union of the two affected folds | unchanged (39+17 = 35+21 = the same 56 clips) |
| model accuracy on that union | 100 % under both partitions |

**Conclusion: not the bug — but not a replicate either.** The identity is forced by the data: the only
clips cs=2 moves are perfectly-classified clips of one class, reshuffled between two folds whose union
is unchanged, so every per-clip prediction is necessarily identical. cs=2 is a *degenerate*
perturbation of cs=0. The LCO clustering term therefore rests on **two** distinct partitions (cs=0,
cs=1), not three, and the margin is reported as a range **+13.3 to +16.9 pp**.

Method note: the fold-shape comparison was run at `--seeds 1 --epochs 1`, which is enough to print
fold construction without paying for training. Screen any future cs=3+ leg for ARI < 0.99 against
cs=0 *before* spending the ~50 min.

### 2026-09-09 — P3 edge latency measured (parity-gated)

Test touches: **zero**. Latency is measured on dummy tensors; the parity gate re-scores the stored
`blind_softmax` that S20's already-logged touch produced, and consumes no new evaluation.

Parity PASS (max abs softmax diff 2.980e-07, argmax 64/64, ONNX top-1 54/64) before any timing was
printed — `--bench` refuses otherwise. Per decision: detector 1745.0 ms (95.6 %), appearance 79.6 ms,
head 0.46 ms, total 1825.0 ms = 0.55 dec/s on M3 ORT-CPU. Transfer calibrated on P1 (measured on both
machines): projected Jetson **1.13 / 1.64 / 1.71 dec/s vs 3.0 required — FAIL**.

Two measurement bugs found and fixed before believing the sweep: duplicate ORT sessions
(oversubscribed thread pools, appearance 82 -> 182 ms) and input generation inside the timed lambda
(benchmarked `np.random.randn` alongside the graph, error scaling with sz²). Post-fix, sweep and bench
agree to 0.03 ms. Both fixes are commented in `benchmark-p3-onnx.py` so the trap is not re-entered.

### 2026-09-17 — BINARY Safe/Unsafe task — PRE-REGISTRATION (G0), before any binary test touch

Test touches: **zero**. This entry declares the binary evaluation protocol before any binary model is trained.

- **Task:** clip-level Safe (source classes 4-7) vs Unsafe (0-3), all 691 clips. Test split = the paper's
  125 clips (64 U / 61 S), **kept intact**. 8 sub-classes = auxiliary supervision + per-sub-class recall only.
- **Dedupe policy (decided by the thesis author 2026-09-16, recorded here 2026-09-17):**
  headline = `task='binary'`, train `drop_conflict` (drops `0_tr123, 1_tr55, 0_tr136, 0_tr138`), test untouched.
  Also reported beside it: raw `none/none`; sensitivity row = train+test `drop_conflict` (120 test clips).
  Audit: `sweep_results/duplicate_audit_binary.json`.
- **Ceilings printed in every binary table:** hard 124/125 (`0_te23` ≡ `4_te9`, both test, opposite labels);
  practical 121/125 under `none`, 124/125 under the headline policy.
- **Two declared test touches only:** (1) E0 = current flat encoder retrained on the binary recipe;
  (2) the winner of the pre-registered rule below. Selection (seed, epoch, threshold τ, temperature) on val/OOF only.
- **Decision rule for crediting the redesigned encoder E1:** Δ(E1−E0) under LCO-691 > 0 with paired-bootstrap
  CI excluding 0 **and** (E1−S2) under LCO-691 with CI excluding 0. If Δ_rand > 0 but the LCO CI includes 0,
  the write-up must say "scene fitting". Headline metrics: balanced accuracy + macro-F1, with unsafe
  recall/precision, PR-AUC, ROC-AUC, ECE, per-sub-class recall; significance = McNemar exact (own-vs-own),
  bound only against published aggregates.
- **Pre-committed limitation L1:** opened/closed panel pair co-clusters at 0.12 on the background descriptor —
  reported per sub-class regardless of outcome.
- **Train-only gate already measured (no test):** detector-rule baseline acc 0.843 vs majority 0.799,
  safe recall ≈ 0.23 (`laneb-2026-09-16-binary-data-prep.md`).

### 2026-09-17 (late) — ADDENDUM after S1/S2 gates, BEFORE any encoder LCO-691 result exists
- Test touches: **zero** (gates = pooled k-fold-691 / LCO-691 OOF; report
  plans/2026-09-16-binary-safe-unsafe-reframe/reports/laneb-2026-09-17-binary-baseline-gates.md).
- Observed: S2 (scene-only) LCO-691 bal_acc 52.7-52.8%, ROC-AUC 0.53 (≈ chance); S1-logreg LCO-691
  bal_acc 59.2-59.7%. So the pre-registered clause "(E1−S2) under LCO CI excludes 0" is now a weak bar.
- The G0 decision rule is **NOT changed**. Added, labelled as a post-hoc secondary criterion written
  after seeing gate results but before any encoder LCO number: report (E1−S1-logreg) under LCO-691
  with paired CI + McNemar exact b/c; the thesis states whether E1 clears it, and "beats detector-only
  information on unseen cameras" may only be claimed if that CI excludes 0.
- Caveat recorded: pooled-OOF ROC/PR-AUC under LCO mixes per-fold score scales (S1 LCO ROC-AUC 0.39-0.48
  < 0.5 while bal_acc 59%); thresholded metrics (bal_acc, macro-F1) are primary under LCO.
