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
| 2026-09-06 01:01 | `sweep_results/r2/temporal_convlstm.pt` | not stored | **50/64** | **Raw-frame ConvLSTM, the arm that is directly comparable to ref [17]'s no-bbox row. Found 2026-10-11, logged retroactively.** Only seed-averaged `Pt` is stored, so `blind_pred` cannot be recovered; ensemble argmax = 50/64 top-1, paper-macro (mean per-class sensitivity) 75.64%, per class 27/32, **2/11**, 13/13, 8/8. Trained with `device='mps'` (the module default at the time) on macOS 14, where AdamW's `addcmul_` is silently wrong (`chuong-03-phuong-phap.txt:484-487`) — that is the hypothesis G21 tests. The file is **gitignored** (`.gitignore:11`) and is **cited nowhere in `report/vn/`**, so no published number depends on it. |

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

### 2026-10-10 — DETECTOR-GENERATION CONTROL, FINE-TUNED vs FINE-TUNED (G20) — PRE-REGISTRATION

**Written before any number exists.** Nothing below a "Result" line existed when the commands were
launched. Test touches: **zero** — both arms are pooled-OOF (LCO-691 / LSO-1100), the same basis the
2026-09-07 and 2026-09-17 entries above declare as touch-free.

**Why this runs after the 2026-10-05 number freeze.** The thesis author directed it on 2026-10-10
after being shown that (a) YOLO26 was already evaluated twice and neither result is an upgrade, and
(b) the freeze had passed. Recorded as the author's decision, not an implicit one.

#### Prior state of evidence — all pre-existing, unchanged by this entry

| run | date | detectors compared | lane | result |
|---|---|---|---|---|
| **G17** (pre-registered 2026-09-29) | 29/09 | `yolo11s.pt` vs `yolo26s.pt`, both **COCO-pretrained** | ISB headline, LSO-1100 | ESTA 83.8% → 81.6%, **−2.2 pp**, CI [−1.7,+6.0] contains 0, McNemar b=75 c=43 p=0.00412 |
| **MT3** (not pre-registered; **its detector mAP and 64-clip McNemar never published** — but see fourth fact below, its *token set* is published) | 29/08 | `detector_best.pt` vs `yolo26_best.pt`, both **fine-tuned**, identical recipe | P1 4-class, 64 clips + LCO-516 | detector mAP@50 0.969 vs 0.962; downstream McNemar **p=1.0000** (61/64 identical); LCO 80.4% vs 77.1% |

Third fact, established 2026-10-10 from `sweep_results/relation_tokens/build_bin8_2026-09-16.log`
line 1: the Unsafe-Net lane `tokens_bin8_*` — the **E3 headline-encoder lane**, carrying the
protocol- inflation measurement and gates G10–G13, G15, G16 — was built with the **fine-tuned
YOLO26** (`yolo26_best.pt`, `classes=[0..7]`, 31,873 train + 6,774 test detections). There is **no
fine-tuned YOLOv11s arm on that lane**. So the detector comparison the thesis never ran is the one
on the lane where the detector actually feeds the decision. *Reading note on that quote.*
`classes=[0..7]` is **not** a detector-class filter. It is the list of source behaviour-class
folders that supply the clip labels (`UNSAFE = [0, 1, 2, 3]`, `build_relation_tokens.py:44`). The
entity loop applies no detector-class filter at all; geometry is computed only for `c in (0, 1)` and
zeroed otherwise (`:226`).

**Fourth fact, established 2026-10-10 while arm B was still extracting — it corrects the MT3 row
above.** That row read "never published in the thesis". Too strong. MT3's *detector mAP table* and
its *64-clip McNemar* are indeed unpublished, but **MT3's token set is the published headline
artifact of lane C.** Section 6.12.2 reports k-fold on `tokens_yolo26app_*` as that lane's headline
and reports a same-protocol fine-tuned-YOLOv11s control on `tokens_yoloapp_*` immediately beside it:

| lane C, k-fold 5x10 on 516 pooled clips | fine-tuned YOLO26 (headline) | fine-tuned YOLOv11s (control) |
|---|---|---|
| fold-mean top-1 | 80.8% ± 4.0 | 80.7% ± 3.6 |
| pooled correct | 421/516 | 421/516 — *identical* |
| macro | 84.2% | 85.1% |
| margin over its own baseline | +3.7 pp, t=1.93, **p=0.0600** | +3.6 pp, t=1.95, **p=0.0574** |

Sources: `sweep_results/relation_tokens/canonical/kfold_repeats10_yolo26app_HEADLINE.log:53-57` and
`sweep_results/relation_tokens/canonical/kfold_repeats10_yoloapp_YOLO11.log:53-57`.

So **a fine-tuned-vs-fine-tuned detector comparison is already in the thesis**, and it already
concludes "gan nhu trung" (near-identical). G20 arm A is therefore the **second** such comparison,
not the first. What arm A adds is a different lane (687 pooled binary clips vs 516 four-class), a
different protocol (LCO-691 pooled OOF vs 5x10 k-fold) and a **cluster-robust CI the lane-C pair
never had**. Section 6.14 must be written to say exactly that and must not claim novelty.

**Fifth fact, closed by measurement rather than by log.** `tokens_yolo26app_*` has **no build log in
the repo**; the only attestation of how it was built was the MT3 report itself. Counted directly
from the tensors (`mask.sum()`): train **26,438**, test **3,390** detections — both exactly MT3's
stated figures. The fine-tuned-YOLOv11s pair gives 25,891 / **3,267**, and 3,267 is exactly the last
line of `sweep_results/relation_tokens/build_yoloapp.log`. The claim is now checkable against the
artifact instead of resting on a secondary report. This belongs in 7.7.1 (resolved by
re-measurement), not 7.7.2.

**Sixth fact — the detection-volume mechanism replicates across both in-domain lanes.**

| lane | unit | fine-tuned YOLOv11s | fine-tuned YOLO26 | ratio |
|---|---|---|---|---|
| Unsafe-Net binary (`v11ft` vs `bin8`) | 566 train clips | 30,781 | 31,873 | 1.035 |
| Unsafe-Net binary | 125 test clips | 6,307 | 6,774 | 1.074 |
| lane C 4-class (`yoloapp` vs `yolo26app`) | 452 train clips | 25,891 | 26,438 | 1.021 |
| lane C 4-class | 64 test clips | 3,267 | 3,390 | 1.038 |
| iSafetyBench, **COCO-pretrained** (G17) | 1,100 clips | 52,997 | 45,512 | **0.859** |

In domain, fine-tuned on the eight factory classes, the newer detector is the **more prolific** one
on **both** lanes. Out of domain, COCO-pretrained, it is the **sparser** one. One lane could be
noise; two lanes in the same direction, against a pretrained pair in the opposite direction, is a
mechanism.

**Seventh fact — G20's fairness premise is now artifact-verified, and it corrects two published
claims.** Both detector checkpoints carry an ultralytics-written `train_args`/`train_metrics` block.
Read 2026-10-10:

| field (library-written in the `.pt`) | `detector_best.pt` | `yolo26_best.pt` |
|---|---|---|
| `date` / ultralytics | 2026-08-09T07:23:09Z / 8.4.116 | 2026-08-24T16:13:29Z / 8.4.127 |
| `train_args.project` | **`/content`**`/runs` | **`/kaggle/working`**`/runs26` |
| `train_args.data` | `/content/ds/data.yaml` | `/kaggle/working/data.yaml` |
| init from / `name` | `stage1_frozen/…/best.pt` / `stage2_finetune` | `runs26/s1/…/best.pt` / `s2` |
| optimizer / lr0 / imgsz | SGD / 0.001 / 800 | SGD / 0.001 / 800 |
| epochs / seed / device / batch | 100 / 0 / 0 / 16 | 100 / 0 / 0 / 16 |
| patience / freeze / pretrained | 30 / 0 / True | 30 / 0 / True |
| `train_metrics` mAP@50 / @50-95 | 0.9670 / 0.8811 | 0.9622 / 0.8780 |
| `train_metrics` P / R | 0.9340 / 0.9674 | 0.9435 / 0.9244 |

Three consequences. **(1)** The thesis attributed *both* weights to Kaggle; `detector_best.pt` is
**Google Colab**. Corrected in §4.5.5 and §7.7.1(a). **(2)** "lời tự khai, không phải bằng chứng"
(self-report, not evidence) was too pessimistic: the configuration and platform are attested by two
independent artifacts — the embedded metadata, and the **tracked** notebooks
`colab/train_detector_colab.ipynb` and `colab/train_yolo26_detector_kaggle.ipynb`, which match it
field for field including the two stage names. What is still absent is only the **per-epoch
history**, plus the fact that neither `.pt` is tracked. **(3)** MT3's "one-thing-changed" fairness
claim — the premise G20 arm A inherits — is now verifiable from artifacts, not from narrative: the
two weights agree on every hyper-parameter above. Caveat recorded with it: the embedded
`train_metrics` are **not** the Table 4.1 measurement. `detector_best.pt`'s P/R agree to four
decimals (0.9340/0.9674 embedded vs 0.9339/0.9673 re-measured) but the mAPs differ (0.9670
vs 0.9695; 0.8811 vs 0.8939); the same pattern holds for the second weight, whose embedded
mAP@50 0.9622 matches MT3's published 0.962 while its embedded mAP@50-95 0.8780 does not match MT3's
0.872. Library-version difference at AP integration (8.4.116 training vs 8.4.27 re-validation) is
recorded as a possibility, not a conclusion.

#### Pre-run measurement, recorded before either arm exists

`scratchpad/probe-ft-detector-on-isb.py`, 24 iSafetyBench clips (seed 0) / 384 decoded frames,
imgsz 640, conf 0.25, N_MAX 8:

| detector | nc | detections | frames with >=1 | mean conf |
|---|---|---|---|---|
| `yolo11s.pt` (pretrained) | 80 | 1,415 | 370/384 = 96.4% | 0.611 |
| `yolo26s.pt` (pretrained) | 80 | 1,245 | 357/384 = 93.0% | 0.632 |
| `detector_best.pt` (v11s fine-tuned) | 8 | 332 | 178/384 = 46.4% | 0.574 |
| `yolo26_best.pt` (26s fine-tuned) | 8 | 311 | 210/384 = 54.7% | 0.616 |

The fine-tuned detectors fire at **~23% of pretrained detection volume** on iSafetyBench and leave
over 45% of frames with no entity. **Arm B below is therefore a detection-starved, out-of-domain
regime — known in advance here, not discovered after the fact.**

#### Arm A — in-domain, Unsafe-Net LCO-691 (primary)

The missing fine-tuned YOLOv11s arm on the lane that already runs fine-tuned YOLO26.

```
# build the v11-fine-tuned arm; recipe copied field-for-field from build_bin8_2026-09-16.log
.venv/bin/python build_relation_tokens.py --appearance --appearance_backbone yolo \
    --detector sweep_results/entity_detector/detector_best.pt \
    --classes 0,1,2,3,4,5,6,7 --out_tag v11ft_ --device mps
# score it with the published E3 LCO command, tag swapped and nothing else
.venv/bin/python run_kfold_relation.py --binary --arch E3 --lco --tag v11ft_ \
    --dedupe drop_conflict --dedupe_task binary \
    --scene_map sweep_results/zone_prior/clip_scene_map_691.json
```

Baseline for the pairing is the already-published E3 run on `--tag bin8_`: pooled 687 clips
(unsafe 512), 8 groups, params 18,044, coord T1, **bal_acc 69.6%** / macro_f1 70.4%
(`sweep_results/relation_tokens/tier3_lco_logs/E3.log`,
`lco_oof_softmax_bin8_binary_E3_drop_conflict.pt`). It is **not** re-run and **not** overwritten.

#### Arm B — out-of-domain, iSafetyBench LSO-1100 (the G17 upgrade as asked)

Requires one code change, declared here before it is written:
`build-isafetybench-clip-features.py:126` maps detector class ids through `ISB.coco_to_group()`, an
**80-entry COCO LUT**. The fine-tuned detectors have 8 factory classes (`Person_Authorized,
Person_Unauthorized, Safe_Walkway, Restricted_Zone, Panel_Closed, Panel_Opened, Forklift_Safe_Load,
Forklift_Overload`), so the LUT would mislabel 7 of 8. The fix is an **identity** class map used for
both arms: class id 0..7 passes through unchanged. Both arms get identical treatment, so no semantic
mapping is invented and nothing is tuned. Arm B's class space is therefore **not** the COCO-group
space of G17.

*Precondition verified before launch, 2026-10-10.* The identity map is only safe if the model's
class vocabulary admits ids 0..7 without folding. `model_entity_spatiotemporal.py:22` sets
`N_ENTITY = 8` and line 70 does `geom[...,0].long().clamp(0, N_ENTITY - 1)`. Both fine-tuned
detectors have exactly 8 classes, so each id lands on its own embedding row and the clamp never
fires; no model change is needed, and the two arms share one vocabulary size with G17's coarse
groups even though the *meaning* of each row differs. Had either detector carried more than 8
classes, every id above 7 would have been silently folded into row 7 — which is why this was
checked rather than assumed.

```
.venv/bin/python build-isafetybench-clip-features.py \
    --detector sweep_results/entity_detector/detector_best.pt --class_map identity --out_tag isbv11ft_
.venv/bin/python build-isafetybench-clip-features.py \
    --detector sweep_results/entity_detector/yolo26_best.pt  --class_map identity --out_tag isb26ft_
# shard protocol resolved from logs/r48-lso-ESTAy26-shard{0..3}.log: 4 shards, 10 LSO folds,
# ESTA, 3 seeds, folds dealt f[shard::4]; then the shards are merged by the same finalise() the
# unsharded path uses. Repeat the block with --tag_suffix _y26ft and feats_isb26ft_all.pt.
for k in 0 1 2 3; do .venv/bin/python run-esta-lso.py --model ESTA \
    --feats sweep_results/relation_tokens/feats_isbv11ft_all.pt \
    --tag_suffix _v11ft --shard $k --nshards 4 & done; wait
.venv/bin/python tools/merge-esta-shards.py --model ESTA --nshards 4 --tag_suffix _v11ft
```

#### Decision rule — fixed in advance

1. **Headline clause, binding (mirrors G17's).** The thesis's headline detector, headline encoder
   (E3), every headline accuracy number and every published gate verdict stay **exactly as they
   are, whatever this run returns.** Renaming a model after seeing a cross-validated result is
   selection on the evaluation data. Arm A and arm B enter the thesis only as a **disclosed
   post-freeze control** in their own subsection — never replacing a frozen number, never averaged
   or ranked against one.
2. **Reporting.** Balanced-accuracy margin + cluster bootstrap 95% CI + McNemar exact b/c, resampled
   at the **group / source** level, per section 3.7.4. Clip-level CIs are not reported as the
   verdict.
3. **"YOLO26 is an upgrade" may be claimed only if** the margin is positive **and** the CI excludes
   0, **and** the claim is scoped to the lane it was measured on.
4. **If the CI contains 0** the recorded conclusion is "khong phan biet duoc" (indistinguishable) —
   not "equivalent", not "upgrade".
5. **Arm B's numbers may not be compared to 83.8% / 81.6%.** Different class space, different
   detection regime. Only arm B's own two arms may be compared to each other.
6. If arm A's build produces a detection count outside 0.5x–2.0x of `bin8_`'s 31,873 train
   detections, the arm is reported **with that ratio stated in the same sentence as the margin**,
   because a starved or flooded arm is not a clean one-thing-changed comparison.

#### Amendment, 2026-10-10 — added after arm A's token build launched, before any score exists

7. **Code-parity re-run of the baseline arm, mandatory.** The published baseline OOF artifact
   `enc_E3_lco-691_headline-drop_conflict_oof.json` was written **2026-09-19 23:33**, ten days
   before `9515ee1 refactor(eval): route every paired comparison through the shared statistics
   module` (2026-09-29 23:22) rewrote the LCO driver's pooling, dedupe and scorecard path
   (`load_pooled_binary`, `report_binary`, `evaluate_suite`; +263 lines in that one file). Pairing
   the 19/09 artifact against an arm scored by today's tree would confound "the detector changed"
   with "the evaluation code changed", and that confound is not separable after the fact. So arm A
   scores **both** detectors with the current tree: the YOLO26-ft baseline is re-scored from the
   already-built `tokens_bin8_*.pt` under `--tag bin8_`, and the margin is computed between the two
   current-code artifacts. No token file is rebuilt and no test touch is consumed — the re-run is
   the same pooled-OOF screening protocol as the original.
   - The published **69.6%** bal_acc is then reported beside the re-run as a **reproduction
     check**. If the two differ, the difference is a property of the refactor, is disclosed as
     such in its own sentence, and the **published number stays the published number** — rule 1
     already covers that, and this clause does not create an exception to it.
   - `enc_bin8_E3_...` and `enc_v11ft_E3_...` are distinct filenames from the 19/09
     `enc_E3_...`, so neither re-run can overwrite the JSON the thesis cites.

**Executed 2026-10-10, and one disclosure.** The re-run reproduced the published baseline exactly:
`balanced_acc=69.6% macro_f1=70.4% unsafe_recall=87.7% unsafe_precision=84.1% PR-AUC=0.856
ROC-AUC=0.742`, all eight held-out-group accuracies and every per-sub-class recall byte-identical
to `tier3_lco_logs/E3.log`. The only differing lines in the whole log are the protocol tag
(`lco` -> `lco-691`) and an added section title. Commit `9515ee1` is therefore **behaviour-neutral
on this path**, and the published LCO-691 E3 number regenerates from the stored tokens under
today's tree.

*Disclosure.* The clash check above covered the JSON name but **not** the tensor name: the
published OOF tensor already carried the tag, so `lco_oof_softmax_bin8_binary_E3_drop_conflict.pt`
was **overwritten** by this re-run, and the file is gitignored, so there was no copy to restore.
It is provably immaterial, and the proof does not depend on the overwritten file: the untouched
19/09 `enc_E3_lco-691_headline-drop_conflict_oof.json` holds `prob_unsafe` for all 687 clips, and
against the regenerated tensor the max absolute difference is **0.0**, with **0/687** predicted
labels and 0 true labels differing. The regenerated artifact is content-identical to the one it
replaced. Remaining G20 output names (`v11ft_`, `isbv11ft_`, `isb26ft_`, `_v11ft`, `_y26ft`) were
each checked against the directory before launch and none collides.

#### Result — arm A

**Ran 2026-10-10. Both arms scored by today's tree, per clause 7.**

Token build, `logs/g20-build-v11ft-2026-10-10.log`, same recipe fields as `build_bin8`
(`appearance=True backbone=yolo corrupt=None classes=[0..7]`, `feat=74` in both):

| split | YOLOv11s-ft (new) | YOLO26-ft (`bin8_`, incumbent) | ratio |
|---|---|---|---|
| train, 566 clips | 30,781 detections | 31,873 | 0.966 |
| test, 125 clips | 6,307 detections | 6,774 | 0.931 |

Both ratios sit well inside the 0.5x-2.0x band, so decision rule 6's special reporting is not
triggered. The direction is still worth stating: **in domain the newer detector is the more
prolific one**, the opposite of the out-of-domain pretrained pair in G17, where YOLO26s produced
14.1% *fewer* detections than YOLOv11s.

Score, `logs/g20-score-v11ft-2026-10-10.log` (687 pooled clips, 8 held-out groups, params 18,044):
`balanced_acc=61.8% macro_f1=60.9% unsafe_recall=75.0% unsafe_precision=81.0% PR-AUC=0.856
ROC-AUC=0.699`.

Paired comparison, `logs/g20-compare-armA-2026-10-10.log`:

```
E3 @ YOLO26-ft 69.6%  -  E3 @ YOLOv11s-ft 61.8%  =  +7.8 pp
[cluster bootstrap over 8 groups, 5000/5000 valid draws]  95% CI [-1.5, +15.2]  INCLUDES 0
[McNemar exact] b=98 c=28 n_discordant=126 p=2.69e-10
[clip-level bootstrap -- WRONG UNIT, disclosure only] 95% CI [+4.3, +11.4]
```

**Recorded conclusion, by rule 4: "khong phan biet duoc" at the group level.** The point estimate
favours the **incumbent** YOLO26-ft by 7.8 pp and the exact McNemar is overwhelming at clip level
(p = 2.7e-10, 126 discordant), but the pre-registered verdict unit is the scene group, there are
only 8 of them, and that CI crosses 0. Rule 3 is therefore not satisfied in either direction and
no "upgrade" claim may be made from arm A. The clip-level interval excludes 0 and is recorded here
only as disclosure; section 3.7.4 forbids reading it as the verdict.

**The wide interval has a mechanism, not just 8 groups.** Per held-out group:

| group | n | YOLO26-ft | YOLOv11s-ft | favours |
|---|---|---|---|---|
| 0 | 22 | 77.3% | 100.0% | YOLOv11s +22.7 |
| 1 | 187 | 75.4% | 38.5% | **YOLO26 +36.9** |
| 2 | 58 | 100.0% | 100.0% | tie |
| 3 | 170 | 81.8% | 81.8% | tie |
| 4 | 121 | 51.2% | 43.0% | YOLO26 +8.3 |
| 5 | 101 | 93.1% | 97.0% | YOLOv11s +4.0 ‡ |
| 6 | 14 | 100.0% | 100.0% | tie |
| 7 | 14 | 100.0% | 100.0% | tie |

**Correction, 2026-10-10 (re-verified against both OOF JSONs row by row).** **Four** groups are
identical (2, 3, 6, 7), **two** favour YOLOv11s-ft (group 0 +22.7, group 5 +4.0), **two** favour
YOLO26-ft (group 1 +36.9, group 4 +8.3). The two margin columns above were previously the
difference of the *rounded* values; they are now the rounded *difference*. Columns are **plain
per-group accuracy, not balanced** — ‡ group 5 holds one class only (101/101 unsafe), so balanced
accuracy is undefined there; pooled plain accuracy is 78.5% vs 68.3% and must not be substituted
for the headline balanced pair 69.6% vs 61.8%. In clip terms the pooled margin is **+70 clips of
687**, of which group 1 alone contributes **+69** (98.6%), group 4 +10, groups 0 and 5 −5 and −4.
So almost the whole margin comes from group 1 — the
largest group — where YOLOv11s-ft collapses to 38.5%. A cluster bootstrap resamples groups, so
draws that omit group 1 give a margin near 0: that is exactly why the group-level CI crosses 0
while the clip-level McNemar is overwhelming. **The in-domain advantage of fine-tuned YOLO26 is
concentrated in one scene group, not spread across the corpus.** Any "upgrade" verdict built on it
would be a statement about one context phrased as a statement about the corpus.

Two consequences that do follow, and neither needs the CI:

1. **"Upgrade the thesis to YOLO26" is not an available action on this lane — it is already the
   state of the lane.** `build_bin8_2026-09-16.log` has run on `yolo26_best.pt` since 16/09, so
   every E3 number in the thesis, the protocol-inflation measurement and gates G10-G13, G15, G16
   already rest on the fine-tuned newer detector.
2. **The arm that was actually missing loses.** Substituting fine-tuned YOLOv11s costs 7.8 pp of
   point estimate, and the per-sub-class pattern shows where: `2:OpenedPanel` recall falls to
   27.5% and `6:ClosedPanel` to 6.2%, i.e. the panel-state distinction degrades most.

**Code parity confirmed, not assumed** (clause 7): the `bin8_` re-score reproduced the published
`69.6% / 70.4% / 87.7% / 84.1% / 0.856 / 0.742` exactly, all eight group accuracies and every
per-sub-class recall byte-identical to `tier3_lco_logs/E3.log`. So the +7.8 pp is attributable to
the detector alone, not to `9515ee1`.

#### Result — arm B

**Ran 2026-10-10.** Both extractions were rebuilt from frames today with the identity class map
declared above; both LSO runs were scored by today's tree, per clause 7.

Token build, `logs/g20-build-isbv11ft-2026-10-10.log` and `logs/g20-build-isb26ft-2026-10-10.log`
(1,100 clips, T=16 sampled frames, `n_max=8`, imgsz 640, conf 0.25, `--class_map identity`):

| extraction | detections | s/clip | wall | labels | sources |
|---|---|---|---|---|---|
| YOLOv11s-ft, `feats_isbv11ft_all.pt` | 13,895 | 1.64 | 30.2 min | 420 haz / 680 norm | 151 |
| YOLO26-ft, `feats_isb26ft_all.pt` | 13,108 | 1.62 | 29.7 min | 420 haz / 680 norm | 151 |

Ratio 13,108 / 13,895 = **0.943**, inside the 0.5x-2.0x band, so rule 6's special reporting is not
triggered. Label counts and source count are identical across arms, which is the drift guard: the
corpus did not move between the two extractions.

**One-thing-changed control, `logs/g20-armB-wholeframe-control.log`.** This is the same control
section 4.3.3 ran for G17, repeated for arm B because arm B also changes the class map:

```
clip order identical : True
labels identical     : True
max |diff| wholeframe: 0.0
```

The whole-frame branch never passes through the detector, so it must be bit-identical between arms;
it is (0.0, byte-identical). Both `meta` dicts agree on every field — `detector_nc` 8, `class_map`
identity, `imgsz` 640, `conf` 0.25, `T` 16, `n_max` 8 — and differ only in the detector path. So
the detector is the only thing that changed.

**Class space verified, not assumed.** The `names` map was read out of both checkpoints. Both
`detector_best.pt` and `yolo26_best.pt` carry the same 8 names in the same order (0
Person_Authorized, 1 Person_Unauthorized, 2 Safe_Walkway, 3 Restricted_Zone, 4 Panel_Closed, 5
Panel_Opened, 6 Forklift_Safe_Load, 7 Forklift_Overload), so the identity map means the same thing
in both arms and no id lands on a different embedding row between them.

Score, `logs/g20-lso-ESTA-v11ft-merged.log` (1,100 clips, 10 held-out LSO folds, merged from 4
shards): `balanced_acc=75.9% macro_f1=74.0% unsafe_recall=82.4% unsafe_precision=62.5%
PR-AUC=0.714 ROC-AUC=0.837`.

Score, `logs/g20-lso-ESTA-26ft-merged.log` (same protocol): `balanced_acc=78.1%
macro_f1=75.9% unsafe_recall=86.7% unsafe_precision=63.7% PR-AUC=0.642 ROC-AUC=0.825`.

Paired comparison, `logs/g20-compare-armB-2026-10-10.log`:

```
ESTA @ YOLO26-ft 78.1%  -  ESTA @ YOLOv11s-ft 75.9%  =  +2.2 pp
[cluster bootstrap over 151 sources, 5000/5000 valid draws]  95% CI [-3.5, +8.8]  INCLUDES 0
[McNemar exact] b=138 c=119 n_discordant=257 p=0.261
[clip-level bootstrap -- WRONG UNIT, disclosure only] 95% CI [-0.5, +5.1]  ALSO includes 0
```

**Recorded conclusion, by rule 4: "khong phan biet duoc" at the source level — and here the null is
strong, not merely unresolved.** Arm A's CI crossed 0 with only 8 clusters, which leaves
"underpowered" as a live alternative explanation. Arm B has **151** clusters, 5000/5000 valid
bootstrap draws, and the null survives anyway: the cluster CI includes 0, the exact McNemar is
nowhere near significance (p = 0.261, 257 discordant clips, b = 138 vs c = 119), and the clip-level
interval — the one that *excluded* 0 in arm A — **also includes 0** here. Both units agree. No
"upgrade" claim may be made from arm B, and unlike arm A no appeal to low cluster count is
available to explain the result away.

**The sign of the margin is decided by two sources that nearly cancel.** The pooled margin is
**+19 clips of 1,100**. Per source: **32 favour YOLO26-ft, 35 favour YOLOv11s-ft, 84 tie.** More
sources favour the *older* detector, and the pooled margin is positive anyway, because one source
carries it:

| source | n | 26ft correct | v11ft correct | margin |
|---|---|---|---|---|
| `0zQ9fI30v1E` | 31 | 27 | 1 | **+26** |
| `1tuWxs8fUOk` | 102 | 54 | 73 | **-19** |
| `H4sC0QCkG7I` | 17 | 17 | 5 | +12 |
| `bxENbC-SPCw` | 14 | 14 | 7 | +7 |
| `0a-WuC1Xdl4` | 9 | 2 | 8 | -6 |

The largest single source contributes +26 — **136.8% of the entire pooled margin** — and the second
largest cancels most of it at -19. Drop `0zQ9fI30v1E` alone and the margin **changes sign** to -7.
Resampling 151 sources is exactly the procedure that exposes this, which is why the cluster CI spans
0 in both directions. Arm A and arm B fail the CI for *opposite* reasons: arm A because its margin
is concentrated in one of 8 clusters, arm B because its margin is a near-wash between two of 151.

**The two detectors also swap order depending on which metric is read.** YOLO26-ft wins balanced
accuracy (78.1% vs 75.9%) but **loses both ranking metrics**: PR-AUC 0.642 vs 0.714, ROC-AUC 0.825
vs 0.837. Counting sources also favours YOLOv11s-ft, 35 to 32. Three of five views favour the older
detector. Recorded because any one-number summary of arm B would be a choice of metric rather than
a finding.

**What the +2.2 pp is, structurally.** It is 18 unsafe clips recovered at essentially no specificity
cost: TP 346 -> 364, FN 74 -> 56, while TN moves 472 -> 473 and FP 208 -> 207. Unsafe recall goes
82.4% -> 86.7% and specificity is flat (69.4% vs 69.6%); unsafe precision is ~63% in both arms. So
out of domain the newer detector is marginally more *sensitive*, not better calibrated and not more
precise. On a safety task that is the useful direction, which is why it is recorded — but it is 18
clips of 1,100 and it is not significant on any unit.

**Rule 5 restated, because the temptation is real.** Arm B's two numbers **may not be compared to
G17's 83.8% / 81.6%.** G17 ran the same encoder on the same 1,100 clips under the same LSO
protocol, which makes the comparison look legitimate, and it is not: G17's entity tokens were
labelled in the **80-entry COCO group space** by `ISB.coco_to_group()`, arm B's in the **8-class
factory space** by the identity map. The embedding rows carry different meanings, and the detection
regime differs by a factor of four (below). Only arm B's own two arms may be compared to each other.

**The detection regime out of domain, measured.** The reason arm B is not a re-run of G17 with a
better detector is that out of domain the fine-tuned detectors are detection-starved. Anchored to
the thesis's own published metric — the share of frames with at least one detection, computed from
the `mask` field, published in section 4.3.3 as 90.2% and 87.9%:

| extraction | class space | detections | frames with >=1 det | slots firing |
|---|---|---|---|---|
| YOLOv11s COCO-pretrained | 8 COCO groups | 52,997 | 15,874 (90.2%) | 8/8 |
| YOLO26s COCO-pretrained | 8 COCO groups | 45,512 | 15,465 (87.9%) | 8/8 |
| YOLOv11s fine-tuned | 8 factory | 13,895 | 8,117 (46.1%) | 4/8 |
| YOLO26 fine-tuned | 8 factory | 13,108 | 8,518 (48.4%) | 4/8 ‡ |

The 90.2% and 87.9% rows reproduce the published figures exactly (15,874 and 15,465 of 17,600
frames), so the new rows are measured on the same ruler. ‡ YOLO26-ft also emits 2 Panel_Closed and
1 Panel_Opened — 3 detections of 13,108, 0.02% — so strictly six slots are touched but four carry
99.98%; YOLOv11s-ft is exactly 4/8 (ids 0, 1, 6, 7 = 13,895 = 100.0%).

**Truncation is excluded by measurement, not by argument.** `n_max=8` could in principle be hiding
classes behind the top-8 confidence cut. In domain, 221 of 11,056 frames saturate for YOLOv11s-ft
(2.0%) and 277 (2.5%) for YOLO26-ft, so 97.5-98% of frames have spare slots and a zone detection
above conf 0.25 would have been kept. Out of domain YOLOv11s-ft saturates in 28 of 17,600 frames
(0.2%) while **9,483 frames (53.9%) are empty**. The missing classes are genuinely below threshold,
not truncated away.

**Two classes never fire anywhere, and that is a source-code fact.** Ids 2 Safe_Walkway and 3
Restricted_Zone are zero in both domains and both arms, so the in-domain baseline is 6 of 8, not 8
of 8. The zone signal on this lane does not come from the detector at all:
`build_relation_tokens.py:105-106` reads the `safe_walkway` / `restricted_zone` **annotation
polygons**, and lines 160-166 turn them into `iou_walkway`, `iou_restricted`, `foot_in_walkway` and
`norm_dist`. Those four geometry channels are annotation-supplied, which bounds what "tự động" can
mean in section 5.1.

**The panel classes are the ones that are lost.** In domain Panel_Closed and Panel_Opened are
**24.6%** of YOLOv11s-ft detections and **24.1%** of YOLO26-ft's, and MT3 reports mAP of 0.990
and 0.962 on them. Out of domain they contribute 0 and 3. The two most reliable in-domain classes
contribute essentially nothing out of domain — iSafetyBench footage simply has no factory panels in
it. This is the mechanism behind the 46.1% / 48.4% detected-frame collapse.

**Direction of the volume effect is set by domain, not by training regime.**

| comparison | 26 / v11 detections | ratio | denser |
|---|---|---|---|
| in domain, fine-tuned | 38,647 / 37,088 | 1.042 | YOLO26 |
| out of domain, fine-tuned | 13,108 / 13,895 | 0.943 | YOLOv11s |
| out of domain, pretrained | 45,512 / 52,997 | 0.859 | YOLOv11s |

Out of domain the newer detector is sparser in **both** regimes; only in domain does it become the
more prolific one. So arm A's "the newer detector detects more" and G17's "the newer detector
detects 14.1% less" are not in conflict — they are the same detector pair measured in two domains.

**One counter-intuitive detail, recorded because it is not what the volume number suggests.**
YOLO26-ft produces fewer detections than YOLOv11s-ft (13,108 vs 13,895) yet covers **more** frames
(48.4% vs 46.1%) and is more confident (mean conf 0.619 vs 0.556). Composition shifts hard:
Person_Unauthorized goes 77.3% -> 90.7% of all detections, Person_Authorized 9.5% -> 3.0%, and
Forklift_Safe_Load 911 -> 166. Fewer, more confident, more spread out, and more collapsed onto one
class.

**Terminology note, to stop a misreading of the recipe line.** `classes=[0..7]` in the arm A token
build is **not** a detector-class filter. It is the list of source behaviour-class folders that
supply clip labels (`UNSAFE = [0, 1, 2, 3]`, `build_relation_tokens.py:44`). The entity loop applies
no detector-class filter at all; geometry is computed only for `c in (0, 1)` and zeroed otherwise
(`:226`).

---

### 2026-10-11 — RAW-FRAME ConvLSTM, DEVICE CONTROL (G21) + BINARY NP-125 (G25) — PRE-REGISTERED

Pre-registration written in full **before any new number existed**, at
`plans/261011-1502-baseline-giao-thuc-bai-bao-goc-unsafe-net/tien-dang-ky-g21-g25.md` (base commit
4598bc5). Read that file for the decision rules; this entry records only the touch itself.

**Why a touch is spent at all.** Ref [17]'s reported no-bbox row (57/64 top-1, 85.78% macro) is a
**test-set** number. Any comparison against it is by construction a test-set comparison; there is no
val-only route to it. Budget: 2 pre-registered, **1 consumed here** (batched — both runs scored in a
single pass), 1 remaining.

**Exact commands** (both detached, logs persisted this time):

```
.venv/bin/python3 -m unsafenet.convlstm --task unsafe4 --device cpu --seeds 5 --epochs 18 \
    --name convlstm_cpu           > sweep_results/r2/convlstm-cpu-unsafe4-2026-10-11.log
.venv/bin/python3 -m unsafenet.convlstm --task binary --policy drop_conflict --device cpu \
    --seeds 5 --epochs 18 --name convlstm_binary > sweep_results/r2/convlstm-cpu-binary-2026-10-11.log
.venv/bin/python3 run_convlstm_paper_baseline.py     # reads artifacts only, trains nothing
```

**One variable.** `device` mps -> cpu. `px=112, t=16, seeds=5, epochs=18, bs=8, lr=3e-4`, the
`blind_split(val_frac=0.18, seed=0)` split and the 64 test clips are all identical to the 2026-09-06
run; the split was verified to reproduce that run's `val_names`, `test_names` and `yt` exactly before
launching. No early stopping on test: last epoch is taken, selection is on val only.

**Declared headline configuration, fixed in advance.** Seed-averaged ensemble argmax over 5 seeds,
last epoch, `policy='none'` for the 4-class arm (so it reproduces the frozen 64-clip protocol) and
train `drop_conflict` + test `keep` for the binary arm.

**Metric set expected, fixed in advance.** 4-class arm: top-1, paper-macro (mean per-class
sensitivity — *not* sklearn macro-F1), per-class sensitivity, and the feasible-discordant-split
McNemar interval against 57/64 and 62/64 (ref [17] publishes no per-clip labels, so an exact McNemar
is not computable; the interval over all feasible 2x2 tables is). Binary arm NP-125: balanced
accuracy primary, Wilson CI, plus the two mandatory controls — scene-only majority and constant
majority class — with exact McNemar against each. NP-64 is reported as a **slice**, unsafe recall
only; specificity, BA and PR-AUC are undefined there and must not be printed.

**Two bounds that are properties of the data, not of any model.** NP-125 majority floor 64/125 =
51.20%; **hard ceiling 124/125 = 99.20%**, because `0_te23.mp4` and `4_te9.mp4` are byte-identical
with opposite binary labels and both sit inside test (`unsafenet/dedupe.py`).

**Pre-existing contamination that this entry does not fix.** Test-set early stopping already
contaminates both lane-C training branches (`chuong-06-danh-gia.txt:1567-1570`). This touch adds no
third contamination path, but it does not retroactively clean the first two.

**TOUCH CONSUMED — results, recorded 2026-10-11 after scoring.** Budget now **2 pre-registered, 1
consumed, 1 remaining**. Both pre-registered gates were adjudicated against the rules above and
**both went against the hypothesis**; neither rule was altered after the numbers appeared.

| arm | result | pre-registered rule | verdict |
|---|---|---|---|
| G21 device control, 4-class/64 | **47/64 = 73.44%**, paper-macro **70.22%**, per class 27/32, 3/11, **9/13**, 8/8 | refuted iff top-1 <= 47/64 | **REFUTED** |
| G25 binary NP-125 | **93/125 = 74.40%**, **BA 74.19%** [66.10, 81.24] | BA >= 66.2% **and** McNemar vs scene-only p < 0.05 | **FAILS** (BA ok, p = 0.4049) |

G21's two falsifiable predictions: c1 recall >= 6/11 -> **3/11, failed**; seed spread not wider ->
cpu 80.56-82.25 vs mps 78.41-80.18, **held**. The device hypothesis is dead: the MPS `addcmul_` bug
does not explain the gap to 85.78%, and fixing the prime suspect **widened** the gap to 15.56 pp.

**The incidental finding is larger than either gate.** Changing only the compute device moved val
macro **+2.78 pp** (76.39 -> 79.17) while **losing 3 test clips**, and moved one class's sensitivity
by **30.77 pp** (c2 OpenedPanel 13/13 -> 9/13). Two consequences, both binding on this log's
purpose: the 83-clip val slice does not predict the 64-clip test set, so model selection on it is
blind; and **no single-run per-class row is comparable to ref [17]'s**, because a pure nuisance
variable moves a per-class cell further than any gap being explained. The 2026-09-06 50/64 sits
inside that nuisance band and must not be used as a headline.

**G25's binary controls, on `clip_scene_map_691.json`** (KMeans on each clip's temporal-median static
background — classifier-independent; `clip_view_map*.json` was NOT used, it is KMeans on the same
appearance features used for classification and therefore circular):

| control | top-1 | BA | McNemar vs ConvLSTM |
|---|---|---|---|
| scene-only, majority label from **train** | 88/125 = 70.40% | 69.94% | b=14, c=9, **p = 0.4049** |
| scene-only **oracle** (uses test labels -> upper bound, not a baseline) | 91/125 = 72.80% | 72.44% | b=11, c=9, p = 0.8238 |
| constant majority class | 64/125 = 51.20% | 50.00% | b=40, c=11, **p = 5.70e-05** |

The classifier beats the constant baseline decisively and is **statistically indistinguishable from
guessing the majority label of the clip's scene** (+4.25 pp BA over it, 23 discordant clips,
p = 0.40). It is scene confounding, measured, not asserted. NP-64 slice: unsafe recall 53/64 =
82.81%; specificity, BA and PR-AUC are undefined on it and are not printed.

Full adjudication:
`plans/261011-1502-baseline-giao-thuc-bai-bao-goc-unsafe-net/reports/phan-quyet-2026-10-11-g21-g25.md`.
Raw numbers: `.../ket-qua-g21-g25.json`.

---

### 2026-10-11 (later) — G23 FRAME-REDUCTION MECHANISM: **ZERO TOUCHES CONSUMED**

Recorded here because a leg that *could* have spent a touch and did not is part of the budget
record, not an absence of one. **Budget unchanged: 2 pre-registered, 1 consumed, 1 remaining.**

[17] attribute +10.03 pp macro (85.78 -> 95.81) to fusing the behaviour detector, naming the
frame-reduction step as the mechanism. That claim splits into two separable questions and only the
second needs test clips:

- **(a) mechanism** — does `det-gateQ` (keep top-q frames by unsafe evidence) beat plain
  aggregation (`det-mean`/`-max`/`-pair`) of the SAME detector's SAME posteriors? Answerable
  entirely on the 83-clip val slice shared by every 4-class member.
- **(b) magnitude** — does the fused system reach 95.81% on the 64 test clips? Needs test.

Only (a) was run. `run_g23_framegate_val.py` reads stored artifacts, trains nothing, and touches no
test label: best gate minus best plain is **+0.13 pp** (`beh_ctx`) and **-0.13 pp** (`beh_tight`),
exact McNemar on val **p = 1.0000** in both directions, against the +10.03 pp [17] attribute to it.
The 2026-09-06 `detC-*` members reproduce the 2026-10-11 `detctx-*` members digit for digit, so
this is deterministic rather than seed noise.

**Test posteriors deliberately left unscored.** `fusion_g23_convlstm_cpu+detT.pt` and
`fusion_g23_convlstm_cpu+detctx.pt` contain `Pt` for the val-selected fusion weights (val macro
88.05% and 85.15%). Scoring either requires the remaining touch **and** a decision rule written
before it is spent. Nothing in this leg may be presented as a test result.

Standing caveat carried from the same day's G21 finding: every ranking above is a ranking **on
val**, and the 83-clip val slice was shown that morning to be blind to the 64-clip test set (CPU
gained +2.78 pp val macro while losing 3 test clips). Val ordering is not a test prediction.

Adjudication:
`plans/261011-1502-baseline-giao-thuc-bai-bao-goc-unsafe-net/reports/phan-quyet-2026-10-11-g23-co-che-tren-val.md`.
Raw numbers: `.../ket-qua-g23-val.json`.

### 2026-10-11 (later still) — NP-125-D DETECTION RE-EXPRESSION: **ZERO TOUCHES CONSUMED**

Recorded because this leg *re-reads* a test posterior that a spent touch produced, and re-reading is
the kind of thing a budget log exists to make visible. **Budget unchanged: 2 pre-registered, 1
consumed, 1 remaining.**

The author asked for the binary result as an **unsafe-detection objective** rather than a class
classification. Balanced accuracy is symmetric — it prices one missed unsafe clip exactly like one
false alarm — so the same run was re-expressed with **positive class = unsafe**.
`run_binary_detection_scorecard.py` trains nothing, runs no inference, and reads the SAME `Pt` in
`sweep_results/r2/temporal_convlstm_binary.pt` that the 2026-10-11 G21/G25 touch scored.

**The new leakage channel was named before measuring, not after.** Re-reading one saved posterior at
many thresholds *is* test probing. Mitigation, pre-registered in
`.../tien-dang-ky-np125d-phat-hien-nhi-phan.md` §2-3: all four operating points are fixed on the
**103-clip val slice** before test is read at them, and the script computes val first. No threshold
was ever chosen on test, not even illustratively.

Result at argmax: recall 82.81%, false-alarm rate 34.43%, precision 71.62%, PR-AUC 0.876. The
verdict does **not** move — no operating point separates from the scene-only control (exact McNemar
on unsafe decisions p = 0.4049 / 0.8450 / 0.3123 / 0.1214), and the control itself reaches **higher**
recall (89.06%). New quantity that needed no touch: restricting Mann-Whitney concordance to
same-context pairs drops AUC **0.8485 -> 0.6426**, so 59% of the apparent discrimination is context.

**Not a gate.** NP-125-D cannot overturn, loosen or soften G25; the gate table stays eight gates,
two pass, six fail. The remaining touch is still unspent and still needs a decision rule written
before it is spent.

Adjudication:
`plans/261011-1502-baseline-giao-thuc-bai-bao-goc-unsafe-net/reports/phan-quyet-2026-10-11-np125d-phat-hien-nhi-phan.md`.
Raw numbers: `.../ket-qua-np125d-phat-hien.json`.

#### Follow-up the same day: the published interval was at the wrong unit — fixed, still **ZERO TOUCHES**

`code-reviewer` found that the confidence interval on the within-context AUC was bootstrapped over
**clips** while the independence unit is the **background-context group** — the precise error this
thesis uses to reject the original protocol. Recomputed at the group unit: **[0.4923, 0.8443]**,
which **contains 0.50**. The 59% point estimate stands, but the data cannot exclude that *all* of the
apparent discrimination is context; the interval is wide because only **6 groups are label-mixed**.
The clip-level interval [0.5106, 0.7680] is retained in the JSON under the key
`ci_clip_resample_WRONG_UNIT` and is never used to conclude anything.

The scorer was re-run **twice** from the same saved `Pt`; a flattened-JSON diff shows **no measured
value changed** (second run: 0 diff entries). Budget unchanged: **2 pre-registered, 1 consumed, 1
remaining**. `report/vn/` §6.15.4, the extended abstract and §8.3.9(c) now print the group-unit
interval and say it contains chance — the correction makes the published claim weaker, not stronger.
