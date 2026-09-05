# W2 — Number-Integrity Fixes (measurement/diagnostic pass)

Scope: settle QA1's 3 defects + 3 more the coordinator found mid-task (folded in, same remit).
`.venv/bin/python` throughout. Every number below has its generating command shown. Sacrificing
grammar for concision per task rules.

---

## DEFECT 1 — cross-pipeline ECE mixing → RECOMPUTED, root cause identified

**Root cause:** scorecard's ECE=0.101 cell sits in the P3 same-dist row (54/64/88.4%) but was
actually computed off a different pipeline's (P1 pairwise-cascade, 56/64) softmax. Fix: recompute
ECE straight from `logits_yolo26.pt`'s own `ens_softmax`.

Command:
```
.venv/bin/python -c "
import torch, numpy as np
from eval_utils.calibration import ece
for f in ['logits_yolo26.pt','logits_s1a.pt']:
    d = torch.load(f'sweep_results/relation_tokens/{f}', weights_only=False)
    P = d['ens_softmax'].numpy(); y = d['labels'].numpy(); pred = P.argmax(1)
    print(f, (pred==y).sum(),'/64 ECE=', round(ece(P,y,15),4))
"
```
Output:
```
logits_yolo26.pt 52 /64 ECE= 0.1009
logits_s1a.pt 54 /64 ECE= 0.1041
```
Cross-checked bit-identical via `run_full_eval.py`'s own inline `ece()` call (see full_eval.json /
full_eval_s1a.json below) — matches to 4 decimals.

**Corrected number: P3 (yolo26) ECE = 0.1009 (15-bin).** S1a ECE = 0.1041 (its own number, as asked;
S1a is NOT the brief.md headline — see Defect 3).

**Caveat that must travel with this number (do not drop):** ECE=0.1009 is computed on
`argmax(ens_softmax)` = 52/64, the seed-AVERAGED ensemble decision — NOT the 54/64 `blind_pred`
that brief.md actually cites as the P3 same-dist headline. `blind_pred`'s own per-seed softmax was
never saved to disk (only decisions + the all-seed average), so no calibration number exists for the
exact 54/64 decision function. See Defect 5 — this is the same underlying issue.

Reliability diagrams + full multi-lens metrics saved:
`figures/relation_confusion.png`, `figures/relation_reliability.png` (currently reflect yolo26,
regenerated last), `sweep_results/relation_tokens/full_eval.json` (yolo26),
`sweep_results/relation_tokens/full_eval_s1a.json` (s1a).

**Cite in thesis:** "P3 (ensemble-basis) ECE = 0.1009 (15-bin, `logits_yolo26.pt`, n=64)" — with the
blind/ensemble caveat above stated in the same sentence or footnote.

---

## DEFECT 2 — k-fold NB significance instability → DIAGNOSED, not resolvable without a new run

Three p-values exist, wrongly treated in other reports as "the same result, different log
versions":

| Log | tag (backbone) | fold scene-only accs | relation top-1 | NB p | df |
|---|---|---|---|---|---|
| `kfold_relation.log` (Aug16) | `yoloapp_` (YOLO11) | 77.9/78.6/76.7/75.7/76.7 | 81.2±2.5% | **0.0375 (sig)** | 4 |
| `_zones_eval.log` KFOLD-ZONES (Sep5) | `zones_l1_` (deleted intermediate, YOLO26-zones-detector) | 77.9/78.6/76.7/75.7/76.7 | 80.0±2.3% | **0.1013 (n.s.)** | 4 |
| `_zones_eval.log` KFOLD-CHAMPION (Sep5) | `yolo26L1_` (YOLO26) | 77.9/78.6/76.7/75.7/76.7 | 80.6±2.8% | **0.0697 (n.s.)** | 4 |

Commands (read-only, no training re-run — see below for why):
```
cat sweep_results/relation_tokens/kfold_relation.log
cat sweep_results/relation_tokens/_zones_eval.log
grep -n "def nadeau_bengio" -A 13 eval_utils/paired_tests.py   # df = len(diffs)-1
```

**Root cause — two independent factors, both confirmed from the logs directly (no retrain needed):**

1. **These are three different feature/backbone configs, not three reruns of one config.** The
   per-fold scene-only baseline (depends only on view-grouping + labels, not on the relation model)
   is byte-for-byte identical across all three logs (77.9/78.6/76.7/75.7/76.7 in every case) —
   proving fold assignment and view-grouping are constant. The only thing that changes is the
   relation-Transformer's own per-fold accuracy, which varies with `--tag` (yoloapp_/YOLO11 vs
   zones_l1_/champion-YOLO26 variants). So the "flip" is genuinely comparing different pipelines
   under one name, not the same experiment misbehaving.
2. **Every one of the three runs used `--repeats 1` (the default), giving k=5 diffs → df=4.** The
   script's own docstring (`run_kfold_relation.py:8-10`) warns explicitly: *"A SINGLE k-fold split
   gives the Nadeau-Bengio test only k=5 difference samples (df=4), which is too few to trust; the
   reported '+/-' is then a 5-point spread, not a confidence interval."* All 3 historical p-values
   sit on this admittedly-underpowered design. `--repeats 10` exists precisely to fix this and was
   never used in any log on disk.

**Why not just rerun with `--repeats 10` now:** timed one seed x 5 epochs on the `yoloapp_` tag:
```
.venv/bin/python -c "
import torch, time
import run_relation_transformer as R
X,M,y = R.load('train','yoloapp_')
t1=time.time()
m,best = R.train_one(X[:400],M[:400],y[:400],X[400:],M[400:],y[400:],seed=0,epochs=5,use_cw=False,n_num=X.shape[-1]-1,aux_w=0.5)
print('elapsed 1 seed x 5 epochs:', time.time()-t1)
"
```
Output: `elapsed 1 seed x 5 epochs: 44.6s` → ~8.9s/epoch → ~9 min/seed at epochs=60 → ~135 min for
one full 5-fold×3-seed×1-repeat pass → **~22+ hours for `--repeats 10`** on CPU. Out of budget for a
diagnostic task; not run. Flagging as required follow-up (est. run overnight, single canonical
config, see below).

**Verdict — can the claim be stated at all:** **No, not as currently evidenced.** Every p-value on
disk is a single-split, df=4 point estimate on top of that config choice being inconsistent across
reports. The instability itself is real and must be disclosed, not resolved by picking whichever log
looks best. Recommended thesis wording: *"A Nadeau-Bengio test for relation-Transformer vs scene-only
was run at repeats=1 (df=4) under three token/backbone configurations, yielding p=0.04-0.10 —
underpowered and configuration-dependent. Not citable as a stable significance result. A properly
powered rerun (`run_kfold_relation.py --tag yolo26app_ --repeats 10`, df=49) on the frozen headline
config is required before either 'significant' or 'n.s.' can be claimed."* If Lane A can spare an
overnight CPU run, that command (matching the tag of the already-cited k-fold headline artifact
`kfold_oof_softmax_yolo26app_.pt`) is the one to run.

**Link to Defect 4 (coordinator's hypothesis) — checked directly:** the scorecard's separate claim
"YOLO11 ablation (k-fold): 419/516=81.2%, NB p=0.0375" traces to `kfold_relation.log` (`--tag
yoloapp_`, pooled-516 k-fold) — a completely different script/artifact from the 64-clip
`logits_yolo11_baseline.pt` byte-duplication bug (Defect 4 below). The two bugs are NOT the same bug,
but they compound: Defect 4 proves the codebase already conflates same-named-but-different-config
artifacts once at the 64-clip level; Defect 2 shows the identical failure mode at the k-fold level
(three different configs, tracked as if one result). Same discipline gap, two independent instances.

---

## DEFECT 3 — stale/ambiguous full_eval.json → REGENERATED with provenance, root cause = missing metadata not staleness

**What was actually wrong:** `full_eval.json` (mtime Aug29 00:30) was NOT stale — it was generated
~1.3 min after `logits_yolo26.pt` was built (Aug29 00:28), from `--logits logits_yolo26.pt` (an
override of the script's *old* default `logits_s1a.pt`). A peer report (`f1-metric-suite.md`) flagged
it as "stale" only because it diffed the JSON against `logits_s1a.pt` (the script's documented
default) and got a mismatch — but the JSON was never generated from s1a at all. **The real defect was
that the JSON carried no record of which `--logits` artifact produced it**, so two independent
reviewers (peer report + this one) both had to reverse-engineer provenance from per-class recall
numbers instead of reading a field.

Command + verification:
```
.venv/bin/python -c "
import torch, numpy as np
from eval_utils.metrics import per_class_recall
d = torch.load('sweep_results/relation_tokens/logits_yolo26.pt', weights_only=False)
P = d['ens_softmax'].numpy(); y = d['labels'].numpy(); pred = P.argmax(1)
print(per_class_recall(y, pred))
"
```
Output: `{0: 71.9, 1: 90.9, 2: 84.6, 3: 100.0}` — exact match to the old `full_eval.json`'s per-class
recall (71.9/90.9/84.6/100.0). Confirms old JSON = yolo26 ensemble, not s1a.

**Fix applied to `run_full_eval.py`** (I own this file per task scope):
- default `--logits` changed from `logits_s1a.pt` → `logits_yolo26.pt` (yolo26 is the actual P3
  headline cited throughout brief.md/QA1; s1a was never a headline number anywhere, it just happened
  to be the leftover default arg from before yolo26 was built).
- output JSON now carries `source_logits`, `pred_basis`, and (when present) `blind_pred_top1_correct`
  + `blind_vs_ensemble_pred_agree_frac` + an explicit note — closing the provenance gap for good.
- output filename now derives from `--logits`: default run still writes the canonical
  `full_eval.json`; any non-default `--logits` writes `full_eval_<tag>.json` instead of silently
  overwriting the headline file.

Regenerated both:
```
.venv/bin/python run_full_eval.py --logits logits_yolo26.pt
.venv/bin/python run_full_eval.py --logits logits_s1a.pt
```
`full_eval.json` (yolo26, canonical) now: `top1_correct=52/64, macro_recall=86.85, ece_15bin=0.1009,
source_logits=logits_yolo26.pt, blind_pred_top1_correct=54/64, blind_vs_ensemble_pred_agree_frac=0.9688`.
`full_eval_s1a.json`: `top1_correct=54/64, macro_recall=88.41, ece_15bin=0.1041,
blind_pred_top1_correct=50/64`.

**Source of truth going forward:** `sweep_results/relation_tokens/full_eval.json`, generated by
`run_full_eval.py` (no args) from `logits_yolo26.pt`. Any thesis chapter citing calibration/ROC-AUC/
safety-metric numbers for P3 same-dist must cite this file AND carry the blind-vs-ensemble caveat
from Defect 1/5.

---

## DEFECT 4 (coordinator finding) — `logits_s1a.pt` ≡ `logits_yolo11_baseline.pt`, byte-identical

Command:
```
.venv/bin/python -c "
import torch, hashlib
def sha(t): return hashlib.sha1(t.numpy().tobytes()).hexdigest()
for f in ['logits_s1a.pt','logits_yolo11_baseline.pt']:
    d = torch.load(f'sweep_results/relation_tokens/{f}', weights_only=False)
    print(f, sha(d['ens_softmax']), int(d['blind_top1']))
"
```
Output: both `12fdecb6ec347ad1d4d081da52dbef143471a5d5`, both `blind_top1=50`.

Confirmed at the log level too — `train_s1a.log` (Aug 13 21:38:33) and `train_yolo11_baseline.log`
(Aug 16 15:04:41) print IDENTICAL per-seed results line for line (same 7 seeds, same val_macro@epoch,
same test counts, same confusion matrix, same z-test), both under `tag='app_'`. Only the header
warning line and the save-path in the final line differ.

**Root cause (inference, not directly provable from disk — flagged as inference):** both runs share
`tag='app_'` and produce bit-identical float32 output 3 days apart. The most parsimonious
explanation is the "yolo11_baseline" run replayed the exact same command as the s1a run (same tag,
same seeds, same aux-loss weight) rather than a genuine no-aux-loss vanilla baseline — CPU
determinism with fixed seeds makes bit-identical reproduction across separate processes plausible if
no code changed in between, which the identical log text supports.

**Does this break "detector generation (YOLO11 vs YOLO26) is within noise"?** No — checked directly.
That scorecard line ("YOLO11 ablation (k-fold): 419/516=81.2%... NB p=0.0375") traces to
`kfold_relation.log`, which comes from `run_kfold_relation.py --tag yoloapp_` (pooled 516-clip
k-fold) — a completely different script and artifact from the 64-clip `logits_yolo11_baseline.pt`.
The two never share data. **The "not detector-limited" conclusion survives**, sourced from
`kfold_relation.log`, not from the duplicated file.

**What must change:** `logits_yolo11_baseline.pt` must not be cited as an independent 64-clip
same-dist data point — it is s1a under another name. Either delete it or regenerate it as a genuine
`aux_w=0.0` vanilla YOLO11 baseline before any report references "yolo11_baseline" at the 64-clip
level.

---

## DEFECT 5 (coordinator finding) — what sits between `ens_softmax` and the headline decision

Traced in `run_relation_transformer.py:153-233` (read, not run — this is the generating script for
every `logits_*.pt` file):

- `ens_softmax` = `logits_sum / args.seeds` — the softmax **averaged across all trained seeds**
  (an ensemble of N independently-trained models), argmax'd post-hoc.
- `blind_pred` = the **raw prediction of the single seed with the best held-out validation macro**
  (`blind = max(rows, key=lambda r: r['val_macro'])`), selected **before** looking at test — this is
  the actually-blind, pre-registration-compliant decision the brief.md headline (54/64 for yolo26)
  is built from.

**There is no calibration/threshold/temperature step between softmax and either quantity** — the gap
the coordinator flagged is not a hidden decision rule, it is that these are **two different models**
(ensemble-of-N vs single-best-val-seed), and only the ensemble's probabilities were persisted to disk.
`blind_pred`'s own single-seed softmax was **never saved** — confirmed by inspecting the `.pt` keys
(`ens_softmax, blind_pred, labels, blind_top1` — no per-seed logits array).

**Direct consequence for the conformal/selective-risk work (flagged for whoever owns that lane):**
any AURC / risk-coverage / conformal-prediction analysis built on `ens_softmax` characterises the
**ensemble decision rule** (52/64 for yolo26, 54/64 for s1a) — **not** the 54/64 blind headline number
quoted everywhere else for yolo26. State this pairing explicitly in that chapter: *"conformal
coverage guarantees in this section apply to the ensemble-averaged predictor (52/64 top-1), a
different decision function than the blind headline (54/64) reported for accuracy elsewhere."*
Do not present both under one "P3" label without this qualifier.

---

## DEFECT 6 (coordinator finding) — `logits_yoloroi.pt` blind=55/64 > headline 54/64: provenance, not promotion

Command:
```
.venv/bin/python -c "
import torch
d = torch.load('sweep_results/relation_tokens/logits_yoloroi.pt', weights_only=False)
print(int(d['blind_top1']), (d['blind_pred'].numpy()==d['labels'].numpy()).sum())
"
```
Output: `55 55` — confirmed, matches `train_yoloroi.log`'s printed `BLIND (val-selected seed 4):
55/64 = 85.9% top1, macro 89.2%`.

**Provenance, by file mtime (chronological, `stat -f "%Sm %N"`):**
```
Aug 16 15:04:41  train_yolo11_baseline.log
Aug 16 16:51:51  build_yoloapp.log            (tokens_yoloapp_*.pt freshly built)
Aug 16 17:07:25  train_yoloroi.log / logits_yoloroi.pt   (tag='yoloapp_', 7 fresh seeds)
Aug 16 17:36:05  lco_yoloroi.log
Aug 16 17:54:26  kfold_relation.log           (same day, same tag, NB p=0.0375)
Aug 29 00:28:58  logits_yolo26.pt             (13 days later — new YOLO26 pipeline, current headline)
```
`find . -iname "test-touch-log*"` → **no file found anywhere in the repo.** Brief.md's hard rule #3
("one declared test touch per pipeline variant, timestamp-logged in `reports/test-touch-log.md`") has
never actually been implemented as a log — there is no artifact to check whether 55/64 was a
pre-registered single touch or an extra peek.

**Which case is it (coordinator's two options):** neither cleanly — a third case. `yoloroi` is a
same-day (Aug 16) exploratory run on the YOLO11-backbone `yoloapp_` tokens, run alongside the k-fold
and LCO evaluations for that same tag, **before** the project moved to the YOLO26 architecture 13
days later. It is not "selected on test and hidden" in the sense of cherry-picking a final number —
it's one of at least 3 separate 64-clip touches recorded for the YOLO11 track alone (`s1a`/duplicate
`yolo11_baseline` at 50/64 blind, `yoloroi` at 55/64 blind, both `tag`-distinct token builds) with no
test-touch-log distinguishing which (if any) was "the" declared touch. **It is superseded by
architecture change, not legitimately overlooked** — the project already moved past the YOLO11 track
to YOLO26 for reasons presumably unrelated to this one clip count (not documented either). Given the
QA1 audit's own N=64 ceiling analysis (±15-20pp needed to resolve a real difference), a 1-clip gap
(54 vs 55) is pure noise regardless.

**Recommendation: do not promote.** But separately: the complete absence of `reports/test-touch-log.md`
despite it being a binding hard rule is itself worth flagging to whoever owns Lane C/protocol
discipline — right now nothing on disk can distinguish a legitimate single touch from an
undisciplined multi-touch for ANY of these artifacts, for any tag.

**`robustness_eval.log` check (closes the last open thread):**
```
diff sweep_results/relation_tokens/robustness_eval.log sweep_results/relation_tokens/train_yoloroi.log
```
Output: lines 1-27 identical in both files; `robustness_eval.log` has 9 extra lines (28-36) appending
an "F/G — robustness / TTA" block (clean/flip/jpeg/noise/blur/dark/TTA accuracy under the same
seed-ensemble). **Not a fourth duplicate touch** — `robustness_eval.log` is the same `yoloroi`
run's log with a robustness-analysis section appended afterward, not an independent retrain. No new
provenance concern; folded into the Aug 16 `yoloroi` touch above.

---

## Canonical artifacts table

| Headline claim | Canonical file | Config / how produced | Status |
|---|---|---|---|
| P3 same-dist top-1/macro (54/64, 88.4%) | `sweep_results/relation_tokens/logits_yolo26.pt`, key `blind_pred` | `run_relation_transformer.py --tag yolo26app_ --save_logits logits_yolo26.pt` (tag inferred from token-build mtime proximity, Aug29 00:14-00:18 → 00:28; **no training log was persisted for this run — provenance gap, unresolved, see below**) | Verified count (QA1 + here); generating command unlogged |
| P3 same-dist ECE | `logits_yolo26.pt`, key `ens_softmax` via `eval_utils/calibration.ece` | ensemble-basis (52/64), NOT blind (54/64) — see Defect 1/5 | **New, corrected: 0.1009** (was mislabeled 0.101 from a P1 artifact) |
| P3 full multi-lens suite (ROC-AUC/PR-AUC/safety/CI) | `sweep_results/relation_tokens/full_eval.json` | `run_full_eval.py` (now defaults to `logits_yolo26.pt`), ensemble-basis | Regenerated this session, provenance-tagged |
| S1a (non-headline) full suite | `sweep_results/relation_tokens/full_eval_s1a.json` | `run_full_eval.py --logits logits_s1a.pt` | Regenerated this session |
| P3 k-fold 516 (417/516, 80.8%/84.0%) | `kfold_oof_softmax_yolo26app_.pt` | `run_kfold_relation.py --tag yolo26app_` | Unaffected by this pass (QA1 already verified exact) |
| k-fold NB significance (relation vs scene-only) | none — **no canonical file exists** | needs `run_kfold_relation.py --tag yolo26app_ --repeats 10` (~22h CPU, not run) | **Unresolved — do not cite any of the 3 existing p-values as final** |
| LCO 516 (78-82% range) | `run_lco_eval.py` outputs | owned by another lane, out of scope here | Not touched |
| YOLO11-vs-YOLO26 "not detector-limited" | `kfold_relation.log` | `run_kfold_relation.py --tag yoloapp_` | Survives Defect 4 (different artifact) |
| **Do NOT cite** | `logits_yolo11_baseline.pt` | byte-identical to `logits_s1a.pt` (Defect 4) | Duplicate, not independent |
| **Do NOT promote** | `logits_yoloroi.pt` (55/64 blind) | superseded YOLO11-track exploratory run, no test-touch-log (Defect 6) | Not headline-eligible as evidenced |

---

## Unresolved questions

1. Exact command that produced `logits_yolo26.pt` is not logged anywhere on disk — inferred as
   `--tag yolo26app_` purely from mtime proximity to the token build (10 min gap). Whoever owns Lane
   A should confirm the actual invocation and, ideally, start persisting `train_<name>.log` for every
   `--save_logits` run going forward (this whole defect class stems from missing provenance metadata).
2. k-fold NB significance cannot be finalized without a `--repeats 10` rerun (~22h CPU on this
   machine) on the canonical `yolo26app_` tag. Not run here — budget/scope. Recommend scheduling
   overnight before Ch3/Ch4 freeze.
3. `reports/test-touch-log.md` (required by brief.md hard rule #3) does not exist anywhere in the
   repo — every "blind"/"one touch" claim in every report (including this one's own citations) is
   currently unverifiable against a real touch ledger. This is bigger than any single defect above
   and should probably become its own task.
4. Whether `logits_yolo11_baseline.pt` was meant to be a vanilla (`aux_w=0.0`) baseline is inferred,
   not confirmed — the exact CLI args used were not captured in either log's header. If someone still
   needs a genuine YOLO11 vanilla-baseline number, it should be regenerated fresh with explicit
   `--aux_w 0.0` and a persisted log.
5. (Resolved during write-up, kept for the record) `robustness_eval.log` vs `train_yoloroi.log` —
   confirmed via `diff` to be the same run (first 27 lines identical), not an independent touch; see
   Defect 6.
