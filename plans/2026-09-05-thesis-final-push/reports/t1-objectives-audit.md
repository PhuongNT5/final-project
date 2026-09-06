# T1 — Objectives & results audit — hard numbers

Date 2026-09-05. Read-only on code/unsafenet. All numbers below either loaded directly from
`sweep_results/r2/*.pt` (torch.load, weights_only=False) / `report_fusion.json`, or computed with
scipy/numpy in `/private/tmp/.../scratchpad/{build_table.py,power.py}`. Scripts kept in scratchpad,
not committed.

## 1. Run table — every experiment to date

### 1a. Round 0/1 (pre-frozen-protocol, PROGRESS.md) — test set touched repeatedly per entry
Protocol note: no val/test split discipline existed yet; each row is a distinct touch of the same
64-clip test set (this is the "improvement ladder" phase-01 flags as resampling, not progress).

| # | Config | Top-1 | Macro | Notes |
|---|---|---|---|---|
| 1 | V3 frozen-COCO YOLO→Transformer (18-model ens.) | 55/64 (85.9%) | 88.84% | |
| 2 | V5 multi-scale frozen (L15/19/22)→Transformer | 53/64 (82.8%) | 88.77% | |
| 3 | V6-det COCO det+scene→cross-attn Tfm | 52/64 (81.2%) | 85.36% | only 27% person det. |
| 4 | V6-ft frame-level fine-tuned YOLO→Tfm | 54/64 (84.4%) | 87.27% | frame labels noisy |
| 5 | VideoMAE frozen-base→MLP | 55/64 (85.9%) | 87.00% | 86M params, same ceiling |
| 6 | Final E2E, 42-model sweep | 54/64 (84.4%) | 87.20% | exhaustive search |
| 7 | YOLOWorld rel.+backbone→Tfm | 55/64 (85.9%) | 89.54% | best macro, UnauthorInt 100% |
| 8 | CrossEns COCO+YOLOWorld | 55/64 (85.9%) | 87.71% | models too correlated |
| 9 | Pairwise 8-class contrastive+cascade+ens | **56/64 (87.5%)** | 89.28% | best top-1 this round; ONNX-verified (CPU-FP32 56/64, INT8 55/64) |
| — | round-1 "previous blind" (cited in R2 review) | 53/64 (82.8%) | 85.4% | referenced baseline going into R2 |

Per-class (best round-0/1 ens., 56/64): WalkwayViol 27/32, UnauthorInt 8/11, OpenedPanel 13/13, CarryOverload 8/8.
An earlier "57/64 calibrated" number was withdrawn — test-set bias-search tuning; honest refit gives 53.4±1.2/64.

### 1b. Round 2 (`unsafenet/`, frozen protocol: val carved from 83 train clips, one test touch per final config)
46 members loaded from `sweep_results/r2/temporal_*.pt` (kind: lstm/tfm/mean = trained temporal heads,
seeds 3 or 8; agg = training-free aggregation, seeds 1; det = detector-conditioned heads, seeds 1,
generated **today 2026-09-05 07:36-07:47**, i.e. AFTER the MPS bug fix). Full table below, sorted by
test top-1 (test numbers are reported ONLY because they were already computed and stored in prior
legitimate single-touch runs per config — none of this table is "test-selected" in the sense of
picking a new winner; the fusion below is the one live selection).

| name | tag | kind | seeds | val_macro | val_top1 | test_top1 | test n | test_macro |
|---|---|---|---|---|---|---|---|---|
| t1-lstm-elg | b1 | lstm | 3 | .8545 | .8193 | **.8594** | 55/64 | **.8586** |
| tx-lstm-elg | b0+b1 | lstm | 3 | .8313 | .8072 | **.8594** | 55/64 | **.8586** |
| t1-tfm-elg | b1 | tfm | 3 | .8406 | .8072 | .8438 | 54/64 | .8508 |
| aggb0-max / t-lstm-el / t-mean-lg / t-tfm-el / t-tfm-lg / tf-b0lstm-lg-s8 / tf-xlstm-lg-s8 / tx-lstm-lg | mixed | mixed | 1-8 | .81-.85 | .78-.83 | .8281 | 53/64 | .82-.85 |
| detT-max | — | det | 1 | **.8805** (2nd-highest val of all 46) | .8554 | .8125 | 52/64 | .8387 |
| 10 more configs (agg/det/t/tx) | | | | .81-.86 | .76-.85 | .8125 | 52/64 | .81-.84 |
| detC-gate25, t-lstm-lg, t1-mean-lg, t1-tfm-lg, tf-lstm-lg-s8, tx-tfm-lg | | | | .81-.86 | .80-.83 | .7969 | 51/64 | .80-.82 |
| aggb0-pair, aggb1-gate25, aggb1-mean, detC-{gate75,mean,pair}, detT-{mean,pair}, **t1-lstm-lg** | | | | .81-.88 | .76-.84 | .7812 | **50/64** | .78-.81 |
| aggb1-{gate50,gate75,max}, detC-gate50, detT-{gate25,gate75} | | | | .81-.86 | .76-.84 | .7656 | 49/64 | .78-.82 |
| detT-gate50 | — | det | 1 | .8578 | .8193 | .7344 | 47/64 | .7320 |
| aggb1-pair | b1 | agg | 1 | .8502 | .8072 | .6562 | 42/64 | .7342 |

Full 46-row CSV-equivalent kept in scratchpad `rows.json` (not committed; regenerate with
`build_table.py` if T4/T5 need it — reads only, no risk).

**Blind fusion (report_fusion.json, mtime 2026-09-05 23:44 — i.e. AFTER det members existed):**
Caruana-selected weights = `{aggb1-pair:.025, t1-lstm-lg:.9, t1-tfm-lg:.05, tf-lstm-lg-s8:.013,
tx-lstm-lg:.013}`. **No det member was selected**, despite detT-max having the 2nd-best val_macro of
all 46 candidates. Val: 68/83=81.93% top1, 85.45% macro. **Test (blind, the headline): 50/64=78.12%
top-1 [66.0,87.5], 80.81% macro [70.6,90.2].**

**Test-selected (best test top-1 across the 46 legitimately-trained-but-multiply-touched configs,
i.e. what you'd report if you cherry-picked post-hoc):** t1-lstm-elg / tx-lstm-elg tied at
**55/64 = 85.94% top-1, 85.86% macro** — matches the number given in the task brief exactly.

**Resolved insight (supersedes phase-01's carried-in note):** the "detector never fed through to a
clip decision" gap is now closed — det{T,C}-* members exist and were available to the Caruana
selector today. Result: standalone detector-conditioned heads cap at 52/64 (detT-max), and the
selector did not want any of them in the blind ensemble. **The paper's own mechanism, now correctly
trained (mAP50 0.79/0.81) and fed through to clip decisions, still does not help** — third
independent confirmation (after frame-reduction ablation and the ConvLSTM-baseline design) that the
YOLO-behaviour stage is not the lever here.

**Val↔test correlation, re-verified independently on this data (not just cited):**
- All 46 configs: Pearson r=-0.247 (p=0.098), Spearman rho=-0.261 (p=0.080).
- 22 "legitimate" trained temporal heads only (excl. 1-seed agg/det): Pearson r=-0.136 (p=0.545).
- Previously reported (thesis-process-review, N=34 configs, a different subset cut): r=-0.066 (p=0.71).
- **Conclusion unchanged across every subset tried: r is small, negative, never significant.**
  Val macro carries no exploitable signal about test macro on this benchmark, however you slice the
  config set. (Exact r/N is sensitive to which configs count as "legitimate" — flagged as unresolved
  below — but the qualitative finding — no signal — is robust.)

### 1c. Behaviour-detector stage (object-detection metric, not clip classification)
| run | boxes | imgsz | epochs | best epoch | mAP50 |
|---|---|---|---|---|---|
| beh_tight | literal paper (tight person box) | 416 | 25 | 15 | 0.7913 |
| beh_ctx | expanded 4.0x/1.8x | 512 | 36 | 15 | 0.8059 |
| behaviour_yolo (pre-fix, MPS) | tight | 416 | 25 | — | 0.0144 (failed) |
| behaviour_yolo_ctx (pre-fix, MPS) | ctx | 512 | 36 | — | 0.00212 (failed) |

Box-context ablation is negative for the *detector* metric direction too (0.806 ctx > 0.791 tight by
+1.5pp mAP50) but, per 1b, neither variant's clip-decision output beats the non-detector ensemble.

### 1d. Fair head-to-head (2026-09-02, different pipeline: YOLO26 entity+relation Transformer, NOT unsafenet/)
| Protocol | N | Top-1 | Macro | CI width (top-1) |
|---|---|---|---|---|
| same-dist 64-clip (blind) | 64 | 54/64 = 84.4% | 88.4% | ±11pp (Clopper-Pearson) |
| k-fold (headline, pooled) | 516 | 417/516 = 80.8±3.1% | 84.0% | **±3pp** (Nadeau-Bengio) |
| leave-camera-out (scene-robust) | 516 | 398/516 = 77.1% | 78.6% | tight (516) |
| scene-only baseline, LCO | 516 | — | 16.9% (collapses) | — |

This is the only evidence in the whole project produced at N>64, and it is the only evidence with a
CI narrow enough to plausibly separate two systems. It uses a different model family (YOLO26
relation-Transformer, code outside `unsafenet/`) — a genuinely different pipeline from round 2, not a
reprocessing of the same numbers.

## 2. Baseline facts (reused per task instruction, cross-checked against sources above)
Paper full (YOLOv4+ConvLSTM, bbox): 62/64=96.88% top-1, 95.81% macro (paper Table 6, cited
thesis-process-review.md L9).
Paper no-reduction (ConvLSTM only): 57/64=89.06% top-1, 85.78% macro (paper Table 5).
Ours blind (r2 fusion, above): 50/64=78.12% top-1, 80.81% macro. — verified directly from report_fusion.json.
Ours test-selected: 55/64=85.94% top-1, 85.86% macro. — verified directly from temporal_t1-lstm-elg.pt / temporal_tx-lstm-elg.pt.
Val/test Pearson r=-0.066 (p=0.71, N=34) — re-verified qualitatively (see 1b); exact N/r subset-dependent, sign/non-significance robust.

## 3. Statistical power at N=64 — the core deliverable

### 3a. Minimum detectable top-1 difference, McNemar paired design, alpha=.05, power=.80 (Connor 1987)
Formula: n = (z_{a/2}√ψ + z_β√(ψ−δ²))² / δ², solved for δ given n=64, ψ = assumed discordant fraction.

| ψ (frac. discordant) | # discordant pairs | min δ (proportion) | **min K clips of net advantage** |
|---|---|---|---|
| 0.05 | 3.2 | .0769 | 4.9 |
| 0.10 | 6.4 | .1087 | 7.0 |
| 0.15 | 9.6 | .1331 | 8.5 |
| 0.20 | 12.8 | .1537 | 9.8 |
| 0.25 | 16.0 | .1719 | 11.0 |
| 0.30 | 19.2 | .1883 | 12.1 |
| 0.40 | 25.6 | .2174 | 13.9 |
| 0.50 | 32.0 | .2431 | 15.6 |

Realistic discordance for two different pipelines on the same 64 clips is plausibly 15-30% (10-19
clips flip at least one way) → **need a net ~9-12 clip advantage to reach 80% power at alpha=.05.**
At low discordance (two very similar pipelines, ψ=.05-.10) the bar drops to 5-7 clips but that
regime requires the two systems to almost never disagree, which is not this comparison.

### 3b. Exact p-value vs paper's 62/64, our_correct in {55..64}, over ALL possible pairings
We lack the paper's per-clip labels, so (b,c) = (we-right/paper-wrong, we-wrong/paper-right) is not
observable — enumerated over every combinatorially valid split given paper_correct=62, paper_wrong=2.

| our_correct | best-case (b,c) | best p | worst-case (b,c) | worst p | best split that FAVORS us (b≥c) |
|---|---|---|---|---|---|
| 55 | (0,7) | 0.0156 | (2,9) | 0.0654 | **none exists** |
| 56 | (0,6) | 0.0312 | (2,8) | 0.1094 | **none exists** |
| 57 | (0,5) | 0.0625 | (2,7) | 0.1797 | **none exists** |
| 58 | (0,4) | 0.1250 | (2,6) | 0.2891 | **none exists** |
| 59 | (0,3) | 0.2500 | (2,5) | 0.4531 | **none exists** |
| 60 | (0,2) | 0.5000 | (2,4) | 0.6875 | **none exists** |
| 61 | (0,1) | 1.0000 | (2,3) | 1.0000 | **none exists** |
| 62 | (0,0) | 1.0000 | (2,2) | 1.0000 | (0,0), tie, p=1 |
| 63 | (1,0) | 1.0000 | (2,1) | 1.0000 | (1,0), p=1 |
| 64 | (2,0) | 0.5000 | (2,0) | 0.5000 | (2,0), **p=0.5, still n.s.** |

**The structural finding: for our_correct ≤ 61, it is mathematically impossible for the discordance to
favor us at all** (b ≤ 2 always, since paper is only wrong twice; c ≥ 3 forced). For our_correct in
{55,56,57} the *best case* (all-favorable pairing) crosses p<0.05 — but "best case" here means the
discordance favors the *paper*, i.e. this is the bound on how badly we could be *losing*
significantly, not a path to beating them.

**Even our_correct=64 (a perfect, error-free run) cannot reach p<0.05 against the paper's 62/64**,
paired (McNemar best case p=0.5) or unpaired (Fisher exact p=0.496) — because the paper made only 2
errors, the maximum possible discordant advantage is capped at 2 clips, and 2 discordant pairs can
never produce p<0.05 under any test. **This is the number that proves claim (a) ("beat 62/64
significantly") is not achievable at N=64, full stop — not a power problem, a ceiling problem.**

Unpaired two-proportion/Fisher exact (the test actually computable without the paper's per-clip
labels) for our_correct vs paper 62/64, independent samples:
| our_correct | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fisher p | .054 | .096 | .164 | .273 | .440 | .680 | 1.00 | 1.00 | 1.00 | .496 |

Note p is non-monotonic near 64 for a subtle reason (small-N exact tests): even so, no value of
our_correct produces p<0.05 in the direction of beating the paper.

### 3c. Clopper-Pearson exact 95% CI, k correct / 64
| k | top-1% | CI low% | CI high% | width pp |
|---|---|---|---|---|
| 50 | 78.12 | 66.03 | 87.49 | 21.46 |
| 51 | 79.69 | 67.77 | 88.72 | 20.94 |
| 52 | 81.25 | 69.54 | 89.92 | 20.38 |
| 53 | 82.81 | 71.32 | 91.10 | 19.77 |
| 54 | 84.38 | 73.14 | 92.24 | 19.11 |
| 55 | 85.94 | 74.98 | 93.36 | 18.39 |
| 56 | 87.50 | 76.85 | 94.45 | 17.60 |
| 57 | 89.06 | 78.75 | 95.49 | 16.73 |
| 58 | 90.62 | 80.70 | 96.48 | 15.78 |
| 59 | 92.19 | 82.70 | 97.41 | 14.71 |
| 60 | 93.75 | 84.76 | 98.27 | 13.51 |
| 61 | 95.31 | 86.91 | 99.02 | 12.12 |
| 62 | 96.88 | 89.16 | 99.62 | 10.46 |
| 63 | 98.44 | 91.60 | 99.96 | 8.36 |
| 64 | 100.00 | 94.40 | 100.00 | 5.60 |

Every k we have ever produced (50-56) has a CI that overlaps the paper's 62/64 CI [89.2,99.6].
Non-overlap first appears only from k≈58-59 upward, and even overlapping CIs at k=62 vs our best
observed (55) only means "cannot rule out equal" — it does not certify equality either (CI overlap is
a heuristic, not a test; see 3b for the actual test).

### 3d. Bootstrap 95% CI width on macro accuracy at N=64 (5000+ resample simulation, per-class n=32/11/13/8)
| Scenario | Simulated 95% CI | Width | Half-width |
|---|---|---|---|
| Blind measured (macro 80.8%) | [73.6, 91.5] | 17.9pp | 9.0pp |
| Mid-range (macro 88%) | [77.0, 94.2] | 17.2pp | 8.6pp |
| Paper-level (macro 95.81%) | [80.6, 96.2] | 15.6pp | 7.8pp |
| report_fusion.json actual (macro 80.81%) | [70.6, 90.2] | 19.6pp | 9.8pp |

**Minimum detectable macro difference at N=64 ≈ 16-20 percentage points** (two CIs stop overlapping
around there). Every macro number this project has ever reported (round0/1: 85.4-89.5%; round2:
78-89%; fair-headtohead: 78.6-88.4%) sits inside a single ~18pp band — the entire project's result
history is statistically one point, not a trend.

## 4. Claim assessment — ranked, with cost and missing evidence

**(a) "Beat 62/64"** — **NOT ACHIEVABLE, stated bluntly.** Proven in 3b: even a hypothetical 64/64
(zero errors) cannot reach p<0.05 against the paper's 62/64 by any test (paired best-case McNemar
p=0.5, unpaired Fisher p=0.496), because the paper is 2 clips from perfect and 2 discordant pairs is
an information-theoretic floor no amount of our accuracy can cross. Our actual best (55/64
test-selected, itself not a legitimate headline) is 7 clips below paper and would need +9 to +12 net
clips (3a) just to have adequate power — a number larger than the 2-clip ceiling in 3b allows even
in principle. Cost to attempt: unbounded (chasing noise); evidence needed: none would suffice —
mathematically closed. **Do not pursue.**

**(b) "Statistically indistinguishable from the paper at N=64, N=64 cannot separate"** —
**ACHIEVABLE NOW, already proven.** Every piece of evidence needed exists: CP intervals overlap (3c),
McNemar/Fisher never significant for our_correct≤61 in the direction that matters (3b), bootstrap
bands 16-20pp swallow every number either side has produced (3d), val/test independence (1b, r
consistently small/negative/n.s. across every subset), and the paper's OWN 62-vs-57 ablation is n.s.
(p≥0.0625, already on file). Missing: nothing structural; would benefit from finishing T4's ConvLSTM
baseline (paper's actual architecture, not just its target number) to make the reproduction claim
complete, but the statistical argument doesn't depend on it. Cost: near zero — this is a writing task
plus the frozen-protocol table already produced above. **Recommended primary claim.**

**(c) "Match accuracy at lower latency"** — **PARTIALLY ACHIEVABLE, cheap, but tighten the evidence
before claiming it.** Measured M4 end-to-end latency 26.8-28.8ms p50 (CPU-FP32, ONNX, `benchmarks.json`)
vs paper reporting **no latency at all** — trivially "faster" but only because there is no comparator,
which is honest and sufficient (paper measured zero deployment metrics). Caveat found during this
audit: the "Jetson" row in `deployment/artifacts/projections.md` is **not a measurement of our
pipeline** — it is cited third-party YOLO11s-backbone-only numbers from Ultralytics/arXiv literature.
Any claim of "measured on Jetson" would be false; the fair-headtohead-scorecard.md line "measured M4 +
Jetson" overstates this and should be corrected to "measured M4; Jetson figures are literature
citations for context, not our pipeline." "Match accuracy" should target the paper's no-reduction row
(89.06%/85.78%), not the full 96.88% (per (a)) — our test-selected 55/64=85.94%/85.86% is within noise
of that row already (2-clip gap, CI-overlapping, same order as the paper's own insignificant internal
ablation). Cost: low (write-up + one honest caveat); missing evidence: a real Jetson run of the actual
pipeline, or drop the device claim to M4-only. **Recommended as secondary/supporting claim**, not
sufficient alone as the thesis's central contribution (the paper's total silence on latency makes any
comparison here easy to state but also easy for a reader to dismiss as unearned unless paired with (b)).

### Rank: (b) primary, (c) supporting, (a) excluded.
Fallback if (b) is judged too "negative" for the committee: pair it explicitly with (c) as
"statistically-tied accuracy at measured lower latency and zero annotation cost" — same evidence,
reframed as a deployment-value claim rather than a pure negative result.

## 5. Frozen evaluation protocol (binding, for T3/T4 without further interpretation)
1. **Splits:** 691 total unsafe+safe clips → 452 train / 64 test, fixed by the paper's own released
   split (do not re-split). Val = 83 clips stratified on (label, view), carved from the 452 train
   clips only. Test = the 64 unsafe clips, held out, never used for selection.
2. **Model-selection rule:** epoch, backbone, seed, temporal-head architecture, and ensemble/fusion
   weights are chosen **only** by val macro (mean per-class recall) on the 83-clip val slice.
   Caruana greedy fusion with replacement, weights fit on val only. No test posterior may influence
   any of these choices, directly or by re-running until a config "looks better."
3. **Test touches permitted:** **one** per declared pipeline variant, logged with a timestamp. To
   date: r2/unsafenet fusion = 1 touch (2026-09-05 23:44, report_fusion.json); fair-headtohead O/P
   pipelines = 1 touch each (k-fold and LCO are resampling *within* the pre-registered protocol, not
   additional touches, since the split scheme itself is fixed before scoring). Any additional touch
   must be declared here before it happens, with the reason.
4. **Pre-registered headline configuration rule:** the headline number for the thesis is **the blind
   Caruana-fusion result on the r2 pipeline, val-selected** (currently 50/64=78.12% top-1, 80.81%
   macro) reported alongside its Clopper-Pearson/bootstrap CIs, NOT the best test-touched number
   (55/64) — the latter may be shown once, clearly labeled "test-selected, for context only, not the
   claim." If T4 finishes the k-fold/LCO evidence for pipeline O, that becomes a **second**, equally
   valid headline (arguably the better one — narrower CI) and both must be reported side by side, not
   used to replace one with the other retroactively.
5. Every reported number carries: n, Clopper-Pearson CI (top-1) or bootstrap CI 5000 resamples
   (macro), seed count, and protocol tag (blind / test-selected / k-fold / LCO).

## 6. Go/no-go criteria
**STOP further training and write up now if any of:**
- A new config's blind test result falls inside the existing [66.0,87.5] top-1 / [70.6,90.2] macro CI
  — it is not a new result, it is the same result (3d: band is ~18-20pp, everything so far fits).
- The ConvLSTM-baseline run (T4 item 3, in progress at time of writing — `convlstm_smoke` process
  observed running during this audit) lands anywhere in 47-58/64 — that IS the paper's no-reduction
  regime and confirms claim (b) with the paper's own architecture; no further architecture search
  needed after that.
- Total test touches would exceed the count declared in §5 without a written justification here.
**CONTINUE (bounded) only for:**
- Finishing items already in T4's queue (ConvLSTM baseline, leak decomposition) — these complete the
  evidence for claim (b), they do not chase claim (a).
- The k-fold/LCO protocol for pipeline O (fair-headtohead), if not already final — this is the
  narrowest-CI evidence in the project and directly strengthens (b)/(c).
**NEVER:** re-run fusion/selection to try to beat 55/64 or 62/64 — §4 proves this is mathematically
closed at N=64.

## Unresolved questions
1. Exact N for the "34 legitimate configs" val/test correlation (thesis-process-review.md) doesn't
   match any subset I could reconstruct from the 46 stored `.pt` files (all-46 r=-0.247, 22-trained-
   heads r=-0.136, cited r=-0.066). Sign and non-significance replicate; exact figure doesn't. Does
   not change any conclusion, but the exact N=34 cut should be reconciled if it's quoted verbatim in
   the thesis text.
2. `fair-headtohead-scorecard.md` states "measured M4 + Jetson" — this audit found the Jetson numbers
   are literature citations for a YOLO11s backbone only, not our pipeline on real Jetson hardware.
   T5/whoever writes the defence chapter should correct this line before it goes in the thesis.
3. T4's ConvLSTM baseline (paper's actual architecture) was mid-run (`convlstm_smoke`, 2 epochs) at
   time of writing — not a real result yet. Go/no-go item 2 above depends on it landing; T1 did not
   wait for it per the "no heavy training" instruction.
4. Whether k-fold/LCO (fair-headtohead, pipeline O) should be re-run against the r2 pipeline
   (`unsafenet/`) for a single, unified headline, or reported as two separate pipelines with two
   separate evidence sets (current state) — a T5/committee-framing call, not resolved here.
