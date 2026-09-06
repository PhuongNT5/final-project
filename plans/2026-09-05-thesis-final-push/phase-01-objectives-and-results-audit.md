# T1 - Objectives & results audit -> master plan

**Priority:** P0 (gates T3). **Status:** ready. **Owns:** `plan.md`, this file.

## Overview
Decide what result this thesis will actually claim, and what evidence closes it. Everything downstream
is scheduling; this is the only track that changes the target.

## Key insights carried in
- The "improvement ladder" 52 -> 53 -> 55/64 across rounds is inside a +/-11 pp bootstrap band. It is
  resampling, not progress.
- Val/test correlation r = -0.066 -> model selection on 83 val clips cannot pick a test winner.
- Paper's headline advantage (62 vs 57) is itself not significant (McNemar p >= 0.0625).
- The paper's YOLO stage is now reproduced and working (mAP50 0.79-0.81) but has never been fed
  through to a clip decision with a *good* detector.

## Requirements
- R1 State the thesis claim in falsifiable form. Candidates:
  (a) "beats 62/64" - requires >= 63/64 blind, p < 0.05 vs paper. Assess feasibility honestly.
  (b) "statistically indistinguishable from the paper at N=64, and here is the proof that N=64 cannot
      separate any method in this range" - a *negative result with teeth*.
  (c) "matches paper accuracy at X x lower latency on edge hardware" - shifts the axis to deployment,
      where our contribution is real and measurable.
- R2 Rank (a)/(b)/(c) by achievability given measured variance; recommend one, keep one as fallback.
- R3 Define the frozen evaluation protocol BEFORE more training (splits, selection rule, test touches).
- R4 Define go/no-go: what result would make us stop training and write up.

## Implementation steps
1. Tabulate every run so far: config, val macro, test top-1/macro, protocol used.
2. Recompute the bootstrap band on 64 clips; state the minimum detectable difference at N=64.
3. Compute what top-1 is needed for p<0.05 vs 62/64 under McNemar - report the required clip count.
4. Cross-check each of (a)/(b)/(c) against that number.
5. Write the recommendation + frozen protocol into this file; update `plan.md` critical path.

## Todo
- [x] Run table across rounds 0-2
- [x] Minimum detectable effect at N=64
- [x] Clips required to beat 62/64 significantly
- [x] Claim recommendation (a/b/c) + fallback
- [x] Frozen evaluation protocol
- [x] Go/no-go criteria

## Success criteria
A one-paragraph thesis claim that (i) is falsifiable, (ii) has a number attached, (iii) survives the
variance analysis, and (iv) T3/T4 can act on without further interpretation.

## Risks
- Choosing (a) when the data cannot support it burns weeks. Mitigation: step 3 answers this numerically
  before any training is queued.

## Recommendation

Full numbers, tables, and derivations: `reports/t1-objectives-audit.md`. Summary here.

**Claim (a) "beat 62/64" is mathematically closed at N=64, not just unlikely.** Paper is wrong on only
2/64 clips; McNemar/Fisher on our_correct=64 (a hypothetical zero-error run) still gives p=0.50 vs
paper's 62/64 — 2 discordant pairs can never reach p<0.05 under any test. No amount of training closes
this. Excluded.

**Recommended primary claim: (b) statistically indistinguishable from the paper at N=64, and N=64
cannot separate any method in this range.** Already fully evidenced: Clopper-Pearson CIs overlap for
every k we've produced (50-56) vs paper's 62; bootstrap macro CI is 16-20pp wide (our whole result
history, 78-89.5% macro across 3 rounds, fits in one band); val/test correlation is small/negative/
non-significant across every subset re-checked (r in [-0.25,-0.07], all p>0.05); the paper's own
62-vs-57 internal ablation is itself n.s. (p>=0.0625). Cost: near zero, mostly writing — the evidence
exists now.

**Recommended supporting claim: (c) match (paper no-reduction row 89.06%/85.78%, not the full 96.88%)
at measured lower latency.** M4 end-to-end 26.8-28.8ms p50 is a real measurement; paper reports zero
latency numbers so "faster" is true by default but weak alone — pair with (b), don't lead with it.
**Correction found during this audit:** `deployment/artifacts/projections.md` "Jetson" row is a
literature citation (Ultralytics/arXiv YOLO11s-backbone-only numbers), NOT a measurement of our
pipeline on Jetson hardware. Do not claim "measured on Jetson" — say "measured on M4; Jetson figures
are literature context" or run a real Jetson pass before claiming it.

**Fallback framing if (b) alone reads too negative for the committee:** merge (b)+(c) into one claim —
"statistically-tied accuracy, zero annotation cost, measured lower latency" — same evidence,
deployment-value framing instead of pure negative result.

**Frozen protocol (binding for T3/T4), condensed — full version in the report:** splits fixed (452
train / 64 test per paper; 83-clip stratified val carved from train); selection ONLY on val macro;
Caruana fusion weights fit on val only; one test touch per declared pipeline variant, logged; headline
= blind val-selected result (currently r2 fusion 50/64=78.12%/80.81% macro), test-selected number
(55/64) may be shown once labeled "for context only, not the claim"; every number carries n + CP or
bootstrap CI + seed count + protocol tag.

**Go/no-go, condensed:** stop and write up if a new blind result lands inside the existing
[66.0,87.5]/[70.6,90.2] CI (it already will — that band is ~18-20pp wide); stop after T4's ConvLSTM
baseline (paper's real architecture, in progress at time of writing) lands anywhere in 47-58/64 — that
confirms (b) with the paper's own method, no more architecture search needed. Never re-run
fusion/selection to chase past 55/64 or 62/64 — closed per above.
