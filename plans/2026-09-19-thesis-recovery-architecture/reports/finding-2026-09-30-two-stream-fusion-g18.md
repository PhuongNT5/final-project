# G18 — two-stream fusion (TSF): pre-registered, run, FAILED

Date 2026-09-30. Protocol LSO-1100, `StratifiedGroupKFold(10, seed=0)` over 151 YouTube sources,
1,100 clips (420 unsafe). Identical folds, clips, recipe and seed count as every Block 1b row.
**0 test touches.**

## What was tested and why

G14 established the entity decomposition does not *replace* a whole-frame representation. It did
not test whether it *complements* one. The discordance pattern said it might: ESTA vs B1w had
b=72, c=78 (p=0.683) — equal accuracy, different errors.

Diagnostic on the existing OOF vectors (no training): error-correlation phi between the entity and
whole-frame arms = **0.49**, against **0.86** for two whole-frame variants of the same family
(the control that says the metric measures what it claims). Oracle "either arm right" = **91.2 %**
vs 84.4 % best-single: **+6.8 pp headroom**, 9.2 % of clips both-wrong.

That sweep touched 9 combinations and its fusion scores are **selection statistics, banned from
the thesis** (risks.md R49). G18 was registered — two tiers, verdicts and consequences written —
before `TwoStreamFusion` existed.

## Model

Both G14 arms kept whole (`EntitySpatioTemporal`, `WholeFrameTemporal`), their clip states fused by
attention over a 2-token sequence with a learned CLS and stream embeddings. Auxiliary head on each
arm at lambda=0.3 (inherited from this repo's binary relation heads, not tuned here).
**686,726 params.** Both streams read the same feature file, so the fusion adds no extraction cost
and cannot be fed a different corpus than its baselines.

Refactor safety: `encode`/`head` was split out of both arms so TSF reuses them verbatim.
`tests/test-two-stream-fusion-arms.py` (6/6) asserts the split is bit-exact, that TSF's aux logits
equal the standalone arms, that both streams reach the fused output, that gradients reach both
arms, that degenerate clips stay finite, and that arm param counts are unchanged.

## Result — FAIL

| comparison | diff | cluster CI | McNemar |
|---|---:|---|---|
| TSF 84.1 % − **B1w 84.4 %** | **−0.3 pp** | [−3.7, +2.5] | b=22 c=29 **p=0.401** |
| TSF − ESTA 83.8 % | +0.2 pp | [−6.4, +5.5] | b=81 c=82 **p=1.000** |
| TSF − B1xl 82.7 % (capacity) | +1.4 pp | [−0.3, +3.2] | b=22 c=12 p=0.121 |

Neither tier met. Pre-registered consequence applied verbatim.

**The mechanism of the failure is legible.** ESTA vs B1w disagreed on 150 clips; TSF vs B1w
disagrees on **51**, and TSF vs its own entity arm is b=81 c=82, p=1.000. The fusion **collapsed
onto the whole-frame stream**. The aux loss kept the entity arm trained but not influential.

**Oracle headroom is not reachable without an oracle.** +6.8 pp exists only for something that
knows which stream to trust per clip. At 1,100 clips / 151 sources no such selector was learnable.
This is a claim about the selector, not about the streams — and it is the honest, transferable
lesson from the run.

## Capacity — a clean positive result, independent of G18

| model | params | balanced acc |
|---|---:|---:|
| B1 | 201,090 | 84.2 % |
| B1w | 333,570 | **84.4 %** |
| TSF | 686,726 | 84.1 % |
| B1xl | 696,962 | **82.7 %** |

B1xl was registered *before* the TSF result was visible, chosen by the rule "the whole-frame
configuration whose parameter count is closest to TSF's" — landing 1.5 % **larger** than TSF. Over
a 3.5x parameter range the whole-frame arm gains +0.2 pp then loses 1.7 pp. **Capacity is ruled out
as the mechanism, in both directions.** Quotable.

Ban: TSF's +1.4 pp over B1xl may **not** be cited for the fusion. It was registered as a condition
on a passing claim; there is no passing claim.

## What the lane now establishes

1. **The appearance representation is the binding constraint.** MobileNetV3-small 64-d (E0,
   80.3 %) -> CLIP ViT-B/32 (B1w, 84.4 %) = **+4.0 pp**, McNemar p=3.01e-04.
2. **Capacity is not the mechanism** — 3.5x parameter range, controlled both ways (above).
3. **Entity decomposition is not the mechanism** — three independent pre-registered tests agree:
   G14 (vs whole-frame, p=0.683), G17 (detector generation; the *newer* YOLO26s is 2.2 pp worse,
   p=0.00412), G18 (fusion, p=0.401).
4. **The cost is a backbone purchase, measured.** CLIP tower 375.0 ms vs B1 head 0.3 ms per
   16-frame clip on host CPU — backbone = 99.9 % of wall clock, 18.3x the MobileNetV3 graph it
   replaced. Host arm64 only; **no R48 graph has been measured on the Jetson**.
5. **Every cluster CI in this lane includes 0.** At 151 sources with 0 carrying both classes (R41),
   no 3-4 pp margin is certifiable at cluster resolution. Clip-level CIs must never be the headline.

This is a coherent systematic ablation: for clip-level unsafe-behaviour detection on iSafetyBench,
representation dominates and structural elaboration does not help. It is a positive methodological
finding, not a critique of prior work.

## Unresolved questions

1. **Framing decision, still the author's.** Headline B1/B1w (84.4 %) with YOLOv11 as localisation
   + edge subject — the pre-registered consequence of G14/G18 — or something else? Three gates now
   point one way, but the author owns the thesis narrative.
2. **Is the corpus the real ceiling?** R41 (0 of 151 sources carry both classes) means source
   identity nearly determines label, which is why every cluster CI is wide. A corpus with
   within-source label variation would make margins of this size certifiable. Author previously
   raised using a more suitable dataset; with 3-10 days left this is a scope decision, not a
   technical one.
3. **Would the selector become learnable with more data?** The oracle gap is real; only the
   selector failed. Untested and not testable on this corpus.
4. **Jetson measurement for the R48 graphs is unstarted** — author operates the device.
