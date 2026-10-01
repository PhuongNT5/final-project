# Finding — two different numbers in this thesis are both "84.4 %", and the edge chapter is about to quote the wrong one

**Date:** 2026-10-01. **Severity:** high — it is a defence-room question, not a typo.

## 1. The collision

| label | model | metric | protocol | N | source |
|---|---|---|---|---:|---|
| **84.4 %** | **B1w** (333.570 params), whole-frame CLIP + 2 temporal layers | **balanced accuracy** | **P-D LSO-1100** | 1.100 | number-sheet line 67 |
| **84.4 %** | **P3**, entity detector + spatial-relation Transformer | **top-1** (54/64), 88,4 % macro | blind same-dist split | **64** | `report/abstract.md:14`, `chapter-06:109,284`, `chapter-05:14`, Ch7 |

Same three digits. Different model, different metric, different protocol, and a 17× difference in
sample size. Nothing in the text currently warns the reader, and `chapter-05` line 14 and
`chapter-06` line 14 carry the P3 84,4 % in a comparison table while the R48 lane's 84,4 % is B1w.

A reader who sees "84,4 %" beside a Jetson latency will not be able to tell which claim is being
made, and an examiner who notices the collision will ask which one the thesis is defending.

**Recommendation:** never write the bare string "84.4 %". Always write it as
`84,4 % balanced accuracy (B1w, LSO-1100, N=1.100)` or
`84,4 % top-1 (P3, blind split, N=64)`. Cheap, and it removes the ambiguity completely.

## 2. The edge chapter must quote 84.2 %, not either 84.4 %

The engine on the device is `r48-b1-clip-temporal-3seed-fp16.engine`, **603.270 params = 3 × 201.090**
— the 3-seed ensemble of **B1**, not B1w (a 3-seed B1w would be 1.000.710). So:

- B1's own OOF balanced accuracy is **84,2 %** (number-sheet line 68).
- The exporter already states the rule, and states it correctly
  (`deployment/scripts/export-b1-clip-temporal-head.py`, docstring): B1 was chosen over the
  "statistically indistinguishable B1w (84.2% vs 84.4%) on the pre-registered *edge* criterion --
  smallest model -- not on accuracy; **its own OOF number is the one to quote, never max(B1, B1w)**."

So any sentence putting 84,4 % beside a measured Jetson latency assigns one model's accuracy to
another model's runtime. §5.6.9 is drafted with **84,2 %** throughout and with the substitution
stated explicitly.

### A defect in the pre-registered prediction P4

Number-sheet line 323 writes P4's consequence as "FP16 latency may be quoted beside **84.4 %**
accuracy". That wording presumes the deployed head is the 84,4 % model. It is not. The **substance**
of P4 is unaffected and stands CONFIRMED — parity was measured on the B1 graph that is actually
deployed (agreement 1,000, max |Δp| 2,66·10⁻³, obtained twice) — so this is a wording defect in the
prediction, recorded rather than silently corrected, exactly as the prediction's own stride defect
was recorded for P2. The licence P4 grants should read **84,2 %**.

## 3. What would make the headline quotable on the device

If the thesis wants its Chapter 4 headline number (B1w, 84,4 % balanced accuracy) to sit beside a
measured edge latency, then a **B1w** head must be exported to ONNX, built as a TensorRT engine on
the device, and put through the same FP16 parity gate. That has **not** been done. The cost is
small and the measurement is cheap: B1w is 333.570 params against B1's 201.090, both are ~1,7 % of
decision latency, so the latency table would barely move — the point of doing it would be
*legitimacy*, not speed.

Until then the honest edge claim is: **84,2 % balanced accuracy (B1, LSO-1100, N=1.100) at
23,8 ms per decision on an Orin Nano Super 8 GB**, with B1w's 84,4 % noted as statistically
indistinguishable and not deployed.

## Unresolved questions

- Is the thesis headline P3's 54/64 or B1w's LSO-1100 balanced accuracy? `abstract.md` and
  `chapter-06` currently say P3; the R48 lane and the confirmed framing say B1/B1w. These cannot
  both be "the headline" without the text saying how they relate.
- Export a B1w engine to make 84,4 % quotable on device — worth the ~20 min, or accept 84,2 %?
