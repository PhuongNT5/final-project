# Jetson runbook — 19/10/2026: what the round-3 towers cost on the device

Lane: R48 / B1 (iSafetyBench, binary). Device: Jetson Orin Nano **Super** 8 GB,
`phuongnguyen-desktop`, JetPack R36.5.0, TensorRT 10.3.0. Power mode stays
**MAXN_SUPER (mode 2)** — the board idles there and this session does not change it.

## Why this session exists

Round 3 measured eight frozen vision towers on the 1,100-clip iSafetyBench LSO protocol and
**no tower survived** the pre-registered Benjamini–Hochberg correction (smallest p = 0.093 against
a rank-1 threshold of 0.05/8 = 0.00625). The accuracy question is therefore closed and stays
closed: nothing here can reopen it.

What is still open is a *deployment* question, and it is open in both directions:

- `siglip2B16` + mean-pool reached 87.25 % balanced accuracy (+4.00 pp over the 83.25 % four-thread
  control). Even unconfirmed, that row is the best evidence of how much representational headroom
  the 151-source corpus still contains. It can only enter the report as an **upper bound on that
  headroom** — never as a headline candidate — and only if its device cost is known. A number
  whose cost is unknown cannot be discussed as a deployment option at all.
- `mobileclip_s1` tied the control (+0.47 pp, p = 0.93) while being *smaller* than the deployed
  B/32. If it is also cheaper on the device, the thesis gains a statable result: the deployed
  accuracy is reachable at lower cost. If it is not cheaper, that is also a result.

The gate is pre-registered and is not renegotiated here: **≥ 3 decisions/s**. The published
composed cadence of the deployed B/32 path is **289.672 ms/decision = 3.452 dec/s** — a margin of
only 1.15×. Every number below is compared to that, and to the published isolated tower figure
**23.386 ms** (FP16, batch 16).

## Predictions, written before any tower was exported

Each prediction names the observation that would refute it. A refuted prediction is reported as
refuted; none of them is rewritten after the fact. (The 01/10 runbook's P1 was refuted this way
and the refutation became §5.7's most useful content.)

**P1 — a B/16 tower costs 3–4.7× the B/32 tower in isolation: 70–110 ms at FP16 batch 16.**
Basis: 17.6 GFLOPs/image against 4.4, i.e. exactly 4× the arithmetic, at the ~3.0 TFLOP/s
effective rate the B/32 engine already achieves on this board. The band is wider than 4× in both
directions because B/16 presents 197 tokens instead of 50, which changes how well the GEMMs fill
the SMs — more parallelism could beat the FLOP ratio, a larger attention share could lose to it.
*Falsifier:* a measured isolated latency outside 70–110 ms.

**P2 — the in-loop penalty shrinks for B/16: the in-loop/isolated ratio is below 2.51×.**
Basis: §5.7 adjudicated the B/32 path's 2.51× (58.630 ms in loop vs 23.386 isolated) to GPU clock
droop across the ~210 ms idle gap between decisions. A tower that occupies the GPU 4× longer
spends proportionally less of each decision idle, so if droop is the cause the penalty must fall.
*Falsifier:* a ratio at or above 2.51×, which would mean the penalty is not droop-driven and the
01/10 adjudication is wrong.

**P3 — a B/16 tower breaks the 3 dec/s gate.** Arithmetic on the published decomposition:
289.672 − 58.630 + (in-loop B/16). Even at the most favourable possible value — no droop penalty
at all, in-loop = isolated = 93.5 ms — the decision lands at 324.5 ms = 3.08 dec/s, barely above
the gate; at P2's expected ~2× it lands near 418 ms = 2.39 dec/s. So `siglip2B16`, `clipB16` and
`openclipB16_l2b` are predicted **not deployable at `--window 48 --stride 24`** on this board.
*Falsifier:* a measured composed cadence ≥ 3.0 dec/s for any B/16 tower.

**P4 — `mobileclip_s1` is free: isolated ≤ 23.386 ms and composed cadence ≥ 3.452 dec/s.**
Basis: fewer parameters and fewer FLOPs than the deployed B/32.
*Falsifier:* either figure worse than the deployed B/32's. This is the prediction most likely to
fail and the most interesting if it does: MobileCLIP's published advantage is measured on mobile
CPU/NPU latency, and a hybrid conv-transformer at batch 16 on TensorRT need not inherit it. A
failure here would be the **third** counterexample this project has recorded to transferring a
published benchmark ranking to this setting, after VideoMAE (−2.37 pp) and DataComp-1B/DFN-2B
(weakest towers despite the best zero-shot scores).

**P5 — parity gates every quoted number.** Each tower's ONNX must match its PyTorch embedding to
`max|diff| ≤ 1e-4` in fp32, and each FP16 engine must match its ONNX to a stated per-frame cosine.
*Falsifier / rule:* a tower that fails parity gets **no latency number quoted anywhere**, because
the latency of a graph that computes something else is not the latency of the tower.

## Decision rule, also pre-registered

1. A latency figure may be quoted only from an artifact whose `quiet` and `clocks_pinned` fields
   were recorded by `jetson-clock-state-utils.py` and are both consistent with the published
   protocol. The 01/10 session measured a **1.79× spread** on byte-identical provenance because a
   second session shared the board; the guard, not a pre-flight check, is what prevents that.
2. A tower is *deployable* only if its composed cadence is ≥ 3.0 dec/s. Deployability is a
   necessary condition for a headline, never a sufficient one: the BH verdict already denies every
   round-3 tower a headline regardless of what this session measures.
3. The head stage is charged at the **published 512-d head** for every tower. For `siglip2B16` the
   deployed head would be 768-d (233,858 vs 201,090 parameters, +16.3 %); the resulting error is
   bounded by 0.163 × the measured head stage and that bound is reported with the number rather
   than absorbed into it.
4. Preprocessing (160.765 ms = 73.4 % of the B/32 path's compute) is **unchanged** across towers —
   all eight round-3 towers consumed the same 224×224 cached frames, so the only moving part is the
   tower stage. Any measured change in the preprocessing stage is a measurement fault, not a
   tower effect, and aborts the comparison.

## Steps

1. `deployment/scripts/export-round3-towers-onnx.py` — export `clipB16`, `siglip2B16`,
   `mobileclip_s1` at 224×224 with dynamic batch and the ONNX-parity gate (P5, first half).
   224 is not a choice here: it is the resolution the round-3 features were extracted at, so an
   export at any other resolution would not be the measured tower.
2. `LANE=r48 JETSON_HOST=jetson bash deployment/scripts/jetson-rsync-from-mac.sh`.
3. On device: `jetson-build-trt-engine.py --mode fp16 --model <tower> --batch_profile 1:16:16`.
   The batch profile is mandatory — a batch-1 engine forces a 16-iteration loop and overstates the
   tower by charging 16 kernel launches instead of one batched call.
4. Device parity (P5, second half), then the clean-window guard: poll until no measurement process
   is running and clocks are restored, settle 90 s, measure, re-read state, and **abort rather than
   report** if the window turned dirty mid-run.
5. `benchmark-r48-clip-pipeline.py --tower <stem>` for the isolated stage (P1, P4) and
   `benchmark-r48-composed-cadence.py --tower <stem>` for the composed cadence (P2, P3, P4).
6. Adjudicate P1–P5 in writing, including every refutation.

## Outcome

Measured 2026-10-19 23:35–23:47, JetPack R36.5.0 / TensorRT 10.3.0, MAXN_SUPER (mode 2, unchanged),
clocks UNPINNED (DVFS) as the baselines were. All six runs were wrapped by
`jetson-clean-window-guard.py`; every one recorded `clean_before` and `clean_after` true with no
dirty reasons, which is what licenses quoting them. Verdict files:
`deployment/artifacts/round3-device/guard-*.json`.

### What the device measured

Isolated stage, FP16, batch 16, 20 runs after 3 warmups:

| tower | p50 | p95 | vs today's B/32 | GFLOPs/decision | effective TFLOP/s |
|---|---|---|---|---|---|
| `r48-clip-vit-b32-vision` (published, control) | 31.5 ms | 35.8 ms | 1.00x | 70.4 | 2.24 |
| `r48-tower-clipB16` | 78.8 ms | 84.8 ms | 2.50x | 281.6 | 3.57 |
| `r48-tower-siglip2B16` | 82.1 ms | 85.5 ms | 2.61x | 281.6 | 3.43 |

Composed cadence, `--window 48 --stride 24`, 12 clips / 109 decisions:

| tower | prep p50 | tower p50 | head p50 | compute/dec p50 | steady-state | dec/s |
|---|---|---|---|---|---|---|
| B/32 (control) | 50.707 ms | 57.441 ms | 0.859 ms | 109.922 ms | 169.843 ms | **5.888** |
| clipB16 | 53.940 ms | 103.683 ms | 0.837 ms | 159.014 ms | 228.629 ms | **4.374** |
| siglip2B16 (768-d cost head) | 54.502 ms | 104.440 ms | 0.818 ms | 159.431 ms | 227.582 ms | **4.394** |

### P1 — a B/16 tower costs 70–110 ms in isolation: CONFIRMED as stated, refuted in its reasoning

78.8 ms and 82.1 ms both land inside the pre-registered band. But the band was derived as 3–4.7x the
*published* 23.386 ms, and the measured ratio against the control taken in the same session is
**2.50x and 2.61x** — below the predicted ratio range. The absolute prediction was right; the ratio
argument that produced it was wrong. A B/16 tower delivers 3.57 against 2.24 effective TFLOP/s,
i.e. **1.60x more arithmetic per unit time** than B/32, because 197 tokens fill the SMs where 50 do
not. The pre-registration widened the band for exactly this reason but still mis-signed the effect.

### P2 — the in-loop/isolated ratio falls below 2.51x: CONFIRMED

clipB16 103.683 / 78.8 = **1.316x**; siglip2B16 104.440 / 82.1 = **1.272x**; today's B/32 control
57.441 / 31.5 = **1.823x**. The predicted ordering also holds inside today's session: the 4x-longer
tower spends less of each decision idle and loses less to clock droop. The published 2.507x
(58.630 / 23.386) does not reproduce — see the drift section; its direction does, its magnitude
does not.

### P3 — a B/16 tower breaks the 3 dec/s gate: **REFUTED**

clipB16 **4.374 dec/s** and siglip2B16 **4.394 dec/s**, both clearing the pre-registered 3.0 dec/s
floor with ~46% headroom.

Why the prediction was wrong is a bookkeeping error of mine, not a device surprise: its arithmetic
started from the 289.672 ms composed baseline with a 160.765 ms preprocessing stage, which is the
**pre-fix** branch of Bảng 5.2 (commit 9c1e219). The submitted thesis's own claim is the post-fix
branch, 167.291 ms with ~50 ms preprocessing, and that is what the device reproduces (see the
reproduction table below, every stage within 2%). Carried out with the number the thesis actually
claims, the B/16 budget is 228.6 ms/decision = 4.37 dec/s -- which is what was measured. I
pre-registered a prediction against a figure my own report had already superseded; the device
behaved exactly as the correct premise implies.

The pre-registration above is left as written, including the superseded premise. A pre-registration
that is edited after the measurement records nothing.

Consequence, stated precisely: **the deployment objection to `siglip2B16` is removed; the
statistical one is not.** Its +4.00 pp carries p = 0.093 against a BH rank-1 threshold of 0.00625,
so 87.25% remains ineligible as a headline. Clearing a deployment gate does not promote a number
that failed the multiple-comparison correction — the two gates are conjunctive, as pre-registered.

### P4 — `mobileclip_s1` is free: **REFUTED**, for two independent reasons

First reason, found before any timing: its FP16 engine does not reproduce its own FP32 tower --
min row cosine **0.932**, rel **0.259**, with the preprocessed-pixel checksum matching the host at
0, so the input is not the cause. Under P5 that alone left P4 unsettleable, because a tower whose
engine is wrong has no quotable cost.

Second reason, measured afterwards at FP32 (see the FP32 section below): it is **2.83x slower than
B/32**, 151.8 ms against 53.6 ms in isolation at the same precision. So P4 fails twice over --
the engine is wrong, *and* the tower is not cheap. Either one is sufficient to retire it.

This is a stronger deployment result than a merely slow number would have been. The tower with
**4.08x fewer image-encoder parameters** (21,533,632 against 87,849,216) is not deployable through
TensorRT 10.3 on this device at *either* precision, and where it does run it is the more expensive
of the two. Its size advantage is unusable here -- not postponed to QAT or to an FP32 path, since
the FP32 path was built and fails too. It is the third counterexample in this project to
transferring published benchmark rankings onto this task, after VideoMAE (−2.37 pp) and the
DataComp-1B / DFN-2B zero-shot inversion, and the first where the published metric (parameter
count) points the opposite way from the measurement by a factor of 2.83.

**The cause is NOT precision.** An FP32 engine was built and measured on 2026-10-20 at the
request of the supervisor, with the parity tolerance tightened to match the precision claimed
(cos >= 0.99999, rel <= 1e-4; the fp16 budget is ~200x looser and would have certified nothing).
It fails too, and by almost exactly the same margin:

| engine | batch 1 | batch 2 | batch 4 | batch 16 |
|---|---|---|---|---|
| fp32, min row cosine / rel | 0.948259 / 0.241 | 0.948259 / 0.241 | 0.941337 / 0.241 | 0.932605 / 0.247 |
| fp16, min row cosine / rel | 0.948011 / 0.241 | 0.947883 / 0.241 | 0.940836 / 0.241 | 0.932087 / 0.259 |

The error is invariant across two precisions and four batch sizes. FP32 carries ~1e-7 relative
resolution, so a precision mechanism would have to show ~0.006% error here, not 24.1%. The chain of
evidence closes on the conversion: the ONNX graph is correct (host torch-vs-ONNX rel 1.22e-05, cos
1.000000), the device's input is bit-identical (pixel checksum difference exactly 0), and the engine
output is wrong at every precision and batch. **TensorRT 10.3.0 mis-converts the MobileCLIP-S1
graph**, so the tower is not deployable through TensorRT on this device *at all* — not merely
"not in FP16". Localising the offending op would require bisecting the graph with truncated
engines and was not done; the deployment conclusion does not depend on it.

### A rejected explanation, recorded because it was wrong

Before the FP32 engine existed I proposed a precision mechanism and it looked quantitatively
convincing: mobileclip's embedding scale is 0.2083 while its signal-path activations reach 121.9,
a **585x** ratio, and FP16's relative resolution at intermediate scale (4.88e-4) maps to 0.0595
absolute = **28.6%** of the embedding scale, against 25.9% measured. The same arithmetic gave
clipB16 0.74% predicted against 0.48% measured. Two towers, both close.

It is wrong. The FP32 engine fails identically, which no precision argument can produce. The
agreement was a coincidence of magnitude, and the second "confirmation" (clipB16) was consistent
with the mechanism only because that tower's error is genuinely small for other reasons. The
warning sign was already visible and I under-weighted it: the same arithmetic over-predicted
siglip2B16 by 28x (13.4% predicted, 0.47% measured, its largest signal-path tensor 1,548.7 in
`layers.9/mlp/fc1` being renormalised downstream). A rule that fits two cases, misses a third by
28x, and is then contradicted outright by a controlled test is not a mechanism. The scale ratio is
kept here only as a screening statistic for which towers deserve a parity test, and the episode is
kept as the reason not to accept a numerical coincidence as a cause.

### Separate landmine, unaffected by the above

siglip2's exported graph holds **3.64e9** inside the GELU-tanh decomposition (the x^3 term,
`layers.9/mlp/activation_fn/Mul_1`), which overflows FP16's 65,504 by 5e4. TensorRT 10.3 fuses the
pattern and the engine passes parity, so this is a property of this runtime, not of the model;
another runtime executing that decomposition literally would break.

### `mobileclip_s1` at FP32: the cost of a graph that does not reproduce the tower

Measured 2026-10-20 at the supervisor's explicit direction, choosing a labelled FP32 number over
P5's silence. **An FP32 number is not comparable to any FP16 baseline in this report**, so a B/32
FP32 control was taken in the same session, under the same guard, to give it a same-precision
reference. Both isolated runs carry `VERDICT: clean before and after`.

| isolated, FP32, batch 16 | tower p50 | tower p95 | per decision | decisions/s |
|---|---|---|---|---|
| r48-clip-vit-b32-vision (control) | 53.6 ms | 56.4 ms | 54.1 ms | 18.50 |
| r48-tower-mobileclip_s1 | 151.8 ms | 169.2 ms | 152.4 ms | 6.56 |

**mobileclip_s1 is 2.83x SLOWER than B/32 at the same precision, with 24.5% of its parameters**
(21,533,632 vs 87,849,216 ONNX initializers). The premise behind P4 — that a small mobile-targeted
tower is the cheap option — is false on this runtime: parameter count predicts the wrong direction
by a factor of 2.83. MobileCLIP-S1 is a conv/transformer hybrid, and TensorRT 10.3.0 evidently
serves its graph badly; that is the same runtime that mis-converts it numerically, and the two
observations are consistent with one cause, though nothing here proves they share it.

Two caveats bind this number and must travel with it anywhere it is cited:
1. It is FP32. Every tower baseline in Chapters 5-6 is FP16. The pairing above is the only
   comparison it licenses.
2. **The engine fails parity at FP32** (cos 0.932605, rel 0.247 against cos >= 0.99999, rel <= 1e-4).
   So 151.8 ms is the cost of executing a graph that does not reproduce MobileCLIP-S1's embeddings.
   It bounds what a *correct* engine would cost only if the mis-conversion is cost-neutral, which
   is unknown. It is quotable as evidence that this tower is not the cheap escape route; it is NOT
   quotable as "MobileCLIP-S1 costs 151.8 ms on Orin Nano".

The MobileNetV3-small reference inside these two runs read 10.1 and 10.2 ms -- agreeing to 1%,
unlike the 11.2 / 7.0 / 7.0 ms spread recorded in the FP16 session below. The instability recorded
there is therefore not a fixed property of that engine either.

### The FP32 composed cadence, complete

The first attempt lost its fourth run because **the board was powered off at the operator's
request** while the chain was still running: SSH died with `client_loop: send disconnect: Broken
pipe`, then ICMP went to 100% loss. Nothing about the device failed, and no measurement below is
evidence about device reliability. The fault was in how I scheduled the work -- a four-run chain
bound to one SSH session, with no statement anywhere of how long the board would be needed. The
re-run was detached with `setsid` so the next shutdown or dropped link cannot take a measurement
with it. The pair was re-taken the same evening after the reboot, **both runs in one session**,
because the drift evidence in this document says a control from another session is not
interchangeable. Every run below carries
`VERDICT: clean before and after`; the re-run guard refused for the first 90 s while the
freshly-booted desktop drove the GPU to 46%, which is the hardened guard doing its job.

| FP32, 12 clips / 109 decisions | tower p50 | prep p50 | compute p50 | steady-state p50 | p95 | dec/s | duty | steady/sum |
|---|---|---|---|---|---|---|---|---|
| B/32 control, 19:39 (re-run) | 87.66 ms | 52.58 | 140.43 | **204.293 ms** | 232.31 | **4.895** | 0.680 | 1.043 |
| mobileclip_s1, 19:41 | 161.31 ms | 52.45 | 214.92 | **281.495 ms** | 308.94 | **3.552** | 0.769 | 1.047 |
| B/32 control, 07:29 (before the failure) | 86.82 ms | 51.89 | 139.08 | 202.717 ms | 240.03 | 4.933 | 0.679 | 1.042 |
| B/32 control, 19/10 — **FP16, for scale only** | 57.44 ms | 50.71 | 109.92 | 169.843 ms | 199.23 | 5.888 | 0.630 | 1.037 |

Four readings, in order of what they settle.

**1. `mobileclip_s1` holds 3.552 dec/s at FP32 — it clears the 3 dec/s gate by only 1.18x**, against
the B/32 control's 4.895 dec/s in the same session. In cadence it is **1.378x slower** than the
tower it was supposed to cheapen, and its tower stage alone is **1.840x slower** (161.31 against
87.66 ms). Both caveats from the isolated section still bind: this is FP32, comparable to nothing
in Chapters 5-6, and the engine **fails parity at FP32**, so 3.552 dec/s is the cadence of a graph
that does not reproduce MobileCLIP-S1. It is quotable as evidence against the tower, never for it.

**2. The control reproduces across a full power cycle to within 1%**: 204.293 against 202.717
ms/decision (+0.78%), tower stage 87.66 against 86.82 ms (+0.96%) -- the two runs sit on either side
of a shutdown, a cold boot, and a desktop session that was still loading the GPU when the guard
first looked. Together with the within-2% reproduction of Chapter 5's post-fix column, the
composed-cadence harness is the most reproducible measurement in this project: three sessions,
16 days, two power cycles.

**3. The isolated-to-in-loop factor is not a property of the loop; it is a property of the tower.**
At the same precision, in the same session: B/32 reads 1.635x its isolated cost in the loop
(87.66 / 53.6), while mobileclip_s1 reads **1.063x** (161.31 / 151.8). A single "in-loop penalty"
constant does not exist, so P2's 2.51x ceiling and the 1.27-1.32x values measured at FP16 are
statements about particular towers, not about the loop. The direction is consistent with gap-filling:
the shorter a tower's kernel, the larger the share of per-decision CPU and launch overhead it
cannot hide -- but that is an interpretation, and nothing here measures the overhead directly.

**4. Precision costs the B/32 lane 1.20x in cadence** (204.293 FP32 against 169.843 FP16, same clip
set, same engines otherwise) and 1.53x in the tower stage alone (87.66 against 57.44). Quoted only
as the internal scale between this document's two precisions; no Chapter 5-6 figure is FP32.

### P5 — parity gates every quoted number: HELD

Host parity on real frames, batch 16: clipB16 rel 2.05e-06 / cos 1.000000; siglip2B16 rel 4.56e-06 /
cos 1.000000; mobileclip_s1 rel 1.22e-05 / cos 1.000000. Device FP16-vs-FP32 parity: clipB16 cos
0.999973 / rel 4.76e-03 PASS; siglip2B16 cos 0.999918 / rel 4.73e-03 PASS; mobileclip_s1 FAIL and
quoted with no latency.

### The submitted Chapter 5's device numbers DO reproduce — correcting what this document first said

**This section replaces a wrong claim I made in it.** I first wrote that two published figures --
289.672 ms/decision = 3.452 dec/s, and preprocessing = 73.4% of compute -- do not reproduce, and
that they should go to the supervisor as a proposed correction. That was an error of attribution,
not of measurement. The numbers in the table below are unchanged; what was wrong was comparing a
new run against the *pre-fix* branch while the submitted chapter had already superseded it.

`report/vn/chuong-05-trien-khai-bien.txt` (Bảng 5.2) presents those two figures only as the
**TRƯỚC** column of an explicit before/after pair, attributed to commit **9c1e219** and measured
01/10, with the post-fix column **SAU: 6319b21** standing as the thesis's actual claim. The chapter
already states, with matched clip sets (24 clips, 5,559 frames, 201 decisions, `per_clip` lists
compared equal) and named source artifacts, that a preprocessing fix cut that stage from 160.765 to
50.867 ms. So the 3.16x gap is not an unexplained anomaly; it is the documented effect of a code
change, and the thesis's current cadence claim is 167.291 ms = 5.978 dec/s, not 3.452 dec/s.

Row by row, the submitted post-fix column against this session's control, 16 days later, on a
12-clip subset rather than the full 24 (so this is a reproduction on overlapping-but-smaller data,
not a bit-level replication):

| quantity | Ch5 SAU (03/10, 24 clips, 201 dec) | 19/10 control (12 clips, 109 dec) | difference |
|---|---|---|---|
| steady-state p50 | 167.291 ms | 169.843 ms | +1.5% |
| preprocess p50 | 50.867 ms | 50.707 ms | −0.3% |
| tower p50, in loop | 58.052 ms | 57.441 ms | −1.1% |
| head p50 | 0.842 ms | 0.859 ms | +2.0% |
| compute per decision p50 | 109.114 ms | 109.922 ms | +0.7% |
| duty cycle | 0.631 | 0.630 | −0.2% |
| preprocessing share of compute | 46.6% | 46.1% | −0.5 pp |

**Every stage of the submitted claim reproduces within 2%.** That is a positive provenance result
and belongs in the report as one: it is the strongest available evidence for Y6 (self-measured,
re-measurable) on the deployment numbers, because it is the same claim re-measured on the same
board after a 16-day gap, three JetPack-level reboots, and an engine rebuild.

For the archive, the full three-session picture that produced the original mistake:

| session | prep p50 | tower p50 (in loop) | steady-state |
|---|---|---|---|
| 01/10, three runs — **pre-fix, commit 9c1e219** | **160.8 ms** | 58.0 / 58.9 / 58.6 ms | 289.7 / 289.3 / 292.8 ms |
| 03/10, three runs — **post-fix, 6319b21; the submitted claim** | 48.9 / 51.4 / 50.9 ms | 58.4 / 58.1 / 57.4 ms | 167.0 / 167.7 / 167.3 ms |
| 19/10, this session (12 clips) | 50.707 ms | 57.441 ms | 169.843 ms |

The GPU-bound tower stage is stable within 1% across all 18 days and all three sessions, including
the pre-fix one -- which is itself the evidence that the 01/10 difference was confined to the
CPU-side stage that was then rewritten, exactly as the chapter says.

One caveat from this section survives, because it was never about these figures: all three 01/10
runs recorded `quiet: true`, and that flag comes from `competing_gpu_processes()`, which this
session proved blind to any process outside a name whitelist frozen that same day -- it called the
board clean while a TensorRT build ran on it. `quiet: true` on any pre-20/10 artifact therefore
certifies less than it appears to. That weakens the *guarantee* attached to old artifacts; it does
not contradict them, and the reproduction above is the reason it no longer needs to.

**Nothing here goes to the supervisor as a correction, and no Chapter 6 or Chapter 5 row changes.**
The one file that does headline 289.672 ms = 3.452 dec/s as a current result is
`chuong-05-trien-khai-bien-ghim-xung-nhip.txt`, which is already archive-only and excluded from
submission; this reproduction is a further reason it must stay excluded.

### A second instability, in the isolated benchmark itself

Within this one guard-clean session the *same* MobileNetV3 engine at the same batch measured
**11.2 ms, then 7.0 ms, then 7.0 ms** in three runs 90 s apart — a 1.60x spread on an unchanged
graph. The isolated B/32 tower likewise reads 31.5 ms today against the published 23.386 ms
(+34.7%), while its in-loop value reproduces within 1%. So the isolated stage is the *least*
reproducible quantity measured here and must not carry three-significant-figure claims. This is an
independent second reason for the existing "a stage sum is not a cadence" finding: the stages are
not only non-additive, they are individually unstable, whereas the composed loop is not.

### Deviations from this pre-registration, recorded

1. Decision rule 4 required preprocessing to be *unchanged* across towers and made any change an
   abort. It was written with no tolerance, which is a drafting error — no stage is bit-identical
   across runs. Measured: 50.707 / 53.940 / 54.502 ms, a 7.5% spread. Not treated as an abort,
   because the quantity under comparison (the tower stage) differs by 80% and a 7.5% preprocessing
   spread cannot account for it. Stated here rather than silently waived.
2. Decision rule 3 bounded the 768-d head error at 0.163x the measured head stage. Superseded by
   direct measurement: the 768-d cost head measures 0.818 ms against the 512-d head's 0.859 ms, and
   the head is under 1% of the decision budget either way, so the bound is moot.
3. The clean-window guard was strengthened mid-session, before any timing run, after it declared a
   window clean while an engine build was running. It now samples peak GPU utilisation
   (`/sys/devices/platform/gpu.0/load`) and detects processes by repo path rather than by a name
   whitelist. `jetson-clock-state-utils.py` was deliberately left unmodified because published
   provenance fields record its output.

4. **FP32 latency was measured and is quoted, against P5's rule.** P5 pre-registered that a tower
   failing parity gets no latency quoted anywhere. The supervisor was shown that consequence and
   directed the alternative: measure at FP32 and label it. The label is carried in full in the FP32
   section (not comparable to FP16; the engine fails parity at FP32 too), and the B/32 FP32 control
   was added so the number has a same-precision reference. P5 is otherwise unchanged: no FP16
   mobileclip latency appears anywhere.
5. **One run was lost to a requested shutdown and re-taken with a fresh control.**
   `cad-fp32-mobileclip` was still pending when the board was powered off at the operator's
   request. I first recorded this as an unexplained board failure and reasoned about power and
   thermal causes from a secondary filesystem's dirty-mount flag; that was wrong and is retracted
   here. There was no device fault, and nothing in this document should be read as evidence about
   this board's stability under sustained load. Rather than pair the re-run with the morning's
   control across a reboot, both runs were re-taken in one evening session; the morning control is
   kept in the table as the reproducibility check it accidentally became. The operational lesson is
   mine: a measurement chain must not depend on an SSH session staying up, and the time the board
   will be occupied has to be stated before it is occupied, not assumed. The repo already holds the
   right tool for this -- `deployment/scripts/jetson-run-then-shutdown.sh`, which runs steps
   detached, times each one out, and leaves the board up on failure -- and I ran an ad-hoc SSH chain
   instead. Future device chains go through that script.

   One unrelated observation made while diagnosing: `mmcblk0p1` -- a removable card auto-mounted by
   the desktop, **not** the NVMe root that holds the project -- carries an ext4 error and aborts its
   journal on mount, so it needs `e2fsck`. It has no bearing on any measurement; all artifacts live
   on the NVMe and were copied to the host.
