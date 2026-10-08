# P3's per-decision latency is a stage sum, not a cadence — the thesis's own headline finding applies to its own P3 number

**Date** 2026-10-03 · **Status** CONFIRMED by code reading, three independent omissions, arithmetic decisive
**Severity** HIGH — withdraws the only PASS verdict in the P3 lane
**Trigger** `deployment/scripts/jetson-p3-multi-stream.py` docstring, L1-30, asserting that
`jetson-measure-latency.py --decision` is "a SINGLE-stream sum of separately-timed stage
percentiles over synthetic detections". Verified. **The docstring is correct, and understates it.**

---

---

> ## ⚠ SELF-CORRECTION — same day, 2026-10-03, later
>
> **§4's "upper bound" argument is WITHDRAWN. The direction of the error is unknown, not upward.**
>
> After writing §1-§9 I read `jetson-measure-latency.py:105-106` and found that the file *defines*
> the deployed stride and then does not use it:
>
> ```python
> DECISION_T_FRAMES, DECISION_IMGSZ, DECISION_CROP = 16, 800, 224
> DECISION_EMIT_EVERY = 8   # cadence contract: one decision per 8 NEW frames (24 fps -> 3.0 dec/s)
> ```
>
> `DECISION_T_FRAMES` is the **window length**; `DECISION_EMIT_EVERY` is the **stride**. The formula
> at L231 multiplies by the window. So the sum charges **16 frames of per-frame work for a decision
> that consumes 8 new frames** — it double-charges one whole stride. Meanwhile `_measure_decode`
> uses the stride correctly ("8 consecutive frames … the per-decision decode increment a live
> pipeline actually pays").
>
> So the stage sum is **too high** by one stride of per-frame work *and* **too low** by the six
> omissions of §3. It is not a bound in either direction. The claim "P3 misses the contract in both
> clock conditions" is **no longer established by arithmetic** — see §10, which replaces §4.
>
> What this does **not** touch: §2 (the construct), §3 (the six omissions), §5's A/B contrast
> (both arms share the same incoherent construct, so the *contrast* is unaffected), §6, §7.

## 1. Verdict

The P3 two-condition headline — pinned **310,00 ms = 3,226 dec/s, "clears the 3,0 dec/s contract by
+7,5 %"** vs DVFS **365,88 ms = 2,733 dec/s, FAIL** — is **not a cadence measurement**. It is
`16·det_p50 + post_p50 + crop_p50 + app_p50 + head_p50`, five stages each timed **in isolation,
back-to-back, warmed, on synthetic fixed tensors**, plus a separately-measured decode term.

This is the exact construction that §5.6.9 / `finding-2026-10-01-stage-sum-is-not-a-cadence.md`
refutes for the B1 lane. The thesis applied the standard to its own headline lane and **not** to P3.
An examiner who reads both sections in order finds the contradiction without looking for it.

**The pinned PASS does not survive**, because a stage sum cannot verdict a cadence contract at
all — not because the corrected number is known to be worse. Verdict: **the +7,5 % PASS is
withdrawn and the contract question is reopened, not answered** (see the self-correction above and
§10). D2 already deploys P1 and already stops forcing P3 to real time, so no decision rests on it.

**§11 now answers it by measurement** (2026-10-03, 17:33-17:47). Composed, nothing omitted, N=1:
**pinned 2,493-3,965 dec/s, median 3,479 — meeting the 3,0 contract on 2 of 3 clips; DVFS
1,141-1,511, median 1,441 — failing on 3 of 3.** The arms are disjoint. So the answer is neither
"P3 passes" nor "P3 misses in both conditions": **P3's contract is clip-dependent and holds only
with the clocks pinned.** The second of those — what §4 tried to prove by arithmetic — would have
been **wrong in the pinned arm**, which is why the measurement was necessary rather than merely
more rigorous. The published stage sum turns out to be 7,98 % too *high* when pinned and 39,3 % too
*low* under DVFS: **its bias changes sign with the clock state**, so no correction factor could have
rescued it.

## 2. The construct, from the code

`deployment/scripts/jetson-measure-latency.py:211`

```python
per_dec = (DECISION_T_FRAMES * t_det['p50'] + t_post['p50'] + t_crop['p50']
           + t_app['p50'] + t_head['p50'])
```

Every term comes from `_timed()` (L120-129): warmup, then `runs` back-to-back calls of **that stage
alone**, returning percentiles. The artifact records the formula verbatim in its own `formula` field
and lists `stages_included` as exactly those five. There is no composed loop anywhere in the file.

The function's own docstring (L134) says "on the real composition, not a sum of one pass per stage".
Read precisely, it is defending against a *different* error: `run_latency()`'s
`end_to_end_naive_sum_ms_p50` counts **one** detector pass and a **batch-1** appearance pass.
`run_decision_latency` fixes the **multiplicities and batch shapes** (16 detector passes, appearance
at `batch=n_crops`). It does not compose. "Real composition" there means *correct multiplicity*, and
the phrasing has been read as *end-to-end loop* ever since — including by me, and including by the
chapter.

## 3. Six omissions. Three are entire stages.

| # | omitted | evidence | B1-lane analogue |
|---|---|---|---|
| 1 | **Detector input preprocessing** — 16 frames → 800×800 resize + float + normalise + HWC→CHW + batch | `grep -n "preprocess\|letterbox\|resize\|blobFromImage" jetson-measure-latency.py` → **zero hits**. Detector is fed `_fixed_inputs()` = `rng.standard_normal(engine_input_shape)`, a tensor already in the engine's layout | **the single largest cost in the whole B1 pipeline**: 160,765 ms pre-fix (55,5 % of the interval), 50,867 ms post-fix (30,4 %) |
| 2 | **Appearance input preprocessing** — 54 uint8 HWC crops → normalised float32 NCHW batch | `extract_crops()` (`benchmark-p3-utils.py:78-87`) returns a Python **list of uint8 HWC arrays**; the appearance engine is then fed `_fixed_inputs(app)`, a synthetic float batch. The conversion between the two is never timed | same class as #1 |
| 3 | **Head input assembly** — the 74-d relation-token construction | head is fed `_fixed_inputs(head)` (L190) | — |
| 4 | **Loop / composition overhead** — ring buffer, `cap.read()` outside the timed span | no loop exists to carry it | **7,4 %** of the B1 interval, unattributed |
| 5 | **Contention** — every stage timed with the others idle | `_timed()` runs one stage at a time | the shared memory bus on an integrated SoC; the mechanism 3b-note 9 names |
| 6 | **Composed percentiles** — `sum(p50) ≠ p50(sum)`; no composed p95/p99 exists at all | the formula sums p50s | B1 reports composed p95 198,224 ms directly |

Omissions 1, 4 and 5 are precisely the three cost classes the 10-01 finding names as *structurally
invisible to a sum of isolated benchmarks*: **preprocessing (absent, not underestimated), state
(back-to-back vs duty-cycled), contention**.

Inputs are also **synthetic**: `synthetic_raw_frames()` makes 300 random boxes/frame with
`conf ~ U(0,1)` for the postprocess timing, and crop extraction runs on a uniform-noise frame. The
crop *counts* (p50=54, p95=106, p99=119, cap=128) are corpus-derived and remain sound; the crop
*content* is not real. Only #1-#6 carry the verdict; synthetic content is a secondary limit.

## 4. ~~The arithmetic is decisive, and needs no new measurement~~ — SUPERSEDED BY §10

> **This section is kept as written and is WRONG in its conclusion.** Its premise — "every omission
> is signed: each can only add latency" — is false: the double-charged stride (see the
> self-correction above) is an error in the opposite direction, and larger than the omission this
> section prices. Read §10 instead. Kept, not deleted, because the error is instructive: it is the
> same class of mistake as the one the finding is about — reasoning about a composed cost from parts
> without checking that the parts compose.

Contract 3,0 dec/s ⇒ budget **333,333 ms**. Pinned stage sum **310,00 ms** ⇒ headroom **23,33 ms**
(the +7,5 % the abstract and Bảng 5.5J quote).

The nearest **measured** value of omission #1 is the B1 lane's **post-fix** preprocessing stage,
**50,867 ms** for 16 frames at 224² destinations — already **2,18× the entire P3 headroom**, and a
**floor** for P3, which resizes to 800² (12,76× the destination pixels) and additionally needs
omissions #2 and #3.

| | stage sum | + omission #1 at the B1 post-fix floor | verdict |
|---|---|---|---|
| pinned | 310,00 ms = 3,226 dec/s (+7,5 %) | ≥ 360,87 ms = ≤ 2,771 dec/s (**−7,6 %**) | **FAIL** |
| default DVFS | 365,88 ms = 2,733 dec/s (−8,9 %) | ≥ 416,75 ms = ≤ 2,400 dec/s (−20,0 %) | FAIL (already) |

Adding B1's unattributed loop overhead (#4, 7,4 % of interval) puts the pinned arm near
**2,58 dec/s**. Every omission is signed: each can only **add** latency. So **3,226 dec/s is an
upper bound on the pinned cadence**, and the bound is exceeded by the smallest credible value of a
single omitted stage. No new measurement is required to withdraw the PASS.

**Calibration from the only lane where both constructs were measured on this device:** the honest B1
stage sum was 72,948 ms against a composed 289,672 ms = **3,97×**; against the post-fix composed
167,291 ms = **2,29×**. P3 at 2,29× would be ~710 ms = 1,41 dec/s. That range is **not** quoted as a
P3 prediction — different graph, different stage mix — but it bounds the plausible scale of the gap,
and a 2,18×-the-headroom floor sits at the very bottom of it.

## 5. What survives, and what does not

**Survives — the A/B contrast.** Both arms omit the *same* stages under the *same* protocol, so
pinned-vs-DVFS is a valid protocol-identical comparison: **310,00 vs 365,88 ms, +18,0 % dec/s**, and
the detector stage's **−17,7 %** (14,246 → 11,724 ms/frame) with its p95/p50 tail narrowing
**1,51 → 1,03**. The clock-state result — *P3's cadence is decided by the device's clock state* — is
unaffected, and is the defensible P3 claim.

**Survives — the per-stage shares**, as *stage* figures: detector 11,724 ms/frame ×16 = 187,58 ms,
postprocess 47,352 ms and its clock-insensitivity (−0,7 % under a change that moves the detector
−17,3 %). These are isolated-stage measurements and were always reported as such.

**Does not survive — any absolute contract verdict**: "clears the 3,0 dec/s contract by 7,5 %",
`meets_3_per_sec_cadence: true`, and the framing of 310,00 ms as a per-decision *interval*.

This inverts the P-PRE4 discipline. There, only the **absolute** figure was quotable. Here, only the
**relative** figure is.

**Does not survive — number-sheet row 236.** It calls the P3 decision "composed", and infers the
contention mechanism from the whole-loop gain (+18,0 %) exceeding the detector-stage gain (+9,5 %).
A stage sum has **no concurrency**, so it cannot evidence contention — the inference is not merely
weak (the row already concedes "two runs under different protocols"), it is unavailable from this
construct. The +18,0 % is better explained by composition: the sum contains CPU-bound stages
(postprocess 47,35 ms, crop extract) that `scaling_min_freq` accelerates, while the soak is
GPU-bound and thermally saturated. Row withdrawn as corroboration; kept as a composition note.

## 6. Why this strengthens the thesis

The thesis's headline methodological contribution **is** "a stage sum is not a cadence". Until now
it demonstrated that claim on one lane (B1, 2,0-4,0× understatement) while quietly violating it on
another (P3). Correcting P3 turns the contradiction into a **second, independent instance of the
same finding, on a different graph** — and one where the understatement flips a PASS to a FAIL,
which is a stronger demonstration than B1's, where the verdict survived.

It also removes the lane's only tension with **D2**. D2 deploys P1 and stops forcing P3 to real
time. A P3 that "clears the contract when pinned" sat awkwardly against that; a P3 that misses it in
both conditions is the evidence D2 was already decided on.

## 7. Claims to patch

| site | current | action |
|---|---|---|
| `report/abstract.md:60-64` | "with the clocks pinned it reaches 3,23 dec/s at median crop load, **clearing the 3,0 dec/s contract by 7,5 %**" | rescope to stage sum + upper bound; state that P3 misses in both conditions |
| `report/chapter-05-edge-deployment.md` Bảng 5.5J + surrounds | two-condition table, cadence framing | label construct, add the omission list, withdraw PASS |
| number-sheet rows 216-220 | "HEADLINE", "FAIL, −8,9 %", `meets_3_per_sec_cadence` | relabel as stage sum, upper bound, A/B-only |
| number-sheet row 236 | "composed P3 decision", contention corroboration | withdraw inference (§5) |
| `deployment/scripts/jetson-measure-latency.py:134` | docstring "on the real composition" | reword — it is the proximate cause of the misreading |

## 8. Impact on the running re-measurement

The 6-run re-measurement launched today (3 × DVFS + 3 × pinned, to close the clock-state provenance
gap on 5 of 6 artifacts) measures **the same stage sum**. It still closes the provenance gap and
still sharpens the A/B — the only P3 claim that survives — so it is worth collecting. It cannot fix
the construct, and its artifacts must not be cited for a contract verdict.

---

## 10. What the arithmetic actually says: three multiplicities, one number

### 10.1 The formula mixes window-, stride- and decision-multiplicity

| term | code | charged for | correct multiplicity (stride 8) | factor |
|---|---|---|---|---|
| detector | `DECISION_T_FRAMES * t_det` (L231) | 16 frames | 8 new frames | **2× too high** |
| postprocess | `[postprocess_frame(r,…) for r in synthetic_raw_frames(16)]` (L207-209) | 16 frames | 8 new frames | **2× too high** |
| crop extract | the clip's whole window crop set (54 crops at p50) | 16 frames | the stride's ~27 | **≈2× too high** |
| appearance | one batched call over the same 54 crops | 16 frames | the stride's ~27 | **≈2× too high** (sub-linear in batch, so ≤2×) |
| head | one pass | 1 decision | 1 decision | correct |
| decode | `stream_frames_per_decision` = "8 consecutive frames … the increment a live pipeline actually pays" | 8 frames | 8 frames | **correct** |

Every **per-frame** term is charged twice; the head and the decode term are right. That is checkable
in one line against the published artifact: `(272,11 − 0,287)/2 + 0,287 = 136,20`.

Evidence that the deployed stride is 8, not 16 — five independent places, all agreeing:

1. `jetson-measure-latency.py:106` — `DECISION_EMIT_EVERY = 8   # cadence contract: one decision per 8 NEW frames (24 fps -> 3.0 dec/s)`
2. `jetson-p3-multi-stream.py:48` — `CADENCE = 3.0   # decisions/s contract: 24 fps / EMIT_EVERY=8`
3. `stream-pipeline-utils.py:27` — `EMIT_EVERY = 8`
4. the composed loop's emit test — `if len(s.buf) == cfg['t_window'] and s.processed % SPU.EMIT_EVERY == 0`
5. the decode term itself, which measures 8 frames

**The 3,0 dec/s contract is *arithmetically derived* from stride 8** (24 fps ÷ 8). A sum that charges
16 frames per decision is therefore not even measuring the pipeline the contract describes. And each
frame's detector/postprocess/crop/appearance work runs exactly **once** in the composed harness —
`s.buf.append((rows, mask))` caches it — and is reused by the next decision.

### 10.2 Coherent stride-8 recomposition of the gated pinned arm

From `decision-latency-fp16-20261003T170736.json`, p50 crop scenario:

```
per-frame = det 11,723 + post 46,925/16 + crop 18,022/16 + app 19,304/16 = 16,989 ms
8 frames  = 135,910   + head 0,287 + decode 38,256 (8 frames)  =  174,45 ms  =  5,73 dec/s
published                                                          310,37 ms  =  3,22 dec/s
```

The published figure is **1,78× higher** than a stride-coherent recomposition of its own inputs. The
double-charge is **135,91 ms** — **5,8× the 23,33 ms headroom** that the withdrawn +7,5 % PASS rested
on, and in the opposite direction from §3's omissions.

### 10.3 Therefore the number is unsigned

| pressure | direction | magnitude (pinned) |
|---|---|---|
| double-charged stride (§10.1) | number too **high** | −135,91 ms |
| omission #1, detector input preprocessing | too **low** | ≥ +25,4 ms (B1 post-fix floor, halved to 8 frames) — and P3 resizes to 800², 12,76× the pixels, so the real value is some unmeasured multiple |
| omissions #2, #3 (appearance preprocess, 74-d assembly) | too **low** | unmeasured |
| omissions #4, #5 (loop overhead, contention) | too **low** | B1 analogue: 7,4 % of interval |
| omission #6, `sum(p50) ≠ p50(sum)` | too **low** at p95/p99 | unmeasured |

Whether the net is above or below 333,333 ms depends on the 800² preprocessing cost, which **has
never been measured on this device**. At the B1 floor the omissions do not cover the double-charge
and P3 would *pass*; at the B1 pixel-ratio scaling they swamp it and P3 fails badly. **Both
conclusions are reachable from the same evidence, which means the evidence decides nothing.**

So §4's "no new measurement is required" was wrong twice: the arithmetic is not decisive, and a
measurement *is* required. It is also *available* — which §4 did not know.

### 10.4 The composed harness already exists

`deployment/scripts/jetson-p3-multi-stream.py` (196 lines, written 2026-10-01, **never run**) is a
genuinely composed P3 harness, and it closes all six omissions at `--streams 1`:

| omission | closed by |
|---|---|
| #1 detector input preprocessing | `pre = functools.partial(P3.preprocess, size=cfg['imgsz'])`, called in `video_reader` **inside** the wall clock |
| #2 appearance input preprocessing | real `extract_crops` → engine path in `P3.frame_tokens` |
| #3 74-d token assembly | `P3.frame_tokens` builds the real `(N_MAX, FEAT+APP_DIM)` rows |
| #4 loop overhead | `decisions_per_sec = len(s.predictions) / elapsed`, wall clock from before the readers start |
| #5 contention | `--streams 2 3 …`; and at `--streams 1` the reader and consumer already overlap |
| #6 `sum(p50) ≠ p50(sum)` | percentiles taken over observed per-frame and per-head latencies, not summed |
| *and* the stride bug | emits on `s.processed % EMIT_EVERY == 0` over a cached sliding window |

It also refuses to report accuracy (the 74-d assembly is unverified on-device), so it answers the
cadence question and only that — which is exactly the question.

The blocker recorded in the audit list ("multi-stream P3 blocked by a hard-coded P1 stage list") is
**stale**: `--detector-stage` and `--head-stage` are parameterised and `P3.build_runners` asserts
shapes rather than assuming them. The blocker belongs to the P1 harness.

Two runs settle it, per clock condition:

* `--fps 1000` — reader effectively unthrottled (`period = 1/fps` sleep), so `decisions_per_sec` is
  the **free-running composed cadence**, directly comparable to the stage sum's 310,37 ms.
* `--fps 24` — the contract condition. `meets_cadence_all_streams = worst ≥ 3,0 and drop_rate == 0`
  is a **direct composed verdict on the contract**, because emission keys off frames *processed*, so
  an overloaded pipeline cannot report a healthy cadence from a staler window.

Timing-only use of `data/test` does not consume a pre-registered test touch, and the harness reports
decision **counts**, never labels — so this costs no touch.

---

## 11. The measurement. P3's composed cadence, both clock conditions

§10.4 said two runs per clock condition would settle it. They ran on 2026-10-03, 17:33-17:47, on
`deployment/scripts/jetson-p3-multi-stream.py` — the harness that already existed, had 21 passing
unit tests, and had never been run only because it had never been rsynced to the device. Eight
artifacts, `p3-composed-cadence-fp16-*.json`, all staged.

The loop is composed: file decode → 800×800 letterbox preprocess → detector → select + zone
geometry → 224² context crops → appearance encoder → 74-d relation-token assembly → binary E3
head, all inside one wall clock. All six omissions of §3 are closed. Both arms were thermally
gated (entry tj ≤ 56 °C, load ≤ 0,60, then 180 s settle) and N=1, 480 frames, three test clips.
Timing only — no labels read, so no pre-registered test touch is consumed.

### 11.1 Free-running: the composed ceiling

| arm | 1_te10 (view 3) | 3_te1 (view 7) | 4_te10 (view 1) | median | vs 3,0 dec/s |
|---|---|---|---|---|---|
| **pinned** | **3,479** dec/s | **3,965** dec/s | 2,493 dec/s | **3,479** | **MEETS 2 of 3** |
| **DVFS** | 1,441 dec/s | 1,511 dec/s | 1,141 dec/s | **1,441** | **FAILS 3 of 3** |
| pinned composed per-frame | 35,32 ms | 31,01 ms | 49,31 ms | 35,32 | contract needs ≤ 41,67 |
| DVFS composed per-frame | 85,32 ms | 81,37 ms | 107,76 ms | 85,32 | — |

The two arms are **disjoint**: max DVFS 1,511 < min pinned 2,493. Pinning is worth **+141,4 %
(2,41×)** on composed cadence, against only +26,4 % (1,26×) on the stage sum.

**The verdict is clip-dependent, and that is the result.** A pinned P3 meets its contract on two of
three clips and misses on the third — `4_te10`, view 1, the slowest clip in both arms — by 1,20×.
Neither "P3 passes" nor "P3 misses in both clock conditions" is true. The second of those is the
claim §4 of this finding tried to establish by arithmetic, and §1 withdrew; it would have been
**wrong in the pinned arm**. The measurement was necessary, not merely more rigorous.

### 11.2 Why pinning is worth 2,41× composed but 1,26× in the stage sum

Machine-recorded in every DVFS artifact, before and after every run: **`policy4` — CPU cores 4-5 —
sat at 115 200 kHz, the absolute floor**, while `policy0` wandered 192 000-1 728 000. In the pinned
arm both clusters held `cur = min = 1 728 000`.

The composed loop is CPU-bound on preprocessing: an 800×800 letterbox, normalise and HWC→CHW per
frame, plus 54 crops at 224². That is precisely the work the stage sum omits. So the omission does
not merely remove cost — **it removes the component most sensitive to the clock condition**, which
is why the stage sum understates the clock effect by nearly 2× as well as misreading the cadence.

### 11.3 The stage sum's error is not consistently signed

| arm | published window sum | stride-coherent recomposition | composed measured | published vs composed |
|---|---|---|---|---|
| pinned | 310,37 ms = 3,222 dec/s | 174,24 ms = 5,739 | **287,44 ms = 3,479** | **7,98 % too HIGH** |
| DVFS | 421,52 ms = 2,372 dec/s | 230,82 ms = 4,332 | **694,00 ms = 1,441** | **39,3 % too LOW** |

In the pinned arm the double-charged stride (+136,13 ms) very nearly cancels the omissions
(−113,20 ms), leaving the published figure accidentally within 8 % of the truth. In the DVFS arm the
omissions dominate and the published figure is optimistic by 272,48 ms. Once the double-charge is
removed, the recomposition is too optimistic in **both** arms — 1,65× pinned, 3,01× DVFS — exactly
as §3's omission argument predicted. Omitted work: **18,89 ms/frame pinned (2,11×), 62,91 ms/frame
DVFS (3,64×)**.

**So the stage sum's bias changes sign with the device's clock state.** No correction factor could
have rescued it. That is the sharpest available form of this finding's own thesis.

### 11.4 Three reading rules, found before they reached a report

Auditing the new harness against its own outputs turned up three ways to misread it. One is a real
defect; two are citation hazards in correct code.

1. **`--fps F` is an additive pacer, not a deadline** — a defect, in the `--fps`-throttled condition
   only. `stream-pipeline-utils.py:84` sleeps `period` *after* all per-frame work, so the inter-frame
   interval is `work + 1/F` and the achievable rate is `1/(1/F + work) < F`. At `--fps 24` the
   ceiling is 3,000 dec/s only if `work = 0`; at the pinned pipeline's 20,27 ms/frame it is ≈ 2,0.
   The data proves it internally: pinned aggregate throughput is **30,17 fps at N=2 but 17,89 fps at
   N=1** — the N=1 figure is pacing-limited, not pipeline-limited. **The `--fps 24` runs are
   therefore not a contract verdict** and §11.1's free-running throughput against 24 fps processed is
   the valid comparison.
2. **`drop_rate = 0` is not evidence of real-time operation at N ≤ 2.** The reader is a coroutine on
   the consumer's event loop and `preprocess_fn` runs inline on it, so the reader is back-pressured
   into lockstep rather than dropping. The signature is exact in all five N ≤ 2 configurations:
   **`read_fps == aggregate_fps` to the cent**, including at `--fps 1000`, where the reader asked for
   1000 fps and delivered 11,72. Only N=3 decouples (18,93 read vs 13,39 processed, 29,24 % dropped).
   Cite `read_fps` against `target_fps`; never `drop_rate`.
3. **`clock_state_stable` means the pin *mode* and the contention did not change — not that the
   frequencies held.** It compares `clocks_pinned` and `quiet` before and after
   (`stream-pipeline-utils.py:140`), so it reads `true` for every DVFS run while `policy0 cur_khz`
   moved 1 574 400 → 192 000 inside a single run. The code does what its docstring says; the field
   *name* is the hazard. Cite observed `cur_khz`, as the runbook already requires.

### 11.5 Multi-stream

Aggregate composed throughput is roughly **flat in N** — ≈ 28-30 fps pinned, ≈ 13-15 fps DVFS — so
concurrency redistributes throughput instead of adding it. Worst-stream cadence falls accordingly
(pinned 2,199 / 1,854 / 1,140 dec/s at N = 1 / 2 / 3), and no N ≥ 2 configuration reaches 3,0 dec/s
per stream. Read these as the device's ceiling, not as contract verdicts, per rule 1 above.

### 11.6 What this does to D2

D2 deploys P1 and stops forcing P3 to real time. The composed measurement **supports** D2 without
needing P3 to fail outright: P3's cadence depends on the clip and on the device's clock state, so a
deployed P3 would carry a contract that holds on some views and not others, and only while the
clocks are pinned. That is a weaker guarantee than a deployed system should rest on, and it is an
honest reason to deploy P1 — stronger than the flat failure §4 wrongly claimed, because it is
measured.


## 9. Unresolved questions

1. **Should P3 get a composed measurement before the 10-05 freeze?** `jetson-p3-multi-stream.py`
   exists and claims to run "the real composition, concurrently, on real video", but is **blocked by
   a hard-coded P1 stage list**. Cost: new measurement code 2 days before the number freeze, against
   a lane that D2 does not deploy. Recommendation: **no** — publish the bound, not a rushed number.
2. **Is `deployment/artifacts/multi-stream-results.json` ("hand-patched") instrument-derived at
   all?** Pre-existing open audit item, now higher priority: it is the one artifact that would carry
   a composed P3 figure.
3. **Does the 74-d token assembly (#3) match the training-time pipeline numerically on device?**
   `jetson-p3-multi-stream.py`'s docstring says this is unverified, which is why it reports decision
   *counts* and not predicted labels. Unchanged by this finding, but it is the same stage.
4. ~~Does B1's harness have the same class of omission?~~ **CHECKED AND CLEARED, same pass.**
   `benchmark-r48-composed-cadence.py:50-81` runs a real loop over real video — sequential
   `cap.read()`, rolling `deque`, stride-based emission — and
   `benchmark-r48-cadence-utils.py:44-62` forms the headline from **`gaps`**: the wall-clock deltas
   between consecutive `emit_sec` timestamps within a clip. So 167,291 ms is a p50 of **real
   inter-decision intervals** (decode, preprocess, tower, head, loop overhead, one clock), and its
   p95 198,224 ms is a genuine p95 of intervals. None of #1-#6 applies.
   **The harness goes further**: it computes `stage_sum_ms_per_decision = decode_p50·stride +
   compute_p50` explicitly, *as the object of refutation*, and reports
   `steady_state_over_stage_sum` beside it. The contrast with P3 is total — **B1 measures the
   cadence and computes the sum in order to refute it; P3 computes only the sum and names it a
   cadence.** The headline lane is sound and no claim on it moves.
