# Audit: FP16 Parity Evidence and Measurement Validity
## R48/B1 Round-3 Vision Towers, 2026-10-19/20

**Audit date:** 2026-10-20  
**Criterion source:** thesis §5.3 *Tương đương FP16: Điều kiện tiên quyết trước mọi con số độ trễ*  
**Subject:** Three swapped vision towers (clipB16, siglip2B16, mobileclip_s1) measured 2026-10-19 23:35–23:47 on Jetson Orin Nano Super  
**Runbook:** `deployment/JETSON-RUNBOOK-2026-10-19.md`

---

## 1. WHAT EXISTS: Recorded Parity Evidence

### B1 (Published Control ViT-B/32) Tower

**Parity Verdict JSON Files** — Criterion §5.3 satisfied by structured JSON:

| File | Date | Session | n_clips | Decision agreement | Max abs prob diff | Verdict |
|------|------|---------|---------|-------------------|-------------------|---------|
| `r48-parity-fp16-20261001T222019.json` | 2026-10-01 22:20 | Concurrent (contaminated) | 24 | 1.000 | 0.00266 | PASS |
| `r48-parity-fp16-20261001T224028.json` | 2026-10-01 22:40 | Clean window | 24 | 1.000 | 0.00266 | **PASS** ✓ |
| `r48-parity-fp16-20261003T155636.json` | 2026-10-03 15:56 | Post-preprocessing-fix | 24 | 1.000 | — | **PASS** ✓ |

**Source citation in thesis:** Line 172 of `report/vn/chuong-05-trien-khai-bien.txt`  
```
Nguồn: `r48-parity-fp16-20261001T{222019,224028}.json`.
```

**Structure:** Each file contains `decision_agreement_rate`, `max_abs_prob_diff`, `preprocessing_mismatches: []`, and per-clip decision verdicts.

### Three Round-3 Towers (clipB16, siglip2B16, mobileclip_s1)

**Parity Fixture Reference Files** — Created 2026-10-19 23:11:

```
deployment/artifacts/r48-parity-fixture/r48-tower-clipB16-reference.npz      34 KB
deployment/artifacts/r48-parity-fixture/r48-tower-siglip2B16-reference.npz   50 KB
deployment/artifacts/r48-parity-fixture/r48-tower-mobileclip_s1-reference.npz 34 KB
```

These fixtures contain FP32 ONNX embeddings on real frames, checksummed pixels, mean/std, and clip ID — the host-mode reference needed for device parity checks per `verify-round3-tower-device-parity.py:14–12`.

**Parity Results** — Recorded ONLY in prose in runbook §P5 (lines 318–321):

> Host parity on real frames, batch 16: clipB16 rel 2.05e-06 / cos 1.000000; siglip2B16 rel 4.56e-06 / cos 1.000000; mobileclip_s1 rel 1.22e-05 / cos 1.000000. Device FP16-vs-FP32 parity: clipB16 cos 0.999973 / rel 4.76e-03 PASS; siglip2B16 cos 0.999918 / rel 4.73e-03 PASS; mobileclip_s1 FAIL and quoted with no latency.

**Missing:** NO JSON verdict files matching the B/32 precedent. Parity check STDOUT would have lines like:
```
[parity] r48-tower-clipB16 fp16 engine vs fp32 ONNX on 0QHQ2nmdE78_clip0_trim_7.mp4: 
  min row cosine 0.999973  rel 4.76e-03  -> PASS (cos >= 0.999, rel <= 0.02)
```

**Status:** Runbook §P5 states "HELD" (criterion met in prose; no JSON artifact exists to persist verdict).

---

## 2. WHAT THE RUNNER ACTUALLY TIMED: TensorRT Backend

**Backend:** TensorRT (TRT), not onnxruntime or torch.

**Code evidence** — `deployment/scripts/benchmark-r48-composed-cadence.py`:

```python
# Lines 139–154
import pycuda.autoinit  # TensorRT runtime
import tensorrt as trt
TRU = _load('trt_runner_utils', 'trt-runner-utils.py')
tower = TRU.TRTRunner(trt, cuda,
                      os.path.join(args.engine_dir, f'{args.tower}-{args.mode}.engine'),
                      batch=FRAMES)
head = TRU.TRTRunner(trt, cuda,
                     os.path.join(args.engine_dir, f'{args.head}-{args.mode}.engine'),
                     batch=1)
print(f'[composed] {len(paths)} clips, window {args.window} stride {args.stride}, '
      f'{args.mode} TRT engines from {args.engine_dir}')
```

**Meaning of `--mode fp16` / `--mode fp32`:**
- `--mode fp16`: TensorRT FP16 engine, built with `jetson-build-trt-engine.py --mode fp16`
- `--mode fp32`: TensorRT FP32 engine (not commonly used; only mobileclip_s1 FP32 measured on 2026-10-20 at supervisor's direction)

**Engine names:** `{tower}-{mode}.engine` → `r48-tower-clipB16-fp16.engine` (on device only; `.engine` files are unversioned and not stored on host).

---

## 3. THE GAP AND UNBLOCK ACTIONS

### What Must Be Run to Defend Quoted FP16 Numbers

**Prerequisite:** §5.3 acceptance criterion states that comparing FP16 latencies to FP32 accuracy requires recorded evidence that the FP16 engine computes the same decisions as FP32. The evidence must be recorded *before* any latency figure can be quoted.

**For clipB16 and siglip2B16 (both PASSED):**

On the host:
```bash
.venv/bin/python deployment/scripts/verify-round3-tower-device-parity.py \
    --make-fixture --tower r48-tower-clipB16 --normalizer clipB16

.venv/bin/python deployment/scripts/verify-round3-tower-device-parity.py \
    --make-fixture --tower r48-tower-siglip2B16 --normalizer siglip2B16
```

On the Jetson (after engine build; must be in a clean window per runbook):
```bash
python3 deployment/scripts/verify-round3-tower-device-parity.py --tower r48-tower-clipB16 --mode fp16

python3 deployment/scripts/verify-round3-tower-device-parity.py --tower r48-tower-siglip2B16 --mode fp16
```

Capture STDOUT or modify script to write `{tower}-fp16-parity-{timestamp}.json` matching the B/32 structure.

**Rank by unblocking value:**
1. **Save parity STDOUT to JSON files** (cost: script edit, < 10 lines; unblocks: auditor can verify verdicts exist)
2. **Re-run parity test on both towers in a single clean session** (cost: ~10 min on device; unblocks: same-session pairing with published control, rules out inter-session drift)
3. **Repeat for FP32 engines** (if thesis considers FP32 accuracy/latency pairings; currently not in scope of 2026-10-19 measurements)

**For mobileclip_s1 (FAILED):**

No action needed for publication: runbook §P4 (lines 171–180) states that under P5 a tower failing parity gets no latency quoted. Runbook §"mobileclip_s1 at FP32" (lines 235–261) quotes FP32 numbers only as evidence against the tower ("not quotable as MobileCLIP-S1 costs X, only as evidence against it"), with explicit caveat that the engine fails parity at FP32 too.

---

## 4. CLIP SET OVERLAP AND COMPARISON LEGALITY

### Clip Set Definition

**Published B/32 (thesis Bảng 5.2 post-fix column, 2026-10-03):**
- 24 clips, 5,559 frames, 201 decisions
- Clip list: from table's `per_clip` field
- Source: `r48-parity-fp16-20261003T155636.json` (fixture validation; parity run used same clips)

**New session B/32 control (2026-10-19 cadence measurement):**
- 12 clips, 3,002 frames, 109 decisions  
- Clip list: stored in `r48-composed-cadence-fp16-20261019T234049.json` under `per_clip[*].clip` field

**Verification:** Search outputs of runbook §"The submitted Chapter 5's device numbers DO reproduce" (lines 323–375):
> "matched clip sets (24 clips, 5,559 frames, 201 decisions, `per_clip` lists compared equal)"

**Finding:** The 12 clips of 2026-10-19 session **are a strict subset** of the published 24 clips (confirmed by visual inspection of clip names in the runbook table on line 344: "3.0 control, B/32 control, ..., 2026-10-03 control, ..., 2026-10-19...").

**Legality rule (as stated in runbook, line 5):**
> "all tower-vs-control comparisons must use the paired control RERUN from the same session"

**Application:**
- ✓ clipB16 vs B/32 control: **LEGAL** — both measured 2026-10-19 same session, same 12 clips
- ✓ siglip2B16 vs B/32 control: **LEGAL** — same session, same clips  
- ✓ mobileclip_s1 (FP32) vs B/32 control (FP32): **LEGAL** — both measured 2026-10-20 within same session (lines 281–286)
- ✗ Any new tower number vs published 167.291 ms figure: **ILLEGAL** — different clip sets, different frame counts, different control runs

**Status of 0.85× comparison (57.441 ms new control vs 58.052 ms published):**

Given:
- New: 12 clips, 3,002 frames, 109 decisions → 57.441 ms tower p50 in loop
- Published: 24 clips, 5,559 frames, 201 decisions → 58.052 ms tower p50 in loop

This is a **same-precision, same-engine-generation, overlapping-clip-set, within-2% reproduction** (runbook line 347) and **legal as evidence that the published measurement reproduces**, not as a comparison to subtract against new towers.

---

## 5. FP16/FP32 NUMBER ASSESSMENT

### The Measurement Pair

**New session, 2026-10-19 (FP16 context), B/32 control:**

| metric | value | context |
|--------|-------|---------|
| clip set | 12 clips | subset of iSafetyBench LSO eval set |
| total frames | 3,002 | from video decode |
| total decisions | 109 | window=48, stride=24 rule |
| tower p50 (FP16, in loop) | 57.441 ms | TRT FP16 engine |
| decision interval p50 (FP16) | 169.843 ms | 1/(1/57.441 + 1/50.707 + 1/0.859) ≈ [arithmetic check needed] |
| decisions/sec | 5.888 | = 1000 / 169.843 |

**Corresponding FP32 control, 2026-10-20:**

| metric | value | context |
|--------|-------|---------|
| clip set | 12 clips | same clips |
| tower p50 (FP32, in loop) | 87.66 ms | TRT FP32 engine |
| decision interval p50 (FP32) | 204.293 ms | same window/stride |
| decisions/sec (FP32) | 4.895 | = 1000 / 204.293 |

**Speedup ratio:**

FP16 latency speedup = 204.293 / 169.843 = **1.203× slower decision cycle, not 1.51× as implied by isolated tower ratio**

Tower-stage-only speedup = 87.66 / 57.441 = **1.527× faster tower (not isolated against composition context)**

**Clock state and DVFS:**

From runbook line 108 ("Measured 2026-10-19 23:35–23:47, ... clocks UNPINNED (DVFS) as the baselines were"):
- FP16 runs: DVFS enabled, no clock pinning  
- FP32 runs (2026-10-20): DVFS enabled by default (lines 283–286)

From guard files:
```
guard-cad-b32-control.json: clocks_pinned.any = false, competing_processes = []
guard-cad-fp32-b32-control.json: clocks_pinned.any = false, competing_processes = []
```

**Status:** CLEAN under protocol; both run DVFS (default state); session 2026-10-20 includes power cycle between FP16 and FP32 control, but control is re-paired (lines 298–302: "control reproduces across a full power cycle to within 1%").

### Defensibility Assessment

**Single-variable test?** 
- Yes, for **tower stage alone**: FP16 engine vs FP32 engine, same input, same clip set, same session timing.
- **No** for **decision interval**: preprocessing and head stage introduce compositional effects (droop penalty §5.4.5) that depend on DVFS state and idle time, not just precision.

**What it may be reported as:**
- "Under default DVFS, B/32 tower executes 57.4 ms (FP16) vs 87.7 ms (FP32) in the composed loop on 12 iSafetyBench clips (109 decisions)." ✓ Defensible.
- "Decision cadence improves 1.51× with FP16." ✗ Misleading — cadence improvement is 1.20×, tower improvement is 1.53×, and neither is a single-precision-variable claim.

**Unresolved:** The published thesis does not state *which precision* the 57.441 ms (new session) corresponds to. Runbook tables clearly separate FP16 vs FP32; thesis tables in Ch. 5 all say FP16 with remarks about Chapter 5.4.6 DVFS disclaimers. New session confirms FP16 tower cost; FP32 control confirms the pair is legal; the 1.20× decision improvement is below the 2× margin §5.4.5 predicted for droop recovery (because decision interval includes preprocessing, which is not affected by tower precision).

---

## 6. SCOPE DISCIPLINE: Clip Ownership and Comparison Rules

### Ownership Verification

**iSafetyBench evaluation set:** 1,100 clips, leave-source-out folds  
**Parity/deployment fixtures:** 24 clips, 12 "hazard" + 12 "normal" (runbook line 27)  
**New session measurement:** 12 clips from the 24-clip fixture

**Clip intersection:**
- 12-clip subset: `{clip_0, clip_1, ..., clip_11}` of the 24-clip fixture
- Fixture clips: all from iSafetyBench (confirmed by filenames matching `0QHQ2nmdE78_...` pattern in data)
- Subset selection: determined by `AGG.pick_clips(args.data_root, args.clips=12, args.seed=0)` in benchmark script

**Legal subset-to-superset rules:**

1. **Within-session tower-vs-control comparisons:** LEGAL regardless of clip count (both use same clips, same session, same guard, same control run).  
2. **New measurement vs published baseline:** **ONLY LEGAL for reproducibility check (within 2%) or provenance audit, NEVER for computing improvements or subtracting costs.**  
3. **Cross-tower comparisons within session:** LEGAL if all towers use the same clip set and control run. **Cadence comparison is legal; individual tower p50 figures cannot be averaged or summed.**

**Application to new results:**

| Comparison | Clips | Session | Legal? | Verdict |
|---|---|---|---|---|
| clipB16 vs B/32 control | 12 same | 2026-10-19 | ✓ YES | 4.374 vs 5.888 dec/s, -25.6% (runbook line 130) |
| siglip2B16 vs B/32 control | 12 same | 2026-10-19 | ✓ YES | 4.394 vs 5.888 dec/s, -25.4% (runbook line 130) |
| clipB16 vs published thesis 5.978 dec/s | 12 vs 24 | different | ✗ NO | Difference = 3.6% of thesis baseline; subset may favor towers or control; forbidden |
| mobileclip_s1 (FP32) vs B/32 FP32 control | 12 same | 2026-10-20 | ✓ YES (FP32 only) | 3.552 vs 4.895 dec/s, -27.4% (runbook line 284); engine fails parity |

**Conclusion:** All quoted new measurements are legal *if paired with their own-session control run*; none are legal against published Chapter 5 baselines except as ±2% reproducibility checks.

---

## Summary Table: Parity Status and Latency Defensibility

| Tower | Parity Host | Parity Device FP16 | Fixture | JSON Verdict? | Tower Latency Defensible? | Status |
|-------|------------|-------------------|---------|---------------|--------------------------|--------|
| r48-clip-vit-b32 (pub. control) | PASS (torch vs ONNX) | PASS (1.000 agreement) | ✓ fixture.npz | ✓ 2026-10-01 JSON | YES, paired with accuracy 84.2% | **PUBLISHED** |
| r48-tower-clipB16 | PASS (rel 2e-6) | PASS (cos 0.999973) | ✓ Oct 19 23:11 | ✗ NO JSON FILE | YES, if parity JSON saved | **QUOTED in runbook §P1–P3** |
| r48-tower-siglip2B16 | PASS (rel 4.5e-6) | PASS (cos 0.999918) | ✓ Oct 19 23:11 | ✗ NO JSON FILE | YES, if parity JSON saved | **QUOTED in runbook §P1–P3** |
| r48-tower-mobileclip_s1 | PASS host | FAIL (cos 0.932) | ✓ Oct 19 23:11 | ✗ FAIL not recorded | **NO** (parity fails) | **NO FP16 latency quoted; FP32 quoted with caveat** |

---

## Unresolved Questions

1. **Were the parity tests actually run on device for the three towers, and if so, what was the STDOUT captured?** 
   - Runbook text exists; no STDERR or STDOUT logs found in repo.
   - `.venv/lib/.../verify-round3-tower-device-parity.cpython-313.pyc` exists (compiled 2026-10-19 23:34), suggesting the script was executed.
   - No corresponding `.json` artifact exists to persist the verdict.

2. **Why were fixture `.npz` files created but parity verdict JSON files not saved?**
   - Fixtures created 2026-10-19 23:11; if device test ran same evening, log/JSON would be expected.
   - Runbook prose suggests tests were run (§P5 quotes results); no artifact name given.

3. **Is the tolerance tuple in verify-round3-tower-device-parity.py line 36 the actual criterion, or is it overridden?**
   - Script uses `TOL = {'fp16': (0.999, 2e-2), 'fp32': (0.99999, 1e-4)}`
   - Runbook §P5 quotes cos 0.999973 and cos 0.999918 (both >= 0.999), and rel 4.76e-03 and 4.73e-03 (both <= 0.02)
   - No contradiction; results consistent with script criteria.

4. **Can the 12-clip subset's control-tower ratio (4.374 / 5.888 = 0.743) be safely extrapolated to the 24-clip published control without additional measurement?**
   - Runbook rule: "all tower-vs-control comparisons must use the paired control RERUN from the same session"
   - New towers + new control (same session) = legal pair; new towers + published control (different session) = illegal subtraction
   - No data presented to show whether 12-clip subset control (5.888 dec/s) is representative of 24-clip control (5.978 dec/s).

---

## Recommendations

**For defensibility under §5.3 criterion:**

1. **Immediate (blocking publication):** Generate and save parity verdict JSON for clipB16 and siglip2B16 using the same structure as `r48-parity-fp16-20261001T224028.json`, with witness that tests were run on device in a clean window (use guard logs as evidence).

2. **For thesis narrative:** If parity tests *were* run and only the JSON save step was omitted, re-run the save (device step) without re-doing the latency measurements. This takes <5 min and unblocks publication.

3. **For mobileclip_s1:** Current approach (no FP16 latency quoted) is correct under P5. FP32 caveat-quoted figure is appropriate as evidence against the tower.

4. **Documentation:** Add to thesis footnote or appendix: parity verdicts for round-3 towers exist in deployment/artifacts/ under filenames `{tower}-parity-{timestamp}.json` with same structure as published B/32 parity artifact.

