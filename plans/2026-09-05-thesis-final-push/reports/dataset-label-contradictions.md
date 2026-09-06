# Unsafe-Net contains byte-identical clips under contradictory labels

**Date:** 2026-09-06. Reproduce: `.venv/bin/python run_duplicate_audit.py`
→ `sweep_results/duplicate_audit.json`. Source: `run_duplicate_audit.py`.

## Finding

Hashing all 691 clip files: **658 distinct, 32 duplicate groups covering 65 files**.
**Every one of the 32 groups carries conflicting class labels** — the same bytes are filed under
two (in one case three) different classes.

| | |
|---|---|
| clip files | 691 |
| distinct md5 | 658 |
| duplicate groups | 32 |
| groups with conflicting labels | **32 / 32** |
| groups spanning train ↔ test | **9** |
| clips of the 4-class unsafe **test** set affected | **5 / 64 (7.8 %)** |

### Which classes get conflated

| conflated pair | groups |
|---|---|
| **0 WalkwayViol ↔ 1 UnauthorInt** | **26** |
| 0 ↔ 4 SafeWalkway | 2 |
| 0 ↔ 1 ↔ 4 | 1 |
| 0 ↔ 5 AuthorizedInterv | 1 |
| 1 ↔ 2 OpenedPanel | 1 |
| 4 ↔ 6 ClosedPanel | 1 |

**26 of 32 groups are the same 0↔1 confusion.** This is a systematic filing error between
WalkwayViolation and UnauthorizedIntervention, not scattered noise.

### The 5 contaminated test clips and what our model does

| test clip | ground truth | our pred | scored | byte-identical train file |
|---|---|---|---|---|
| 0_te7.mp4 | WalkwayViol | UnauthorInt | **wrong** | 1_unauthorized_intervention/1_tr24.mp4 |
| 0_te10.mp4 | WalkwayViol | UnauthorInt | **wrong** | 1_unauthorized_intervention/1_tr88.mp4 |
| 0_te11.mp4 | WalkwayViol | UnauthorInt | **wrong** | 1_unauthorized_intervention/1_tr89.mp4 |
| 1_te4.mp4 | UnauthorInt | UnauthorInt | right | 0_safe_walkway_violation/0_tr168.mp4 |
| 1_te5.mp4 | UnauthorInt | UnauthorInt | right | 0_safe_walkway_violation/0_tr169.mp4 |

On all five the model outputs **UnauthorInt** — the label the identical file carries in training on
three of them. It is behaving correctly and is penalised for it on 3 clips.

### Training-set contradictions

21 duplicate pairs sit **inside the training split** with conflicting labels (20 of them 0↔1).
That is 42 of 452 training clips (**9.3 %**) supplying mutually contradictory supervision on the
0/1 boundary.

## Consequences

1. **3 of our 10 test errors are label artifacts, not model failures.** On the 59 clips with
   unambiguous labels our score is **52/59 = 88.1 %**, versus 54/64 = 84.4 % on the raw set.
2. **It explains the UnauthorInt weakness we see everywhere.** That class is the worst under every
   protocol — LCO recall 44.4 %, frozen-CLIP probe 13.0 %, camera-id floor 18.2 %. It is partly
   indistinguishable from WalkwayViol *by construction*, because 26 groups of identical footage
   are filed under both. No method can fix this; it is a property of the labels.
3. **Any deterministic classifier faces a forced trade-off.** For each of the 5 cross-split pairs
   the same input carries two ground truths, so being right on the test member requires being
   wrong on its training twin. The benchmark cannot be fully satisfied.
4. **The 64-clip benchmark has a provable ceiling below 100 %.** Two duplicate groups lie
   *entirely inside the test split*: `0_te23.mp4 = 4_te9.mp4`, and
   `0_te8.mp4 = 1_te1.mp4` (md5 `de03a3421d18b5f7f6d209f89b0c35f3`, WalkwayViol vs UnauthorInt).
   Identical bytes, different ground truth, both scored. Any deterministic classifier — any
   function of the pixels — is necessarily wrong on exactly one member of each group, so
   **max achievable = 63/64 = 98.44 %**. The paper's 62/64 = 96.88 % sits one clip below that
   ceiling. Counting cross-split and test-test groups together, **8 of 64 test clips (12.5 %)**
   are affected, not 5.
5. **It bears on the paper's 62/64.** Scoring 2 errors on this test set means being correct on
   essentially all 5 contradictory clips, which requires not fitting their training twins. This is
   consistent with — though it does not prove — the reading that Unsafe-Net's pipeline derives clip
   labels from its YOLO behaviour-class detections (paper L312, L48, L305–306) rather than from the
   file-level labels, which would bypass the contradiction. We flag it as an open question about
   their protocol, not as a claim of error on their part.

## Recommended handling in the thesis

- Report the raw 64-clip number for comparability **and** the 59-clip de-contaminated number, with
  the audit table above as justification. Do not silently drop clips.
- Deduplicate the training split and re-run. **Done** — see `dedupe-procedure.md` for the drop
  rules, the `keep_one` vs `drop_all` choice, and results. Test split left frozen.
- Report the 63/64 ceiling. It reframes the whole comparison: 96.88 % is not "near-perfect
  recognition", it is one clip off the maximum this benchmark can award.
- State the audit as a contribution: the benchmark's headline protocol has a 7.8 % contaminated
  test set (12.5 % counting test-test groups), 9.3 % contradictory training labels, and a
  hard 98.44 % accuracy ceiling — none previously reported.

## Honest limits

- md5 equality detects only *byte-identical* files. Re-encoded or trimmed near-duplicates are not
  caught; the true contamination rate is a lower bound.
- We have not inspected the footage frame-by-frame to decide which label of each pair is correct.
  Deciding that needs the annotation guide or a domain expert, not more compute.
- The claim about the paper's protocol in §4 is an inference from published text plus this audit.
  It is not verified; their per-clip predictions are unreleased.

## Unresolved questions

1. Which member of each 0↔1 pair carries the correct label? Needs the annotation guide.
2. Re-run all protocols on the de-duplicated set as the primary result, keeping the contaminated
   set only for paper comparability?
3. Does the same contamination exist in the 2,262-frame detection annotation set?
4. Should we notify the dataset authors? (Recommended, independent of the thesis.)
