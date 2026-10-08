# Binary Safe/Unsafe duplicate-clip audit + dedupe policy (691 clips)

Date 2026-09-16. Recomputes `plans/2026-09-05-thesis-final-push/reports/dedupe-procedure.md` /
`dataset-label-contradictions.md` for the BINARY task (`unsafenet/config.py`: `UNSAFE=[0,1,2,3]`
"unsafe", classes 4-7 "safe"). The old machinery classified all 32 byte-duplicate groups as
"conflicting" because it compared 8-class folder labels — wrong question for binary, where 0↔1
(26/32 groups) share the same binary value.

Tools: `run_duplicate_audit_binary.py` (new, reuses `run_duplicate_audit.scan()` so group
membership is guaranteed identical to the original audit) → `sweep_results/duplicate_audit_binary.json`
(new file, `duplicate_audit.json` untouched). `unsafenet/dedupe.py` extended with `task='binary'`.
Tests: `tests/test_dedupe_binary.py`.

## 1. Group taxonomy (all 32 groups, all 691 clips)

| kind | count | definition | action |
|---|---:|---|---|
| **benign** | 22 | same split, same binary label | none — harmless duplicate copy |
| **leakage** | 6 | crosses train/test, same binary label | flagged, not dropped by default (see §6) |
| **opposite** | 4 | members disagree on binary label | drives the ceiling + dedupe policy |

22+6+4 = 32, matches the pre-supplied verified facts exactly; independently re-derived by
re-hashing all 691 files in `run_duplicate_audit_binary.py` (not just re-reading the old JSON).

### The 4 opposite (label-contradictory) groups

| md5 group | members | binary labels | split shape | note |
|---|---|---|---|---|
| both-test | `0_te23.mp4`, `4_te9.mp4` | unsafe, safe | test / test | **hard ceiling driver** — no train twin exists, irreducible |
| 3-member | `0_tr123.mp4`, `1_tr55.mp4`, `4_te4.mp4` | unsafe, unsafe, safe | train, train / test | 2 train members to drop |
| 2-member | `0_tr136.mp4`, `5_te13.mp4` | unsafe, safe | train / test | |
| 2-member | `0_tr138.mp4`, `4_te7.mp4` | unsafe, safe | train / test | |

Opposite-group **train** clips (4, all happen to be unsafe-class): `0_tr123, 0_tr136, 0_tr138, 1_tr55`.
Opposite-group **test** clips (5): `0_te23, 4_te4, 4_te7, 4_te9, 5_te13`.

### The 6 leakage groups (same binary label, cross-split — no contradiction, but byte overlap)

train `0_tr168/0_tr169/1_tr24/1_tr88/1_tr89/4_tr10` ↔ test `1_te4/1_te5/0_te7/0_te10/0_te11/6_te11`
(same pairing as before; label agrees under binary, so not a supervision conflict — but the
model literally saw these test clips' exact bytes, correctly labelled, during training. See §6.)

## 2. The favourable nuance, quantified precisely

Old 8-class audit: **32/32 groups (100 %)** label-contradictory, 26/32 (81 %) being the 0↔1
(WalkwayViol↔UnauthorInt) pair — and **both 0 and 1 are UNSAFE**.

New binary audit: **4/32 groups (12.5 %)** remain label-contradictory. All 26 of the 0↔1 groups
resolve to non-contradictory (20 become `benign`, 5 become `leakage`, 1 — the both-test
`0_te8/1_te1` pair — also becomes `benign`, dissolving what used to be the *second* "both-test"
hard-ceiling contributor in the old audit). The single 1↔2 group and the single 4↔6 group also
resolve (both-unsafe, both-safe respectively). Only the groups that conflate an unsafe class (0)
with a safe class (4 or 5) remain contradictory — exactly 4 of the 32.

Stated precisely for the thesis: **the binary reframe removes 28/32 (87.5 %) of the previously
reported label contradictions**, specifically all 26 instances of the 0↔1 filing confusion,
because that confusion never crossed the safe/unsafe boundary. Do not overstate this as
"resolves all contamination" — 4 groups (12.5 %), including the one entirely inside test, remain
genuine binary-label contradictions.

## 3. Recommended policy + reasoning

**Recommend `--dedupe drop_conflict` on train, `keep` (default) on test**, i.e. drop the 4 opposite
group's train members and leave all 125 test clips untouched. Report the stricter
train+test policy (`drop_conflict` on both) as a secondary sensitivity number only.

Reasoning, addressing the tension explicitly:

- The test split is the paper's fixed evaluation set (125 clips: 64 unsafe/61 safe). Removing
  test clips breaks every historical and future comparison against it, and the instructions'
  standing principle is never to silently shrink the scored set.
- But **3 of the 5 opposite-group test clips (`4_te4`, `5_te13`, `4_te7`) have their contradiction
  confined to TRAIN**: their train twin is what carries the wrong binary label, not the test
  clip itself. Dropping those train twins removes the actual poison (a model can no longer
  memorize the opposite label for that exact byte content) while leaving the test clip itself, and
  the 125-clip scoring set, fully intact and comparable. This is a strict improvement with zero
  comparability cost.
- The remaining 2 test clips in opposite groups (`0_te23`, `4_te9`) have **no train twin at
  all** — they conflict only with each other, both inside test. No train-only policy can ever fix
  this; it is the irreducible hard ceiling (§4). Dropping them (the stricter sensitivity policy)
  is the only way to reach a 100 %-attainable eval set, but it costs 2 test clips (1 unsafe + the
  rest of the stricter policy's test drops add 3 more safe clips) and breaks the frozen-125
  comparison — hence sensitivity-only, never headline, mirroring the existing `unsafe4`
  `drop_contradictory` precedent.
- The 6 leakage groups are deliberately **not** dropped by this policy (see §6): they carry no
  label conflict, so removing them would cost train clips for zero benefit under the label-noise
  argument. They are flagged separately as a residual memorization/leakage caveat.
- Default policy is `none` for both splits (project principle) — no historical command's number
  changes unless `--dedupe drop_conflict` is passed explicitly.

## 4. Accuracy ceilings (binary task, ready for results tables)

| policy | scored test clips | hard ceiling (any deterministic classifier) | practical ceiling (model that fits its training data) |
|---|---:|---:|---:|
| `none` (raw, contaminated — current default) | 125 | **124/125 = 99.2 %** | **121/125 = 96.8 %** |
| `drop_conflict` train only (**recommended**) | 125 | **124/125 = 99.2 %** | **124/125 = 99.2 %** |
| `drop_conflict` train+test (sensitivity) | 120 | **120/120 = 100 %** | 120/120 = 100 % |

Derivation:
- **Hard ceiling** comes solely from the one group entirely inside test (`0_te23` vs `4_te9`,
  opposite binary labels, no train involvement) — no policy that leaves test untouched can ever
  move it. 124/125 = 99.2 %.
- **Practical ceiling under `none`**: a model that perfectly fits contradictory training
  supervision will additionally reproduce the wrong (opposite) label on the 3 test clips whose
  train twin carries it (`4_te4`, `5_te13`, `4_te7`) — 3 more forced errors on top of the 1
  irreducible hard-ceiling error → 125 − 1 − 3 = 121/125 = 96.8 %.
- **Practical ceiling under recommended `drop_conflict` (train)**: the 4 poisoning train twins are
  gone, so that memorization pathway is closed; practical ceiling collapses back to the hard
  ceiling, 124/125 = 99.2 %. (This is an upper bound from removing the *memorization* mechanism,
  not a claim that a real model attains it — actual accuracy still depends on generalization.)
- **Under the stricter train+test policy**: all 5 opposite-group test clips are gone, so no
  contradiction remains in the 120-clip eval set at all → ceiling 100 %.

## 5. Per-policy clip counts and class balance

Baseline (raw, `none`): train 566 = **452 unsafe / 114 safe** (ratio 3.97:1); test 125 =
**64 unsafe / 61 safe** (ratio 1.05:1, near-balanced).

| policy | train unsafe/safe (total) | train ratio | test unsafe/safe (total) | test ratio |
|---|---|---:|---|---:|
| `none` | 452 / 114 (566) | 3.97 | 64 / 61 (125) | 1.05 |
| `drop_conflict` train (**recommended**) | **448 / 114 (562)** | **3.93** | 64 / 61 (125) unchanged | 1.05 |
| `drop_conflict` train+test (sensitivity) | 448 / 114 (562) | 3.93 | **63 / 57 (120)** | 1.11 |

Imbalance check (explicitly requested):
- **Train**: all 4 dropped clips are unsafe-class members (`0_tr123, 0_tr136, 0_tr138, 1_tr55` —
  class 0/1 are both unsafe). Dropping them removes 4/452 unsafe clips and 0/114 safe clips, so
  the recommended policy **slightly reduces** train imbalance (3.97 → 3.93), it does not worsen
  it.
- **Test, stricter policy only**: of the 5 dropped test clips, 1 is unsafe (`0_te23`) and 4 are
  safe (`4_te4`, `4_te7`, `4_te9`, `5_te13`). Since safe is already the smaller test class (61 vs
  64), this policy **does worsen** test balance (1.05 → 1.11) and shrinks the minority class by
  4/61 = 6.6 %. This is a second, independent reason (beyond breaking comparability) to keep the
  stricter policy as a sensitivity number only, never the headline.

## 6. Residual leakage groups — not dropped, flagged

The 6 `leakage` groups are same-binary-label, so they inject no label noise — but their test
member's exact bytes also sit in train, correctly labelled. A model can trivially "solve" those 6
test clips by memorizing training data rather than generalizing, inflating apparent accuracy on
exactly those clips regardless of dedupe policy above. Recommend reporting this as a documented
limitation (train/test byte overlap, 6/125 = 4.8 % of test, label-consistent) rather than acting
on it — removing them would cost 6 train clips (all label-consistent, so no noise benefit) purely
to close a leakage channel that a `--dedupe drop_conflict` policy is not designed to address. If
reviewers push back, the fix would be a new policy variant dropping the leakage-group train
members too (not implemented — YAGNI until asked for, straightforward to add analogous to
`opposite`-group handling in `unsafenet/dedupe.py` if needed).

## 7. Files changed

- **New** `run_duplicate_audit_binary.py` — regenerates the binary audit; reuses
  `run_duplicate_audit.scan()` so hashing/grouping is identical to the original, only the
  label-conflict classification differs.
- **New** `sweep_results/duplicate_audit_binary.json` — full 32-group audit with md5, member
  paths, 8-class labels, binary labels, splits, and `kind` (benign/leakage/opposite) per group.
  `sweep_results/duplicate_audit.json` untouched.
- **Modified** `unsafenet/dedupe.py` — added `task='binary'` branch to `drop_names`, `keep_mask`,
  `summary` (new default parameter, appended after existing positional args, so every existing
  call site — `run_relation_transformer.py`, `run_kfold_relation.py`, `run_lco_eval.py`,
  `run_representation_ladder.py`, `unsafenet/scene_confound.py` — is unaffected; none of those
  files were touched). `task='unsafe4'` behaviour verified byte-for-byte unchanged (52/22/0/0/4
  drop counts, same as `dedupe-procedure.md`).
- **New** `tests/test_dedupe_binary.py` — 10 tests.

No training run, no test-split evaluation was performed, per instructions.

## Test output (actual, `.venv/bin/python tests/test_dedupe_binary.py`)

```
  PASS  test_audit_binary_group_taxonomy_matches_documented_counts
  PASS  test_binary_both_in_test_group_untouched_by_train_policy
  PASS  test_binary_default_policy_is_none_and_drops_nothing
  PASS  test_binary_invalid_policy_raises
  PASS  test_binary_keep_mask_aligns_with_clip_id_order_and_drops_only_conflicts
  PASS  test_binary_leakage_and_benign_groups_never_dropped_by_any_policy
  PASS  test_binary_test_drop_conflict_is_exactly_five_clips
  PASS  test_binary_three_member_group_drops_both_train_members_not_the_test_one
  PASS  test_binary_train_drop_conflict_is_exactly_the_four_opposite_train_clips
  PASS  test_unsafe4_task_unchanged_backward_compat
ALL PASS
```

Regression check, existing suite unaffected:
```
.venv/bin/python tests/test_paired_significance.py -> ALL PASS (unchanged)
```

`unsafenet/dedupe.py` CLI self-test (`.venv/bin/python -m unsafenet.dedupe`):
```
-- task=unsafe4 (original 8-class-label framing) --
train/drop_all               drops  52  [...]
train/keep_one               drops  22  [...]
train/none                   drops   0  []
test/keep                    drops   0  []
test/drop_contradictory      drops   4  ['0_te23.mp4', '0_te8.mp4', '1_te1.mp4', '4_te9.mp4']
-- task=binary (Safe/Unsafe framing) --
train/none                   drops   0  []
train/drop_conflict          drops   4  ['0_tr123.mp4', '0_tr136.mp4', '0_tr138.mp4', '1_tr55.mp4']
test/keep                    drops   0  []
test/drop_conflict           drops   5  ['0_te23.mp4', '4_te4.mp4', '4_te7.mp4', '4_te9.mp4', '5_te13.mp4']
```

## Unresolved questions

1. Should the 6 leakage groups also get a train-drop option (§6), or is flagging them as a
   documented limitation sufficient for the thesis? Not implemented — easy to add if needed.
2. Same open question as the original audit: which label of each opposite pair is "correct"? Not
   resolvable from file metadata; would need the annotation guide (see
   `footage-verification-of-label-contradictions.md` — footage review suggests both labels
   describe real simultaneous events, i.e. the single-label assumption itself is the deeper issue,
   not a filing error).
3. Has any binary-task training/eval actually been re-run with `--dedupe drop_conflict --dedupe_task binary`
   yet? No — this task was audit + dedupe machinery only, no training or test-split evaluation was
   performed, per instructions. A call site needs a small additive change (new `--dedupe_task`
   flag defaulting to `'unsafe4'`) before any binary run can use this; not made here since none of
   the training/eval call sites are in this phase's file ownership.
