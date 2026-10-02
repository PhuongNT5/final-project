# Finding — the McNemar aggregate-only bound omitted the marginal constraint

**Date:** 2026-09-29 23:05
**Severity:** HIGH — reverses two thesis conclusions, both in the direction that flattered the author.
**Status:** code fixed, tests added and passing; thesis text sweep in progress (§6).
**Test touches consumed:** 0. This is arithmetic over *published aggregate counts*; no test-set
prediction was read.

## 1. What was wrong

When comparing our per-clip predictions against a paper that publishes only a total score, the
paired 2×2 table (b = we right / they wrong, c = we wrong / they right) cannot be reconstructed.
The repo's answer is to enumerate every split consistent with the observed difference and report
the `[min_p, max_p]` interval of the exact McNemar p.

Both enumerators constrained only

```
b - c = correct_a - correct_b      and      b + c <= n_total
```

and **not** each classifier's own wrong-count. The feasible set is smaller:

```
b <= n_total - correct_b     (they cannot be wrong more often than they are wrong)
c <= n_total - correct_a     (nor can we)
```

Without those caps the enumeration admits tables that contradict the marginals — e.g. for a paper
scoring 62/64 it happily considers splits with 8 clips the paper got wrong, when the paper is wrong
on exactly 2. Those impossible splits have large discordance D and therefore large p, so they
drove `max_p` up.

Affected: `unsafenet/metrics_stats.py:mcnemar_bound` and
`eval_utils/paired_tests.py:_max_p_over_discordant_splits` (the latter took only `diff`, so it
could not apply the caps even in principle — its signature is now
`(n_total, correct_a, correct_b)`).

`min_p` is **unaffected**: it is attained at the minimal discordance, which is always feasible.
So every published *lower* bound stands; only the upper bounds were wrong.

## 2. Corrected numbers (n = 64, exact McNemar)

| comparison | published (loose) | **corrected (feasible)** | D range | change |
| --- | --- | --- | --- | --- |
| ensemble 54/64 vs paper no-bbox 57/64 | [0.250, 0.801] | **[0.250, 0.629]** | [3, 17] | tighter, same reading |
| ensemble 54/64 vs paper **full-bbox 62/64** | [0.008, 0.382] | **[0.008, 0.039]** | [8, 12] | **REVERSES — significant at every feasible split** |
| hybrid 51/64 vs paper no-bbox 57/64 | [0.031, 0.532] | **[0.031, 0.263]** | [6, 20] | tighter, same reading |
| hybrid 51/64 vs paper **full-bbox 62/64** | [0.001, 0.207] | **[0.001, 0.007]** | [11, 15] | **REVERSES — significant at every feasible split** |
| gated 53/64 vs paper no-bbox 57/64 | [0.125, 0.708] | **[0.125, 0.481]** | [4, 18] | tighter, same reading |
| best-case 56/64 vs paper no-bbox 57/64 | — | **[1.000, 1.000]** | [1, 15] | new row; p = 1 at *every* split |
| best-case 56/64 vs paper **full-bbox 62/64** | — | **[0.031, 0.109]** | [6, 10] | new row; straddles 0.05 |

Reproduce: `python3 -c "from eval_utils.paired_tests import mcnemar_discordance_bound as B;
print(B(64, 54, 62)['p_value_bound'])"`.

## 3. Consequence for the thesis — the correction goes against us

`report/vn/bao-cao-tien-do-2026-09-17.txt` §3.4 concluded:

> "Không còn kết quả nào trước đây ghi 'có ý nghĩa thống kê' đứng vững. Hai kết quả mất tính ý nghĩa
> đều là các kết luận rằng mô hình của ta KÉM HƠN paper; việc sửa có lợi cho ta…"

That is now **false**, and the passage even flagged the risk correctly: the two results that
"lost" significance were both findings that *we are worse than the paper's full pipeline*, and
both are restored by the fix. Under the feasible split set:

- **ensemble 54/64 vs full-bbox 62/64: p ∈ [0.008, 0.039] — significant at α = 0.05 no matter how
  the discordance falls.**
- **hybrid 51/64 vs full-bbox 62/64: p ∈ [0.001, 0.007] — likewise.**

So the honest statement is: **the deficit against the paper's full-bbox pipeline IS statistically
significant**, and the thesis may not describe it as "not robustly significant". This strengthens,
not weakens, the thesis's own data-volume-bound argument — the gap to full-bbox is real and
measurable, which is exactly what the V10 negative result and the §4.6.2 cosine-collapse analysis
claim. Nothing about the *no-bbox baseline* comparison changes: that one remains non-significant
under every bound, and the macro-parity claim is untouched.

The `56/64 vs 57/64` row deserves its own note: a **one-clip** difference on n = 64 yields exact
p = 1.000 at *every* feasible split. It is not merely "not significant"; it is maximally
uninformative, and the old z-test's p = 0.7872 understated that.

## 4. Why the bug survived review

The unit test (`test_mcnemar_discordance_bound_reproduces_audit_table`) asserted only
`min_p_over_splits` — the one quantity the defect does not touch — so it passed throughout. The
docstrings described the feasible set as "consistent with the two aggregate accuracies", which
reads as if the marginals were applied; the code implemented only the difference constraint.

## 5. Fix

- `unsafenet/metrics_stats.py:mcnemar_bound` — added the two marginal caps.
- `eval_utils/paired_tests.py:_max_p_over_discordant_splits` — signature changed to take the
  marginals; caps applied; docstring records the reversal.
- `eval_utils/tests/test_statistics.py::test_mcnemar_bound_respects_marginal_wrong_counts` — new
  regression test pinning all seven bounds and asserting
  `max_p(54 vs 62) < 0.05`, the conclusion the loose bound wrongly withdrew.
- Full suite: **78 passed**.
- No recorded artifact needs regeneration: no JSON on disk contains `max_p_over_splits` or
  `p_value_bound` (checked). The stale values existed only in prose.

## 6. Text sweep

| file | status |
| --- | --- |
| `report/vn/bao-cao-tien-do-2026-09-17.txt` §3.4 table + conclusion | **done** |
| `report/chapter-04-experiments-and-results.md` §4.7.9.2 / §4.7.9.3 Table 4.16c | **done** |
| `plans/2026-09-16-binary-safe-unsafe-reframe/phase-07-number-sheet-freeze.md:27-29` | **done** |
| `README.md:94` (z-test line) | **done** |
| `report/chapter-05-edge-deployment.md:614` (z-test line) | **done** |

## Unresolved questions

1. **Does the fix change any *internal* ablation comparison?** No — internal comparisons have both
   sides' per-clip predictions and use `mcnemar_exact`, not the bound. Only paper comparisons are
   affected. Stated for the record; not re-verified run-by-run.
2. **Should the thesis lead with the restored significance?** It is a deficit-vs-paper result, so
   it belongs in §4.7.9 and §6.2 as support for the data-bound diagnosis, not in the headline.
   Author's call on emphasis.
3. The 2026-09-13 statistics review (which caught the original z-test-vs-McNemar error) did not
   catch this one. Whether its checklist should be re-run over the other bound-style statistics is
   not decided here.
