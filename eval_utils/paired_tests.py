"""Paired / proportion significance tests for small-N (N=64) comparisons.

- mcnemar_exact : exact binomial McNemar for two paired classifiers (needs per-clip
  correctness of both models on the SAME clips).
- two_proportion_z : unpaired z-test (used against the paper baselines, whose
  per-clip predictions are not published).
- wilson_ci : Wilson score interval for a single proportion.
- nadeau_bengio : corrected resampled paired t-test for k-fold differences.

All flag the N=64 underpowering where relevant; CIs remain the primary evidence.
"""
from __future__ import annotations
import numpy as np
from scipy import stats as _sst


def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar (sign-binomial on the discordant pairs) for two classifiers
    scored on the SAME clips — the only valid significance test for that setup (see
    plans/2026-09-06-thesis-2week-final/reports/review-2026-09-13-eval-statistics.md finding #1;
    two_proportion_z is invalid here because it assumes 2 independent samples).

    Always uses the exact binomial CDF (never the chi-square approximation), which matters because
    the discordant count b+c is typically tiny at N=64-125 (~6-8) where the chi-square
    approximation is unreliable.

    'b'/'c' follow the task-standard naming: b = a correct & b wrong, c = a wrong & b correct
    (b01/b10 are kept for backward compatibility with existing call sites/tests).
    Also returns a percentile-bootstrap 95% CI (over clips, pairing preserved by resampling the
    same index into both arrays each draw) for the paired accuracy difference acc(a)-acc(b) —
    chosen over the closed-form Wald paired-difference SE because that SE's normal approximation
    is unreliable when b+c is this small; the bootstrap makes no distributional assumption.

    On N=64 the discordant count is typically ~6-8 -> underpowered; the returned
    'underpowered' flag makes that explicit.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    b01 = int((~a & b).sum())   # a wrong, b right
    b10 = int((a & ~b).sum())   # a right, b wrong
    n_disc = b01 + b10
    if n_disc == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        p = float(min(1.0, 2.0 * _sst.binom.cdf(k, n_disc, 0.5)))
    diff_pp, ci_lo, ci_hi = paired_diff_bootstrap_ci(a, b)
    return {
        'b01': b01, 'b10': b10, 'b': b10, 'c': b01, 'n_discordant': n_disc, 'p_value': p,
        'diff_accuracy_pp': diff_pp, 'diff_accuracy_ci95_pp': [ci_lo, ci_hi],
        'ci_method': 'percentile_bootstrap_over_clips_5000',
        'underpowered': n_disc < 10,
        'n_clips': int(len(a)),
        # The note used to hardcode 'N=64', which is wrong on any corpus other than the Unsafe-Net
        # blind split (it printed N=64 beside an n=1100 grouped-CV comparison). Report the real n.
        'note': (f'N={len(a)}, {n_disc} discordant cells -> underpowered, use bootstrap CIs as '
                 f'primary evidence' if n_disc < 10 else
                 f'N={len(a)}, {n_disc} discordant cells'),
    }


def paired_diff_bootstrap_ci(correct_a, correct_b, n_boot=5000, seed=0, alpha=0.05):
    """Percentile-bootstrap 95% CI for the paired accuracy difference acc(a)-acc(b), in
    percentage points. Resamples clip indices WITH replacement and applies the SAME index to both
    arrays each draw, which preserves the pairing (a paired closed-form alternative is the Wald SE
    on (b-c)/n, but that normal approximation is unreliable when the discordant count is small, as
    it typically is here — b+c ~ 6-8 of N=64). No distributional assumption needed.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n = len(a)
    diff = float(a.mean() - b.mean()) * 100.0
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = (a[idx].mean() - b[idx].mean()) * 100.0
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return diff, float(lo), float(hi)


def two_proportion_z(k1, n1, k2, n2):
    """Unpaired two-proportion z-test. Returns z, p (two-sided), delta_pp."""
    p1, p2 = k1 / n1, k2 / n2
    ph = (k1 + k2) / (n1 + n2)
    se = (ph * (1 - ph) * (1 / n1 + 1 / n2)) ** 0.5
    z = (p1 - p2) / se if se > 0 else 0.0
    return {'z': float(z), 'p_value': float(2 * (1 - _sst.norm.cdf(abs(z)))),
            'delta_pp': float((p1 - p2) * 100)}


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a single proportion (returns lo, hi in 0..1)."""
    if n == 0:
        return (0.0, 0.0)
    z = _sst.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (float(center - half), float(center + half))


def mcnemar_discordance_bound(n_total, correct_a, correct_b):
    """McNemar bound when only total-correct counts are known for one side (the
    opponent's per-clip predictions are unavailable) — see
    plans/2026-09-05-thesis-final-push/reports/t1-objectives-audit.md S3b for the
    required semantics. Thin wrapper around unsafenet.metrics_stats.mcnemar_bound
    (reused, not reimplemented) plus the "could this favour b?" flag that section
    3b's table reports as "best split that favors us".

    Enumerates every discordant split (b01, b10) consistent with
    correct_a - correct_b = b01 - b10 and returns the smallest exact two-sided
    McNemar p across all such splits: a LOWER BOUND on p (p >= this value), which
    can only be tightened with the real per-clip predictions.

    diff = correct_a - correct_b. If diff > 0, b01 - b10 = diff > 0 for every valid
    split, i.e. discordance is structurally forced to favour a, not b — no split
    can show b ahead (`favours_b=False`). If diff < 0 the reverse holds. diff == 0
    is a tie at the aggregate level.
    """
    from unsafenet.metrics_stats import mcnemar_bound
    base = mcnemar_bound(n_total, correct_a, correct_b)
    diff = base['diff']
    max_p = _max_p_over_discordant_splits(n_total, correct_a, correct_b)
    return {
        **base,
        'max_p_over_splits': max_p,
        'p_value_bound': [base['min_p_over_splits'], max_p],
        'favours_b': diff < 0,
        'note': ('bound on how significantly a leads' if diff > 0 else
                 'bound on how significantly b leads' if diff < 0 else
                 'aggregate tie: no discordance-favoured direction'),
    }


def _max_p_over_discordant_splits(n_total, correct_a, correct_b):
    """Companion to unsafenet.metrics_stats.mcnemar_bound: that function returns the MINIMUM
    (most-significant / most-favourable) exact McNemar p across every feasible discordant split
    (b, c); this returns the MAXIMUM (least-significant / most-conservative) p over the same
    split set, so together they give the full [min_p, max_p] bound interval the aggregate-only
    comparison can support (task: 'the most and least significant p-value consistent with the
    observed marginal accuracies').

    Feasible means every cell of the 2x2 paired table is non-negative, not merely that the split
    reproduces the difference - the rule lives in unsafenet.metrics_stats.feasible_discordant_splits
    and is shared with mcnemar_bound so the two cannot drift apart (they did: the same defect then
    needed the same fix in both files). Dropping the caps (the pre-2026-09-29 behaviour, which took
    only `diff`) inflated max_p enough to reverse conclusions - the 54/64-vs-62/64 comparison read
    [0.008, 0.382] ("not robustly significant") when the feasible range is [0.008, 0.039], i.e.
    significant under EVERY admissible split. See
    reports/finding-2026-09-29-mcnemar-bound-marginal-constraint.md.
    """
    from unsafenet.metrics_stats import feasible_discordant_splits
    worst_p = 0.0
    for b, c in feasible_discordant_splits(n_total, correct_a, correct_b):
        n = b + c
        p = 1.0 if n == 0 else 2.0 * _sst.binom.cdf(min(b, c), n, 0.5)
        worst_p = max(worst_p, min(p, 1.0))
    return float(worst_p)


def paired_vs_aggregate_baseline(n, our_correct, other_correct, label='baseline'):
    """Primary 'vs baseline' significance stat for comparisons against a SAME-clip-set baseline
    (e.g. an unpublished paper score) whose per-clip predictions are unavailable, so the paired
    (b, c) table cannot be reconstructed and exact McNemar (mcnemar_exact) cannot be run.

    Returns the McNemar DISCORDANCE BOUND (min/max exact two-sided p consistent with the two
    aggregate accuracies) as the primary, valid statistic — clearly labelled as a bound, not an
    exact p — plus `two_proportion_z` kept ONLY as a legacy/diagnostic line, explicitly flagged
    invalid (it assumes 2 independent samples, which does not hold: same clips scored twice).
    See plans/2026-09-06-thesis-2week-final/reports/review-2026-09-13-eval-statistics.md finding #1.

    If real per-clip predictions for `other` ever become available, call `mcnemar_exact` instead
    (exact, not a bound) — this function is only for the aggregate-count-only case.
    """
    bound = mcnemar_discordance_bound(n, our_correct, other_correct)
    z = two_proportion_z(our_correct, n, other_correct, n)
    return {
        'label': label, 'n': n, 'our_correct': our_correct, 'other_correct': other_correct,
        'primary_test': 'mcnemar_discordance_bound (baseline per-clip predictions unavailable)',
        'p_value_bound': bound['p_value_bound'],
        'min_p_over_splits': bound['min_p_over_splits'], 'max_p_over_splits': bound['max_p_over_splits'],
        'diff': bound['diff'], 'favours_b': bound['favours_b'],
        'delta_pp': z['delta_pp'],
        'diagnostic_two_proportion_z': {'z': z['z'], 'p_value': z['p_value']},
        'diagnostic_warning': ('INVALID independence assumption on same-clip paired data (assumes '
                                '2 independent n=' + str(n) + ' samples) — kept only for continuity '
                                'with older reports; do NOT cite this p as the significance test.'),
    }


def nadeau_bengio(diffs, n_train, n_test):
    """Corrected resampled paired t-test (Nadeau & Bengio, 2003) for k-fold diffs.

    Corrects the naive t-test variance for the train/test overlap across folds.
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean, var = d.mean(), d.var(ddof=1)
    if var == 0:
        return {'t': 0.0, 'p_value': 1.0, 'mean_diff': float(mean)}
    corr = 1.0 / n + n_test / max(1, n_train)
    t = mean / ((var * corr) ** 0.5)
    p = 2 * (1 - _sst.t.cdf(abs(t), df=n - 1))
    return {'t': float(t), 'p_value': float(p), 'mean_diff': float(mean)}
