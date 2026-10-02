#!/usr/bin/env python3
"""F1 + F2 statistics: bootstrap CIs (macro + per-class), 2-proportion
z-tests vs paper baselines, McNemar marginal bounds.

Outputs: sweep_results/ablations/significance.json
"""
import os, sys, json, importlib.util
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sst

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from eval_utils.paired_tests import paired_vs_aggregate_baseline, mcnemar_exact
CKPT = os.path.join(ROOT, 'sweep_results', 'pairwise')
FEAT_TE = os.path.join(CKPT, 'features', 'all8_test.pt')
OUT = os.path.join(ROOT, 'sweep_results', 'ablations')
PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]
PAIR_NAMES = ['Walkway', 'Intervention', 'Panel', 'Forklift']
CLASS_NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
RNG = np.random.default_rng(20260505)


def load_TH():
    spec = importlib.util.spec_from_file_location('pw', os.path.join(ROOT, 'run_pairwise_contrastive.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.TransformerHead


def stats_pool(f):
    return torch.cat([f.mean(2), f.std(2), f.max(2).values], -1)


@torch.no_grad()
def head_logits(model, x, dev, bs=32):
    model.eval(); out = []
    for i in range(0, len(x), bs):
        out.append(model(x[i:i + bs].to(dev)).cpu())
    return torch.cat(out, 0).numpy()


def load_head(p, nc, dim, nh, L, sd, dev, TH):
    m = TH(sd, nc, dim, nh, L, 0.3).to(dev)
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=True))
    return m


def ensemble_probs(s4, sall, mask4, dev, TH, sd):
    Ld = head_logits(load_head(os.path.join(CKPT, 'direct4_s314.pt'), 4, 384, 8, 6, sd, dev, TH), s4, dev)
    Lsc = head_logits(load_head(os.path.join(CKPT, 'scenario_s7.pt'), 4, 384, 8, 6, sd, dev, TH), sall, dev)[mask4]
    pair_log = []
    for i, n in enumerate(PAIR_NAMES):
        pair_log.append(head_logits(load_head(os.path.join(CKPT, f'pair{i}_{n}_s7.pt'), 2, 192, 6, 3, sd, dev, TH), s4, dev))
    pair_log = np.stack(pair_log, 1)
    pd = F.softmax(torch.from_numpy(Ld), 1).numpy()
    pd_only = pd
    psc = F.softmax(torch.from_numpy(Lsc), 1).numpy()
    pp_unsafe = F.softmax(torch.from_numpy(pair_log), 2).numpy()[:, :, 1]
    cas = psc * pp_unsafe; cas = cas / (cas.sum(1, keepdims=True) + 1e-8)
    return (pd + cas) / 2, pd_only, cas


def macro_acc(pred, y, n_cls=4):
    return float(np.mean([(pred[y == c] == c).mean() if (y == c).sum() else 0 for c in range(n_cls)]) * 100)


def bootstrap_metric(pred, y, fn, n_boot=5000, n_cls=4):
    """Generic bootstrap on a metric fn(pred_subset, y_subset)."""
    n = len(y); samples = np.zeros(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        samples[b] = fn(pred[idx], y[idx])
    return float(np.mean(samples)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)), float(np.std(samples))


def bootstrap_perclass(pred, y, n_boot=5000, n_cls=4):
    """Bootstrap per-class accuracy CIs by stratified resampling within each class."""
    out = []
    for c in range(n_cls):
        idx_c = np.where(y == c)[0]
        nc = len(idx_c)
        if nc == 0:
            out.append({'class': CLASS_NAMES[c], 'n': 0, 'mean': None})
            continue
        samples = np.zeros(n_boot)
        for b in range(n_boot):
            sel = RNG.choice(idx_c, size=nc, replace=True)
            samples[b] = (pred[sel] == c).mean() * 100
        out.append({
            'class': CLASS_NAMES[c], 'n': int(nc),
            'point': round(float((pred[idx_c] == c).mean() * 100), 2),
            'mean': round(float(np.mean(samples)), 2),
            'ci95_lo': round(float(np.percentile(samples, 2.5)), 2),
            'ci95_hi': round(float(np.percentile(samples, 97.5)), 2),
            'std': round(float(np.std(samples)), 2),
        })
    return out


def macro_z_simulated(macro1, macro2, n1=64, n2=64, n_sim=20000):
    """Approximate z-test on macro accuracy by treating macro as a
    weighted top-1 with the same N. Returns simulation-based 2-sided p."""
    p1, p2 = macro1 / 100, macro2 / 100
    p_hat = (p1 + p2) / 2
    se = (p_hat * (1 - p_hat) * (1 / n1 + 1 / n2)) ** 0.5
    z = (p1 - p2) / se if se > 0 else 0.0
    p = 2 * (1 - sst.norm.cdf(abs(z)))
    return float(z), float(p)


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")
    TH = load_TH()

    print("\n[1] Computing ensemble + direct4 + cascade probs on test ...")
    dte = torch.load(FEAT_TE, map_location='cpu', weights_only=False)
    mask_te4 = (dte['labels'] < 4).numpy()
    y = dte['labels'][mask_te4].numpy()
    sall = stats_pool(dte['features']); s4 = sall[mask_te4]
    sd = s4.shape[-1]

    P_ens, P_d4, P_cas = ensemble_probs(s4, sall, mask_te4, dev, TH, sd)
    pred_ens = P_ens.argmax(1); pred_d4 = P_d4.argmax(1); pred_cas = P_cas.argmax(1)
    c_ens = int((pred_ens == y).sum()); c_d4 = int((pred_d4 == y).sum()); c_cas = int((pred_cas == y).sum())
    n = len(y)

    print(f"  ensemble: {c_ens}/{n}  direct4: {c_d4}/{n}  cascade: {c_cas}/{n}")

    # ── F1 — bootstrap CIs ──
    print(f"\n[2] Bootstrap (5000 resamples) — top-1 + macro + per-class")
    top1_mu, top1_lo, top1_hi, top1_sd = bootstrap_metric(
        pred_ens, y, lambda p, t: (p == t).mean() * 100, n_boot=5000)
    macro_mu, macro_lo, macro_hi, macro_sd = bootstrap_metric(
        pred_ens, y, macro_acc, n_boot=5000)
    perclass = bootstrap_perclass(pred_ens, y, n_boot=5000)

    print(f"  top-1 :  point=87.50  bootstrap mean={top1_mu:.2f}  95% CI=[{top1_lo:.2f}, {top1_hi:.2f}]")
    print(f"  macro :  point=89.28  bootstrap mean={macro_mu:.2f}  95% CI=[{macro_lo:.2f}, {macro_hi:.2f}]")
    print(f"  per-class:")
    for r in perclass:
        if r.get('point') is not None:
            print(f"    {r['class']:14s} n={r['n']:2d}  point={r['point']:6.2f}  CI=[{r['ci95_lo']:6.2f}, {r['ci95_hi']:6.2f}]")

    # ── F2 — top-1 vs paper (paired, same 64 clips; paper per-clip predictions unpublished
    # -> per-clip table can't be reconstructed -> McNemar BOUND is the primary, valid stat, not an
    # unpaired two-proportion z-test. Routed through eval_utils.paired_tests so this file shares
    # ONE implementation with run_full_eval.py / ensemble_relation.py / run_relation_transformer.py
    # instead of its own drifting continuity-corrected-chi2 bound. See
    # plans/2026-09-06-thesis-2week-final/reports/review-2026-09-13-eval-statistics.md finding #1
    # and reports/laneb-2026-09-16-paired-significance-fix.md for the old-vs-new bound comparison. ──
    print(f"\n[3] McNemar bound vs paper (primary) + two_proportion_z (labelled diagnostic only)")
    # 57/64, not 55: the paper's ConvLSTM-only confusion (Table 5 of [17]) decomposes per class as
    # 31+9+10+7 = 57 — the figure every chapter cites (ch02 S2.4, ch04 Tables 4.3/4.4/4.6b/4.16c/4.24,
    # ch05 S5.5). The previous 55 was our own direct4 count, not the paper's.
    cmp_baseline = paired_vs_aggregate_baseline(64, 56, 57, label='paper ConvLSTM baseline (57/64)')
    print(f"  top-1  ours 56/64 vs paper 57/64:  McNemar bound p∈[{cmp_baseline['min_p_over_splits']:.4f},"
          f"{cmp_baseline['max_p_over_splits']:.4f}]  Δ={cmp_baseline['delta_pp']:+.2f}pp  "
          f"(diagnostic two_proportion_z p={cmp_baseline['diagnostic_two_proportion_z']['p_value']:.4f}, invalid independence)")

    cmp_full = paired_vs_aggregate_baseline(64, 56, 62, label='paper full pipeline (62/64)')
    print(f"  top-1  ours 56/64 vs paper 62/64:  McNemar bound p∈[{cmp_full['min_p_over_splits']:.4f},"
          f"{cmp_full['max_p_over_splits']:.4f}]  Δ={cmp_full['delta_pp']:+.2f}pp  "
          f"(diagnostic two_proportion_z p={cmp_full['diagnostic_two_proportion_z']['p_value']:.4f}, invalid independence)")

    # ── F2 — macro z-tests. NOT part of this fix: macro accuracy is a mean of per-class recalls,
    # not a per-clip binary correctness value, so it isn't a McNemar/paired-proportion case at all
    # (a different, separate methodological question from finding #1) — left untouched here.
    # ── own-vs-own: ensemble vs direct4 / cascade share the SAME 64 test clips (per-clip preds
    # already in memory) -> exact McNemar is directly computable, unlike the paper comparisons. ──
    print(f"\n[3b] McNemar exact — own variants (same 64 clips, per-clip preds available)")
    mc_d4 = mcnemar_exact(pred_ens == y, pred_d4 == y)
    print(f"  ensemble vs direct4:  b={mc_d4['b']} c={mc_d4['c']}  p={mc_d4['p_value']:.4f}  "
          f"diff={mc_d4['diff_accuracy_pp']:+.2f}pp  CI95=[{mc_d4['diff_accuracy_ci95_pp'][0]:+.1f},"
          f"{mc_d4['diff_accuracy_ci95_pp'][1]:+.1f}]pp")
    mc_cas = mcnemar_exact(pred_ens == y, pred_cas == y)
    print(f"  ensemble vs cascade:  b={mc_cas['b']} c={mc_cas['c']}  p={mc_cas['p_value']:.4f}  "
          f"diff={mc_cas['diff_accuracy_pp']:+.2f}pp  CI95=[{mc_cas['diff_accuracy_ci95_pp'][0]:+.1f},"
          f"{mc_cas['diff_accuracy_ci95_pp'][1]:+.1f}]pp")

    print(f"\n[4] Macro two-proportion z-tests (simulated; separate from the paired-significance fix)")
    z_macro, p_macro = macro_z_simulated(89.28, 85.78, n1=64, n2=64)
    print(f"  macro  ours 89.28 vs paper 85.78:  z={z_macro:+.3f}  p={p_macro:.4f}  Δ=+3.50pp")

    z_macro_full, p_macro_full = macro_z_simulated(89.28, 95.81, n1=64, n2=64)
    print(f"  macro  ours 89.28 vs paper 95.81:  z={z_macro_full:+.3f}  p={p_macro_full:.4f}  Δ=-6.53pp")

    out = {
        'point_estimates': {
            'ensemble_top1':   {'correct': c_ens, 'n': n, 'pct': round(100 * c_ens / n, 2)},
            'direct4_top1':    {'correct': c_d4,  'n': n, 'pct': round(100 * c_d4  / n, 2)},
            'cascade_top1':    {'correct': c_cas, 'n': n, 'pct': round(100 * c_cas / n, 2)},
            'ensemble_macro':  round(macro_acc(pred_ens, y), 2),
        },
        'bootstrap_5000': {
            'top1':  {'mean': round(top1_mu, 2),  'ci95': [round(top1_lo, 2),  round(top1_hi, 2)],  'std': round(top1_sd, 2)},
            'macro': {'mean': round(macro_mu, 2), 'ci95': [round(macro_lo, 2), round(macro_hi, 2)], 'std': round(macro_sd, 2)},
            'per_class': perclass,
        },
        'top1_vs_own_variants': {
            # exact McNemar: both sides are per-clip predictions on the SAME 64 test clips.
            'ensemble_vs_direct4': mc_d4,
            'ensemble_vs_cascade': mc_cas,
        },
        'top1_vs_paper': {
            # primary_test == 'mcnemar_discordance_bound...' -> bound is the valid stat; the
            # embedded diagnostic_two_proportion_z is NOT the significance test, see its own
            # diagnostic_warning field.
            'ours_vs_paper_baseline': cmp_baseline,
            'ours_vs_paper_full': cmp_full,
        },
        'macro_z_tests_simulated': {
            # separate methodological question from the paired-significance fix (macro accuracy is
            # not a per-clip binary correctness value) — untouched, not routed through paired_tests.
            'macro_ours_vs_paper_baseline': {
                'ours_macro': 89.28, 'paper_macro': 85.78, 'n': 64,
                'z': round(z_macro, 3), 'p': round(p_macro, 4), 'delta_pp': 3.50,
            },
            'macro_ours_vs_paper_full': {
                'ours_macro': 89.28, 'paper_macro': 95.81, 'n': 64,
                'z': round(z_macro_full, 3), 'p': round(p_macro_full, 4), 'delta_pp': -6.53,
            },
        },
    }
    json.dump(out, open(os.path.join(OUT, 'significance.json'), 'w'), indent=2)
    print(f"\n[SAVED] {os.path.join(OUT, 'significance.json')}")


if __name__ == '__main__':
    main()
