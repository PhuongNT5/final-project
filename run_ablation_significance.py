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


def two_prop_z(k1, n1, k2, n2):
    """Unpaired two-proportion z-test. Returns z, p (two-sided)."""
    p1, p2 = k1 / n1, k2 / n2
    p_hat = (k1 + k2) / (n1 + n2)
    se = (p_hat * (1 - p_hat) * (1 / n1 + 1 / n2)) ** 0.5
    z = (p1 - p2) / se if se > 0 else 0.0
    p = 2 * (1 - sst.norm.cdf(abs(z)))
    return float(z), float(p), float(p1 - p2)


def macro_z_simulated(macro1, macro2, n1=64, n2=64, n_sim=20000):
    """Approximate z-test on macro accuracy by treating macro as a
    weighted top-1 with the same N. Returns simulation-based 2-sided p."""
    p1, p2 = macro1 / 100, macro2 / 100
    p_hat = (p1 + p2) / 2
    se = (p_hat * (1 - p_hat) * (1 / n1 + 1 / n2)) ** 0.5
    z = (p1 - p2) / se if se > 0 else 0.0
    p = 2 * (1 - sst.norm.cdf(abs(z)))
    return float(z), float(p)


def mcnemar_marginal_bounds(k_a, k_b, n):
    """Given two paired classifiers' top-1 counts k_a, k_b on same n,
    bound the McNemar chi-sq. Discordant cells satisfy
        b01 - b10 = k_a - k_b   (signed)
        b01 + b10 = D ≥ |k_a - k_b|, ≤ min(2*max-marginal-mismatch, n).

    Returns chi-sq under the BEST and WORST possible D for the
    less-accurate model, and the corresponding 2-sided p-values.
    """
    diff = abs(k_a - k_b)
    # D_min = diff (one method strictly dominates discordant cells)
    # D_max = 2 * min(k_a, k_b, n - k_a, n - k_b) + diff  (max overlap of
    # disagreements consistent with both marginals); upper-bounded by n.
    d_min = diff
    d_max = min(n, 2 * min(k_a, n - k_a, k_b, n - k_b) + diff)
    # Continuity-corrected McNemar: chi2 = (|b01-b10| - 1)^2 / (b01+b10)
    # b01 - b10 fixed at ±diff, so chi2 maximized when D minimized.
    out = {}
    for label, D in [('lower_bound (max D)', d_max), ('upper_bound (min D)', d_min)]:
        if D == 0:
            out[label] = {'D': D, 'chi2': None, 'p_value': None,
                          'note': 'zero discordant cells — perfect agreement'}
            continue
        chi2 = (max(0, diff - 1)) ** 2 / D
        p = 1 - sst.chi2.cdf(chi2, df=1)
        out[label] = {'D': int(D), 'chi2': round(chi2, 4),
                      'p_value': round(float(p), 6)}
    return out


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

    # ── F2 — z-tests ──
    print(f"\n[3] Two-proportion z-tests")

    # Top-1: ours 56/64 vs paper ConvLSTM 55/64 (paired? no — different test runs.
    # Use unpaired as conservative)
    z_top1, p_top1, d_top1 = two_prop_z(56, 64, 55, 64)
    print(f"  top-1  ours 56/64 vs paper 55/64:  z={z_top1:+.3f}  p={p_top1:.4f}  Δ={d_top1*100:+.2f}pp  → not significant")

    # Top-1: ours 56/64 vs paper full 62/64 (negative direction)
    z_top1_full, p_top1_full, d_top1_full = two_prop_z(56, 64, 62, 64)
    print(f"  top-1  ours 56/64 vs paper 62/64:  z={z_top1_full:+.3f}  p={p_top1_full:.4f}  Δ={d_top1_full*100:+.2f}pp")

    # Macro: 89.28 vs 85.78 (paper Table 7)
    z_macro, p_macro = macro_z_simulated(89.28, 85.78, n1=64, n2=64)
    print(f"  macro  ours 89.28 vs paper 85.78:  z={z_macro:+.3f}  p={p_macro:.4f}  Δ=+3.50pp")

    # Macro: 89.28 vs paper full 95.81
    z_macro_full, p_macro_full = macro_z_simulated(89.28, 95.81, n1=64, n2=64)
    print(f"  macro  ours 89.28 vs paper 95.81:  z={z_macro_full:+.3f}  p={p_macro_full:.4f}  Δ=-6.53pp")

    # ── McNemar bounds vs paper full pipeline ──
    print(f"\n[4] McNemar marginal bounds (paired, same N=64 test set, "
          f"unknown per-clip overlap)")
    mc_bounds_full = mcnemar_marginal_bounds(56, 62, 64)
    mc_bounds_baseline = mcnemar_marginal_bounds(56, 55, 64)
    print(f"  vs paper full 62/64:")
    for k, v in mc_bounds_full.items():
        print(f"    {k}:  D={v.get('D')}  chi2={v.get('chi2')}  p={v.get('p_value')}")
    print(f"  vs paper baseline 55/64:")
    for k, v in mc_bounds_baseline.items():
        print(f"    {k}:  D={v.get('D')}  chi2={v.get('chi2')}  p={v.get('p_value')}")

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
        'z_tests': {
            'top1_ours_vs_paper_baseline': {
                'ours_k': 56, 'ours_n': 64, 'paper_k': 55, 'paper_n': 64,
                'z': round(z_top1, 3), 'p': round(p_top1, 4), 'delta_pp': round(d_top1 * 100, 2),
            },
            'top1_ours_vs_paper_full': {
                'ours_k': 56, 'ours_n': 64, 'paper_k': 62, 'paper_n': 64,
                'z': round(z_top1_full, 3), 'p': round(p_top1_full, 4), 'delta_pp': round(d_top1_full * 100, 2),
            },
            'macro_ours_vs_paper_baseline': {
                'ours_macro': 89.28, 'paper_macro': 85.78, 'n': 64,
                'z': round(z_macro, 3), 'p': round(p_macro, 4), 'delta_pp': 3.50,
            },
            'macro_ours_vs_paper_full': {
                'ours_macro': 89.28, 'paper_macro': 95.81, 'n': 64,
                'z': round(z_macro_full, 3), 'p': round(p_macro_full, 4), 'delta_pp': -6.53,
            },
        },
        'mcnemar_marginal_bounds': {
            'vs_paper_full_pipeline': mc_bounds_full,
            'vs_paper_convlstm_baseline': mc_bounds_baseline,
            'note': 'Paper\'s per-clip predictions are not published, so D=b01+b10 is bounded; chi-squared is reported at both extremes of D.',
        },
    }
    json.dump(out, open(os.path.join(OUT, 'significance.json'), 'w'), indent=2)
    print(f"\n[SAVED] {os.path.join(OUT, 'significance.json')}")


if __name__ == '__main__':
    main()
