#!/usr/bin/env python3
"""Bootstrap robustness: confidence intervals on top-1 / macro-F1.

The test set is N=64 — any single-number accuracy claim is noisy.
This script bootstraps the test set 5,000 times (with replacement) and
reports the 95% percentile interval on top-1 and macro-F1 for each
ensemble component. Also reports paired-test confidence intervals on
ensemble-vs-direct4 deltas (clip-level matched pairs).

Output: sweep_results/ablations/bootstrap.json
"""
import os, sys, json, importlib.util
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
CKPT = os.path.join(ROOT, 'sweep_results', 'pairwise')
FEAT = os.path.join(CKPT, 'features', 'all8_test.pt')
OUT = os.path.join(ROOT, 'sweep_results', 'ablations')
N_BOOT = 5000
RNG = np.random.default_rng(42)


def load_TH():
    spec = importlib.util.spec_from_file_location('pw', os.path.join(ROOT, 'run_pairwise_contrastive.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.TransformerHead


def stats(f): return torch.cat([f.mean(2), f.std(2), f.max(2).values], -1)


@torch.no_grad()
def hp(m, x, dev):
    m.eval(); return F.softmax(m(x.to(dev)), 1).cpu().numpy()


def lh(p, nc, dim, nh, L, sd, dev, TH):
    m = TH(sd, nc, dim, nh, L, 0.3).to(dev)
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=True)); return m


def macro_f1(p, y, nc=4):
    pr = p.argmax(1); f = []
    for c in range(nc):
        tp = ((pr == c) & (y == c)).sum(); fp = ((pr == c) & (y != c)).sum(); fn = ((pr != c) & (y == c)).sum()
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        if prec + rec > 0: f.append(2 * prec * rec / (prec + rec))
        else: f.append(0.0)
    return float(np.mean(f))


def boot_ci(probs, y, n=N_BOOT):
    N = len(y); accs, mfs = [], []
    for _ in range(n):
        idx = RNG.integers(0, N, N)
        accs.append(float((probs[idx].argmax(1) == y[idx]).mean()))
        mfs.append(macro_f1(probs[idx], y[idx]))
    return {
        "acc": {"mean": round(float(np.mean(accs)), 4),
                "ci_lo": round(float(np.percentile(accs, 2.5)), 4),
                "ci_hi": round(float(np.percentile(accs, 97.5)), 4)},
        "f1": {"mean": round(float(np.mean(mfs)), 4),
               "ci_lo": round(float(np.percentile(mfs, 2.5)), 4),
               "ci_hi": round(float(np.percentile(mfs, 97.5)), 4)},
    }


def paired_delta(probs_a, probs_b, y, n=N_BOOT):
    """Bootstrap distribution of (acc_a - acc_b) on matched test set."""
    N = len(y); deltas = []
    for _ in range(n):
        idx = RNG.integers(0, N, N)
        a = (probs_a[idx].argmax(1) == y[idx]).mean()
        b = (probs_b[idx].argmax(1) == y[idx]).mean()
        deltas.append(float(a - b))
    return {
        "mean_delta": round(float(np.mean(deltas)), 4),
        "ci_lo": round(float(np.percentile(deltas, 2.5)), 4),
        "ci_hi": round(float(np.percentile(deltas, 97.5)), 4),
        "p_a_better": round(float(np.mean(np.array(deltas) > 0)), 4),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {dev}  N_BOOT={N_BOOT}")
    TH = load_TH()
    d = torch.load(FEAT, map_location='cpu', weights_only=False)
    mask4 = d['labels'] < 4; y = d['labels'][mask4].numpy()
    fz4 = d['features'][mask4]; fzall = d['features']
    s4 = stats(fz4); sall = stats(fzall)
    sd = s4.shape[-1]

    p_d4 = hp(lh(os.path.join(CKPT, 'direct4_s314.pt'), 4, 384, 8, 6, sd, dev, TH), s4, dev)
    p_sc = hp(lh(os.path.join(CKPT, 'scenario_s7.pt'), 4, 384, 8, 6, sd, dev, TH), sall, dev)[mask4.numpy()]
    pu = []
    for i, n in enumerate(['Walkway', 'Intervention', 'Panel', 'Forklift']):
        pu.append(hp(lh(os.path.join(CKPT, f'pair{i}_{n}_s7.pt'), 2, 192, 6, 3, sd, dev, TH), s4, dev)[:, 1])
    pu = np.stack(pu, 1)
    p_cas = p_sc * pu; p_cas = p_cas / (p_cas.sum(1, keepdims=True) + 1e-8)
    p_ens = (p_d4 + p_cas) / 2

    res = {"n_test": int(len(y)), "n_boot": N_BOOT, "components": {}, "deltas": {}}

    print("\n[Marginal CIs]")
    for name, p in [("direct4", p_d4), ("scenario", p_sc), ("cascade", p_cas), ("ensemble", p_ens)]:
        ci = boot_ci(p, y)
        res["components"][name] = ci
        print(f"  {name:9s} acc={ci['acc']['mean']:.3f} [{ci['acc']['ci_lo']:.3f}, {ci['acc']['ci_hi']:.3f}]  "
              f"F1={ci['f1']['mean']:.3f} [{ci['f1']['ci_lo']:.3f}, {ci['f1']['ci_hi']:.3f}]")

    print("\n[Paired deltas (bootstrap)]")
    for a_name, a_p, b_name, b_p in [
        ("ensemble", p_ens, "direct4", p_d4),
        ("ensemble", p_ens, "cascade", p_cas),
        ("cascade",  p_cas, "direct4", p_d4),
    ]:
        d_ = paired_delta(a_p, b_p, y)
        res["deltas"][f"{a_name}_minus_{b_name}"] = d_
        print(f"  {a_name:9s} - {b_name:9s}  Δ={d_['mean_delta']:+.3f} "
              f"[{d_['ci_lo']:+.3f}, {d_['ci_hi']:+.3f}]  P({a_name}>{b_name})={d_['p_a_better']:.3f}")

    json.dump(res, open(os.path.join(OUT, 'bootstrap.json'), 'w'), indent=2)
    print(f"\n[SAVED] {os.path.join(OUT, 'bootstrap.json')}")


if __name__ == '__main__':
    main()
