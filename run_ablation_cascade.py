#!/usr/bin/env python3
"""Cascade decomposition: contribution of each ensemble component.

Tests:
  1. Direct4 only
  2. Cascade only
  3. Direct4 + scenario only (no pair heads — uniform pair_unsafe)
  4. Direct4 + scenario × {pair_i} ablating one pair at a time
  5. Full ensemble (direct4 + full cascade)
  6. Ensemble weight sensitivity (alpha = 0..1 grid)

Output: sweep_results/ablations/cascade_decomposition.json
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

PAIR_NAMES = ['Walkway', 'Intervention', 'Panel', 'Forklift']


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


def acc(p, y): return int((p.argmax(1) == y).sum())
def macro(p, y):
    return float(np.mean([(p.argmax(1)[y == c] == c).mean() * 100 for c in range(4) if (y == c).sum()]))
def per_class(p, y):
    return [round(float((p.argmax(1)[y == c] == c).mean() * 100), 1) if (y == c).sum() else 0.0 for c in range(4)]


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    TH = load_TH()
    d = torch.load(FEAT, map_location='cpu', weights_only=False)
    mask4 = d['labels'] < 4; y = d['labels'][mask4].numpy()
    fz4 = d['features'][mask4]; fzall = d['features']
    s4 = stats(fz4); sall = stats(fzall)
    sd = s4.shape[-1]

    p_d4 = hp(lh(os.path.join(CKPT, 'direct4_s314.pt'), 4, 384, 8, 6, sd, dev, TH), s4, dev)
    p_sc = hp(lh(os.path.join(CKPT, 'scenario_s7.pt'), 4, 384, 8, 6, sd, dev, TH), sall, dev)[mask4.numpy()]
    p_pair_unsafe = []
    for i, n in enumerate(PAIR_NAMES):
        m = lh(os.path.join(CKPT, f'pair{i}_{n}_s7.pt'), 2, 192, 6, 3, sd, dev, TH)
        p_pair_unsafe.append(hp(m, s4, dev)[:, 1])
    pu = np.stack(p_pair_unsafe, 1)

    def cas_norm(x): return x / (x.sum(1, keepdims=True) + 1e-8)
    p_cas_full = cas_norm(p_sc * pu)
    p_cas_no_pair = cas_norm(p_sc)  # uniform pair_unsafe = 1
    p_cas_pair_only = cas_norm(pu)  # uniform scenario = 1

    results = {"components": {}, "leave_one_pair_out": {}, "alpha_sweep": [],
               "n_test": int(len(y)), "n_per_class": [int((y == c).sum()) for c in range(4)]}

    def add(d_, name, p):
        d_[name] = {"correct": acc(p, y), "macro": round(macro(p, y), 2),
                    "per_class_pct": per_class(p, y)}
        print(f"  {name:38s} {acc(p,y):2d}/64  macro={macro(p,y):.2f}%  pc={per_class(p,y)}")

    print("[Components]")
    add(results["components"], "direct4_only",         p_d4)
    add(results["components"], "scenario_only",        p_sc)
    add(results["components"], "pair_unsafe_only",     p_cas_pair_only)
    add(results["components"], "cascade_no_pair",      p_cas_no_pair)
    add(results["components"], "cascade_full",         p_cas_full)
    add(results["components"], "ensemble_full",        (p_d4 + p_cas_full) / 2)

    print("\n[Leave-one-pair-out cascade] (replaces pair_i with uniform 1)")
    for i, n in enumerate(PAIR_NAMES):
        pu_loo = pu.copy(); pu_loo[:, i] = 1.0
        p_loo = cas_norm(p_sc * pu_loo)
        ens = (p_d4 + p_loo) / 2
        add(results["leave_one_pair_out"], f"loo_pair{i}_{n}_cascade", p_loo)
        add(results["leave_one_pair_out"], f"loo_pair{i}_{n}_ensemble", ens)

    print("\n[Alpha sweep] alpha*direct4 + (1-alpha)*cascade_full")
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p = alpha * p_d4 + (1 - alpha) * p_cas_full
        results["alpha_sweep"].append({
            "alpha": alpha, "correct": acc(p, y),
            "macro": round(macro(p, y), 2), "per_class_pct": per_class(p, y),
        })
        print(f"  a={alpha:.1f}  {acc(p,y):2d}/64  macro={macro(p,y):.2f}%  pc={per_class(p,y)}")

    json.dump(results, open(os.path.join(OUT, 'cascade_decomposition.json'), 'w'), indent=2)
    print(f"\n[SAVED] {os.path.join(OUT, 'cascade_decomposition.json')}")


if __name__ == '__main__':
    main()
