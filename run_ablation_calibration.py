#!/usr/bin/env python3
"""Ablation: temperature calibration sensitivity.

Sweeps temperature T over [0.5, 0.6, ..., 2.0] and measures top-1, NLL,
expected calibration error (ECE), and per-class accuracy at each T.

Produces:
  - sweep_results/ablations/calibration.json
  - figures/calibration_curve.png
"""
import os, sys, json, importlib.util
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
CKPT = os.path.join(ROOT, 'sweep_results', 'pairwise')
FEAT = os.path.join(CKPT, 'features', 'all8_test.pt')
OUT = os.path.join(ROOT, 'sweep_results', 'ablations')
FIG = os.path.join(ROOT, 'figures')


def load_TH():
    spec = importlib.util.spec_from_file_location('pw', os.path.join(ROOT, 'run_pairwise_contrastive.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.TransformerHead


def stats(f): return torch.cat([f.mean(2), f.std(2), f.max(2).values], -1)


@torch.no_grad()
def logits(model, x, dev, bs=32):
    model.eval(); out = []
    for i in range(0, len(x), bs):
        out.append(model(x[i:i + bs].to(dev)).cpu())
    return torch.cat(out, 0).numpy()


def lh(p, nc, dim, nh, L, sd, dev, TH):
    m = TH(sd, nc, dim, nh, L, 0.3).to(dev)
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=True)); return m


def ece(p, y, n_bins=10):
    """Expected Calibration Error."""
    conf = p.max(1); pred = p.argmax(1); correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            e += (m.sum() / len(y)) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def nll(p, y):
    eps = 1e-12; return float(-np.log(p[np.arange(len(y)), y] + eps).mean())


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    TH = load_TH()
    d = torch.load(FEAT, map_location='cpu', weights_only=False)
    mask4 = d['labels'] < 4; y = d['labels'][mask4].numpy()
    fz4 = d['features'][mask4]; fzall = d['features']
    s4 = stats(fz4); sall = stats(fzall)
    sd = s4.shape[-1]

    # Get logits for direct4 + cascade ensemble
    Ld = logits(lh(os.path.join(CKPT, 'direct4_s314.pt'), 4, 384, 8, 6, sd, dev, TH), s4, dev)
    Lsc = logits(lh(os.path.join(CKPT, 'scenario_s7.pt'), 4, 384, 8, 6, sd, dev, TH), sall, dev)[mask4.numpy()]
    pair_log = []
    for i, n in enumerate(['Walkway', 'Intervention', 'Panel', 'Forklift']):
        pair_log.append(logits(lh(os.path.join(CKPT, f'pair{i}_{n}_s7.pt'), 2, 192, 6, 3, sd, dev, TH), s4, dev))
    pair_log = np.stack(pair_log, 1)  # (64, 4, 2)

    sweep = []
    for T in np.arange(0.5, 2.05, 0.1):
        pd = F.softmax(torch.from_numpy(Ld / T), 1).numpy()
        psc = F.softmax(torch.from_numpy(Lsc / T), 1).numpy()
        pp_unsafe = F.softmax(torch.from_numpy(pair_log / T), 2).numpy()[:, :, 1]
        cas = psc * pp_unsafe; cas = cas / (cas.sum(1, keepdims=True) + 1e-8)
        ens = (pd + cas) / 2
        sweep.append({
            "T": round(float(T), 2),
            "correct": int((ens.argmax(1) == y).sum()),
            "macro": round(float(np.mean([(ens.argmax(1)[y == c] == c).mean() * 100 for c in range(4) if (y == c).sum()])), 2),
            "nll": round(nll(ens, y), 4),
            "ece": round(ece(ens, y), 4),
            "mean_confidence": round(float(ens.max(1).mean()), 4),
        })
    for r in sweep: print(f"  T={r['T']:.2f}  ens={r['correct']:2d}/64  macro={r['macro']:.2f}%  "
                          f"NLL={r['nll']:.4f}  ECE={r['ece']:.4f}  conf={r['mean_confidence']:.3f}")

    json.dump({"sweep": sweep}, open(os.path.join(OUT, 'calibration.json'), 'w'), indent=2)
    print(f"\n[SAVED] {os.path.join(OUT, 'calibration.json')}")

    # Plot
    Ts = [r["T"] for r in sweep]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(Ts, [r["correct"] for r in sweep], 'o-', color='#264653', linewidth=2, markersize=6)
    ax[0].set_xlabel('Temperature T'); ax[0].set_ylabel('Correct (out of 64)')
    ax[0].set_title('Top-1 vs T'); ax[0].grid(alpha=0.3); ax[0].axvline(1.0, ls='--', color='gray', alpha=0.5)
    ax[1].plot(Ts, [r["nll"] for r in sweep], 'o-', color='#e63946', linewidth=2, markersize=6)
    ax[1].set_xlabel('Temperature T'); ax[1].set_ylabel('NLL')
    ax[1].set_title('NLL vs T'); ax[1].grid(alpha=0.3); ax[1].axvline(1.0, ls='--', color='gray', alpha=0.5)
    ax[2].plot(Ts, [r["ece"] for r in sweep], 'o-', color='#2a9d8f', linewidth=2, markersize=6)
    ax[2].set_xlabel('Temperature T'); ax[2].set_ylabel('ECE')
    ax[2].set_title('ECE vs T'); ax[2].grid(alpha=0.3); ax[2].axvline(1.0, ls='--', color='gray', alpha=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'calibration_curve.png'), dpi=140); plt.close()
    print(f"[SAVED] {os.path.join(FIG, 'calibration_curve.png')}")


if __name__ == '__main__':
    main()
