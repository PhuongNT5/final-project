#!/usr/bin/env python3
"""Probe: t-SNE / UMAP of [CLS] embeddings from each ensemble member.

Visualizes whether the pairwise-contrastive training successfully separates
the WalkwayViol ↔ UnauthorInt confusion pair in [CLS] space.

For each model:
  - Extract [CLS] vector per test clip (after final encoder layer, before MLP head)
  - Compute pairwise cosine similarity matrix → cite the 0.94+ confusion-pair stat
  - 2-D t-SNE projection → save PNG
  - Per-pair cosine distance summary

Output:
  sweep_results/ablations/embedding_probe.json
  figures/embedding_tsne_*.png
"""
import os, sys, json, importlib.util
import numpy as np
import torch
import torch.nn as nn
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

CLASS_NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
COLORS = ['#e63946', '#2a9d8f', '#264653', '#f4a261']


def load_TH():
    spec = importlib.util.spec_from_file_location('pw', os.path.join(ROOT, 'run_pairwise_contrastive.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.TransformerHead


def stats(f): return torch.cat([f.mean(2), f.std(2), f.max(2).values], -1)


@torch.no_grad()
def get_cls_embeddings(model, x, dev, bs=32):
    """Forward through model up to final encoder, return [CLS] embedding (post-LN)."""
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(dev)
        if xb.dim() == 4: xb = xb.mean(2)
        b, t, _ = xb.shape; dvc = xb.device
        a = (model.appear(xb)
             + model.tpos(torch.arange(t, device=dvc))
             + model.te(torch.ones(t, dtype=torch.long, device=dvc)))
        diff = xb[:, 1:] - xb[:, :-1]
        m_ = (model.motion(diff)
              + model.tpos(torch.arange(t - 1, device=dvc))
              + model.te(torch.full((t - 1,), 2, dtype=torch.long, device=dvc)))
        cls = (model.cls.expand(b, -1, -1)
               + model.te(torch.zeros(1, dtype=torch.long, device=dvc)))
        h = model.transformer(model.pre(torch.cat([cls, a, m_], 1)))
        out.append(model.post(h[:, 0]).cpu())
    return torch.cat(out, 0).numpy()


def cos_sim(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return An @ Bn.T


def pair_stats(emb, y, c1, c2):
    e1 = emb[y == c1]; e2 = emb[y == c2]
    if len(e1) == 0 or len(e2) == 0: return None
    s = cos_sim(e1, e2)
    return {"between_mean": float(s.mean()), "between_max": float(s.max()),
            "within_c1_mean": float(cos_sim(e1, e1)[np.triu_indices(len(e1), 1)].mean()) if len(e1) > 1 else None,
            "within_c2_mean": float(cos_sim(e2, e2)[np.triu_indices(len(e2), 1)].mean()) if len(e2) > 1 else None}


def tsne_plot(emb, y, title, save_path):
    from sklearn.manifold import TSNE
    z = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto').fit_transform(emb)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for c in range(4):
        m = y == c
        if m.sum() == 0: continue
        ax.scatter(z[m, 0], z[m, 1], c=COLORS[c], label=f'{CLASS_NAMES[c]} (n={int(m.sum())})',
                   s=80, alpha=0.85, edgecolor='k', linewidth=0.6)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9, loc='best')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(save_path, dpi=140); plt.close()
    return z


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    TH = load_TH()
    d = torch.load(FEAT, map_location='cpu', weights_only=False)
    mask4 = d['labels'] < 4; y = d['labels'][mask4].numpy()
    fz4 = d['features'][mask4]; s4 = stats(fz4)
    sd = s4.shape[-1]

    models = {
        'direct4_s314': dict(p='direct4_s314.pt', nc=4, dim=384, nh=8, L=6),
        'pair0_Walkway': dict(p='pair0_Walkway_s7.pt', nc=2, dim=192, nh=6, L=3),
        'pair1_Intervention': dict(p='pair1_Intervention_s7.pt', nc=2, dim=192, nh=6, L=3),
        'pair2_Panel': dict(p='pair2_Panel_s7.pt', nc=2, dim=192, nh=6, L=3),
        'pair3_Forklift': dict(p='pair3_Forklift_s7.pt', nc=2, dim=192, nh=6, L=3),
    }

    pair_keys_to_label = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    summary = {"models": {}, "raw_yolo_baseline": {}}

    # Raw frozen YOLO features (mean-pool across tokens & time → 512-D vector per clip)
    raw = fz4.mean(2).mean(1).numpy()  # (64, 512)
    print("[Raw YOLO features] cosine similarities between class pairs:")
    raw_pairs = {}
    for c1, c2 in pair_keys_to_label:
        s = pair_stats(raw, y, c1, c2)
        if s: raw_pairs[f'{CLASS_NAMES[c1]}_vs_{CLASS_NAMES[c2]}'] = s
        print(f"  {CLASS_NAMES[c1]:14s} vs {CLASS_NAMES[c2]:14s} : "
              f"between={s['between_mean']:.3f} (max={s['between_max']:.3f})")
    summary['raw_yolo_baseline'] = raw_pairs
    tsne_plot(raw, y, 'Raw frozen YOLOv11s features (mean-pool)', os.path.join(FIG, 'embedding_tsne_raw.png'))

    for name, c in models.items():
        print(f"\n[{name}]")
        path = os.path.join(CKPT, c['p'])
        m = TH(sd, c['nc'], c['dim'], c['nh'], c['L'], 0.3).to(dev)
        m.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
        emb = get_cls_embeddings(m, s4, dev)
        print(f"  [CLS] embedding shape: {emb.shape}")
        pairs = {}
        for c1, c2 in pair_keys_to_label:
            s = pair_stats(emb, y, c1, c2)
            if s: pairs[f'{CLASS_NAMES[c1]}_vs_{CLASS_NAMES[c2]}'] = s
            if s and (c1, c2) in [(0, 1), (0, 2), (0, 3), (2, 3)]:
                print(f"  {CLASS_NAMES[c1]:14s} vs {CLASS_NAMES[c2]:14s} : "
                      f"between={s['between_mean']:.3f}  within={s['within_c1_mean']:.3f}/{s['within_c2_mean']:.3f}")
        summary['models'][name] = {"emb_dim": int(emb.shape[1]), "pairs": pairs}
        tsne_plot(emb, y, f'{name} — [CLS] embedding (4-class)',
                  os.path.join(FIG, f'embedding_tsne_{name}.png'))

    json.dump(summary, open(os.path.join(OUT, 'embedding_probe.json'), 'w'), indent=2)
    print(f"\n[SAVED] {os.path.join(OUT, 'embedding_probe.json')}")
    print(f"[SAVED] figures: embedding_tsne_*.png")


if __name__ == '__main__':
    main()
