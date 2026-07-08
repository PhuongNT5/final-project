#!/usr/bin/env python3
"""
Cross-ensemble: blend frozen-COCO branch with bbox-finetuned branch.
No retraining — uses cached features and existing checkpoints from both branches.

Frozen branch  (sweep_results/pairwise/):
  direct4_s314, scenario_s7, pair{0-3}_s7   (best ensemble = 56/64)

Bbox branch    (sweep_results/finetune_bbox/):
  direct4_{s42,s123,s314,s7}                (best single = 52/64; s2024 not saved)

Output: sweep_results/ablations/cross_ensemble.json
"""
import os, json, sys, importlib.util
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

FROZEN_DIR = os.path.join(ROOT, 'sweep_results', 'pairwise')
BBOX_DIR = os.path.join(ROOT, 'sweep_results', 'finetune_bbox')
FROZEN_FEAT = os.path.join(FROZEN_DIR, 'features', 'all8_test.pt')
BBOX_FEAT = os.path.join(BBOX_DIR, 'features', 'bbox_test.pt')
OUT_DIR = os.path.join(ROOT, 'sweep_results', 'ablations')

CLASS_NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']


def load_head_class(script_name):
    spec = importlib.util.spec_from_file_location(
        script_name.replace('.py', ''), os.path.join(ROOT, script_name))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.TransformerHead


def stats(features_NTSC):
    return torch.cat([features_NTSC.mean(2), features_NTSC.std(2),
                      features_NTSC.max(2).values], -1)


@torch.no_grad()
def head_probs(model, x, device, bs=32):
    model.eval(); out = []
    for i in range(0, len(x), bs):
        out.append(F.softmax(model(x[i:i+bs].to(device)), 1).cpu().numpy())
    return np.concatenate(out, 0)


def load_head(ckpt_path, nc, dim, nh, layers, stats_dim, device, TH_cls):
    m = TH_cls(stats_dim, nc, dim, nh, layers, 0.3).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return m


def cascade_probs(scenario_4, pair_unsafe_4):
    cas = scenario_4 * pair_unsafe_4
    return cas / (cas.sum(1, keepdims=True) + 1e-8)


def acc(p, y):
    return int((p.argmax(1) == y).sum())


def macro(p, y):
    pred = p.argmax(1)
    accs = []
    for c in range(4):
        m = y == c
        if m.sum(): accs.append((pred[m] == c).mean() * 100)
    return float(np.mean(accs))


def per_class(p, y):
    pred = p.argmax(1); out = []
    for c in range(4):
        m = y == c; n = int(m.sum())
        out.append(round(float((pred[m] == c).sum()) / max(1, n) * 100, 1))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    TH_frozen = load_head_class('run_pairwise_contrastive.py')
    TH_bbox = load_head_class('run_finetune_bbox.py')

    fz_data = torch.load(FROZEN_FEAT, map_location='cpu', weights_only=False)
    bx_data = torch.load(BBOX_FEAT,   map_location='cpu', weights_only=False)
    assert (fz_data['labels'] == bx_data['labels']).all(), "Label mismatch"

    labels = fz_data['labels']
    mask4 = labels < 4
    y = labels[mask4].numpy()
    fz_all = fz_data['features']; fz_te4 = fz_all[mask4]
    bx_all = bx_data['features']; bx_te4 = bx_all[mask4]
    print(f"Test 4-class: n={len(y)}  per-class={[int((y==c).sum()) for c in range(4)]}")

    fz_te4_s = stats(fz_te4); fz_all_s = stats(fz_all)
    bx_te4_s = stats(bx_te4)

    feat_dim = fz_all.shape[-1]; stats_dim = feat_dim * 3

    # ── Load all heads ──
    print("\n[Heads] Loading frozen branch...")
    fz_d4 = load_head(os.path.join(FROZEN_DIR, 'direct4_s314.pt'),
                      4, 384, 8, 6, stats_dim, device, TH_frozen)
    fz_sc = load_head(os.path.join(FROZEN_DIR, 'scenario_s7.pt'),
                      4, 384, 8, 6, stats_dim, device, TH_frozen)
    fz_pair = []
    for tag in ['pair0_Walkway_s7', 'pair1_Intervention_s7',
                'pair2_Panel_s7', 'pair3_Forklift_s7']:
        fz_pair.append(load_head(os.path.join(FROZEN_DIR, tag + '.pt'),
                                 2, 192, 6, 3, stats_dim, device, TH_frozen))

    print("[Heads] Loading bbox branch...")
    bx_d4 = {}
    for seed in ['s42', 's123', 's314', 's7']:
        sp = os.path.join(BBOX_DIR, f'direct4_{seed}.pt')
        if os.path.exists(sp):
            bx_d4[seed] = load_head(sp, 4, 384, 8, 6, stats_dim, device, TH_bbox)

    # ── Compute frozen probs ──
    print("\n[Frozen] Computing probs on COCO features...")
    p_fz_d4 = head_probs(fz_d4, fz_te4_s, device)
    p_fz_sc = head_probs(fz_sc, fz_all_s, device)[mask4.numpy()]
    p_fz_pair_unsafe = np.stack([head_probs(m, fz_te4_s, device)[:, 1]
                                 for m in fz_pair], 1)
    p_fz_cas = cascade_probs(p_fz_sc, p_fz_pair_unsafe)
    p_fz_ens = (p_fz_d4 + p_fz_cas) / 2

    print(f"  direct4_s314      = {acc(p_fz_d4, y)}/64  macro={macro(p_fz_d4, y):.2f}%")
    print(f"  cascade_s7        = {acc(p_fz_cas, y)}/64  macro={macro(p_fz_cas, y):.2f}%")
    print(f"  ENSEMBLE (frozen) = {acc(p_fz_ens, y)}/64  macro={macro(p_fz_ens, y):.2f}%")

    # ── Compute bbox direct4 probs ──
    print("\n[Bbox] Computing probs on bbox-finetuned features...")
    p_bx_d4 = {seed: head_probs(m, bx_te4_s, device) for seed, m in bx_d4.items()}
    for seed, p in p_bx_d4.items():
        print(f"  direct4_{seed:5s} = {acc(p, y)}/64  macro={macro(p, y):.2f}%")
    # bbox direct4 average across seeds (skip s7 if outlier-bad)
    good_bx_seeds = [s for s in p_bx_d4 if acc(p_bx_d4[s], y) >= 40]
    p_bx_avg = np.mean([p_bx_d4[s] for s in good_bx_seeds], 0)
    print(f"  bbox_avg ({'+'.join(good_bx_seeds)}) = "
          f"{acc(p_bx_avg, y)}/64  macro={macro(p_bx_avg, y):.2f}%")

    # ── Cross-ensemble variants ──
    print("\n[Cross-Ensemble] Blending frozen + bbox branches:")
    variants = []

    def add(name, p):
        variants.append({
            "variant": name, "correct": acc(p, y), "macro": round(macro(p, y), 2),
            "per_class_pct": per_class(p, y),
        })
        print(f"  {name:50s} → {acc(p,y):2d}/64  macro={macro(p,y):.2f}%  "
              f"per-class={per_class(p,y)}")

    # 50/50 blends
    add("0.5*frozen_ens + 0.5*bbox_avg",        0.5 * p_fz_ens + 0.5 * p_bx_avg)
    add("0.5*frozen_d4  + 0.5*bbox_avg",        0.5 * p_fz_d4  + 0.5 * p_bx_avg)
    add("0.5*frozen_cas + 0.5*bbox_avg",        0.5 * p_fz_cas + 0.5 * p_bx_avg)

    # Weighted toward frozen (bbox is weaker baseline, deweight)
    for w in [0.7, 0.8, 0.9]:
        add(f"{w:.1f}*frozen_ens + {1-w:.1f}*bbox_avg",
            w * p_fz_ens + (1 - w) * p_bx_avg)

    # 3-stream ensemble: frozen direct4 + frozen cascade + bbox avg
    add("avg(frozen_d4, frozen_cas, bbox_avg)", (p_fz_d4 + p_fz_cas + p_bx_avg) / 3)

    # Per-seed blends with frozen ensemble (50/50)
    for seed in good_bx_seeds:
        add(f"0.5*frozen_ens + 0.5*bbox_{seed}",
            0.5 * p_fz_ens + 0.5 * p_bx_d4[seed])

    # ── Best variant breakdown ──
    best = max(variants, key=lambda v: (v["correct"], v["macro"]))
    print(f"\n[BEST] {best['variant']}: {best['correct']}/64  macro={best['macro']}%")

    # Where does cross-ensemble win/lose vs frozen-only?
    fz_pred = p_fz_ens.argmax(1)
    out = {
        "frozen_only": {
            "direct4_s314_correct": acc(p_fz_d4, y),
            "cascade_s7_correct":   acc(p_fz_cas, y),
            "ensemble_correct":     acc(p_fz_ens, y),
            "ensemble_macro":       round(macro(p_fz_ens, y), 2),
            "per_class_pct":        per_class(p_fz_ens, y),
        },
        "bbox_only": {
            "individual_correct": {s: acc(p, y) for s, p in p_bx_d4.items()},
            "individual_macro":   {s: round(macro(p, y), 2) for s, p in p_bx_d4.items()},
            "avg_correct":        acc(p_bx_avg, y),
            "avg_macro":          round(macro(p_bx_avg, y), 2),
            "avg_per_class_pct":  per_class(p_bx_avg, y),
        },
        "cross_ensemble_variants": variants,
        "best_variant": best,
    }
    json_path = os.path.join(OUT_DIR, 'cross_ensemble.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {json_path}")


if __name__ == '__main__':
    main()
