#!/usr/bin/env python3
"""
Inference-time ablation suite — uses cached test features + existing checkpoints.
No re-extraction, no re-training.

Reports:
  1. Test-time T robustness: T_eval in {8,16,24,32}, model trained at T=32
  2. Spatial-pool robustness: pool {1,2,4} (1=mean over all 16 tokens)
  3. Per-stream accuracy of each ensemble member (audit)

Output: sweep_results/ablations/inference_ablation.json + Markdown table.
"""
import os, json, sys, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CKPT_DIR = os.path.join(ROOT, 'sweep_results', 'pairwise')
FEAT_PATH = os.path.join(CKPT_DIR, 'features', 'all8_test.pt')
OUT_DIR = os.path.join(ROOT, 'sweep_results', 'ablations')

CLASS_NAMES_4 = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]


def load_head_class():
    spec = importlib.util.spec_from_file_location(
        'pw', os.path.join(ROOT, 'run_pairwise_contrastive.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TransformerHead


def stats_from_tokens(features_NTSC):
    return torch.cat([features_NTSC.mean(2), features_NTSC.std(2),
                      features_NTSC.max(2).values], -1)


def subsample_T(features_NTSC, T_target):
    T = features_NTSC.shape[1]
    if T_target >= T:
        return features_NTSC
    idx = np.linspace(0, T - 1, T_target, dtype=int)
    return features_NTSC[:, idx]


@torch.no_grad()
def head_probs(model, stats, device, batch=32):
    model.eval()
    out = []
    for i in range(0, len(stats), batch):
        x = stats[i:i + batch].to(device)
        out.append(F.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(out, 0)


def load_existing_probs(device, te4_stats, all_te_stats, mask_te4_np, TransformerHead, stats_dim):
    """Reproduce the 5 ensemble-member outputs from cached checkpoints."""
    cfg = {
        'direct4_s314': dict(nc=4, dim=384, head=8, layers=6, drop=0.3, input='te4'),
        'scenario_s7':  dict(nc=4, dim=384, head=8, layers=6, drop=0.3, input='all'),
        'pair0_Walkway_s7':       dict(nc=2, dim=192, head=6, layers=3, drop=0.3, input='te4'),
        'pair1_Intervention_s7':  dict(nc=2, dim=192, head=6, layers=3, drop=0.3, input='te4'),
        'pair2_Panel_s7':         dict(nc=2, dim=192, head=6, layers=3, drop=0.3, input='te4'),
        'pair3_Forklift_s7':      dict(nc=2, dim=192, head=6, layers=3, drop=0.3, input='te4'),
    }
    probs = {}
    for tag, c in cfg.items():
        sp = os.path.join(CKPT_DIR, tag + '.pt')
        if not os.path.exists(sp):
            print(f"  [SKIP] {tag} missing")
            continue
        m = TransformerHead(stats_dim, c['nc'], c['dim'], c['head'], c['layers'], c['drop']).to(device)
        m.load_state_dict(torch.load(sp, map_location=device, weights_only=True))
        x = te4_stats if c['input'] == 'te4' else all_te_stats
        probs[tag] = head_probs(m, x, device)
    return probs


def cascade_from_parts(scenario_4, pair_unsafe_4):
    cas = scenario_4 * pair_unsafe_4
    return cas / (cas.sum(1, keepdims=True) + 1e-8)


def ensemble_correct(probs_list, ynp4):
    if not probs_list:
        return 0
    avg = np.mean(probs_list, 0)
    return int((avg.argmax(1) == ynp4).sum())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    TransformerHead = load_head_class()

    test_data = torch.load(FEAT_PATH, map_location='cpu', weights_only=False)
    all_te = test_data['features']  # (125, 32, 16, 512)
    all_te_labels = test_data['labels']

    mask_te4 = all_te_labels < 4
    te4_full = all_te[mask_te4]
    ynp4 = all_te_labels[mask_te4].numpy()
    print(f"Test: all={tuple(all_te.shape)}, 4-class={tuple(te4_full.shape)}")

    feat_dim = int(all_te.shape[-1])
    stats_dim = feat_dim * 3

    # ────────── Ablation A: test-time T robustness ──────────
    print("\n[Ablation A] Test-time T robustness (model trained at T=32):")
    a_results = []
    for T_eval in [8, 16, 24, 32]:
        te4_T = subsample_T(te4_full, T_eval)
        all_te_T = subsample_T(all_te, T_eval)
        te4_stats = stats_from_tokens(te4_T)
        all_te_stats = stats_from_tokens(all_te_T)

        probs = load_existing_probs(device, te4_stats, all_te_stats,
                                    mask_te4.numpy(), TransformerHead, stats_dim)
        if not probs:
            continue

        d4 = probs['direct4_s314']
        sc = probs['scenario_s7'][mask_te4.numpy()]
        pair_keys = ['pair0_Walkway_s7', 'pair1_Intervention_s7',
                     'pair2_Panel_s7', 'pair3_Forklift_s7']
        pair_unsafe = np.stack([probs[k][:, 1] for k in pair_keys], 1)  # (64, 4)
        cas = cascade_from_parts(sc, pair_unsafe)
        ens = ensemble_correct([d4, cas], ynp4)
        d4_acc = int((d4.argmax(1) == ynp4).sum())
        cas_acc = int((cas.argmax(1) == ynp4).sum())
        per_class = []
        avg = (d4 + cas) / 2
        ens_pnp = avg.argmax(1)
        for c in range(4):
            m = ynp4 == c; n = int(m.sum())
            per_class.append(int((ens_pnp[m] == c).sum()) / max(1, n) * 100)
        macro = float(np.mean(per_class))
        a_results.append({
            "T": T_eval, "direct4": d4_acc, "cascade": cas_acc,
            "ensemble": ens, "macro": round(macro, 2),
            "per_class": [round(x, 1) for x in per_class],
        })
        print(f"  T={T_eval:2d}  direct4={d4_acc:2d}/64  cascade={cas_acc:2d}/64  "
              f"ensemble={ens}/64  macro={macro:.2f}%  "
              f"per-class={[f'{x:.0f}' for x in per_class]}")

    # ────────── Ablation B: per-member audit at T=32 ──────────
    print("\n[Ablation B] Ensemble-member audit at T=32:")
    te4_stats_32 = stats_from_tokens(te4_full)
    all_te_stats_32 = stats_from_tokens(all_te)
    probs32 = load_existing_probs(device, te4_stats_32, all_te_stats_32,
                                  mask_te4.numpy(), TransformerHead, stats_dim)
    b_results = []
    for k, p in probs32.items():
        if p.shape[0] == 64:
            acc = int((p.argmax(1) == ynp4).sum())
            b_results.append({"member": k, "correct_4cls": acc, "shape": list(p.shape)})
            print(f"  {k:30s} → {acc}/64")
        else:
            b_results.append({"member": k, "correct_4cls": None, "shape": list(p.shape),
                              "note": "evaluated on all 8-class test set, not 4-class"})
            print(f"  {k:30s} → on full test set (shape {p.shape})")

    # ────────── Ablation C: spatial-pool robustness ──────────
    print("\n[Ablation C] Spatial-pool robustness (test-time only):")
    c_results = []
    for pool_target in [1, 2, 4]:
        if pool_target == 4:
            te4_pooled = te4_full
        else:
            S = te4_full.shape[2]
            grid = int(S ** 0.5)
            x = te4_full.reshape(-1, te4_full.shape[1], grid, grid, te4_full.shape[3])
            x = x.permute(0, 1, 4, 2, 3)
            shape = x.shape
            x = x.reshape(-1, shape[2], grid, grid)
            x = F.adaptive_avg_pool2d(x, (pool_target, pool_target))
            x = x.reshape(shape[0], shape[1], shape[2], pool_target, pool_target)
            x = x.permute(0, 1, 3, 4, 2)
            te4_pooled = x.reshape(shape[0], shape[1], pool_target * pool_target, shape[2])

        s = torch.cat([te4_pooled.mean(2), te4_pooled.std(2), te4_pooled.max(2).values], -1)
        m = TransformerHead(stats_dim, 4, 384, 8, 6, 0.3).to(device)
        m.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'direct4_s314.pt'),
                                     map_location=device, weights_only=True))
        p = head_probs(m, s, device)
        acc = int((p.argmax(1) == ynp4).sum())
        c_results.append({"pool_grid": pool_target, "tokens": pool_target * pool_target,
                          "direct4_correct": acc})
        print(f"  pool={pool_target}x{pool_target} ({pool_target * pool_target} tokens) → "
              f"direct4={acc}/64")

    out = {"ablation_A_T_robustness": a_results,
           "ablation_B_member_audit": b_results,
           "ablation_C_pool_robustness": c_results}
    json_path = os.path.join(OUT_DIR, 'inference_ablation.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {json_path}")


if __name__ == '__main__':
    main()
