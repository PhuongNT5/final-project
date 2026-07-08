#!/usr/bin/env python3
"""
Blind held-out evaluation — removes the test-set leakage in run_pairwise_contrastive.py.

Original leakage (acknowledged in thesis §4.1): every head selects its best epoch by
accuracy on the 64-clip TEST set (train_classifier(..., test_x=te4_stats, ...)), and the
headline ensemble members were greedy-selected on test. This script instead:

  1. Carves a stratified validation slice from TRAIN (split_seed fixed, 8-class stratified).
  2. Selects each head's best epoch on that VALIDATION slice — the 64 test clips are NEVER
     seen during training/selection.
  3. Uses the FIXED headline ensemble definition (direct4 + scenario×pair cascade) — no
     greedy-on-test member search, no post-hoc bias calibration.
  4. Evaluates ONCE on the 64-clip test set → blind top-1 + macro.

Reports per-seed and mean±std over seeds for: direct4, cascade, ensemble.

Usage:
    .venv/bin/python3 run_blind_eval.py [--seeds 314 42 123] [--val-frac 0.2] [--split-seed 0]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

import run_pairwise_contrastive as pw  # reuse: TransformerHead, FeatDS, train_classifier, etc.

CLASS_NAMES_4 = pw.CLASS_NAMES_4
PAIRS = pw.PAIRS


def macro_acc(pred, y):
    return float(np.mean([100 * int((pred[y == i] == i).sum()) / max(1, int((y == i).sum()))
                          for i in range(4)]))


def stratified_fit_val(labels8, val_frac, split_seed):
    """Stratified split over the 8-class TRAIN labels → (fit_mask, val_mask) boolean arrays."""
    rng = np.random.RandomState(split_seed)
    y = labels8.numpy() if torch.is_tensor(labels8) else np.asarray(labels8)
    val = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val[idx[:n_val]] = True
    return ~val, val


def fit_select_model(model, fit_stats, fit_labels, val_stats, val_labels,
                     lr, device, epochs=300, patience=60):
    """Train on fit, select best epoch on val; return the loaded best model."""
    ds = pw.FeatDS(fit_stats, fit_labels, aug=True)
    best_st, _, _ = pw.train_classifier(model, ds, val_stats, val_labels, lr, device,
                                        sp=None, epochs=epochs, patience=patience)
    if best_st is None:
        best_st = model.state_dict()
    model.load_state_dict(best_st)
    model.eval()
    return model


def probs_on(model, stats, device):
    with torch.no_grad():
        return F.softmax(model(stats.to(device)), 1).cpu().numpy()


def _cascade(scenario4, pairwise):
    c = scenario4 * pairwise
    return c / (c.sum(1, keepdims=True) + 1e-8)


def run_seed(seed, data, device, val_frac, split_seed):
    (all_tr_stats, all_tr_labels, all_te_stats, te4_stats, ynp4,
     stats_dim, mask_te4, scenario_tr_labels, scenario_te_labels) = data

    fit_mask, val_mask = stratified_fit_val(all_tr_labels, val_frac, split_seed)
    fit_mask_t = torch.from_numpy(fit_mask); val_mask_t = torch.from_numpy(val_mask)
    lt4 = all_tr_labels < 4
    fit4 = fit_mask_t & lt4; val4 = val_mask_t & lt4
    val4_stats = all_tr_stats[val4]
    yval4 = all_tr_labels[val4].numpy()

    # ---- direct4 ----
    pw.set_seed(seed)
    m = pw.TransformerHead(stats_dim, 4, 384, 8, 6, 0.3).to(device)
    m = fit_select_model(m, all_tr_stats[fit4], all_tr_labels[fit4],
                         val4_stats, all_tr_labels[val4], 1.5e-4, device)
    direct4_te = probs_on(m, te4_stats, device)
    direct4_va = probs_on(m, val4_stats, device)

    # ---- scenario (8→4 scenario labels) ----
    pw.set_seed(seed)
    m = pw.TransformerHead(stats_dim, 4, 384, 8, 6, 0.3).to(device)
    m = fit_select_model(m, all_tr_stats[fit_mask_t], scenario_tr_labels[fit_mask_t],
                         all_tr_stats[val_mask_t], scenario_tr_labels[val_mask_t], 1.5e-4, device)
    sp4_te = probs_on(m, all_te_stats, device)[mask_te4.numpy()]
    sp4_va = probs_on(m, val4_stats, device)  # scenario idx == class idx on 4-class clips

    # ---- 4 pair binary heads ----
    pu_te, pu_va = [], []
    for pi, (u, s) in enumerate(PAIRS):
        fit_p = fit_mask_t & ((all_tr_labels == u) | (all_tr_labels == s))
        val_p = val_mask_t & ((all_tr_labels == u) | (all_tr_labels == s))
        pw.set_seed(seed)
        m = pw.TransformerHead(stats_dim, 2, 192, 6, 3, 0.3).to(device)
        m = fit_select_model(m, all_tr_stats[fit_p], (all_tr_labels[fit_p] == u).long(),
                             all_tr_stats[val_p], (all_tr_labels[val_p] == u).long(), 2e-4, device)
        pu_te.append(probs_on(m, te4_stats, device)[:, 1])
        pu_va.append(probs_on(m, val4_stats, device)[:, 1])

    def _pw(stack):
        ps = np.stack(stack, axis=1)
        return ps / (ps.sum(1, keepdims=True) + 1e-8)
    pwise_te, pwise_va = _pw(pu_te), _pw(pu_va)

    cascade_te = _cascade(sp4_te, pwise_te); cascade_va = _cascade(sp4_va, pwise_va)
    ens_te = np.mean([direct4_te, cascade_te], axis=0)
    ens_va = np.mean([direct4_va, cascade_va], axis=0)

    configs = {'direct4': (direct4_va, direct4_te),
               'cascade': (cascade_va, cascade_te),
               'ensemble': (ens_va, ens_te)}
    out = {}
    for name, (va, te) in configs.items():
        out[name] = {
            'val_top1': int((va.argmax(1) == yval4).sum()), 'val_n': len(yval4),
            'val_macro': round(macro_acc(va.argmax(1), yval4), 2),
            'test_top1': int((te.argmax(1) == ynp4).sum()),
            'test_macro': round(macro_acc(te.argmax(1), ynp4), 2)}
    # honest selection: best config BY VALIDATION (tiebreak val_macro)
    best_cfg = max(out, key=lambda k: (out[k]['val_top1'], out[k]['val_macro']))
    out['_val_selected_config'] = best_cfg
    out['_val_score'] = out[best_cfg]['val_top1'] / max(1, out[best_cfg]['val_n'])
    out['_blind_test_top1'] = out[best_cfg]['test_top1']
    out['_blind_test_macro'] = out[best_cfg]['test_macro']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[314, 42, 123])
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--split-seed', type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | seeds={args.seeds} | val_frac={args.val_frac} "
          f"| split_seed={args.split_seed}\n")

    train_data, test_data = pw.extract_all_features(device)
    all_tr_feats, all_tr_labels = train_data['features'], train_data['labels']
    all_te_feats, all_te_labels = test_data['features'], test_data['labels']

    all_tr_stats = torch.cat([all_tr_feats.mean(2), all_tr_feats.std(2), all_tr_feats.max(2).values], -1)
    all_te_stats = torch.cat([all_te_feats.mean(2), all_te_feats.std(2), all_te_feats.max(2).values], -1)
    stats_dim = int(all_tr_stats.shape[-1])

    mask_te4 = all_te_labels < 4
    te4_stats = all_te_stats[mask_te4]
    ynp4 = all_te_labels[mask_te4].numpy()

    scenario_tr_labels = all_tr_labels.clone()
    scenario_te_labels = all_te_labels.clone()
    for u, s in PAIRS:
        scenario_tr_labels[all_tr_labels == s] = u
        scenario_te_labels[all_te_labels == s] = u

    n_tr4 = int((all_tr_labels < 4).sum())
    print(f"  4-class TRAIN clips={n_tr4} | TEST clips={len(ynp4)} "
          f"(val slice ≈ {int(round(n_tr4*args.val_frac))} held FROM TRAIN)\n")

    data = (all_tr_stats, all_tr_labels, all_te_stats, te4_stats, ynp4,
            stats_dim, mask_te4, scenario_tr_labels, scenario_te_labels)

    per_seed = {}
    for seed in args.seeds:
        print(f"── seed {seed} ──", flush=True)
        r = run_seed(seed, data, device, args.val_frac, args.split_seed)
        per_seed[seed] = r
        for name in ['direct4', 'cascade', 'ensemble']:
            print(f"    {name:9s}: val {r[name]['val_top1']}/{r[name]['val_n']} "
                  f"→ TEST {r[name]['test_top1']}/64  macro={r[name]['test_macro']}%")
        print(f"    val-selected → {r['_val_selected_config']}: "
              f"TEST {r['_blind_test_top1']}/64 macro={r['_blind_test_macro']}%\n", flush=True)

    print("=" * 64)
    print("BLIND HELD-OUT SUMMARY (everything selected on VAL; test untouched)")
    print("=" * 64)
    agg = {}
    for name in ['direct4', 'cascade', 'ensemble']:
        t1 = np.array([per_seed[s][name]['test_top1'] for s in args.seeds], dtype=float)
        mc = np.array([per_seed[s][name]['test_macro'] for s in args.seeds], dtype=float)
        agg[name] = {'test_top1_mean': round(t1.mean(), 2), 'test_top1_std': round(t1.std(), 2),
                     'test_top1_per_seed': t1.tolist(),
                     'test_macro_mean': round(mc.mean(), 2), 'test_macro_std': round(mc.std(), 2)}
        print(f"  {name:9s}: TEST {t1.mean():.2f} ± {t1.std():.2f} /64 "
              f"({'/'.join(str(int(x)) for x in t1)})  macro {mc.mean():.2f}±{mc.std():.2f}%")

    # honest selection 1: per-seed val-best config, averaged
    bt = np.array([per_seed[s]['_blind_test_top1'] for s in args.seeds], dtype=float)
    bm = np.array([per_seed[s]['_blind_test_macro'] for s in args.seeds], dtype=float)
    print(f"\n  [val-selected config, mean over seeds]  TEST {bt.mean():.2f} ± {bt.std():.2f} /64 "
          f"({bt.mean()/64*100:.1f}%)  macro {bm.mean():.2f}%")

    # honest selection 2: pick the single seed with best VAL score → report its test
    best_seed = max(args.seeds, key=lambda s: per_seed[s]['_val_score'])
    bs = per_seed[best_seed]
    print(f"  [val-selected seed={best_seed} + config={bs['_val_selected_config']}]  "
          f"TEST {bs['_blind_test_top1']}/64 ({bs['_blind_test_top1']/64*100:.1f}%) "
          f"macro {bs['_blind_test_macro']}%")
    print(f"\n  baseline (no-bbox): 57/64=89.06% top-1, 85.78% macro | "
          f"best-case test-selected: 56/64, 89.28%")

    out_dir = 'plans/260613-thesis-defense-revision/reports'
    os.makedirs(out_dir, exist_ok=True)
    result = {'protocol': 'val-selected epoch AND val-selected config/seed; no calibration; '
                          'test untouched until final eval',
              'seeds': args.seeds, 'val_frac': args.val_frac, 'split_seed': args.split_seed,
              'per_seed': {str(k): v for k, v in per_seed.items()},
              'per_config_agg': agg,
              'val_selected_config_mean_test_top1': round(bt.mean(), 2),
              'val_selected_config_mean_test_macro': round(bm.mean(), 2),
              'val_selected_seed': best_seed,
              'val_selected_seed_test_top1': bs['_blind_test_top1'],
              'val_selected_seed_test_macro': bs['_blind_test_macro'],
              'runtime_s': round(time.time() - t0, 1)}
    tag = f"v2-vf{args.val_frac}"
    json.dump(result, open(os.path.join(out_dir, f'blind-eval-{tag}.json'), 'w'), indent=2)
    print(f"\n[SAVED] {out_dir}/blind-eval-{tag}.json | {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
