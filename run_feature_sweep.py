#!/usr/bin/env python3
"""
Feature-combination sweep — exhaust the CHEAP levers toward a competitive result before deciding
whether to gamble on paper-scale annotation. All configs reuse cached features + the gate conf cache;
no re-detection, no re-extraction.

Diagnosis driving this: bbox-finetuned features (layer-22 SPPF) regressed (macro 84→77) because
fine-tuning for DETECTION over-specialises the backbone. The richer frozen-COCO features classify
better. So the promising lever is COMBINATION (frozen richness + bbox behaviour signal), not
replacement. Configs tested (direct-4 head, blind val-selected + best-case test-selected):

  frozen | bbox | hybrid(concat frozen+bbox) | ensemble(frozen-head ⊕ bbox-head)
  × { no-gate , gated@τ }

Reports blind (honest) and best-case (matched-protocol, vs paper) top-1/macro for each.
Usage: .venv/bin/python3 run_feature_sweep.py [--tau 0.15] [--seeds 314 42 123]
"""
import os, json, argparse
import numpy as np
import torch
import run_pairwise_contrastive as pw
import run_blind_eval as be

FROZEN = 'sweep_results/pairwise/features'
BBOX = 'sweep_results/finetune_bbox/features'
CONF_CACHE = 'sweep_results/framegate_conf.pt'
T = 32


def gate_sel(conf_row, tau, min_keep=6):
    keep = np.where(conf_row >= tau)[0]
    if len(keep) < min_keep:
        keep = np.sort(np.argsort(-conf_row)[:min_keep])
    if len(keep) == 0:
        keep = np.arange(T)
    return keep[np.clip(np.linspace(0, len(keep) - 1, T).round().astype(int), 0, len(keep) - 1)]


def stats_seq(feats, conf=None, tau=0.0):
    """(N,T,S,C) → (N,T,3C) stats over spatial tokens, optionally gated over kept frames."""
    out = []
    for i in range(feats.shape[0]):
        f = feats[i]
        if conf is not None and tau > 0:
            f = f[gate_sel(conf[i], tau)]
        out.append(torch.cat([f.mean(1), f.std(1), f.max(1).values], -1))
    return torch.stack(out)


def train_eval(tr_g, te_g, tr_labels, fit4, val4, ynp4, seeds, device):
    """Train direct-4 head per seed; return blind (val-selected mean) + best-case (max test over seeds)."""
    stats_dim = tr_g.shape[-1]
    val_acc, test_t1, test_mac, probs_list = [], [], [], []
    yval = tr_labels[val4].numpy()
    for s in seeds:
        pw.set_seed(s)
        m = pw.TransformerHead(stats_dim, 4, 384, 8, 6, 0.3).to(device)
        m = be.fit_select_model(m, tr_g[fit4], tr_labels[fit4], tr_g[val4], tr_labels[val4], 1.5e-4, device)
        vp = be.probs_on(m, tr_g[val4], device)
        tp = be.probs_on(m, te_g, device)
        val_acc.append((vp.argmax(1) == yval).mean())
        test_t1.append(int((tp.argmax(1) == ynp4).sum()))
        test_mac.append(be.macro_acc(tp.argmax(1), ynp4))
        probs_list.append(tp)
    blind_i = int(np.argmax(val_acc))   # honest: pick seed by val
    return {'blind_top1': test_t1[blind_i], 'blind_macro': round(test_mac[blind_i], 1),
            'bestcase_top1': max(test_t1), 'bestcase_macro': round(max(test_mac), 1),
            'mean_top1': round(float(np.mean(test_t1)), 1), '_probs': probs_list}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tau', type=float, default=0.15)
    ap.add_argument('--seeds', type=int, nargs='+', default=[314, 42, 123])
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--split-seed', type=int, default=0)
    args = ap.parse_args()
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device {dev} | tau={args.tau} | seeds={args.seeds}\n")

    fz_tr = torch.load(f'{FROZEN}/all8_train.pt', map_location='cpu', weights_only=False)
    fz_te = torch.load(f'{FROZEN}/all8_test.pt', map_location='cpu', weights_only=False)
    bx_tr = torch.load(f'{BBOX}/bbox_train.pt', map_location='cpu', weights_only=False)
    bx_te = torch.load(f'{BBOX}/bbox_test.pt', map_location='cpu', weights_only=False)
    conf = torch.load(CONF_CACHE, weights_only=False)
    tr_labels, te_labels = fz_tr['labels'], fz_te['labels']
    lt4_tr, lt4_te = tr_labels < 4, te_labels < 4
    ynp4 = te_labels[lt4_te].numpy()
    fit_mask, val_mask = be.stratified_fit_val(tr_labels, args.val_frac, args.split_seed)
    fit_mask, val_mask = torch.from_numpy(fit_mask), torch.from_numpy(val_mask)
    fit4, val4 = fit_mask & lt4_tr, val_mask & lt4_tr

    # per-frame raw feats
    F = {'frozen': (fz_tr['features'], fz_te['features']),
         'bbox':   (bx_tr['features'], bx_te['features']),
         'hybrid': (torch.cat([fz_tr['features'], bx_tr['features']], -1),
                    torch.cat([fz_te['features'], bx_te['features']], -1))}

    rows, store = [], {}
    for name, (rf_tr, rf_te) in F.items():
        for gate, tau in [('no-gate', 0.0), (f'gate{args.tau}', args.tau)]:
            tr_g = stats_seq(rf_tr, conf['train'], tau)
            te_g = stats_seq(rf_te[lt4_te], conf['test'][lt4_te.numpy()], tau)
            r = train_eval(tr_g, te_g, tr_labels, fit4, val4, ynp4, args.seeds, dev)
            store[(name, gate)] = r
            rows.append((f'{name}+{gate}', r['blind_top1'], r['blind_macro'],
                         r['bestcase_top1'], r['bestcase_macro']))
            print(f"  {name:7}+{gate:9}  blind {r['blind_top1']}/64 ({r['blind_macro']}%)  "
                  f"best-case {r['bestcase_top1']}/64 ({r['bestcase_macro']}%)", flush=True)

    # decision-level ensemble: frozen-head ⊕ bbox-head (no-gate), averaged probs
    print("\n  -- decision-level ensembles --")
    for combo in [('frozen', 'no-gate', 'bbox', 'no-gate'), ('frozen', f'gate{args.tau}', 'bbox', 'no-gate'),
                  ('frozen', 'no-gate', 'hybrid', 'no-gate')]:
        a, b = store[(combo[0], combo[1])]['_probs'], store[(combo[2], combo[3])]['_probs']
        t1s, macs = [], []
        for pa, pb in zip(a, b):
            pred = ((np.array(pa) + np.array(pb)) / 2).argmax(1)
            t1s.append(int((pred == ynp4).sum())); macs.append(be.macro_acc(pred, ynp4))
        tag = f'ens({combo[0]}+{combo[1]} ⊕ {combo[2]}+{combo[3]})'
        rows.append((tag, '-', '-', max(t1s), round(max(macs), 1)))
        print(f"  {tag:38}  best-case {max(t1s)}/64 ({round(max(macs),1)}%)  mean {round(np.mean(t1s),1)}")

    print("\n" + "=" * 70)
    print("SWEEP SUMMARY (vs paper: no-bbox 57/64=85.78% | full 62/64=95.81%)")
    print("=" * 70)
    print(f"  {'config':40} {'blind':>10} {'best-case':>14}")
    for tag, b1, bm, bc1, bcm in rows:
        print(f"  {tag:40} {str(b1)+'/'+str(bm):>10} {str(bc1)+'/64 '+str(bcm)+'%':>14}")
    best = max(rows, key=lambda r: r[3] if isinstance(r[3], int) else 0)
    print(f"\n  >> BEST best-case: {best[0]} = {best[3]}/64 ({best[4]}% macro)")
    print(f"     ceiling check: does any config beat 56/64 best-case or 84.5% macro?")
    json.dump({'rows': [[r[0], r[1], r[2], r[3], r[4]] for r in rows]},
              open('plans/260613-thesis-defense-revision/reports/feature-sweep-result.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
