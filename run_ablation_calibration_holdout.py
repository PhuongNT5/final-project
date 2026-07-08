#!/usr/bin/env python3
"""Honest 57/64 retry: per-class bias calibration on a TRAIN holdout.

The previously reported 57/64 used a 20,000-step random bias search fit
directly on the test set (audited as test-set tuning in Ch4 §4.7.7).
This script re-fits the same bias search on a HELD-OUT slice of the
TRAINING set and applies the resulting bias to the test set untouched.

Caveat: the heads themselves were trained with model selection on the
test set (early stopping picks the epoch with best test accuracy). The
bias-search added on top is what this script audits, not the head
training. Calibration on train is therefore "more honest than test-
tuning" but not zero-leakage.

Outputs: sweep_results/ablations/calibration_holdout.json
"""
import os, sys, json, importlib.util
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
CKPT = os.path.join(ROOT, 'sweep_results', 'pairwise')
FEAT_TR = os.path.join(CKPT, 'features', 'all8_train.pt')
FEAT_TE = os.path.join(CKPT, 'features', 'all8_test.pt')
OUT = os.path.join(ROOT, 'sweep_results', 'ablations')

PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]
PAIR_NAMES = ['Walkway', 'Intervention', 'Panel', 'Forklift']
N_SEARCH = 20_000


def load_TH():
    spec = importlib.util.spec_from_file_location('pw', os.path.join(ROOT, 'run_pairwise_contrastive.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.TransformerHead


def stats(f):
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
    """Compute (direct4 + cascade) / 2 ensemble probabilities."""
    Ld = head_logits(load_head(os.path.join(CKPT, 'direct4_s314.pt'), 4, 384, 8, 6, sd, dev, TH), s4, dev)
    Lsc = head_logits(load_head(os.path.join(CKPT, 'scenario_s7.pt'), 4, 384, 8, 6, sd, dev, TH), sall, dev)[mask4]
    pair_log = []
    for i, n in enumerate(PAIR_NAMES):
        pair_log.append(head_logits(load_head(os.path.join(CKPT, f'pair{i}_{n}_s7.pt'), 2, 192, 6, 3, sd, dev, TH), s4, dev))
    pair_log = np.stack(pair_log, 1)  # (N, 4, 2)

    pd = F.softmax(torch.from_numpy(Ld), 1).numpy()
    psc = F.softmax(torch.from_numpy(Lsc), 1).numpy()
    pp_unsafe = F.softmax(torch.from_numpy(pair_log), 2).numpy()[:, :, 1]
    cas = psc * pp_unsafe; cas = cas / (cas.sum(1, keepdims=True) + 1e-8)
    return (pd + cas) / 2


def stratified_holdout(labels_np, frac, seed):
    """Stratified split: returns (train_idx, holdout_idx) where holdout
    contains frac of each class."""
    rng = np.random.default_rng(seed)
    holdout = []; tr = []
    for c in range(int(labels_np.max()) + 1):
        idx = np.where(labels_np == c)[0]
        rng.shuffle(idx)
        n_hold = max(1, int(round(len(idx) * frac)))
        holdout.append(idx[:n_hold]); tr.append(idx[n_hold:])
    return np.concatenate(tr), np.concatenate(holdout)


def bias_search(probs, y, n_steps, seed):
    """Random bias search over R^4. Returns (best_bias, best_correct)."""
    rng = np.random.default_rng(seed)
    best_b = np.zeros(4); best_c = int((probs.argmax(1) == y).sum())
    for _ in range(n_steps):
        b = rng.uniform(-0.5, 0.5, 4)
        c = int(((probs + b).argmax(1) == y).sum())
        if c > best_c: best_c, best_b = c, b.copy()
    return best_b, best_c


def macro(preds, y, n_cls=4):
    return float(np.mean([100 * int((preds[y == c] == c).sum()) / max(1, int((y == c).sum())) for c in range(n_cls)]))


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")
    TH = load_TH()

    print("\n[1] Loading features and computing ensemble probs ...")
    dtr = torch.load(FEAT_TR, map_location='cpu', weights_only=False)
    dte = torch.load(FEAT_TE, map_location='cpu', weights_only=False)

    mask_tr4 = (dtr['labels'] < 4).numpy()
    mask_te4 = (dte['labels'] < 4).numpy()
    y_tr4 = dtr['labels'][mask_tr4].numpy()
    y_te4 = dte['labels'][mask_te4].numpy()

    sall_tr = stats(dtr['features']); s4_tr = sall_tr[mask_tr4]
    sall_te = stats(dte['features']); s4_te = sall_te[mask_te4]
    sd = s4_tr.shape[-1]

    P_tr = ensemble_probs(s4_tr, sall_tr, mask_tr4, dev, TH, sd)
    P_te = ensemble_probs(s4_te, sall_te, mask_te4, dev, TH, sd)
    print(f"  train ensemble: {P_tr.shape}, test ensemble: {P_te.shape}")
    print(f"  train labels: {np.bincount(y_tr4)}  test labels: {np.bincount(y_te4)}")

    base_tr = int((P_tr.argmax(1) == y_tr4).sum()); base_te = int((P_te.argmax(1) == y_te4).sum())
    base_tr_macro = macro(P_tr.argmax(1), y_tr4); base_te_macro = macro(P_te.argmax(1), y_te4)
    print(f"\n[BASELINE — no calibration]")
    print(f"  train: {base_tr}/{len(y_tr4)} ({100*base_tr/len(y_tr4):.1f} %)  macro={base_tr_macro:.2f} %")
    print(f"  test : {base_te}/{len(y_te4)} ({100*base_te/len(y_te4):.1f} %)  macro={base_te_macro:.2f} %")

    print(f"\n[REFERENCE — bias search fit on TEST (this is the audited 57/64)]")
    b_te, c_te = bias_search(P_te, y_te4, N_SEARCH, seed=0)
    pred_te = (P_te + b_te).argmax(1)
    print(f"  test-fit bias: {np.round(b_te, 3).tolist()}")
    print(f"  test-fit accuracy: {c_te}/{len(y_te4)} macro={macro(pred_te, y_te4):.2f} %")

    print(f"\n[A — bias search fit on FULL TRAIN, applied to TEST]")
    b_tr, c_tr_fit = bias_search(P_tr, y_tr4, N_SEARCH, seed=0)
    pred_tr_full = (P_tr + b_tr).argmax(1)
    pred_te_full = (P_te + b_tr).argmax(1)
    test_with_train_full = int((pred_te_full == y_te4).sum())
    print(f"  bias from full train: {np.round(b_tr, 3).tolist()}")
    print(f"  train fit accuracy: {c_tr_fit}/{len(y_tr4)} macro={macro(pred_tr_full, y_tr4):.2f} %")
    print(f"  TEST with this bias: {test_with_train_full}/{len(y_te4)} macro={macro(pred_te_full, y_te4):.2f} %")

    print(f"\n[B — 5-fold stratified train holdout (frac=0.2 per fold)]")
    fold_results = []
    for k in range(5):
        _, hold_idx = stratified_holdout(y_tr4, 0.2, seed=1000 + k)
        P_hold = P_tr[hold_idx]; y_hold = y_tr4[hold_idx]
        b_h, c_h_fit = bias_search(P_hold, y_hold, N_SEARCH, seed=1000 + k)
        pred_te_h = (P_te + b_h).argmax(1)
        c_te_h = int((pred_te_h == y_te4).sum())
        m_te_h = macro(pred_te_h, y_te4)
        fold_results.append({
            'fold': k,
            'holdout_size': int(len(hold_idx)),
            'holdout_fit_correct': int(c_h_fit),
            'holdout_fit_acc': round(100 * c_h_fit / len(y_hold), 2),
            'test_correct': int(c_te_h),
            'test_acc': round(100 * c_te_h / len(y_te4), 2),
            'test_macro': round(m_te_h, 2),
            'bias': [round(float(x), 4) for x in b_h.tolist()],
        })
        print(f"  fold {k}: holdout={len(hold_idx)} fit={c_h_fit}/{len(hold_idx)}  "
              f"→ TEST={c_te_h}/{len(y_te4)} ({100*c_te_h/len(y_te4):.1f} %) macro={m_te_h:.2f} %")

    test_corrects = [r['test_correct'] for r in fold_results]
    print(f"\n  5-fold TEST mean: {np.mean(test_corrects):.2f}  std: {np.std(test_corrects):.2f}")
    print(f"  5-fold TEST range: [{min(test_corrects)}, {max(test_corrects)}]")

    print(f"\n[C — repeated random search on FULL TRAIN, 5 different seeds]")
    seed_results = []
    for s in range(5):
        b_s, c_s_fit = bias_search(P_tr, y_tr4, N_SEARCH, seed=2000 + s)
        pred_te_s = (P_te + b_s).argmax(1)
        c_te_s = int((pred_te_s == y_te4).sum())
        m_te_s = macro(pred_te_s, y_te4)
        seed_results.append({
            'search_seed': 2000 + s,
            'train_fit_correct': int(c_s_fit),
            'test_correct': int(c_te_s),
            'test_acc': round(100 * c_te_s / len(y_te4), 2),
            'test_macro': round(m_te_s, 2),
            'bias': [round(float(x), 4) for x in b_s.tolist()],
        })
        print(f"  seed {2000+s}: train fit={c_s_fit}/{len(y_tr4)}  "
              f"→ TEST={c_te_s}/{len(y_te4)} macro={m_te_s:.2f} %")

    seed_corrects = [r['test_correct'] for r in seed_results]
    print(f"\n  5-seed TEST mean: {np.mean(seed_corrects):.2f}  std: {np.std(seed_corrects):.2f}")

    out = {
        'baseline': {
            'train': {'correct': base_tr, 'total': int(len(y_tr4)), 'macro': round(base_tr_macro, 2)},
            'test':  {'correct': base_te, 'total': int(len(y_te4)), 'macro': round(base_te_macro, 2)},
        },
        'test_set_tuning_reference': {
            'description': 'Bias search fit directly on test set (Ch4 §4.7.7 audit target)',
            'test_correct': int(c_te), 'test_total': int(len(y_te4)),
            'test_macro': round(macro(pred_te, y_te4), 2),
            'bias': [round(float(x), 4) for x in b_te.tolist()],
        },
        'full_train_calibration': {
            'description': 'Bias search fit on full 4-class train (452 clips), applied to test',
            'train_fit_correct': int(c_tr_fit),
            'train_total': int(len(y_tr4)),
            'test_correct': int(test_with_train_full),
            'test_total': int(len(y_te4)),
            'test_acc': round(100 * test_with_train_full / len(y_te4), 2),
            'test_macro': round(macro(pred_te_full, y_te4), 2),
            'bias': [round(float(x), 4) for x in b_tr.tolist()],
        },
        'kfold_holdout_calibration': {
            'description': '5-fold stratified train holdout (frac=0.2 per fold); bias fit on holdout, applied to test',
            'folds': fold_results,
            'mean_test_correct': float(np.mean(test_corrects)),
            'std_test_correct': float(np.std(test_corrects)),
            'min_test_correct': int(min(test_corrects)),
            'max_test_correct': int(max(test_corrects)),
        },
        'multi_seed_full_train': {
            'description': '5 random-search seeds, fit on full train, applied to test',
            'seeds': seed_results,
            'mean_test_correct': float(np.mean(seed_corrects)),
            'std_test_correct': float(np.std(seed_corrects)),
        },
    }
    json.dump(out, open(os.path.join(OUT, 'calibration_holdout.json'), 'w'), indent=2)
    print(f"\n[SAVED] {os.path.join(OUT, 'calibration_holdout.json')}")

    # Verdict
    print(f"\n{'═'*70}")
    print("VERDICT")
    print(f"{'═'*70}")
    print(f"  Baseline (no cal):              56/64 = 87.5 %  (expected)")
    print(f"  Test-tuned (audit target):      {c_te}/64 = {100*c_te/64:.1f} %")
    print(f"  Full-train calibration:         {test_with_train_full}/64 = {100*test_with_train_full/64:.1f} %")
    print(f"  5-fold holdout (mean):          {np.mean(test_corrects):.1f}/64 = {100*np.mean(test_corrects)/64:.2f} %")
    print(f"  5-seed full-train (mean):       {np.mean(seed_corrects):.1f}/64 = {100*np.mean(seed_corrects)/64:.2f} %")


if __name__ == '__main__':
    main()
