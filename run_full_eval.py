#!/usr/bin/env python3
"""Full multi-lens evaluation of the best model (S1a appearance+aux ensemble) on the 64-clip test.

Covers the thesis eval spec: accuracy battery (top-1, macro=mean-recall, macro-F1, balanced-acc,
per-class P/R, Cohen's κ, MCC, confusion), safety (per-class FNR, cost-sensitive), operating point
(one-vs-rest ROC-AUC/PR-AUC), calibration (ECE + reliability), statistics (BCa/Wilson CIs, McNemar
exact vs scene-only baseline). Writes JSON + confusion/reliability PNGs. Timing measured separately.

Usage: .venv/bin/python run_full_eval.py
"""
import json
import os
import numpy as np
import torch
from sklearn.metrics import (f1_score, balanced_accuracy_score, cohen_kappa_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, average_precision_score)
from eval_utils.metrics import top1, macro_accuracy, per_class_recall, per_class_fnr, confusion
from eval_utils.bootstrap_ci import bootstrap_ci
from eval_utils.paired_tests import mcnemar_exact, wilson_ci, two_proportion_z
from eval_utils.calibration import ece, reliability_bins
from eval_utils.safety_metrics import cost_sensitive_score

ROOT = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(ROOT, 'sweep_results', 'relation_tokens')
FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']


def scene_only_pred():
    """Camera→majority-class baseline on the 64 test clips (for a paired McNemar)."""
    from collections import Counter, defaultdict
    m = json.load(open(os.path.join(ROOT, 'sweep_results', 'zone_prior', 'clip_view_map.json')))
    tab = defaultdict(Counter)
    for r in m:
        if r.get('subset') == 'train':
            tab[str(r['view'])][r['label']] += 1
    major = {v: tab[v].most_common(1)[0][0] for v in tab}
    te = [r for r in m if r.get('subset') == 'test']
    # order matches tokens_test (sorted by class then clip) — rebuild the same order
    d = torch.load(os.path.join(TOK, 'tokens_app_test.pt'), weights_only=False)
    view_by_clip = {r['basename']: str(r['view']) for r in m}
    return np.array([major.get(view_by_clip.get(c, 'na'), 0) for c in d['clip_ids']])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # DEFAULT CHANGED 2026-09-06 (number-integrity fix): logits_yolo26.pt is the P3 headline pipeline
    # cited throughout brief.md/QA1 audit (blind_pred = 54/64 = 84.4%, macro 88.4%). logits_s1a.pt was
    # left as the old default from before yolo26 became the headline config — it is a DIFFERENT model
    # (S1a appearance+aux ablation), not cited as a headline number anywhere in brief.md. Pass
    # --logits logits_s1a.pt explicitly to reproduce the old (non-headline) artifact.
    ap.add_argument('--logits', default='logits_yolo26.pt')
    args = ap.parse_args()
    d = torch.load(os.path.join(TOK, args.logits), weights_only=False)
    y = d['labels'].numpy()
    # Prefer the BLIND seed's own probabilities when the artifact carries them. Before 2026-09-06 only
    # the seed-averaged ens_softmax was persisted, which forced every probability metric (ECE/ROC/PR)
    # onto the ensemble predictor while top-1/macro/kappa came from blind_pred - two different models in
    # one table. logits_yolo26.pt now stores blind_softmax (provenance recovered and bit-verified; see
    # reports/test-touch-log.md), so both halves describe the same predictor. Artifacts without it fall
    # back to the ensemble, and pred_basis records which happened.
    if 'blind_softmax' in d:
        P = d['blind_softmax'].numpy(); basis = 'blind_seed_softmax (val-selected single seed)'
    else:
        P = d['ens_softmax'].numpy(); basis = 'ensemble_seed_avg_argmax (softmax averaged over seeds, then argmax)'
    pred = P.argmax(1)
    n = len(y)
    Y = np.eye(4)[y]                                          # one-hot for AUC

    out = {}
    # provenance — avoids the exact ambiguity this fix is closing (a JSON with no record of which
    # .pt/predictor produced it). pred_basis flags that every metric below (incl. ECE) is computed on
    # the seed-AVERAGED ensemble argmax, which is NOT always identical to the file's own 'blind_pred'
    # (the val-selected single-seed headline decision) — see blind_vs_ensemble_agreement below.
    out['source_logits'] = args.logits
    out['pred_basis'] = basis
    if 'provenance' in d:
        out['source_provenance'] = d['provenance']
    if 'blind_pred' in d:
        bp = d['blind_pred'].numpy()
        out['blind_pred_top1_correct'] = f"{int((bp == y).sum())}/{n}"
        out['blind_vs_ensemble_pred_agree_frac'] = round(float((bp == d['ens_softmax'].numpy().argmax(1)).mean()), 4)
        out['blind_vs_ensemble_note'] = (
            "blind_pred is the val-selected single-seed decision (the brief.md headline protocol tag). "
            "When pred_basis is blind_seed_softmax, every metric in this JSON describes that same "
            "predictor and blind_pred_top1_correct will equal top1_correct. When it is the ensemble "
            "fallback, the calibration/AUC numbers characterise the seed-averaged variant instead, and "
            "a difference between those two counts is the size of that mismatch.")
    # --- accuracy battery ---
    out['top1'] = round(100 * top1(y, pred), 2)
    out['top1_correct'] = f"{int((pred==y).sum())}/{n}"
    out['macro_recall'] = round(100 * macro_accuracy(y, pred), 2)
    out['macro_f1'] = round(100 * f1_score(y, pred, average='macro'), 2)
    out['balanced_acc'] = round(100 * balanced_accuracy_score(y, pred), 2)
    out['cohen_kappa'] = round(cohen_kappa_score(y, pred), 3)
    out['mcc'] = round(matthews_corrcoef(y, pred), 3)
    out['per_class'] = {NAMES[c]: {
        'precision': round(100 * precision_score(y, pred, labels=[c], average='macro', zero_division=0), 1),
        'recall': round(100 * recall_score(y, pred, labels=[c], average='macro', zero_division=0), 1),
        'fnr': round(100 * per_class_fnr(y, pred)[c], 1), 'n': int((y == c).sum())} for c in range(4)}
    out['confusion'] = confusion(y, pred, 4).tolist()

    # --- safety ---
    out['cost_sensitive_miss_x10'] = cost_sensitive_score(y, pred, miss_cost=10, false_alarm_cost=1)

    # --- operating point (one-vs-rest) ---
    out['roc_auc_ovr_macro'] = round(roc_auc_score(Y, P, average='macro', multi_class='ovr'), 3)
    out['pr_auc_ovr_macro'] = round(float(np.mean([average_precision_score(Y[:, c], P[:, c]) for c in range(4)])), 3)

    # --- calibration ---
    out['ece_15bin'] = round(ece(P, y, 15), 4)

    # --- statistics ---
    _, t_lo, t_hi, _ = bootstrap_ci(y, pred, lambda a, b: top1(a, b) * 100, n_boot=5000, seed=0)
    _, m_lo, m_hi, _ = bootstrap_ci(y, pred, lambda a, b: macro_accuracy(a, b) * 100, n_boot=5000, seed=0)
    out['top1_bca_ci95'] = [round(t_lo, 1), round(t_hi, 1)]
    out['macro_bca_ci95'] = [round(m_lo, 1), round(m_hi, 1)]
    out['top1_wilson_ci95'] = [round(100 * x, 1) for x in wilson_ci(int((pred == y).sum()), n)]
    sp = scene_only_pred()
    out['mcnemar_vs_scene_only'] = mcnemar_exact(pred == y, sp == y)
    out['vs_paper_no_bbox_top1'] = two_proportion_z(int((pred == y).sum()), n, 57, n)
    out['vs_paper_full_bbox_top1'] = two_proportion_z(int((pred == y).sum()), n, 62, n)

    # canonical headline output keeps the plain 'full_eval.json' name; any non-default --logits writes
    # to a suffixed file instead, so re-running against a different artifact never silently overwrites
    # the headline file with an unlabeled, ambiguous JSON (the exact bug this fix is closing).
    out_name = 'full_eval.json' if args.logits == 'logits_yolo26.pt' else f"full_eval_{args.logits.replace('logits_', '').replace('.pt', '')}.json"
    out_path = os.path.join(ROOT, 'sweep_results', 'relation_tokens', out_name)
    json.dump(out, open(out_path, 'w'), indent=2)

    # --- figures ---
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        cm = np.array(out['confusion'])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(NAMES, rotation=45, ha='right', fontsize=7); ax.set_yticklabels(NAMES, fontsize=7)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
        ax.set_xlabel('pred'); ax.set_ylabel('true'); ax.set_title('S1a ensemble — confusion (64-clip)')
        fig.tight_layout(); fig.savefig(os.path.join(FIG, 'relation_confusion.png'), dpi=150); plt.close(fig)
        rb = reliability_bins(P, y, 15)
        xs = [b['confidence'] for b in rb if b['count']]; ys = [b['accuracy'] for b in rb if b['count']]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], '--', color='gray'); ax.plot(xs, ys, 'o-')
        ax.set_xlabel('confidence'); ax.set_ylabel('accuracy'); ax.set_title(f"Reliability (ECE={out['ece_15bin']})")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, 'relation_reliability.png'), dpi=150); plt.close(fig)
        out['figures'] = ['figures/relation_confusion.png', 'figures/relation_reliability.png']
    except Exception as e:
        print("(figure gen skipped:", str(e)[:60], ")")

    # --- print ---
    print("=== FULL EVAL — S1a appearance+aux ensemble (64-clip test) ===")
    for k in ['top1_correct', 'top1', 'macro_recall', 'macro_f1', 'balanced_acc', 'cohen_kappa', 'mcc',
              'roc_auc_ovr_macro', 'pr_auc_ovr_macro', 'ece_15bin', 'cost_sensitive_miss_x10',
              'top1_bca_ci95', 'macro_bca_ci95']:
        print(f"  {k:22s}: {out[k]}")
    print("  per-class (P / R / FNR):")
    for c in NAMES:
        pc = out['per_class'][c]; print(f"    {c:14s} P={pc['precision']:5.1f} R={pc['recall']:5.1f} FNR={pc['fnr']:5.1f} (n={pc['n']})")
    print(f"  McNemar vs scene-only: p={out['mcnemar_vs_scene_only']['p_value']:.3f} "
          f"(discordant {out['mcnemar_vs_scene_only']['n_discordant']}, {out['mcnemar_vs_scene_only']['note'][:40]})")
    print(f"  vs paper no-bbox (57/64): Δ={out['vs_paper_no_bbox_top1']['delta_pp']:+.1f}pp p={out['vs_paper_no_bbox_top1']['p_value']:.3f}")
    print(f"  saved: sweep_results/relation_tokens/{out_name} + figures/")


if __name__ == '__main__':
    main()
