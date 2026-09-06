"""Quantify the camera/scene confound in the Unsafe-Net clip set.

The dataset assigns each clip a fixed-CCTV view id. If view predicts label, then
same-distribution accuracy partly measures camera memorisation rather than behaviour.
This module builds the *no-pixel* floor - a classifier whose only input is the camera id -
and scores every protocol against it. A method that cannot beat this floor has shown no
behaviour signal, whatever its top-1.

Usage: .venv/bin/python -m unsafenet.scene_confound [--logits logits_yolo26.pt]
"""
import argparse, collections, json, os
import numpy as np
import torch
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
from unsafenet import dedupe
from unsafenet import metrics_stats as MS

TOK = 'sweep_results/relation_tokens'
CLS = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']


def load(tag='', dedupe_policy='none'):
    """Test half is never filtered here: it is the paper's frozen 64-clip set."""
    a = torch.load(f'{TOK}/tokens_{tag}train.pt', weights_only=False)
    b = torch.load(f'{TOK}/tokens_{tag}test.pt', weights_only=False)
    f = lambda d, k: np.array([str(x) for x in d[k]]) if k == 'views' else np.array(list(d[k]))
    va, ya = f(a, 'views'), f(a, 'labels')
    if dedupe_policy != 'none':
        keep = dedupe.keep_mask(a['clip_ids'], 'train', dedupe_policy)
        print(f"[dedupe:{dedupe_policy}] dropped {int((~keep).sum())} of {len(keep)} train clips")
        va, ya = va[keep], ya[keep]
    return (va, ya, f(b, 'views'), f(b, 'labels'))


def scene_prior(vtr, ytr):
    """Majority label per camera, fit on TRAIN only. Sees no pixels."""
    return {v: collections.Counter(ytr[vtr == v]).most_common(1)[0][0] for v in set(vtr)}


def confound_stats(v, y):
    """Dependence between camera id and label over the pooled clip set."""
    uv = sorted(set(v), key=int)
    ct = np.array([[int(((v == a) & (y == c)).sum()) for c in range(4)] for a in uv])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = len(y)
    ent = lambda counts: -sum((c / n) * np.log2(c / n) for c in counts if c)
    hy = ent(collections.Counter(y.tolist()).values())
    mi = mutual_info_score(v, y) / np.log(2)
    return {'chi2': chi2, 'dof': dof, 'p': p,
            'cramers_v': float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))),
            'H_label_bits': hy, 'I_view_label_bits': mi, 'frac_entropy_explained': mi / hy,
            'views_single_class': [uv[i] for i in range(len(uv)) if (ct[i] > 0).sum() == 1]}


def report(logits='logits_yolo26.pt', dedupe_policy='none'):
    vtr, ytr, vte, yte = load(dedupe_policy=dedupe_policy)
    v, y = np.concatenate([vtr, vte]), np.concatenate([ytr, yte])
    out = {'n_pooled': len(y), 'n_test': len(yte), 'dedupe': dedupe_policy,
           'confound': confound_stats(v, y)}

    maj = scene_prior(vtr, ytr)
    sp_te = np.array([maj[x] for x in vte])
    # pooled floor uses a prior fit on all pooled clips (optimistic ceiling for the floor)
    maj_all = scene_prior(v, y)
    sp_all = np.array([maj_all[x] for x in v])
    rec = [float((sp_all[y == c] == c).mean()) for c in range(4)]
    out['floor_pooled'] = {'top1': float((sp_all == y).mean()), 'macro': float(np.mean(rec)),
                           'per_class_recall': dict(zip(CLS, rec))}
    out['floor_test'] = {'correct': int((sp_te == yte).sum()), 'n': len(yte),
                         'top1': float((sp_te == yte).mean()),
                         'ci95': [float(x) for x in MS.clopper_pearson(int((sp_te == yte).sum()), len(yte))]}

    path = f'{TOK}/{logits}'
    if os.path.exists(path):
        pred = torch.load(path, weights_only=False)['blind_pred'].numpy()
        out['model_test'] = {'correct': int((pred == yte).sum()), 'n': len(yte),
                             'top1': float((pred == yte).mean()),
                             'ci95': [float(x) for x in MS.clopper_pearson(int((pred == yte).sum()), len(yte))]}
        out['model_vs_floor_mcnemar'] = MS.mcnemar_exact(yte, pred, sp_te)
        hard = sp_te != yte
        out['conditional'] = {
            'floor_wrong_n': int(hard.sum()), 'model_right_when_floor_wrong': int((pred[hard] == yte[hard]).sum()),
            'floor_right_n': int((~hard).sum()), 'model_right_when_floor_right': int((pred[~hard] == yte[~hard]).sum())}
        out['per_view_test'] = [
            {'view': vv, 'n': int((vte == vv).sum()), 'n_classes': len(set(yte[vte == vv].tolist())),
             'floor': float((sp_te[vte == vv] == yte[vte == vv]).mean()),
             'model': float((pred[vte == vv] == yte[vte == vv]).mean())}
            for vv in sorted(set(vte), key=int)]
    return out


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--logits', default='logits_yolo26.pt')
    a.add_argument('--out', default='sweep_results/scene_confound.json')
    a.add_argument('--dedupe', default='none', choices=['none', 'drop_all', 'keep_one'])
    v = a.parse_args()
    r = report(v.logits, v.dedupe)
    os.makedirs(os.path.dirname(v.out), exist_ok=True)
    json.dump(r, open(v.out, 'w'), indent=2)
    c = r['confound']
    print(f"pooled {r['n_pooled']} clips | Cramer's V={c['cramers_v']:.3f} "
          f"| I(view;label)={c['I_view_label_bits']:.3f}/{c['H_label_bits']:.3f} bits "
          f"({100*c['frac_entropy_explained']:.1f}% of label entropy)")
    print(f"single-class cameras: {c['views_single_class']}")
    print(f"NO-PIXEL FLOOR pooled: top1={100*r['floor_pooled']['top1']:.1f}% macro={100*r['floor_pooled']['macro']:.1f}%")
    print(f"NO-PIXEL FLOOR test  : {r['floor_test']['correct']}/{r['floor_test']['n']} = {100*r['floor_test']['top1']:.1f}%")
    if 'model_test' in r:
        m = r['model_test']; print(f"MODEL test           : {m['correct']}/{m['n']} = {100*m['top1']:.1f}%  "
                                   f"McNemar vs floor p={r['model_vs_floor_mcnemar']['p']:.3f}")
    print(f"-> written {v.out}")
