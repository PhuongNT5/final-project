"""Data loading + CV helpers for the binary Safe/Unsafe baseline GATES (S1 detector-rule, S2
scene-appearance). Kept separate from run_binary_baseline_gates.py to stay under the 200-line cap
(docs/code-standards.md). No dependency on model_relation_head.py / run_relation_transformer.py /
run_kfold_relation.py — those are owned by the concurrent encoder-implementation agent (phase-06);
S1/S2 are plain sklearn baselines and need none of that machinery.

Binary convention (fixed repo-wide, design-2026-09-16-binary-encoder-redesign.md §3.1):
  class_names=['Safe','Unsafe'], index 1 = Unsafe = positive. Source folder classes 0-3 = unsafe
  sub-classes, 4-7 = safe sub-classes.
"""
from __future__ import annotations
import json
import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from unsafenet import dedupe

ROOT = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(ROOT, 'sweep_results', 'relation_tokens')
SCENE_MAP_691 = os.path.join(ROOT, 'sweep_results', 'zone_prior', 'clip_scene_map_691.json')
OOF_DIR = os.path.join(ROOT, 'sweep_results', 'binary_gates')
NAMES = ['Safe', 'Unsafe']
SUB_NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload',
             'SafeWalkway', 'AuthorInt', 'ClosedPanel', 'SafeCarrying']


def to_binary(sub_label):
    """source class 0-3 (unsafe) -> 1 (Unsafe); 4-7 (safe) -> 0 (Safe)."""
    return (np.asarray(sub_label) < 4).astype(int)


def load_bin8(split):
    d = torch.load(os.path.join(TOK, f'tokens_bin8_{split}.pt'), weights_only=False)
    return d['tokens'].numpy(), d['mask'].numpy(), d['labels'].numpy(), np.array(d['clip_ids'])


def load_wf(split, tag='bin8_'):
    d = torch.load(os.path.join(TOK, f'tokens_wf_{tag}{split}.pt'), weights_only=False)
    return d['feats'].numpy(), d['labels'].numpy(), np.array(d['clip_ids'])


def s1_descriptor(tokens, mask):
    """(N,T,8,74) + (N,T,8) -> (N,8) max detector confidence per entity class (0..7).
    feat_layout: [class, xc, yc, w, h, conf, ...] -> class=idx0, conf=idx5."""
    cls = tokens[..., 0]
    conf = tokens[..., 5]
    m = mask.astype(bool)
    n = tokens.shape[0]
    out = np.zeros((n, 8), dtype=np.float32)
    for k in range(8):
        sel = m & (cls == k)
        vals = np.where(sel, conf, -1.0)
        mx = vals.max(axis=(1, 2))
        out[:, k] = np.where(mx < 0, 0.0, mx)
    return out


def load_pool(dedupe_policy):
    """Pool train+test (691 clips, matching the existing 4-class k-fold/LCO pooling precedent —
    run_kfold_relation.py / run_lco_eval.py both pool 516; phase-06 logs this as 0 test touches).
    Returns dict with s1 descriptor, s2 mean-pooled wf feature, sub_label, binary y, clip_ids, group
    (scene-map cluster id, str), aligned and filtered by the given binary dedupe policy."""
    tok_tr, mask_tr, lab_tr, cid_tr = load_bin8('train')
    tok_te, mask_te, lab_te, cid_te = load_bin8('test')
    wf_tr, wflab_tr, wfcid_tr = load_wf('train')
    wf_te, wflab_te, wfcid_te = load_wf('test')
    assert np.array_equal(cid_tr, wfcid_tr) and np.array_equal(cid_te, wfcid_te), \
        'tokens_bin8_* and tokens_wf_bin8_* clip order mismatch — rebuild both from the same glob'
    assert np.array_equal(lab_tr, wflab_tr) and np.array_equal(lab_te, wflab_te)

    s1_tr, s1_te = s1_descriptor(tok_tr, mask_tr), s1_descriptor(tok_te, mask_te)
    s2_tr, s2_te = wf_tr.mean(axis=1), wf_te.mean(axis=1)

    keep_tr = dedupe.keep_mask(cid_tr, 'train', dedupe_policy, task='binary')
    keep_te = dedupe.keep_mask(cid_te, 'test', 'keep', task='binary')

    s1 = np.concatenate([s1_tr[keep_tr], s1_te[keep_te]])
    s2 = np.concatenate([s2_tr[keep_tr], s2_te[keep_te]])
    sub = np.concatenate([lab_tr[keep_tr], lab_te[keep_te]])
    cids = np.concatenate([cid_tr[keep_tr], cid_te[keep_te]])
    dropped = int((~keep_tr).sum() + (~keep_te).sum())

    sm = json.load(open(SCENE_MAP_691))
    group = np.array([str(sm.get(c, 'na')) for c in cids])
    return {'s1': s1, 's2': s2, 'sub': sub, 'y': to_binary(sub), 'clip_ids': cids,
            'group': group, 'dropped': dropped, 'n': len(sub)}


def _fit_predict_proba(model_name, Xtr, ytr, Xte):
    if model_name == 's1_logreg':
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, random_state=0).fit(sc.transform(Xtr), ytr)
        return clf.predict_proba(sc.transform(Xte))
    if model_name == 's1_tree':
        clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xtr, ytr)
        return clf.predict_proba(Xte)
    if model_name == 's2_logreg':
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=0).fit(
            sc.transform(Xtr), ytr)
        return clf.predict_proba(sc.transform(Xte))
    raise ValueError(model_name)


def kfold_oof(X, y, model_name, k=5, repeats=5):
    """Pooled stratified k-fold OOF probabilities (repeat 0 = canonical OOF set, matching
    run_kfold_relation.py) + per-fold accuracy across all repeats (for NB-corrected significance).
    fold_id (repeat-0 fold index per row) is returned so the OOF dump can record which fold held
    out each clip in the canonical set."""
    n = len(y)
    oof_prob = np.zeros((n, 2))
    fold_id = np.full(n, -1, dtype=int)
    fold_acc = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=r)
        for j, (tr, te) in enumerate(skf.split(np.zeros(n), y)):
            prob = _fit_predict_proba(model_name, X[tr], y[tr], X[te])
            if r == 0:
                oof_prob[te] = prob
                fold_id[te] = j
            fold_acc.append(float((prob.argmax(1) == y[te]).mean()) * 100)
    n_tr_mean = int(n * (k - 1) / k)
    n_te_mean = int(n / k)
    return oof_prob, fold_acc, n_tr_mean, n_te_mean, fold_id


def save_oof(model_name, protocol, policy_tag, clip_ids, y_true, prob, pred, fold_id):
    """Dump per-clip OOF predictions so downstream own-vs-own McNemar (e.g. encoder vs S1/S2) can
    be run on the exact same clip set without re-fitting these baselines. One JSON per
    model x protocol x dedupe-policy: sweep_results/binary_gates/{model}_{protocol}_{policy}_oof.json."""
    os.makedirs(OOF_DIR, exist_ok=True)
    path = os.path.join(OOF_DIR, f'{model_name}_{protocol}_{policy_tag}_oof.json')
    rows = [{'clip_id': str(c), 'y_true': int(yt), 'prob_unsafe': float(p1), 'y_pred': int(yp),
             'fold_id': (int(f) if isinstance(f, (int, np.integer)) else str(f))}
            for c, yt, p1, yp, f in zip(clip_ids, y_true, prob[:, 1], pred, fold_id)]
    with open(path, 'w') as fh:
        json.dump({'model': model_name, 'protocol': protocol, 'dedupe_policy': policy_tag,
                   'n': len(rows), 'rows': rows}, fh, indent=1)
    return path


def lco_oof(X, y, group, model_name):
    """Leave-one-scene-cluster-out, pooled OOF probabilities across all groups (never per-fold
    accuracy — clip_scene_map_691.json groups are severely class-skewed, phase-06 §Key Insights)."""
    n = len(y)
    oof_prob = np.zeros((n, 2))
    uniq = sorted(set(group.tolist()))
    for g in uniq:
        te = np.where(group == g)[0]
        tr = np.where(group != g)[0]
        if len(te) == 0 or len(set(y[tr].tolist())) < 2:
            oof_prob[te] = np.bincount(y[tr], minlength=2) / max(1, len(tr))
            continue
        oof_prob[te] = _fit_predict_proba(model_name, X[tr], y[tr], X[te])
    return oof_prob, uniq
