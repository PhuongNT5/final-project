"""Loading + grouped-CV plumbing for the iSafetyBench relation tokens.

Kept separate from run_kfold_relation.py's Unsafe-Net loaders because the two corpora differ in three
structural ways, none of which belong in the same function:
  * one token file (`tokens_isb_all.pt`) instead of a train/test pair — there is no frozen blind split
    on this corpus and none is created here;
  * no 8-way sub-class taxonomy — labels are natively binary (hazard=1 / normal=0);
  * no byte-identical-twin dedupe audit (that audit is Unsafe-Net-specific).

Grouping is the source YouTube video id, recovered from the filename at token-build time and carried
in the 'views' field. 0/150 sources contain both classes (R41), so grouped CV is mandatory and random
splits are meaningless on this corpus.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

TOK = 'sweep_results/relation_tokens'
PERSON_GROUP = 0          # COCO->coarse group id for `person` (build_isafetybench_tokens.COCO_GROUPS)


def load(tag='isb_'):
    """-> X (n,T,N,74) float, M (n,T,N) float, y (n,) long 1=hazard, groups (n,) str, clip_ids (n,) str."""
    d = torch.load(f'{TOK}/tokens_{tag}all.pt', weights_only=False)
    X, M = d['tokens'].float(), d['mask'].float()
    y = d['labels'].long()
    groups = np.array([str(v) for v in d['views']])
    clip_ids = np.array([str(c) for c in d['clip_ids']])
    return X, M, y, groups, clip_ids


def feat_layout(tag='isb_'):
    """The token file's own column names, so feature_ablation never hardcodes an index."""
    d = torch.load(f'{TOK}/tokens_{tag}all.pt', weights_only=False)
    return d.get('feat_layout')


def folds(y, groups, n_splits=10, seed=0):
    """Leave-source-out CV as grouped folds. StratifiedGroupKFold (not plain GroupKFold) because the
    top 5 sources hold ~24% of clips and every source is single-class, so unstratified grouped folds
    swing the class ratio badly (feasibility report unresolved Q3)."""
    yn = y.numpy() if torch.is_tensor(y) else np.asarray(y)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(yn)), yn, groups))


def placebo_groups(y, groups, seed):
    """Fake source ids that preserve every structural property of the real ones EXCEPT provenance.

    The G19 control for the confound the inflation measurement cannot otherwise rule out: the
    leave-source-out number could be lower simply because grouped folds are structurally harder
    (whole indivisible single-class sources, uneven fold sizes), not because source identity leaks.
    This builds groups that match the real ones on all three of those properties --
      * the same number of groups,
      * the same multiset of group sizes, within each class,
      * the same class purity (every group single-class, as all 151 real sources are, R41) --
    while scattering each real video's clips across different fake groups. Source identity is the
    only thing removed, so whatever accuracy survives here is NOT attributable to fold geometry.
    """
    yn = y.numpy() if torch.is_tensor(y) else np.asarray(y)
    rng = np.random.default_rng(seed)
    fake = np.empty(len(yn), dtype=object)
    for c in np.unique(yn):
        idx = np.where(yn == c)[0]
        # group-size multiset of the REAL sources of this class, preserved exactly
        sizes = [int((groups[idx] == g).sum()) for g in dict.fromkeys(groups[idx])]
        order = rng.permutation(idx)
        at = 0
        for j, n in enumerate(sizes):
            fake[order[at:at + n]] = f'plc{c}_{j}'
            at += n
        assert at == len(idx), f'size multiset did not tile class {c}: {at} vs {len(idx)}'
    return np.array([str(v) for v in fake])


def count_features(X, M):
    """The 3 features of the pre-registered R42 count-rule shortcut: persons/frame, objects/frame,
    mean detection confidence. Recomputed here from the same T=16 tokens the encoder consumes, so the
    shortcut and the encoder are scored on identical inputs and identical folds."""
    m = M.numpy() if torch.is_tensor(M) else M
    cls = (X[..., 0].numpy() if torch.is_tensor(X) else X[..., 0])
    conf = (X[..., 5].numpy() if torch.is_tensor(X) else X[..., 5])
    n_frames = max(m.shape[1], 1)
    objs = m.sum(axis=(1, 2)) / n_frames
    persons = ((cls == PERSON_GROUP) * m).sum(axis=(1, 2)) / n_frames
    tot = m.sum(axis=(1, 2))
    mean_conf = np.divide((conf * m).sum(axis=(1, 2)), tot, out=np.zeros_like(objs), where=tot > 0)
    return np.stack([persons, objs, mean_conf], axis=1)


def count_rule_oof(X, M, y, splits):
    """Pooled out-of-fold predictions of the R42 count-rule (class-weighted logistic regression).
    -> (pred (n,), prob_unsafe (n,)). Standardisation is fit inside each fold — no leakage."""
    F = count_features(X, M)
    yn = y.numpy() if torch.is_tensor(y) else np.asarray(y)
    pred = np.full(len(yn), -1)
    prob = np.zeros(len(yn), dtype=np.float32)
    for tr, te in splits:
        sc = StandardScaler().fit(F[tr])
        lr = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(F[tr]), yn[tr])
        p = lr.predict_proba(sc.transform(F[te]))[:, 1]
        prob[te] = p
        pred[te] = (p >= 0.5).astype(int)
    return pred, prob
