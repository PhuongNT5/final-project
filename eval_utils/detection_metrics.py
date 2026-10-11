"""Detection-framed metrics for the binary Safe/Unsafe task: positive class = unsafe.

Balanced accuracy prices a missed unsafe clip exactly like a false alarm on a safe clip. For an
alarm system those two costs differ, so these helpers report the asymmetric quantities instead -
recall, false-alarm rate, precision, F1 - plus two diagnostics that separate genuine discrimination
from context guessing (`within_scene_auc`, `scene_alarm_profile`) and one that measures how much of
the false-alarm rate the dataset's own labels force (`duplicate_floor`).

Thresholds are never chosen here from the data being scored; `pick_thresholds` takes the validation
scores only, and the caller applies the result to test.
"""
import json

import numpy as np

POS = 1  # unsafe


def counts(y, dec):
    """Detection confusion with unsafe as the positive class."""
    y = np.asarray(y); dec = np.asarray(dec)
    tp = int(((y == POS) & (dec == POS)).sum()); fn = int(((y == POS) & (dec != POS)).sum())
    fp = int(((y != POS) & (dec == POS)).sum()); tn = int(((y != POS) & (dec != POS)).sum())
    rec = tp / (tp + fn) if tp + fn else float('nan')
    fpr = fp / (fp + tn) if fp + tn else float('nan')
    prec = tp / (tp + fp) if tp + fp else float('nan')
    if np.isnan(rec):               # no positive in the ground truth -> F1 is undefined, not 0
        f1 = float('nan')
    elif np.isnan(prec) or prec + rec == 0:   # nothing predicted positive, or tp == 0
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'recall_unsafe': rec,
            'false_alarm_rate': fpr, 'specificity': 1 - fpr if not np.isnan(fpr) else fpr,
            'miss_rate': 1 - rec if not np.isnan(rec) else rec, 'precision_unsafe': prec,
            'f1_unsafe': f1, 'youden_j': (rec - fpr) if not np.isnan(rec * fpr) else float('nan')}


def recall_fn(y, dec):
    return counts(y, dec)['recall_unsafe']


def fpr_fn(y, dec):
    return counts(y, dec)['false_alarm_rate']


def pick_thresholds(sv, yv, scene_fpr_val):
    """All four operating points, chosen on VAL only (pre-registration §3).

    Ties are broken towards the SMALLER threshold in all three data-chosen points (op1 maximises
    val F1 and takes the lowest tau among equals; op2 is the lowest tau meeting the alarm budget;
    op3 takes the lowest tau among equally close alarm matches), so the choice is deterministic
    and does not depend on grid order.
    """
    grid = np.unique(np.concatenate([[0.0], np.sort(sv), [1.0]]))
    rows = [(float(t), counts(yv, (sv >= t).astype(int))) for t in grid]
    op1 = max(rows, key=lambda r: (r[1]['f1_unsafe'], -r[0]))[0]
    budget = [r for r in rows if r[1]['false_alarm_rate'] <= 0.10]
    op2 = min(budget, key=lambda r: r[0])[0] if budget else 1.0
    op3 = min(rows, key=lambda r: (abs(r[1]['false_alarm_rate'] - scene_fpr_val), r[0]))[0]
    return {'OP0_argmax': 0.5, 'OP1_val_max_f1': op1, 'OP2_val_fpr_le_10pct': op2,
            'OP3_val_alarm_matched_to_scene': op3}


def duplicate_floor(names, y, dec, audit_path):
    """False alarms that the LABELS force on any deterministic detector.

    A byte-identical pair with opposite binary labels cannot both be right: whichever decision the
    detector emits, one member is wrong.  Four such pairs touch a SAFE test clip -- one lies wholly
    inside test, three straddle train->test with the unsafe member in train, so a detector that fits
    its training data is pushed to raise the alarm on the safe test twin.
    """
    pos = {n: i for i, n in enumerate(names)}
    forced = []
    for g in json.load(open(audit_path))['groups']:
        if g['kind'] != 'opposite':
            continue
        safe_in_test = [m['name'] for m in g['members']
                        if m['binary'] == 'safe' and m['name'] in pos]
        unsafe_elsewhere = [m['name'] for m in g['members'] if m['binary'] == 'unsafe']
        if safe_in_test and unsafe_elsewhere:
            for n in safe_in_test:
                forced.append({'safe_test_clip': n, 'unsafe_twin': unsafe_elsewhere,
                               'twin_in_test': all(t in pos for t in unsafe_elsewhere),
                               'alarm_raised': bool(dec[pos[n]] == POS)})
    n_safe = int((y != POS).sum())
    # Perfection is ruled out only by a pair lying WHOLLY inside the scored set: the same bytes
    # then carry both labels, so one of the two decisions is wrong whatever the detector emits.
    both_in = any(f['twin_in_test'] for f in forced)
    return {'n_safe_test_clips': n_safe, 'n_forced_candidates': len(forced),
            'n_actually_false_alarmed': sum(f['alarm_raised'] for f in forced),
            'label_induced_fpr_floor': sum(f['alarm_raised'] for f in forced) / n_safe,
            'perfect_detection_possible': not both_in, 'detail': forced}


def per_unsafe_class_recall(names, y, dec):
    """Which KIND of unsafe gets detected: the filename prefix is the original 4-class label."""
    lab = {'0': 'WalkwayViolation', '1': 'UnauthorisedIntrusion',
           '2': 'OpenedPanel', '3': 'CarryOverload'}
    out = {}
    for k, nm in lab.items():
        m = np.array([n.split('_')[0] == k for n in names]) & (y == POS)
        if m.any():
            out[nm] = {'n': int(m.sum()), 'detected': int((dec[m] == POS).sum()),
                       'recall': float(np.mean(dec[m] == POS))}
    return out


def within_scene_auc(y, score, sid, n_boot=5000, seed=0):
    """Mann-Whitney concordance restricted to pairs from the SAME background-context group.

    The pooled ROC-AUC rewards a model for ranking an unsafe clip of a mostly-unsafe scene above a
    safe clip of a mostly-safe scene, which the scene-only control also does.  Conditioning on the
    scene removes that channel and leaves only discrimination the model adds WITHIN a context.
    """
    y = np.asarray(y); score = np.asarray(score); sid = np.asarray(sid)
    pairs = []  # (scene, concordant+0.5*ties, n_pairs)
    for g in np.unique(sid):
        m = sid == g
        pu, pa = score[m & (y == POS)], score[m & (y != POS)]
        if not len(pu) or not len(pa):
            continue
        diff = pu[:, None] - pa[None, :]
        pairs.append((int(g), float((diff > 0).sum() + 0.5 * (diff == 0).sum()), int(diff.size)))
    tot_c = sum(c for _, c, _ in pairs); tot_n = sum(n for _, _, n in pairs)
    idx_by = {int(g): np.where(sid == g)[0] for g in np.unique(sid)}

    def concordance(groups_idx):
        cc = nn = 0.0
        for idx in groups_idx:
            pu, pa = score[idx][y[idx] == POS], score[idx][y[idx] != POS]
            if not len(pu) or not len(pa):
                continue
            diff = pu[:, None] - pa[None, :]
            cc += (diff > 0).sum() + 0.5 * (diff == 0).sum(); nn += diff.size
        return (cc / nn) if nn else float('nan')

    # GROUP unit: resample whole scenes with replacement.  This is the interval to report -- the
    # background-context group is the unit of independence (pre-registration §6), and with only a
    # handful of label-mixed scenes it is wide by construction.
    keys = list(idx_by)
    rng = np.random.default_rng(seed)
    grp = [concordance([idx_by[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
           for _ in range(n_boot)]
    grp = [v for v in grp if not np.isnan(v)]
    g_lo, g_hi = (np.percentile(grp, [2.5, 97.5]) if grp else (float('nan'),) * 2)

    # CLIP unit: scenes stay fixed and only clips inside them are resampled.  Kept for disclosure
    # and labelled WRONG_UNIT -- it treats clips of one scene as independent evidence.
    rng = np.random.default_rng(seed)
    clp = [concordance([idx[rng.integers(0, len(idx), len(idx))] for idx in idx_by.values()])
           for _ in range(n_boot)]
    clp = [v for v in clp if not np.isnan(v)]
    c_lo, c_hi = (np.percentile(clp, [2.5, 97.5]) if clp else (float('nan'),) * 2)

    return {'within_scene_auc': tot_c / tot_n if tot_n else float('nan'),
            'n_within_scene_pairs': tot_n, 'n_mixed_scenes': len(pairs),
            'ci_group_resample': [float(g_lo), float(g_hi)],
            'ci_group_resample_n_valid_draws': len(grp),
            'ci_clip_resample_WRONG_UNIT': [float(c_lo), float(c_hi)],
            'per_scene': {str(g): {'auc': c / n, 'n_pairs': n} for g, c, n in pairs}}


def scene_alarm_profile(y, dec, sid):
    """Per-scene alarm rate against the scene's true unsafe rate, and their correlation."""
    y = np.asarray(y); dec = np.asarray(dec); sid = np.asarray(sid)
    rows = {}
    for g in np.unique(sid):
        m = sid == g; u = m & (y == POS); a = m & (y != POS)
        rows[str(int(g))] = {
            'n': int(m.sum()), 'n_unsafe': int(u.sum()), 'n_safe': int(a.sum()),
            'recall': float(dec[u].mean()) if u.any() else None,
            'false_alarm_rate': float(dec[a].mean()) if a.any() else None,
            'true_unsafe_rate': float(y[m].mean()), 'alarm_rate': float(dec[m].mean())}
    ar = np.array([r['alarm_rate'] for r in rows.values()])
    tr = np.array([r['true_unsafe_rate'] for r in rows.values()])
    within = {k: r['recall'] - r['false_alarm_rate'] for k, r in rows.items()
              if r['recall'] is not None and r['false_alarm_rate'] is not None}
    pearson = (float(np.corrcoef(ar, tr)[0, 1]) if ar.std() > 0 and tr.std() > 0
               else float('nan'))      # a constant column makes the correlation undefined
    return {'per_scene': rows, 'pearson_alarm_vs_true_rate': pearson,
            'within_scene_youden_j': within}


