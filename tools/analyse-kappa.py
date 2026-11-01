#!/usr/bin/env python3
"""Analysis for the iSafetyBench binary label-ceiling study. WRITTEN AND COMMITTED BEFORE ANY
CLIP WAS ANNOTATED -- that is the point of this file, not an incidental fact about it. The 31/10
panel found the project's gate charters entered git AFTER the results they adjudicate (risks.md
01/10 vs the G14 result 29/09), a provenance hole no re-run can patch. This study does not
repeat it: the sampler, the codebook and this script are one commit, dated before the data.

Everything below is fixed in advance. Reading a number out of here is allowed to change what
the thesis CONCLUDES; it is not allowed to change what this script COMPUTES.

Primary estimand: agreement between an independent re-annotation and iSafetyBench's own binary
label, on a census of all 151 sources (one clip each, 76 hazard / 75 normal by construction).
Reported as Cohen's kappa AND Gwet's AC1 -- kappa alone is prevalence-sensitive, and while this
sample is near-balanced the subgroup cuts are not.

Second estimand, and the one that actually decides the thesis argument: human balanced accuracy
against the same labels, on the same 151 clips, compared with the B1 model's balanced accuracy
restricted to those same clips (read from the already-published out-of-fold predictions -- no
model is run, no test set is touched, see the pre-registration's budget declaration).
"""
import json, pathlib, sys
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
BASE = REPO / 'annotation_guide/kappa-isb-binary'
OOF = REPO / 'sweep_results/relation_tokens/enc_isbclip_B1_lso-1100_headline-none_oof.json'

# Pre-registered exclusion. K096's real path was printed by a server traceback during the
# 01/11 build-and-test session, in a transcript the annotator can read, so its label may have
# been seen before annotation. Excluded in advance rather than argued about afterwards; n = 150,
# and the 01/11 pre-registration records this with the reason.
EXCLUDE = {'K096'}
B = 10000
RNG = np.random.default_rng(20261101)


def kappa(a, b):
    po = float((a == b).mean())
    pe = sum(float((a == k).mean()) * float((b == k).mean()) for k in (0, 1))
    return (po - pe) / (1 - pe) if pe < 1 else float('nan')


def ac1(a, b):
    """Gwet's AC1: same observed agreement, chance term that does not collapse under skew."""
    po = float((a == b).mean())
    pi = (float((a == 1).mean()) + float((b == 1).mean())) / 2
    pe = 2 * pi * (1 - pi)
    return (po - pe) / (1 - pe) if pe < 1 else float('nan')


def bal_acc(y, yhat):
    out = []
    for k in (0, 1):
        m = y == k
        out.append(float((yhat[m] == k).mean()) if m.any() else float('nan'))
    return float(np.mean(out))


def boot_ci(fn, *arrays):
    """Bootstrap over SOURCES. By design each source contributes exactly one clip, so the
    clusters are singletons and this reduces to the ordinary clip bootstrap -- an intended
    property of the census design, not a missing cluster correction."""
    n = len(arrays[0])
    vals = []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        v = fn(*[a[i] for a in arrays])
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_exact(c1, c2):
    """Exact two-sided McNemar on per-clip correctness. Same test the thesis uses elsewhere."""
    from math import comb
    b = int(((c1 == 1) & (c2 == 0)).sum())
    c = int(((c1 == 0) & (c2 == 1)).sum())
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return float(min(1.0, p)), b, c


def line(name, val, ci=None, extra=''):
    s = f'{name:<44} {val:.4f}'
    if ci:
        s += f'  CI95 [{ci[0]:.4f}, {ci[1]:.4f}]'
    print(s + ('  ' + extra if extra else ''))


def main():
    key = json.loads((BASE / 'key.json').read_text())
    ann_path = BASE / 'annotations.jsonl'
    if not ann_path.exists():
        sys.exit('no annotations yet: annotations.jsonl absent')
    rows = [json.loads(l) for l in ann_path.read_text().splitlines() if l.strip()]
    # last write per (kid, pass) wins; the UI is append-only so a correction is a later line
    seen = {}
    for r in rows:
        seen[(r['kid'], r['pass'])] = r
    p1 = {k[0]: v for k, v in seen.items() if k[1] == 1}
    p2 = {k[0]: v for k, v in seen.items() if k[1] == 2}
    if len(p1) != 151:
        print(f'WARNING: pass 1 has {len(p1)}/151 clips -- partial run', file=sys.stderr)

    kids = sorted(k for k in p1 if k not in EXCLUDE)
    if EXCLUDE:
        print(f'== loai tru truoc dang ky: {sorted(EXCLUDE)}')
    gold = np.array([key[k]['gold'] for k in kids])
    hum = np.array([p1[k]['label'] for k in kids])
    conf = np.array([p1[k]['confidence'] for k in kids])
    recog = np.array([1 if p1[k]['recognised'] else 0 for k in kids])

    print(f'== n = {len(kids)} clip, {int((gold==1).sum())} hazard / {int((gold==0).sum())} normal')
    print(f'== nhan doc lap: {int((hum==1).sum())} hazard / {int((hum==0).sum())} normal\n')

    print('-- dong thuan nguoi <-> nhan goc iSafetyBench')
    line('agreement tho', float((hum == gold).mean()))
    line("Cohen kappa", kappa(hum, gold), boot_ci(kappa, hum, gold))
    line("Gwet AC1", ac1(hum, gold), boot_ci(ac1, hum, gold))

    print('\n-- tran nhan do bang chinh dai luong cua luan van')
    hba = bal_acc(gold, hum)
    line('balanced accuracy cua NGUOI', hba, boot_ci(lambda g, h: bal_acc(g, h), gold, hum))

    oof = {r['clip_id']: r for r in json.loads(OOF.read_text())['rows']}
    mdl = np.array([oof[key[k]['clip_id']]['y_pred'] for k in kids])
    assert all(oof[key[k]['clip_id']]['y_true'] == key[k]['gold'] for k in kids), 'gold mismatch vs OOF'
    mba = bal_acc(gold, mdl)
    line('balanced accuracy cua MO HINH (cung 151 clip)', mba,
         boot_ci(lambda g, m: bal_acc(g, m), gold, mdl))
    line('hieu NGUOI - MO HINH', hba - mba,
         boot_ci(lambda g, h, m: bal_acc(g, h) - bal_acc(g, m), gold, hum, mdl))
    p, b, c = mcnemar_exact((hum == gold).astype(int), (mdl == gold).astype(int))
    print(f"{'McNemar exact (nguoi vs mo hinh)':<44} p = {p:.4g}  "
          f'nguoi dung/mo hinh sai = {b}, nguoc lai = {c}')

    print('\n-- do tin cay cua chinh nguoi gan nhan (test-retest)')
    both = sorted((set(p1) & set(p2)) - EXCLUDE)
    if both:
        a = np.array([p1[k]['label'] for k in both])
        bb = np.array([p2[k]['label'] for k in both])
        line(f'kappa noi-nguoi tren {len(both)} clip lap', kappa(a, bb))
        print(f"{'so clip tu doi y':<44} {int((a != bb).sum())}/{len(both)}")
    else:
        print('chua co lan hien thi thu hai')

    print('\n-- cat theo bien da tien dang ky')
    for nm, m in (('clip NHAN RA (nho tu truoc)', recog == 1), ('clip KHONG nhan ra', recog == 0),
                  ('tu tin cao (3)', conf == 3), ('tu tin thap (1-2)', conf <= 2)):
        if m.sum() >= 10:
            line(f'  kappa | {nm} (n={int(m.sum())})', kappa(hum[m], gold[m]))
        else:
            print(f'  kappa | {nm}: n={int(m.sum())} < 10, khong bao cao')


if __name__ == '__main__':
    main()
