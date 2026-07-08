#!/usr/bin/env python3
"""
κ Stage 2 — compute Cohen's κ between annotator-1 (you) and annotator-2 (blind).

Reads kappa_groundtruth.csv (annotator-1) + kappa_annotator2.csv (filled by annotator-2), aligns by
anon_id, and reports Cohen's κ on the 5-way frame label {0,1,2,3,bg}, raw agreement, per-class
agreement, and the confusion matrix. Result saved for §4.6.

Usage:  .venv/bin/python3 run_kappa_compute.py
"""
import csv, json, os
from collections import Counter
from sklearn.metrics import cohen_kappa_score, confusion_matrix

GT = 'annotation_guide/kappa_groundtruth.csv'
A2 = 'annotation_guide/kappa_annotator2.csv'
LABELS = ['0', '1', '2', '3', 'bg']


def read(path, key, val):
    # auto-detect delimiter (Excel/Numbers may save ';' by locale)
    first = open(path).readline()
    delim = ';' if first.count(';') > first.count(',') else ','
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter=delim):
            out[r[key].strip()] = r[val].strip().lower()   # normalise 'Bg'→'bg'
    return out


def main():
    a1 = read(GT, 'anon_id', 'annotator1_label')
    a2 = read(A2, 'anon_id', 'label')
    pairs = [(a1[k], a2[k]) for k in a1 if a2.get(k, '') != '']
    missing = [k for k in a1 if a2.get(k, '') == '']
    if not pairs:
        print(f"  ⚠ annotator-2 has not filled any labels yet ({len(a1)} rows blank). "
              f"Fill {A2} first."); return
    if missing:
        print(f"  ⚠ {len(missing)} of {len(a1)} rows not yet labelled by annotator-2; "
              f"computing κ on the {len(pairs)} completed rows.")

    y1, y2 = zip(*pairs)
    # normalise unexpected tokens to bg
    norm = lambda v: v if v in LABELS else 'bg'
    y1 = [norm(v) for v in y1]; y2 = [norm(v) for v in y2]

    kappa = cohen_kappa_score(y1, y2, labels=LABELS)
    raw = sum(a == b for a, b in zip(y1, y2)) / len(y1)
    cm = confusion_matrix(y1, y2, labels=LABELS)

    interp = ('almost perfect' if kappa >= 0.81 else 'substantial' if kappa >= 0.61 else
              'moderate' if kappa >= 0.41 else 'fair' if kappa >= 0.21 else 'slight/poor')
    print(f"\n  Cohen's κ = {kappa:.3f}  ({interp}; target ≥ 0.61)")
    print(f"  Raw agreement = {raw*100:.1f}%  (n = {len(y1)})")
    print(f"\n  Confusion matrix (rows=annotator-1, cols=annotator-2), labels {LABELS}:")
    for lab, row in zip(LABELS, cm):
        print(f"    {lab:>3}  {row}")
    # per-class agreement (one-vs-rest)
    print(f"\n  Per-class agreement:")
    for lab in LABELS:
        n1 = sum(v == lab for v in y1)
        agree = sum(a == b == lab for a, b in zip(y1, y2))
        print(f"    {lab:>3}: annotator-1 had {n1}, both agreed {agree}")

    out = {'kappa': round(kappa, 4), 'interpretation': interp, 'raw_agreement': round(raw, 4),
           'n': len(y1), 'labels': LABELS, 'confusion_matrix': cm.tolist()}
    os.makedirs('plans/260613-thesis-defense-revision/reports', exist_ok=True)
    json.dump(out, open('plans/260613-thesis-defense-revision/reports/kappa-result.json', 'w'),
              indent=2)
    print(f"\n  Saved → plans/260613-thesis-defense-revision/reports/kappa-result.json")
    if kappa < 0.61:
        print(f"  ⚠ κ below 0.61 — refine the labeling guide on ambiguous cases and re-label.")


if __name__ == '__main__':
    main()
