#!/usr/bin/env python3
"""Byte-level duplicate / label-contradiction audit of the Unsafe-Net clip set.

Hashes every clip file and reports groups of byte-identical videos filed under different
class labels. Such groups make the benchmark internally contradictory: the same pixels carry
two ground truths, so no deterministic classifier can be correct on every member.

Usage: .venv/bin/python run_duplicate_audit.py [--data data] [--out sweep_results/duplicate_audit.json]
"""
import argparse, collections, hashlib, json, os

VID = ('.mp4', '.avi', '.mov')
UNSAFE = {'0', '1', '2', '3'}


def scan(root):
    rows = []
    for split in ('train', 'test'):
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        for cls in sorted(os.listdir(d)):
            cd = os.path.join(d, cls)
            if not os.path.isdir(cd):
                continue
            for f in sorted(os.listdir(cd)):
                if f.lower().endswith(VID):
                    p = os.path.join(cd, f)
                    rows.append({'split': split, 'cls': cls, 'name': f, 'size': os.path.getsize(p),
                                 'md5': hashlib.md5(open(p, 'rb').read()).hexdigest()})
    return rows


def audit(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r['md5']].append(r)
    groups = [v for v in by.values() if len(v) > 1]
    conflict = [g for g in groups if len({r['cls'] for r in g}) > 1]
    cross = [g for g in conflict if len({r['split'] for r in g}) > 1]
    pairs = collections.Counter()
    for g in conflict:
        pairs[tuple(sorted({r['cls'].split('_')[0] for r in g}))] += 1
    leaked_test = sorted({r['name'] for g in cross for r in g if r['split'] == 'test'})
    unsafe_test = [n for n in leaked_test if n.split('_')[0] in UNSAFE]
    return {'n_files': len(rows), 'n_distinct': len(by), 'n_groups': len(groups),
            'n_conflicting_label': len(conflict), 'n_cross_split': len(cross),
            'test_clips_with_train_twin': leaked_test,
            'unsafe_test_clips_affected': unsafe_test,
            'class_pairs_conflated': {' <-> '.join(k): v for k, v in pairs.most_common()},
            'groups': [[{k: r[k] for k in ('split', 'cls', 'name')} for r in g] for g in conflict]}


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--data', default='data')
    a.add_argument('--out', default='sweep_results/duplicate_audit.json')
    v = a.parse_args()
    r = audit(scan(v.data))
    os.makedirs(os.path.dirname(v.out), exist_ok=True)
    json.dump(r, open(v.out, 'w'), indent=2)
    print(f"{r['n_files']} clips, {r['n_distinct']} distinct -> {r['n_groups']} duplicate groups")
    print(f"  groups with CONFLICTING labels : {r['n_conflicting_label']}")
    print(f"  of those, spanning train/test  : {r['n_cross_split']}")
    print(f"  4-class UNSAFE test clips hit  : {len(r['unsafe_test_clips_affected'])}/64 "
          f"-> {r['unsafe_test_clips_affected']}")
    print(f"  class pairs conflated          : {r['class_pairs_conflated']}")
    print(f"-> written {v.out}")
