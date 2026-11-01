#!/usr/bin/env python3
"""Deterministic blinded sample for the iSafetyBench binary label-ceiling study (kappa).

Design (pre-registered in plans/261031-.../tien-dang-ky-kappa-nhi-phan-isafetybench.md):
  * CENSUS of all 151 source videos, exactly ONE clip per source. 151 sources are 76 hazard /
    75 normal, so the sample is label-balanced by construction -- no stratification needed and
    no kappa prevalence paradox. One clip per source also means zero intra-source correlation
    (ICC by source is 0.43, so n=200 with two clips per source carries LESS information than
    this n=151) and, critically, no within-source label leakage: every source is single-class,
    so seeing a second clip of a source you already judged would hand you the answer.
  * 25 planted repeats (seeded choice) re-shown later in the order, to measure the annotator's
    test-retest agreement with themselves. Without that number the between-rater kappa is
    uninterpretable: a low kappa could be the label's noise or just the annotator's.
  * Masked ids. The real filename encodes the source id AND the folder encodes the label, so the
    annotation UI must never see either. Events carry K-ids only; the key stays server-side.

Output is a deterministic function of (seed, manifest order_sha256, this file). Nothing is
sealed or secret -- it is REPRODUCIBLE, which is strictly stronger. Re-running must reproduce
events_sha256 byte for byte; the pre-registration records it.
"""
import argparse, hashlib, json, pathlib, sys

SEED = 20261101
N_REPEATS = 25
MIN_GAP = 40        # minimum events between a clip's two showings
REPO = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPO / 'kaggle/isb-clip-manifest.json'
EXPECTED_ORDER_SHA = '11f757c01b26b3b14dd7ece9b7640005e7c3dc05807346900e72dfc6d267ae4f'
ROOT = pathlib.Path('data/isafetybench')          # kept relative: key.json must be portable
OUT = REPO / 'annotation_guide/kappa-isb-binary'


def stable_rand(*parts):
    """Deterministic uniform in [0,1) from a seed and strings. Not numpy: no version drift."""
    h = hashlib.sha256(('|'.join(str(p) for p in parts)).encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='write the files (default: dry run)')
    a = ap.parse_args()

    man = json.loads(MANIFEST.read_text())
    if man['order_sha256'] != EXPECTED_ORDER_SHA:
        sys.exit(f"manifest order changed: {man['order_sha256']}")

    by_src = {}
    for cid, lab, src in zip(man['clip_ids'], man['labels'], man['views']):
        by_src.setdefault(src, []).append((cid, lab))
    if len(by_src) != 151:
        sys.exit(f'expected 151 sources, got {len(by_src)}')
    for src, rows in by_src.items():
        if len({l for _, l in rows}) != 1:
            sys.exit(f'source {src} is not single-class')

    # one clip per source: the lowest stable_rand, so the choice is seed-determined not order-determined
    chosen = []
    for src in sorted(by_src):
        rows = sorted(by_src[src])
        cid, lab = min(rows, key=lambda r: stable_rand(SEED, 'pick', src, r[0]))
        chosen.append({'source': src, 'clip_id': cid, 'gold': lab})
    assert len(chosen) == 151

    # K-ids assigned in seeded order so the id itself leaks nothing about label or source
    for c in chosen:
        c['_o'] = stable_rand(SEED, 'kid', c['clip_id'])
    for i, c in enumerate(sorted(chosen, key=lambda c: c['_o']), 1):
        c['kid'] = f'K{i:03d}'

    # presentation order: all 151 in a seeded first pass, then the 25 repeats. A repeat must sit
    # at least MIN_GAP events after its first showing, else it is spotted as a repeat from recency
    # alone and the test-retest number measures memory, not judgement. Drawing the repeat pool from
    # the first (151 - MIN_GAP) positions satisfies the gap for every tail position.
    first = [c['kid'] for c in sorted(chosen, key=lambda c: stable_rand(SEED, 'ord', c['kid']))]
    pos = {k: i for i, k in enumerate(first)}
    eligible = [k for k in first if pos[k] <= len(first) - MIN_GAP]
    repeats = sorted(eligible, key=lambda k: stable_rand(SEED, 'rep', k))[:N_REPEATS]
    tail = sorted(repeats, key=lambda k: stable_rand(SEED, 'tail', k))
    events = [{'event': i + 1, 'kid': k, 'pass': 1} for i, k in enumerate(first)]
    for j, k in enumerate(tail):
        if len(events) - pos[k] < MIN_GAP:
            sys.exit(f'repeat {k} too close to its first showing')
        events.append({'event': len(events) + 1, 'kid': k, 'pass': 2})

    key = {c['kid']: {'clip_id': c['clip_id'], 'source': c['source'], 'gold': c['gold'],
                      'path': str(ROOT / ('hazard' if c['gold'] == 1 else 'normal') / c['clip_id'])}
           for c in chosen}
    for v in key.values():
        if not (REPO / v['path']).exists():
            sys.exit(f"missing clip: {v['path']}")

    ev_json = json.dumps({'seed': SEED, 'n_clips': len(chosen), 'n_events': len(events),
                          'order_sha256': man['order_sha256'], 'events': events},
                         indent=1, sort_keys=True)
    ev_sha = hashlib.sha256(ev_json.encode()).hexdigest()
    key_json = json.dumps(key, indent=1, sort_keys=True)

    n_haz = sum(1 for c in chosen if c['gold'] == 1)
    print(f'clips {len(chosen)} | hazard {n_haz} normal {len(chosen)-n_haz} | events {len(events)}'
          f' | repeats {len(repeats)}\nevents_sha256 {ev_sha}\nkey_sha256    '
          f'{hashlib.sha256(key_json.encode()).hexdigest()}')

    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'events.json').write_text(ev_json)
        (OUT / 'key.json').write_text(key_json)
        print(f'wrote {OUT}/events.json and {OUT}/key.json')
    else:
        print('dry run; pass --write')


if __name__ == '__main__':
    main()
