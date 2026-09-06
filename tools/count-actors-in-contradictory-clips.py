"""Test the multi-label hypothesis for the byte-identical / conflicting-label groups.

Hypothesis: the duplicate filings are not a packaging bug. The same footage was filed under two
classes because two labelled behaviours are simultaneously visible (different actors, same clip).
Prediction: clips in contradictory groups contain MORE simultaneous people than control clips.

Counts COCO-person detections per sampled frame with a generic detector (yolo11s, unmodified),
so the count is independent of anything we trained on this dataset.

RESULT (2026-09-06): the prediction FAILED. Contradictory clips carry 3.12 max simultaneous
persons on average vs 3.03 for controls (Mann-Whitney p=0.38). Multi-actor scenes are the norm
across the WHOLE dataset (>=2 people in 83% of control clips), not a property of the duplicated
groups. Reported as such; see footage-verification-of-label-contradictions.md.

With --behaviour-model, additionally counts how many DISTINCT clip-level behaviour classes a
detector fires per clip. CAVEAT: that detector was trained on this dataset with clip labels
propagated to boxes, so it inherits the very ambiguity under test — supporting evidence only,
not independent evidence.

Usage: .venv/bin/python tools/count-actors-in-contradictory-clips.py [--frames 8] [--n-control 60]
       [--behaviour-model runs/detect/sweep_results/r2/beh_ctx/weights/best.pt]
"""
import argparse, glob, json, os, random
import cv2
import numpy as np
from ultralytics import YOLO

AUDIT = 'sweep_results/duplicate_audit.json'
OUT = 'sweep_results/contradictory_actor_counts.json'


def path_of(rec):
    return f"data/{rec['split']}/{rec['cls']}/{rec['name']}"


def distinct_behaviours(model, clip, n_frames, conf):
    """How many distinct clip-level behaviour classes the detector fires anywhere in the clip."""
    cap = cv2.VideoCapture(clip)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seen = set()
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(i * (total - 1) / max(n_frames - 1, 1))))
        ok, frame = cap.read()
        if not ok:
            continue
        r = model.predict(frame, conf=conf, verbose=False, device='mps')[0]
        seen.update(int(c) for c in r.boxes.cls.tolist())
    cap.release()
    return sorted(seen)


def counts(model, clip, n_frames, conf):
    cap = cv2.VideoCapture(clip)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(round(i * (total - 1) / max(n_frames - 1, 1))) for i in range(n_frames)]
    per_frame = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        r = model.predict(frame, conf=conf, classes=[0], verbose=False, device='mps')[0]
        per_frame.append(int(len(r.boxes)))
    cap.release()
    return per_frame


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--frames', type=int, default=8)
    a.add_argument('--n-control', type=int, default=60)
    a.add_argument('--conf', type=float, default=0.35)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--behaviour-model', default=None)
    v = a.parse_args()

    groups = json.load(open(AUDIT))['groups']
    dup_names = {r['name'] for g in groups for r in g}
    # one representative per group: members are byte-identical, so one read suffices
    reps = [g[0] for g in groups]

    all_clips = sorted(glob.glob('data/*/*/*.mp4'))
    control = [c for c in all_clips if os.path.basename(c) not in dup_names]
    random.Random(v.seed).shuffle(control)
    control = control[:v.n_control]

    model = YOLO('yolo11s.pt')
    beh = YOLO(v.behaviour_model) if v.behaviour_model else None
    res = {'frames_per_clip': v.frames, 'conf': v.conf, 'contradictory': [], 'control': []}

    for i, rec in enumerate(reps):
        pf = counts(model, path_of(rec), v.frames, v.conf)
        rec_out = {'group': i, 'clip': rec['name'],
                   'classes': sorted({r['cls'][0] for r in groups[i]}),
                   'per_frame': pf, 'max': max(pf), 'mean': float(np.mean(pf))}
        if beh:
            rec_out['behaviours'] = distinct_behaviours(beh, path_of(rec), v.frames, v.conf)
        res['contradictory'].append(rec_out)
        print(f"  grp{i:2d} {rec['name']:16s} max={max(pf)} mean={np.mean(pf):.2f} {pf}")

    for c in control:
        pf = counts(model, c, v.frames, v.conf)
        rec_out = {'clip': os.path.basename(c), 'per_frame': pf,
                   'max': max(pf), 'mean': float(np.mean(pf))}
        if beh:
            rec_out['behaviours'] = distinct_behaviours(beh, c, v.frames, v.conf)
        res['control'].append(rec_out)

    cm = np.array([d['max'] for d in res['contradictory']])
    km = np.array([d['max'] for d in res['control']])
    res['summary'] = {
        'contradictory_n': len(cm), 'control_n': len(km),
        'contradictory_max_mean': float(cm.mean()), 'control_max_mean': float(km.mean()),
        'contradictory_frac_ge2': float((cm >= 2).mean()), 'control_frac_ge2': float((km >= 2).mean()),
    }
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(cm, km, alternative='greater')
    res['summary']['mannwhitney_u'] = float(u); res['summary']['p_greater'] = float(p)
    json.dump(res, open(OUT, 'w'), indent=2)
    s = res['summary']
    print(f"\nmax simultaneous persons per clip")
    print(f"  contradictory (n={s['contradictory_n']}): mean {s['contradictory_max_mean']:.2f}  "
          f">=2 people in {100*s['contradictory_frac_ge2']:.0f}% of clips")
    print(f"  control       (n={s['control_n']}): mean {s['control_max_mean']:.2f}  "
          f">=2 people in {100*s['control_frac_ge2']:.0f}% of clips")
    print(f"  Mann-Whitney U (contradictory > control) p={s['p_greater']:.4g}")
    if beh:
        for k in ('contradictory', 'control'):
            nb = np.array([len(d['behaviours']) for d in res[k]])
            res['summary'][f'{k}_behaviours_mean'] = float(nb.mean())
            res['summary'][f'{k}_frac_multi_behaviour'] = float((nb >= 2).mean())
            print(f"  {k:14s}: {nb.mean():.2f} distinct behaviour classes/clip, "
                  f">=2 in {100*(nb >= 2).mean():.0f}% of clips")
        json.dump(res, open(OUT, 'w'), indent=2)
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
