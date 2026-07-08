#!/usr/bin/env python3
"""
WS4 helper — extract one representative frame per camera-viewpoint cluster so a human can
draw a restricted-zone polygon per view (the manual half of "virtual fencing").

Unsafe-Net has no camera/viewpoint labels, but clips cluster strongly by background
(silhouette peaks ≈0.45 at k=6). We k-means the 516 four-class clips on their time-averaged
spatial features, pick the clip nearest each cluster centroid, and dump its middle frame to
`annotation_guide/zones/view_{id}.png`. The operator draws ONE polygon (the walkway / allowed
zone) per image; `tools/apply_zone_prior.py` (follow-up) then computes person-in-zone features.

Reconstructs the SAME clip ordering as run_pairwise_contrastive.extract_all_features so cluster
indices map back to real video files.

Usage:
  .venv/bin/python3 tools/extract_viewpoint_frames.py [--k 8]
"""
import os, json, argparse
import numpy as np
import cv2
import torch
from sklearn.cluster import KMeans

DATA_DIR = 'data'
FEAT = 'sweep_results/pairwise/features'
OUT = 'annotation_guide/zones'
VID_EXT = ('.mp4', '.avi', '.mov')


def clip_paths():
    """Replicate extract_all_features ordering: train (8 sorted classes) then test."""
    paths, labels = [], []
    for subset in ['train', 'test']:
        sp = os.path.join(DATA_DIR, subset)
        for folder in sorted(os.listdir(sp)):
            full = os.path.join(sp, folder)
            if not os.path.isdir(full):
                continue
            lab = int(folder.split('_')[0])
            for vf in sorted(os.listdir(full)):
                if vf.lower().endswith(VID_EXT):
                    paths.append(os.path.join(full, vf)); labels.append(lab)
    return paths, np.array(labels)


def save_mid_frame(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, frame = cap.read(); cap.release()
    if ok:
        cv2.imwrite(out_path, frame)
        return frame.shape[1], frame.shape[0]   # w, h
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    tr = torch.load(os.path.join(FEAT, 'all8_train.pt'), map_location='cpu')
    te = torch.load(os.path.join(FEAT, 'all8_test.pt'), map_location='cpu')
    feats = torch.cat([tr['features'], te['features']], 0)
    labels = torch.cat([tr['labels'], te['labels']], 0)
    paths, plabels = clip_paths()
    assert len(paths) == len(labels), f"path/feature mismatch {len(paths)} vs {len(labels)}"

    m4 = (labels < 4).numpy()
    rep = feats.mean(1).reshape(len(labels), -1).numpy()[m4]    # (516, 8192)
    y4 = labels.numpy()[m4]
    paths4 = [p for p, keep in zip(paths, m4) if keep]
    names = ['Walk', 'Unauth', 'Panel', 'Forklift']

    km = KMeans(args.k, random_state=0, n_init=10).fit(rep)
    g = km.labels_
    manifest = []
    print(f"{len(y4)} four-class clips → {args.k} viewpoint clusters\n")
    for c in range(args.k):
        idx = np.where(g == c)[0]
        d = np.linalg.norm(rep[idx] - km.cluster_centers_[c], axis=1)
        pick = idx[int(np.argmin(d))]
        dist = [int((y4[idx] == j).sum()) for j in range(4)]
        out_png = os.path.join(OUT, f'view_{c}.png')
        wh = save_mid_frame(paths4[pick], out_png)
        rec = {'view': c, 'n_clips': int(len(idx)), 'class_dist': dict(zip(names, dist)),
               'representative_clip': os.path.basename(paths4[pick]),
               'frame_wh': wh, 'frame_png': out_png}
        manifest.append(rec)
        mix = 'MIXED Walk+Unauth' if dist[0] > 0 and dist[1] > 0 else ''
        print(f"  view {c}: n={len(idx):3d} {dict(zip(names, dist))} "
              f"→ {os.path.basename(paths4[pick])} {wh} {mix}")

    json.dump({'k': args.k, 'note': 'draw ONE allowed/walkway polygon per view_{id}.png; '
               'save polygons to annotation_guide/zones/zones.json as '
               '{"view_id": [[x,y],...]} in pixel coords of frame_wh',
               'views': manifest}, open(os.path.join(OUT, 'viewpoints.json'), 'w'), indent=2)
    print(f"\n[SAVED] {OUT}/view_*.png + viewpoints.json")


if __name__ == '__main__':
    main()
