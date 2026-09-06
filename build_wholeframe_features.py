#!/usr/bin/env python3
"""S3 diagnostic — extract WHOLE-FRAME frozen features (no detector, no crops) per clip.

This represents the paper's scene-dependent approach: classify behaviour from the full frame (which
encodes the camera/scene). Compared later against actor-crop + our entity model under same-distribution
AND leave-camera-out, to quantify how much accuracy is scene memorisation.

Output: sweep_results/relation_tokens/tokens_wf_{train,test}.pt  →  (Nclips, T, APP_DIM) + labels + views
Usage: .venv/bin/python build_wholeframe_features.py
"""
import glob
import os
import json
import numpy as np
import cv2
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'sweep_results', 'relation_tokens')
VIEW_MAP = os.path.join(ROOT, 'sweep_results', 'zone_prior', 'clip_view_map.json')
T, APP_DIM, UNSAFE = 16, 64, [0, 1, 2, 3]


def get_model(device):
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    w = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    net = mobilenet_v3_small(weights=w)
    feat = torch.nn.Sequential(net.features, net.avgpool, torch.nn.Flatten()).to(device).eval()
    rng = np.random.default_rng(0)
    proj = torch.tensor((rng.standard_normal((576, APP_DIM)) / np.sqrt(APP_DIM)).astype('float32'), device=device)
    return feat, w.transforms(), proj


@torch.no_grad()
def process(path, feat, tf, proj, device):
    from PIL import Image
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or T
    idxs = [int(x) for x in np.linspace(0, max(0, total - 1), T)]
    ims = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i, max(0, total - 2)))
        ok, fr = cap.read()
        ims.append(tf(Image.fromarray(fr[:, :, ::-1])) if ok else torch.zeros(3, 224, 224))
    cap.release()
    x = torch.stack(ims).to(device)
    f = feat(x) @ proj                       # (T, APP_DIM)
    return f.cpu().numpy()


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    feat, tf, proj = get_model(device)
    vmap = {r['basename']: str(r['view']) for r in json.load(open(VIEW_MAP))}
    os.makedirs(OUT, exist_ok=True)
    for sp in ('train', 'test'):
        clips = []
        for c in UNSAFE:
            clips += sorted(glob.glob(os.path.join(ROOT, 'data', sp, f'{c}_*', '*.mp4')))
        feats, labels, views = [], [], []
        for k, path in enumerate(clips):
            base = os.path.basename(path)
            feats.append(process(path, feat, tf, proj, device))
            labels.append(int(base.split('_')[0])); views.append(vmap.get(base, 'na'))
            if (k + 1) % 100 == 0:
                print(f"  [{sp}] {k+1}/{len(clips)}", flush=True)
        torch.save({'feats': torch.tensor(np.stack(feats)), 'labels': torch.tensor(labels), 'views': views},
                   os.path.join(OUT, f'tokens_wf_{sp}.pt'))
        print(f"[{sp}] {len(clips)} clips → tokens_wf_{sp}.pt  shape {tuple(np.stack(feats).shape)}")


if __name__ == '__main__':
    main()
