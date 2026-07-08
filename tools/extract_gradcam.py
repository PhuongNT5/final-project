#!/usr/bin/env python3
"""
Extract Grad-CAM saliency maps for the 64-clip 4-class test set using the
direct4_s314 checkpoint. Phase 04 Step 1-4: interpretability artefact for
§4.5 error-analysis (decision-gate output for whether to pursue pseudo-bbox).

Outputs:
  figures/gradcam/cams_npz/clip_NN.npz       — (T, 224, 224) float16 CAMs
  figures/gradcam/overlays/clip_NN_*.png     — middle-frame heatmap overlay
  figures/gradcam/metadata.json              — predictions + paths
"""
import os, sys, json, gc

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms

from run_pairwise_contrastive import TransformerHead, CLASS_NAMES_4

DATA_DIR = 'data'
CKPT = 'sweep_results/pairwise/direct4_s314.pt'
OUT_DIR = 'figures/gradcam'
TARGET_LAYER_IDX = 22
T = 32
POOL = 4
DEVICE = 'cpu'

NORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def list_test_clips_4class():
    """Test clips in the order the cached test feature tensor was built with."""
    out, idx = [], 0
    test_root = os.path.join(DATA_DIR, 'test')
    for folder in sorted(os.listdir(test_root)):
        full = os.path.join(test_root, folder)
        if not os.path.isdir(full): continue
        label = int(folder.split('_')[0])
        if label >= 4: continue
        for vf in sorted(os.listdir(full)):
            if vf.lower().endswith(('.mp4', '.avi', '.mov')):
                idx += 1
                out.append((idx, os.path.join(full, vf), label))
    return out


def load_models(device):
    from ultralytics import YOLO
    yolo = YOLO('yolo11s.pt')
    backbone = yolo.model.to(device); backbone.eval()
    for p in backbone.parameters(): p.requires_grad_(True)  # need autograd graph
    head = TransformerHead(in_dim=512 * 3, nc=4, dim=384, nhead=8, layers=6, drop=0.3).to(device)
    head.load_state_dict(torch.load(CKPT, map_location=device))
    head.eval()
    return backbone, head


def gradcam_for_clip(video_path, backbone, head, device):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(1, total - 1), T, dtype=int)

    rgb_frames, feats_per_frame = [], []
    captured = {}

    def hook(_m, _i, output):
        out = output[0] if isinstance(output, (list, tuple)) else output
        captured['feat'] = out

    handle = backbone.model[TARGET_LAYER_IDX].register_forward_hook(hook)
    try:
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                rgb_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                feats_per_frame.append(None)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(cv2.resize(rgb, (224, 224)).astype(np.uint8))
            tensor = NORM(Image.fromarray(rgb)).unsqueeze(0).to(device)
            captured.clear()
            backbone(tensor)  # WITHOUT torch.no_grad — graph required
            feat = captured['feat']
            feat.retain_grad()
            feats_per_frame.append(feat)
    finally:
        cap.release()
        handle.remove()

    # Replace any None with zero placeholder matching shape of first valid feat
    valid = next((f for f in feats_per_frame if f is not None), None)
    if valid is None:
        return None, None, None, None
    for i, f in enumerate(feats_per_frame):
        if f is None:
            feats_per_frame[i] = torch.zeros_like(valid)

    # Pool 4×4 → 16 spatial tokens, then mean/std/max stats over S=16 → (1, T, 1536)
    pooled_tokens = []
    for feat in feats_per_frame:
        fh, fw = feat.shape[2], feat.shape[3]
        if fh % POOL != 0 or fw % POOL != 0:
            nh = max(POOL, (fh // POOL) * POOL)
            nw = max(POOL, (fw // POOL) * POOL)
            feat_for_pool = F.interpolate(feat, size=(nh, nw), mode='bilinear', align_corners=False)
        else:
            feat_for_pool = feat
        p = F.adaptive_avg_pool2d(feat_for_pool, (POOL, POOL))
        pooled_tokens.append(p.flatten(2).transpose(1, 2).squeeze(0))  # (16, C)

    seq = torch.stack(pooled_tokens).unsqueeze(0)  # (1, T, 16, C)
    stats = torch.cat([seq.mean(2), seq.std(2), seq.max(2).values], -1)  # (1, T, 3C)
    logits = head(stats)
    probs = F.softmax(logits, 1).detach().cpu().numpy()[0]
    pred = int(probs.argmax())

    head.zero_grad(set_to_none=True)
    logits[0, pred].backward()

    cams = []
    for feat in feats_per_frame:
        if feat.grad is None:
            cams.append(np.zeros((224, 224), dtype=np.float32)); continue
        weights = feat.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feat).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cams.append(cam.astype(np.float32))

    return np.stack(cams), np.stack(rgb_frames), pred, probs


def overlay(rgb, cam):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (0.5 * rgb + 0.5 * heat).clip(0, 255).astype(np.uint8)


def main():
    os.makedirs(os.path.join(OUT_DIR, 'cams_npz'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'overlays'), exist_ok=True)
    print(f"Device: {DEVICE} | ckpt: {CKPT}")
    backbone, head = load_models(DEVICE)
    clips = list_test_clips_4class()
    print(f"Test clips: {len(clips)} (4-class subset)\n")

    metadata = []
    for clip_idx, vpath, true_label in clips:
        cams, rgb, pred, probs = gradcam_for_clip(vpath, backbone, head, DEVICE)
        if cams is None:
            print(f"  [{clip_idx:2d}] SKIP (no readable frames): {vpath}", flush=True); continue
        outcome = 'correct' if pred == true_label else 'wrong'
        np.savez_compressed(
            os.path.join(OUT_DIR, 'cams_npz', f'clip_{clip_idx:02d}.npz'),
            cams=cams.astype(np.float16))
        mid = T // 2
        png = (f"clip_{clip_idx:02d}_true-{CLASS_NAMES_4[true_label]}"
               f"_pred-{CLASS_NAMES_4[pred]}_{outcome}.png")
        Image.fromarray(overlay(rgb[mid], cams[mid])).save(
            os.path.join(OUT_DIR, 'overlays', png))
        metadata.append({
            'clip_idx': clip_idx, 'video': vpath,
            'true_label': true_label, 'true_class': CLASS_NAMES_4[true_label],
            'pred_label': pred, 'pred_class': CLASS_NAMES_4[pred],
            'pred_prob': float(probs[pred]),
            'all_probs': [float(p) for p in probs],
            'outcome': outcome,
        })
        print(f"  [{clip_idx:2d}/{len(clips)}] {os.path.basename(vpath):40s} "
              f"true={CLASS_NAMES_4[true_label]:12s} pred={CLASS_NAMES_4[pred]:12s} "
              f"({probs[pred]:.2f}) [{outcome}]", flush=True)
        gc.collect()

    with open(os.path.join(OUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    n_wrong = sum(1 for m in metadata if m['outcome'] == 'wrong')
    print(f"\nDone. {len(metadata)} clips, {n_wrong} errors")
    print(f"  CAMs:     {OUT_DIR}/cams_npz/")
    print(f"  Overlays: {OUT_DIR}/overlays/")
    print(f"  Metadata: {OUT_DIR}/metadata.json")


if __name__ == '__main__':
    main()
