"""G16 — frozen VideoMAE-base clip features for iSafetyBench (the B2 opponent).

B2 exists to argue against us. ESTA decomposes a clip into detected entities; a video-pretrained
backbone models the clip directly, with no detector at all. If B2 matches or beats ESTA, the
detector is not earning its place on accuracy grounds even if G14 passes, and the pre-registration
says so.

Frames come from `frame_sampling.decode`, the same sampler the CLIP builder uses, so B2 and ESTA see
the same 16 frames of the same 1,100 clips. VideoMAE tubelets span 2 frames, so 16 frames produce 8
temporal states; each state is the mean over its 196 patch tokens, giving (8, 768) per clip, which
feeds the same temporal Transformer B1 uses.

Usage:
  .venv/bin/python build-isafetybench-videomae-features.py            # full 1,100 clips
  .venv/bin/python build-isafetybench-videomae-features.py --limit 8  # smoke test
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch

import build_relation_tokens as B
import build_isafetybench_tokens as ISB
import frame_sampling as FS

# Kinetics-400 fine-tuned, not the self-supervised checkpoint: an MAE encoder that has never been
# trained on a labelled action task is a weak opponent, and beating a weak opponent proves nothing.
# --backbone MCG-NJU/videomae-base selects the SSL variant if the contrast is ever wanted.
VMAE_ID = 'MCG-NJU/videomae-base-finetuned-kinetics'
VMAE_DIM = 768
PATCHES = 14 * 14        # 224/16 per side
OUT = 'sweep_results/relation_tokens/feats_isbvmae_all.pt'


def load_backbone(device, model_id=VMAE_ID):
    from transformers import VideoMAEImageProcessor, VideoMAEModel
    m = VideoMAEModel.from_pretrained(model_id).eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    proc = VideoMAEImageProcessor.from_pretrained(model_id)
    mean = torch.tensor(proc.image_mean, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=device).view(1, 1, 3, 1, 1)
    return {'model': m, 'mean': mean, 'std': std, 'device': device}


def to_video(frames, device):
    """BGR frames -> (1, T, 3, 224, 224) normalised. A dropped frame repeats the last good one."""
    last = np.zeros((224, 224, 3), dtype=np.float32)
    out = []
    for fr in frames:
        if fr is not None:
            last = cv2.resize(fr[:, :, ::-1], (224, 224)).astype('float32') / 255.0
        out.append(last)
    v = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    return v


@torch.no_grad()
def embed(frames, bb):
    v = (to_video(frames, bb['device']) - bb['mean']) / bb['std']
    h = bb['model'](pixel_values=v).last_hidden_state          # (1, T/2 * 196, 768)
    return h.reshape(-1, PATCHES, VMAE_DIM).mean(1).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--backbone', default=VMAE_ID)
    args = ap.parse_args()

    clips = ISB.clips()[:args.limit] if args.limit else ISB.clips()
    bb = load_backbone(args.device, args.backbone)
    print(f"[vmae] {len(clips)} clips | backbone={args.backbone} device={args.device} T={B.T}", flush=True)

    feats, labels, ids, views, t0 = [], [], [], [], time.time()
    for i, (path, label, src) in enumerate(clips):
        frames, _, _ = FS.decode(path, B.T)
        feats.append(embed(frames, bb))
        labels.append(label); ids.append(os.path.basename(path)); views.append(src)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  [vmae] {i+1}/{len(clips)}  {el/(i+1):.2f}s/clip  "
                  f"eta {(len(clips)-i-1)*el/(i+1)/60:.1f} min", flush=True)

    torch.save({'wholeframe': torch.from_numpy(np.stack(feats)),
                'labels': torch.tensor(labels).long(), 'clip_ids': np.array(ids),
                'views': np.array(views), 'backbone': args.backbone, 'dataset': 'isafetybench'}, args.out)
    print(f"[vmae] saved {args.out}  shape={np.stack(feats).shape}")


if __name__ == '__main__':
    main()
