"""CLIP few-shot transfer stream (T2's ranked-1 upgrade for a 452-clip training set).

Rationale: full fine-tuning of VideoMAE / TimeSformer / Video Swin needs thousands of clips; a frozen
CLIP image tower with a light temporal head is validated down to 8 shots per class, and we have ~113.
Two of our four behaviours (OpenedPanel, CarryOverload) are object/scene-state rather than pose, which
is exactly what CLIP's visual features already separate.

Emits `frameout_clip_{split}.pt` in the same schema as the trained frame recognisers, so the existing
`agg` and `temporal` stages consume it with `--tag clip` and no further changes.  `logits` here are
zero-shot text-prompt similarities - a genuinely training-free signal - and `feats` are the pooled
image embeddings the temporal heads learn on.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import torch

from . import config as C
from .data_index import load_clips
from .frame_cache import frame_paths

MODEL = 'openai/clip-vit-base-patch32'
PROMPTS = [
    'a worker walking in a restricted area outside the safety walkway',
    'an unauthorized worker intervening with industrial machinery',
    'an electrical control panel with its cover opened',
    'a forklift carrying an unsafe overloaded pallet',
    'a worker walking safely inside the marked pedestrian walkway',
    'an authorized technician servicing machinery with permission',
    'an electrical control panel with its cover closed',
    'a forklift carrying a safe normal load',
]


def _vec(out):
    """transformers 5.x returns an output object here, 4.x a bare tensor."""
    return out if torch.is_tensor(out) else out.pooler_output


def _encode(model, proc, paths, device, bs=32):
    from PIL import Image
    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert('RGB') for p in paths[i:i + bs]]
        px = proc(images=ims, return_tensors='pt')['pixel_values'].to(device)
        with torch.no_grad():
            f = _vec(model.get_image_features(pixel_values=px))
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu())
    return torch.cat(out)


def build(subsets=('train', 'test'), device='mps', tag='clip', t=C.T):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    with torch.no_grad():
        tin = proc(text=PROMPTS, return_tensors='pt', padding=True).to(device)
        W = torch.nn.functional.normalize(_vec(model.get_text_features(**tin)), dim=-1).cpu()
    for sp in subsets:
        out = os.path.join(C.OUT_DIR, f'frameout_{tag}_{sp}.pt')
        if os.path.exists(out):
            print(f'[clip] {sp}: cached', flush=True)
            continue
        clips = load_clips([sp])
        F = torch.zeros(len(clips), t, W.shape[1])
        for i, c in enumerate(clips):
            F[i] = _encode(model, proc, frame_paths(c, t), device)
            if (i + 1) % 100 == 0:
                print(f'  [{sp}] {i+1}/{len(clips)}', flush=True)
        L = F @ W.T * 100.0                                  # CLIP logit scale; zero-shot 8-way
        torch.save({'names': [c.name for c in clips], 'labels': [c.label for c in clips],
                    'views': [c.view for c in clips], 'logits': L.half(), 'feats': F.half()},
                   out)
        print(f'[clip] {sp}: feats={tuple(F.shape)} logits={tuple(L.shape)}', flush=True)


def zero_shot(tag='clip'):
    """Training-free baseline: no labels used at all. A useful floor for the thesis."""
    for sp in ('train', 'test'):
        d = torch.load(os.path.join(C.OUT_DIR, f'frameout_{tag}_{sp}.pt'), weights_only=False)
        y = np.array(d['labels']); keep = y < 4
        L = d['logits'].float()[keep].mean(1)[:, :4]         # temporal mean-pool, unsafe head
        p = L.argmax(-1).numpy(); yk = y[keep]
        mac = np.mean([(p[yk == k] == k).mean() for k in range(4) if (yk == k).any()])
        print(f'[clip zero-shot] {sp}: top1={ (p==yk).mean()*100:.1f} macro={mac*100:.1f} '
              f'({(p==yk).sum()}/{len(yk)})', flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--device', default='mps'); a.add_argument('--tag', default='clip')
    a.add_argument('--zero-shot', action='store_true')
    v = a.parse_args()
    build(device=v.device, tag=v.tag)
    zero_shot(tag=v.tag)
