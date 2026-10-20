#!/usr/bin/env python3
"""STAGE 2 on Kaggle: run frozen vision towers over the cached frames. GPU, minutes per tower.

What this does and does NOT do
------------------------------
It produces ONLY the whole-frame arm, the arm B1 actually reads, in the published clip order, and
saves one file per tower. It trains nothing. Head training stays on the local machine at
OMP_NUM_THREADS=4, because that is the only environment in which a row is comparable to the 83.25%
four-thread control (section 4.5.6): moving training to a GPU would change the reduction order, the
checkpoint selected by the saturated val_macro, and therefore the number -- a second drift source
on top of the one the thesis already measures. So Kaggle supplies representations; the local
machine supplies every reported margin.

The control row is mandatory, not optional
------------------------------------------
`clipB32_control` re-extracts the published backbone from the Kaggle-decoded cache. Locally,
`verify-and-install-tower-features.py` compares it against feats_isbclip_all.pt. If it matches, the
Kaggle decode path is proven not to shift the input side and every other tower here is paired with
the published control on identical pixels. If it does not match, the difference is itself the
measurement, and no tower row may be reported as a clean tower effect until it is explained.
Extract it FIRST, verify it, and only then spend the session on the rest.

Normalisation is fp32 by default. --amp halves the time but introduces a second numeric drift
source; the thesis already spends a section on one, so the default stays off.

Usage:
  pip install -q open_clip_torch                      # only for the open_clip-backend towers
  python kaggle-extract-tower-features.py --cache /kaggle/working/cache \
      --manifest /kaggle/input/isb-manifest/isb-clip-manifest.json --towers clipB32_control
  python kaggle-extract-tower-features.py --list
"""
import argparse
import json
import os
import time

import numpy as np
import torch

# backend 'hf' -> transformers AutoModel + get_image_features (CLIPModel and SiglipModel both
# expose it, so the projected embedding is taken the same way the published B/32 file took it).
# backend 'open_clip' -> open_clip.encode_image, for checkpoints released only in that format.
TOWERS = {
    'clipB32_control': ('hf', 'openai/clip-vit-base-patch32'),
    'clipL14_control': ('hf', 'openai/clip-vit-large-patch14'),
    'clipB16':         ('hf', 'openai/clip-vit-base-patch16'),
    # measured 16/10: both LAION repos ship an open_clip-style config.json with no `model_type`,
    # so AutoModel refuses them. They are open_clip checkpoints, not HF-format ones.
    'openclipB16_dc1b': ('open_clip', 'hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K'),
    'openclipB16_l2b': ('open_clip', 'hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K'),
    'metaclipB16':     ('hf', 'facebook/metaclip-b16-400m'),
    'siglip2B16':      ('hf', 'google/siglip2-base-patch16-224'),
    'dfn2bB16':        ('open_clip', 'hf-hub:apple/DFN2B-CLIP-ViT-B-16'),
    'mobileclip_s1':   ('open_clip', 'hf-hub:apple/MobileCLIP-S1-OpenCLIP'),
    # measured 16/10: the timm repo returns 404 for open_clip_config.json, so the hf-hub
    # form cannot load it. open_clip carries it under its own name + pretrained tag.
    'eva02B16':        ('open_clip', 'EVA02-B-16|merged2b_s8b_b131k'),
}


def _as_tensor(r):
    """transformers 4.x returns the projected embedding as a tensor from get_image_features;
    5.x returns a ModelOutput whose `pooler_output` carries it. Kaggle's pin is not ours, so both
    shapes are accepted and the PROJECTED embedding -- the one the published B/32 file stores -- is
    the one taken in either case."""
    if hasattr(r, 'shape'):
        return r
    for k in ('image_embeds', 'pooler_output'):
        if getattr(r, k, None) is not None:
            return getattr(r, k)
    raise TypeError(f'cannot find an embedding in {type(r).__name__}: {list(r.keys())}')


def load_hf(model_id, device):
    from transformers import AutoImageProcessor, AutoModel
    m = AutoModel.from_pretrained(model_id).eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    proc = AutoImageProcessor.from_pretrained(model_id)
    mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1)
    return (lambda x: _as_tensor(m.get_image_features(pixel_values=x))), mean, std


def load_open_clip(model_id, device):
    import open_clip
    from torchvision.transforms import Normalize
    # 'name|pretrained' for checkpoints whose HF repo carries no open_clip_config.json
    name, _, tag = model_id.partition('|')
    m, _, pre = open_clip.create_model_and_transforms(name, pretrained=tag or None)
    m = m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    norm = [t for t in pre.transforms if isinstance(t, Normalize)]
    assert norm, f'{model_id}: no Normalize in its own preprocess; cannot match its training stats'
    mean = torch.tensor(norm[0].mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(norm[0].std, device=device).view(1, 3, 1, 1)
    return (lambda x: _as_tensor(m.encode_image(x))), mean, std


@torch.no_grad()
def embed_all(fwd, mean, std, frames, bs, device, amp):
    """frames (n,T,224,224,3) uint8 RGB -> (n,T,D) float32. Same arithmetic as to_batch: /255 then
    (x-mean)/std, on pixels already resized and channel-swapped by stage 1."""
    n, t = frames.shape[0], frames.shape[1]
    flat, out, D = n * t, None, None
    t0 = time.time()
    for i in range(0, flat, bs):
        sl = slice(i, min(i + bs, flat))
        # np.array, not asarray: the cache is opened read-only as a memmap and torch
        # refuses to wrap a non-writable buffer without warning about it.
        chunk = np.array(frames.reshape(flat, *frames.shape[2:])[sl])
        x = torch.from_numpy(chunk).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - mean) / std
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            e = fwd(x)
        e = e.float().cpu()
        if out is None:
            D = e.shape[-1]
            out = torch.zeros(flat, D, dtype=torch.float32)
        out[sl] = e
        if (i // bs) % 20 == 0:
            done = sl.stop
            print(f'    {done}/{flat} frames  {(time.time()-t0)/max(done,1)*1000:.1f} ms/frame',
                  flush=True)
    return out.view(n, t, D)


def run_tower(name, cache, man, out_dir, bs, device, amp):
    backend, model_id = TOWERS[name]
    frames = np.load(f'{cache}/frames.npy', mmap_mode='r')
    valid = np.load(f'{cache}/valid.npy')
    n = frames.shape[0]
    fwd, mean, std = (load_hf if backend == 'hf' else load_open_clip)(model_id, device)
    t0 = time.time()
    wf = embed_all(fwd, mean, std, frames, bs, device, amp)
    # Dead frames carry zeros, exactly as the local builder leaves them, so frame validity is
    # identical across towers and the temporal mask downstream stays the same object.
    wf[torch.from_numpy(valid) == 0] = 0.0
    res = {'wholeframe': wf,
           'labels': torch.tensor(man['labels'][:n]),
           'clip_ids': man['clip_ids'][:n], 'views': man['views'][:n],
           'valid': torch.from_numpy(valid).float(),
           'backbone': model_id, 'backend': backend, 'dataset': 'isafetybench',
           'wholeframe_only': True,
           'meta': {'built_on': 'kaggle', 'amp_fp16': bool(amp), 'batch': bs,
                    'order_sha256': man['order_sha256'], 'dim': int(wf.shape[-1]),
                    'minutes': round((time.time() - t0) / 60, 1),
                    'torch': torch.__version__,
                    'cache_meta': json.load(open(f'{cache}/cache-meta.json'))}}
    os.makedirs(out_dir, exist_ok=True)
    path = f'{out_dir}/feats_{name}_all.pt'
    torch.save(res, path)
    mb = os.path.getsize(path) / 1e6
    print(f'[{name}] {model_id} dim={wf.shape[-1]} '
          f'{res["meta"]["minutes"]} min -> {path} ({mb:.0f} MB)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/kaggle/working/cache')
    ap.add_argument('--manifest', required=False)
    ap.add_argument('--out', default='/kaggle/working/feats')
    ap.add_argument('--towers', nargs='*', default=['clipB32_control'])
    ap.add_argument('--bs', type=int, default=64,
                    help='LEAVE AT 64. Measured 2026-10-16 on the 1,100-clip cache: at 64 '
                         'the B/32 re-extraction is bit-identical to the published tensor '
                         '(max|diff| = 0), at 32 it differs by 1.5e-05. Batch size changes '
                         'the GEMM reduction order, so it is a drift source of the same '
                         'order as the CPU thread count section 4.5.6 measures -- and that '
                         'one moves balanced accuracy by about 1 pp through the saturated '
                         'val_macro checkpoint rule.')
    ap.add_argument('--amp', action='store_true', help='fp16 autocast; adds a numeric drift source')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    if args.list:
        for k, (b, m) in TOWERS.items():
            print(f'{k:20s} {b:10s} {m}')
        return
    man = json.load(open(args.manifest))
    # Apple Silicon is offered too: the local machine ran the published extraction on mps,
    # and a B/16 tower over 17,600 frames is minutes there, so Kaggle is a convenience for
    # running many towers at once, not a requirement for running one.
    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'[towers] {len(args.towers)} towers on {device}, bs={args.bs}, amp={args.amp}, '
          f'{man["n"]} clips x {man["T"]} frames', flush=True)
    for name in args.towers:
        assert name in TOWERS, f'{name} not in TOWERS; --list to see them'
        run_tower(name, args.cache, man, args.out, args.bs, device, args.amp)


if __name__ == '__main__':
    main()
