"""R48 — CLIP features for iSafetyBench, both arms from ONE decode pass.

Why this file exists: the existing tokens carry appearance as MobileNetV3-small crushed through a
*fixed random projection 576 -> 64*, and the seven-condition ablation showed the signal on this
corpus is entirely crop appearance (`appearance_only` 75.6% vs full 75.2%, p=0.551). So the 64-d
projection, not the relational head, is the suspected ceiling (R48).

Both arms are extracted here together, from the SAME decoded frames and the SAME detector pass:
  - entity arm     : top-8 boxes per frame, 30% context padding -> CLIP ViT-B/32 -> 512-d
  - whole-frame arm: the undetected full frame           -> CLIP ViT-B/32 -> 512-d

Sharing the decode is not only cheaper, it is the fairness control G14 rests on: the baseline cannot
be accused of seeing different frames, a different backbone, or a different preprocessing path.

Frame sampling, crop padding, N_MAX, T, detector and confidence threshold are copied from
`build_relation_tokens.process_clip` so the clip set, the ordering and the folds stay byte-identical
to every run already on the number sheet.

Usage:
  .venv/bin/python build-isafetybench-clip-features.py                 # full 1,100 clips
  .venv/bin/python build-isafetybench-clip-features.py --limit 8       # smoke test
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import build_relation_tokens as B
import build_isafetybench_tokens as ISB
import frame_sampling as FS

ROOT = os.path.dirname(os.path.abspath(__file__))
CLIP_ID = 'openai/clip-vit-base-patch32'
CLIP_DIM = 512
GEOM = 6          # class, xc, yc, w, h, conf — kept beside the embedding, never projected away


def load_clip(device, model_id=None):
    """Frozen CLIP vision tower + its own processor stats. Offline; the model is already cached."""
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    model_id = model_id or CLIP_ID
    m = CLIPVisionModelWithProjection.from_pretrained(model_id).eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    proc = CLIPImageProcessor.from_pretrained(model_id)
    mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1)
    return {'model': m, 'mean': mean, 'std': std, 'device': device}


def to_batch(crops_bgr, clip):
    """BGR uint8 crops -> normalised CLIP input tensor (K,3,224,224).

    The channel swap follows the resize, not precedes it: bit-identical pixels (gated by
    deployment/scripts/profile-r48-preprocess.py and tests/test-r48-preprocess-variants.py), but it
    does not force OpenCV to copy the full-resolution crop to escape a negative-stride view."""
    ims = [cv2.resize(c, (224, 224))[:, :, ::-1].astype('float32') / 255.0 for c in crops_bgr]
    x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).to(clip['device'])
    return (x - clip['mean']) / clip['std']


@torch.no_grad()
def embed(crops_bgr, clip, bs=128):
    """CLIP image embeddings for a list of BGR crops, in batches."""
    if not crops_bgr:
        return np.zeros((0, CLIP_DIM), dtype=np.float32)
    out = []
    for i in range(0, len(crops_bgr), bs):
        e = clip['model'](pixel_values=to_batch(crops_bgr[i:i + bs], clip)).image_embeds
        out.append(e.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def box_px(cx, cy, w, h, W, H):
    """The exact 30%-context box `build_relation_tokens.embed_boxes` uses, in pixels.

    Split out from `crop_box` so the demo can draw the same rectangle it embeds without
    re-deriving the padding rule. A degenerate box (zero area after clamping) returns None and the
    caller falls back to the whole frame, which is what the original single-expression form did."""
    x1 = max(0, int((cx - w / 2 * 1.3) * W)); y1 = max(0, int((cy - h / 2 * 1.3) * H))
    x2 = min(W, int((cx + w / 2 * 1.3) * W)); y2 = min(H, int((cy + h / 2 * 1.3) * H))
    return (x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None


def crop_box(frame, cx, cy, w, h, W, H):
    """The 30%-context crop. Unchanged behaviour; the geometry now lives in `box_px`."""
    b = box_px(cx, cy, w, h, W, H)
    return frame[b[1]:b[3], b[0]:b[2]] if b else frame


def decode(path):
    """T evenly spaced frames, the same indices `process_clip` takes. Shared with the VideoMAE
    builder via frame_sampling so both provably read the same frames."""
    return FS.decode(path, B.T)


def process_clip(path, det, clip, lut, device, wf_only=False):
    """-> (entity (T,N,512), geom (T,N,6), mask (T,N), wholeframe (T,512))."""
    frames, W, H = decode(path)
    ent = np.zeros((B.T, B.N_MAX, CLIP_DIM), dtype=np.float32)
    geo = np.zeros((B.T, B.N_MAX, GEOM), dtype=np.float32)
    msk = np.zeros((B.T, B.N_MAX), dtype=np.float32)
    wf = np.zeros((B.T, CLIP_DIM), dtype=np.float32)
    valid = [(t, fr) for t, fr in enumerate(frames) if fr is not None]
    if not valid:
        return ent, geo, msk, wf

    # whole-frame arm: the undetected frame, same CLIP, same preprocessing
    wf_emb = embed([fr for _, fr in valid], clip)
    for (t, _), e in zip(valid, wf_emb):
        wf[t] = e
    if wf_only:
        # E2 ladder: B1/B1w read only the whole-frame arm, so the detector pass and the 8x larger
        # crop workload buy nothing here. `mask`/`geom` are borrowed from the B/32 file in main()
        # so frame-validity stays byte-identical and the tower is the only variable.
        return ent, geo, msk, wf

    # entity arm: one detector pass over the same frames
    res = det.predict([fr for _, fr in valid], imgsz=B.IMGSZ, conf=B.CONF, verbose=False, device=device)
    crops, slots = [], []
    for (t, fr), r in zip(valid, res):
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xywhn, cls, conf = b.xywhn.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
        for j, oi in enumerate(np.argsort(-conf)[:B.N_MAX]):
            cx, cy, w, h = xywhn[oi]
            geo[t, j] = [lut[int(np.clip(cls[oi], 0, len(lut) - 1))], cx, cy, w, h, float(conf[oi])]
            msk[t, j] = 1.0
            crops.append(crop_box(fr, cx, cy, w, h, W, H)); slots.append((t, j))
    for (t, j), e in zip(slots, embed(crops, clip)):
        ent[t, j] = e
    return ent, geo, msk, wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='cap clips (smoke test)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--detector', default=os.path.join(ROOT, 'yolo11s.pt'))
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--out_tag', default='isbclip_')
    ap.add_argument('--class_map', default='coco', choices=['coco', 'identity'],
                    help="geom class channel. 'coco' maps the 80 COCO ids onto 8 coarse groups and "
                         "is correct for COCO-pretrained detectors. 'identity' passes the detector's "
                         "own class id through unchanged — required for the fine-tuned 8-class "
                         "factory detectors, whose ids are NOT COCO ids (G20 arm B).")
    ap.add_argument('--backbone', default=CLIP_ID,
                    help='frozen vision tower. The E2 ladder changes ONLY this.')
    ap.add_argument('--wholeframe_only', action='store_true',
                    help='extract the whole-frame arm only (what B1/B1w read) and borrow '
                         'mask/geom from --mask_from, so frame-validity is byte-identical')
    ap.add_argument('--mask_from', default=None,
                    help='feature file to copy mask/geom from (default: the B/32 headline file)')
    args = ap.parse_args()

    global CLIP_DIM
    B.IMGSZ, B.CONF, B.CORRUPT = args.imgsz, args.conf, None
    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    clip = load_clip(device, args.backbone)
    CLIP_DIM = int(clip['model'].config.projection_dim)
    det = None if args.wholeframe_only else YOLO(args.detector)
    lut = (ISB.coco_to_group() if args.class_map == 'coco' or det is None
           else np.arange(len(det.names), dtype=np.int64))
    items = ISB.clips()[:args.limit] if args.limit else ISB.clips()
    print(f'[clip-feat] {len(items)} clips | '
          f"detector={'(skipped)' if det is None else os.path.basename(args.detector)} "

          f'device={device} backbone={args.backbone} dim={CLIP_DIM} T={B.T} '
          f'N_MAX={B.N_MAX} wholeframe_only={args.wholeframe_only}', flush=True)

    E, G, M, WF, labels, cids, srcs = [], [], [], [], [], [], []
    t0 = time.time()
    for k, (path, label, src) in enumerate(items):
        e, g, m, w = process_clip(path, det, clip, lut, device, args.wholeframe_only)
        E.append(e); G.append(g); M.append(m); WF.append(w)
        labels.append(label); cids.append(os.path.basename(path)); srcs.append(src)
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(f'  [clip-feat] {k+1}/{len(items)}  {el/(k+1):.2f}s/clip  '
                  f'eta {(len(items)-k-1)*el/(k+1)/60:.1f} min', flush=True)

    out = {'entity': torch.tensor(np.stack(E)), 'geom': torch.tensor(np.stack(G)),
           'mask': torch.tensor(np.stack(M)), 'wholeframe': torch.tensor(np.stack(WF)),
           'labels': torch.tensor(labels), 'clip_ids': cids, 'views': srcs,
           'geom_layout': ['class', 'xc', 'yc', 'w', 'h', 'conf'],
           'entity_names': (ISB.GROUP_NAMES if args.class_map == 'coco' or det is None
                            else [det.names[i] for i in range(len(det.names))]),
           'backbone': args.backbone, 'dataset': 'isafetybench',
           'wholeframe_only': args.wholeframe_only,
           # Provenance, added 2026-10-10: no stored feature or token file recorded which detector
           # built it, so establishing that rested on log archaeology. Written into the artifact now.
           'meta': {'detector': os.path.relpath(os.path.abspath(args.detector), ROOT),
                    'detector_nc': (None if det is None else len(det.names)),
                    'class_map': args.class_map,
                    'imgsz': B.IMGSZ, 'conf': B.CONF, 'T': B.T, 'n_max': B.N_MAX,
                    'argv': ' '.join(sys.argv)}}
    if args.wholeframe_only:
        # B1/B1w read the whole-frame arm plus `mask` (frame validity). `parts_for` silently falls
        # back to all-ones when `mask` is absent, which would let the new tower see MORE frames than
        # the B/32 headline -- a second variable. Borrow the mask instead, keyed by clip_id.
        ref = torch.load(args.mask_from or os.path.join(B.OUT, 'feats_isbclip_all.pt'),
                         map_location='cpu', weights_only=False)
        pos = {str(c): i for i, c in enumerate(ref['clip_ids'])}
        perm = torch.as_tensor([pos[str(c)] for c in cids])
        if not torch.equal(torch.as_tensor(ref['labels']).long()[perm], out['labels'].long()):
            raise SystemExit('label mismatch vs --mask_from -- refusing to borrow a mask')
        out['mask'] = torch.as_tensor(ref['mask']).float()[perm]
        out['geom'] = torch.as_tensor(ref['geom']).float()[perm]
        out['entity'] = torch.zeros(0)      # never written here; dropping it saves ~430 MB of zeros
        out['mask_borrowed_from'] = os.path.basename(args.mask_from or 'feats_isbclip_all.pt')
        print(f"  [clip-feat] mask/geom borrowed from {out['mask_borrowed_from']}: "
              f"{int(out['mask'].sum())} detections, frame-validity identical to B/32", flush=True)
    os.makedirs(B.OUT, exist_ok=True)
    path = os.path.join(B.OUT, f'feats_{args.out_tag}all.pt')
    torch.save(out, path)
    print(f"[clip-feat] -> {os.path.relpath(path, ROOT)} | entity {tuple(out['entity'].shape)} "
          f"wholeframe {tuple(out['wholeframe'].shape)} | detections={int(out['mask'].sum())} "
          f"| hazard={int(sum(labels))} normal={len(items)-int(sum(labels))} "
          f"| sources={len(set(srcs))} | {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
