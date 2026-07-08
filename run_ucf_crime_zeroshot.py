#!/usr/bin/env python3
"""F4 — Cross-dataset zero-shot evaluation: apply the Unsafe-Net 4-class
ensemble (direct4_s314 + scenario × pair{0..3} cascade) to UCF-Crime
anomaly clips. Reports prediction distribution, mean confidence, and
the fraction of clips routed to UnauthorInt (semantic closest match).

Output: sweep_results/ablations/ucf_crime_zeroshot.json + .md
"""
import os, sys, json, importlib.util, glob
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'data', 'ucf_crime', 'Anomaly')
CKPT = os.path.join(ROOT, 'sweep_results', 'pairwise')
OUT_DIR = os.path.join(ROOT, 'sweep_results', 'ablations')
OUT_JSON = os.path.join(OUT_DIR, 'ucf_crime_zeroshot.json')
OUT_MD = os.path.join(OUT_DIR, 'ucf_crime_zeroshot.md')

CLASS_NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
PAIR_NAMES = ['Walkway', 'Intervention', 'Panel', 'Forklift']
T_FRAMES = 32


def load_TH():
    spec = importlib.util.spec_from_file_location(
        'pw', os.path.join(ROOT, 'run_pairwise_contrastive.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.TransformerHead


def extract_features(video_paths, device):
    """Extract YOLOv11s layer-22 features per clip (T,16,512)."""
    from ultralytics import YOLO
    yolo = YOLO('yolo11s.pt')
    bb = yolo.model
    for p in bb.parameters(): p.requires_grad = False
    bb.eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])])
    feats = []
    for vi, vp in enumerate(video_paths):
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release(); feats.append(torch.zeros(T_FRAMES, 16, 512)); continue
        idx = np.linspace(0, total - 1, T_FRAMES, dtype=int)
        toks = []
        for j in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
            ok, frame = cap.read()
            if not ok: toks.append(torch.zeros(16, 512)); continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = tf(img).unsqueeze(0)
            captured = None
            def hook(_m, _i, o):
                nonlocal captured; captured = o
            h = bb.model[22].register_forward_hook(hook)
            with torch.no_grad(): bb(tensor)
            h.remove()
            f = captured[0] if isinstance(captured, (list, tuple)) else captured
            fh, fw = f.shape[2], f.shape[3]
            if fh % 4 or fw % 4:
                f = F.interpolate(f, size=(max(4, (fh // 4) * 4), max(4, (fw // 4) * 4)),
                                  mode='bilinear', align_corners=False)
            pooled = F.adaptive_avg_pool2d(f, (4, 4))
            toks.append(pooled.flatten(2).transpose(1, 2).squeeze(0))
        cap.release()
        while len(toks) < T_FRAMES: toks.append(toks[-1])
        feats.append(torch.stack(toks[:T_FRAMES]))
        if (vi + 1) % 5 == 0 or vi + 1 == len(video_paths):
            print(f'  feat {vi + 1}/{len(video_paths)}', flush=True)
    return torch.stack(feats)


def stats_pool(f):
    return torch.cat([f.mean(2), f.std(2), f.max(2).values], -1)


def load_head(p, nc, dim, nh, L, sd, dev, TH):
    m = TH(sd, nc, dim, nh, L, 0.3).to(dev)
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=True))
    m.eval(); return m


@torch.no_grad()
def head_logits(model, x, dev, bs=32):
    out = []
    for i in range(0, len(x), bs):
        out.append(model(x[i:i + bs].to(dev)).cpu())
    return torch.cat(out, 0).numpy()


def main():
    if not os.path.isdir(DATA):
        sys.exit(f'UCF-Crime data not found at {DATA}')
    classes = sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))
    clips = []
    for c in classes:
        for f in sorted(glob.glob(os.path.join(DATA, c, '*.mp4'))):
            clips.append((c, f))
    print(f'UCF-Crime classes: {classes}')
    print(f'Total clips: {len(clips)} ({Counter(c for c, _ in clips)})')
    if not clips: sys.exit('no clips found')

    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}')

    print('\n[1/3] Extracting frozen YOLOv11s features...')
    paths = [p for _, p in clips]
    feats = extract_features(paths, dev)
    s = stats_pool(feats); sd = s.shape[-1]
    print(f'  features: {tuple(feats.shape)} -> stats: {tuple(s.shape)}')

    print('\n[2/3] Loading 4-class ensemble heads...')
    TH = load_TH()
    direct4 = load_head(os.path.join(CKPT, 'direct4_s314.pt'), 4, 384, 8, 6, sd, dev, TH)
    scenario = load_head(os.path.join(CKPT, 'scenario_s7.pt'), 4, 384, 8, 6, sd, dev, TH)
    pairs = [load_head(os.path.join(CKPT, f'pair{i}_{n}_s7.pt'), 2, 192, 6, 3, sd, dev, TH)
             for i, n in enumerate(PAIR_NAMES)]

    print('\n[3/3] Running ensemble...')
    Ld = head_logits(direct4, s, dev)
    Lsc = head_logits(scenario, s, dev)
    pair_log = np.stack([head_logits(p, s, dev) for p in pairs], 1)
    Pd = F.softmax(torch.from_numpy(Ld), 1).numpy()
    Psc = F.softmax(torch.from_numpy(Lsc), 1).numpy()
    Pp_unsafe = F.softmax(torch.from_numpy(pair_log), 2).numpy()[:, :, 1]
    cas = Psc * Pp_unsafe; cas = cas / (cas.sum(1, keepdims=True) + 1e-8)
    ens = (Pd + cas) / 2
    pred = ens.argmax(1)
    maxp = ens.max(1)

    per_class_dist = {CLASS_NAMES[c]: int((pred == c).sum()) for c in range(4)}
    by_ucf = {}
    for c in classes:
        idx = [i for i, (cc, _) in enumerate(clips) if cc == c]
        by_ucf[c] = {
            'n_clips': len(idx),
            'pred_dist': {CLASS_NAMES[cl]: int((pred[idx] == cl).sum()) for cl in range(4)},
            'frac_unauthor_int': round(float((pred[idx] == 1).mean()), 4),
            'mean_max_prob': round(float(maxp[idx].mean()), 4),
            'frac_high_conf': round(float((maxp[idx] > 0.5).mean()), 4),
        }

    summary = {
        'dataset': 'UCF-Crime (Anomaly subset, partial)',
        'ucf_classes_evaluated': classes,
        'total_clips': len(clips),
        'thesis_classes': CLASS_NAMES,
        'overall_pred_dist': per_class_dist,
        'overall_frac_unauthor_int': round(float((pred == 1).mean()), 4),
        'overall_mean_max_prob': round(float(maxp.mean()), 4),
        'overall_frac_high_conf': round(float((maxp > 0.5).mean()), 4),
        'baseline_uniform_25pct': 0.25,
        'in_domain_unsafenet_test_mean_max_prob': None,  # filled below
        'per_ucf_class': by_ucf,
        'note': 'UCF-Crime labels do not map to Unsafe-Net classes; the metric '
                'is structural (prediction distribution + confidence), not top-1.',
    }
    # In-domain baseline confidence for context
    try:
        d = torch.load(os.path.join(CKPT, 'features', 'all8_test.pt'),
                       map_location='cpu', weights_only=False)
        m4 = (d['labels'] < 4).numpy()
        s4 = stats_pool(d['features'])[m4]
        Lt = head_logits(direct4, s4, dev)
        Pt = F.softmax(torch.from_numpy(Lt), 1).numpy()
        summary['in_domain_unsafenet_test_mean_max_prob'] = round(float(Pt.max(1).mean()), 4)
    except Exception as e:
        print(f'  [warn] could not compute in-domain baseline: {e}')

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, 'w') as f: json.dump(summary, f, indent=2)
    print(f'\n[SAVED] {OUT_JSON}')

    md = ['# UCF-Crime cross-dataset zero-shot evaluation\n',
          f'Dataset: {summary["dataset"]}',
          f'Classes evaluated: {classes}  (total {len(clips)} clips)\n',
          '## Overall',
          f"- pred dist: {summary['overall_pred_dist']}",
          f"- frac → UnauthorInt: {summary['overall_frac_unauthor_int']:.3f} (uniform = 0.25)",
          f"- mean max-prob: {summary['overall_mean_max_prob']:.3f}",
          f"- frac high-conf (max-p>0.5): {summary['overall_frac_high_conf']:.3f}",
          f"- in-domain Unsafe-Net mean max-prob: {summary['in_domain_unsafenet_test_mean_max_prob']}\n",
          '## Per UCF-Crime class', '',
          '| UCF class | N | Walk | Unauth | Panel | Forklift | frac→Unauth | mean p | high-conf |',
          '|-----------|--:|-----:|-------:|------:|---------:|------------:|-------:|----------:|']
    for c, r in by_ucf.items():
        d = r['pred_dist']
        md.append(f"| {c} | {r['n_clips']} | {d['WalkwayViol']} | {d['UnauthorInt']} | "
                  f"{d['OpenedPanel']} | {d['CarryOverload']} | {r['frac_unauthor_int']:.3f} | "
                  f"{r['mean_max_prob']:.3f} | {r['frac_high_conf']:.3f} |")
    with open(OUT_MD, 'w') as f: f.write('\n'.join(md) + '\n')
    print(f'[SAVED] {OUT_MD}')

    print(f"\nfrac→UnauthorInt = {summary['overall_frac_unauthor_int']:.3f} "
          f"(uniform=0.25; in-domain UnauthorInt rate ≈ 11/64 = 0.172)")


if __name__ == '__main__':
    main()
