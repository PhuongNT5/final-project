#!/usr/bin/env python3
"""F5 — Resolution ablation: 224² vs 384² for the frozen YOLOv11s + direct4
head pipeline. Extracts 384² features (train + test), trains a single direct4
seed, and compares per-class + overall accuracy against the cached 224²
baseline.

Outputs:
  sweep_results/pairwise/features/all8_train_384.pt
  sweep_results/pairwise/features/all8_test_384.pt
  sweep_results/pairwise/direct4_s314_384.pt
  sweep_results/ablations/resolution_ablation.json + .md
"""
import os, sys, json, importlib.util, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, 'data')
FEAT_DIR = os.path.join(ROOT, 'sweep_results', 'pairwise', 'features')
CKPT_DIR = os.path.join(ROOT, 'sweep_results', 'pairwise')
OUT_DIR = os.path.join(ROOT, 'sweep_results', 'ablations')
OUT_JSON = os.path.join(OUT_DIR, 'resolution_ablation.json')
OUT_MD = os.path.join(OUT_DIR, 'resolution_ablation.md')

CLASS_NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
T_FRAMES = 32
RESOLUTION = 384
SEED = 314


def extract_384_features(subset, dev):
    out_path = os.path.join(FEAT_DIR, f'all8_{subset}_384.pt')
    if os.path.exists(out_path):
        print(f'  cached: {out_path}')
        return torch.load(out_path, map_location='cpu', weights_only=False)

    from ultralytics import YOLO
    yolo = YOLO('yolo11s.pt'); bb = yolo.model
    for p in bb.parameters(): p.requires_grad = False
    bb.eval()
    tf = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    subset_path = os.path.join(DATA_DIR, subset)
    samples = []
    for folder in sorted(os.listdir(subset_path)):
        full = os.path.join(subset_path, folder)
        if not os.path.isdir(full): continue
        label = int(folder.split('_')[0])
        for vf in sorted(os.listdir(full)):
            if vf.lower().endswith(('.mp4', '.avi', '.mov')):
                samples.append((os.path.join(full, vf), label))

    print(f'  {subset} @ {RESOLUTION}²: {len(samples)} videos')
    all_feats, all_labels = [], []
    t0 = time.time()
    for vi, (vp, label) in enumerate(samples):
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release(); all_feats.append(torch.zeros(T_FRAMES, 16, 512))
            all_labels.append(label); continue
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
        all_feats.append(torch.stack(toks[:T_FRAMES])); all_labels.append(label)
        if (vi + 1) % 25 == 0 or vi + 1 == len(samples):
            dt = time.time() - t0
            eta = dt / (vi + 1) * (len(samples) - vi - 1)
            print(f'    {vi + 1}/{len(samples)}  ({dt:.0f}s elapsed, eta {eta:.0f}s)',
                  flush=True)
    feats = torch.stack(all_feats)
    labels = torch.tensor(all_labels, dtype=torch.long)
    print(f'  {subset}: {feats.shape}')
    os.makedirs(FEAT_DIR, exist_ok=True)
    torch.save({'features': feats, 'labels': labels}, out_path)
    return {'features': feats, 'labels': labels}


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}')

    print(f'\n[1/3] Extracting {RESOLUTION}² features...')
    train = extract_384_features('train', dev)
    test = extract_384_features('test', dev)

    spec = importlib.util.spec_from_file_location(
        'pw', os.path.join(ROOT, 'run_pairwise_contrastive.py'))
    pw = importlib.util.module_from_spec(spec); spec.loader.exec_module(pw)

    # Filter to 4-class unsafe subset (labels 0..3)
    tr_mask = (train['labels'] < 4)
    te_mask = (test['labels'] < 4)
    f_tr = pw.stats_pool(train['features'][tr_mask]) if hasattr(pw, 'stats_pool') \
        else torch.cat([train['features'][tr_mask].mean(2),
                        train['features'][tr_mask].std(2),
                        train['features'][tr_mask].max(2).values], -1)
    y_tr = train['labels'][tr_mask]
    f_te = torch.cat([test['features'][te_mask].mean(2),
                      test['features'][te_mask].std(2),
                      test['features'][te_mask].max(2).values], -1)
    y_te = test['labels'][te_mask]
    sd = f_tr.shape[-1]
    print(f'  4cls train: {f_tr.shape}  test: {f_te.shape}')

    print(f'\n[2/3] Training direct4 head @ {RESOLUTION}², seed={SEED}...')
    torch.manual_seed(SEED); np.random.seed(SEED)
    ds = pw.FeatDS(f_tr, y_tr, aug=True)
    model = pw.TransformerHead(sd, 4, 384, 8, 6, 0.3).to(dev)
    sp = os.path.join(CKPT_DIR, f'direct4_s{SEED}_384.pt')
    pw.train_classifier(model, ds, f_te, y_te, lr=1e-4, device=dev, sp=sp,
                        epochs=300, patience=60)
    model.load_state_dict(torch.load(sp, map_location=dev, weights_only=True))
    model.eval()

    print(f'\n[3/3] Evaluating @ {RESOLUTION}²...')
    with torch.no_grad():
        logits = model(f_te.to(dev)).cpu()
    pred = logits.argmax(1).numpy(); y = y_te.numpy()
    correct = int((pred == y).sum())
    per_class = {}
    for c in range(4):
        m = y == c; n = int(m.sum())
        per_class[CLASS_NAMES[c]] = {
            'n': n, 'correct': int((pred[m] == c).sum()),
            'acc': round(float((pred[m] == c).mean()) if n else 0.0, 4)}
    macro = round(float(np.mean([per_class[c]['acc'] for c in CLASS_NAMES])), 4)

    # Load 224² baseline direct4 from cached checkpoint for head-to-head
    base_sp = os.path.join(CKPT_DIR, 'direct4_s314.pt')
    base_feat = torch.load(os.path.join(CKPT_DIR, 'features', 'all8_test.pt'),
                           map_location='cpu', weights_only=False)
    bm4 = base_feat['labels'] < 4
    bf = torch.cat([base_feat['features'][bm4].mean(2),
                    base_feat['features'][bm4].std(2),
                    base_feat['features'][bm4].max(2).values], -1)
    by = base_feat['labels'][bm4].numpy()
    base = pw.TransformerHead(bf.shape[-1], 4, 384, 8, 6, 0.3).to(dev)
    base.load_state_dict(torch.load(base_sp, map_location=dev, weights_only=True))
    base.eval()
    with torch.no_grad():
        bl = base(bf.to(dev)).cpu()
    bp = bl.argmax(1).numpy()
    base_correct = int((bp == by).sum())
    base_per = {}
    for c in range(4):
        m = by == c; n = int(m.sum())
        base_per[CLASS_NAMES[c]] = {
            'n': n, 'correct': int((bp[m] == c).sum()),
            'acc': round(float((bp[m] == c).mean()) if n else 0.0, 4)}
    base_macro = round(float(np.mean([base_per[c]['acc'] for c in CLASS_NAMES])), 4)

    summary = {
        'seed': SEED, 'frames': T_FRAMES,
        'configurations': {
            '224x224 (baseline)': {'top1': base_correct, 'total': len(by),
                                   'top1_pct': round(100 * base_correct / len(by), 2),
                                   'macro': base_macro, 'per_class': base_per},
            f'{RESOLUTION}x{RESOLUTION}': {'top1': correct, 'total': len(y),
                                           'top1_pct': round(100 * correct / len(y), 2),
                                           'macro': macro, 'per_class': per_class},
        },
        'delta_top1_pp': round(100 * (correct / len(y) - base_correct / len(by)), 2),
        'delta_macro_pp': round(100 * (macro - base_macro), 2),
        'note': 'direct4 single-head only (no cascade, no ensemble); fair '
                'isolation of the resolution effect at the dominant '
                'computational stage.',
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, 'w') as f: json.dump(summary, f, indent=2)
    print(f'\n[SAVED] {OUT_JSON}')

    md = ['# Resolution ablation — direct4 single-head, seed=314\n',
          '| Resolution | Top-1 | Top-1 % | Macro | Walk | Unauth | Panel | Forklift |',
          '|------------|------:|--------:|------:|-----:|-------:|------:|---------:|']
    for k, r in summary['configurations'].items():
        pc = r['per_class']
        md.append(f"| **{k}** | {r['top1']}/{r['total']} | {r['top1_pct']:.2f} | "
                  f"{r['macro']:.4f} | {pc['WalkwayViol']['acc']:.3f} | "
                  f"{pc['UnauthorInt']['acc']:.3f} | {pc['OpenedPanel']['acc']:.3f} | "
                  f"{pc['CarryOverload']['acc']:.3f} |")
    md.append(f"\nΔ top-1 (384 − 224): {summary['delta_top1_pp']:+.2f} pp")
    md.append(f"Δ macro (384 − 224): {summary['delta_macro_pp']:+.2f} pp")
    with open(OUT_MD, 'w') as f: f.write('\n'.join(md) + '\n')
    print(f'[SAVED] {OUT_MD}')

    print(f"\n224²: {base_correct}/{len(by)} ({100 * base_correct / len(by):.2f}% top-1, "
          f"macro {base_macro:.3f})")
    print(f"384²: {correct}/{len(y)} ({100 * correct / len(y):.2f}% top-1, "
          f"macro {macro:.3f})")
    print(f"Δ: {summary['delta_top1_pp']:+.2f} pp top-1, "
          f"{summary['delta_macro_pp']:+.2f} pp macro")


if __name__ == '__main__':
    main()
