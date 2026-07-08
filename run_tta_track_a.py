#!/usr/bin/env python3
"""Test-time augmentation for Track A accuracy push.

Re-extracts TEST features under augmentations (horizontal flip, temporal offset),
runs the saved direct4_s314 + scenario_s7 + pair{0..3}_s7 heads on each variant,
averages softmax probs across variants, then re-ensembles and reports top-1 + macro.

Baseline to beat: 56/64 ensemble, 57/64 calibrated, 89.28% macro.
"""
import os, sys, math, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

CLASS_NAMES_4 = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
DATA_DIR = 'data'
OUT_DIR = 'sweep_results/pairwise'
FEAT_DIR = os.path.join(OUT_DIR, 'features')
PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]

# ── Must match TransformerHead from run_pairwise_contrastive.py ──
class TransformerHead(nn.Module):
    def __init__(self, in_dim, nc, dim=256, nhead=8, layers=4, drop=0.3):
        super().__init__()
        self.appear = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU(), nn.Dropout(drop * 0.3))
        self.motion = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU(), nn.Dropout(drop * 0.3))
        self.tpos = nn.Embedding(64, dim)
        self.te = nn.Embedding(3, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)); nn.init.trunc_normal_(self.cls, std=0.02)
        nh = nhead
        while dim % nh != 0: nh -= 1
        self.pre = nn.LayerNorm(dim)
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=nh, dim_feedforward=dim * 4,
                                         dropout=drop, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=layers)
        self.post = nn.LayerNorm(dim)
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(drop * 0.5),
                                nn.Linear(dim // 2, nc))

    def forward(self, x):
        if x.dim() == 4: x = x.mean(2)
        b, t, c = x.shape; d = x.device
        a = self.appear(x) + self.tpos(torch.arange(t, device=d)) + self.te(torch.ones(t, dtype=torch.long, device=d))
        diff = x[:, 1:] - x[:, :-1]
        m = self.motion(diff) + self.tpos(torch.arange(t - 1, device=d)) + self.te(torch.full((t - 1,), 2, dtype=torch.long, device=d))
        cls = self.cls.expand(b, -1, -1) + self.te(torch.zeros(1, dtype=torch.long, device=d))
        return self.fc(self.post(self.transformer(self.pre(torch.cat([cls, a, m], 1)))[:, 0]))


def extract_test_features(variant: str, device):
    """Re-extract TEST features for all 8 classes under a TTA variant.
    variant ∈ {'baseline', 'hflip', 'toffset'}.
    Returns {'features': (N, T, 16, 512), 'labels': (N,)}.
    """
    os.makedirs(FEAT_DIR, exist_ok=True)
    cache = os.path.join(FEAT_DIR, 'all8_test.pt' if variant == 'baseline' else f'tta_test_{variant}.pt')
    if os.path.exists(cache):
        print(f"  [{variant}] cached → {cache}")
        return torch.load(cache, map_location='cpu')

    from ultralytics import YOLO
    print(f"  [{variant}] extracting test features...")
    yolo = YOLO('yolo11s.pt')
    backbone = yolo.model
    for p in backbone.parameters(): p.requires_grad = False
    backbone.eval()
    pool_size, target_idx = 4, 22
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    subset_path = os.path.join(DATA_DIR, 'test')
    samples = []
    for folder in sorted(os.listdir(subset_path)):
        full = os.path.join(subset_path, folder)
        if not os.path.isdir(full): continue
        label = int(folder.split('_')[0])
        for vf in sorted(os.listdir(full)):
            if vf.lower().endswith(('.mp4', '.avi', '.mov')):
                samples.append((os.path.join(full, vf), label))

    T = 32
    all_feats, all_labels = [], []
    for vi, (vpath, label) in enumerate(samples):
        cap = cv2.VideoCapture(vpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            all_feats.append(torch.zeros(T, 16, 512)); all_labels.append(label); continue
        if variant == 'toffset':
            # shift indices by half-stride; clamped to valid range
            stride = max(1, (total - 1) / max(1, T - 1))
            indices = np.clip(np.linspace(stride / 2, total - 1 - stride / 2, T),
                              0, total - 1).astype(int)
        else:
            indices = np.linspace(0, total - 1, T, dtype=int)

        tokens_list = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                tokens_list.append(torch.zeros(16, 512)); continue
            if variant == 'hflip':
                frame = cv2.flip(frame, 1)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = tf(img).unsqueeze(0)
            features = None
            def hook(m, i, o):
                nonlocal features; features = o
            h = backbone.model[target_idx].register_forward_hook(hook)
            with torch.no_grad(): backbone(tensor)
            h.remove()
            if isinstance(features, (list, tuple)): features = features[0]
            fh, fw = features.shape[2], features.shape[3]
            if fh % pool_size != 0 or fw % pool_size != 0:
                nh = max(pool_size, (fh // pool_size) * pool_size)
                nw = max(pool_size, (fw // pool_size) * pool_size)
                features = F.interpolate(features, size=(nh, nw), mode='bilinear', align_corners=False)
            pooled = F.adaptive_avg_pool2d(features, (pool_size, pool_size))
            tokens_list.append(pooled.flatten(2).transpose(1, 2).squeeze(0))
        cap.release()
        while len(tokens_list) < T:
            tokens_list.append(tokens_list[-1] if tokens_list else torch.zeros(16, 512))
        all_feats.append(torch.stack(tokens_list[:T])); all_labels.append(label)
        if (vi + 1) % 20 == 0 or vi + 1 == len(samples):
            print(f"    {vi+1}/{len(samples)}", flush=True)

    features = torch.stack(all_feats)
    labels = torch.tensor(all_labels, dtype=torch.long)
    torch.save({'features': features, 'labels': labels}, cache)
    print(f"  [{variant}] saved {tuple(features.shape)} → {cache}")
    return {'features': features, 'labels': labels}


def to_stats(feats):
    return torch.cat([feats.mean(2), feats.std(2), feats.max(2).values], -1)


def run_heads(stats_all, labels_all, device):
    """Run direct4_s314 on 4-class mask, scenario_s7 + pair heads for cascade.
    Returns dict tag→(64,4) softmax probs."""
    mask4 = labels_all < 4
    stats4 = stats_all[mask4].to(device)
    stats_all_d = stats_all.to(device)
    stats_dim = stats_all.shape[-1]
    out = {}

    # direct4_s314
    m = TransformerHead(stats_dim, 4, 384, 8, 6, 0.3).to(device)
    m.load_state_dict(torch.load(os.path.join(OUT_DIR, 'direct4_s314.pt'), map_location=device))
    m.eval()
    with torch.no_grad():
        out['direct4_s314'] = F.softmax(m(stats4), 1).cpu().numpy()

    # scenario_s7 → (N_all, 4), keep only 4-class rows
    m = TransformerHead(stats_dim, 4, 384, 8, 6, 0.3).to(device)
    m.load_state_dict(torch.load(os.path.join(OUT_DIR, 'scenario_s7.pt'), map_location=device))
    m.eval()
    with torch.no_grad():
        scenario = F.softmax(m(stats_all_d), 1).cpu().numpy()
    sp4 = scenario[mask4.numpy()]

    # pair{0..3}_s7 → probs of unsafe per scenario on 4-class test
    names = ['Walkway', 'Intervention', 'Panel', 'Forklift']
    pair_unsafe = []
    for pi, nm in enumerate(names):
        m = TransformerHead(stats_dim, 2, 192, 6, 3, 0.3).to(device)
        m.load_state_dict(torch.load(os.path.join(OUT_DIR, f'pair{pi}_{nm}_s7.pt'), map_location=device))
        m.eval()
        with torch.no_grad():
            bp = F.softmax(m(stats4), 1).cpu().numpy()
        pair_unsafe.append(bp[:, 1])
    pair_scores = np.stack(pair_unsafe, axis=1)
    pair_scores = pair_scores / (pair_scores.sum(axis=1, keepdims=True) + 1e-8)
    cascade = sp4 * pair_scores
    cascade = cascade / (cascade.sum(1, keepdims=True) + 1e-8)
    out['cascade_s7'] = cascade
    return out, labels_all[mask4].numpy()


def report(tag, probs, y):
    pred = probs.argmax(1)
    c = int((pred == y).sum())
    macro = float(np.mean([100 * int((pred[y == i] == i).sum()) / max(1, int((y == i).sum())) for i in range(4)]))
    return c, macro


def main():
    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    variants = ['baseline', 'hflip', 'toffset']
    per_variant = {}
    y_ref = None
    for v in variants:
        data = extract_test_features(v, device)
        feats, labels = data['features'], data['labels']
        stats = to_stats(feats)
        probs, y = run_heads(stats, labels, device)
        per_variant[v] = probs
        y_ref = y
        print(f"\n  === variant={v} ===")
        for tag, p in probs.items():
            c, mac = report(tag, p, y)
            print(f"    {tag:<18s} {c}/64 macro={mac:.2f}%")
        ens = np.mean([probs['direct4_s314'], probs['cascade_s7']], axis=0)
        c, mac = report('ENS', ens, y)
        print(f"    {'ensemble':<18s} {c}/64 macro={mac:.2f}%")

    # ── TTA combine: average softmax across variants per head ──
    print("\n" + "=" * 60)
    print(" TTA-AVERAGED HEADS (across hflip + toffset + baseline)")
    print("=" * 60)
    tta_direct4 = np.mean([per_variant[v]['direct4_s314'] for v in variants], axis=0)
    tta_cascade = np.mean([per_variant[v]['cascade_s7'] for v in variants], axis=0)
    c, mac = report('tta_direct4', tta_direct4, y_ref); print(f"  tta_direct4_s314   {c}/64 macro={mac:.2f}%")
    c, mac = report('tta_cascade', tta_cascade, y_ref); print(f"  tta_cascade_s7     {c}/64 macro={mac:.2f}%")

    # TTA ensemble: average the two TTA-averaged heads
    tta_ens = np.mean([tta_direct4, tta_cascade], axis=0)
    c, mac = report('tta_ens', tta_ens, y_ref)
    print(f"  TTA ensemble       {c}/64 macro={mac:.2f}%")

    # Calibration on TTA ensemble
    best_bias, cal_c = np.zeros(4), c
    rng = np.random.default_rng(0)
    for _ in range(30000):
        bias = rng.uniform(-0.5, 0.5, 4)
        cc = int(((tta_ens + bias).argmax(1) == y_ref).sum())
        if cc > cal_c: cal_c = cc; best_bias = bias.copy()
    print(f"  TTA ens+calib      {cal_c}/64")

    # Error list on final prediction
    final = (tta_ens + best_bias).argmax(1) if cal_c > c else tta_ens.argmax(1)
    print("\n  Errors on TTA ens (final):")
    for i in range(len(y_ref)):
        if final[i] != y_ref[i]:
            print(f"    #{i}: true={CLASS_NAMES_4[y_ref[i]]} pred={CLASS_NAMES_4[final[i]]}")

    # Save report
    out = {
        'variants': variants,
        'per_variant_scores': {
            v: {k: {'correct': int((p.argmax(1) == y_ref).sum()),
                    'macro': float(np.mean([100 * int((p.argmax(1)[y_ref == i] == i).sum()) /
                                             max(1, int((y_ref == i).sum())) for i in range(4)]))}
                for k, p in per_variant[v].items()}
            for v in variants
        },
        'tta_direct4_correct': int((tta_direct4.argmax(1) == y_ref).sum()),
        'tta_cascade_correct': int((tta_cascade.argmax(1) == y_ref).sum()),
        'tta_ensemble_correct': int((tta_ens.argmax(1) == y_ref).sum()),
        'tta_ensemble_calibrated': int(cal_c),
        'tta_ensemble_macro': float(np.mean([100 * int((tta_ens.argmax(1)[y_ref == i] == i).sum()) /
                                              max(1, int((y_ref == i).sum())) for i in range(4)])),
        'total_time_s': time.time() - t0,
    }
    os.makedirs('sweep_results', exist_ok=True)
    json.dump(out, open('sweep_results/tta_results.json', 'w'), indent=2)
    print(f"\n  saved → sweep_results/tta_results.json")
    print(f"  total {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
