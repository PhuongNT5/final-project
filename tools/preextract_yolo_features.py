import os
import sys
import argparse
import torch
import numpy as np
from torchvision import transforms

# ensure project root is importable when running from tools/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_arch import YOLO_Temporal_Transformer
from temporal_video_loader import UnsafeNetTemporalDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='data')
    ap.add_argument('--subset', type=str, default='train', choices=['train','test'])
    ap.add_argument('--out_dir', type=str, default='features')
    ap.add_argument('--sequence_length', type=int, default=16)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--yolo_weights', type=str, default='yolo11s.pt')
    ap.add_argument('--feature_layer_idx', type=int, default=-3)
    ap.add_argument('--embed_dim_mult', type=int, default=2)
    ap.add_argument('--max_items', type=int, default=0, help='0 = all videos, else limit for debugging')
    ap.add_argument('--pool', type=int, default=1, help='AdaptiveAvgPool grid size. 1=global avg pool (old), 4=keep spatial layout')
    ap.add_argument(
        '--layout',
        type=str,
        default='flat',
        choices=['flat', 'tokens'],
        help='Feature layout when pool>1: flat=(T,C*pool*pool) (old Exp1), tokens=(T,S,C) where S=pool*pool (recommended).',
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"{args.subset}_T{args.sequence_length}_img{args.img_size}_pool{args.pool}_{args.layout}_{os.path.splitext(os.path.basename(args.yolo_weights))[0]}.pt",
    )

    print('[PreExtract] subset=', args.subset)
    print('[PreExtract] out_path=', out_path)
    print('[PreExtract] yolo_weights=', args.yolo_weights)
    print('[PreExtract] pool=', args.pool)
    print('[PreExtract] layout=', args.layout)

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ds = UnsafeNetTemporalDataset(args.root, args.subset, args.sequence_length, tf)

    device = 'cpu'
    model = YOLO_Temporal_Transformer(
        num_classes=8,
        nhead=8,
        num_layers=4,
        dropout=0.5,
        yolo_weights=args.yolo_weights,
        feature_layer_idx=args.feature_layer_idx,
        embed_dim_mult=args.embed_dim_mult,
    ).to(device)
    model.eval()

    feats = []
    labels = []

    with torch.no_grad():
        n = len(ds)
        if args.max_items and args.max_items > 0:
            n = min(n, int(args.max_items))
        for i in range(n):
            seq, y = ds[i]  # (T,3,H,W)
            b = 1
            t = seq.shape[0]
            x = seq.unsqueeze(0)  # (1,T,3,H,W)
            x2 = x.view(b * t, *x.shape[2:]).to(device)
            f = model._extract_features(x2)
            if f is None:
                raise RuntimeError('Feature hook returned None')

            # Exp1+: keep spatial layout by pooling to (pool,pool)
            if args.pool and int(args.pool) > 1:
                p = int(args.pool)
                f = torch.nn.functional.adaptive_avg_pool2d(f, output_size=(p, p))  # (T,C,p,p)

                if args.layout == 'tokens':
                    # (T,C,p,p) -> (T,S,C) where S=p*p; keeps spatial tokens per frame
                    f = f.flatten(2).transpose(1, 2).contiguous()  # (T, p*p, C)
                else:
                    # legacy: (T, C*p*p)
                    f = f.flatten(1)
            else:
                # pool=1: match old behavior (T,C)
                f = model.adapter(f).flatten(1)  # (T,C)

            feats.append(f.cpu())
            labels.append(int(y))

            if (i + 1) % 10 == 0:
                print(f"Extracted {i+1}/{n}")

    # Stack: pool=1 => (N,T,C); pool>1 tokens => (N,T,S,C)
    feats = torch.stack(feats, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    payload = {
        'features': feats,
        'labels': labels,
        'sequence_length': args.sequence_length,
        'img_size': args.img_size,
        'yolo_weights': args.yolo_weights,
        'feature_layer_idx': args.feature_layer_idx,
        'pool': int(args.pool),
        'layout': args.layout,
    }

    torch.save(payload, out_path)
    print('Saved:', out_path)
    print('features shape:', tuple(feats.shape))


if __name__ == '__main__':
    main()
