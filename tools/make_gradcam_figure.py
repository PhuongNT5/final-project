#!/usr/bin/env python3
"""
Compose §4.5 thesis figure: Grad-CAM error analysis grid.

Top row:    one correct example per class (4 columns).
Rows 2-N:   error clips, 4 per row (up to 12 errors shown).

Reads pre-extracted overlays from figures/gradcam/overlays/ and metadata.json.
Writes figures/gradcam-error-analysis.{png,pdf}.
"""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

OVERLAY_DIR = 'figures/gradcam/overlays'
META = 'figures/gradcam/metadata.json'
OUT_PNG = 'figures/gradcam-error-analysis.png'
OUT_PDF = 'figures/gradcam-error-analysis.pdf'


def find_overlay(clip_idx):
    prefix = f"clip_{clip_idx:02d}_"
    for fn in os.listdir(OVERLAY_DIR):
        if fn.startswith(prefix):
            return os.path.join(OVERLAY_DIR, fn)
    return None


def main():
    metadata = json.load(open(META))
    errors = sorted([m for m in metadata if m['outcome'] == 'wrong'],
                    key=lambda m: m['clip_idx'])
    correct_per_class = {}
    for m in metadata:
        if m['outcome'] == 'correct' and m['true_label'] not in correct_per_class:
            correct_per_class[m['true_label']] = m

    n_err = min(len(errors), 12)
    n_err_rows = (n_err + 3) // 4
    n_rows = 1 + n_err_rows
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    plt.subplots_adjust(hspace=0.4, wspace=0.05)

    for i in range(4):
        ax = axes[0, i]
        if i in correct_per_class:
            m = correct_per_class[i]
            path = find_overlay(m['clip_idx'])
            if path: ax.imshow(Image.open(path))
            ax.set_title(f"#{m['clip_idx']} CORRECT: {m['true_class']}\n"
                         f"(p={m['pred_prob']:.2f})", fontsize=10, color='#0a7d27')
        ax.axis('off')

    for i, m in enumerate(errors[:n_err]):
        r, c = 1 + i // 4, i % 4
        ax = axes[r, c]
        path = find_overlay(m['clip_idx'])
        if path: ax.imshow(Image.open(path))
        ax.set_title(f"#{m['clip_idx']} ERROR: {m['true_class']}→{m['pred_class']}\n"
                     f"(p={m['pred_prob']:.2f})", fontsize=10, color='#c0392b')
        ax.axis('off')

    for i in range(n_err, n_err_rows * 4):
        r, c = 1 + i // 4, i % 4
        axes[r, c].axis('off')

    fig.suptitle("Grad-CAM Saliency: YOLOv11s C3k2 layer × Direct4-S314 head\n"
                 "Correct predictions (top row) vs persistent errors (rows 2+)",
                 fontsize=12, y=0.995)
    plt.savefig(OUT_PNG, dpi=140, bbox_inches='tight')
    plt.savefig(OUT_PDF, bbox_inches='tight')
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Errors shown: {n_err}/{len(errors)}")


if __name__ == '__main__':
    main()
