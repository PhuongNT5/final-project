"""Expected Calibration Error (ECE) + reliability bins (numpy-only).

Equal-width confidence binning of Guo et al. (2017). No netcal dependency.
"""
from __future__ import annotations
import numpy as np


def reliability_bins(probs, y_true, n_bins=15):
    """probs: (N, C) softmax rows. Returns a list of per-bin dicts."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(m.sum())
        bins.append({
            'lo': float(lo), 'hi': float(hi), 'count': cnt,
            'confidence': float(conf[m].mean()) if cnt else None,
            'accuracy': float(correct[m].mean()) if cnt else None,
        })
    return bins


def ece(probs, y_true, n_bins=15):
    """Expected Calibration Error in [0, 1]. Lower is better-calibrated."""
    probs = np.asarray(probs, dtype=float)
    N = len(y_true)
    e = 0.0
    for b in reliability_bins(probs, y_true, n_bins):
        if b['count']:
            e += (b['count'] / N) * abs(b['accuracy'] - b['confidence'])
    return float(e)


def plot_reliability_diagram(bins, save_path, title=None):
    """Render a reliability diagram (accuracy vs confidence, equal-width bins) from
    `reliability_bins()` output and save it to `save_path`. Non-fatal if matplotlib
    is unavailable (returns None instead of raising) so this never blocks a headless
    metrics run."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return None

    xs = [b['confidence'] for b in bins if b['count']]
    ys = [b['accuracy'] for b in bins if b['count']]
    weights = [b['count'] for b in bins if b['count']]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='perfect calibration')
    ax.scatter(xs, ys, s=[20 + 2 * w for w in weights], alpha=0.8)
    ax.plot(xs, ys, '-', alpha=0.5)
    ax.set_xlabel('confidence')
    ax.set_ylabel('accuracy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or 'Reliability diagram')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
