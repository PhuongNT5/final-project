"""Clip index + blind held-out protocol splits.

Blind protocol (thesis SS4.7.7): every hyper-parameter, epoch and seed is selected on a
validation slice carved from the TRAIN clips.  The 64-clip test set is scored once per
final configuration and never used for selection.
"""
from __future__ import annotations
import glob
import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from . import config as C


@dataclass(frozen=True)
class Clip:
    path: str
    name: str          # basename, e.g. '0_tr125.mp4'
    label: int         # 0..7 source class
    subset: str        # 'train' | 'test'
    view: str          # camera view id, 'na' when unknown

    @property
    def key(self) -> str:
        return f'{self.subset}/{self.name}'

    @property
    def is_unsafe(self) -> bool:
        return self.label in C.UNSAFE


def _view_map() -> dict:
    if not os.path.exists(C.VIEW_MAP):
        return {}
    return {r['basename']: str(r['view']) for r in json.load(open(C.VIEW_MAP))}


def load_clips(subsets=('train', 'test'), labels=None) -> list[Clip]:
    """All clips of the requested subsets/labels, sorted for determinism."""
    vmap = _view_map()
    labels = list(range(8)) if labels is None else list(labels)
    out = []
    for sp in subsets:
        for lab in labels:
            d = os.path.join(C.DATA_DIR, sp, C.CLASS_DIRS[lab])
            for p in sorted(glob.glob(os.path.join(d, '*.mp4'))):
                b = os.path.basename(p)
                out.append(Clip(p, b, lab, sp, vmap.get(b, 'na')))
    return out


def blind_split(clips: list[Clip], val_frac: float = 0.18, seed: int = 0):
    """Stratified clip-level fit/val split of the TRAIN clips (no test contact).

    Stratification is on (label, view) so every camera is represented on both sides;
    strata with a single clip fall back to the fit side.
    """
    rng = np.random.default_rng(seed)
    strata: dict[tuple, list[int]] = {}
    for i, c in enumerate(clips):
        strata.setdefault((c.label, c.view), []).append(i)
    fit, val = [], []
    for _, idx in sorted(strata.items(), key=lambda kv: str(kv[0])):
        idx = list(idx)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        if len(idx) >= 3:
            n_val = max(1, n_val)
        val += idx[:n_val]
        fit += idx[n_val:]
    return sorted(fit), sorted(val)


def label_array(clips: list[Clip]) -> np.ndarray:
    return np.array([c.label for c in clips], dtype=np.int64)


def view_array(clips: list[Clip]) -> np.ndarray:
    return np.array([c.view for c in clips], dtype=object)


def describe(clips: list[Clip]) -> str:
    import collections
    cnt = collections.Counter(c.label for c in clips)
    return ' '.join(f'{C.CLASS_NAMES8[k]}={cnt.get(k, 0)}' for k in range(8))


def save_index(path: str | None = None) -> str:
    path = path or os.path.join(C.OUT_DIR, 'clip_index.json')
    rows = [asdict(c) for c in load_clips()]
    json.dump(rows, open(path, 'w'), indent=1)
    return path
