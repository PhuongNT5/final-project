"""pipeline-utils.py — Shared helpers for multi-stream pipeline & inference scripts."""
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PAIR_KEYS = ["pair0", "pair1", "pair2", "pair3"]
HEAD_FILES = {
    "direct4": "direct4-s314", "scenario": "scenario-s7",
    "pair0": "pair0-s7", "pair1": "pair1-s7", "pair2": "pair2-s7", "pair3": "pair3-s7",
}


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame → (3,224,224) float32. PIL BILINEAR to match training."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return ((arr - MEAN) / STD).transpose(2, 0, 1)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def stats_agg(buf: np.ndarray) -> np.ndarray:
    """(T, S, C) → (T, 3C) by concatenating mean/std/max over spatial dim S.
    Typical shapes: buf=(32, 16, 512) → (32, 1536). Works for any (T, S, C).
    For the (N, T, S, C) 4-D form used in benchmark-harness, use bench-utils.run_stats."""
    return np.concatenate([buf.mean(1), buf.std(1), buf.max(1)], axis=-1)


def percentiles(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    a = np.asarray(xs)
    return {"p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99))}
