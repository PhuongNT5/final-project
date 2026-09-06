"""Zone geometry: per-view polygons aligned to each clip by background homography.

The earlier pipeline used one canonical polygon per camera "view" and got foot_in_walkway ~ 0.12
for BOTH confused classes - the polygons simply did not line up with the individual clips.  Here
each clip's static background (temporal median) is registered to its view's reference background
with ORB+RANSAC, and the polygons are warped through that homography before any geometry is
computed.  A colour-based green-walkway mask is kept as an independent second opinion.
"""
from __future__ import annotations
import json
import os
import glob

import cv2
import numpy as np

from . import config as C
from .frame_cache import bg_path

_ZONES: dict | None = None
_H_CACHE: dict[str, np.ndarray] = {}
SX, SY = C.CACHE_W / 1920.0, C.CACHE_H / 1080.0


def load_zones() -> dict:
    global _ZONES
    if _ZONES is not None:
        return _ZONES
    z = {}
    for p in sorted(glob.glob(os.path.join(C.ZONES_DIR, 'view_*.json'))):
        d = json.load(open(p))
        w, h = d.get('image_size', [1920, 1080])
        sx, sy = C.CACHE_W / w, C.CACHE_H / h

        def cv(polys):
            return [np.array(q, np.float32) * [sx, sy] for q in polys if len(q) >= 3]
        z[str(d['view'])] = {'walkway': cv(d.get('safe_walkway', [])),
                             'restricted': cv(d.get('restricted_zone', [])),
                             'ref': d.get('source_frame', '')}
    _ZONES = z
    return z


def _ref_bg(view: str):
    """Background image of the clip the view's polygons were drawn on."""
    from .data_index import load_clips
    ref = load_zones().get(view, {}).get('ref', '')
    stem = ref.rsplit('_', 1)[0] if ref else ''
    for c in load_clips():
        if c.name == stem + '.mp4':
            p = bg_path(c)
            return cv2.imread(p) if os.path.exists(p) else None
    return None


def homography(clip, view: str) -> np.ndarray:
    """Warp matrix ref-background -> this clip's background (identity if registration fails)."""
    key = f'{view}|{clip.key}'
    if key in _H_CACHE:
        return _H_CACHE[key]
    H = np.eye(3, dtype=np.float32)
    ref, cur = _ref_bg(view), (cv2.imread(bg_path(clip)) if os.path.exists(bg_path(clip)) else None)
    if ref is not None and cur is not None:
        g1, g2 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(3000)
        k1, d1 = orb.detectAndCompute(g1, None); k2, d2 = orb.detectAndCompute(g2, None)
        if d1 is not None and d2 is not None and len(k1) > 12 and len(k2) > 12:
            m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
            m = sorted(m, key=lambda x: x.distance)[:400]
            if len(m) >= 12:
                src = np.float32([k1[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
                dst = np.float32([k2[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
                Hf, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
                if Hf is not None and 0.3 < abs(np.linalg.det(Hf[:2, :2])) < 3.0:
                    H = Hf.astype(np.float32)
    _H_CACHE[key] = H
    return H


def clip_polygons(clip, view: str) -> dict:
    H = homography(clip, view)
    z = load_zones().get(view, {'walkway': [], 'restricted': []})
    out = {}
    for k in ('walkway', 'restricted'):
        out[k] = [cv2.perspectiveTransform(p.reshape(-1, 1, 2), H).reshape(-1, 2) for p in z[k]]
    return out


def green_walkway_mask(bgr: np.ndarray) -> np.ndarray:
    """Colour prior for the painted walkway: green floor paint, keeping only large low-lying blobs."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (38, 45, 30), (90, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 2500 and st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] > 0.35 * m.shape[0]:
            out[lab == i] = 255
    return out


def _signed_dist(polys, pt) -> float:
    """Max over polygons of the signed distance (px, >0 inside), normalised by frame width."""
    if not polys:
        return 0.0
    d = max(cv2.pointPolygonTest(p.astype(np.float32), (float(pt[0]), float(pt[1])), True) for p in polys)
    return float(d) / C.CACHE_W


def person_zone_feats(box_xywhn, polys, mask=None) -> list[float]:
    """[in_walkway, sdist_walkway, in_restricted, sdist_restricted, green_under_foot]."""
    xc, yc, w, h = box_xywhn
    foot = (xc * C.CACHE_W, (yc + h / 2) * C.CACHE_H)
    dw = _signed_dist(polys.get('walkway', []), foot)
    dr = _signed_dist(polys.get('restricted', []), foot)
    g = 0.0
    if mask is not None:
        x, y = int(np.clip(foot[0], 0, C.CACHE_W - 1)), int(np.clip(foot[1], 0, C.CACHE_H - 1))
        r = max(4, int(0.12 * h * C.CACHE_H))
        patch = mask[max(0, y - r):y + r, max(0, x - r):x + r]
        g = float(patch.mean() / 255.0) if patch.size else 0.0
    return [float(dw > 0), dw, float(dr > 0), dr, g]
