#!/usr/bin/env python3
"""
Auto-derive a walkway polygon per camera view from the PAINTED green path, so the GZT
go/no-go test can run with zero manual annotation. The factory floor marks the allowed
walkway in green (see annotation_guide/zones/README-draw-zones.md); we HSV-threshold that
green, take the largest contour, and emit one polygon per view.

Output: annotation_guide/zones/zones_auto.json  -> {"view_id": {"walkway": [[x,y],...]}}
        annotation_guide/zones/overlay_view_{id}.png  (polygon drawn on the view, for review)

These auto polygons are a fast proxy for the go/no-go probe; for the final thesis a human can
hand-trace (labelme) and drop a manual zones.json — apply_zone_prior.py prefers that if present.

Usage:
  .venv/bin/python3 tools/make_zones_auto.py
  .venv/bin/python3 tools/make_zones_auto.py --hsv 35 40 40 90 255 255 --min-area 0.02
"""
import os, json, argparse
import numpy as np
import cv2

ZONE_DIR = 'annotation_guide/zones'


def green_polygon(img, hsv_lo, hsv_hi, min_area_frac, approx_eps):
    """Return the largest green-region polygon as Nx2 int pixel coords, or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))
    # close gaps (people/forklifts occluding the paint), then drop speckle
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    area_frac = cv2.contourArea(c) / float(img.shape[0] * img.shape[1])
    if area_frac < min_area_frac:
        return None, area_frac
    poly = cv2.approxPolyDP(c, approx_eps * cv2.arcLength(c, True), True)
    return poly.reshape(-1, 2).astype(int), area_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hsv', type=int, nargs=6, default=[35, 40, 40, 90, 255, 255],
                    help='Hlo Slo Vlo Hhi Shi Vhi (OpenCV HSV, H in 0-179)')
    ap.add_argument('--min-area', type=float, default=0.015,
                    help='min green area fraction to accept a walkway polygon')
    ap.add_argument('--approx-eps', type=float, default=0.01,
                    help='approxPolyDP epsilon as fraction of contour perimeter')
    args = ap.parse_args()

    vp = json.load(open(os.path.join(ZONE_DIR, 'viewpoints.json')))
    hsv_lo, hsv_hi = args.hsv[:3], args.hsv[3:]
    zones, summary = {}, []

    for v in vp['views']:
        vid = v['view']
        png = os.path.join(ZONE_DIR, f"view_{vid}.png")
        img = cv2.imread(png)
        if img is None:
            print(f"  view {vid}: MISSING {png}"); continue
        poly, area = green_polygon(img, hsv_lo, hsv_hi, args.min_area, args.approx_eps)
        dist = v['class_dist']
        mixed = dist.get('Walk', 0) > 0 or dist.get('Unauth', 0) > 0
        if poly is None:
            summary.append((vid, 'no-green', area, mixed, dist))
            print(f"  view {vid}: no walkway found (green area {area:.3%})  dist={dist}")
            continue
        zones[str(vid)] = {'walkway': poly.tolist()}
        # overlay for human verification
        ov = img.copy()
        cv2.polylines(ov, [poly.reshape(-1, 1, 2)], True, (0, 0, 255), 4)
        cv2.imwrite(os.path.join(ZONE_DIR, f"overlay_view_{vid}.png"), ov)
        summary.append((vid, f'{len(poly)}-gon', area, mixed, dist))
        print(f"  view {vid}: walkway {len(poly)}-gon, area {area:.2%}  "
              f"{'MIXED Walk/Unauth' if mixed else ''}  dist={dist}")

    out = os.path.join(ZONE_DIR, 'zones_auto.json')
    json.dump(zones, open(out, 'w'), indent=2)
    n_mixed = sum(1 for s in summary if s[3])
    n_mixed_ok = sum(1 for s in summary if s[3] and str(s[0]) in zones)
    print(f"\n[SAVED] {out}  ({len(zones)} polygons; {n_mixed_ok}/{n_mixed} mixed Walk/Unauth views covered)")
    print(f"[REVIEW] overlays at {ZONE_DIR}/overlay_view_*.png — eyeball before trusting the probe.")
    if n_mixed_ok < n_mixed:
        print("[WARN] some mixed views have no auto-walkway; widen --hsv or hand-draw those in zones.json.")


if __name__ == '__main__':
    main()
