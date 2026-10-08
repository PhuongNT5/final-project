"""Regression guard + benchmark for the optimised zone-geometry path in build_relation_tokens.

postprocess (select + geometry) is 47.29 ms/decision on Jetson = 14.3 % of the decision budget
(Table 5.5H), second only to the detector, and geometry is ~28 % of that at median load.

The optimisation now SHIPS inside build_relation_tokens.geometry(). All of it is
exact-output-preserving:
  1. |A u B| = |A| + |B| - |A n B|  -- drops one constructed geometry per pair.
  2. Hoist polygon area/centroid/bounds/prepared-index to a per-view precompute (_zone_pre).
  3. shapely.prepared.prep() for the foot-point containment test.
  4. Bounding-box reject before any intersection (disjoint => IoU 0).

So this file keeps a VERBATIM copy of the pre-optimisation formulation as the reference and
checks the shipped function against it. Do not "simplify" the reference to call the shipped
function: that would make the equivalence check compare the optimisation with itself and pass
unconditionally. Equally, do not re-copy the optimisation here -- it must have exactly one home.

Run: python3 tools/geometry-postprocess-bench.py
"""
import json
import os
import sys
import time

import numpy as np
from shapely.geometry import Point, box as shp_box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_relation_tokens import geometry as geometry_shipped, load_zones  # noqa: E402

W, H = 1920, 1080
T_FRAMES, N_BOX = 16, 8
# A real decision does not fill every token slot. The baseline token tensors give mean 56.3
# boxes/clip, median 53, and geometry runs only for classes 0/1 = 72.5 % of them -> 40.8 boxes
# at median load. Reporting only the 128-box worst case overstates the saving by ~3x.
MEDIAN_GEOM_BOXES = 41


def geometry_reference(cx, cy, w, h, W, H, zones):
    """Verbatim pre-optimisation implementation. The equivalence reference, never optimised."""
    x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
    pbox = shp_box(x1, y1, x2, y2)
    foot = Point((x1 + x2) / 2, y2)

    def max_iou(polys):
        best = 0.0
        for pg in polys:
            uni = pbox.union(pg).area
            if uni > 0:
                best = max(best, pbox.intersection(pg).area / uni)
        return best

    iou_w = max_iou(zones['walkway']) if zones else 0.0
    iou_r = max_iou(zones['restricted']) if zones else 0.0
    foot_in = 1.0 if (zones and any(pg.contains(foot) for pg in zones['walkway'])) else 0.0
    nd = 0.0
    if zones and zones['restricted']:
        pc = Point((x1 + x2) / 2, (y1 + y2) / 2)
        nd = min(pc.distance(pg.centroid) for pg in zones['restricted']) / max(W, 1)
    return [iou_w, iou_r, foot_in, nd]


def _boxes(rng, n):
    """Random plausible boxes plus the degenerate/out-of-frame cases a sweep of uniforms misses:
    zero-area, whole-frame, and fully outside. The bounds-reject optimisation only differs from
    the reference on disjoint pairs, so those cases are the ones that can actually break it."""
    edge = [(0.5, 0.5, 0.0, 0.0), (0.5, 0.5, 1.0, 1.0), (1.5, 1.5, 0.1, 0.2), (-0.4, 0.5, 0.2, 0.3)]
    return edge + [(float(rng.uniform(.1, .9)), float(rng.uniform(.1, .9)),
                    float(rng.uniform(.05, .3)), float(rng.uniform(.1, .5))) for _ in range(n)]


def _time(fn, zones_for_view, views, boxes, reps=5):
    t0 = time.perf_counter()
    for _ in range(reps):
        for v in views:
            for (cx, cy, w, h) in boxes:
                fn(cx, cy, w, h, W, H, zones_for_view[v])
    return (time.perf_counter() - t0) / (reps * len(views)) * 1000


def main():
    zones = load_zones()
    views = sorted(zones)
    print(f'views with zones: {views}')
    rng = np.random.default_rng(0)
    boxes = _boxes(rng, T_FRAMES * N_BOX)

    worst = 0.0
    for v in views:
        for (cx, cy, w, h) in boxes:
            a = geometry_reference(cx, cy, w, h, W, H, zones[v])
            b = geometry_shipped(cx, cy, w, h, W, H, zones[v])
            worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
    for (cx, cy, w, h) in boxes[:4]:                      # the zones=None short-circuit
        assert geometry_reference(cx, cy, w, h, W, H, None) == \
               geometry_shipped(cx, cy, w, h, W, H, None) == [0.0, 0.0, 0.0, 0.0]
    n_pairs = len(views) * len(boxes)
    print(f'max |reference - shipped| over {n_pairs} box/view pairs: {worst:.3e}')
    if worst > 1e-12:
        raise SystemExit(f'FAIL: shipped geometry() is no longer output-equivalent ({worst:.3e}). '
                         'Every recorded token tensor was built with the reference semantics.')

    results = {}
    for load_name, load in (('median decision', boxes[4:4 + MEDIAN_GEOM_BOXES]),
                            ('worst case (all slots)', boxes)):
        print(f'\n{load_name} -- {len(load)} boxes with geometry, 1 view:')
        base = None
        for name, fn in (('reference', geometry_reference), ('shipped', geometry_shipped)):
            for _ in range(2):                            # warm the per-view precompute
                fn(*load[0], W, H, zones[views[0]])
            dt = _time(fn, zones, views, load)
            print(f'  {name:10s} {dt:7.3f} ms per decision')
            if name == 'reference':
                base = dt
            else:
                print(f'  speedup x{base / dt:.2f}  -> saves {base - dt:.2f} ms/decision here')
                results[load_name] = {'boxes': len(load), 'reference_ms': round(base, 3),
                                      'shipped_ms': round(dt, 3),
                                      'speedup': round(base / dt, 3),
                                      'saved_ms': round(base - dt, 3)}

    out = 'deployment/artifacts/geometry-postprocess-bench.json'
    json.dump({'note': 'Host-side geometry microbenchmark; ms are THIS host, not Jetson. The '
                       'speedup RATIO is the transferable quantity, not the absolute ms.',
               'max_abs_diff_reference_vs_shipped': worst, 'views': views,
               'box_loads': results}, open(out, 'w'), indent=1)
    print(f'\n[SAVED] {out}')


if __name__ == '__main__':
    main()
