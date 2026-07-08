#!/usr/bin/env python3
"""
Click-to-draw zone polygons (the manual half of GZT). Use it to trace the MACHINE / DANGER
zone per camera view — the area around presses/panels where UnauthorizedIntervention happens
(the go/no-go probe showed this, not the walkway, is the discriminative region).

For each view it opens annotation_guide/zones/view_{id}.png with the auto green WALKWAY drawn
in green as reference; you click the DANGER-zone vertices, press Enter to finish that view.

Controls (matplotlib ginput):
  left-click  = add vertex
  right-click = undo last vertex
  Enter       = finish this view (need >=3 vertices) and go to next
  close window / 0 vertices + Enter = skip this view

Saves/merges to annotation_guide/zones/zones_manual.json:
  {"view_id": {"danger": [[x,y],...]}}   (pixel coords, 1920x1080)
apply_zone_prior.py then merges this on top of the auto walkway and computes z4/z5.

Usage:
  .venv/bin/python3 tools/draw_zone.py --views 1 3 5 --type danger
  .venv/bin/python3 tools/draw_zone.py --views 5 --type walkway     # also fix a walkway by hand
"""
import os, json, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ZONE_DIR = 'annotation_guide/zones'
OUT = os.path.join(ZONE_DIR, 'zones_manual.json')


def load_existing():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def ref_walkway(view):
    p = os.path.join(ZONE_DIR, 'zones_auto.json')
    if os.path.exists(p):
        return json.load(open(p)).get(str(view), {}).get('walkway')
    return None


def draw_one(view, zone_type):
    png = os.path.join(ZONE_DIR, f'view_{view}.png')
    img = plt.imread(png)
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.imshow(img)
    wk = ref_walkway(view)
    if wk:
        poly = np.array(wk + [wk[0]])
        ax.plot(poly[:, 0], poly[:, 1], '-', color='lime', lw=2, label='auto walkway (ref)')
        ax.legend(loc='upper right')
    ax.set_title(f"view {view}: click the {zone_type.upper()} zone (machine/press area) — "
                 f"left=add, right=undo, Enter=done", fontsize=11)
    ax.set_xlabel("left-click vertices around the equipment/danger area, then press Enter")
    pts = plt.ginput(n=-1, timeout=0, show_clicks=True,
                     mouse_add=1, mouse_pop=3, mouse_stop=2)
    plt.close(fig)
    return [[int(round(x)), int(round(y))] for x, y in pts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', type=int, nargs='+', required=True)
    ap.add_argument('--type', choices=['danger', 'walkway'], default='danger')
    args = ap.parse_args()

    try:
        matplotlib.get_backend()
        plt.figure(); plt.close()
    except Exception as e:
        raise SystemExit(f"matplotlib GUI backend unavailable ({e}). Run locally with a display, "
                         f"or draw in labelme and convert.")

    data = load_existing()
    for v in args.views:
        png = os.path.join(ZONE_DIR, f'view_{v}.png')
        if not os.path.exists(png):
            print(f"  view {v}: missing {png}, skip"); continue
        poly = draw_one(v, args.type)
        if len(poly) < 3:
            print(f"  view {v}: <3 vertices, skipped"); continue
        data.setdefault(str(v), {})[args.type] = poly
        json.dump(data, open(OUT, 'w'), indent=2)   # save after each view (crash-safe)
        print(f"  view {v}: saved {len(poly)}-gon '{args.type}' -> {OUT}")

    print(f"\n[DONE] {OUT}")
    print("[next] re-probe:")
    print("  .venv/bin/python3 tools/apply_zone_prior.py --classes 0 1 --frames 16")
    print("  .venv/bin/python3 tools/check_zone_separation.py --min-coverage 0.2")


if __name__ == '__main__':
    main()
