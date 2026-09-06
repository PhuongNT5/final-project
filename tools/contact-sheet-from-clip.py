"""Contact sheet from a clip: N frames evenly spaced, labelled with frame index.

Used to adjudicate the byte-identical / contradictory-label groups found by
run_duplicate_audit.py against the original footage in data/.

Usage: .venv/bin/python tools/contact-sheet-from-clip.py <clip.mp4> <out.png> [--n 6] [--cols 3]
"""
import argparse
import cv2
from PIL import Image, ImageDraw

a = argparse.ArgumentParser()
a.add_argument('clip'); a.add_argument('out')
a.add_argument('--n', type=int, default=6); a.add_argument('--cols', type=int, default=3)
a.add_argument('--width', type=int, default=560, help='per-tile width')
a.add_argument('--title', default='')
v = a.parse_args()

cap = cv2.VideoCapture(v.clip)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
idxs = [int(round(i * (total - 1) / max(v.n - 1, 1))) for i in range(v.n)]

tiles = []
for fi in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    if not ok:
        continue
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h = int(img.height * v.width / img.width)
    img = img.resize((v.width, h))
    d = ImageDraw.Draw(img)
    txt = f'f{fi}'
    d.rectangle([0, 0, 70, 26], fill=(0, 0, 0))
    d.text((6, 6), txt, fill=(255, 255, 0))
    tiles.append(img)
cap.release()

cols = min(v.cols, len(tiles)); rows = (len(tiles) + cols - 1) // cols
tw, th = tiles[0].size
pad, top = 4, 30 if v.title else 0
sheet = Image.new('RGB', (cols * tw + (cols + 1) * pad, top + rows * th + (rows + 1) * pad), (24, 24, 24))
if v.title:
    ImageDraw.Draw(sheet).text((8, 8), v.title, fill=(255, 255, 255))
for i, t in enumerate(tiles):
    r, c = divmod(i, cols)
    sheet.paste(t, (pad + c * (tw + pad), top + pad + r * (th + pad)))
sheet.save(v.out)
print(f'{v.clip}: {total} frames @ {fps:.1f}fps -> {v.out} ({len(tiles)} tiles)')
