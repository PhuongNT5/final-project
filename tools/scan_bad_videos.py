import os
import cv2
import argparse

def iter_videos(root):
    for subset in ["train", "test"]:
        subset_dir = os.path.join(root, subset)
        if not os.path.isdir(subset_dir):
            continue
        for cls in sorted(os.listdir(subset_dir)):
            cdir = os.path.join(subset_dir, cls)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                if fn.lower().endswith((".mp4", ".avi", ".mov")):
                    yield subset, cls, os.path.join(cdir, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--max", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    bad = []
    n = 0
    for subset, cls, path in iter_videos(args.root):
        n += 1
        if args.max and n > args.max:
            break
        cap = cv2.VideoCapture(path)
        ok = cap.isOpened()
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else -1
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if ok else -1
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if ok else -1
        cap.release()
        if (not ok) or frames <= 0 or w <= 0 or h <= 0:
            bad.append((subset, cls, path, ok, frames, w, h))

    print(f"Scanned {n} videos")
    print(f"Bad videos: {len(bad)}")
    for item in bad[:200]:
        print(item)
    if len(bad) > 200:
        print("... truncated ...")

if __name__ == "__main__":
    main()
