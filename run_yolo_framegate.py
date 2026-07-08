#!/usr/bin/env python3
"""
YOLO frame-gate — replicate the PAPER's actual mechanism (Önal & Dandıl 2025).

Paper: YOLO is a FRAME FILTER, not a feature extractor. It keeps frames containing an unsafe
behaviour and discards the rest; the reduced sequence goes to the temporal classifier. The paper's
+10pp (85.78 → 95.81) IS this filtering. V10 wrongly used YOLO as a feature extractor.

This script:
  1. Runs the fine-tuned YOLO detector (4 unsafe classes) over each clip's 32 cached frames → a
     per-frame max-unsafe-confidence array (cached to disk; the only new compute).
  2. Gates: keep frames with conf ≥ τ (fallback: top-K by conf), then resample the kept frames back
     to T=32 so the sequence is unsafe-frames-only but fixed-length (batchable). τ=0 ⇒ no-gate baseline.
  3. Builds gated stats features from the ALREADY-CACHED frozen per-frame features (all8_*.pt) — no
     re-extraction — and trains the direct-4 head under the BLIND protocol (epoch + τ selected on a
     train-carved val slice), reporting the honest gated vs no-gate test numbers.

⚠ READY TO RUN AFTER YOLO FINE-TUNE (run_finetune_bbox.py). Untested until weights exist. On first run
   VERIFY: (a) frame channel order (npy RGB vs YOLO BGR) — see CHANNEL_BGR flag; (b) that the dataset
   iteration order here matches run_pairwise_contrastive.extract_all_features (sorted folders → sorted
   video files), so YOLO masks align row-for-row with all8_*.pt features.

Usage:
  .venv/bin/python3 run_yolo_framegate.py --weights sweep_results/pairwise/yolo_finetuned/weights/best.pt
"""
import os, sys, argparse, json
import numpy as np
import torch

import run_pairwise_contrastive as pw
import run_blind_eval as be   # stratified_fit_val, macro_acc, fit_select_model, probs_on

DATA = 'data'
FEAT_DIR = 'sweep_results/pairwise/features'
CONF_CACHE = 'sweep_results/framegate_conf.pt'
UNSAFE_CLASSES = {0, 1, 2, 3}
CHANNEL_BGR = True  # npy frames assumed BGR-compatible for ultralytics; flip if verification fails
T = 32


def canonical_clips(subset):
    """Iterate clips in the SAME order as extract_all_features → (label, video_path)."""
    base = os.path.join(DATA, subset)
    out = []
    for folder in sorted(os.listdir(base)):
        full = os.path.join(base, folder)
        if not os.path.isdir(full):
            continue
        label = int(folder.split('_')[0])
        for vf in sorted(os.listdir(full)):
            if vf.lower().endswith(('.mp4', '.avi', '.mov')):
                out.append((label, os.path.join(full, vf)))
    return out


def compute_frame_conf(weights, device):
    """Per-clip (T,) max-unsafe-class YOLO confidence. Cached."""
    if os.path.exists(CONF_CACHE):
        print(f"  frame-conf cached: {CONF_CACHE}")
        return torch.load(CONF_CACHE, weights_only=False)
    from ultralytics import YOLO
    import cv2
    model = YOLO(weights)
    yolo_dev = 'cpu'  # MPS guard: detection on MPS is unreliable in this torch build; CPU is correct
    print(f"  running YOLO frame-gate on NATIVE-res frames, {yolo_dev} (matches training res; "
          f"index aligns with frozen features via linspace-{T})...")
    out = {}
    for subset in ['train', 'test']:
        clips = canonical_clips(subset)
        confs = []
        for ci, (label, vpath) in enumerate(clips):
            # extract the SAME T linspace frames the frozen features use, at NATIVE resolution
            cap = cv2.VideoCapture(vpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release(); confs.append(np.zeros(T, dtype=np.float32)); continue
            idxs = np.linspace(0, total - 1, T).astype(int)
            imgs = []
            for fi in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, fr = cap.read()
                imgs.append(fr if ok else np.zeros((1080, 1920, 3), dtype=np.uint8))  # BGR (cv2 native)
            cap.release()
            res = model.predict(imgs, verbose=False, device=yolo_dev, imgsz=640)
            per = []
            for r in res:
                if r.boxes is None or len(r.boxes) == 0:
                    per.append(0.0); continue
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                m = [cf[k] for k in range(len(cls)) if cls[k] in UNSAFE_CLASSES]
                per.append(float(max(m)) if m else 0.0)
            confs.append(np.array(per[:T] + [0.0] * max(0, T - len(per)), dtype=np.float32))
            if (ci + 1) % 50 == 0:
                print(f"    {subset}: {ci+1}/{len(clips)}", flush=True)
        out[subset] = np.stack(confs)  # (N, T)
    torch.save(out, CONF_CACHE)
    print(f"  saved {CONF_CACHE}")
    return out


def gate_indices(conf_row, tau, min_keep=6):
    """Indices of kept frames (conf ≥ τ), fallback top-min_keep, then resample to T."""
    keep = np.where(conf_row >= tau)[0]
    if len(keep) < min_keep:
        keep = np.argsort(-conf_row)[:min_keep]
        keep = np.sort(keep)
    if len(keep) == 0:
        keep = np.arange(T)
    sel = keep[np.clip(np.linspace(0, len(keep) - 1, T).round().astype(int), 0, len(keep) - 1)]
    return sel


def gated_stats(feats, conf, tau):
    """feats (N,T,S,C) frozen per-frame → gated (N,T,3C) stats over kept+resampled frames."""
    N = feats.shape[0]
    out = []
    for i in range(N):
        sel = gate_indices(conf[i], tau)
        f = feats[i][sel]  # (T,S,C)
        out.append(torch.cat([f.mean(1), f.std(1), f.max(1).values], -1))  # (T,3C)
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='fine-tuned YOLO best.pt (4 unsafe classes)')
    ap.add_argument('--taus', type=float, nargs='+', default=[0.0, 0.15, 0.25, 0.35, 0.5])
    ap.add_argument('--seeds', type=int, nargs='+', default=[314, 42, 123])
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--split-seed', type=int, default=0)
    ap.add_argument('--feat-train', default=os.path.join(FEAT_DIR, 'all8_train.pt'),
                    help='train feature .pt — frozen (all8) or bbox-finetuned (bbox_train.pt)')
    ap.add_argument('--feat-test', default=os.path.join(FEAT_DIR, 'all8_test.pt'))
    args = ap.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | taus={args.taus} | feats={os.path.basename(args.feat_train)}\n")

    conf = compute_frame_conf(args.weights, device)
    tr = torch.load(args.feat_train, map_location='cpu', weights_only=False)
    te = torch.load(args.feat_test, map_location='cpu', weights_only=False)
    tr_feats, tr_labels = tr['features'], tr['labels']
    te_feats, te_labels = te['features'], te['labels']
    assert tr_feats.shape[0] == conf['train'].shape[0], "train align mismatch — check iteration order"
    assert te_feats.shape[0] == conf['test'].shape[0], "test align mismatch — check iteration order"

    lt4_tr, lt4_te = tr_labels < 4, te_labels < 4
    ynp4 = te_labels[lt4_te].numpy()
    stats_dim = tr_feats.shape[-1] * 3

    # τ and epoch selected on a TRAIN-carved val slice; test touched once per τ (blind).
    fit_mask, val_mask = be.stratified_fit_val(tr_labels, args.val_frac, args.split_seed)
    fit_mask, val_mask = torch.from_numpy(fit_mask), torch.from_numpy(val_mask)
    fit4 = fit_mask & lt4_tr; val4 = val_mask & lt4_tr

    results = {}
    for tau in args.taus:
        tr_g = gated_stats(tr_feats, conf['train'], tau)   # (N,T,3C)
        te_g = gated_stats(te_feats, conf['test'], tau)
        te4 = te_g[lt4_te]
        per_seed_val, per_seed_test, per_seed_macro = [], [], []
        for seed in args.seeds:
            pw.set_seed(seed)
            m = pw.TransformerHead(stats_dim, 4, 384, 8, 6, 0.3).to(device)
            m = be.fit_select_model(m, tr_g[fit4], tr_labels[fit4],
                                    tr_g[val4], tr_labels[val4], 1.5e-4, device)
            vp = be.probs_on(m, tr_g[val4], device)
            tp = be.probs_on(m, te4, device)
            per_seed_val.append(int((vp.argmax(1) == tr_labels[val4].numpy()).sum()) / int(val4.sum()))
            per_seed_test.append(int((tp.argmax(1) == ynp4).sum()))
            per_seed_macro.append(round(be.macro_acc(tp.argmax(1), ynp4), 2))
        results[tau] = {'val_acc_mean': round(float(np.mean(per_seed_val)), 4),
                        'test_top1_mean': round(float(np.mean(per_seed_test)), 2),
                        'test_top1_per_seed': per_seed_test,
                        'test_macro_mean': round(float(np.mean(per_seed_macro)), 2)}
        print(f"  τ={tau:.2f}  val={results[tau]['val_acc_mean']:.3f}  "
              f"TEST {results[tau]['test_top1_mean']:.2f}/64 "
              f"({'/'.join(map(str, per_seed_test))})  macro {results[tau]['test_macro_mean']}%")

    print("\n" + "=" * 64)
    print("FRAME-GATE RESULTS — two numbers for two purposes")
    print("=" * 64)

    # (1) BLIND (honest): τ + epoch selected on validation; mean over seeds. Generalization estimate.
    best_tau = max(results, key=lambda t: results[t]['val_acc_mean'])
    blind = results[best_tau]
    print(f"  [BLIND, honest]   τ*={best_tau} (val-selected) → {blind['test_top1_mean']}/64 "
          f"({blind['test_top1_mean']/64*100:.1f}%), macro {blind['test_macro_mean']}%")

    # (2) MATCHED-PROTOCOL best-case (paper reports its single best, test-selected): max over τ AND seed.
    best_single = max(((t, max(results[t]['test_top1_per_seed'])) for t in results), key=lambda x: x[1])
    best_mean_tau = max(results, key=lambda t: results[t]['test_top1_mean'])
    bm = results[best_mean_tau]
    print(f"  [MATCHED-PROTOCOL vs paper] best single (τ={best_single[0]}, best seed): "
          f"{best_single[1]}/64 ({best_single[1]/64*100:.1f}%)")
    print(f"  [MATCHED-PROTOCOL mean] best-τ mean: {bm['test_top1_mean']}/64, macro {bm['test_macro_mean']}%")

    # (3) no-gate baseline (= frozen pipeline, all frames): the thing the gate must beat
    ng = results.get(0.0, {})
    print(f"  [NO-GATE baseline τ=0] blind {ng.get('test_top1_mean','?')}/64 "
          f"(best seed {max(ng.get('test_top1_per_seed',[0]))}/64)")
    print(f"\n  → Gate LIFT (blind): {blind['test_top1_mean'] - ng.get('test_top1_mean',0):+.2f} samples vs no-gate")
    print(f"  → vs paper: best-case {best_single[1]}/64 vs paper 62/64 (full) / 57/64 (no-bbox)")

    out_dir = 'plans/260613-thesis-defense-revision/reports'
    os.makedirs(out_dir, exist_ok=True)
    json.dump({'weights': args.weights, 'results': {str(k): v for k, v in results.items()},
               'blind_val_selected_tau': best_tau, 'blind_top1_mean': blind['test_top1_mean'],
               'blind_macro_mean': blind['test_macro_mean'],
               'matched_protocol_best_single': {'tau': best_single[0], 'top1': best_single[1]},
               'matched_protocol_best_mean': {'tau': best_mean_tau, 'top1': bm['test_top1_mean'],
                                              'macro': bm['test_macro_mean']},
               'no_gate_top1_mean': ng.get('test_top1_mean')},
              open(os.path.join(out_dir, 'framegate-result.json'), 'w'), indent=2)
    print(f"  saved {out_dir}/framegate-result.json")


if __name__ == '__main__':
    main()
