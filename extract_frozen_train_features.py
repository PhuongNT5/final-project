#!/usr/bin/env python3
"""One-shot extractor: frozen COCO YOLOv11s features for ALL 8 classes, train + test.
Calls run_pairwise_contrastive.extract_all_features and exits.
"""
import sys, os, importlib.util, torch
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location('pw', os.path.join(ROOT, 'run_pairwise_contrastive.py'))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
device = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
tr, te = mod.extract_all_features(device)
print(f"Train: {tuple(tr['features'].shape)}  Test: {tuple(te['features'].shape)}")
print(f"Saved to: {mod.FEAT_DIR}")
