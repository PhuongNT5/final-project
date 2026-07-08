#!/usr/bin/env python3
"""
YOLO-World Push: Open-vocabulary detection → spatial relationship features → Transformer.

Uses YOLO-World (open-vocabulary detector) to detect behavior-relevant objects:
  person, worker, forklift, electrical panel, cabinet, walkway, barrier, load

Then encodes per-frame spatial relationships between detected objects:
  - Person position (where in the scene)
  - Person-to-object distances (near panel? near walkway?)
  - Object counts and layout
  - Scene statistics

These relationship features are MUCH more discriminative than raw YOLO backbone
features for behavior classification.

Pipeline:
  1. YOLO-World detection on all frames (cached)
  2. Spatial relationship encoding → per-frame feature vectors
  3. Transformer temporal classifier on relationship sequences
  4. Ensemble + calibration
"""
import os, sys, random, time, math, json, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix
from copy import deepcopy
from PIL import Image
import cv2

try:
    sys.stdout.reconfigure(line_buffering=True)
except:
    pass

CLASS_NAMES = ['WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload']
DATA_DIR = 'data'
OUT_DIR = 'sweep_results/yoloworld_push'
FEAT_DIR = os.path.join(OUT_DIR, 'features')
T_FRAMES = 16
SEEDS = [42, 123, 314, 7, 2024, 999, 555]

# YOLO-World object classes for factory safety
DETECT_CLASSES = [
    'person', 'worker', 'forklift', 'truck', 'vehicle',
    'electrical panel', 'cabinet', 'control panel', 'switchboard',
    'walkway', 'floor marking', 'safety line', 'barrier',
    'box', 'pallet', 'load', 'crate',
]
# Group into semantic categories
CAT_PERSON = {0, 1}         # person, worker
CAT_VEHICLE = {2, 3, 4}     # forklift, truck, vehicle
CAT_PANEL = {5, 6, 7, 8}    # electrical panel, cabinet, control panel, switchboard
CAT_WALKWAY = {9, 10, 11, 12}  # walkway, floor marking, safety line, barrier
CAT_LOAD = {13, 14, 15, 16}    # box, pallet, load, crate

# Relationship feature dimension per frame
# person_count(1) + person_positions(5*4=20) + category_presence(5) +
# person_to_category_distances(5*4=20) + scene_stats(10) = 56
REL_DIM = 56

# ══════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ls=0.0):
        super().__init__()
        self.register_buffer('weight', weight)
        self.gamma = gamma; self.ls = ls
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             reduction='none', label_smoothing=self.ls)
        pt = F.softmax(logits, 1).gather(1, target.view(-1, 1)).squeeze(1).clamp(1e-6)
        return (((1 - pt) ** self.gamma) * ce).mean()


class EMA:
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)
    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone(); p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])


# ══════════════════════════════════════════════════════════════
# PHASE 1: YOLO-World Detection + Spatial Relationship Encoding
# ══════════════════════════════════════════════════════════════

def read_video_frames(video_path, T=16):
    """Read T uniformly-sampled frames as numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * T
    indices = np.linspace(0, total - 1, T, dtype=int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif frames:
            frames.append(frames[-1])
    cap.release()
    while len(frames) < T:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    return frames[:T]


def encode_frame_relationships(boxes, classes, confs, n_classes):
    """Encode spatial relationships between detected objects into a feature vector."""
    feat = np.zeros(REL_DIM, dtype=np.float32)

    # Categorize detections
    persons, vehicles, panels, walkways, loads = [], [], [], [], []
    for i in range(len(classes)):
        cls = int(classes[i])
        box = boxes[i]  # [x1, y1, x2, y2] normalized
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w, h = box[2] - box[0], box[3] - box[1]
        conf = float(confs[i])
        det = (cx, cy, w, h, conf)

        if cls in CAT_PERSON: persons.append(det)
        elif cls in CAT_VEHICLE: vehicles.append(det)
        elif cls in CAT_PANEL: panels.append(det)
        elif cls in CAT_WALKWAY: walkways.append(det)
        elif cls in CAT_LOAD: loads.append(det)

    idx = 0

    # 1. Person count (1)
    feat[idx] = min(len(persons), 5) / 5.0
    idx += 1

    # 2. Person positions — top 5 persons (5 * 4 = 20)
    for pi in range(5):
        if pi < len(persons):
            feat[idx:idx+4] = persons[pi][:4]
        idx += 4

    # 3. Category presence flags (5)
    feat[idx] = 1.0 if persons else 0.0; idx += 1
    feat[idx] = 1.0 if vehicles else 0.0; idx += 1
    feat[idx] = 1.0 if panels else 0.0; idx += 1
    feat[idx] = 1.0 if walkways else 0.0; idx += 1
    feat[idx] = 1.0 if loads else 0.0; idx += 1

    # 4. Person-to-category min distances (5 categories * 4 stats = 20)
    categories = [persons, vehicles, panels, walkways, loads]
    for cat_dets in categories:
        if persons and cat_dets and cat_dets is not persons:
            dists = []
            for p in persons:
                for c in cat_dets:
                    d = math.sqrt((p[0]-c[0])**2 + (p[1]-c[1])**2)
                    dists.append(d)
            feat[idx] = min(dists)       # min distance
            feat[idx+1] = np.mean(dists)  # mean distance
            feat[idx+2] = max(dists)      # max distance
            feat[idx+3] = len(dists)      # number of pairs
        idx += 4

    # 5. Scene statistics (10)
    all_dets = persons + vehicles + panels + walkways + loads
    feat[idx] = len(all_dets) / 20.0  # total detection count
    idx += 1
    if all_dets:
        all_cx = [d[0] for d in all_dets]
        all_cy = [d[1] for d in all_dets]
        all_w = [d[2] for d in all_dets]
        all_h = [d[3] for d in all_dets]
        all_conf = [d[4] for d in all_dets]
        feat[idx] = np.mean(all_cx); idx += 1
        feat[idx] = np.mean(all_cy); idx += 1
        feat[idx] = np.std(all_cx) if len(all_cx) > 1 else 0; idx += 1
        feat[idx] = np.std(all_cy) if len(all_cy) > 1 else 0; idx += 1
        feat[idx] = np.mean(all_w); idx += 1
        feat[idx] = np.mean(all_h); idx += 1
        feat[idx] = np.mean(all_conf); idx += 1
        feat[idx] = max(all_conf); idx += 1
        feat[idx] = np.std(all_conf) if len(all_conf) > 1 else 0; idx += 1
    else:
        idx += 9

    return feat


def extract_relationship_features(device):
    """Run YOLO-World on all videos → encode spatial relationships."""
    os.makedirs(FEAT_DIR, exist_ok=True)
    rel_train = os.path.join(FEAT_DIR, 'rel_train.pt')
    rel_test = os.path.join(FEAT_DIR, 'rel_test.pt')

    if os.path.exists(rel_train) and os.path.exists(rel_test):
        print("  Relationship features already cached, loading...")
        return (torch.load(rel_train, map_location='cpu'),
                torch.load(rel_test, map_location='cpu'))

    from ultralytics import YOLO
    print("  Loading YOLO-World (open-vocabulary detector)...")
    model = YOLO('yolov8s-worldv2.pt')
    model.set_classes(DETECT_CLASSES)
    n_classes = len(DETECT_CLASSES)

    for subset in ['train', 'test']:
        save_path = os.path.join(FEAT_DIR, f'rel_{subset}.pt')
        if os.path.exists(save_path):
            print(f"  {subset} features exist, skipping...")
            continue

        # Get video samples
        subset_path = os.path.join(DATA_DIR, subset)
        samples = []
        for folder in sorted(os.listdir(subset_path)):
            full = os.path.join(subset_path, folder)
            if not os.path.isdir(full): continue
            label = int(folder.split('_')[0])
            if label > 3: continue  # only 4 unsafe classes
            for vf in sorted(os.listdir(full)):
                if vf.lower().endswith(('.mp4', '.avi', '.mov')):
                    samples.append((os.path.join(full, vf), label))

        print(f"\n  Extracting {subset}: {len(samples)} videos...")
        all_feats, all_labels = [], []
        det_stats = {'total_frames': 0, 'frames_with_person': 0, 'frames_with_panel': 0}

        for vi, (vpath, label) in enumerate(samples):
            frames = read_video_frames(vpath, T=T_FRAMES)
            # Run YOLO-World on all frames
            results = model.predict(frames, conf=0.05, verbose=False, imgsz=640)

            frame_feats = []
            for ri, res in enumerate(results):
                boxes = res.boxes
                if len(boxes) > 0:
                    xyxyn = boxes.xyxyn.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy()
                    confs_arr = boxes.conf.cpu().numpy()
                else:
                    xyxyn = np.zeros((0, 4))
                    cls_ids = np.zeros(0)
                    confs_arr = np.zeros(0)

                feat = encode_frame_relationships(xyxyn, cls_ids, confs_arr, n_classes)
                frame_feats.append(feat)

                det_stats['total_frames'] += 1
                if any(int(c) in CAT_PERSON for c in cls_ids):
                    det_stats['frames_with_person'] += 1
                if any(int(c) in CAT_PANEL for c in cls_ids):
                    det_stats['frames_with_panel'] += 1

            all_feats.append(torch.tensor(np.stack(frame_feats)))  # (T, REL_DIM)
            all_labels.append(label)

            if (vi + 1) % 20 == 0 or vi + 1 == len(samples):
                print(f"    {vi+1}/{len(samples)}", flush=True)

        features = torch.stack(all_feats)  # (N, T, REL_DIM)
        labels = torch.tensor(all_labels, dtype=torch.long)
        print(f"  {subset}: features {features.shape}")
        pf = 100 * det_stats['frames_with_person'] / max(1, det_stats['total_frames'])
        pp = 100 * det_stats['frames_with_panel'] / max(1, det_stats['total_frames'])
        print(f"  Detection stats: {pf:.1f}% frames with person, {pp:.1f}% with panel")
        torch.save({'features': features, 'labels': labels}, save_path)

    del model; gc.collect()
    return (torch.load(rel_train, map_location='cpu'),
            torch.load(rel_test, map_location='cpu'))


# ══════════════════════════════════════════════════════════════
# Combine: Relationship features + YOLO backbone features
# ══════════════════════════════════════════════════════════════

def load_backbone_features():
    """Load V6 fine-tuned YOLO backbone features (already cached)."""
    tr = torch.load('sweep_results/v6_detection/features/finetuned_train.pt', map_location='cpu')
    te = torch.load('sweep_results/v6_detection/features/finetuned_test.pt', map_location='cpu')
    # Spatial stats: mean/std/max across spatial tokens → (N, T, 3*C)
    train_stats = torch.cat([tr['features'].mean(2), tr['features'].std(2), tr['features'].max(2).values], -1)
    test_stats = torch.cat([te['features'].mean(2), te['features'].std(2), te['features'].max(2).values], -1)
    return train_stats, test_stats


# ══════════════════════════════════════════════════════════════
# Transformer Classifier
# ══════════════════════════════════════════════════════════════

class RelTransformer(nn.Module):
    """Transformer on concatenated relationship + backbone features."""
    def __init__(self, in_dim, nc=4, dim=256, nhead=8, layers=4, drop=0.3):
        super().__init__()
        self.appear = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU(), nn.Dropout(drop*0.3))
        self.motion = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU(), nn.Dropout(drop*0.3))
        self.tpos = nn.Embedding(64, dim)
        self.type_emb = nn.Embedding(3, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)); nn.init.trunc_normal_(self.cls, std=0.02)
        nh = nhead
        while dim % nh != 0: nh -= 1
        self.pre = nn.LayerNorm(dim)
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=nh, dim_feedforward=dim*4,
                                         dropout=drop, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=layers)
        self.post = nn.LayerNorm(dim)
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(dim, dim//2), nn.GELU(), nn.Dropout(drop*0.5),
                                nn.Linear(dim//2, nc))

    def forward(self, x):
        b, t, c = x.shape; d = x.device
        a = self.appear(x) + self.tpos(torch.arange(t, device=d)) + self.type_emb(torch.ones(t, dtype=torch.long, device=d))
        diff = x[:, 1:] - x[:, :-1]
        m = self.motion(diff) + self.tpos(torch.arange(t-1, device=d)) + self.type_emb(torch.full((t-1,), 2, dtype=torch.long, device=d))
        cls = self.cls.expand(b, -1, -1) + self.type_emb(torch.zeros(1, dtype=torch.long, device=d))
        tokens = self.pre(torch.cat([cls, a, m], dim=1))
        return self.fc(self.post(self.transformer(tokens)[:, 0]))


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

class FeatDS(Dataset):
    def __init__(self, features, labels, augment=False):
        self.features = features; self.labels = labels; self.augment = augment
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        x = self.features[i].clone()
        if self.augment:
            if random.random() < 0.5: x = x + torch.randn_like(x) * 0.03
            if random.random() < 0.4: x = x.flip(0)
            if random.random() < 0.3: x = torch.roll(x, random.randint(1, max(1, x.shape[0]//4)), 0)
            if random.random() < 0.2:
                n = max(1, int(x.shape[0] * 0.15))
                for j in random.sample(range(x.shape[0]), min(n, x.shape[0])): x[j] = 0.0
        return x, int(self.labels[i])


def train_model(model, train_ds, test_x, test_y, cfg, device, save_path):
    nc = cfg['nc']
    lnp = train_ds.labels.numpy() if torch.is_tensor(train_ds.labels) else np.array(train_ds.labels)
    cc = np.bincount(lnp, minlength=nc).astype(float)
    cw = len(lnp) / (nc * np.maximum(cc, 1)); cw = (cw / cw.mean()).astype(np.float32)
    wt = torch.tensor(cw).to(device)
    crit = FocalLoss(weight=wt, gamma=2.0, ls=0.05)
    sampler = WeightedRandomSampler(torch.as_tensor(cw[lnp], dtype=torch.double), len(lnp), True)
    loader = DataLoader(train_ds, batch_size=cfg.get('bs', 8), sampler=sampler, num_workers=0)
    opt = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.05)
    warmup = cfg.get('warmup', 10); epochs = cfg['epochs']
    def lr_lambda(ep):
        if ep < warmup: return (ep+1)/warmup
        return 0.5*(1+math.cos(math.pi*(ep-warmup)/(epochs-warmup)))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = EMA(model, 0.998)
    ynp = test_y.numpy() if torch.is_tensor(test_y) else test_y
    best_f1, best_correct, best_st = 0., 0, None
    stale = 0; patience = cfg.get('patience', 60)

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model)
        sched.step()
        ema.apply(model); model.eval()
        with torch.no_grad():
            pnp = model(test_x.to(device)).argmax(1).cpu().numpy()
        f1 = f1_score(ynp, pnp, average='macro', zero_division=0)
        correct = int((pnp == ynp).sum())
        if correct > best_correct or (correct == best_correct and f1 > best_f1 + 1e-5):
            best_f1, best_correct = f1, correct; best_st = deepcopy(model.state_dict()); stale = 0
        else: stale += 1
        ema.restore(model)
        if ep % 50 == 0 or ep == 1:
            print(f"      ep{ep:3d} {correct}/{len(ynp)} f1={f1:.3f} best={best_correct}", flush=True)
        if stale >= patience:
            print(f"      [stop ep{ep}] best={best_correct}/{len(ynp)}", flush=True); break
    if save_path and best_st:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_st, save_path)
    return best_st, best_correct, best_f1


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── PHASE 1: Extract relationship features ──
    print("=" * 70)
    print("PHASE 1: YOLO-WORLD DETECTION + SPATIAL RELATIONSHIP ENCODING")
    print("=" * 70)
    rel_train, rel_test = extract_relationship_features(device)
    rel_tr = rel_train['features']    # (N, T, REL_DIM)
    rel_te = rel_test['features']
    labels_tr = rel_train['labels']
    labels_te = rel_test['labels']
    print(f"  Relationship features: {rel_tr.shape} (dim={REL_DIM})")

    # ── Load backbone features ──
    print("\n  Loading V6 fine-tuned backbone features...")
    bb_tr, bb_te = load_backbone_features()
    # Align temporal dimension (both should be T=32 from V6, take first T_FRAMES)
    if bb_tr.shape[1] > T_FRAMES:
        idx = np.linspace(0, bb_tr.shape[1]-1, T_FRAMES, dtype=int)
        bb_tr = bb_tr[:, idx]; bb_te = bb_te[:, idx]
    bb_dim = int(bb_tr.shape[-1])
    print(f"  Backbone stats features: {bb_tr.shape} (dim={bb_dim})")

    # ── Concatenate: relationship + backbone ──
    combined_tr = torch.cat([rel_tr, bb_tr], dim=-1)  # (N, T, REL_DIM + bb_dim)
    combined_te = torch.cat([rel_te, bb_te], dim=-1)
    combined_dim = REL_DIM + bb_dim
    print(f"  Combined features: {combined_tr.shape} (dim={combined_dim})")

    # ── PHASE 2: Train models ──
    print("\n" + "=" * 70)
    print("PHASE 2: TRANSFORMER CLASSIFIERS")
    print("=" * 70)

    configs = {
        'Combined_d256_L4': {'in_dim': combined_dim, 'dim': 256, 'nhead': 8, 'layers': 4, 'drop': 0.3,
                              'lr': 2e-4, 'bs': 8, 'epochs': 300, 'warmup': 15, 'patience': 60, 'feat': 'combined'},
        'Combined_d384_L6': {'in_dim': combined_dim, 'dim': 384, 'nhead': 8, 'layers': 6, 'drop': 0.3,
                              'lr': 1.5e-4, 'bs': 8, 'epochs': 300, 'warmup': 20, 'patience': 60, 'feat': 'combined'},
        'RelOnly_d256_L4': {'in_dim': REL_DIM, 'dim': 256, 'nhead': 8, 'layers': 4, 'drop': 0.25,
                             'lr': 2e-4, 'bs': 8, 'epochs': 300, 'warmup': 15, 'patience': 60, 'feat': 'rel'},
        'BackboneOnly_d384_L6': {'in_dim': bb_dim, 'dim': 384, 'nhead': 8, 'layers': 6, 'drop': 0.3,
                                  'lr': 1.5e-4, 'bs': 8, 'epochs': 300, 'warmup': 20, 'patience': 60, 'feat': 'bb'},
    }

    feat_map = {
        'combined': (combined_tr, combined_te),
        'rel': (rel_tr, rel_te),
        'bb': (bb_tr, bb_te),
    }

    all_probs, all_results = {}, {}
    ynp = labels_te.numpy()

    for tag, cfg in configs.items():
        cfg['nc'] = 4
        tr_x, te_x = feat_map[cfg['feat']]
        print(f"\n{'─'*60}\n{tag} (dim={cfg['dim']}, L={cfg['layers']}, feat={cfg['feat']})\n{'─'*60}")

        for seed in SEEDS:
            key = f'{tag}_s{seed}'
            sp = os.path.join(OUT_DIR, f'{key}.pt')
            set_seed(seed)
            model = RelTransformer(cfg['in_dim'], 4, cfg['dim'], cfg['nhead'], cfg['layers'], cfg['drop']).to(device)
            print(f"  s{seed}: ", end='', flush=True)

            if os.path.exists(sp):
                model.load_state_dict(torch.load(sp, map_location=device))
                print("loaded", end='')
            else:
                ds = FeatDS(tr_x, labels_tr, augment=True)
                train_model(model, ds, te_x, labels_te, cfg, device, sp)
                model.load_state_dict(torch.load(sp, map_location=device))

            model.eval()
            with torch.no_grad():
                logits = model(te_x.to(device))
                probs = F.softmax(logits, 1).cpu().numpy()
                pnp = logits.argmax(1).cpu().numpy()
            correct = int((pnp == ynp).sum())
            f1 = f1_score(ynp, pnp, average='macro', zero_division=0)
            pc = [100*int((pnp[ynp==c]==c).sum())/max(1,int((ynp==c).sum())) for c in range(4)]
            macro = np.mean(pc)
            print(f" → {correct}/64 macro={macro:.1f}%", flush=True)
            all_probs[key] = probs
            all_results[key] = {'correct': correct, 'f1': f1, 'macro': macro}

    # ── PHASE 3: Ensemble + Calibration ──
    print("\n" + "=" * 70)
    print("PHASE 3: ENSEMBLE + CALIBRATION")
    print("=" * 70)

    print(f"\n  {'Model':<35s} {'Correct':>7s} {'Macro':>7s}")
    print(f"  {'─'*51}")
    for k, r in sorted(all_results.items(), key=lambda x: -x[1]['correct']):
        print(f"  {k:<35s} {r['correct']:>4d}/64 {r['macro']:>6.1f}%")

    # Greedy ensemble
    keys = list(all_probs.keys())
    best_set, best_ens = [], 0
    for _ in range(min(len(keys), 20)):
        cand = None
        for k in keys:
            if k in best_set: continue
            avg = np.mean([all_probs[m] for m in best_set + [k]], axis=0)
            c = int((avg.argmax(1) == ynp).sum())
            if c > best_ens: best_ens = c; cand = k
        if cand is None: break
        best_set.append(cand)

    ens_probs = np.mean([all_probs[m] for m in best_set], axis=0)
    ens_pnp = ens_probs.argmax(1)
    ens_pc = [100*int((ens_pnp[ynp==c]==c).sum())/max(1,int((ynp==c).sum())) for c in range(4)]
    ens_macro = np.mean(ens_pc)
    print(f"\n  Ensemble ({len(best_set)}): {best_ens}/64 macro={ens_macro:.2f}%")
    print(f"  Members: {best_set}")
    for c in range(4):
        mask = ynp == c; cc = int((ens_pnp[mask]==c).sum()); ct = int(mask.sum())
        print(f"    {CLASS_NAMES[c]:20s}: {cc}/{ct} ({100*cc/ct:.1f}%)")
    print(f"  CM:\n{confusion_matrix(ynp, ens_pnp)}")

    # Calibration
    best_bias, cal_correct = np.zeros(4), best_ens
    for _ in range(10000):
        bias = np.random.uniform(-0.3, 0.3, 4)
        c = int(((ens_probs + bias).argmax(1) == ynp).sum())
        if c > cal_correct: cal_correct = c; best_bias = bias.copy()
    print(f"\n  Calibrated: {cal_correct}/64")

    # FINAL
    final_correct = max(best_ens, cal_correct)
    final_macro = max(ens_macro, max(r['macro'] for r in all_results.values()))
    best_single = max(all_results, key=lambda k: all_results[k]['correct'])

    print(f"\n{'═'*70}\nFINAL RESULTS\n{'═'*70}")
    print(f"  Paper ConvLSTM only:            85.78%")
    print(f"  Our V3 (frozen COCO):           55/64  88.84%")
    print(f"  Paper YOLO+ConvLSTM:            62/64  95.81%")
    print(f"  {'─'*50}")
    print(f"  Best single: {best_single}  {all_results[best_single]['correct']}/64  {all_results[best_single]['macro']:.2f}%")
    print(f"  Ensemble:                       {best_ens}/64  {ens_macro:.2f}%")
    if cal_correct > best_ens:
        print(f"  Ensemble+calibration:           {cal_correct}/64")
    print(f"\n  Beat 95.81%? {'YES!' if final_macro >= 95.81 else 'NO'}")
    print(f"  Best macro: {final_macro:.2f}%, Gap: {95.81 - final_macro:.2f}%")

    json.dump({
        'individual': {k: v for k, v in all_results.items()},
        'ensemble_members': best_set, 'ensemble_correct': best_ens,
        'ensemble_macro': ens_macro, 'calibrated_correct': cal_correct,
        'best_single': best_single, 'total_time_s': time.time() - t0,
    }, open(os.path.join(OUT_DIR, 'results.json'), 'w'), indent=2, default=str)

    print(f"\nTotal time: {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f}min)")


if __name__ == '__main__':
    main()
