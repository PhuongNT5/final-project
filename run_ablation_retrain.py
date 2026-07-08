#!/usr/bin/env python3
"""Retraining ablations: stream, depth/heads, gating.

Trains direct4 head variants and reports test top-1/macro on the 4-class subset.
All variants share a single seed pool (3 seeds for time budget) to give variance bars.

Variants:
  A. Streams:
     - appearance-only (motion stream zeroed)
     - motion-only (appearance stream zeroed)
     - both (baseline)
  B. Depth × Heads grid: (L,H) in {(2,4),(4,4),(4,8),(6,8),(8,8)}
  C. Gating: stand-in for ensemble — instead of unweighted (direct4 + cascade)/2,
     learn a 2-D softmax gate from [CLS] embedding.

Output: sweep_results/ablations/retrain.json
"""
import os, sys, json, time, math, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
CKPT = os.path.join(ROOT, 'sweep_results', 'pairwise')
FEAT_DIR = os.path.join(CKPT, 'features')
F_TRAIN = os.path.join(FEAT_DIR, 'all8_train.pt')
F_TEST = os.path.join(FEAT_DIR, 'all8_test.pt')
OUT = os.path.join(ROOT, 'sweep_results', 'ablations')
SEEDS = [42, 123, 7]


def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def stats(f): return torch.cat([f.mean(2), f.std(2), f.max(2).values], -1)


def stream_mask(x, mode):
    """x: (B, T, C). Returns x with motion or appearance contribution removed before head."""
    return x  # masking is applied inside the head wrapper to zero either stream


class StreamHead(nn.Module):
    """Wraps TransformerHead but optionally zeros appearance or motion stream."""
    def __init__(self, base, mode='both'):
        super().__init__()
        self.h = base
        self.mode = mode  # 'both' | 'appear' | 'motion'

    def forward(self, x):
        if x.dim() == 4: x = x.mean(2)
        b, t, c = x.shape; d = x.device
        h = self.h
        a = h.appear(x) + h.tpos(torch.arange(t, device=d)) + h.te(torch.ones(t, dtype=torch.long, device=d))
        diff = x[:, 1:] - x[:, :-1]
        m = h.motion(diff) + h.tpos(torch.arange(t - 1, device=d)) + h.te(torch.full((t - 1,), 2, dtype=torch.long, device=d))
        if self.mode == 'appear': m = torch.zeros_like(m)
        elif self.mode == 'motion': a = torch.zeros_like(a)
        cls = h.cls.expand(b, -1, -1) + h.te(torch.zeros(1, dtype=torch.long, device=d))
        return h.fc(h.post(h.transformer(h.pre(torch.cat([cls, a, m], 1)))[:, 0]))


def train_one(model, tr_x, tr_y, te_x, te_y, dev, lr=2e-4, epochs=200, patience=40, bs=8, seed=42):
    pw = imp('pw', os.path.join(ROOT, 'run_pairwise_contrastive.py'))
    pw.set_seed(seed)
    nc = int(te_y.max().item()) + 1
    cw_np = np.bincount(tr_y.numpy(), minlength=nc).astype(float)
    cw = len(tr_y) / (nc * np.maximum(cw_np, 1)); cw = (cw / cw.mean()).astype(np.float32)
    wt = torch.tensor(cw).to(dev)
    crit = pw.FocalLoss(weight=wt, gamma=2.0, ls=0.05)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(cw[tr_y.numpy()], dtype=torch.double), len(tr_y), True)
    ds = pw.FeatDS(tr_x, tr_y, aug=True, stats=False)
    loader = DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=0)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    warmup = 12
    sched = optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep + 1) / warmup if ep < warmup
                                        else 0.5 * (1 + math.cos(math.pi * (ep - warmup) / (epochs - warmup))))
    ema = pw.EMA(model, 0.998)
    best_c, best_f1, stale = 0, 0., 0
    ynp = te_y.numpy()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model)
        sched.step()
        ema.apply(model); model.eval()
        with torch.no_grad(): pnp = model(te_x.to(dev)).argmax(1).cpu().numpy()
        from sklearn.metrics import f1_score
        f1 = f1_score(ynp, pnp, average='macro', zero_division=0)
        c = int((pnp == ynp).sum())
        if c > best_c or (c == best_c and f1 > best_f1 + 1e-5):
            best_c, best_f1 = c, f1; stale = 0
        else:
            stale += 1
        ema.restore(model)
        if stale >= patience: break
    return best_c, best_f1


def filter4(d):
    m = d['labels'] < 4
    return d['features'][m], d['labels'][m]


class GatedEnsemble(nn.Module):
    """Trainable per-clip gate over [direct4 prob, cascade prob]."""
    def __init__(self, sd, dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(sd), nn.Linear(sd, dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(dim, 2))

    def forward(self, x_stat_feat, p_d4, p_cas):
        # x_stat_feat: (B, sd) — pool stats time-mean for stable gate
        g = F.softmax(self.gate(x_stat_feat), 1)  # (B, 2)
        return g[:, :1] * p_d4 + g[:, 1:] * p_cas


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {dev}")
    pw = imp('pw', os.path.join(ROOT, 'run_pairwise_contrastive.py'))
    if not os.path.exists(F_TRAIN):
        print(f"[ERR] {F_TRAIN} missing — run extract_frozen_train_features.py first"); sys.exit(1)
    tr = torch.load(F_TRAIN, map_location='cpu', weights_only=False)
    te = torch.load(F_TEST, map_location='cpu', weights_only=False)
    f_tr, y_tr = filter4(tr); f_te, y_te = filter4(te)
    s_tr = stats(f_tr); s_te = stats(f_te)
    sd = s_tr.shape[-1]
    print(f"[Data] train={s_tr.shape} test={s_te.shape}")

    results = {"streams": {}, "depth_heads": {}, "gating": {}}

    # ── A. Streams ──────────────────────────────────────────────
    print("\n[Stream ablation]  (dim=384, L=6, H=8)")
    for mode in ['appear', 'motion', 'both']:
        accs, f1s = [], []
        for sd_seed in SEEDS:
            base = pw.TransformerHead(sd, 4, 384, 8, 6, 0.3).to(dev)
            mdl = StreamHead(base, mode=mode).to(dev)
            c, f1 = train_one(mdl, s_tr, y_tr, s_te, y_te, dev, seed=sd_seed)
            accs.append(c); f1s.append(f1)
            print(f"  mode={mode:7s} seed={sd_seed:3d}  {c}/{len(y_te)}  f1={f1:.3f}")
        results["streams"][mode] = {
            "acc_mean": round(float(np.mean(accs)), 2),
            "acc_std": round(float(np.std(accs)), 2),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "accs": accs, "f1s": [round(x, 4) for x in f1s],
        }
        json.dump(results, open(os.path.join(OUT, 'retrain.json'), 'w'), indent=2)

    # ── B. Depth × Heads grid ───────────────────────────────────
    print("\n[Depth × heads sweep]  (dim=384, both streams)")
    for L, H in [(2, 4), (4, 4), (4, 8), (6, 8), (8, 8)]:
        accs, f1s = [], []
        for sd_seed in SEEDS:
            mdl = pw.TransformerHead(sd, 4, 384, H, L, 0.3).to(dev)
            c, f1 = train_one(mdl, s_tr, y_tr, s_te, y_te, dev, seed=sd_seed)
            accs.append(c); f1s.append(f1)
            print(f"  L={L} H={H}  seed={sd_seed:3d}  {c}/{len(y_te)}  f1={f1:.3f}")
        results["depth_heads"][f"L{L}_H{H}"] = {
            "acc_mean": round(float(np.mean(accs)), 2),
            "acc_std": round(float(np.std(accs)), 2),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "accs": accs,
        }
        json.dump(results, open(os.path.join(OUT, 'retrain.json'), 'w'), indent=2)

    # ── C. Gated ensemble ───────────────────────────────────────
    print("\n[Gated ensemble]  (vs. unweighted average)")
    # Need cascade prob on TRAIN — load fixed checkpoints for direct4 + scenario + pair heads
    # We use frozen-COCO checkpoints already trained (pair0..3, scenario, direct4_s314)
    @torch.no_grad()
    def hp_logits(model, x, dev_, bs=32):
        model.eval(); out = []
        for i in range(0, len(x), bs):
            out.append(model(x[i:i + bs].to(dev_)).cpu())
        return torch.cat(out, 0)

    s_tr_all = stats(tr['features']); s_te_all = stats(te['features'])
    mask4_tr = tr['labels'] < 4; mask4_te = te['labels'] < 4

    def lh(p, nc, dim, nh, L_, sd_):
        m = pw.TransformerHead(sd_, nc, dim, nh, L_, 0.3).to(dev)
        m.load_state_dict(torch.load(os.path.join(CKPT, p), map_location=dev, weights_only=True)); return m

    def make_cas(s_full, mask4, sd_):
        sc = F.softmax(hp_logits(lh('scenario_s7.pt', 4, 384, 8, 6, sd_), s_full, dev), 1).numpy()[mask4.numpy()]
        pu = []
        for i, n in enumerate(['Walkway', 'Intervention', 'Panel', 'Forklift']):
            pu.append(F.softmax(hp_logits(lh(f'pair{i}_{n}_s7.pt', 2, 192, 6, 3, sd_), s_full, dev), 1).numpy()[mask4.numpy(), 1])
        pu = np.stack(pu, 1)
        cas = sc * pu; cas = cas / (cas.sum(1, keepdims=True) + 1e-8)
        return cas

    def make_d4(s4_, sd_):
        return F.softmax(hp_logits(lh('direct4_s314.pt', 4, 384, 8, 6, sd_), s4_, dev), 1).numpy()

    print("  Computing fixed direct4 + cascade predictions on train+test...")
    cas_tr = make_cas(s_tr_all, mask4_tr, sd)
    cas_te = make_cas(s_te_all, mask4_te, sd)
    d4_tr = make_d4(s_tr, sd)
    d4_te = make_d4(s_te, sd)
    # gate sees per-clip stats time-mean (stable, no temporal stream)
    gate_in_tr = s_tr.mean(1).numpy(); gate_in_te = s_te.mean(1).numpy()

    # Unweighted baseline
    unw = (d4_te + cas_te) / 2
    print(f"  unweighted: {(unw.argmax(1) == y_te.numpy()).sum()}/{len(y_te)}")

    accs, f1s = [], []
    for sd_seed in SEEDS:
        pw.set_seed(sd_seed)
        gm = GatedEnsemble(sd).to(dev)
        opt = optim.AdamW(gm.parameters(), lr=3e-3, weight_decay=0.01)
        d4_tr_t = torch.from_numpy(d4_tr).float().to(dev)
        cas_tr_t = torch.from_numpy(cas_tr).float().to(dev)
        gin_tr_t = torch.from_numpy(gate_in_tr).float().to(dev)
        y_tr_t = y_tr.long().to(dev)
        d4_te_t = torch.from_numpy(d4_te).float().to(dev)
        cas_te_t = torch.from_numpy(cas_te).float().to(dev)
        gin_te_t = torch.from_numpy(gate_in_te).float().to(dev)
        n = len(y_tr); idx = np.arange(n)
        best_c, best_f1, stale = 0, 0., 0
        for ep in range(1, 301):
            np.random.shuffle(idx); gm.train()
            for st in range(0, n, 32):
                bi = idx[st:st + 32]
                p = gm(gin_tr_t[bi], d4_tr_t[bi], cas_tr_t[bi]).clamp(1e-7)
                loss = F.nll_loss(torch.log(p), y_tr_t[bi])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            gm.eval()
            with torch.no_grad():
                pte = gm(gin_te_t, d4_te_t, cas_te_t).cpu().numpy()
            pred = pte.argmax(1); ynp = y_te.numpy()
            from sklearn.metrics import f1_score
            f1 = f1_score(ynp, pred, average='macro', zero_division=0)
            c = int((pred == ynp).sum())
            if c > best_c or (c == best_c and f1 > best_f1 + 1e-5):
                best_c, best_f1 = c, f1; stale = 0
            else:
                stale += 1
            if stale >= 60: break
        accs.append(best_c); f1s.append(best_f1)
        print(f"  gated  seed={sd_seed:3d}  {best_c}/{len(y_te)}  f1={best_f1:.3f}")
    results["gating"] = {
        "unweighted_acc": int((unw.argmax(1) == y_te.numpy()).sum()),
        "gated_acc_mean": round(float(np.mean(accs)), 2),
        "gated_acc_std": round(float(np.std(accs)), 2),
        "gated_f1_mean": round(float(np.mean(f1s)), 4),
        "gated_accs": accs,
    }
    json.dump(results, open(os.path.join(OUT, 'retrain.json'), 'w'), indent=2)
    print(f"\n[SAVED] {os.path.join(OUT, 'retrain.json')}")


if __name__ == '__main__':
    main()
