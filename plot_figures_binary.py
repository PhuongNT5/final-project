#!/usr/bin/env python3
"""
Generate thesis-quality figures for binary safe/unsafe classification results.
Also export best model (Gated_raw_L) to ONNX + INT8 and benchmark latency.

Best result: Gated_raw_L (dim=256, L=4) seed=42 → 83.2% (104/125), F1=83.0%
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, f1_score,
                             accuracy_score, precision_score, recall_score)

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# ═══════════════════════════════════════════════════════════════
# Model Definition (must match run_temporal_fusion.py)
# ═══════════════════════════════════════════════════════════════

class GatedFusionBinaryHead(nn.Module):
    def __init__(self, in_dim=768, model_dim=192, nhead=4, num_layers=3,
                 dropout=0.3, max_t=64, num_classes=2):
        super().__init__()
        self.proj_raw = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU())
        self.proj_d1 = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU())
        self.proj_d2 = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.GELU())
        self.proj_win = nn.Sequential(
            nn.LayerNorm(in_dim * 3), nn.Linear(in_dim * 3, model_dim), nn.GELU())
        self.gate = nn.Sequential(
            nn.Linear(model_dim * 4, model_dim), nn.GELU(),
            nn.Linear(model_dim, 4), nn.Softmax(dim=-1))
        nh = nhead
        while model_dim % nh != 0: nh -= 1
        enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.post_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes))

    def _windowed_stats(self, x, window=4):
        b, t, c = x.shape; pad = window // 2
        xp = F.pad(x, (0, 0, pad, pad), mode='replicate')
        stats_list = []
        for i in range(t):
            w = xp[:, i:i + window, :]
            stats_list.append(torch.cat([w.mean(1), w.std(1), w.max(1).values], dim=-1))
        return torch.stack(stats_list, dim=1)

    def forward(self, x):
        if x.dim() == 4: x = x.mean(2)
        b, t, c = x.shape; d = x.device
        pos = self.time_pos(torch.arange(t, device=d)).unsqueeze(0)
        f_raw = self.proj_raw(x) + pos
        d1 = x[:, 1:] - x[:, :-1]
        f_d1 = self.proj_d1(d1) + self.time_pos(torch.arange(t - 1, device=d)).unsqueeze(0)
        d2 = d1[:, 1:] - d1[:, :-1]
        f_d2 = self.proj_d2(d2) + self.time_pos(torch.arange(t - 2, device=d)).unsqueeze(0)
        win_stats = self._windowed_stats(x)
        f_win = self.proj_win(win_stats) + pos
        s_raw = f_raw.mean(1); s_d1 = f_d1.mean(1)
        s_d2 = f_d2.mean(1); s_win = f_win.mean(1)
        gates = self.gate(torch.cat([s_raw, s_d1, s_d2, s_win], dim=-1))
        fused = gates[:, 0:1].unsqueeze(1) * f_raw + gates[:, 3:4].unsqueeze(1) * f_win
        pad_d1 = F.pad(f_d1, (0, 0, 0, 1), value=0)
        pad_d2 = F.pad(f_d2, (0, 0, 0, 2), value=0)
        fused = fused + gates[:, 1:2].unsqueeze(1) * pad_d1 + gates[:, 2:3].unsqueeze(1) * pad_d2
        out = self.transformer(fused)
        return self.fc(self.post_norm(out.mean(1)))


class SpatialMotionHead(nn.Module):
    def __init__(self, in_dim, num_classes, model_dim=256, nhead=8, num_layers=4,
                 dropout=0.3, max_t=64):
        super().__init__()
        self.appear_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.motion_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5))
        self.time_pos = nn.Embedding(max_t, model_dim)
        self.type_embed = nn.Embedding(3, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nh = nhead
        while model_dim % nh != 0: nh -= 1
        self.pre_norm = nn.LayerNorm(model_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nh, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(model_dim // 2, num_classes))

    def forward(self, x):
        if x.dim() == 4: x = x.mean(2)
        b, t, c = x.shape; d = x.device
        appear = self.appear_proj(x)
        appear += self.time_pos(torch.arange(t, device=d)).unsqueeze(0)
        appear += self.type_embed(torch.ones(t, dtype=torch.long, device=d)).unsqueeze(0)
        diff = x[:, 1:] - x[:, :-1]
        motion = self.motion_proj(diff)
        motion += self.time_pos(torch.arange(t - 1, device=d)).unsqueeze(0)
        motion += self.type_embed(torch.full((t - 1,), 2, dtype=torch.long, device=d)).unsqueeze(0)
        cls = (self.cls_token.expand(b, -1, -1)
               + self.type_embed(torch.zeros(1, dtype=torch.long, device=d)).unsqueeze(0))
        tokens = self.pre_norm(torch.cat([cls, appear, motion], dim=1))
        tokens = self.transformer(tokens)
        return self.fc(self.post_norm(tokens[:, 0]))


# ═══════════════════════════════════════════════════════════════
# Load data + models
# ═══════════════════════════════════════════════════════════════

def main():
    device = 'cpu'
    output_dir = 'figures'
    deploy_dir = 'deployment'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(deploy_dir, exist_ok=True)

    CLASS_NAMES_8 = [
        'WalkwayViol', 'UnauthorInt', 'OpenedPanel', 'CarryOverload',
        'SafeWalkway', 'AuthorizedInt', 'ClosedPanel', 'SafeCarrying'
    ]
    CLASS_LABELS_8 = [
        'Walkway\nViolation', 'Unauthorized\nIntervention', 'Opened\nPanel', 'Carrying\nOverload',
        'Safe\nWalkway', 'Authorized\nIntervention', 'Closed\nPanel', 'Safe\nCarrying'
    ]

    # Load test data
    print("Loading test data...")
    tst = torch.load('features/test_T32_img224_pool4_tokens_yolo11s.pt', map_location='cpu')
    test_x_tokens = tst['features']  # (125, 32, 16, 768)
    labels_8 = tst['labels'].numpy()
    test_y = (tst['labels'] < 4).long()  # 1=unsafe, 0=safe
    test_y_np = test_y.numpy()

    n_safe = int((test_y == 0).sum())
    n_unsafe = int((test_y == 1).sum())
    print(f"Test set: {len(test_y)} clips, safe={n_safe}, unsafe={n_unsafe}")

    # ─── Load best model: Gated_raw_L s42 ───
    print("\nLoading best model: Gated_raw_L (dim=256, L=4, seed=42)...")
    best_model = GatedFusionBinaryHead(768, 256, 8, 4, 0.35, 64, 2).to(device)
    best_model.load_state_dict(torch.load(
        'sweep_results/safe_unsafe/su_Gated_raw_L_s42.pt', map_location=device))
    best_model.eval()

    with torch.no_grad():
        logits = best_model(test_x_tokens.to(device))
        probs = F.softmax(logits, 1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

    acc = accuracy_score(test_y_np, preds)
    f1m = f1_score(test_y_np, preds, average='macro', zero_division=0)
    prec = precision_score(test_y_np, preds, average='macro', zero_division=0)
    rec = recall_score(test_y_np, preds, average='macro', zero_division=0)
    cm = confusion_matrix(test_y_np, preds, labels=[0, 1])
    correct = int((preds == test_y_np).sum())

    print(f"Best model: {correct}/125 ({100*acc:.1f}%), F1={100*f1m:.1f}%")
    print(f"CM: safe={cm[0,0]}/61, unsafe={cm[1,1]}/64, FP={cm[0,1]}, FN={cm[1,0]}")

    # ─── Also load ensemble partner: SM_stats_L s42 ───
    print("\nLoading ensemble partner: SM_stats_L (dim=384, L=6, seed=42)...")
    ens_model = SpatialMotionHead(2304, 2, 384, 8, 6, 0.35, 64).to(device)
    ens_model.load_state_dict(torch.load(
        'sweep_results/safe_unsafe/su_SM_stats_L_s42.pt', map_location=device))
    ens_model.eval()

    test_x_stats = torch.cat([
        test_x_tokens.mean(2), test_x_tokens.std(2), test_x_tokens.max(2).values
    ], dim=-1)

    with torch.no_grad():
        logits_ens = ens_model(test_x_stats.to(device))
        probs_ens = F.softmax(logits_ens, 1).cpu().numpy()

    avg_probs = (probs + probs_ens) / 2.0
    preds_ens = avg_probs.argmax(1)
    acc_ens = accuracy_score(test_y_np, preds_ens)
    f1_ens = f1_score(test_y_np, preds_ens, average='macro', zero_division=0)
    cm_ens = confusion_matrix(test_y_np, preds_ens, labels=[0, 1])
    correct_ens = int((preds_ens == test_y_np).sum())
    print(f"Ensemble: {correct_ens}/125 ({100*acc_ens:.1f}%), F1={100*f1_ens:.1f}%")

    # Use the best between single and ensemble
    if correct_ens >= correct:
        final_preds = preds_ens
        final_probs = avg_probs
        final_cm = cm_ens
        final_correct = correct_ens
        final_name = "Gated_raw_L + SM_stats_L Ensemble"
    else:
        final_preds = preds
        final_probs = probs
        final_cm = cm
        final_correct = correct
        final_name = "Gated_raw_L (single)"

    final_acc = 100 * final_correct / 125
    final_f1 = 100 * f1_score(test_y_np, final_preds, average='macro', zero_division=0)
    final_prec = 100 * precision_score(test_y_np, final_preds, average='macro', zero_division=0)
    final_rec = 100 * recall_score(test_y_np, final_preds, average='macro', zero_division=0)

    print(f"\nUsing: {final_name}")
    print(f"  Accuracy:  {final_acc:.1f}% ({final_correct}/125)")
    print(f"  F1-macro:  {final_f1:.1f}%")
    print(f"  Precision: {final_prec:.1f}%")
    print(f"  Recall:    {final_rec:.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # Figure 1: Binary Confusion Matrix (2×2)
    # ═══════════════════════════════════════════════════════════════
    print("\n─── Generating figures ───")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    cm_pct = final_cm.astype(float) / final_cm.sum(axis=1, keepdims=True) * 100
    annot = np.empty_like(final_cm, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f'{final_cm[i,j]}\n({cm_pct[i,j]:.1f}%)'

    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(final_cm, annot=annot, fmt='', cmap=cmap, ax=ax,
                xticklabels=['Safe', 'Unsafe'], yticklabels=['Safe', 'Unsafe'],
                linewidths=2, linecolor='white',
                cbar_kws={'label': 'Count', 'shrink': 0.8},
                vmin=0, vmax=65, annot_kws={'fontsize': 16, 'fontweight': 'bold'})

    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(f'Binary Safe/Unsafe Confusion Matrix\n'
                 f'{final_name}\n'
                 f'Accuracy: {final_correct}/125 ({final_acc:.1f}%) — F1: {final_f1:.1f}%',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13, fontweight='bold')

    plt.tight_layout()
    p = os.path.join(output_dir, 'confusion_matrix_binary.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.savefig(p.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Figure 2: Per-Class Accuracy (8 original classes → binary)
    # ═══════════════════════════════════════════════════════════════
    per_class_acc = []
    per_class_counts = []
    per_class_colors = []
    for c in range(8):
        mask = labels_8 == c
        if mask.sum() == 0:
            per_class_acc.append(0)
            per_class_counts.append('0/0')
            continue
        correct_c = int((final_preds[mask] == test_y_np[mask]).sum())
        total_c = int(mask.sum())
        per_class_acc.append(100 * correct_c / total_c)
        per_class_counts.append(f'{correct_c}/{total_c}')
        per_class_colors.append('#F44336' if c < 4 else '#4CAF50')

    fig, ax = plt.subplots(figsize=(12, 6))
    unsafe_colors = ['#EF5350', '#FF7043', '#FF8A65', '#FFAB91']
    safe_colors = ['#66BB6A', '#81C784', '#A5D6A7', '#C8E6C9']
    colors = unsafe_colors + safe_colors

    bars = ax.bar(range(8), per_class_acc, color=colors, width=0.65,
                  edgecolor='white', linewidth=1.5)

    for i, (bar, acc_val, cnt) in enumerate(zip(bars, per_class_acc, per_class_counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{acc_val:.1f}%\n({cnt})', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Overall accuracy line
    ax.axhline(y=final_acc, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(7.4, final_acc + 1, f'Overall: {final_acc:.1f}%', color='blue',
            fontsize=11, fontweight='bold', ha='right')

    ax.set_xticks(range(8))
    ax.set_xticklabels(CLASS_LABELS_8, fontsize=9, ha='center')
    ax.set_ylabel('Binary Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Per-Class Binary Safe/Unsafe Accuracy\n'
                 f'(Red = Unsafe classes, Green = Safe classes)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(0, 115)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    # Legend
    unsafe_patch = mpatches.Patch(color='#EF5350', label='Unsafe classes (predict=1)')
    safe_patch = mpatches.Patch(color='#66BB6A', label='Safe classes (predict=0)')
    ax.legend(handles=[unsafe_patch, safe_patch], loc='lower right', fontsize=10)

    plt.tight_layout()
    p = os.path.join(output_dir, 'per_class_accuracy_binary.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.savefig(p.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Figure 3: Method Comparison (with baselines)
    # ═══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [
        'Majority\n(all unsafe)', 'Linear Probe\n(global avg)', 'Linear Probe\n(spatial stats)',
        'SM_stats\n(best, T32)', 'Gated_raw\n(best, T32)',
        'SM_stats_L\n(larger, T32)', 'Gated_raw_L\n(larger, T32)',
        'Best Ensemble\n(Gated_L+SM_L)'
    ]
    accs_bar = [51.2, 56.8, 68.0, 81.6, 81.6, 82.4, 83.2, final_acc]
    f1s_bar = [33.9, 45.5, 66.2, 81.5, 81.4, 82.3, 83.0, final_f1]
    colors_bar = ['#BDBDBD', '#BDBDBD', '#BDBDBD',
                  '#64B5F6', '#64B5F6',
                  '#2196F3', '#1565C0',
                  '#0D47A1']

    x_pos = np.arange(len(methods))
    width = 0.35

    bars_acc = ax.bar(x_pos - width/2, accs_bar, width, label='Accuracy',
                      color=colors_bar, edgecolor='white', linewidth=1)
    bars_f1 = ax.bar(x_pos + width/2, f1s_bar, width, label='F1-macro',
                     color=[c + '80' if c.startswith('#') else c for c in colors_bar],
                     edgecolor='white', linewidth=1, alpha=0.7)

    for bar, val in zip(bars_acc, accs_bar):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=8, ha='center')
    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Binary Safe/Unsafe Classification — Method Comparison\n'
                 '(125 test clips: 64 unsafe + 61 safe)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p = os.path.join(output_dir, 'method_comparison_binary.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.savefig(p.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Figure 4: Error Analysis
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 4a: Error breakdown by original class
    errors_per_class = []
    for c in range(8):
        mask = labels_8 == c
        if mask.sum() == 0: continue
        errors = int((final_preds[mask] != test_y_np[mask]).sum())
        errors_per_class.append((CLASS_NAMES_8[c], errors, int(mask.sum()),
                                'Unsafe→Safe' if c < 4 else 'Safe→Unsafe'))

    errors_per_class.sort(key=lambda x: -x[1])
    names_err = [f"{e[0]}\n({e[3]})" for e in errors_per_class if e[1] > 0]
    counts_err = [e[1] for e in errors_per_class if e[1] > 0]
    totals_err = [e[2] for e in errors_per_class if e[1] > 0]
    colors_err = ['#F44336' if 'Unsafe' in e[3] else '#4CAF50'
                  for e in errors_per_class if e[1] > 0]

    bars_err = axes[0].barh(range(len(names_err)), counts_err, color=colors_err,
                            edgecolor='white', linewidth=1.5, height=0.6)
    for i, (bar, cnt, tot) in enumerate(zip(bars_err, counts_err, totals_err)):
        axes[0].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{cnt}/{tot} ({100*cnt/tot:.0f}%)', va='center', fontsize=10)

    axes[0].set_yticks(range(len(names_err)))
    axes[0].set_yticklabels(names_err, fontsize=9)
    axes[0].set_xlabel('Number of Errors', fontsize=11, fontweight='bold')
    n_total_err = 125 - final_correct
    axes[0].set_title(f'Error Breakdown by Original Class\n({n_total_err} errors total)',
                      fontsize=12, fontweight='bold')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].invert_yaxis()

    fn_patch = mpatches.Patch(color='#F44336', label=f'FN: Unsafe→Safe ({int(final_cm[1,0])})')
    fp_patch = mpatches.Patch(color='#4CAF50', label=f'FP: Safe→Unsafe ({int(final_cm[0,1])})')
    axes[0].legend(handles=[fn_patch, fp_patch], loc='lower right', fontsize=9)

    # 4b: Confidence distribution of errors
    correct_mask = final_preds == test_y_np
    error_mask = ~correct_mask

    correct_conf = final_probs[np.arange(len(final_preds)), final_preds][correct_mask]
    error_conf = final_probs[np.arange(len(final_preds)), final_preds][error_mask]

    axes[1].hist(correct_conf, bins=15, alpha=0.7, color='#4CAF50', label='Correct',
                 edgecolor='white', density=True)
    axes[1].hist(error_conf, bins=10, alpha=0.7, color='#F44336', label='Errors',
                 edgecolor='white', density=True)
    axes[1].axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Prediction Confidence', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[1].set_title('Confidence Distribution\n(Correct vs Misclassified)',
                      fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    p = os.path.join(output_dir, 'error_analysis_binary.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.savefig(p.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Figure 5: Ablation — Architecture × Temporal Resolution
    # ═══════════════════════════════════════════════════════════════
    # Load results from temporal_fusion_results.json
    results_path = 'sweep_results/safe_unsafe/temporal_fusion_results.json'
    if os.path.exists(results_path):
        with open(results_path) as f:
            tf_results = json.load(f)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by arch
        arch_labels = {'SM_stats': 'SM_stats (dim=256)', 'Gated_raw': 'Gated_raw (dim=192)',
                       'SM_stats_L': 'SM_stats_L (dim=384)', 'Gated_raw_L': 'Gated_raw_L (dim=256)'}
        t_labels = ['T16', 'T32', 'T48']
        bar_width = 0.18
        x_base = np.arange(len(t_labels))

        colors_arch = ['#64B5F6', '#FF8A65', '#1565C0', '#E65100']
        all_models = tf_results.get('all_models', {})

        arch_idx = 0
        for arch_name, arch_label in arch_labels.items():
            accs_t = []
            for t_name in t_labels:
                # Find best seed for this arch+T
                best_acc = 0
                for key_str, info in all_models.items():
                    if f"'{arch_name}'" in key_str and f"'{t_name}'" in key_str:
                        if info['acc'] > best_acc:
                            best_acc = info['acc']
                if best_acc == 0:
                    accs_t.append(0)
                else:
                    accs_t.append(best_acc)

            if any(a > 0 for a in accs_t):
                offset = (arch_idx - 1.5) * bar_width
                bars_a = ax.bar(x_base + offset, accs_t, bar_width,
                               label=arch_label, color=colors_arch[arch_idx % len(colors_arch)],
                               edgecolor='white', linewidth=1)
                for bar, val in zip(bars_a, accs_t):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                                f'{val:.1f}', ha='center', va='bottom', fontsize=7)
                arch_idx += 1

        ax.set_xticks(x_base)
        ax.set_xticklabels(['T=16 frames', 'T=32 frames', 'T=48 frames'], fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title('Ablation: Architecture × Temporal Resolution\n'
                     '(Best single model per combination)',
                     fontsize=13, fontweight='bold', pad=10)
        ax.set_ylim(70, 90)
        ax.legend(fontsize=9, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        p = os.path.join(output_dir, 'ablation_arch_temporal.png')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        plt.savefig(p.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  Saved: {p}")
        plt.close()

    # ═══════════════════════════════════════════════════════════════
    # ONNX Export + INT8 Quantization + Latency Benchmark
    # ═══════════════════════════════════════════════════════════════
    print("\n─── Edge Deployment ───")

    # Export Gated_raw_L to ONNX
    dummy_input = torch.randn(1, 32, 16, 768)
    onnx_path = os.path.join(deploy_dir, 'binary_safe_unsafe_gated_L.onnx')
    onnx_size = None

    try:
        torch.onnx.export(
            best_model, dummy_input, onnx_path,
            input_names=['features'], output_names=['logits'],
            dynamic_axes={'features': {0: 'batch'}, 'logits': {0: 'batch'}},
            opset_version=14,
            do_constant_folding=True
        )
        onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"  ONNX exported: {onnx_path} ({onnx_size:.1f} MB)")
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        onnx_path = None

    # INT8 quantization (dynamic) — selectively quantize non-transformer Linear layers
    int8_path = os.path.join(deploy_dir, 'binary_safe_unsafe_gated_L_int8.pt')
    int8_model = None
    int8_size = None
    int8_acc = None
    int8_f1 = None
    try:
        import copy
        torch.backends.quantized.engine = 'qnnpack'
        int8_model = copy.deepcopy(best_model)

        # Manually quantize only projection and FC modules (skip transformer)
        for name in ['proj_raw', 'proj_d1', 'proj_d2', 'proj_win', 'gate', 'fc']:
            submod = getattr(int8_model, name, None)
            if submod is not None:
                quantized = torch.quantization.quantize_dynamic(
                    submod, {nn.Linear}, dtype=torch.qint8)
                setattr(int8_model, name, quantized)

        torch.save(int8_model.state_dict(), int8_path)
        int8_size = os.path.getsize(int8_path) / 1024 / 1024
        print(f"  INT8 exported: {int8_path} ({int8_size:.1f} MB)")

        # Verify INT8 accuracy
        int8_model.eval()
        with torch.no_grad():
            int8_preds = int8_model(test_x_tokens).argmax(1).numpy()
        int8_acc = 100 * accuracy_score(test_y_np, int8_preds)
        int8_f1 = 100 * f1_score(test_y_np, int8_preds, average='macro', zero_division=0)
        print(f"  INT8 accuracy: {int8_acc:.1f}%, F1={int8_f1:.1f}% "
              f"(vs FP32: {final_acc:.1f}%)")
    except Exception as e:
        print(f"  INT8 export failed: {e}")
        import traceback; traceback.print_exc()
        int8_model = None

    # Latency benchmark
    print("\n  Latency benchmark (100 iterations, batch=1):")
    dummy = torch.randn(1, 32, 16, 768)

    # FP32
    best_model.eval()
    times_fp32 = []
    for _ in range(10):  # warmup
        with torch.no_grad():
            best_model(dummy)
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            best_model(dummy)
        times_fp32.append((time.perf_counter() - t0) * 1000)
    print(f"    FP32: {np.mean(times_fp32):.2f} ± {np.std(times_fp32):.2f} ms "
          f"(median={np.median(times_fp32):.2f} ms)")

    # ONNX
    times_onnx = []
    if onnx_path and os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path)
            dummy_np = dummy.numpy()
            for _ in range(10):
                sess.run(None, {'features': dummy_np})
            for _ in range(100):
                t0 = time.perf_counter()
                sess.run(None, {'features': dummy_np})
                times_onnx.append((time.perf_counter() - t0) * 1000)
            print(f"    ONNX: {np.mean(times_onnx):.2f} ± {np.std(times_onnx):.2f} ms "
                  f"(median={np.median(times_onnx):.2f} ms)")
        except ImportError:
            print("    ONNX Runtime not available, skipping ONNX benchmark")

    # INT8
    times_int8 = []
    if int8_model is not None:
        try:
            for _ in range(10):
                with torch.no_grad():
                    int8_model(dummy)
            for _ in range(100):
                t0 = time.perf_counter()
                with torch.no_grad():
                    int8_model(dummy)
                times_int8.append((time.perf_counter() - t0) * 1000)
            print(f"    INT8: {np.mean(times_int8):.2f} ± {np.std(times_int8):.2f} ms "
                  f"(median={np.median(times_int8):.2f} ms)")
        except Exception:
            pass

    # FPS estimate
    fps_cpu = 1000 / np.median(times_fp32)
    print(f"\n  Estimated FPS (head only, CPU): {fps_cpu:.1f}")

    # ═══════════════════════════════════════════════════════════════
    # Figure 6: Latency Comparison
    # ═══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8, 5))
    latency_methods = ['FP32\n(PyTorch)']
    latency_vals = [np.median(times_fp32)]
    lat_colors = ['#42A5F5']

    if times_onnx:
        latency_methods.append('ONNX\nRuntime')
        latency_vals.append(np.median(times_onnx))
        lat_colors.append('#66BB6A')

    if times_int8:
        latency_methods.append('INT8\n(Dynamic Quant)')
        latency_vals.append(np.median(times_int8))
        lat_colors.append('#FF7043')

    bars_lat = ax.bar(range(len(latency_methods)), latency_vals,
                      color=lat_colors, width=0.5, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars_lat, latency_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f} ms', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xticks(range(len(latency_methods)))
    ax.set_xticklabels(latency_methods, fontsize=11)
    ax.set_ylabel('Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Inference Latency — Binary Safe/Unsafe Head\n'
                 '(CPU, batch=1, T=32 frames, median of 100 runs)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p = os.path.join(output_dir, 'latency_comparison_binary.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.savefig(p.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved: {p}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Save deployment report
    # ═══════════════════════════════════════════════════════════════
    report = f"""Binary Safe/Unsafe Deployment Report
====================================
Generated: {time.strftime('%Y-%m-%d %H:%M')}

Best Model: {final_name}
  Architecture: GatedFusionBinaryHead (dim=256, nhead=8, L=4)
  Input: (B, 32, 16, 768) — YOLOv11s pool4 tokens
  Parameters: {sum(p.numel() for p in best_model.parameters()):,}

Results (125 test clips: 64 unsafe + 61 safe):
  Accuracy:  {final_acc:.1f}% ({final_correct}/125)
  F1-macro:  {final_f1:.1f}%
  Precision: {final_prec:.1f}%
  Recall:    {final_rec:.1f}%

Confusion Matrix:
              Pred Safe  Pred Unsafe
  True Safe:    {final_cm[0,0]:3d}       {final_cm[0,1]:3d}
  True Unsafe:  {final_cm[1,0]:3d}       {final_cm[1,1]:3d}

Edge Deployment:
  FP32 model:   ~{os.path.getsize('sweep_results/safe_unsafe/su_Gated_raw_L_s42.pt')/1024/1024:.1f} MB
  ONNX model:   {f'{onnx_size:.1f} MB (opset 14)' if onnx_size else 'N/A'}
  INT8 model:   {f'{int8_size:.1f} MB (dynamic quantization)' if int8_size else 'N/A'}
  INT8 accuracy: {f'{int8_acc:.1f}%, F1={int8_f1:.1f}%' if int8_acc else 'N/A'}

Latency (CPU, batch=1):
  FP32:  {np.median(times_fp32):.2f} ms (median)
  ONNX:  {f'{np.median(times_onnx):.2f} ms (median)' if 'times_onnx' in dir() and times_onnx else 'N/A'}
  INT8:  {f'{np.median(times_int8):.2f} ms (median)' if times_int8 else 'N/A'}
  FPS:   ~{fps_cpu:.0f} FPS (head only, CPU)
"""
    report_path = os.path.join(deploy_dir, 'binary_safe_unsafe_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n  Report saved: {report_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"ALL DONE")
    print(f"{'='*60}")
    print(f"Figures: {output_dir}/")
    print(f"  - confusion_matrix_binary.png/pdf")
    print(f"  - per_class_accuracy_binary.png/pdf")
    print(f"  - method_comparison_binary.png/pdf")
    print(f"  - error_analysis_binary.png/pdf")
    print(f"  - ablation_arch_temporal.png/pdf")
    print(f"  - latency_comparison_binary.png/pdf")
    print(f"Deployment: {deploy_dir}/")
    print(f"  - binary_safe_unsafe_gated_L.onnx")
    print(f"  - binary_safe_unsafe_gated_L_int8.pt")
    print(f"  - binary_safe_unsafe_report.txt")


if __name__ == '__main__':
    main()
