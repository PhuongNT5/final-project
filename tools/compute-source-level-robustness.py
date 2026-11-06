"""Source-level robustness audit for the iSafetyBench B1 headline (POST-FREEZE, POST-HOC).

Two structural facts, not currently carried side by side in the thesis, about the 1,100-clip binary
iSafetyBench leave-source-out (LSO) lane: (a) DAMAGING -- 0/151 sources carry both labels, so
holding out a source holds out a whole label block, and clip counts per source are very uneven (one
source alone supplies ~102/1,100 clips); (b) FAVOURABLE -- macro-averaging balanced accuracy (BA)
over sources (each weighted once, not once per clip) comes out slightly ABOVE the clip-level
headline, i.e. the headline is not an artifact of the big sources.

Reads only the already-frozen B1 out-of-fold predictions + the manifest's grouping; retrains
nothing, changes no headline number. Fixed seeds -> byte-identical JSON on re-run.
Usage: cd <repo root> && PYTHONPATH=. python3 tools/compute-source-level-robustness.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
import isafetybench_dataset as ISB
from eval_utils.cluster_bootstrap import balanced_accuracy

MANIFEST, SCRIPT = 'kaggle/isb-clip-manifest.json', 'tools/compute-source-level-robustness.py'
OOF = 'sweep_results/relation_tokens/enc_isbclip_B1_lso-1100_headline-none_oof.json'
OUT = Path('sweep_results/review_261101/source-level-robustness.json')
SEED, N_BOOT, ALPHA = 0, 5000, 0.05


def gini(counts):
    """Gini coefficient of a non-negative count vector (0 = all sources equal size)."""
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float((2.0 * np.sum(i * x) - (n + 1) * x.sum()) / (n * x.sum()))


def anova_icc(error, groups):
    """Unbalanced one-way ANOVA ICC (Fisher 1958; Donner 1986) on the binary per-clip error
    indicator: share of error variance BETWEEN sources vs within. 23/151 sources have 1 clip
    (0 within-source variance by construction) -- reported as a caveat, not hidden."""
    e = np.asarray(error, dtype=float)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    k, N, grand = len(uniq), len(e), e.mean()
    sizes = np.bincount(inv).astype(float)
    group_means = np.bincount(inv, weights=e) / sizes
    ssb = float(np.sum(sizes * (group_means - grand) ** 2))
    ssw = float(np.sum((e - group_means[inv]) ** 2))
    df_b, df_w = k - 1, N - k
    msb, msw = ssb / df_b, (ssw / df_w if df_w > 0 else float('nan'))
    n0 = (N - np.sum(sizes ** 2) / N) / df_b
    denom = msb + (n0 - 1) * msw
    return {'icc': float((msb - msw) / denom) if denom else float('nan'), 'msb': msb, 'msw': msw,
            'k_sources': int(k), 'n0_avg_cluster_size': float(n0), 'df_between': int(df_b),
            'df_within': int(df_w)}


def main():
    manifest = json.load(open(MANIFEST))
    clip_to_src = dict(zip(manifest['clip_ids'], manifest['views']))
    clip_to_lab = dict(zip(manifest['clip_ids'], manifest['labels']))
    rows = json.load(open(OOF))['rows']
    assert len(rows) == 1100, f'expected 1,100 clips, got {len(rows)}'
    for r in rows:  # canonical grouping (manifest 'views') must agree with the artifact's fold_id
        assert clip_to_src[r['clip_id']] == r['fold_id'], f"source mismatch at {r['clip_id']}"
        assert clip_to_lab[r['clip_id']] == r['y_true'], f"label mismatch at {r['clip_id']}"

    clip_ids = [r['clip_id'] for r in rows]
    y = np.array([r['y_true'] for r in rows])
    p = np.array([r['y_pred'] for r in rows])
    src = np.array([clip_to_src[c] for c in clip_ids])
    uniq_src = sorted(set(src))
    k_src = len(uniq_src)
    counts = {s: int((src == s).sum()) for s in uniq_src}
    labels = {s: int(y[src == s][0]) for s in uniq_src}
    both = [s for s in uniq_src if len(set(y[src == s].tolist())) > 1]
    acc_by_src = {s: float((y[src == s] == p[src == s]).mean()) for s in uniq_src}
    correct_n = {s: int((y[src == s] == p[src == s]).sum()) for s in uniq_src}
    top10 = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    top5 = [s for s, _ in top10[:5]]
    normal_src = [s for s in uniq_src if labels[s] == 0]
    hazard_src = [s for s in uniq_src if labels[s] == 1]
    clip_ba = balanced_accuracy(y, p) * 100
    macc_normal = float(np.mean([acc_by_src[s] for s in normal_src])) * 100
    macc_hazard = float(np.mean([acc_by_src[s] for s in hazard_src])) * 100
    src_macro_ba = (macc_normal + macc_hazard) / 2.0
    mean_acc_all = float(np.mean(list(acc_by_src.values()))) * 100

    # CI for source-macro BA: resample SOURCES (not clips) w/ replacement, k_src picks, from the
    # precomputed per-source accuracy/label table (no retraining) -- mirrors the resample-the-
    # grouping-unit idea of eval_utils.cluster_bootstrap, but that helper's metric(y_true, pred)
    # can't express a per-source-averaged statistic, hence this bespoke loop (draws losing a label
    # stratum are discarded, same disclosure convention as that helper).
    rng = np.random.default_rng(SEED)
    accs_a = np.array([acc_by_src[s] for s in uniq_src])
    labs_a = np.array([labels[s] for s in uniq_src])
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, k_src, k_src)
        a, l = accs_a[pick], labs_a[pick]
        if len(set(l.tolist())) < 2:
            continue
        boots.append((a[l == 0].mean() + a[l == 1].mean()) / 2.0 * 100)
    ci_lo, ci_hi = (float(x) for x in np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]))

    # Leave-one-source-out: recompute the pooled headline BA with the source's clips removed from
    # scoring (predictions already exist out-of-fold; nothing is retrained).
    loso = []
    for s in uniq_src:
        mask = src != s
        ba_wo = balanced_accuracy(y[mask], p[mask]) * 100
        loso.append({'source': s, 'n_clips': counts[s], 'label': labels[s],
                     'ba_without_source_pct': ba_wo, 'leverage_pp': ba_wo - clip_ba})
    loso_rank = sorted(loso, key=lambda r: -abs(r['leverage_pp']))
    ba_wo_vals = [r['ba_without_source_pct'] for r in loso]
    src_102 = next(r for r in loso if r['source'] == max(counts, key=counts.get))
    n_full_correct = sum(1 for s in uniq_src if correct_n[s] == counts[s])
    n_full_wrong = sum(1 for s in uniq_src if correct_n[s] == 0)
    icc = anova_icc((y != p).astype(int), src)
    _, _, yt, groups, tok_clip_ids = ISB.load('isb_')
    splits = ISB.folds(yt, groups, n_splits=10, seed=SEED)
    tok_y = yt.numpy()
    pred_by_clip = {r['clip_id']: r['y_pred'] for r in rows}
    pred_aligned = np.array([pred_by_clip[str(c)] for c in tok_clip_ids])
    per_fold = [balanced_accuracy(tok_y[te], pred_aligned[te]) * 100 for _, te in splits]
    big = max(counts.values())
    results = {
        'source_census': {
            'n_sources': k_src, 'n_clips': len(clip_ids),
            'clip_count_distribution': {'min': min(counts.values()),
                                         'median': float(np.median(list(counts.values()))),
                                         'max': big, 'mean': float(np.mean(list(counts.values())))},
            'gini_concentration': gini(list(counts.values())),
            'top10_sources_by_clip_count': [{'source': s, 'n_clips': n, 'label': labels[s]} for s, n in top10],
            'top5_sources_share_of_corpus_pct': 100.0 * sum(counts[s] for s in top5) / len(clip_ids),
            'label_purity': {'n_sources_label_normal': len(normal_src), 'n_sources_label_hazard': len(hazard_src),
                              'n_sources_both_labels': len(both),
                              'note': 'both_labels is COMPUTED, not assumed; 0 here reproduces thesis R41.'},
        },
        'clip_level_vs_source_macro_ba': {
            'clip_level_ba_pct': clip_ba, 'clip_level_ba_display': f'{clip_ba:.2f}%',
            'unweighted_mean_accuracy_over_all_sources_pct': mean_acc_all,
            'source_macro_ba': {
                'mean_accuracy_normal_label_sources_pct': macc_normal,
                'mean_accuracy_hazard_label_sources_pct': macc_hazard,
                'source_macro_ba_pct': src_macro_ba, 'source_macro_ba_display': f'{src_macro_ba:.2f}%',
                'diff_vs_clip_level_pp': src_macro_ba - clip_ba,
                'bootstrap_ci_pct': [ci_lo, ci_hi], 'n_valid_draws': len(boots),
            },
            'per_source_balanced_accuracy': {
                'n_sources_with_both_classes': len(both),
                'note': 'Undefined for every source: a per-source BA needs both class recalls inside ONE source, but 0/151 sources carry both labels (R41), so no source supplies the second recall. source_macro_ba above is the computable source-weighted analogue: average accuracy within each label stratum of sources first, then average the two strata.',
            },
        },
        'leave_one_source_out_leverage': {
            'full_headline_ba_pct': clip_ba,
            'ba_without_source_min_pct': float(min(ba_wo_vals)), 'ba_without_source_max_pct': float(max(ba_wo_vals)),
            'swing_pp': float(max(ba_wo_vals) - min(ba_wo_vals)),
            'top10_by_abs_leverage': loso_rank[:10], 'the_102_clip_source': src_102,
        },
        'error_clustering_by_source': {
            'n_sources_fully_correct': n_full_correct, 'n_sources_fully_wrong': n_full_wrong,
            'n_sources_mixed': k_src - n_full_correct - n_full_wrong,
            'icc_error_by_source': {**icc,
                'estimator': 'unbalanced one-way ANOVA ICC (Fisher 1958 / Donner 1986), binary error indicator',
                'caveat': '23/151 sources have exactly 1 clip, contributing 0 within-source variance by construction.'},
        },
        'per_fold_ba_spread_10_lso_folds': {
            'protocol': 'StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0) via isafetybench_dataset.folds',
            'per_fold_ba_pct': per_fold, 'min_pct': float(min(per_fold)), 'max_pct': float(max(per_fold)),
            'spread_pp': float(max(per_fold) - min(per_fold)),
        },
        'doi_dap': {
            'damaging_source_label_confound': [
                'Hạn chế cần nêu thẳng: trên iSafetyBench, 0/151 nguồn mang cả hai nhãn, nên danh tính nguồn là một proxy nhãn hoàn hảo và giữ ra một nguồn cũng giữ ra trọn một khối nhãn.',
                f'Phân phối số clip trên mỗi nguồn rất lệch — một nguồn ({big} clip) chiếm {100 * big / len(clip_ids):.1f}% của cả 1,100 clip — nên một fold rơi đúng vào nguồn này không so sánh được với các fold khác.',
            ],
            'favourable_source_macro_robustness': [
                f'Phản hồi cho hạn chế trên: tiêu đề 84.2% không phải lạm phát bởi các nguồn lớn — macro-trung bình độ chính xác cân bằng theo từng nguồn (mỗi nguồn tính một lần, không theo clip) đạt {src_macro_ba:.2f}%, CAO HƠN mức theo clip {clip_ba:.2f}% {src_macro_ba - clip_ba:+.2f} điểm phần trăm.',
                f'Khoảng tin cậy 95% theo bootstrap cấp nguồn (resample {k_src} nguồn, {N_BOOT} lượt, {len(boots)} lượt hợp lệ) cho con số macro-theo-nguồn là [{ci_lo:.2f}%, {ci_hi:.2f}%].',
            ],
        },
    }

    doc = {
        'note': 'POST-FREEZE (computed 2026-11-01, after the 2026-10-05 number freeze); POST-HOC / NOT A PRE-REGISTERED GATE. May be used to DESCRIBE structural properties of the already-frozen B1 headline (source concentration, label purity, source-macro BA, leave-one-source-out leverage, error clustering, per-fold spread). May NOT be used to adjust, re-certify, or replace 84.2%/84.17%, to retrain, or to compare across the iSafetyBench 1,100-clip binary lane and the Unsafe-Net 64/516-clip 4-class lane.',
        'lane': 'iSafetyBench, 1,100 clips, binary (hazard/normal), leave-source-out, 151 sources (LSO-1100) -- same lane as thesis Table 6.1 / Bang 6.1',
        'resampling_unit': 'source', 'n_boot': N_BOOT, 'seed': SEED, 'alpha': ALPHA,
        'source_artifacts': [MANIFEST, OOF, 'sweep_results/relation_tokens/tokens_isb_all.pt', 'isafetybench_dataset.py'],
        'script': SCRIPT,
        'results': results,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=False, ensure_ascii=False) + '\n')
    print(f'wrote {OUT} ({OUT.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
