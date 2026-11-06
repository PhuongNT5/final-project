#!/usr/bin/env python3
"""Pairwise-vs-B1 bootstrap + McNemar + Benjamini-Hochberg for the iSafetyBench headline table.

GAP: Bang 6.1 ranks 7 models by point estimate over a ~2.9 pp spread but only ONE pair (G14, ESTA vs
B1w) was ever significance-tested; the other 6 rows were never tested against B1, the config
actually deployed (Ch.5). Runs that missing family vs B1, paired on clip_id: source-level cluster
bootstrap + McNemar (wrong unit, disclosed) + BH q=0.05. POST-HOC: this 6-pair family was fixed
after every OOF file already existed (not a pre-registered gate), so BH here is post-hoc, not a
pre-registered multiplicity control (see "note" below).

Resampling unit: source video (`fold_id`, cross-checked against the canonical clip->source grouping
in isb-clip-manifest.json's `views`, never filename regex). Uses eval_utils.cluster_bootstrap for the
CI (hides its raw draws, so `_boot_draws` redraws the identical resample to recover a p-value, and
`compare_row` asserts its CI matches the helper bit for bit) and mcnemar_exact for the McNemar line.

Run: cd <repo root> && PYTHONPATH=. python3 tools/compute-table3-pairwise-bh.py (seed=0, deterministic)
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_utils.cluster_bootstrap import balanced_accuracy, paired_metric_cluster_bootstrap
from eval_utils.paired_tests import mcnemar_exact

TOK = 'sweep_results/relation_tokens'
MANIFEST = 'kaggle/isb-clip-manifest.json'
OUT_DIR = 'sweep_results/review_261101'
OUT = f'{OUT_DIR}/table3-pairwise-bh.json'
SCRIPT = 'tools/compute-table3-pairwise-bh.py'
BASELINE = 'B1'

# Every Bang 6.1 row -> its OOF file, by the one naming pattern all 7 share. ESTA_26ft/ESTA_v11ft/
# ESTA_smoke/B1_REPROCHECK/isbclipL_B1 exist on disk but are NOT Bang 6.1 rows (other sweep variants
# and the environment-drift control, the filename trap) -- excluded by listing only these 7 names.
ROWS = {m: f'{TOK}/enc_isbclip_{m}_lso-1100_headline-none_oof.json'
        for m in ('B1', 'B1w', 'TSF', 'ESTA', 'B1xl', 'ESTA_y26', 'B2')}
VI_NAME = {'B1': 'B1 (toàn khung, 1 lớp) [TK, đã triển khai]', 'B1w': 'B1w (toàn khung, 2 lớp)',
           'TSF': 'TSF (hợp nhất hai luồng)', 'ESTA': 'ESTA (phân rã theo thực thể, @YOLOv11s)',
           'B1xl': 'B1xl (toàn khung, đối chứng dung lượng)',
           'ESTA_y26': 'ESTA @ bộ phát hiện YOLO26s', 'B2': 'B2 (VideoMAE-base Kinetics)'}
PRINTED_BA = {'B1': 84.2, 'B1w': 84.4, 'TSF': 84.1, 'ESTA': 83.8,  # Bang 6.1, 1dp; identity check only
              'B1xl': 82.7, 'ESTA_y26': 81.6, 'B2': 81.5}
N_BOOT, SEED, ALPHA, Q_BH = 5000, 0, 0.05, 0.05

def load_oof(path):
    return {r['clip_id']: r for r in json.load(open(path))['rows']}

def _boot_draws(y_true, pred_a, pred_b, groups, n_boot=N_BOOT, seed=SEED):
    """Same resampling as paired_metric_cluster_bootstrap, but returns the raw per-draw diffs
    (not exposed by that helper) so a p-value can be read off the SAME draws as the published CI."""
    y_true, pred_a, pred_b, groups = map(np.asarray, (y_true, pred_a, pred_b, groups))
    uniq = np.unique(groups)
    idx_by_group = [np.where(groups == g)[0] for g in uniq]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_group[p] for p in pick])
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append((balanced_accuracy(y_true[idx], pred_a[idx])
                      - balanced_accuracy(y_true[idx], pred_b[idx])) * 100.0)
    return np.array(boots)

def bootstrap_asl(boots):
    """Two-sided achieved significance level, floored at 1/B."""
    b = len(boots)
    return float(max(min(2 * min((boots <= 0).sum(), (boots >= 0).sum()) / b, 1.0), 1.0 / b))

def benjamini_hochberg(names_pvals, q):
    """Standard BH step-up over [(name, p)]. -> (dict name->rejected, m, k)."""
    m = len(names_pvals)
    ordered = sorted(names_pvals, key=lambda kv: kv[1])
    k = 0
    for i, (_, p) in enumerate(ordered, 1):
        if p <= q * i / m:
            k = i
    return {name: (i <= k) for i, (name, _) in enumerate(ordered, 1)}, m, k

def compare_row(name, base, data, canon):
    d = data[name]
    ids = sorted(set(base) & set(d))
    pairing_valid = len(ids) == len(base) == len(d)
    y_mismatch = sum(d[c]['y_true'] != base[c]['y_true'] for c in ids)
    fold_mismatch = sum(str(d[c]['fold_id']) != str(base[c]['fold_id']) for c in ids)
    canon_mismatch = (sum(str(base[c]['fold_id']) != str(canon.get(c)) for c in ids)
                      + sum(str(d[c]['fold_id']) != str(canon.get(c)) for c in ids))
    identity_ok = pairing_valid and y_mismatch == 0 and fold_mismatch == 0 and canon_mismatch == 0
    y = np.array([base[c]['y_true'] for c in ids])
    pa = np.array([d[c]['y_pred'] for c in ids])       # candidate row
    pb = np.array([base[c]['y_pred'] for c in ids])    # B1 baseline
    g = np.array([str(base[c]['fold_id']) for c in ids])
    ba_a, ba_b = balanced_accuracy(y, pa) * 100.0, balanced_accuracy(y, pb) * 100.0
    identity_ok = identity_ok and abs(round(ba_a, 1) - PRINTED_BA[name]) < 0.05
    diff, lo, hi, n_valid = paired_metric_cluster_bootstrap(y, pa, pb, g, n_boot=N_BOOT,
                                                            seed=SEED, alpha=ALPHA)
    boots = _boot_draws(y, pa, pb, g)
    lo2, hi2 = np.percentile(boots, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    assert len(boots) == n_valid and abs(lo - lo2) < 1e-9 and abs(hi - hi2) < 1e-9, (
        f'{name}: redrawn bootstrap does not reproduce the library CI -- refusing the derived p')
    p_boot = bootstrap_asl(boots)
    mc = mcnemar_exact(pa == y, pb == y)
    return {
        'row': name, 'name_vi': VI_NAME[name], 'baseline': BASELINE,
        'identity_verified': identity_ok, 'n_clips': len(ids), 'n_sources': int(len(set(g))),
        'ba_row_pct': ba_a, 'ba_baseline_pct': ba_b,
        'ba_row_pct_display': f'{ba_a:.1f}%', 'ba_baseline_pct_display': f'{ba_b:.1f}%',
        'diff_pp': diff, 'diff_pp_display': f'{diff:+.1f} pp',
        'cluster_bootstrap': {'unit': 'source_video', 'n_boot': N_BOOT, 'seed': SEED,
                               'alpha': ALPHA, 'kind': 'percentile', 'n_valid_draws': int(n_valid),
                               'ci_lo_pp': lo, 'ci_hi_pp': hi,
                               'ci_excludes_zero': bool(lo > 0 or hi < 0),
                               'p_value_two_sided': p_boot},
        'mcnemar_clip_level': {'b': mc['b'], 'c': mc['c'], 'n_discordant': mc['n_discordant'],
                                'p_value': mc['p_value'], 'tag': 'WRONG_UNIT',
                                'why_shown': ('McNemar ở mức clip coi 1.100 clip là độc lập; đơn '
                                              'vị độc lập thật là 151 nguồn, nên p này chỉ là đối '
                                              'chứng chẩn đoán, không phải kiểm định chính (xem '
                                              'cluster_bootstrap.p_value_two_sided).')},
    }

def reading_for(r):
    diff = r['diff_pp']
    lo, hi = r['cluster_bootstrap']['ci_lo_pp'], r['cluster_bootstrap']['ci_hi_pp']
    excl, rej = r['cluster_bootstrap']['ci_excludes_zero'], r['bh_reject_q05']
    if excl and diff < 0:
        sep = f"tách khỏi B1 theo hướng KÉM HƠN (CI theo cụm 95% [{lo:+.1f}, {hi:+.1f}] loại trừ 0)"
    elif excl:
        sep = f"tách khỏi B1 theo hướng VƯỢT TRỘI (CI theo cụm 95% [{lo:+.1f}, {hi:+.1f}] loại trừ 0)"
    else:
        sep = f"không tách khỏi B1 (CI theo cụm 95% [{lo:+.1f}, {hi:+.1f}] chứa 0)"
    bh = ("sau Benjamini-Hochberg q=0.05 vẫn còn ý nghĩa" if (excl and rej) else
          "nhưng KHÔNG còn ý nghĩa sau hiệu chỉnh Benjamini-Hochberg q=0.05" if excl else
          "Benjamini-Hochberg không cần xét vì CI đã chứa 0")
    return f"{r['name_vi']}: điểm ước lượng {diff:+.1f} pp so với B1, {sep}; {bh}."

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = json.load(open(MANIFEST))
    canon = dict(zip(manifest['clip_ids'], manifest['views']))
    data = {name: load_oof(path) for name, path in ROWS.items()}
    base = data[BASELINE]
    rows = [compare_row(name, base, data, canon) for name in ROWS if name != BASELINE]
    assert all(r['identity_verified'] for r in rows), 'row identity / pairing check failed'
    rejected, m, k = benjamini_hochberg([(r['row'], r['cluster_bootstrap']['p_value_two_sided'])
                                         for r in rows], Q_BH)
    for r in rows:
        r['bh_reject_q05'] = rejected[r['row']]
    reading = {r['row']: reading_for(r) for r in rows}
    n_sep = sum(r['cluster_bootstrap']['ci_excludes_zero'] for r in rows)
    n_sep_bh = sum(r['bh_reject_q05'] for r in rows)
    overall = (f"Trên {len(rows)} hàng đối với B1 (triển khai), {n_sep} hàng có CI theo cụm loại "
               f"trừ 0 trước hiệu chỉnh và {n_sep_bh} hàng còn lại sau Benjamini-Hochberg q=0.05; "
               f"thứ hạng theo điểm ước lượng trong Bảng 6.1 (trải ~2.9 pp) KHÔNG tương đương thứ "
               f"hạng có kiểm định — phần lớn chênh lệch 0.1-0.5 pp giữa các hàng liền kề là nhiễu "
               f"thống kê dưới phép lấy mẫu theo nguồn, không phải bằng chứng xếp hạng.")
    note = ('POST-FREEZE (after the 2026-10-05 number freeze), POST-HOC / NOT A PRE-REGISTERED '
            'GATE: these 6 B1-vs-row comparisons were defined after all 7 OOF files already '
            'existed, so BH here is a post-hoc multiplicity control, not a pre-registered one. May '
            'be used to show which Bang 6.1 rows separate from deployed B1 under proper '
            'source-level resampling with multiplicity control; may NOT be used as a new '
            'pre-registered gate verdict, may NOT be averaged/compared with any Unsafe-Net-lane '
            '(64/516-clip, 4-class) number, and does not supersede the pre-registered G14 (ESTA '
            'vs B1w) verdict in Bang 6.2.')
    lane = ('iSafetyBench binary (Safe/Unsafe), leave-source-out, 1,100 clips, 151 sources '
            '(canonical grouping: isb-clip-manifest.json "views"); every number below belongs to '
            'this lane only -- never compare to the Unsafe-Net 64/516-clip 4-class lane (sec 6.1.1).')
    fam_def = ('the 6 row-vs-B1 comparisons this script runs (Bang 6.1 rows minus B1 itself); '
               'family fixed AFTER the data existed -- post-hoc, not pre-registered')
    artifact = {
        'note': note, 'lane': lane, 'resampling_unit': 'source_video', 'n_boot': N_BOOT,
        'seed': SEED, 'alpha': ALPHA, 'source_artifacts': [MANIFEST] + list(ROWS.values()),
        'script': SCRIPT,
        'results': {
            'baseline': BASELINE, 'baseline_ba_pct': rows[0]['ba_baseline_pct'],
            'baseline_ba_pct_display': rows[0]['ba_baseline_pct_display'], 'rows': rows,
            'multiple_comparisons': {
                'method': 'Benjamini-Hochberg', 'q': Q_BH, 'm': m, 'n_rejected': k,
                'family_definition': fam_def,
                'rejected_rows': [r['row'] for r in rows if r['bh_reject_q05']],
            },
        },
        'reading': {**reading, 'overall': overall},
    }
    with open(OUT, 'w') as f:
        json.dump(artifact, f, indent=1, ensure_ascii=False, sort_keys=False)
    print(f'wrote {OUT}')
    for r in rows:
        cb = r['cluster_bootstrap']
        print(f"{r['row']:10s} diff={r['diff_pp']:+.3f}pp CI=[{cb['ci_lo_pp']:+.2f},"
              f"{cb['ci_hi_pp']:+.2f}] p={cb['p_value_two_sided']:.4f} BH={r['bh_reject_q05']}")

if __name__ == '__main__':
    main()
