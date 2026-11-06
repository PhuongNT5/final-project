#!/usr/bin/env python3
"""repro-band: disclosure table for the headline BA non-reproducibility gap (POST-FREEZE, POST-HOC).

Recomputes BA/macro-F1/confusion/per-fold BA from the on-disk OOF artifacts for the published B1
headline run and its post-freeze reruns (thesis chuong-04 S4.5.6). Single lane: iSafetyBench
1,100-clip binary leave-source-out -- never the Unsafe-Net lane. Deterministic (seed fixed for the
one bootstrap step) -> rerun is byte-identical.
Run: cd <repo root> && PYTHONPATH=. python3 tools/compute-repro-band.py
"""
import datetime, json, os, re, sys
import numpy as np

sys.path.insert(0, '.')
from eval_utils.cluster_bootstrap import balanced_accuracy, paired_metric_cluster_bootstrap
from eval_utils.metrics import confusion

MANIFEST = 'kaggle/isb-clip-manifest.json'
FOLD_APPENDIX = 'report/vn/phu-luc-b-cau-hinh-fold.txt'
CHUONG04 = 'report/vn/chuong-04-chi-tiet-thuc-nghiem.txt'
FILES = {
    'published_2026-09-29': 'sweep_results/relation_tokens/enc_isbclip_B1_lso-1100_headline-none_oof.json',
    'reprocheck_2026-10-14': 'sweep_results/relation_tokens/enc_isbclip_B1_REPROCHECK_lso-1100_headline-none_oof.json',
    'posthoc_threads1': 'sweep_results/posthoc_sensitivity/B1_published_config_threads1_oof.json',
    'posthoc_threads4': 'sweep_results/posthoc_sensitivity/B1_published_config_threads4_oof.json',
    'posthoc_default': 'sweep_results/posthoc_sensitivity/B1_published_config_oof.json',
}
OUT_PATH = 'sweep_results/review_261101/headline-reproducibility-band.json'
SCRIPT_PATH = 'tools/compute-repro-band.py'

def macro_f1(y_true, y_pred):
    cm = confusion(y_true, y_pred, n_classes=2)
    f1s = []
    for c in (0, 1):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))

def parse_fold_appendix(path):
    """Authoritative source->fold(1-10) map, read from Appendix B.1 (not re-derived by rerunning
    StratifiedGroupKFold, which could silently drift with the sklearn version on disk)."""
    text = open(path, encoding='utf-8').read()
    block_re = re.compile(
        r'FOLD (\d+) — test (\d+) clip: (\d+) nguy hiểm, (\d+) bình thường\n'
        r'\s+(\d+) nguồn.*?\n(.*?)(?=\n\nFOLD |\n-{10,}\nB\.2)', re.S)
    source_to_fold, fold_meta = {}, {}
    for m in block_re.finditer(text):
        fnum = int(m.group(1))
        fold_meta[str(fnum)] = {'n_clip': int(m.group(2)), 'n_unsafe': int(m.group(3)),
                                 'n_safe': int(m.group(4)), 'n_sources': int(m.group(5))}
        for line in m.group(6).splitlines():
            mm = re.match(r'^\s*(\S+)\s+(\d+) clip\s+([HN])\s*$', line)
            if mm:
                source_to_fold[mm.group(1)] = fnum
    assert len(source_to_fold) == 151, f'expected 151 sources, parsed {len(source_to_fold)}'
    assert sum(v['n_sources'] for v in fold_meta.values()) == 151 and \
        sum(v['n_clip'] for v in fold_meta.values()) == 1100
    return source_to_fold, fold_meta

def extract(text, start, end):
    i = text.index(start); j = text.index(end, i) + len(end)
    return text[i:j]

def load_oof(path, clip_to_source):
    raw = json.load(open(path))
    rows = raw['rows']
    for r in rows:
        if 'fold_id' not in r:
            r['fold_id'] = clip_to_source[r['clip_id']]
    st = os.stat(path)
    meta = {k: v for k, v in raw.items() if k != 'rows'}
    meta['file_size_bytes'] = st.st_size
    meta['file_mtime_utc'] = datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat()
    return rows, meta

def summarize(rows):
    y = np.array([r['y_true'] for r in rows]); p = np.array([r['y_pred'] for r in rows]); cm = confusion(y, p, n_classes=2)
    return {'n': len(rows), 'balanced_accuracy_pct': balanced_accuracy(y, p) * 100,
            'macro_f1_pct': macro_f1(y, p) * 100, 'top1_pct': float((y == p).mean()) * 100,
            'confusion': {'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])}}

def per_fold_ba(rows, source_to_fold):
    by_fold = {}
    for r in rows:
        f = source_to_fold[r['fold_id']]; by_fold.setdefault(f, {'y': [], 'p': []})
        by_fold[f]['y'].append(r['y_true']); by_fold[f]['p'].append(r['y_pred'])
    out = {}
    for f, v in sorted(by_fold.items(), key=lambda kv: int(kv[0])):
        y, p = np.array(v['y']), np.array(v['p'])
        out[f] = {'n': len(y), 'ba_pct': balanced_accuracy(y, p) * 100, 'top1_pct': float((y == p).mean()) * 100}
    return out

def disagreement(ref_rows, other_rows, source_to_fold):
    ref = {r['clip_id']: r for r in ref_rows}; oth = {r['clip_id']: r for r in other_rows}
    ids = sorted(set(ref) & set(oth))
    diffs = [c for c in ids if ref[c]['y_pred'] != oth[c]['y_pred']]
    by_fold, by_source = {}, {}
    for c in diffs:
        src = ref[c]['fold_id']; f = source_to_fold[src]
        by_fold[f] = by_fold.get(f, 0) + 1; by_source[src] = by_source.get(src, 0) + 1
    return {'n_clips_compared': len(ids), 'n_disagree': len(diffs), 'disagree_clip_ids': diffs,
            'by_fold': dict(sorted(by_fold.items(), key=lambda kv: int(kv[0]))),
            'by_source': dict(sorted(by_source.items(), key=lambda kv: (-kv[1], kv[0])))}

def paired_ci(rows_a, rows_b):
    a = {r['clip_id']: r for r in rows_a}; b = {r['clip_id']: r for r in rows_b}
    ids = sorted(set(a) & set(b))
    y = np.array([a[i]['y_true'] for i in ids]); pa = np.array([a[i]['y_pred'] for i in ids])
    pb = np.array([b[i]['y_pred'] for i in ids]); g = np.array([str(a[i]['fold_id']) for i in ids])
    d, lo, hi, nv = paired_metric_cluster_bootstrap(y, pa, pb, g, n_boot=5000, seed=0, alpha=0.05)
    return {'diff_pp': d, 'ci95_pp': [lo, hi], 'n_valid_draws': nv,
            'excludes_zero': bool(lo > 0 or hi < 0)}

def main():
    manifest = json.load(open(MANIFEST)); clip_to_source = dict(zip(manifest['clip_ids'], manifest['views']))
    source_to_fold, fold_meta = parse_fold_appendix(FOLD_APPENDIX)
    runs, rows_by_run = {}, {}
    for name, path in FILES.items():
        rows, meta = load_oof(path, clip_to_source)
        rows_by_run[name] = rows
        runs[name] = dict(summarize(rows), provenance=meta, per_fold=per_fold_ba(rows, source_to_fold))
    pub = rows_by_run['published_2026-09-29']
    band_members = {k: runs[k]['balanced_accuracy_pct']
                     for k in ('posthoc_threads1', 'published_2026-09-29', 'posthoc_threads4')}
    lo, hi = min(band_members.values()), max(band_members.values())
    ctrl_ba = runs['posthoc_threads4']['balanced_accuracy_pct']
    rep, t1, t4 = rows_by_run['reprocheck_2026-10-14'], rows_by_run['posthoc_threads1'], rows_by_run['posthoc_threads4']
    disagreements = {
        'published_vs_reprocheck': disagreement(pub, rep, source_to_fold),
        'published_vs_posthoc_threads1': disagreement(pub, t1, source_to_fold),
        'published_vs_posthoc_threads4': disagreement(pub, t4, source_to_fold),
        'reprocheck_vs_posthoc_threads4': disagreement(rep, t4, source_to_fold),
    }
    paired_cluster_ci = {'published_minus_reprocheck': paired_ci(pub, rep),
                          'published_minus_posthoc_threads1': paired_ci(pub, t1),
                          'published_minus_posthoc_threads4': paired_ci(pub, t4)}
    chuong04 = open(CHUONG04, encoding='utf-8').read()
    sec_start = '4.5.6. Cái giá đo được của việc chọn điểm kiểm tra bằng một đại lượng đã bão hoà'
    sec_end = 'khuyết điểm được ghi thành một hạn chế có số đo kèm theo, thay vì một câu nhận xét không số.'
    quotes = {
        'full_text_verbatim': extract(chuong04, sec_start, sec_end),
        'headline_unchanged_verbatim': extract(chuong04, 'Hàng tiêu đề **không đổi**.',
                                                'chỉ khai báo độ chính xác của chữ số thập phân.'),
        'band_statement_verbatim': extract(chuong04, 'Dải là **1.46 điểm phần trăm**', ', không ở biên trên.'),
        'mechanism_verbatim': extract(chuong04, '(1) `val_macro` đã **bão hoà**', 'dự đoán của fold đó.'),
        'mandate_control_verbatim': extract(chuong04, 'Mọi phép đo thêm sau đóng băng', '**83.25% ở 4 luồng**.'),
    }
    disclosure_sentence_vn = (
        'BA đầu đề 84.2% (tạo tác đóng băng 05/10, làn iSafetyBench 1,100 clip nhị phân, '
        'leave-source-out) chỉ tái lập được đến khoảng ±0.7 điểm phần trăm khi chạy lại đúng '
        'lệnh, cùng fold và cùng seed trên cùng tệp đặc trưng đã cache (dải đo được qua ba môi '
        f'trường luồng CPU: {lo:.2f}%-{hi:.2f}%, rộng {hi - lo:.2f} điểm phần trăm), vì bước '
        'chọn epoch dùng val_macro đã bão hoà quanh 98.0%-99.1% và phá trùng điểm bằng quy tắc '
        '"epoch đầu tiên đạt max" (dấu > thay vì >=), nên một sai số dấu phẩy động cỡ 1e-7 do '
        'thứ tự cộng dồn tensor khác nhau theo số luồng CPU đủ đổi epoch được chốt và đổi dự '
        'đoán của fold yếu nhất (fold 6, dao động 10.4 điểm phần trăm). Hàng tiêu đề 84.2% và '
        'mọi phán quyết cổng đã đóng băng không đổi; mọi phép đo thêm sau đóng băng trên làn '
        f'này phải so với hàng đối chứng {ctrl_ba:.2f}% (4 luồng CPU), không so với 84.2%.'
    )

    result = {
        'note': ('POST-FREEZE (after the 2026-10-05 freeze); POST-HOC, NOT A PRE-REGISTERED GATE. May be '
                 'used to show the headline 84.2% BA reproduces only to about +/-0.7pp under an unmodified '
                 'rerun on cached features, and to name the 83.25% (4-thread) control row for post-freeze '
                 'comparisons. Must NOT move the published headline, reopen any frozen gate verdict (thesis '
                 '6.13), or compare any number here against the Unsafe-Net 4-class lane.'),
        'lane': ('iSafetyBench, 1,100 clips, binary (nguy_hiem/binh_thuong), leave-source-out '
                 '(StratifiedGroupKFold, n_splits=10, 151 sources, group=source video id). Single lane '
                 'only -- never compared against the 64-/516-clip Unsafe-Net 4-class lane.'),
        'resampling_unit': 'source_video (the leave-source-out grouping key, i.e. fold_id)',
        'n_boot': 5000,
        'seed': 0,
        'alpha': 0.05,
        'source_artifacts': [MANIFEST, FOLD_APPENDIX, CHUONG04] + list(FILES.values()),
        'script': SCRIPT_PATH,
        'results': {
            'fold_membership_check': {'n_sources': len(source_to_fold), 'n_folds': len(fold_meta),
                                       'fold_sizes': fold_meta},
            'runs': runs,
            'band': {'members_pct': band_members, 'min_pct': lo, 'max_pct': hi, 'width_pp': hi - lo,
                     'control_file': FILES['posthoc_threads4'], 'control_ba_pct': ctrl_ba,
                     'display': f'{lo:.2f}-{hi:.2f}% (rộng {hi - lo:.2f} điểm phần trăm); '
                                f'đối chứng hậu-đóng-băng = {ctrl_ba:.2f}% (4 luồng)'},
            'per_clip_disagreement': disagreements,
            'paired_cluster_bootstrap_ci_pp': paired_cluster_ci,
            'thesis_section_4_5_6_quotes': quotes,
            'disclosure_sentence_vn_ready_to_paste': disclosure_sentence_vn,
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write('\n')
    print(f'wrote {OUT_PATH}')

if __name__ == '__main__':
    main()
