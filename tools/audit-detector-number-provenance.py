#!/usr/bin/env python3
"""Provenance map for detector-checkpoint numbers cited in the thesis.
POST-FREEZE (run 2026-11-01), POST-HOC / NOT A PRE-REGISTERED GATE: audits citation
hazards in numbers already printed, proves no new hypothesis. Resolves which checkpoint
backs which number, flags a sign slip in the thesis's own G17 sentence
(report/vn/chuong-06-danh-gia.txt:424), gives safe citation wording. Must NOT be used to
flip a gate verdict, compare/average lanes, or revive a withdrawn number.
Run: cd <repo> && PYTHONPATH=. python3 tools/audit-detector-number-provenance.py
(every bootstrap/seed fixed at 0 -> byte-identical reruns)"""
import glob, json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from eval_utils.cluster_bootstrap import balanced_accuracy, metric_cluster_bootstrap, paired_metric_cluster_bootstrap
from eval_utils.paired_tests import mcnemar_exact
ROOT = Path(__file__).resolve().parent.parent
RT = ROOT / 'sweep_results' / 'relation_tokens'
ED = ROOT / 'sweep_results' / 'entity_detector'
OUT_PATH = ROOT / 'sweep_results' / 'review_261101' / 'detector-number-provenance.json'
N_BOOT, SEED, ALPHA = 5000, 0, 0.05
rel = lambda p: str(Path(p).resolve().relative_to(ROOT))
def scan_detector_lines():
    """Every 'detector=<path>' occurrence under sweep_results/**/*.log, read verbatim (full file, not just the header)."""
    out = []
    for f in sorted(glob.glob(str(ROOT / 'sweep_results' / '**' / '*.log'), recursive=True)):
        try:
            content = open(f, errors='ignore').read()
        except OSError:
            continue
        for line in content.splitlines():
            m = re.search(r'detector=(\S+)', line)
            if m:
                out.append({'log': rel(f), 'checkpoint_as_written': m.group(1),
                            'checkpoint_basename': Path(m.group(1)).name, 'raw_line': line.strip()})
    return out

def build_evidence(log_path):
    content = open(log_path, errors='ignore').read()
    return [l.strip() for l in content.splitlines() if re.search(r'\.pt\b', l) or 'detections=' in l]

def checkpoint_registry():
    reg = {}
    for h in scan_detector_lines():
        reg.setdefault(h['checkpoint_basename'], []).append(h)
    return {ck: {'raw_detector_lines': v,
                 'build_output_evidence': [{'log': e['log'], 'lines': build_evidence(ROOT / e['log'])} for e in v]}
            for ck, v in sorted(reg.items())}

def load_oof(path):
    rows = json.load(open(path))['rows']
    return (np.array([int(r['y_true']) for r in rows]), np.array([int(r['y_pred']) for r in rows]),
            np.array([str(r['fold_id']) for r in rows]), [r['clip_id'] for r in rows])

def ba_entry(path, label, checkpoint, thesis_row, thesis_pct, unit):
    y, p, g, cids = load_oof(path)
    pt, lo, hi, nv = metric_cluster_bootstrap(y, p, g, balanced_accuracy, n_boot=N_BOOT, seed=SEED, alpha=ALPHA)
    e = {'label': label, 'checkpoint': checkpoint, 'oof_file': rel(path), 'n_clips': int(len(y)),
         'n_groups': int(len(set(g))), 'resampling_unit': unit, 'balanced_accuracy': pt,
         'display': f'{pt * 100:.1f}%', 'ci95_cluster_bootstrap_pp': [lo, hi], 'n_valid_boot_draws': nv,
         'thesis_row': thesis_row, 'thesis_printed_ba_pct': thesis_pct,
         'recomputed_minus_thesis_pp': None if thesis_pct is None else round(pt * 100 - thesis_pct, 4)}
    e['_y'], e['_p'], e['_g'], e['_cids'] = y, p, g, cids
    return e

def pair_entry(A, B, label, gate, thesis_d, thesis_ci, thesis_p):
    ids = sorted(set(A['_cids']) & set(B['_cids']))
    ia, ib = {c: i for i, c in enumerate(A['_cids'])}, {c: i for i, c in enumerate(B['_cids'])}
    idx_a, idx_b = [ia[c] for c in ids], [ib[c] for c in ids]
    y = A['_y'][idx_a]
    assert (y == B['_y'][idx_b]).all(), f"label mismatch {A['label']} vs {B['label']}"
    pa, pb, g = A['_p'][idx_a], B['_p'][idx_b], A['_g'][idx_a]
    d, lo, hi, nv = paired_metric_cluster_bootstrap(y, pa, pb, g, n_boot=N_BOOT, seed=SEED, alpha=ALPHA)
    mc = mcnemar_exact(pa == y, pb == y)
    return {'gate_or_label': label, 'named_pre_registered_gate': gate,
            'A': {'label': A['label'], 'checkpoint': A['checkpoint'], 'oof_file': A['oof_file']},
            'B': {'label': B['label'], 'checkpoint': B['checkpoint'], 'oof_file': B['oof_file']},
            'direction': 'A minus B', 'n_clips_intersection': len(ids), 'resampling_unit': A['resampling_unit'],
            'recomputed_diff_pp': d, 'display': f'{d:+.1f} pp', 'ci95_cluster_bootstrap_pp': [lo, hi],
            'n_valid_boot_draws': nv, 'thesis_printed_delta_pp': thesis_d, 'thesis_ci_cluster_pp': thesis_ci,
            'thesis_mcnemar_p': thesis_p,
            'mcnemar_exact': {'b': mc['b'], 'c': mc['c'], 'n_discordant': mc['n_discordant'], 'p_value': mc['p_value']}}

def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if not str(k).startswith('_')}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj

def parse_reval_log(path):
    t = open(path, errors='ignore').read()
    g = lambda pat: float(re.search(pat, t).group(1))
    return {'images': int(g(r'Split\s*:\s*(\d+) images')), 'instances': int(g(r'(\d+) instances')),
            'map50': g(r'overall mAP@50\s*=\s*([\d.]+)'), 'map50_95': g(r'overall mAP@50-95\s*=\s*([\d.]+)'),
            'precision': g(r'overall precision\s*=\s*([\d.]+)'), 'recall': g(r'recall\s*=\s*([\d.]+)')}

def parse_mt3_report(path):
    t = open(path, errors='ignore').read()
    m50 = re.search(r'\|\s*mAP@50 \(all\)\s*\|\s*([\d.]+)\s*\|\s*\*\*([\d.]+)\*\*', t)
    m95 = re.search(r'\|\s*mAP@50-95 \(all\)\s*\|\s*([\d.]+)\s*\|\s*\*\*([\d.]+)\*\*', t)
    mh = re.search(r'Person_Unauthorized mAP@50 \(hard\)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', t)
    return {'yolo26s_map50': float(m50.group(1)), 'yolo11s_map50_crosscheck': float(m50.group(2)),
            'yolo26s_map50_95': float(m95.group(1)), 'yolo11s_map50_95_crosscheck': float(m95.group(2)),
            'yolo26s_person_unauthorized_map50': float(mh.group(1)),
            'yolo11s_person_unauthorized_map50_crosscheck': float(mh.group(2)), 'report_date': '2026-08-29',
            'freeze_date': '2026-10-05', 'status': 'PRE-FREEZE report; quoted verbatim, not recomputed here'}

def main():
    manifest = json.load(open(ROOT / 'kaggle' / 'isb-clip-manifest.json'))
    views = dict(zip(manifest['clip_ids'], manifest['views']))
    esta = {
        'ESTA_v11s_pretrained': ba_entry(RT / 'enc_isbclip_ESTA_lso-1100_headline-none_oof.json', 'ESTA @ YOLOv11s (COCO-pretrained)', 'yolo11s.pt', 'Bang 6.1 "ESTA phan ra theo thuc the"', 83.8, 'source_video'),
        'ESTA_y26_pretrained': ba_entry(RT / 'enc_isbclip_ESTA_y26_lso-1100_headline-none_oof.json', 'ESTA @ YOLO26s (COCO-pretrained)', 'yolo26s.pt', 'Bang 6.1 "ESTA @ bo phat hien YOLO26s"', 81.6, 'source_video'),
        'ESTA_v11ft': ba_entry(RT / 'enc_isbclip_ESTA_v11ft_lso-1100_headline-none_oof.json', 'ESTA @ YOLOv11s fine-tuned', 'detector_best.pt', 'muc 6.14.2 nhanh B, "YOLOv11s-ft"', 75.9, 'source_video'),
        'ESTA_26ft': ba_entry(RT / 'enc_isbclip_ESTA_26ft_lso-1100_headline-none_oof.json', 'ESTA @ YOLO26-ft fine-tuned', 'yolo26_best.pt', 'muc 6.14.2 nhanh B, "YOLO26-ft"', 78.1, 'source_video'),
    }
    assert all(views.get(cid) == fid for e in esta.values() for cid, fid in zip(e['_cids'], e['_g'])), 'fold_id != manifest views'
    pairs = [
        pair_entry(esta['ESTA_v11s_pretrained'], esta['ESTA_y26_pretrained'], 'ESTA@v11s - ESTA@26s (both pretrained)', 'G17 (contre-epreuve, no verdict)', -2.2, [-1.7, 6.0], 4.12e-3),
        pair_entry(esta['ESTA_26ft'], esta['ESTA_v11ft'], 'ESTA@26-ft - ESTA@v11s-ft (both fine-tuned)', 'G20 nhanh B (post-hoc, no verdict)', 2.2, [-3.5, 8.8], 0.261),
        pair_entry(esta['ESTA_v11s_pretrained'], esta['ESTA_v11ft'], 'ESTA@v11s pretrained - v11s-ft', None, None, None, None),
        pair_entry(esta['ESTA_y26_pretrained'], esta['ESTA_26ft'], 'ESTA@26s pretrained - 26s-ft', None, None, None, None),
        pair_entry(esta['ESTA_v11s_pretrained'], esta['ESTA_26ft'], 'ESTA@v11s pretrained - 26-ft', None, None, None, None),
        pair_entry(esta['ESTA_y26_pretrained'], esta['ESTA_v11ft'], 'ESTA@26s pretrained - v11s-ft', None, None, None, None),
    ]
    e3_v11ft = ba_entry(RT / 'enc_v11ft_E3_lco-691_headline-drop_conflict_oof.json', 'E3 @ YOLOv11s-ft (Unsafe-Net)', 'detector_best.pt', 'muc 6.14.1 nhanh A, "E3 @ YOLOv11s-ft"', 61.8, 'scene_cluster')
    e3_26ft = ba_entry(RT / 'enc_bin8_E3_lco-691_headline-drop_conflict_oof.json', 'E3 @ YOLO26-ft (Unsafe-Net, headline)', 'yolo26_best.pt', 'muc 6.14.1 nhanh A, "E3 @ YOLO26-ft"', 69.6, 'scene_cluster')
    e3_pair = pair_entry(e3_26ft, e3_v11ft, 'E3@26-ft - E3@v11s-ft (Unsafe-Net, in-domain)', 'G20 nhanh A (post-hoc, no verdict)', 7.8, [-1.5, 15.2], 2.69e-10)

    det_best_map = parse_reval_log(ED / 'reval-2026-10-02-detector-best-val343.log')
    y26_best_map = parse_mt3_report(ROOT / 'plans' / '2026-07-09-spatial-relation-yolo26-3month' / 'reports' / 'mt3-yolo26-results.md')
    detection_metric_side = {
        'claim_in_thesis': 'chuong-04 muc 4.2.3: Bang 4.1 (mAP@50=0.9695, mAP@50-95=0.8939, 343 anh/988 instance) do tren detector_best.pt -- checkpoint nay la nhanh THUA ca hai doi chung bo phat hien.',
        'detector_best_pt': {'recomputed_or_read': 'read (reval log, re-derived 02/10/2026, post-freeze)', **det_best_map,
                              'feeds_oof': ['ESTA_v11ft (loses G20 nhanh B)', 'E3@v11s-ft (loses G20 nhanh A)'],
                              'may_be_printed_beside': 'tokens_app_*/tokens_yoloapp_* rows only; never beside the swap winner.'},
        'yolo26_best_pt': {'recomputed_or_read': 'read (mt3-yolo26-results.md, PRE-FREEZE 2026-08-29; no post-freeze re-val on disk)', **y26_best_map,
                            'feeds_oof': ['ESTA_26ft (wins G20 nhanh B, CI contains 0)', 'E3@26-ft (wins G20 nhanh A, CI contains 0)'],
                            'may_be_printed_beside': 'tokens_bin8_*/tokens_yolo26app_* rows only; never beside mAP 0.9695/0.8939.'},
        'irony_for_the_caption': 'The checkpoint behind printed Table-4.1 mAP (0.9695) LOSES both swaps (CIs contain 0 -> "not distinguishable", not "certified worse"); the swap-winner checkpoint has a LOWER, pre-freeze-only mAP (0.962). Pairing "0.9695 mAP" with "the better detector" names the wrong checkpoint.',
    }
    diff_rounded = round(esta['ESTA_v11s_pretrained']['balanced_accuracy'] * 100 - esta['ESTA_y26_pretrained']['balanced_accuracy'] * 100, 4)
    sign_audit = {
        'source': 'report/vn/chuong-06-danh-gia.txt:424 (quoted, not edited)',
        'quoted_sentence': 'ESTA @ YOLOv11s 83.8% − ESTA @ YOLO26s 81.6% = −2.2 pp, tuc bo phat hien MOI HON cho ket qua THAP HON',
        'operand_a_thesis_rounded': 83.8, 'operand_b_thesis_rounded': 81.6,
        'operand_a_recomputed_pct': esta['ESTA_v11s_pretrained']['balanced_accuracy'] * 100,
        'operand_b_recomputed_pct': esta['ESTA_y26_pretrained']['balanced_accuracy'] * 100,
        'diff_from_thesis_rounded_operands_pp': round(83.8 - 81.6, 4), 'diff_from_recomputed_operands_pp': diff_rounded,
        'diff_printed_in_sentence_pp': -2.2,
        'narrative_implies_sign': '"newer detector gives LOWER result" needs v11s>26s, i.e. diff must be POSITIVE (+2.2).',
        'printed_sign_consistent_with_its_own_operands': False, 'printed_sign_consistent_with_its_own_narrative': False,
        'thesis_own_mitigating_note': "6.2.5 note (ii) says the CI bracket may have been printed in reversed order; that does not resolve the headline sentence, whose own two printed operands subtract to +2.2.",
        'verdict_impact': 'None: G17 is "contre-epreuve, no verdict" and its CI contains 0 under either sign.',
        'disposition': 'Reported per assignment; report/vn/ left untouched.',
    }
    citation_lines = [
        {'number': '0.9695 / 0.8939 (mAP@50 / mAP@50-95)', 'vi': 'mAP@50 0.9695 / mAP@50-95 0.8939 la cua detector_best.pt (YOLOv11s-ft), do lai 02/10/2026, 343 anh/988 instance -- khong phai mAP cua bo phat hien thang trong doi chung E3/G20.'},
        {'number': '0.962 / 0.872 (mAP@50 / mAP@50-95)', 'vi': 'mAP@50 0.962 / mAP@50-95 0.872 la cua yolo26_best.pt, do truoc dong bang (bao cao MT3, 29/08/2026) -- khong co so do lai sau dong bang cho checkpoint nay.'},
        {'number': '+2.2 pp (G17, cap tien huan luyen)', 'vi': 'ESTA@YOLOv11s (83,8%) tru ESTA@YOLO26s (81,6%) = +2,2 pp, lan iSafetyBench, n=1.100 clip/151 nguon, CI theo cum chua 0 -- doi chung G17, khong phan quyet. Khong trich -2,2 pp nhu cau goc 6.2.4.'},
        {'number': '+2.2 pp (G20 nhanh B, cap tinh chinh)', 'vi': 'ESTA@YOLO26-ft (78,1%) tru ESTA@YOLOv11s-ft (75,9%) = +2,2 pp, lan iSafetyBench, n=1.100 clip/151 nguon, CI [-3,5; +8,8] chua 0, hau dong bang -- doi chung G20, khong phan quyet.'},
        {'number': '+7.8 pp (G20 nhanh A, Unsafe-Net E3)', 'vi': 'E3@YOLO26-ft (69,6%) tru E3@YOLOv11s-ft (61,8%) = +7,8 pp, lan Unsafe-Net, n=687 clip/8 nhom boi canh, CI [-1,5; +15,2] chua 0, hau dong bang -- doi chung G20, khong phan quyet. Khong dat canh 83,8%/81,6% cua G17 (khac lan).'},
    ]
    results = {
        'checkpoint_registry': checkpoint_registry(),
        'isafetybench_lane': {'lane': 'iSafetyBench, binary, leave-source-out, 1,100 clips, 151 source videos (manifest "views")',
                               'esta_variants': {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')} for k, v in esta.items()},
                               'pairwise_deltas': pairs},
        'unsafenet_e3_lane': {'lane': 'Unsafe-Net, E3 encoder, LCO-691 pooled, 687 clips, 8 scene clusters',
                               'variants': {'E3_26ft': {k: v for k, v in e3_26ft.items() if not k.startswith('_')},
                                            'E3_v11ft': {k: v for k, v in e3_v11ft.items() if not k.startswith('_')}},
                               'pairwise_delta': e3_pair,
                               'cross_lane_ban': 'NOT comparable to isafetybench_lane: different corpus, task, resampling unit.'},
        'detection_metric_side': detection_metric_side, 'sign_audit': sign_audit, 'cach_trich_dan_dung': citation_lines,
    }
    src_fixed = {rel(RT / 'enc_isbclip_ESTA_lso-1100_headline-none_oof.json'), rel(RT / 'enc_isbclip_ESTA_y26_lso-1100_headline-none_oof.json'),
                 rel(RT / 'enc_isbclip_ESTA_v11ft_lso-1100_headline-none_oof.json'), rel(RT / 'enc_isbclip_ESTA_26ft_lso-1100_headline-none_oof.json'),
                 rel(RT / 'enc_v11ft_E3_lco-691_headline-drop_conflict_oof.json'), rel(RT / 'enc_bin8_E3_lco-691_headline-drop_conflict_oof.json'),
                 rel(ED / 'reval-2026-10-02-detector-best-val343.log'),
                 rel(ROOT / 'plans' / '2026-07-09-spatial-relation-yolo26-3month' / 'reports' / 'mt3-yolo26-results.md'),
                 rel(ROOT / 'kaggle' / 'isb-clip-manifest.json'), 'report/vn/chuong-04-chi-tiet-thuc-nghiem.txt', 'report/vn/chuong-06-danh-gia.txt'}
    payload = {
        'note': 'POST-FREEZE (2026-11-01) provenance audit; POST-HOC, NOT A PRE-REGISTERED GATE. May resolve which checkpoint backs which number, flag the G17 sign slip, give safe citation wording. May NOT overturn a gate verdict, average/compare lanes, or revive a withdrawn number (none referenced here).',
        'lane': 'Multi-lane; every number carries its own lane tag (isafetybench_lane / unsafenet_e3_lane). Never compared across lanes.',
        'resampling_unit': 'source_video (iSafetyBench, 151 groups) or scene_cluster (Unsafe-Net E3, 8 groups); see each block',
        'n_boot': N_BOOT, 'seed': SEED, 'alpha': ALPHA,
        'source_artifacts': sorted({h['log'] for h in scan_detector_lines()} | src_fixed),
        'script': 'tools/audit-detector-number-provenance.py', 'results': results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(clean(payload), indent=2, sort_keys=True, ensure_ascii=True))
    print(f'wrote {OUT_PATH}')

if __name__ == '__main__':
    main()
