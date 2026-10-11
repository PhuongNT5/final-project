"""Vẽ hình chương 5 (triển khai biên) và hình demo từ tạo tác đo thật.

Mọi con số trong các hình này được đọc lại từ chính tệp JSON mà chương 5 trích
dẫn, rồi đối chiếu với số đã công bố bằng hàm `kiem`. Script không đo lại gì.

Chạy:  .venv/bin/python3 tools/ve-hinh-trien-khai-va-demo.py
"""
import glob
import json
import statistics as tk

import cv2
import matplotlib.pyplot as plt
import numpy as np

from hinh_luan_van_style import MAU, dung_kieu, luu

JM = 'deployment/artifacts/jetson-measurements'
DEMO = 'demo/out'

HOP_DONG_P3 = 3.0          # quyết định/giây, hợp đồng của làn P3 (mục 5.5)
SAN_B1 = (1.249, 2.500)    # hợp đồng clip trung vị kho và clip 60 khung/giây

# Số đã công bố trong chương 5, dùng để đối chiếu — KHÔNG dùng để vẽ.
P3_CONG_BO = {
    ('pinned', '1_te10'): 3.479, ('pinned', '3_te1'): 3.965, ('pinned', '4_te10'): 2.493,
    ('dvfs', '1_te10'): 1.441, ('dvfs', '3_te1'): 1.511, ('dvfs', '4_te10'): 1.141,
}
B1_CONG_BO = {'truoc_ms': 289.672, 'sau_ms': 167.291, 'sau_p95_ms': 198.224,
              'ms_1080p': 227.214}

TEN_DOAN = {'1_te10': 'phân đoạn 1_te10', '3_te1': 'phân đoạn 3_te1',
            '4_te10': 'phân đoạn 4_te10'}


def kiem(ten, tinh_lai, cong_bo, dung_sai=0.01):
    lech = abs(tinh_lai - cong_bo)
    print(f'  {"OK" if lech <= dung_sai else "!!"}  {ten}: '
          f'tính lại {tinh_lai:.3f} vs công bố {cong_bo:.3f}')
    return tinh_lai


def _mot_tep(mau):
    tep = sorted(glob.glob(f'{JM}/{mau}'))
    if len(tep) != 1:
        raise SystemExit(f'cần đúng một tạo tác khớp {mau}, thấy {len(tep)}')
    return tep[0]


def doc_p3(trang_thai, doan):
    """Đọc nhịp luồng xấu nhất của một lần đo P3 tổ hợp."""
    d = json.load(open(_mot_tep(f'p3-composed-cadence-fp16-{trang_thai}-free-{doan}-*.json')))
    c = d['configs'][0]
    assert d['cadence_contract_decisions_per_sec'] == HOP_DONG_P3
    assert d['device_state']['quiet'] and d['clock_state_stable']
    return c['worst_stream_decisions_per_sec']


def hinh_nhip_p3():
    """Hình 5.1 — nhịp tổ hợp P3 ở hai trạng thái xung nhịp, so với hợp đồng 3.0."""
    doan = ['1_te10', '3_te1', '4_te10']
    gia_tri = {}
    for tt in ('pinned', 'dvfs'):
        for d in doan:
            gia_tri[(tt, d)] = kiem(f'P3 {tt} {d}', doc_p3(tt, d), P3_CONG_BO[(tt, d)])

    x = np.arange(len(doan))
    rong = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for i, (tt, nhan, mau) in enumerate([
            ('pinned', 'xung nhịp ghim (MAXN_SUPER, jetson_clocks)', MAU['chinh']),
            ('dvfs', 'xung nhịp DVFS mặc định', MAU['phu'])]):
        v = [gia_tri[(tt, d)] for d in doan]
        thanh = ax.bar(x + (i - 0.5) * rong, v, rong, label=nhan, color=mau,
                       edgecolor='white', linewidth=0.6)
        for b, giatri in zip(thanh, v):
            ax.text(b.get_x() + b.get_width() / 2, giatri + 0.07, f'{giatri:.3f}',
                    ha='center', fontsize=8,
                    fontweight='bold' if giatri >= HOP_DONG_P3 else 'normal')

    ax.axhline(HOP_DONG_P3, color=MAU['canh_bao'], lw=1.3, ls='--')
    ax.text(len(doan) - 0.52, HOP_DONG_P3 + 0.09, f'hợp đồng {HOP_DONG_P3:.1f} qđ/giây',
            color=MAU['canh_bao'], fontsize=8, ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels([TEN_DOAN[d] for d in doan])
    ax.set_ylabel('nhịp luồng xấu nhất (quyết định/giây)')
    ax.set_ylim(0, 4.6)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left', ncol=1)
    ax.set_title('Nhịp tổ hợp P3 trên Jetson Orin Nano Super, FP16, cửa sổ yên tĩnh')
    return luu(fig, 'hinh-5-1-nhip-to-hop-p3-theo-trang-thai-xung-nhip.png')


def hinh_nhip_b1():
    """Hình 5.2 — nhịp B1 đã triển khai và biên so với hai mức hợp đồng."""
    sau = [json.load(open(_mot_tep(f'r48-composed-cadence-fp16-20261003T{t}.json')))
           ['steady_state_ms_per_decision']['p50'] for t in ('155728', '155806', '155843')]
    truoc = [json.load(open(_mot_tep(f'r48-composed-cadence-fp16-20261001T{t}.json')))
             ['steady_state_ms_per_decision']['p50'] for t in ('233159', '233324', '233449')]
    ms_1080 = json.load(open(_mot_tep('r48-composed-cadence-fp16-20261003T160245.json'))
                        )['steady_state_ms_per_decision']['p50']
    ms_truoc = kiem('B1 trước khi sửa (ms)', tk.median(truoc), B1_CONG_BO['truoc_ms'])
    ms_sau = kiem('B1 sau khi sửa (ms)', tk.median(sau), B1_CONG_BO['sau_ms'])
    kiem('B1 1080p (ms)', ms_1080, B1_CONG_BO['ms_1080p'])

    hang = [('B1 trước khi sửa tiền xử lý\n(720p, 29.97 khung/giây)', 1000 / ms_truoc, MAU['phu']),
            ('B1 đã triển khai\n(720p, 29.97 khung/giây)', 1000 / ms_sau, MAU['chinh']),
            ('B1 trên nguồn 1920×1080\n(đo riêng, mục 5.4.4)', 1000 / ms_1080, MAU['chinh'])]
    kiem('B1 đã triển khai (qđ/giây)', hang[1][1], 5.978)
    kiem('B1 1080p (qđ/giây)', hang[2][1], 4.401)

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    y = np.arange(len(hang))[::-1]
    ax.barh(y, [h[1] for h in hang], 0.52, color=[h[2] for h in hang],
            edgecolor='white', linewidth=0.6)
    for yy, h in zip(y, hang):
        ax.text(h[1] - 0.12, yy, f'{h[1]:.3f} qđ/giây', va='center', ha='right',
                color='white', fontweight='bold', fontsize=8.5)
    for san, kieu in zip(SAN_B1, ('--', ':')):
        ax.axvline(san, color=MAU['canh_bao'], lw=1.3, ls=kieu)
        ax.text(san, len(hang) - 0.52, f'  {san:.3f}', color=MAU['canh_bao'], fontsize=8)
    ax.text(SAN_B1[1] + 0.9, len(hang) - 0.52,
            'hợp đồng: clip trung vị kho | clip 60 khung/giây',
            color=MAU['canh_bao'], fontsize=8, va='center')
    ax.set_yticks(y)
    ax.set_yticklabels([h[0] for h in hang], fontsize=8)
    ax.set_xlim(0, 6.9)
    ax.set_ylim(-0.6, len(hang) - 0.15)
    ax.set_xlabel('nhịp trạng thái dừng (quyết định/giây), suy ra từ trung vị p50 của ba lần lặp')
    ax.grid(axis='y', visible=False)
    return luu(fig, 'hinh-5-2-nhip-b1-da-trien-khai-va-bien-hop-dong.png')


def _nap_dong_thoi_gian():
    ra = []
    for f in sorted(glob.glob(f'{DEMO}/*-timeline.json')):
        d = json.load(open(f))
        assert d['model'] == 'B1' and d['threshold'] == 0.5
        ra.append((f.split('/')[-1].replace('-timeline.json', ''), d))
    return ra


def hinh_dong_thoi_gian():
    """Hình 5.3 — xác suất hiểm hoạ theo thời gian của hai clip demo."""
    dt = _nap_dong_thoi_gian()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    kieu = {'hazard': (MAU['canh_bao'], 'o', 'clip hiểm hoạ'),
            'normal': (MAU['chinh'], 's', 'clip bình thường')}
    for ten, d in dt:
        lop = 'hazard' if '/hazard/' in d['video'] else 'normal'
        mau, danh, nhan = kieu[lop]
        t = [w['start_sec'] for w in d['windows']]
        p = [w['prob_unsafe'] for w in d['windows']]
        lo = [p[i] - min(w['per_seed']) for i, w in enumerate(d['windows'])]
        hi = [max(w['per_seed']) - p[i] for i, w in enumerate(d['windows'])]
        ax.errorbar(t, p, yerr=[lo, hi], color=mau, marker=danh, ms=5, lw=1.6,
                    capsize=3, elinewidth=1.0, label=f'{nhan} — {ten}')
    ax.axhline(0.5, color=MAU['trung_tinh'], lw=1.2, ls='--')
    ax.text(4.55, 0.52, 'ngưỡng quyết định 0.5', fontsize=8, ha='right',
            color=MAU['trung_tinh'])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.2, 4.6)
    ax.set_xlabel('thời điểm bắt đầu cửa sổ (giây); cửa sổ 48 khung, bước nhảy 24 khung')
    ax.set_ylabel('xác suất hiểm hoạ')
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.74))
    ax.set_title('Đường quyết định của B1 trên hai clip demo (thanh đứng = biên độ ba hạt giống)')
    return luu(fig, 'hinh-5-3-demo-duong-quyet-dinh-theo-thoi-gian.png')


def _khung(duong_dan, chi_so, rong=360):
    cap = cv2.VideoCapture(duong_dan)
    cap.set(cv2.CAP_PROP_POS_FRAMES, chi_so)
    ok, anh = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f'không đọc được khung {chi_so} của {duong_dan}')
    h, w = anh.shape[:2]
    anh = cv2.resize(anh, (rong, int(round(h * rong / w))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)


def hinh_khung_co_chu_giai():
    """Hình 5.4 — khung đã chú giải, lấy ở đúng các mốc cửa sổ của dòng thời gian."""
    dt = _nap_dong_thoi_gian()
    k = 4
    fig, truc = plt.subplots(len(dt), k, figsize=(8.6, 3.5),
                             gridspec_kw={'hspace': 0.06, 'wspace': 0.04})
    for hang, (ten, d) in enumerate(dt):
        mp4 = f'{DEMO}/{ten}-annotated.mp4'
        cua_so = [d['windows'][i] for i in np.linspace(0, len(d['windows']) - 1, k).astype(int)]
        lop = 'hiểm hoạ' if '/hazard/' in d['video'] else 'bình thường'
        for cot, w in enumerate(cua_so):
            ax = truc[hang][cot]
            # Khung giữa cửa sổ thì đại diện cho cửa sổ hơn là khung đầu.
            ax.imshow(_khung(mp4, w['start_frame'] + d['window'] // 2))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            mau = MAU['canh_bao'] if w['verdict'] == 'unsafe' else MAU['chinh']
            for v in ax.spines.values():
                v.set_visible(True); v.set_color(mau); v.set_linewidth(1.6)
            ax.set_title(f"{w['start_sec']:.1f} s  ·  p = {w['prob_unsafe']:.3f}",
                         fontsize=8, color=mau, pad=2)
            if cot == 0:
                ax.set_ylabel(f'nhãn kho: {lop}\n{ten}', fontsize=7.5)
    fig.suptitle('Khung demo đã chú giải — hộp của bộ định vị YOLO, nhãn do đầu B1 phát',
                 fontsize=10, y=0.985)
    return luu(fig, 'hinh-5-4-demo-khung-co-chu-giai.png', le=0.22)


if __name__ == '__main__':
    dung_kieu()
    for ve in (hinh_nhip_p3, hinh_nhip_b1, hinh_dong_thoi_gian, hinh_khung_co_chu_giai):
        print(ve.__doc__.splitlines()[0])
        ve()
