"""Sinh ba hình về DỮ LIỆU cho Chương 4: hai bảng khung mẫu và một hình thành phần.

Khung ảnh được trích TRỰC TIẾP từ các tệp .mp4 trong `data/`, không dùng ảnh minh hoạ.
Số clip được ĐẾM từ đĩa, không gõ lại từ bảng trong luận văn; script in ra bảng đếm để
đối chiếu với Chương 4.

Dùng:  .venv/bin/python tools/ve-hinh-du-lieu-luan-van.py
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from hinh_luan_van_style import MAU, dung_kieu, luu

ISB = 'data/isafetybench'
UNSAFE_NET = [
    ('0_safe_walkway_violation', 'Vi phạm lối đi\n(WalkwayViol)', False),
    ('1_unauthorized_intervention', 'Can thiệp không phép\n(UnauthorInt)', False),
    ('2_opened_panel cover', 'Tủ điện mở\n(OpenedPanel)', False),
    ('3_carrying_overload_with_forklift', 'Xe nâng quá tải\n(CarryOverload)', False),
    ('4_safe_walkway', 'Lối đi an toàn', True),
    ('5_authorized_intervention', 'Can thiệp có phép', True),
    ('6_closed_panel_cover', 'Tủ điện đóng', True),
    ('7_safe_carrying', 'Chở hàng an toàn', True),
]


def khung_giua(duong_dan_video, canh=320):
    """Khung ở GIỮA clip, đổi sang RGB và co về ô vuông `canh`. Giữa clip để tránh khung đầu đen."""
    cap = cv2.VideoCapture(duong_dan_video)
    tong = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if tong > 2:
        cap.set(cv2.CAP_PROP_POS_FRAMES, tong // 2)
    ok, khung = cap.read()
    cap.release()
    if not ok:
        return np.full((canh, canh, 3), 230, np.uint8)
    return cv2.cvtColor(cv2.resize(khung, (canh, canh)), cv2.COLOR_BGR2RGB)


def chon_deu(tep, k):
    """k tệp rải đều trên danh sách đã sắp — chọn tất định, không dùng số ngẫu nhiên."""
    tep = sorted(tep)
    return [tep[i] for i in np.linspace(0, len(tep) - 1, k).astype(int)]


def luoi_khung(duong_dan_nhan, hang, cot, ten_tep, cao_moi_o=2.05):
    fig, axes = plt.subplots(hang, cot, figsize=(cot * 1.85, hang * cao_moi_o))
    for ax, (duong_dan, nhan, mau) in zip(np.ravel(axes), duong_dan_nhan):
        ax.imshow(khung_giua(duong_dan))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for canh in ax.spines.values():
            canh.set_visible(True); canh.set_color(mau); canh.set_linewidth(1.8)
        ax.set_title(nhan, fontsize=7.5, color=MAU['trung_tinh'], pad=3)
    fig.subplots_adjust(wspace=0.05, hspace=0.16)
    return luu(fig, ten_tep)


def hinh_isafetybench():
    """Hình 4.1 — 5 clip hiểm hoạ và 5 clip bình thường của iSafetyBench."""
    muc = []
    for lop, nhan, mau in [('hazard', 'hiểm hoạ', MAU['canh_bao']), ('normal', 'bình thường', MAU['dat'])]:
        for p in chon_deu(glob.glob(f'{ISB}/{lop}/*.mp4'), 5):
            ten = os.path.basename(p).replace('.mp4', '')
            muc.append((p, f'{nhan}\n{ten[:18]}', mau))
    return luoi_khung(muc, 2, 5, 'hinh-4-1-mau-clip-isafetybench.png')


def hinh_unsafe_net():
    """Hình 4.2 — một khung đại diện cho mỗi lớp trong 8 lớp của Unsafe-Net."""
    muc = []
    for thu_muc, nhan, an_toan in UNSAFE_NET:
        p = chon_deu(glob.glob(f'data/train/{thu_muc}/*.mp4'), 1)[0]
        muc.append((p, nhan, MAU['dat'] if an_toan else MAU['canh_bao']))
    return luoi_khung(muc, 2, 4, 'hinh-4-2-mau-clip-unsafe-net-8-lop.png', cao_moi_o=2.0)


def dem_clip():
    """Đếm .mp4 trên đĩa cho cả hai kho. In ra để đối chiếu với Chương 4."""
    isb = {lop: len(glob.glob(f'{ISB}/{lop}/*.mp4')) for lop in ('hazard', 'normal')}
    un = {thu_muc: (len(glob.glob(f'data/train/{thu_muc}/*.mp4')),
                    len(glob.glob(f'data/test/{thu_muc}/*.mp4'))) for thu_muc, _, _ in UNSAFE_NET}
    print(f"  iSafetyBench: hiểm hoạ {isb['hazard']}, bình thường {isb['normal']}, "
          f"tổng {sum(isb.values())}")
    print(f"  Unsafe-Net: huấn luyện {sum(a for a, _ in un.values())}, "
          f"kiểm thử {sum(b for _, b in un.values())}")
    return isb, un


def hinh_thanh_phan(isb, un):
    """Hình 4.3 — thành phần hai kho dữ liệu, đếm từ đĩa."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.6),
                                   gridspec_kw={'width_ratios': [1, 1.5], 'wspace': 0.30})

    ax1.barh(['hiểm hoạ', 'bình thường'], [isb['hazard'], isb['normal']],
             color=[MAU['canh_bao'], MAU['dat']], height=0.55)
    for y, v in enumerate([isb['hazard'], isb['normal']]):
        ax1.text(v - 14, y, f'{v} clip\n({v / sum(isb.values()) * 100:.1f}%)', va='center',
                 ha='right', fontsize=8, color='white', fontweight='bold')
    ax1.set_xlim(0, max(isb.values()) * 1.08)
    ax1.set_xlabel('số clip')
    ax1.set_title(f"(a) iSafetyBench — {sum(isb.values()):,} clip, nhị phân".replace(',', ','))
    ax1.grid(axis='y', visible=False)

    nhan = [n.split('\n')[0] for _, n, _ in UNSAFE_NET]
    tr = [un[t][0] for t, _, _ in UNSAFE_NET]
    te = [un[t][1] for t, _, _ in UNSAFE_NET]
    y = np.arange(len(nhan))
    ax2.barh(y, tr, color=MAU['chinh'], height=0.6, label='huấn luyện')
    ax2.barh(y, te, left=tr, color=MAU['phu'], height=0.6, label='kiểm thử')
    for i, (a, b) in enumerate(zip(tr, te)):
        ax2.text(a + b + 3, i, f'{a}+{b}', va='center', fontsize=7.5)
    ax2.set_yticks(y); ax2.set_yticklabels(nhan, fontsize=7.5)
    ax2.tick_params(axis='y', pad=1)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(a + b for a, b in zip(tr, te)) * 1.22)
    ax2.set_xlabel('số clip')
    ax2.set_title(f'(b) Unsafe-Net — {sum(tr)} huấn luyện / {sum(te)} kiểm thử, 8 lớp')
    ax2.legend(loc='lower right')
    ax2.grid(axis='y', visible=False)
    return luu(fig, 'hinh-4-3-thanh-phan-hai-bo-du-lieu.png')


if __name__ == '__main__':
    dung_kieu()
    print('Hình dữ liệu (Chương 4):')
    isb, un = dem_clip()
    hinh_isafetybench()
    hinh_unsafe_net()
    hinh_thanh_phan(isb, un)
