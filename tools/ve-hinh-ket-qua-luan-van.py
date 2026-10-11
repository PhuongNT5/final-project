"""Sinh năm hình KẾT QUẢ cho Chương 6, tính lại từ các tệp dự đoán ngoài fold đã lưu.

Không con số nào trong các hình này được gõ lại từ bảng của luận văn, trừ bảng phán quyết
cổng (Bảng 6.8) vốn là đầu ra của `tools/adjudicate-r48-gates.py`. Script tự ĐỐI CHIẾU các
giá trị tính lại với con số đã công bố và in cảnh báo nếu lệch.

Dùng:  .venv/bin/python tools/ve-hinh-ket-qua-luan-van.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from hinh_luan_van_style import MAU, dung_kieu, luu

SW = 'sweep_results/relation_tokens'
TIEU_DE = 84.2          # Bảng 6.1 — BA của cấu hình được triển khai B1
SAN_DEM = 63.3          # Bảng 6.1 — đường sàn ba đặc trưng đếm, tiền đăng ký
BA_FOLD_CONG_BO = [85.8, 88.4, 74.7, 79.1, 83.6, 67.5, 97.2, 96.9, 81.0, 86.2]   # mục 6.2.1


def _nap_cong_cu():
    """Mượn fold_map/per_fold_ba của tools/analyse-backbone-ladder.py — cùng bộ chia với driver."""
    spec = importlib.util.spec_from_file_location('abl', 'tools/analyse-backbone-ladder.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def doc(ten):
    d = json.load(open(f'{SW}/{ten}'))
    y = np.array([r['y_true'] for r in d['rows']])
    p = np.array([r['y_pred'] for r in d['rows']])
    pr = np.array([r['prob_unsafe'] for r in d['rows']])
    return y, p, pr


def ba_cua(y, p):
    return 50 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())


def kiem(ten, tinh_lai, cong_bo, dung_sai=0.1):
    dat = abs(tinh_lai - cong_bo) <= dung_sai
    print(f"  {'OK ' if dat else '!! '} {ten}: tính lại {tinh_lai:.1f} / công bố {cong_bo:.1f}")
    return dat


def hinh_ba_theo_fold():
    """Hình 6.1 — BA từng fold giữ lại của B1 dưới leave-source-out."""
    abl = _nap_cong_cu()
    fmap = abl.fold_map()
    pf = abl.per_fold_ba(abl.rows_of(f'{SW}/enc_isbclip_B1_lso-1100_headline-none_oof.json'), fmap)
    gia_tri = [pf[f'f{i}'] for i in range(1, 11)]
    for i, (a, b) in enumerate(zip(gia_tri, BA_FOLD_CONG_BO), 1):
        kiem(f'BA fold {i}', a, b)

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    mau = [MAU['canh_bao'] if v < TIEU_DE else MAU['chinh'] for v in gia_tri]
    ax.bar(range(1, 11), gia_tri, color=mau, width=0.66)
    for i, v in enumerate(gia_tri, 1):
        ax.text(i, v + 1.2, f'{v:.1f}', ha='center', fontsize=7.5)
    ax.hlines(TIEU_DE, 0.4, 10.6, color=MAU['trung_tinh'], lw=1.3, ls='--')
    ax.text(10.8, TIEU_DE, f'tiêu đề\n{TIEU_DE}%', va='center', fontsize=7.5,
            color=MAU['trung_tinh'])
    ax.hlines(SAN_DEM, 0.4, 10.6, color=MAU['sàn'], lw=1.3, ls=':')
    ax.text(10.8, SAN_DEM, f'sàn đếm\n{SAN_DEM}%', va='center', fontsize=7.5, color=MAU['sàn'])
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.4, 13.2)
    ax.set_ylim(50, 103)
    ax.set_xlabel('fold giữ lại (leave-source-out, 10 fold × 3 hạt giống)')
    ax.set_ylabel('độ chính xác cân bằng (%)')
    ax.grid(axis='x', visible=False)
    return luu(fig, 'hinh-6-1-ba-theo-fold-cua-b1.png')


def _ve_ma_tran(ax, cm, tieu_de):
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ty = cm[i, j] / cm[i].sum() * 100
            ax.text(j, i, f'{cm[i, j]}\n({ty:.1f}%)', ha='center', va='center', fontsize=9,
                    color='white' if ty > 55 else MAU['trung_tinh'])
    ax.set_xticks([0, 1]); ax.set_xticklabels(['dự đoán:\nbình thường', 'dự đoán:\nhiểm hoạ'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['thật:\nbình thường', 'thật:\nhiểm hoạ'])
    ax.set_title(tieu_de, fontsize=9)
    ax.grid(False)


def hinh_ma_tran_nham_lan():
    """Hình 6.2 — ma trận nhầm lẫn của B1 dưới hai giao thức chia dữ liệu."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.5))
    for ax, tep, nhan in [
            (ax1, 'enc_isbclip_B1_lso-1100_headline-none_oof.json', 'leave-source-out (công bố)'),
            (ax2, 'enc_isbclip_B1_rand-1100_headline-none_oof.json', 'chia ngẫu nhiên (tài liệu)')]:
        y, p, _ = doc(tep)
        cm = confusion_matrix(y, p)
        ba = ba_cua(y, p)
        kiem(f'BA {nhan}', ba, 84.2 if 'lso' in tep else 97.7, 0.1)
        _ve_ma_tran(ax, cm, f'({"a" if ax is ax1 else "b"}) {nhan}\nBA = {ba:.1f}%, '
                            f'bỏ sót {cm[1, 0]}, báo động giả {cm[0, 1]}')
    fig.subplots_adjust(wspace=0.45)
    return luu(fig, 'hinh-6-2-ma-tran-nham-lan-b1-hai-giao-thuc.png')


def hinh_roc_va_do_tin_cay():
    """Hình 6.3 — đường ROC và biểu đồ độ tin cậy của B1, hai giao thức."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.7))
    for tep, nhan, mau in [
            ('enc_isbclip_B1_lso-1100_headline-none_oof.json', 'leave-source-out', MAU['chinh']),
            ('enc_isbclip_B1_rand-1100_headline-none_oof.json', 'chia ngẫu nhiên', MAU['phu'])]:
        y, _, pr = doc(tep)
        auc, ap = roc_auc_score(y, pr), average_precision_score(y, pr)
        if 'lso' in tep:
            kiem('ROC-AUC (LSO)', auc * 100, 89.7, 0.1)
            kiem('PR-AUC (LSO)', ap * 100, 79.2, 0.1)
        fpr, tpr, _ = roc_curve(y, pr)
        ax1.plot(fpr, tpr, color=mau, lw=1.8, label=f'{nhan} — ROC-AUC {auc:.3f}')
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(pr, bins) - 1, 0, 9)
        x = [pr[idx == b].mean() for b in range(10) if (idx == b).any()]
        yy = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
        ax2.plot(x, yy, 'o-', color=mau, lw=1.6, ms=4, label=nhan)
    ax1.plot([0, 1], [0, 1], color=MAU['sàn'], ls=':', lw=1)
    ax1.set_xlabel('tỷ lệ báo động giả'); ax1.set_ylabel('độ nhạy với hiểm hoạ')
    ax1.set_title('(a) đường ROC'); ax1.legend(loc='lower right')
    ax2.plot([0, 1], [0, 1], color=MAU['sàn'], ls=':', lw=1, label='hiệu chỉnh hoàn hảo')
    ax2.set_xlabel('xác suất dự đoán là hiểm hoạ')
    ax2.set_ylabel('tần suất hiểm hoạ quan sát được')
    ax2.set_title('(b) biểu đồ độ tin cậy'); ax2.legend(loc='upper left')
    fig.subplots_adjust(wspace=0.3)
    return luu(fig, 'hinh-6-3-roc-va-do-tin-cay-b1.png')


def _ve_doc(ax, bo, noi_bat, lech):
    """Vẽ một bảng dốc: mỗi mô hình một đoạn từ leave-source-out sang chia ngẫu nhiên."""
    for (nhan, goc, cb_lso, cb_rand), dy in zip(bo, lech):
        lso = ba_cua(*doc(f'{goc}_lso-1100_headline-none_oof.json')[:2])
        rand = ba_cua(*doc(f'{goc}_rand-1100_headline-none_oof.json')[:2])
        kiem(f'{nhan} LSO', lso, cb_lso); kiem(f'{nhan} ngẫu nhiên', rand, cb_rand)
        mau = MAU['chinh'] if nhan == noi_bat else MAU['sàn']
        ax.plot([0, 1], [lso, rand], 'o-', color=mau, ms=5,
                lw=2.2 if nhan == noi_bat else 1.3, zorder=3 if nhan == noi_bat else 2)
        ax.text(-0.05, lso, f'{lso:.1f}', ha='right', va='center', fontsize=8, color=mau)
        ax.text(1.05, rand + dy, f'{rand:.1f}  {nhan}', ha='left', va='center', fontsize=8,
                color=mau)
    ax.set_xticks([0, 1])
    # Nhãn trục ngắn để hai mốc không chạm nhau; chú giải đầy đủ ở nhan đề hình.
    ax.set_xticklabels(['LSO\n(công bố)', 'ngẫu nhiên\n(tài liệu)'], fontsize=8)
    ax.grid(axis='x', visible=False)


def hinh_lam_phat_giao_thuc():
    """Hình 6.4 — cùng mô hình, cùng clip, chỉ đổi phép chia; hai làn vẽ RIÊNG."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.0),
                                   gridspec_kw={'wspace': 0.70})

    _ve_doc(ax1, [('B1 toàn khung', 'enc_isbclip_B1', 84.2, 97.7)], 'B1 toàn khung', [0])
    ax1.set_xlim(-0.42, 1.95)
    ax1.set_ylim(80, 100)
    ax1.set_ylabel('độ chính xác cân bằng (%)')
    ax1.set_title('(a) làn A — cấu hình được triển khai', fontsize=9.5)
    ax1.text(0.52, 86.3, '+13.5 pp\nCI theo cụm [+8.0, +20.0]\np = 1.39e-41', ha='left',
             fontsize=8, color=MAU['chinh'])

    bo_b = [('E3 DeepSets', 'enc_isb_E3', 75.2, 90.6), ('E1 d=64', 'enc_isb_E1_d64', 78.1, 91.3),
            ('E1 d=96', 'enc_isb_E1_d96', 77.0, 91.9),
            ('E0 RelTransformer', 'enc_isb_E0', 80.3, 90.7)]
    _ve_doc(ax2, bo_b, 'E3 DeepSets', [-1.3, 0.6, 1.3, -0.1])
    ax2.set_xlim(-0.42, 2.75)
    ax2.set_ylim(72, 95)
    ax2.set_title('(b) làn B — thang encoder quan hệ', fontsize=9.5)
    ax2.text(1.06, 78.6, 'E3: +15.4 pp\nCI theo cụm\n[+10.6, +21.5]', ha='left', fontsize=8,
             color=MAU['chinh'])

    fig.suptitle('Lạm phát giao thức: cùng mô hình, cùng clip, chỉ đổi phép chia\n'
                 'LSO = leave-source-out (giao thức công bố); ngẫu nhiên = giao thức tài liệu',
                 fontsize=10.5, y=1.045)
    fig.text(0.5, -0.07,
             'Hai bảng là hai phép đo ĐỘC LẬP của cùng một hiện tượng trên hai làn khác nhau: '
             'không lấy trung bình,\nkhông đặt cạnh nhau như hai lần đo của một đại lượng '
             '(mục 6.2.7, điều (2)).', ha='center', fontsize=8, color=MAU['trung_tinh'])
    return luu(fig, 'hinh-6-4-lam-phat-giao-thuc.png')


def hinh_phan_quyet_cong():
    """Hình 6.5 — tám cổng phán quyết tiền đăng ký, Bảng 6.8 (tools/adjudicate-r48-gates.py)."""
    cong = [('G10  encoder > sàn đếm', 8.0, 1.9, 13.5, True),
            ('G11  encoder > toàn khung', 1.1, -4.8, 6.9, False),
            ('G12  giao thức rò rỉ thổi phồng', 15.4, 10.6, 21.5, True),
            ('G13  dung lượng tăng → kết quả tăng', -1.1, None, None, False),
            ('G14  phân rã thực thể > toàn khung', -0.5, -5.5, 5.0, False),
            ('G15  ESTA > encoder thực thể tốt nhất', 3.5, -2.3, 10.0, False),
            ('G16  ESTA > xương sống video', 2.3, -2.3, 7.5, False),
            ('G18  hợp nhất hai luồng > đơn luồng', -0.3, -3.7, 2.5, False)]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    vi_tri, ten = [], []
    for i, (nhan, d, lo, hi, dat) in enumerate(cong):
        y = len(cong) - i
        vi_tri.append(y); ten.append(nhan)
        mau = MAU['dat'] if dat else MAU['canh_bao']
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=mau, lw=2, solid_capstyle='butt')
            for x in (lo, hi):
                ax.plot([x, x], [y - .16, y + .16], color=mau, lw=1.4)
        else:
            ax.text(d + 1.0, y, 'không có khoảng tin cậy', fontsize=7, va='center',
                    color=MAU['sàn'])
        ax.plot(d, y, 'o', color=mau, ms=6, zorder=3)
        ax.text(24.0, y, 'ĐẠT' if dat else 'TRƯỢT', fontsize=8, color=mau, va='center',
                fontweight='bold' if dat else 'normal')
    ax.axvline(0, color=MAU['trung_tinh'], lw=1.1)
    ax.set_yticks(vi_tri); ax.set_yticklabels(ten, fontsize=8)
    ax.set_ylim(0.3, len(cong) + 0.7)
    ax.set_xlim(-7.5, 27.5)
    ax.set_xticks(range(-5, 26, 5))
    ax.set_xlabel('Δ độ chính xác cân bằng (điểm phần trăm)\n'
                  'thanh ngang = khoảng tin cậy bootstrap theo cụm trên 151 nguồn')
    ax.grid(axis='y', visible=False)
    return luu(fig, 'hinh-6-5-phan-quyet-tam-cong-tien-dang-ky.png')


if __name__ == '__main__':
    dung_kieu()
    print('Hình kết quả (Chương 6) — đối chiếu tính lại với số đã công bố:')
    hinh_ba_theo_fold()
    hinh_ma_tran_nham_lan()
    hinh_roc_va_do_tin_cay()
    hinh_lam_phat_giao_thuc()
    hinh_phan_quyet_cong()
