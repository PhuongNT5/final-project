"""Kiểu trình bày dùng chung cho mọi hình của luận văn tiếng Việt.

Một chỗ duy nhất đặt phông, cỡ chữ, bảng màu và hàm lưu, để 12 hình trong
`figures/luan-van/` trông như một bộ chứ không như 12 hình rời.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'figures/luan-van'

# DejaVu Sans phủ đủ dấu tiếng Việt; không dùng phông hệ thống để hình tái lập được ở máy khác.
PLT_RC = {
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#d9d9d9',
    'grid.linewidth': 0.6,
    'legend.frameon': False,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
}

# Bảng màu: an toàn với người mù màu đỏ–lục và vẫn phân biệt được khi in đen trắng.
MAU = {
    'chinh': '#1f4e79',      # xanh đậm — cấu hình được triển khai / giao thức công bố
    'phu': '#9ecae1',        # xanh nhạt — đối chứng / giao thức rò rỉ
    'canh_bao': '#c0504d',   # đỏ gạch — trượt, bỏ sót, đường hợp đồng
    'dat': '#2e7d32',        # lục — đạt
    'sàn': '#8c8c8c',        # xám — đường sàn
    'trung_tinh': '#404040',
}


def dung_kieu():
    plt.rcParams.update(PLT_RC)


def luu(fig, ten, le=None):
    """Lưu PNG 300 dpi vào figures/luan-van/ và in đường dẫn để nhật ký có vết.

    `le` nới thêm lề trắng khi nhãn trục nằm sát mép và bị cắt mất một nét.
    """
    os.makedirs(OUT, exist_ok=True)
    duong_dan = os.path.join(OUT, ten)
    fig.savefig(duong_dan, pad_inches=le if le is not None else PLT_RC['savefig.pad_inches'])
    plt.close(fig)
    print(f'  đã lưu  {duong_dan}')
    return duong_dan
