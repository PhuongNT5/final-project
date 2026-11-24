#!/usr/bin/env python3
"""Pass-3 analysis: did unlimited time resolve the low-confidence clips?

WRITTEN AND COMMITTED BEFORE ANY PASS-3 ROW EXISTS. Pre-registration: amendment 3 of
tien-dang-ky-kappa-nhi-phan-isafetybench.md. Reading numbers out of here may change the thesis's
CONCLUSIONS; it may not change what this script computes.

The subgroup was chosen after seeing pass-1 results, so nothing here revises the primary estimand
(Cohen kappa 0.6933 on n = 150). This measures a mechanism, not the ceiling.
"""
import hashlib, json, pathlib, sys
import numpy as np

BASE = pathlib.Path(__file__).resolve().parent.parent / 'annotation_guide/kappa-isb-binary'
SEED = 'pass3-20261124'
L1, C1 = 22 / 36, 32 / 36          # pass-1 accuracy, both groups -- fixed in amendment 3
DWELL_MIN = 20.0                   # E0: median dwell on the test group, seconds
RNG = np.random.default_rng(20261124)


def kappa(a, b):
    po = (a == b).mean()
    pe = sum(((a == v).mean() * (b == v).mean()) for v in (0, 1))
    return (po - pe) / (1 - pe) if pe < 1 else float('nan')


def boot_ci(fn, *arr, b=10000):
    n = len(arr[0])
    vals = [fn(*[x[i] for x in arr]) for i in (RNG.integers(0, n, n) for _ in range(b))]
    vals = [v for v in vals if v == v]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else None


def line(name, val, ci=None, extra=''):
    s = f'{name:<44} {val:.4f}'
    if ci:
        s += f'  CI95 [{ci[0]:.4f}, {ci[1]:.4f}]'
    print(s + ('  ' + extra if extra else ''))


def main():
    key = json.loads((BASE / 'key.json').read_text())
    rows = [json.loads(l) for l in (BASE / 'annotations.jsonl').read_text().splitlines() if l.strip()]
    p1 = {r['kid']: r for r in rows if r['pass'] == 1 and r['kid'] != 'K096'}
    p3 = {}
    for r in rows:                                     # append-only: last row per kid wins
        if r['pass'] == 3:
            p3[r['kid']] = r
    if not p3:
        sys.exit('chua co du lieu pass 3 -- chay:\n'
                 '  .venv/bin/python tools/kappa-annotation-server.py events-pass3.json')

    low = sorted(k for k, r in p1.items() if r['confidence'] < 3)
    high = sorted(k for k, r in p1.items() if r['confidence'] == 3)
    ctrl = sorted(high, key=lambda k: hashlib.sha256(f'{SEED}|{k}'.encode()).hexdigest())[:36]
    want = set(low) | set(ctrl)
    missing = sorted(want - set(p3))
    if missing:
        print(f'CANH BAO: thieu {len(missing)}/72 clip -- chay mot phan, luat E chua ap dung\n',
              file=sys.stderr)

    def arrs(ks):
        ks = [k for k in ks if k in p3]
        g = np.array([key[k]['gold'] for k in ks])
        a1 = np.array([p1[k]['label'] for k in ks])
        a3 = np.array([p3[k]['label'] for k in ks])
        d3 = np.array([p3[k]['dwell_ms'] / 1000 for k in ks])
        return ks, g, a1, a3, d3

    kl, gl, l1a, l3a, dl = arrs(low)
    kc, gc, c1a, c3a, dc = arrs(ctrl)

    print('== pass 1: 17/11/2026   pass 3: 24/11/2026   cach nhau 7 ngay')
    print(f'== nhom thu nghiem n={len(kl)}   nhom doi chung n={len(kc)}\n')

    print('-- E0: phep can thiep co xay ra khong')
    med = float(np.median(dl))
    print(f"{'dwell trung vi nhom thu nghiem':<44} {med:.1f}s   (pass 1: 11.2s, nguong {DWELL_MIN}s)")
    e0 = med >= DWELL_MIN
    print(f"{'  -> phep can thiep':<44} {'DAT' if e0 else 'THAT BAI -- E1/E1b/E2 KHONG ap dung'}")
    print(f"{'dwell trung vi nhom doi chung':<44} {float(np.median(dc)):.1f}s   (pass 1: 7.7s)")

    print('\n-- do chinh xac so voi nhan goc')
    l3, c3 = float((l3a == gl).mean()), float((c3a == gc).mean())
    line('thu nghiem  pass 1', L1, extra=f'({int(L1*36)}/36)')
    line('thu nghiem  pass 3', l3, boot_ci(lambda g, a: (a == g).mean(), gl, l3a),
         f'({int((l3a == gl).sum())}/{len(kl)})')
    line('  thay doi L3 - L1', l3 - L1, extra=f'= {(l3-L1)*100:+.1f} pp')
    line('doi chung   pass 1', C1, extra=f'({int(C1*36)}/36)')
    line('doi chung   pass 3', c3, boot_ci(lambda g, a: (a == g).mean(), gc, c3a),
         f'({int((c3a == gc).sum())}/{len(kc)})')
    line('  thay doi C3 - C1', c3 - C1, extra=f'= {(c3-C1)*100:+.1f} pp')
    print(f"{'moc tam thuong (tra normal het)':<44} 0.5556   (20/36)")

    print('\n-- quy tac mac dinh "khong chac thi chon binh thuong" con khong')
    for nm, a1, a3, n in (('thu nghiem', l1a, l3a, len(kl)), ('doi chung', c1a, c3a, len(kc))):
        print(f"  {nm:<14} goi hazard: pass 1 = {int((a1==1).sum())}/{n}  ->  "
              f"pass 3 = {int((a3==1).sum())}/{n}")

    print('\n-- test-retest 7 ngay (thay cho con so 0.8397 da bi thoi phong)')
    ga1 = np.concatenate([l1a, c1a]); ga3 = np.concatenate([l3a, c3a])
    line('kappa pass 1 <-> pass 3, ca 72 clip', kappa(ga1, ga3), boot_ci(kappa, ga1, ga3))
    print(f"{'so clip tu doi y':<44} {int((ga1 != ga3).sum())}/{len(ga1)}")
    line('  rieng nhom doi chung', kappa(c1a, c3a))
    line('  rieng nhom thu nghiem', kappa(l1a, l3a))
    print(f"{'nhan ra clip (recognised)':<44} "
          f"{sum(1 for k in kl+kc if p3[k]['recognised'])}/{len(kl)+len(kc)}")

    print('\n-- kappa so voi nhan goc, tung nhom')
    line('thu nghiem pass 3', kappa(l3a, gl), boot_ci(kappa, l3a, gl))
    line('doi chung  pass 3', kappa(c3a, gc), boot_ci(kappa, c3a, gc))

    print('\n== PHAN QUYET (luat E chot truoc trong Sua doi 3)')
    dL, dC = (l3 - L1) * 100, (c3 - C1) * 100
    if missing:
        v = 'CHUA XET -- du lieu chua day du'
    elif c3 <= C1 - 0.10:
        v = ('E4: nguoi doc DA TROI khoi so tay. TOAN BO pass 3 KHONG DIEN GIAI DUOC, '
             'ke ca nhom thu nghiem')
    elif not e0:
        v = ('E0 that bai: chua bo duoc ap luc thoi gian. Chi bao cao duoc duoi dang '
             'test-retest 7 ngay; E1/E1b/E2 khong ap dung')
    elif dL >= 15 and c3 >= C1 - 0.05:
        v = ('E1: PHAN XU THAT. Tran nhan CAO HON 84.67%, mo hinh CON DU DIA. '
             'Lap luan "sat tran nhan" o muc 6 YEU DI va PHAI VIET LAI. Bat loi cho luan van')
    elif dL >= 15:
        v = ('E1b: CHI DICH NGUONG, khong phan xu. Khong nang uoc luong tran nhan. '
             'Muc 6 KHONG DOI')
    elif abs(dL) <= 10 and abs(dC) <= 5:
        v = ('E2: NHAP NHANG THAT, xac nhan bang phep can thiep. Thoi gian khong gioi han '
             'KHONG giai duoc nhung clip do. Lap luan "84.2% sat tran nhan" DUOC CHONG DO MANH')
    else:
        v = 'E3: bao cao so, KHONG nang thanh tuyen bo nao manh hon so cho phep'
    print(f'   L3-L1 = {dL:+.1f} pp   C3-C1 = {dC:+.1f} pp')
    print(f'   {v}')
    k72 = kappa(ga1, ga3)
    if k72 == k72 and k72 < 0.80 and not missing:
        print(f'\n   D3 NO: kappa test-retest = {k72:.4f} < 0.80 theo tien-dang-ky muc 5 '
              f'=> TOAN BO phep do tuyen bo KHONG DIEN GIAI DUOC')
    print('\n   Dai luong chinh KHONG doi: Cohen kappa 0.6933 tren n = 150. D2 van hieu luc.')
    print('   Bang cong khong doi: tam cong, hai dat, sau truot.   Cham test: 0 lan.')


if __name__ == '__main__':
    main()
