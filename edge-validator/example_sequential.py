#!/usr/bin/env python3
"""example_sequential.py — confidence sequence が『覗き見の罠』を防ぐことの実証。

真の効果がゼロ(null)のデータを、母数を増やしながら何度も覗いて
「有意になったら止める」をやると――
  ・素朴な固定NのCI → 高確率で偽陽性（自分を騙す）
  ・confidence sequence → 誤り率を alpha 付近に保つ
を、シミュレーションで示す。データは有界 {-1,0,+1}（ペア差の形）なので
CSは sigma=値域/2=1（保守版・finite-sample に誤り率を理論保証）を使う。
"""
import sys, random, statistics
from math import sqrt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # リポジトリルート
from shikinseki import edge_validator as ev

ALPHA = 0.10
TRIALS = 500
MAX_N = 500
MIN_N, STEP = 30, 10
SIGMA = 1.0  # {-1,0,1} の値域2 → sub-Gaussian σ=1（保守・厳密）


def null_value(rng):
    """平均ゼロ（効果なし）。多くは0、±1を対称に＝ペア差を模す。"""
    u = rng.random()
    if u < 0.12:
        return 1
    if u < 0.24:
        return -1
    return 0


def naive_fixed_ci_decides(xs):
    """素朴: 正規近似の90%CIが0を外れたら『発見』とみなす。"""
    n = len(xs)
    if n < 2:
        return False
    m = sum(xs) / n
    sd = statistics.pstdev(xs)
    if sd == 0:
        return m != 0
    se = sd / sqrt(n)
    return (m - 1.645 * se) > 0 or (m + 1.645 * se) < 0


def main():
    rng = random.Random(7)
    naive_fp = cs_fp = 0
    for _ in range(TRIALS):
        xs = [null_value(rng) for _ in range(MAX_N)]
        # 覗き見しながら『外れたら止める』
        hit = False
        n = MIN_N
        while n <= MAX_N:
            if naive_fixed_ci_decides(xs[:n]):
                hit = True
                break
            n += STEP
        naive_fp += hit
        cs = ev.sequential_scan(xs, alpha=ALPHA, sigma=SIGMA, min_n=MIN_N, step=STEP)
        cs_fp += (cs["crossed_at"] is not None)

    naive_rate = naive_fp / TRIALS
    cs_rate = cs_fp / TRIALS
    print("=" * 64)
    print(f"覗き見の罠 実証（真の効果=ゼロ・{TRIALS}試行・{MIN_N}〜{MAX_N}件を{STEP}刻みで覗く）")
    print("=" * 64)
    print(f"素朴な固定CI（覗いて外れたら止める）の偽陽性率: {naive_rate*100:5.1f}%"
          f"  ← 目標{int(ALPHA*100)}%を大幅超過")
    print(f"confidence sequence の偽陽性率:               {cs_rate*100:5.1f}%"
          f"  ← 目標{int(ALPHA*100)}%以内に制御")
    assert naive_rate >= 0.20, f"素朴CIの偽陽性が膨らむはず: {naive_rate}"
    assert cs_rate <= ALPHA + 0.02, f"CSは誤り率を保つはず: {cs_rate}"
    print("\n✅ 実証完了：『増やしながら覗いて、有意で止める』は素朴CIだと自分を騙す。")
    print("   confidence sequence なら、何度覗いても誤り率が保たれる＝安全に逐次確認できる。")


if __name__ == "__main__":
    main()
