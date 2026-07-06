#!/usr/bin/env python3
"""example_advanced.py — 拡張4本柱の自己検証。
  A 過学習診断（PBO / PSR / DSR）
  B 選択的推論の3段化（staged_scan）
  C 測定の一致（Cohen's κ）
  D 意思決定層（採用判断 / 情報の価値）
"""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # リポジトリルート
from shikinseki import edge_validator as ev


def section(t):
    print("\n" + "=" * 62 + f"\n{t}\n" + "=" * 62)


# ── A 過学習診断 ──
section("A. 過学習診断（PBO / PSR / DSR）")
rng = random.Random(3)
S, T = 20, 240
noise = [[rng.gauss(0, 1) for _ in range(T)] for _ in range(S)]   # 実力ゼロの20戦略
pbo = ev.pbo_cscv(noise, n_blocks=10)
print(f"ノイズ20戦略から最良を選ぶ → PBO={pbo['pbo']*100:.0f}%（{pbo['n_combos']}通り）"
      f"  ＝高い＝in-sample最良はOOSで崩れる＝過学習")
assert pbo["pbo"] >= 0.3, pbo

real = [rng.gauss(0.15, 1) for _ in range(T)]                     # 本物のエッジ(平均+0.15)
psr_real = ev.probabilistic_sharpe_ratio(real)
# ノイズは1本だと運で良く見える（＝本プロジェクトの教訓そのもの）→平均で評価
noise_psrs = [ev.probabilistic_sharpe_ratio([rng.gauss(0, 1) for _ in range(T)])["psr"]
              for _ in range(50)]
mean_noise = sum(noise_psrs) / len(noise_psrs)
print(f"PSR: 本物エッジ={psr_real['psr']*100:.1f}%  ノイズ50本平均={mean_noise*100:.1f}%（≈50%＝無意味）")
assert psr_real["psr"] > 0.9, psr_real   # 分割＋値付き＝落ちたとき原因が見える
assert mean_noise < 0.7, mean_noise

trial_srs = [ev.sharpe(s) for s in noise] + [ev.sharpe(real)]    # 21本試した中の最良扱い
dsr = ev.deflated_sharpe_ratio(real, trial_srs)
print(f"DSR（21試行で割引）={dsr['dsr']*100:.1f}%  ＜ 素のPSR {psr_real['psr']*100:.1f}%"
      f"  ＝試行数を勘定すると確信度は下がる")
assert dsr["dsr"] < psr_real["psr"]

# 過学習が無いケース：1本だけ＝割引なしに近い
dsr1 = ev.deflated_sharpe_ratio(real, [ev.sharpe(real), 0.0])
print(f"（参考）試行2本だとDSR={dsr1['dsr']*100:.1f}%＝割引が小さい")


# ── B 選択的推論3段化 ──
section("B. 選択的推論：探索→検証→確認（staged_scan）")
# baselineに勝つ"本物の特徴"を1つ仕込み、無関係な特徴を混ぜる
recs = []
for i in range(1200):
    t = f"2025-{1 + i % 9:02d}-01"          # 9時点に分散→3分割可
    base = rng.random() * 0.6                # baseline予測(0-0.6・クリップ回避)
    flag = 1 if rng.random() < 0.5 else 0    # 本物の特徴
    outcome = 1 if rng.random() < base + 0.25 * flag else 0  # flagで明確に底上げ
    noise_feat = 1 if rng.random() < 0.5 else 0
    recs.append({"group": f"g{i}", "time": t, "baseline": base,
                 "outcome": outcome, "real": flag, "noise": noise_feat})
specs = [("本物の特徴", lambda r: r["real"] == 1),
         ("無関係ノイズ", lambda r: r["noise"] == 1)]
st = ev.staged_scan(recs, specs, estimand="P(outcome|特徴) − baseline織込み（予測残差）")
print(f"estimand: {st['estimand']}")
print(f"探索{st['n_disc']} / 検証{st['n_val']} / 確認{st['n_conf']}  確認済survivors={st['confirmed']}")
assert "本物の特徴" in st["confirmed"] and "無関係ノイズ" not in st["confirmed"]
st0 = ev.staged_scan(recs, specs)  # estimand未宣言
assert st0["note"].startswith("⚠️")
print(f"estimand未宣言なら警告: {st0['note']}")


# ── C 測定の一致 ──
section("C. アノテータ間一致（Cohen's κ）")
a = ["pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg"]
b_hi = list(a)                                   # 完全一致
b_lo = ["pos", "pos", "pos", "pos", "pos", "neg", "neg", "neg", "neg", "neg"]
print(f"一致大: κ={ev.cohens_kappa(a, b_hi)['kappa']:.2f}  "
      f"一致小: κ={ev.cohens_kappa(a, b_lo)['kappa']:.2f}")
assert ev.cohens_kappa(a, b_hi)["kappa"] == 1.0
assert ev.cohens_kappa(a, b_lo)["kappa"] < 0.3


# ── D 意思決定層 ──
section("D. 意思決定：採用判断 / 情報の価値")
# 効果+7.3pp・CI下限+2.7pp（感情分析の例）。1pp=価値1、Bの切替コスト=4とする
dv = ev.decision_value_ab(effect_mean=7.3, effect_lo=2.7, value_per_unit=1.0, switch_cost=4.0)
print(f"採用判断: 期待純益={dv['expected_net']:+.1f} 悲観純益={dv['worst_net']:+.1f} → {dv['recommendation']}")
assert dv["expected_net"] > 0
# 切替コストが高い(10)なら見送り
dv2 = ev.decision_value_ab(7.3, 2.7, 1.0, switch_cost=10.0)
print(f"切替コスト高: → {dv2['recommendation']}")
assert "見送り" in dv2["recommendation"]
# 情報の価値(VoI)：効果が明確なら集める価値は小、不確実なら大
se = (11.8 - 2.7) / (2 * 1.645)   # 感情分析の90%CI幅からSE概算(pp)
clear = ev.value_of_more_data(effect_mean=7.3, se=se, value_per_unit=1.0, sampling_cost=2.0)
print(f"明確なケース(効果+7.3/SE{se:.1f}): EVPI={clear['evpi']:.3f} vs コスト2.0 → {clear['verdict']}")
# 効果が小さく不確実(0±)なら、確かめる価値が出る
uncertain = ev.value_of_more_data(effect_mean=0.5, se=3.0, value_per_unit=1.0, sampling_cost=0.3)
print(f"灰色なケース(効果+0.5/SE3.0): EVPI={uncertain['evpi']:.3f} vs コスト0.3 → {uncertain['verdict']}")
assert clear["evpi"] < uncertain["evpi"]      # 不確実なほど情報の価値は大きい
assert uncertain["worth_collecting"] and not clear["worth_collecting"]

print("\n✅ 拡張4本柱すべて自己検証PASS（過学習診断/3段化/一致度/意思決定）")
