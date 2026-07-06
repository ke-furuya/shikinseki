#!/usr/bin/env python3
"""edge_validator の回帰テスト（標準ライブラリ unittest のみ・追加依存なし）。

方針＝「検証ツール自身が壊れていないことを検証する」。
各テストは①数学的に答えが分かる入力 か ②設計が主張する統計的性質 のどちらかで、
実装が正しいことを外から確認する。実行: python3 -m unittest -v （または python3 test_edge_validator.py）
"""
import math
import random
import unittest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shikinseki import edge_validator as ev


def const_records(value, n=40, value_key="value"):
    """n個の独立group（各1レコード）に同じ値。CIは点に潰れるはず。"""
    return [{"group": f"g{i}", value_key: value} for i in range(n)]


class TestNormalFunctions(unittest.TestCase):
    """正規分布のCDF/PPF＝数学的に厳密な答えがある。"""

    def test_cdf_at_zero_is_half(self):
        self.assertAlmostEqual(ev._norm_cdf(0.0), 0.5, places=9)

    def test_cdf_symmetry(self):
        for x in (0.3, 1.0, 2.5):
            self.assertAlmostEqual(ev._norm_cdf(x) + ev._norm_cdf(-x), 1.0, places=9)

    def test_ppf_known_quantiles(self):
        # 標準正規の 97.5% 点 = 1.959964（教科書値）
        self.assertAlmostEqual(ev._norm_ppf(0.975), 1.959963985, places=4)
        self.assertAlmostEqual(ev._norm_ppf(0.5), 0.0, places=6)

    def test_cdf_ppf_roundtrip(self):
        for p in (0.05, 0.2, 0.5, 0.8, 0.95):
            self.assertAlmostEqual(ev._norm_cdf(ev._norm_ppf(p)), p, places=4)


class TestMeanCI(unittest.TestCase):
    """group単位ブートストラップCI（seed固定で決定的）。"""

    def test_constant_collapses_to_point(self):
        r = ev.mean_ci(const_records(1.0))
        self.assertAlmostEqual(r["mean"], 1.0, places=9)
        self.assertAlmostEqual(r["ci_low"], 1.0, places=9)
        self.assertAlmostEqual(r["ci_high"], 1.0, places=9)

    def test_positive_constant_is_robust_positive(self):
        r = ev.mean_ci(const_records(0.4))
        self.assertTrue(r["robust_positive"])
        self.assertFalse(r["robust_negative"])

    def test_symmetric_zero_mean_straddles_zero(self):
        # 平均0付近・0をまたぐ → 頑健に非ゼロと言えない
        recs = [{"group": f"g{i}", "value": (1 if i % 2 else -1)} for i in range(40)]
        r = ev.mean_ci(recs)
        self.assertAlmostEqual(r["mean"], 0.0, places=9)
        self.assertFalse(r["robust_positive"])
        self.assertFalse(r["robust_negative"])

    def test_reliable_flag_tracks_min_groups(self):
        self.assertFalse(ev.mean_ci(const_records(0.5, n=10))["reliable"])
        self.assertTrue(ev.mean_ci(const_records(0.5, n=40))["reliable"])

    def test_empty_returns_none(self):
        self.assertIsNone(ev.mean_ci([]))


class TestRoiCI(unittest.TestCase):
    """看板 roi_ci のスモーク＋符号の妥当性。"""

    def test_profitable_bets_positive_roi(self):
        # payoff>cost が続けば ROI>100（=利益）で頑健に正
        recs = [{"group": f"g{i}", "cost": 1.0, "payoff": 1.5, "outcome": 1}
                for i in range(40)]
        r = ev.roi_ci(recs)
        self.assertIsNotNone(r)
        self.assertIn("ci_low", r)
        self.assertGreater(r["ci_low"], 100.0)


class TestConfidenceSequence(unittest.TestCase):
    """anytime-valid CI＝『覗き見の罠』を防ぐという設計主張を検証。"""

    def test_wider_than_fixed_ci(self):
        # 同じデータで CS の半径 > 固定正規CI(1.645σ/√n)。"何度でも覗ける自由"の代償。
        rng = random.Random(1)
        xs = [rng.gauss(0, 1) for _ in range(200)]
        cs = ev.confidence_sequence(xs, alpha=0.10)
        sigma = __import__("statistics").pstdev(xs)
        fixed_half = 1.645 * sigma / math.sqrt(len(xs))
        self.assertGreater(cs["radius"], fixed_half)

    def test_coverage_of_true_zero(self):
        # 真値0を1000回、CSが0を含む率が高い（誤検出<<alpha＝保守側の理論通り）。
        rng = random.Random(7)
        false_positive = 0
        trials = 1000
        for _ in range(trials):
            xs = [rng.gauss(0, 1) for _ in range(60)]
            cs = ev.confidence_sequence(xs, alpha=0.10)
            if cs["robust_positive"] or cs["robust_negative"]:
                false_positive += 1
        # 漸近CSは保守側＝名目alpha=0.10よりかなり低い偽陽性率であるべき
        self.assertLess(false_positive / trials, 0.05)

    def test_sequential_scan_decides_on_strong_effect(self):
        rng = random.Random(3)
        strong = [1.0 + rng.gauss(0, 0.3) for _ in range(200)]  # 明確な正の効果
        self.assertIsNotNone(ev.sequential_scan(strong)["crossed_at"])

    def test_degenerate_input_is_not_robust(self):
        # 【回帰】単一要素・全同値で「頑健に正」と誤判定しないこと（旧バグ）
        single = ev.confidence_sequence([5.0])
        self.assertTrue(single["degenerate"])
        self.assertFalse(single["robust_positive"])
        self.assertFalse(single["robust_negative"])
        allsame = ev.confidence_sequence([3.0] * 25)
        self.assertTrue(allsame["degenerate"])
        self.assertFalse(allsame["robust_positive"])


class TestNoMutation(unittest.TestCase):
    """【回帰】scan系が呼び出し側のレコードを汚染しないこと（旧バグ：_bin/_residのin-place注入）。"""

    def _records(self):
        rng = random.Random(0)
        return [{"group": f"g{i}", "time": f"t{i%6}", "baseline": rng.random(),
                 "outcome": rng.randint(0, 1), "kind": "x"} for i in range(120)]

    def test_residual_scan_does_not_mutate(self):
        recs = self._records()
        before = [dict(r) for r in recs]
        ev.residual_scan(recs, [("all", lambda r: True)])
        for r, b in zip(recs, before):
            self.assertNotIn("_resid", r)
            self.assertNotIn("_bin", r)
            self.assertEqual(r, b)

    def test_holdout_scan_does_not_mutate(self):
        recs = self._records()
        before = [dict(r) for r in recs]
        ev.holdout_scan(recs, [("all", lambda r: True)])
        for r, b in zip(recs, before):
            self.assertNotIn("_resid", r)
            self.assertEqual(r, b)


class TestCalibrationSparseBins(unittest.TestCase):
    """【回帰】較正のfit集合が nbins 未満でも binrate の参照がずれないこと
    （旧バグ：bin番号が飛ぶと _apply_calibration が欠番を引き、_resid が『outcome−0』になった）。"""

    def test_apply_matches_fit_when_bins_sparse(self):
        # fit側8件 < nbins=15 ＝ bin番号が飛ぶ。outcomeは全て1 → fit自身への適用残差は厳密に0のはず
        fit = [{"group": f"g{i}", "baseline": i / 10, "outcome": 1} for i in range(8)]
        cuts, binrate = ev._fit_calibration(fit, nbins=15)
        for r in ev._apply_calibration(fit, cuts, binrate):
            self.assertAlmostEqual(r["_resid"], 0.0, places=9)


class TestNaNHandling(unittest.TestCase):
    """【回帰】NaN/Infが黙って平均に伝播しないこと。"""

    def test_mean_ci_drops_nan(self):
        recs = ([{"group": f"g{i}", "value": 1.0} for i in range(20)]
                + [{"group": "gx", "value": float("nan")}])
        r = ev.mean_ci(recs)
        self.assertTrue(math.isfinite(r["mean"]))
        self.assertAlmostEqual(r["mean"], 1.0, places=9)


class TestPowerRequired(unittest.TestCase):
    """検出力＝必要Nの単調性と二次スケールを検証。"""

    def test_smaller_effect_needs_more_n(self):
        rng = random.Random(5)
        xs = [rng.gauss(0, 1) for _ in range(100)]
        big = ev.power_required_mean(xs, mde=0.5)["need_n"]
        small = ev.power_required_mean(xs, mde=0.1)["need_n"]
        self.assertLess(big, small)

    def test_halving_mde_quadruples_n(self):
        xs = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5]  # 固定SD
        n1 = ev.power_required_mean(xs, mde=0.4)["need_n"]
        n2 = ev.power_required_mean(xs, mde=0.2)["need_n"]
        self.assertAlmostEqual(n2 / n1, 4.0, places=6)


class TestCohensKappa(unittest.TestCase):
    """名義一致κ＝完全一致/完全不一致で厳密な答え。"""

    def test_perfect_agreement(self):
        r = ev.cohens_kappa([1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0])
        self.assertAlmostEqual(r["kappa"], 1.0, places=9)
        self.assertAlmostEqual(r["percent_agreement"], 1.0, places=9)

    def test_complete_disagreement(self):
        # po=0, pe=0.5 → κ=-1
        r = ev.cohens_kappa([1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1])
        self.assertAlmostEqual(r["kappa"], -1.0, places=9)


class TestSharpeAndOverfit(unittest.TestCase):
    """Sharpe/DSR/PBO＝過学習診断の性質を検証。"""

    def test_sharpe_known_value(self):
        # [1,2,3]: mean=2, 標本SD(n-1)=stdev=1.0 → sharpe=2.0
        self.assertAlmostEqual(ev.sharpe([1, 2, 3]), 2.0, places=9)

    def test_sharpe_constant_is_zero(self):
        self.assertEqual(ev.sharpe([5, 5, 5]), 0.0)

    def test_dsr_kills_best_of_pure_noise(self):
        # 真のエッジ無しの多戦略から選んだ最良は「本物」でない → DSR低い
        rng = random.Random(11)
        trial_sharpes = [ev.sharpe([rng.gauss(0, 1) for _ in range(50)]) for _ in range(60)]
        best = max(trial_sharpes)
        best_series = [rng.gauss(best * 1.0, 1) for _ in range(50)]  # それっぽい系列
        dsr = ev.deflated_sharpe_ratio(best_series, trial_sharpes)
        self.assertLess(dsr["dsr"], 0.9)  # 多試行で割り引かれ「確信」に至らない
        self.assertGreater(dsr["n_trials"], 2)

    def test_pbo_high_for_pure_noise(self):
        # 全戦略が純ノイズ → in-sample最良はout-of-sampleで崩れる → PBO高め
        rng = random.Random(13)
        perf = [[rng.gauss(0, 1) for _ in range(120)] for _ in range(20)]
        self.assertGreater(ev.pbo_cscv(perf)["pbo"], 0.3)

    def test_pbo_low_for_one_real_edge(self):
        # 1戦略だけ全期間で明確に優位 → out-of-sampleでも勝ち続ける → PBO低い
        rng = random.Random(17)
        perf = [[rng.gauss(0, 1) for _ in range(120)] for _ in range(19)]
        perf.append([rng.gauss(3.0, 1) for _ in range(120)])  # 本物のエッジ
        self.assertLess(ev.pbo_cscv(perf)["pbo"], 0.2)


class TestDecisionLayer(unittest.TestCase):
    """意思決定層＝しきい値ロジックの厳密な分岐。"""

    def test_adopt_when_worst_case_positive(self):
        r = ev.decision_value_ab(effect_mean=0.1, effect_lo=0.05,
                                  value_per_unit=1000, switch_cost=10)
        self.assertGreater(r["worst_net"], 0)
        self.assertIn("採用", r["recommendation"])

    def test_reject_when_no_expected_value(self):
        r = ev.decision_value_ab(effect_mean=0.0, effect_lo=-0.1,
                                  value_per_unit=1000, switch_cost=10)
        self.assertIn("見送り", r["recommendation"])

    def test_more_data_not_worth_when_certain(self):
        # se=0（不確実性ゼロ）なら追加データの価値ゼロ → 集める価値なし
        r = ev.value_of_more_data(effect_mean=0.2, se=0.0,
                                  value_per_unit=1000, sampling_cost=5)
        self.assertFalse(r["worth_collecting"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
