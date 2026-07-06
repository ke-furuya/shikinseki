#!/usr/bin/env python3
"""leak_guard の回帰テスト＝H/M/L?/L 4分級と重複検出を既知の答えで固定する。"""
import unittest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shikinseki import leak_guard as lg


class TestReappearanceLevels(unittest.TestCase):
    """4分級すべてを1つの合成データで固定（従来はexampleのH判定のみだった）。"""

    def test_all_levels(self):
        recs = [
            # r1: 本命A（baseline最小）が後で昇格再登場 → H（結果が逆算可・最危険）
            {"group": "r1", "entity": "A", "time": 1, "baseline": 0.1, "level": 1},
            {"group": "r1", "entity": "B", "time": 1, "baseline": 0.5, "level": 1},
            {"group": "r2", "entity": "A", "time": 2, "baseline": 0.3, "level": 2},
            # r3: 本命でない上位Dが昇格 → M
            {"group": "r3", "entity": "C", "time": 1, "baseline": 0.2, "level": 1},
            {"group": "r3", "entity": "D", "time": 1, "baseline": 0.4, "level": 1},
            {"group": "r4", "entity": "D", "time": 2, "baseline": 0.4, "level": 2},
            # r5: 2件が昇格なしで再登場 → L?（履歴で結果露出の可能性＝一瞥）
            {"group": "r5", "entity": "E", "time": 1, "baseline": 0.3, "level": 1},
            {"group": "r5", "entity": "F", "time": 1, "baseline": 0.6, "level": 1},
            {"group": "r6", "entity": "E", "time": 2, "baseline": 0.3, "level": 1},
            {"group": "r6", "entity": "F", "time": 2, "baseline": 0.6, "level": 1},
        ]
        flags = lg.reappearance_flags(recs, level_key="level")
        self.assertEqual(flags["r1"], "H")
        self.assertEqual(flags["r3"], "M")
        self.assertEqual(flags["r5"], "L?")
        self.assertEqual(flags["r6"], "L")  # 以後の再登場なし＝安全

    def test_without_level_key_reappearance_is_weak_risk(self):
        # level_keyなし＝「後に再登場した事実」自体を弱いリスクとして扱う
        recs = [{"group": "g1", "entity": "A", "time": 1, "baseline": 0.1},
                {"group": "g2", "entity": "A", "time": 2, "baseline": 0.1}]
        self.assertEqual(lg.reappearance_flags(recs)["g1"], "H")  # 本命Aが再登場


class TestDuplicateCheck(unittest.TestCase):

    def test_duplicate_detected(self):
        recs = [{"entity": "X", "time": 1}, {"entity": "X", "time": 1},
                {"entity": "Y", "time": 1}]
        dup = lg.duplicate_check(recs)
        self.assertEqual(dup, [{"entity": "X", "time": 1, "count": 2}])

    def test_no_duplicates_empty(self):
        recs = [{"entity": "X", "time": 1}, {"entity": "X", "time": 2}]
        self.assertEqual(lg.duplicate_check(recs), [])


class TestSplitContamination(unittest.TestCase):

    def test_straddling_group_flagged(self):
        recs = [{"group": "a", "time": 1}, {"group": "a", "time": 3},   # またぐ
                {"group": "b", "time": 1}, {"group": "c", "time": 3}]   # またがない
        self.assertEqual(lg.split_contamination(recs, split_time=2), ["a"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
