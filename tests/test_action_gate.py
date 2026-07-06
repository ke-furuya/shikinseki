#!/usr/bin/env python3
"""action_gate の回帰テスト＝「止める」側の経路を固定する。

このモジュールの存在意義は実行前ブロック。「不正なアクションを黙ってcommitする」が
最悪の故障モードなので、各invariantの違反検出と例外経路（＝否定側）を既知の答えで検証する。
"""
import unittest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shikinseki import action_gate as ag


class TestGateBlocks(unittest.TestCase):
    """error違反があれば actions は空（＝実行させない）。"""

    def test_required_fields_violation_blocks(self):
        policy = [("r", lambda s: True, lambda s: {"type": "x", "legs": None, "cost": 1})]
        res = ag.run(policy, {}, [ag.inv_required_fields(["type", "legs", "cost"])],
                     verbose=False)
        self.assertFalse(res["committed"])
        self.assertEqual(res["actions"], [])  # ブロック時は実行可アクションを返さない
        self.assertEqual(len(res["generated"]), 1)  # 生成自体は監査のため残る

    def test_contradiction_blocks(self):
        policy = [("up", lambda s: True, lambda s: {"target": "scale_up"}),
                  ("dn", lambda s: True, lambda s: {"target": "scale_down"})]
        res = ag.run(policy, {},
                     [ag.inv_not_contradict("target", [("scale_up", "scale_down")])],
                     verbose=False)
        self.assertFalse(res["committed"])
        self.assertEqual(res["actions"], [])

    def test_budget_pass_and_block_boundary(self):
        policy = [("a", lambda s: True, lambda s: [{"id": 1, "cost": 60}, {"id": 2, "cost": 40}])]
        ok = ag.run(policy, {}, [ag.inv_budget("cost", 100)], verbose=False)   # ちょうど=合格
        self.assertTrue(ok["committed"])
        ng = ag.run(policy, {}, [ag.inv_budget("cost", 99)], verbose=False)    # 超過=ブロック
        self.assertFalse(ng["committed"])


class TestExceptionPaths(unittest.TestCase):
    """例外は黙って握り潰さない＝監査に残す／検査自体の例外はerrorに昇格。"""

    def test_invariant_exception_escalates_to_error(self):
        # 検査関数が例外を吐いたら warn ではなく error（=ブロック）に昇格するはず
        boom = ("boom", lambda a, s: 1 / 0, "warn")
        res = ag.check([{"a": 1}], {}, [boom])
        self.assertFalse(res["passed"])
        self.assertEqual(res["errors"][0]["severity"], "error")

    def test_rule_exception_recorded_not_fatal(self):
        policy = [("bad", lambda s: 1 / 0, lambda s: {"a": 1}),
                  ("ok", lambda s: True, lambda s: {"b": 2})]
        actions, audit = ag.generate(policy, {})
        self.assertEqual(actions, [{"b": 2}])                       # 他ルールは生き残る
        self.assertTrue(any("bad" in e for e in audit["_ERROR"]))   # 例外は監査に残る


if __name__ == "__main__":
    unittest.main(verbosity=2)
