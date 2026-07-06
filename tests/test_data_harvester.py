#!/usr/bin/env python3
"""data_harvester の回帰テスト＝完全性ゲートとSSRFガードを既知の答えで固定する。"""
import io
import unittest
from contextlib import redirect_stdout

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shikinseki import data_harvester as dh


class TestHttpJsonSchemeGuard(unittest.TestCase):
    """【回帰】http(s)以外のスキームはネットワークに触る前に拒否（SSRF/ローカル読み取り防止）。"""

    def test_file_scheme_rejected(self):
        with self.assertRaises(ValueError):
            dh.http_json("file:///etc/hosts")

    def test_ftp_scheme_rejected(self):
        with self.assertRaises(ValueError):
            dh.http_json("ftp://example.com/x.json")


class TestMapping(unittest.TestCase):
    """dig / code_map ＝ 決定論的な既知の答え。"""

    def test_dig_str_and_list_paths(self):
        raw = {"a": {"b": [10, {"c": 7}]}}
        self.assertEqual(dh.dig(raw, "a.b.1.c"), 7)
        self.assertEqual(dh.dig(raw, ["a", "b", 0]), 10)
        self.assertIsNone(dh.dig(raw, "a.x.y"))  # 欠損は例外でなくNone

    def test_unknown_code_is_visible(self):
        # 未知コードは黙ってNoneにせず "?"+元値 で可視化（ゲートが拾う）
        self.assertEqual(dh.apply_code_map(9, {"1": "G1"}), "?9")
        self.assertEqual(dh.apply_code_map(1, {"1": "G1"}), "G1")


class TestCompletenessGate(unittest.TestCase):
    """ゲート＝必須欠損/取得失敗/未収集で FAIL、全充足で PASS。"""

    SCHEMA = {"required": ["odds"], "optional_empty": ["pad"], "code_maps": {}}

    def _gate(self, records, **kw):
        with redirect_stdout(io.StringIO()):
            return dh.completeness_gate(records, self.SCHEMA, **kw)

    def test_missing_required_fails(self):
        r = self._gate({"1": {"odds": None, "pad": ""}})
        self.assertFalse(r["passed"])
        self.assertEqual(r["missing"], {"odds": 1})

    def test_fetch_error_fails_and_optional_empty_ok(self):
        r = self._gate({"1": {"odds": 2.5, "pad": ""}, "2": {"_error": "HTTP 502"}})
        self.assertFalse(r["passed"])           # 取得失敗はFAIL
        self.assertEqual(r["missing"], {})      # padの空は正当な欠落＝欠損に数えない

    def test_reconciliation_detects_uncollected(self):
        r = self._gate({"1": {"odds": 2.5}}, expected_ids=["1", "2"])
        self.assertFalse(r["passed"])
        self.assertEqual(r["recon"]["missing"], {"2"})

    def test_all_present_passes(self):
        r = self._gate({"1": {"odds": 2.5, "pad": "A"}}, expected_ids=["1"])
        self.assertTrue(r["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
