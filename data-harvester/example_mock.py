#!/usr/bin/env python3
"""
example_mock.py — data_harvester を、わざと欠陥を仕込んだ合成ソースで実証。
「完全性ゲートが、欠損・未知コード・取得失敗・未収集を全部捕まえる」ことを証明する。

別ドメインへの移植は、この fetch_fn と スキーマ を書き換えるだけ（エンジンは不変）。
末尾に実APIアダプタの書き方の雛形も載せてある（取得→検証の繋ぎ方）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # リポジトリルート
from shikinseki import data_harvester as dh

# ── 合成ソース（生API想定・ネスト構造・わざと欠陥入り）──
RAW = {
    "A01": {"info": {"price": 3.4, "gate": 5}, "tag": {"grade": "1"}},          # 正常
    "A02": {"info": {"price": 7.1, "gate": 2}, "tag": {"grade": "9"}},          # grade=9 は未知コード
    "A03": {"info": {"price": 12.0},            "tag": {"grade": "3"}},          # gate 欠損(必須)
    "A04": "__BOOM__",                                                          # 取得失敗を誘発
    # A05 は expected に居るが RAW に無い → 未収集
}

def fake_fetch(entity):
    raw = RAW[entity]
    if raw == "__BOOM__":
        raise ConnectionError("502 Bad Gateway")
    return raw

# 生コード → きれいなスキーマ への対応表
CODE_MAPS = {"grade": {"1": "G1", "2": "G2", "3": "OP", "4": "条件"}}  # 9 は未定義 → "?9"

# 生のネストパス → 出力フィールド
FIELD_SPEC = [
    ("odds",  "info.price"),
    ("gate",  "info.gate"),
    ("grade", "tag.grade", "grade"),  # コードマップ適用
]

# 下流（分析）が必要とするものの定義＝ゲートの基準
SCHEMA = {
    "required": ["odds", "gate"],     # 欠ければ赤で止める
    "optional_empty": [],
    "code_maps": CODE_MAPS,
}

if __name__ == "__main__":
    entities = ["A01", "A02", "A03", "A04"]
    expected = ["A01", "A02", "A03", "A04", "A05"]   # A05 は本来あるべき

    print("="*60 + "\n① 欠陥入りデータ → ゲートは FAIL を返すはず\n" + "="*60)
    raw = dh.harvest(entities, fake_fetch)
    clean = {rid: (r if (isinstance(r, dict) and r.get("_error"))
                   else dh.map_record(r, FIELD_SPEC, CODE_MAPS))
             for rid, r in raw.items()}
    rep = dh.completeness_gate(clean, SCHEMA, expected_ids=expected, label="件")
    assert rep["passed"] is False
    assert rep["missing"].get("gate") == 1          # A03 の gate 欠損を検出
    assert rep["unknown_codes"].get("grade") == 1   # A02 の grade=9 を検出
    assert len(rep["fetch_errors"]) == 1            # A04 の取得失敗を検出
    assert rep["recon"]["missing"] == {"A04", "A05"}  # A04(取得失敗=使えない)とA05(未取得)を検出

    print("\n" + "="*60 + "\n② 欠陥を直したデータ → ゲートは PASS を返すはず\n" + "="*60)
    RAW["A02"]["tag"]["grade"] = "2"
    RAW["A03"]["info"]["gate"] = 7
    RAW["A05"] = {"info": {"price": 5.0, "gate": 1}, "tag": {"grade": "4"}}
    entities2 = ["A01", "A02", "A03", "A04b", "A05"]
    RAW["A04b"] = {"info": {"price": 2.2, "gate": 3}, "tag": {"grade": "1"}}
    raw2 = dh.harvest(entities2, fake_fetch)
    clean2 = {rid: dh.map_record(r, FIELD_SPEC, CODE_MAPS) for rid, r in raw2.items()}
    rep2 = dh.completeness_gate(clean2, SCHEMA, expected_ids=entities2, label="件")
    assert rep2["passed"] is True

    print("\n✅ 実証完了：ゲートが欠損・未知コード・取得失敗・未収集を全て捕捉し、")
    print("   修正後は PASS。これが『ゴミを下流に流さない』仕組み。")


# ──────────────────────────────────────────────────────────────
# 参考：実APIアダプタの雛形（取得→edge-validator への繋ぎ方）
#   1エンティティにつき複数エンドポイント(summary/detail/stats…)を取得して統合する型。
#   ※URLは架空。自分の対象APIに合わせて fetch_fn とスキーマの3点だけ書き換える。
#
# def api_fetch(eid):
#     base = "https://api.example.com"
#     def J(t):  # /entity/{eid}/{type} が JSON を返す想定
#         return dh.http_json(f"{base}/entity/{eid}/{t}")["body"]
#     return {"summary": J("summary"), "detail": J("detail"), "stats": J("stats")}
#
# CODE_MAPS = {"grade": {"1": "A", "2": "B", "3": "C"}}
# FIELD_SPEC = [
#     ("price", "summary.price"), ("rank", "detail.rank"), ("grade", "stats.grade", "grade"),
# ]
# SCHEMA = {"required": ["price", "rank"], "optional_empty": ["grade"], "code_maps": CODE_MAPS}
#
# raw   = dh.harvest(eid_list, api_fetch, retries=2, pause=0.3)
# clean = {e: dh.map_record(v, FIELD_SPEC, CODE_MAPS) for e, v in raw.items()}
# gate  = dh.completeness_gate(clean, SCHEMA, expected_ids=eid_list)
# if gate["passed"]:
#     ... clean を edge-validator のデータ契約に載せて検証へ ...
# ──────────────────────────────────────────────────────────────
