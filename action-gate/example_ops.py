#!/usr/bin/env python3
"""
example_ops.py — action_gate を、決定論生成＋機械ゲートで実証。
「手組みなら起きる買い忘れ・重複・予算超過・必須欠落を、構造で防ぐ」ことを証明する。

題材は汎用の「アクション生成」（競馬の買い目でも、インフラの是正アクションでも同型）。
移植は policy（ルール）と invariants（不変条件）を書き換えるだけ。
"""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # リポジトリルート
from shikinseki import action_gate as ag

# 状態（state）: 1イベント分の入力。◎○▲＋pool＋オッズ＋予算（汎用の例）
STATE = {
    "maru": "A", "o": "B", "ba": "C", "pool": ["D", "E"],
    "odds": {"A": 2.0, "B": 4.0, "C": 6.0, "D": 9.0, "E": 12.0},
    "budget": 500, "unit": 100,
}

# 方針（policy）: 検証済みルールを「条件→アクション」で宣言。手で買い目を組まない。
POLICY = [
    ("軸-対抗",   lambda s: True,                    lambda s: {"type": "連", "legs": sorted([s["maru"], s["o"]]), "cost": s["unit"]}),
    ("軸-3番手",  lambda s: True,                    lambda s: {"type": "連", "legs": sorted([s["maru"], s["ba"]]), "cost": s["unit"]}),
    ("軸-pool流し", lambda s: len(s["pool"]) > 0,      lambda s: [{"type": "連", "legs": sorted([s["maru"], p]), "cost": s["unit"]} for p in s["pool"]]),
]

# 機械ゲート（invariants）: 実行前の不変条件。
INVARIANTS = [
    ag.inv_no_duplicate(),
    ag.inv_required_fields(["type", "legs", "cost"]),
    ag.inv_budget("cost", STATE["budget"]),
    # 期待件数 = 1(対抗) + 1(3番手) + pool数。買い忘れ/出し過ぎを検出
    ag.inv_expected_count(lambda s: 2 + len(s["pool"])),
]

if __name__ == "__main__":
    print("="*60 + "\n① 正常: 決定論生成＋ゲート通過 → commit\n" + "="*60)
    res = ag.run(POLICY, STATE, INVARIANTS)
    assert res["committed"] is True
    assert len(res["actions"]) == 4   # 対抗 + 3番手 + pool2 = 4点

    print("\n" + "="*60 + "\n② 決定論の証明: ルール順をシャッフルしても出力は不変\n" + "="*60)
    shuffled = POLICY[:]
    random.Random(1).shuffle(shuffled)
    a1, _ = ag.generate(POLICY, STATE)
    a2, _ = ag.generate(shuffled, STATE)
    assert a1 == a2
    print(f"  ルール順を変えても生成アクションは完全一致（{len(a1)}件）＝再現可能")

    print("\n" + "="*60 + "\n③ 予算超過: poolが増えて500円を超える → ゲートがブロック\n" + "="*60)
    big = dict(STATE, pool=["D", "E", "F", "G", "H"],
               odds=dict(STATE["odds"], F=20, G=30, H=40))
    res3 = ag.run(POLICY, big, INVARIANTS)
    assert res3["committed"] is False
    assert res3["actions"] == []   # ブロック時は実行可アクションを返さない

    print("\n" + "="*60 + "\n④ 買い忘れの構造的検出: poolを1つ手で落とした不正ポリシー → 件数不一致で止まる\n" + "="*60)
    bad_policy = POLICY[:2] + [   # pool流しルールを「最初の1頭だけ」に改悪（=人為的買い忘れ）
        ("軸-pool流し(壊)", lambda s: True, lambda s: {"type": "連", "legs": sorted([s["maru"], s["pool"][0]]), "cost": s["unit"]})
    ]
    res4 = ag.run(bad_policy, STATE, INVARIANTS)
    assert res4["committed"] is False   # 期待4件に対し3件 → ブロック

    print("\n✅ 実証完了：決定論生成（順序非依存・再現可能）＋機械ゲートが、")
    print("   予算超過・買い忘れ（件数不一致）を実行前に物理的にブロック。手組みの事故を構造で防ぐ。")


# 参考：インフラ運用への移植イメージ（同じエンジン・policyとinvariantsだけ差し替え）
#   state = 現在のメトリクス/アラート状況
#   policy = [("CPU高→スケールアウト", cond, act), ("ディスク逼迫→クリーンアップ", cond, act), ...]
#   invariants = [inv_no_duplicate(), inv_budget("instances", MAX), inv_not_contradict("target", [("scale_up","scale_down")])]
#   → 是正アクションを決定論生成し、矛盾/上限超過を実行前にブロック（runbook-as-code）
