#!/usr/bin/env python3
"""
example_leak.py — leak_guard を、わざとリークを仕込んだ合成データで実証。
「再登場リーク・境界またぎ・ターゲットリーク・重複」を全部捕まえることを証明する。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # リポジトリルート
from shikinseki import leak_guard as lg

# データ契約: group(イベント) / time / entity / baseline(小=強) / outcome / 特徴量
# 仕込んだリーク：
#  ① 古いレース g_old(time=1) の本命 horseA が、後の time=5 で level=3(昇格) に再登場 → H のはず
#  ② g_span が time=2 と time=4 に跨る → 境界またぎ(split=3)
#  ③ feature "ans" は outcome と一致 → ターゲットリーク
#  ④ (horseZ, 9) が重複
records = [
    # g_old（古い・採点対象）: 本命A
    {"group": "g_old", "time": 1, "entity": "horseA", "baseline": 1.5, "level": 1, "outcome": 1, "ans": 1, "pad": 0},
    {"group": "g_old", "time": 1, "entity": "horseB", "baseline": 4.0, "level": 1, "outcome": 0, "ans": 0, "pad": 1},
    {"group": "g_old", "time": 1, "entity": "horseC", "baseline": 9.0, "level": 1, "outcome": 0, "ans": 0, "pad": 0},
    # 後の時点で horseA が昇格(level 1→3)再登場 → g_old の結果が逆算可能
    {"group": "g_new", "time": 5, "entity": "horseA", "baseline": 2.0, "level": 3, "outcome": 1, "ans": 1, "pad": 1},
    # 境界またぎ g_span（split_time=3 の前後に出る）
    {"group": "g_span", "time": 2, "entity": "horseX", "baseline": 2.0, "level": 2, "outcome": 1, "ans": 1, "pad": 0},
    {"group": "g_span", "time": 4, "entity": "horseY", "baseline": 3.0, "level": 2, "outcome": 0, "ans": 0, "pad": 1},
    # 重複
    {"group": "g_dup", "time": 9, "entity": "horseZ", "baseline": 1.0, "level": 1, "outcome": 0, "ans": 0, "pad": 0},
    {"group": "g_dup", "time": 9, "entity": "horseZ", "baseline": 1.0, "level": 1, "outcome": 0, "ans": 0, "pad": 0},
]
# 分離度検査に最低件数が要るので、ノイズ行を足す（ans=outcome一致は維持、padは無情報）
import itertools
for i in range(40):
    o = i % 2
    records.append({"group": f"g{i}", "time": 6, "entity": f"h{i}", "baseline": 3.0,
                    "level": 1, "outcome": o, "ans": o, "pad": (i % 3 == 0) and 1 or 0})

if __name__ == "__main__":
    res = lg.run_all(records, level_key="level", split_time=3, feature_keys=["ans", "pad"])

    rf = res["reappearance"]
    assert rf["g_old"] == "H", f'本命の昇格再登場はHのはず: {rf["g_old"]}'
    assert "g_span" in lg.split_contamination(records, 3)
    tl = [t["feature"] for t in lg.target_leak_scan(records, ["ans", "pad"])]
    assert "ans" in tl and "pad" not in tl, f'ansだけリーク判定のはず: {tl}'
    assert lg.duplicate_check(records)[0]["entity"] == "horseZ"
    assert res["clean"] is False

    print("\n✅ 実証完了：再登場リーク(H)・境界またぎ・ターゲットリーク(ans)・重複(horseZ)を")
    print("   全て捕捉。pad(無情報)は誤検出せず。これが『汚染データを検証に流さない』防御層。")
