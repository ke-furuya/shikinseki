#!/usr/bin/env python3
"""BGL実ログから障害トリアージ評価用 testset.json を決定論生成する。

データ＝LogHub (logpai/loghub) の BGL_2k サンプル。BlueGene/L スーパーコンピュータの
実RASログで、各行に「アラート種別タグ or '-'（非アラート）」のラベルが付いている。
出典: Oliner & Stearley "What Supercomputers Say" (DSN'07) / LogHub (Zhu et al., ISSRE'23)。

タスク＝「この1行は対応が必要なアラートか、それとも周辺ノイズか」。
面白いのは **normal側にもFATALが204行ある** こと（クラッシュ周辺のレジスタダンプ等）。
「FATALという単語に飛びつく」キーワードルールは88%止まり＝自明でない実タスク。

試金石の4層をそのまま通す：
  ①取得: LogHubから取得→completeness_gateで欠損検査
  ②防御: 行頭のラベルトークンを残すと target_leak_scan が分離度1.0で検出（＝答えの混入）
         →本文から必ず剥がす。同一テンプレの行は相関する→group=EventIdで束ねる
  ③検証: run_eval_logtriage.py（edge-validatorで判定）
実行: python3 prepare_testset.py  （testset.json を再生成。コミット済みなので通常は不要）
"""
import csv
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from shikinseki import data_harvester as dh
from shikinseki import leak_guard as lg

RAW_URL = "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log_structured.csv"
SEED = 42
CAP_ALERT, CAP_HARD, CAP_EASY = 5, 3, 2   # テンプレあたり上限（同一テンプレの重複で母数を水増ししない）
N_HARD_MAX, N_EASY_MAX = 90, 60


def fetch_rows():
    """LogHubの構造化CSVを取得。①取得層＝completeness_gateで使う前に検査。"""
    raw = urllib.request.urlopen(RAW_URL, timeout=60).read().decode("utf-8", errors="replace")
    rows = list(csv.DictReader(io.StringIO(raw)))
    records = {r["LineId"]: r for r in rows}
    gate = dh.completeness_gate(
        records, {"required": ["Label", "Level", "Content", "EventId"], "optional_empty": []},
        label="BGL lines")
    assert gate["passed"], "取得データが不完全＝ここで止める（下流に進ませない）"
    return rows


def build(rows):
    rng = random.Random(SEED)
    # 層別プール（kind）: alert / hard_negative(normalなのにFATAL/ERROR) / easy_negative
    pools = {"alert": [], "hard_negative": [], "easy_negative": []}
    for r in rows:
        is_alert = r["Label"] != "-"
        kind = ("alert" if is_alert
                else "hard_negative" if r["Level"] in ("FATAL", "ERROR")
                else "easy_negative")
        pools[kind].append(r)

    def sample(pool, cap_per_template, n_max=None):
        by_ev = {}
        for r in pool:
            by_ev.setdefault(r["EventId"], []).append(r)
        picked = []
        for ev in sorted(by_ev):  # 決定論：EventId順
            rs = by_ev[ev]
            rng.shuffle(rs)
            picked.extend(rs[:cap_per_template])
        rng.shuffle(picked)
        return picked[:n_max] if n_max else picked

    chosen = (sample(pools["alert"], CAP_ALERT)
              + sample(pools["hard_negative"], CAP_HARD, N_HARD_MAX)
              + sample(pools["easy_negative"], CAP_EASY, N_EASY_MAX))
    rng.shuffle(chosen)

    # ②防御の実演：ラベルトークン由来の特徴を残すと「答えの混入」として検出されることを機械で示す
    leak_records = [{"entity": r["LineId"], "time": r["LineId"],
                     "outcome": 1 if r["Label"] != "-" else 0,
                     "has_label_token": 1 if r["Label"] != "-" else 0,   # ←行頭タグをそのまま特徴化
                     "level_is_fatal": 1 if r["Level"] == "FATAL" else 0}
                    for r in chosen]
    flags = lg.target_leak_scan(leak_records, ["has_label_token", "level_is_fatal"])
    flagged = {f["feature"] for f in flags}
    assert "has_label_token" in flagged, "ラベルトークンはtarget_leak_scanが検出するはず"
    assert "level_is_fatal" not in flagged, "severityは正当な入力（分離不完全）＝誤検出しないはず"
    print(f"リーク検査: has_label_token→検出(分離度{[f['separation'] for f in flags if f['feature']=='has_label_token'][0]})"
          f" / level_is_fatal→非検出 ＝ 行頭タグだけを剥がせばよい")

    testset = []
    for r in chosen:
        # 本文＝ラベルトークン以外を再構成（severity FATAL等は運用者も見る正当な入力なので残す）
        text = " ".join([r["Timestamp"], r["Date"], r["Node"], r["Time"], r["NodeRepeat"],
                         r["Type"], r["Component"], r["Level"], r["Content"]])
        testset.append({
            "id": int(r["LineId"]),
            "text": text,
            "label": "alert" if r["Label"] != "-" else "normal",
            "group": r["EventId"],   # 同一テンプレは相関＝再標本化の単位（1行=1群にしない）
            "kind": ("alert" if r["Label"] != "-"
                     else "hard_negative" if r["Level"] in ("FATAL", "ERROR")
                     else "easy_negative"),
        })

    n_groups = len(set(t["group"] for t in testset))
    n_alert = sum(1 for t in testset if t["label"] == "alert")
    print(f"testset: {len(testset)}行 / alert {n_alert} / group(テンプレ) {n_groups}")
    json.dump(testset, open(HERE / "testset.json", "w"), ensure_ascii=False, indent=1)
    print(f"保存: {HERE / 'testset.json'}")


if __name__ == "__main__":
    build(fetch_rows())
