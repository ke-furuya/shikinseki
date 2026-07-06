#!/usr/bin/env python3
"""
data_harvester.py — 品質ゲート付き汎用データ収集エンジン。

競馬データの収集パイプラインを
競馬から切り離して汎用化したもの。edge-validator の相棒＝「取得→検証」を完結させる上流。

普通のスクレイパとの違い＝**完全性ゲート**：
  生データをそのまま使わせない。"使う前に" 下流（分析）が必要とする項目が揃っているかを
  機械検査し、欠損があれば赤で止める。「ゴミを入れて気づかず分析」を物理的に防ぐ。

3つの責務：
  1. harvest()         … エンドポイントを巡回し、生データを取得（リトライ・部分失敗に頑健）
  2. map_record()      … 生コード→きれいなスキーマに変換（コードマップ・未知コードは "?" で可視化）
  3. completeness_gate() … 下流要件に対して充足検査＋期待リスト突合＋PASS/FAIL判定

依存：標準ライブラリのみ。fetch は差し替え可能（HTTP/ブラウザ貼付/ローカル何でも）。
"""
import json, time, urllib.request, urllib.error
from collections import Counter

__all__ = ["dig", "apply_code_map", "map_record", "http_json", "harvest",
           "completeness_gate"]


# ── スキーマ定義（下流が必要とするもの＝ゲートの基準）──
# {
#   "required":       [...],   欠損/null = 🔴エラー（採点/分析に進ませない）
#   "optional_empty": [...],   空でも可（正当な理由＝古データ・初回・対象外）
#   "code_maps":      {field: {raw: clean, ...}},  生コード→値。未マップは "?"（要マップ更新の合図）
# }

def dig(raw, path):
    """ネストした生データを path（キー列）で辿る。'a.b.0' でも ['a','b',0] でも可。"""
    if isinstance(path, str):
        path = [int(p) if p.isdigit() else p for p in path.split(".")]
    cur = raw
    for p in path:
        try:
            cur = cur[p]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


def apply_code_map(value, code_map):
    """生コードをマップ変換。未知コードは '?'+元値 で可視化（ゲートが検出する）。"""
    if value is None:
        return None
    key = str(value)
    return code_map.get(key, "?" + key)


def map_record(raw, field_spec, code_maps=None):
    """raw → きれいな dict。
    field_spec: list[(out_name, path, code_map_name_or_None)]"""
    code_maps = code_maps or {}
    out = {}
    for item in field_spec:
        out_name, path = item[0], item[1]
        cm = item[2] if len(item) > 2 else None
        val = dig(raw, path)
        if cm and cm in code_maps:
            val = apply_code_map(val, code_maps[cm])
        out[out_name] = val
    return out


# ── 1. 収集ループ（部分失敗に頑健）──
def http_json(url, headers=None, timeout=20):
    # 注意: 信頼できないURLを渡さない（urllibはfile://等も開ける＝SSRF/ローカル読み取りの恐れ）。http(s)限定を推奨。
    """既定の fetch：HTTP GET で JSON を返す。"""
    req = urllib.request.Request(url, headers=headers or {"accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def harvest(entities, fetch_fn, retries=2, pause=0.0, on_progress=None):
    """entities: list[任意のキー（id/パラメータ）]。
    fetch_fn(entity) -> 生データ（dict）。例外時は retries 回まで再試行。
    返り値: {str(entity): 生データ or {'_error': '...'}}。1件失敗しても全体は止まらない。"""
    out = {}
    for i, ent in enumerate(entities):
        last = None
        for attempt in range(retries + 1):
            try:
                out[str(ent)] = fetch_fn(ent)
                last = None
                break
            except Exception as e:  # noqa: BLE001 — 収集は何が来ても止めない
                last = e
                if attempt < retries and pause:
                    time.sleep(pause)
        if last is not None:
            out[str(ent)] = {"_error": f"{type(last).__name__}: {last}"}
        if on_progress:
            on_progress(i + 1, len(entities), str(ent), last is None)
    return out


# ── 3. 完全性ゲート（収集の品質を採点前に保証する）──
def completeness_gate(records, schema, expected_ids=None, label="records"):
    """records: {id: きれいなdict（map_record後）}。schema: 上記の辞書。
    expected_ids: 取れているべきid集合（あれば未収集/余分を突合）。
    欠損があれば PASS=False。下流に進ませないための pre-flight。"""
    required = schema.get("required", [])
    opt_empty = schema.get("optional_empty", [])
    code_fields = list(schema.get("code_maps", {}).keys())

    n = 0
    miss = Counter()
    unknown = Counter()
    n_empty_ok = Counter()
    fetch_err = []

    for rid, r in records.items():
        if isinstance(r, dict) and r.get("_error"):
            fetch_err.append((rid, r["_error"]))
            continue
        n += 1
        for f in required:
            if r.get(f) in (None, ""):
                miss[f] += 1
        for f in opt_empty:
            if r.get(f) in (None, ""):
                n_empty_ok[f] += 1
        for f in code_fields:
            v = r.get(f)
            if isinstance(v, str) and v.startswith("?"):
                unknown[f] += 1

    print(f"収集 {n} {label}（取得失敗 {len(fetch_err)}）")
    if fetch_err:
        for rid, e in fetch_err[:5]:
            print(f"  🔴 取得失敗 {rid}: {e}")
    if opt_empty:
        es = " / ".join(f"{f} 空{n_empty_ok[f]}" for f in opt_empty if n_empty_ok[f])
        if es:
            print(f"  許容空（正当な欠落）: {es}")
    for f, c in unknown.items():
        print(f"  ⚠️ 未知コード '{f}' {c}件（コードマップ要更新）")

    if miss:
        print("  🔴 必須フィールド欠損（下流に進めない）:")
        for f, c in miss.most_common():
            print(f"      {f}: {c}件")
    elif n:
        print("  ✅ 必須フィールド全充足＝抜け漏れなし")

    recon = None
    if expected_ids is not None:
        got = set(k for k, v in records.items() if not (isinstance(v, dict) and v.get("_error")))
        missing = set(map(str, expected_ids)) - got
        extra = got - set(map(str, expected_ids))
        recon = {"missing": missing, "extra": extra}
        print(f"  突合: 期待{len(expected_ids)} / 収集{len(got)} / 未収集{len(missing)} / 余分{len(extra)}")
        if missing:
            print(f"    未収集: {sorted(missing)[:10]}")

    passed = (not miss) and (not fetch_err) and (recon is None or not recon["missing"])
    print("ゲート: " + ("✅PASS（下流GO）" if passed else "⚠️FAIL（欠損を埋めてから使う）"))
    return {"passed": passed, "n": n, "missing": dict(miss),
            "unknown_codes": dict(unknown), "fetch_errors": fetch_err, "recon": recon}


if __name__ == "__main__":
    print(__doc__)
