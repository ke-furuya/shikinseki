#!/usr/bin/env python3
"""BGL実ログ障害トリアージ＝プロンプトA/B評価（試金石の第2実ドメイン適用）。

問い：「SRE風の工夫プロンプトBは、素朴なAより本当に良いのか？ そして単純な
キーワードルール（FATAL→alert・88%）に、LLMはそもそも勝てるのか？」

使い方:
  export FAL_KEY=<your-fal-api-key>        # https://fal.ai で取得
  python3 run_eval_logtriage.py [model]    # 201行×2プロンプトを実測（約400コール/モデル）
  # predictions_real_<model>.json が既にあればAPIを呼ばず判定だけ再実行（再現用・APIキー不要）

判定は edge-validator。group=EventId（同一ログテンプレの行は相関する＝1行1群にしない。
これを怠ると重複だらけのログで母数を水増しして「有意」を捏造できてしまう）。
"""
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from shikinseki import edge_validator as ev

MODEL = sys.argv[1] if len(sys.argv) > 1 else "google/gemini-2.5-flash-lite"
OUTPATH = HERE / f"predictions_real_{MODEL.replace('/', '_')}.json"

# ── プロンプト（Bは結果を見る前に1回だけ書いた＝チューニングループなし。READMEに明記）──
PROMPT_A = (
    "You are triaging supercomputer system logs. Is the following log line an actionable "
    "alert (a failure event an operator must attend to) or normal? "
    "Answer with exactly one word: alert or normal.\n\nLog line: {text}"
)
PROMPT_B = (
    "You are an experienced SRE triaging RAS logs from a BlueGene/L supercomputer.\n"
    "Decide whether the following single log line is an actionable ALERT or NORMAL noise.\n"
    "Triage principles:\n"
    "1. Judge whether the line itself reports a failure EVENT an operator must act on.\n"
    "2. Severity FATAL does not automatically mean alert: register dumps, memory addresses "
    "and state printouts that accompany a crash are diagnostic detail, not actionable events.\n"
    "3. Hardware error interrupts, kernel panics, and I/O failures on control streams are "
    "actionable; state dumps mentioning scary words are not.\n"
    "Answer with exactly one word: alert or normal.\n\nLog line: {text}"
)

testset = json.load(open(HERE / "testset.json"))


def rule_baseline(item):
    """超えるべき相手＝素朴なキーワードルール（severity FATAL → alert）。全体88%精度。"""
    return "alert" if " FATAL " in item["text"] else "normal"


def parse(out):
    s = (out or "").lower()
    ia, im = s.find("alert"), s.find("normal")
    if ia < 0 and im < 0:
        return "unk"
    if ia < 0:
        return "normal"
    if im < 0:
        return "alert"
    return "alert" if ia < im else "normal"


def measure():
    key = os.environ.get("FAL_KEY")
    if not key:
        raise SystemExit("環境変数 FAL_KEY を設定してください（https://fal.ai のAPIキー）")

    def call(prompt_template, text):
        body = json.dumps({"model": MODEL, "system_prompt": "",
                           "prompt": prompt_template.format(text=text)}).encode()
        req = urllib.request.Request("https://fal.run/fal-ai/any-llm", data=body,
            headers={"Authorization": f"Key {key}", "Content-Type": "application/json"})
        last = None
        for _ in range(3):
            try:
                out = json.load(urllib.request.urlopen(req, timeout=90)).get("output", "")
                return parse(out)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise SystemExit(f"認証エラー HTTP {e.code}: FAL_KEY を確認してください")
                last = e
            except Exception as e:  # noqa: BLE001 — 一時障害は再試行
                last = e
        print(f"  ⚠️ APIコール3回失敗: {type(last).__name__}: {last}", file=sys.stderr)
        return "err"

    def classify_item(item):
        return str(item["id"]), {"A": call(PROMPT_A, item["text"]),
                                 "B": call(PROMPT_B, item["text"])}

    print(f"分類中… model={MODEL}  N={len(testset)}行 ×2プロンプト = {len(testset)*2}コール")
    with ThreadPoolExecutor(max_workers=8) as ex:
        preds = dict(ex.map(classify_item, testset))
    json.dump({"_model": MODEL, "_prompt_A": PROMPT_A, "_prompt_B": PROMPT_B, **preds},
              open(OUTPATH, "w"), ensure_ascii=False, indent=1)
    print(f"保存: {OUTPATH.name}")
    return preds


def judge(preds):
    from collections import defaultdict
    A_rec, B_rec, diff_rec, vs_rule = [], [], [], []
    strata = defaultdict(list)
    errs = 0
    for item in testset:
        p = preds[str(item["id"])]
        if p["A"] in ("err", "unk") or p["B"] in ("err", "unk"):
            errs += 1
        lab, g = item["label"], item["group"]   # group=EventId＝同一テンプレは1群
        ac, bc = 1 if p["A"] == lab else 0, 1 if p["B"] == lab else 0
        rc = 1 if rule_baseline(item) == lab else 0
        A_rec.append({"group": g, "value": ac})
        B_rec.append({"group": g, "value": bc})
        diff_rec.append({"group": g, "value": bc - ac})
        vs_rule.append({"group": g, "value": bc - rc})
        strata[f"kind={item['kind']}"].append({"group": g, "value": bc - ac})

    N = len(testset)
    accA, accB, gap = ev.mean_ci(A_rec), ev.mean_ci(B_rec), ev.mean_ci(diff_rec)
    rule_acc = sum(1 for i in testset if rule_baseline(i) == i["label"]) / N
    beat = ev.mean_ci(vs_rule)

    print("=" * 68)
    print(f"BGL障害トリアージ 実API判定  model={MODEL}  N={N}行 (分類失敗 {errs}件)")
    print(f"group=ログテンプレ {gap['n_groups']}種（1行1群にしない＝重複行で母数を水増ししない）")
    print("=" * 68)
    print(f"キーワードルール(FATAL→alert)  正解率 {rule_acc*100:5.1f}%  ← 超えるべき相手")
    print(f"プロンプトA(素朴)              正解率 {accA['mean']*100:5.1f}%  "
          f"90%CI {accA['ci_low']*100:.1f}〜{accA['ci_high']*100:.1f}%")
    print(f"プロンプトB(SRE風の工夫)       正解率 {accB['mean']*100:5.1f}%  "
          f"90%CI {accB['ci_low']*100:.1f}〜{accB['ci_high']*100:.1f}%")
    print("-" * 68)
    g, lo, hi = gap["mean"] * 100, gap["ci_low"] * 100, gap["ci_high"] * 100
    print(f"B−A の差          {g:+.1f}pp   90%CI {lo:+.1f}〜{hi:+.1f}pp")
    verdict = ("✅ Bが頑健に良い" if gap["robust_positive"]
               else "❌ Bが頑健に悪い（工夫が逆効果）" if gap["robust_negative"]
               else "⚠️ まぐれと区別できない")
    print(f"判定: {verdict}（90%CI・group単位）")
    bg, bl, bh = beat["mean"] * 100, beat["ci_low"] * 100, beat["ci_high"] * 100
    rv = ("✅ LLMがルールに頑健に勝つ" if beat["robust_positive"]
          else "❌ ルールに頑健に負ける" if beat["robust_negative"]
          else "⚠️ ルールとの差はまぐれと区別できない")
    print(f"B−ルール          {bg:+.1f}pp   90%CI {bl:+.1f}〜{bh:+.1f}pp → {rv}")

    pw = ev.power_required_mean([r["value"] for r in diff_rec], mde=0.05)
    if pw:
        print(f"\n検出力: 事前指定MDE=5ppの証明に要N≈{pw['need_n']:.0f}（現在{N}・実効group{gap['n_groups']}）")

    print(f"\n[層別 B−A（効果はどこに居るか）]")
    for key in sorted(strata):
        s = ev.mean_ci(strata[key])
        note = "" if s["reliable"] else f"（group{s['n_groups']}<30＝判定保留）"
        flag = "✅" if s["robust_positive"] else ("❌" if s["robust_negative"] else "・")
        print(f"  {key:<18} n={s['n']:<3} B−A {s['mean']*100:+5.1f}pp"
              f"  90%CI {s['ci_low']*100:+.1f}〜{s['ci_high']*100:+.1f}pp {flag}{note}")


if __name__ == "__main__":
    if OUTPATH.exists():
        print(f"（{OUTPATH.name} を再判定＝APIコールなし。測り直すにはファイルを削除）")
        saved = json.load(open(OUTPATH))
        judge({k: v for k, v in saved.items() if not k.startswith("_")})
    else:
        judge(measure())
