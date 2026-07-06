#!/usr/bin/env python3
"""感情分析 A/B評価 ＝ 実API・独立モデル版（「評価者≒モデル」の弱点を潰す）。

run_eval.py の分類は対話セッションのLLM自身が行っていた（身内採点＝内部妥当性が弱い）。
本スクリプトは fal-ai/any-llm 経由で独立モデルに 110件×2プロンプト を実際に投げ、
誰でも再現できる predictions を生成 → 試金石(edge-validator)で判定する。

使い方:
  export FAL_KEY=<your-fal-api-key>     # https://fal.ai で取得
  python3 run_eval_realapi.py [model]
    model 既定 = google/gemini-2.5-flash-lite（安価）
    例: openai/gpt-4o-mini / meta-llama/llama-3.2-3b-instruct
出力:  predictions_real_<model>.json ＋ 判定をコンソール出力
コスト目安: 220コール ≒ 数円〜数十円/モデル
"""
import json, os, sys, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # リポジトリルート
from shikinseki import edge_validator as ev

MODEL = sys.argv[1] if len(sys.argv) > 1 else "google/gemini-2.5-flash-lite"

KEY = os.environ.get("FAL_KEY")
if not KEY:
    raise SystemExit("環境変数 FAL_KEY を設定してください（https://fal.ai のAPIキー）")

preds_meta = json.load(open(HERE / "predictions.json"))
PROMPT_A = preds_meta["_prompt_A"]   # 素朴
PROMPT_B = preds_meta["_prompt_B"]   # 工夫
testset = json.load(open(HERE / "testset.json"))

def call(system_prompt, text):
    body = json.dumps({"model": MODEL, "system_prompt": "",
                       "prompt": system_prompt.format(text=text)}).encode()
    req = urllib.request.Request("https://fal.run/fal-ai/any-llm", data=body,
        headers={"Authorization": f"Key {KEY}", "Content-Type": "application/json"})
    for _ in range(3):
        try:
            out = json.load(urllib.request.urlopen(req, timeout=90)).get("output", "")
            s = (out or "").lower()
            if "neg" in s: return "neg"
            if "pos" in s: return "pos"
            return "unk"
        except Exception:
            continue
    return "err"

def classify_item(item):
    a = call(PROMPT_A, item["text"])
    b = call(PROMPT_B, item["text"])
    return str(item["id"]), {"A": a, "B": b}

print(f"分類中… model={MODEL}  N={len(testset)}件 ×2プロンプト = {len(testset)*2}コール")
with ThreadPoolExecutor(max_workers=8) as ex:
    preds = dict(ex.map(classify_item, testset))

outpath = HERE / f"predictions_real_{MODEL.replace('/','_')}.json"
json.dump({"_model": MODEL, "_prompt_A": PROMPT_A, "_prompt_B": PROMPT_B, **preds},
          open(outpath, "w"), ensure_ascii=False, indent=1)
print(f"保存: {outpath.name}")

# ── 試金石で判定（run_eval.py と同じロジックの核）──
A_rec, B_rec, diff_rec = [], [], []
errs = 0
for item in testset:
    rid = str(item["id"]); lab = item["label"]
    p = preds[rid]
    if p["A"] in ("err", "unk") or p["B"] in ("err", "unk"):
        errs += 1
    ac = 1 if p["A"] == lab else 0
    bc = 1 if p["B"] == lab else 0
    A_rec.append({"group": rid, "value": ac})
    B_rec.append({"group": rid, "value": bc})
    diff_rec.append({"group": rid, "value": bc - ac})

N = len(A_rec)
accA, accB, gap = ev.mean_ci(A_rec), ev.mean_ci(B_rec), ev.mean_ci(diff_rec)
print("=" * 64)
print(f"実API・独立モデル判定  model={MODEL}  N={N}  (分類失敗 {errs}件)")
print("=" * 64)
print(f"A(素朴) {accA['mean']*100:5.1f}%  90%CI {accA['ci_low']*100:.1f}〜{accA['ci_high']*100:.1f}%")
print(f"B(工夫) {accB['mean']*100:5.1f}%  90%CI {accB['ci_low']*100:.1f}〜{accB['ci_high']*100:.1f}%")
g, lo, hi = gap["mean"]*100, gap["ci_low"]*100, gap["ci_high"]*100
print(f"B−A     {g:+.1f}pp   90%CI {lo:+.1f}〜{hi:+.1f}pp")
if gap["robust_positive"]:
    print("判定: ✅ Bが頑健に良い（90%CIが0を除外・正側）")
elif gap["robust_negative"]:
    print("判定: ❌ Bが頑健に悪い（90%CIが0を除外・負側）＝工夫プロンプトが逆効果")
else:
    print("判定: ⚠️ まぐれと区別不能（90%CIが0をまたぐ）")
pw = ev.power_required_mean([r["value"] for r in diff_rec], mde=0.05)
print(f"検出力: 5pp差の証明に要N≈{pw['need_n']:.0f}（現在{N}）")
