#!/usr/bin/env python3
"""フロンティア級judgeの追加実験（Round3・任意）。

仮説（Round2の発見から）＝「judge間の不一致は弱いjudgeに集中する。
なら強いjudge同士はさらに一致するはずで、一致しなければrubricの問題」。
これを1-2個の強モデルjudgeで検証する。

やること:
  1. STRONG_MODELS 候補を1コールずつprobe（応答しないモデルはerror本文を表示
     ＝fal側の許可モデル一覧が載ることが多く、カタログ確定にも使える）
  2. 応答した強モデル（最大2つ）で RAG回答30問を採点
  3. rag_stats.py で全judge総当たり集計

  export FAL_KEY=<key>
  python3 run_strong_judges.py
コール数目安: probe 4 + 採点 30×(生存数≤2) ≈ 70
"""
import subprocess
import sys
from pathlib import Path

from fal_llm import MODELS, STRONG_MODELS, call_llm_raw

HERE = Path(__file__).parent


def main():
    print("== フロンティアjudge候補のprobe ==")
    alive = []
    for alias, model in STRONG_MODELS.items():
        resp = call_llm_raw(model, 'Reply with ONLY this JSON: {"ok": 1}')
        out = resp.get("output", "").strip()
        if out:
            print(f"  ✅ {alias} ({model}): 応答あり")
            alive.append(alias)
        else:
            err = str(resp.get("error", ""))[:400]
            print(f"  ❌ {alias} ({model}): 空応答  error={err or '(理由なし)'}")
    if not alive:
        raise SystemExit("強モデル候補が全滅。error本文の許可モデル一覧からIDを直して再実行。")

    use = alive[:2]
    print(f"\n→ 採点に使う強judge: {use}")
    for j in use:
        r = subprocess.run([sys.executable, "rag_judge.py", "gemini", j], cwd=HERE)
        if r.returncode != 0:
            raise SystemExit(f"採点失敗: {j}")

    subprocess.run([sys.executable, "rag_stats.py"], cwd=HERE)
    print("\n== 強judge実験 完了（rag_statsのJUDGE_ALIASESに追加済みなら総当たりに反映） ==")


if __name__ == "__main__":
    main()
