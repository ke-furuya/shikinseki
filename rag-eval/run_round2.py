#!/usr/bin/env python3
"""Round2 一括実行（けんてぃがターミナルでこれ1本を叩くだけ）。

やること（順に・失敗したら止まる）:
  0. llama系judgeのprobe（3.2-3b→3.1-8b→3.2-1bの順で応答するものを自動選択。errorも表示）
  1. RAG回答30問を再生成（KB事実修正＋doc-dedup検索を反映）
  2. no-RAG対照30問を生成
  3. judge採点: gpt4o-mini / gemini / (生きているllama系) × RAG/no-RAG
  4. 機械citationチェック → 統計集計 → 出荷verifyゲート

  export FAL_KEY=<key>
  python3 run_round2.py
コール数目安: 60生成 + 最大180採点 ≈ 240（fal any-llm・数分）
"""
import subprocess
import sys
from pathlib import Path

from fal_llm import pick_llama_judge

HERE = Path(__file__).parent


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run([sys.executable] + cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"失敗: {' '.join(cmd)} (exit {r.returncode})")


def main():
    print("== 0. llama系judgeのprobe ==")
    llama_alias, err = pick_llama_judge()
    if llama_alias:
        print(f"  → judge3に採用: {llama_alias}")
    else:
        print(f"  → llama系は全滅（error: {err or '空応答・理由不明'}）＝judge2つで続行（結果に明記）")

    judges = ["gpt4o-mini", "gemini"] + ([llama_alias] if llama_alias else [])

    run(["rag_answer.py", "gemini"])
    run(["rag_answer.py", "gemini", "--norag"])
    for j in judges:
        run(["rag_judge.py", "gemini", j])
        run(["rag_judge.py", "gemini", j, "--norag"])

    run(["rag_citation_check.py", "answers_gemini.json"])
    run(["rag_stats.py"])
    run(["_verify_publish.py"])
    print("\n== Round2 完了 ==")


if __name__ == "__main__":
    main()
