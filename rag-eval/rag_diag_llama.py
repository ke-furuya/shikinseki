#!/usr/bin/env python3
"""Llama が judge で空応答だった原因を切り分ける診断。

3つのテストで「llama自体は生きてるが長いjudgeプロンプトで落ちる」のか
「fal側でllamaが応答しない」のかを見る。call_llm と違い raw レスポンス全体を表示。
  export FAL_KEY=<key>
  python3 rag_diag_llama.py
"""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://fal.run/fal-ai/any-llm"
LLAMA = "meta-llama/llama-3.2-3b-instruct"
RESULT_PATH = Path(__file__).parent / "diag_llama_result.txt"

_LOG = []


def log(msg=""):
    print(msg)
    _LOG.append(str(msg))


def raw_call(model, prompt):
    """生レスポンス(JSON全体)とHTTP状態を返す。retryなし・エラーも握りつぶさない。"""
    key = os.environ["FAL_KEY"]
    body = json.dumps({"model": model, "system_prompt": "", "prompt": prompt}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
        headers={"Authorization": f"Key {key}", "Content-Type": "application/json"})
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=120))
        out = resp.get("output", "")
        log(f"    HTTP 200  output長={len(out)}文字  他キー={[k for k in resp if k!='output']}")
        log(f"    output(先頭300): {out[:300]!r}")
        return out
    except urllib.error.HTTPError as e:
        log(f"    HTTP {e.code}: {e.read().decode()[:300]}")
    except Exception as e:  # noqa: BLE001
        log(f"    例外: {type(e).__name__}: {e}")
    return None


SHORT = 'You are a judge. Reply with ONLY this JSON and nothing else: {"score": 2}'

LONG = """You are a strict evaluator. Score the ANSWER against the GOLD answer.
Use this rubric, each 0, 1, or 2: faithfulness, citation, coverage.
Return ONLY a JSON object: {"faithfulness": <0-2>, "citation": <0-2>, "coverage": <0-2>, "reason": "<one sentence>"}
QUESTION: What triggered the 2017 Amazon S3 outage?
GOLD ANSWER: A command run with a typo removed far more servers than intended.
ANSWER TO SCORE: The outage was caused by a typo in a command that removed too many servers [3].
JSON:"""


def main():
    if not os.environ.get("FAL_KEY"):
        raise SystemExit("FAL_KEY を設定してください")
    log("【テスト1】Llama に “超短いJSON” を頼む（llama自体が生きてるか）")
    raw_call(LLAMA, SHORT)
    log("\n【テスト2】Llama に “実際のjudgeプロンプト(長い+JSON)” を投げる")
    raw_call(LLAMA, LONG)
    log("\n【テスト3・比較】同じ長いプロンプトを Gemini に（プロンプト自体は正常か）")
    raw_call("google/gemini-2.5-flash-lite", LONG)
    log("\n→ T1成功&T2空 なら『llamaは生きてるが長い採点+JSONに落ちる＝モデルの能力』")
    log("  T1も空/エラーなら『fal側でllamaが応答しない＝API/モデル可用性』")
    RESULT_PATH.write_text("\n".join(_LOG))
    print(f"\n（結果を {RESULT_PATH.name} に保存した）")


if __name__ == "__main__":
    main()
