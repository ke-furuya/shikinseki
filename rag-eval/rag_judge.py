#!/usr/bin/env python3
"""LLM-as-judge 採点（Round2仕様・並列化）。

  python3 rag_judge.py <gen_alias> <judge_alias> [--norag]
入力: answers_<gen>{_norag}.json → 出力: judged_<gen>{_norag}_by_<judge>.json
rubric（3軸 0-2）は Round1 と同一＝ラウンド間比較のため変更しない。
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fal_llm import MODELS, call_llm

HERE = Path(__file__).parent

JUDGE_PROMPT = """You are a strict evaluator of a retrieval-augmented answer. Score the ANSWER against the GOLD answer and the retrieved evidence.
Use this rubric, each 0, 1, or 2:
- faithfulness: 2 = no claim contradicts the gold/evidence; 1 = minor unsupported detail; 0 = a clear factual error or hallucination. For a question with no answer in the documents, faithfulness=2 ONLY if the answer says the documents do not contain the information (does not fabricate).
- citation: 2 = cites the correct supporting passage numbers [n]; 1 = cites but partially wrong/missing; 0 = no or wrong citations.
- coverage: 2 = captures the key points of the gold answer; 1 = partial; 0 = misses the point.

Return ONLY a JSON object: {{"faithfulness": <0-2>, "citation": <0-2>, "coverage": <0-2>, "reason": "<one sentence>"}}

QUESTION: {question}
GOLD ANSWER: {gold}
RETRIEVED DOC IDS: {retrieved}
ANSWER TO SCORE: {answer}

JSON:"""


def parse_json(text: str) -> dict:
    """LLM出力から最初のJSONオブジェクトを取り出す（```で囲まれても拾う）。"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"faithfulness": None, "citation": None, "coverage": None, "reason": "parse_fail", "_raw": text[:200]}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"faithfulness": None, "citation": None, "coverage": None, "reason": "parse_fail", "_raw": text[:200]}
    for kk in ("faithfulness", "citation", "coverage"):
        obj.setdefault(kk, None)
    return obj


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: rag_judge.py <gen_alias> <judge_alias> [--norag]")
    norag = "--norag" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    gen_alias, judge_alias = pos[0], pos[1]
    judge_model = MODELS[judge_alias]
    suffix = "_norag" if norag else ""

    data = json.loads((HERE / f"answers_{gen_alias}{suffix}.json").read_text())
    answers = data["answers"]
    print(f"judge={judge_model} が gen={gen_alias}{suffix} の回答 {len(answers)}問 を採点")

    def score(a):
        out = call_llm(judge_model, JUDGE_PROMPT.format(
            question=a["question"], gold=a["gold_answer"],
            retrieved=a["retrieved_doc_ids"] or "none (no-retrieval baseline)",
            answer=a["answer"] or "(empty)"))
        verdict = parse_json(out)
        print(f"  {a['id']} f={verdict.get('faithfulness')} c={verdict.get('citation')} cov={verdict.get('coverage')}")
        return {"id": a["id"], "type": a["type"], **{
            k: verdict.get(k) for k in ("faithfulness", "citation", "coverage", "reason")}}

    with ThreadPoolExecutor(max_workers=8) as ex:
        scored = list(ex.map(score, answers))

    out_path = HERE / f"judged_{gen_alias}{suffix}_by_{judge_alias}.json"
    out_path.write_text(json.dumps({"_gen": gen_alias, "_mode": suffix or "rag",
                                    "_judge": judge_model, "scores": scored},
                                   ensure_ascii=False, indent=2))
    for axis in ("faithfulness", "citation", "coverage"):
        vals = [s[axis] for s in scored if isinstance(s[axis], (int, float))]
        if vals:
            print(f"  mean {axis} = {sum(vals)/len(vals):.2f}  (n={len(vals)})")
    print(f"→ {out_path.name} に保存")


if __name__ == "__main__":
    main()
