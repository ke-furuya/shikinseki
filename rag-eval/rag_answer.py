#!/usr/bin/env python3
"""RAG 回答生成（Round2仕様）。

RAGモード: doc重複排除検索(top5の異なるdoc全文)を文脈に、根拠[n]付きで回答。
no-RAGモード(--norag): 文脈なしで同じ質問に回答＝RAGの効果を測る対照群。

  export FAL_KEY=<fal.ai key>
  python3 rag_answer.py [model_alias] [--norag]   # default: gemini
出力: answers_<alias>.json / answers_<alias>_norag.json
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fal_llm import MODELS, call_llm
from rag_min import BM25, chunk_documents, load_jsonl, KB_PATH, QA_PATH

HERE = Path(__file__).parent
TOP_K = 5

ANSWER_PROMPT = """You are an incident-analysis assistant. Answer the question using ONLY the numbered context passages below. \
Cite the passages you rely on with their numbers in square brackets, e.g. [1]. \
If the context does not contain the answer, reply exactly: "The provided documents do not contain this information." \
Do not use outside knowledge.

Context:
{context}

Question: {question}

Answer (grounded, with [n] citations):"""

NORAG_PROMPT = """You are an incident-analysis assistant. Answer the question from your own knowledge, concisely. \
If you do not know or are not sure, reply exactly: "The provided documents do not contain this information."

Question: {question}

Answer:"""


def build_context(results) -> str:
    return "\n".join(f"[{i}] ({d['id']}) {d['title']}. {d['text']}"
                     for i, (d, _s) in enumerate(results, start=1))


def main():
    args = [a for a in sys.argv[1:]]
    norag = "--norag" in args
    alias = next((a for a in args if not a.startswith("--")), "gemini")
    model = MODELS[alias]
    docs = load_jsonl(KB_PATH)
    qa = load_jsonl(QA_PATH)
    bm25 = BM25(chunk_documents(docs))
    mode = "no-RAG対照" if norag else f"RAG(doc-dedup top{TOP_K})"
    print(f"回答生成 model={model} ({alias})  N={len(qa)}問  mode={mode}")

    def gen(q):
        if norag:
            retrieved_ids = []
            prompt = NORAG_PROMPT.format(question=q["question"])
        else:
            results = bm25.search_docs(q["question"], docs, k=TOP_K)
            retrieved_ids = [d["id"] for d, _ in results]
            prompt = ANSWER_PROMPT.format(context=build_context(results), question=q["question"])
        answer = call_llm(model, prompt)
        print(f"  {q['id']} ✓")
        return {"id": q["id"], "type": q["type"], "question": q["question"],
                "gold_answer": q["gold_answer"], "gold_doc_ids": q["gold_doc_ids"],
                "retrieved_doc_ids": retrieved_ids, "answer": answer.strip()}

    with ThreadPoolExecutor(max_workers=8) as ex:
        answers = list(ex.map(gen, qa))

    suffix = "_norag" if norag else ""
    out = HERE / f"answers_{alias}{suffix}.json"
    out.write_text(json.dumps({"_model": model, "_mode": mode, "top_k": TOP_K,
                               "answers": answers}, ensure_ascii=False, indent=2))
    print(f"→ {out.name} に保存（{len(answers)}問）")


if __name__ == "__main__":
    main()
