#!/usr/bin/env python3
"""最小RAG（検索部）＋検索の評価。

依存は numpy のみ（埋め込みAPI・LLMキー不要）。
- 知識ベース(postmortems.jsonl)を文単位にチャンク分割
- BM25 で質問→根拠チャンクを検索
- gold_doc_ids と照合して Recall@k / MRR を出す

STEP3 の「チャンク分割→検索→根拠提示」までを API 非依存で動かす骨格。
回答生成(LLM)と judge 採点は APIキー確認後に足す（rag_min はそこへ根拠を渡す土台）。
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
KB_PATH = HERE / "knowledge" / "postmortems.jsonl"
QA_PATH = HERE / "qa" / "qa_set.jsonl"

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_documents(docs: list[dict]) -> list[dict]:
    """各ドキュメントを文単位のチャンクに割る（最小構成）。
    チャンクは doc_id を保持し、根拠の出所を追える。"""
    chunks = []
    for d in docs:
        # タイトル＋本文を対象に、素朴に文分割
        sentences = re.split(r"(?<=[.!?])\s+", d["text"].strip())
        for i, s in enumerate(sentences):
            s = s.strip()
            if not s:
                continue
            chunks.append({
                "chunk_id": f"{d['id']}::{i}",
                "doc_id": d["id"],
                "title": d["title"],
                "text": f"{d['title']}. {s}",
            })
    return chunks


class BM25:
    """純 numpy の BM25。埋め込みなしのスパース検索ベースライン。"""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.corpus_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_len = np.array([len(t) for t in self.corpus_tokens], dtype=float)
        self.avgdl = self.doc_len.mean()
        self.N = len(chunks)

        df = Counter()
        for toks in self.corpus_tokens:
            for term in set(toks):
                df[term] += 1
        # BM25+ の idf（非負）
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(toks) for toks in self.corpus_tokens]

    def search(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        q_terms = tokenize(query)
        scores = np.zeros(self.N)
        for i in range(self.N):
            tf_i, dl = self.tf[i], self.doc_len[i]
            s = 0.0
            for term in q_terms:
                if term not in tf_i:
                    continue
                freq = tf_i[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf.get(term, 0.0) * (freq * (self.k1 + 1)) / denom
            scores[i] = s
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def search_docs(self, query: str, docs: list[dict], k: int = 5) -> list[tuple[dict, float]]:
        """doc単位の検索（重複排除版）。チャンクスコアの最大値でdocをランク付けし、
        上位k件の“異なる”docを返す。横断問でチャンク重複が枠を食う問題への対処。"""
        chunk_hits = self.search(query, k=self.N)  # 全チャンクをスコア順に
        doc_best: dict = {}
        for c, s in chunk_hits:
            if c["doc_id"] not in doc_best or s > doc_best[c["doc_id"]]:
                doc_best[c["doc_id"]] = s
        by_id = {d["id"]: d for d in docs}
        ranked = sorted(doc_best.items(), key=lambda x: -x[1])[:k]
        return [(by_id[doc_id], score) for doc_id, score in ranked]


def evaluate_retrieval(bm25: BM25, qa: list[dict], k: int = 5, docs=None) -> dict:
    """gold_doc_ids を使った検索評価。docs を渡すと doc重複排除モードで評価。
    unanswerable（gold 空）は検索評価から除外し別集計。"""
    recalls, rrs = [], []
    per_q = []
    for q in qa:
        gold = set(q.get("gold_doc_ids", []))
        if docs is not None:
            retrieved_docs = [d["id"] for d, _ in bm25.search_docs(q["question"], docs, k=k)]
        else:
            results = bm25.search(q["question"], k=k)
            retrieved_docs = [c["doc_id"] for c, _ in results]
        if not gold:  # unanswerable（q25 等）は別扱い
            per_q.append({"id": q["id"], "type": q["type"], "gold": [], "top_docs": retrieved_docs[:k], "hit": None})
            continue
        # Recall@k: gold のうち上位kに現れた割合
        hit_set = gold & set(retrieved_docs)
        recall = len(hit_set) / len(gold)
        recalls.append(recall)
        # MRR: gold の最初の1件が現れた順位
        rr = 0.0
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc in gold:
                rr = 1.0 / rank
                break
        rrs.append(rr)
        per_q.append({"id": q["id"], "type": q["type"], "gold": sorted(gold),
                      "top_docs": retrieved_docs[:k], "recall": round(recall, 3), "rr": round(rr, 3)})
    return {
        "k": k,
        "n_scored": len(recalls),
        "n_unanswerable": sum(1 for q in qa if not q.get("gold_doc_ids")),
        "mean_recall@k": round(float(np.mean(recalls)), 3),
        "mrr": round(float(np.mean(rrs)), 3),
        "per_q": per_q,
    }


def main():
    docs = load_jsonl(KB_PATH)
    qa = load_jsonl(QA_PATH)
    chunks = chunk_documents(docs)
    bm25 = BM25(chunks)

    print(f"知識ベース: {len(docs)} ドキュメント → {len(chunks)} チャンク")
    print(f"質問セット: {len(qa)} 問\n")

    print("== 検索評価（doc重複排除モード＝Round2の本番構成） ==")
    for k in (3, 5):
        r = evaluate_retrieval(bm25, qa, k=k, docs=docs)
        print(f"  @k={k}: mean Recall@{k} = {r['mean_recall@k']}   MRR = {r['mrr']}"
              f"   取りこぼし {sum(1 for q in r['per_q'] if q.get('recall') is not None and q['recall'] < 1.0)}問")
    print()

    for k in (3, 5):
        r = evaluate_retrieval(bm25, qa, k=k)
        print(f"== 検索評価（旧チャンクモード） @k={k} ==")
        print(f"  採点対象 {r['n_scored']}問 / unanswerable {r['n_unanswerable']}問(除外)")
        print(f"  mean Recall@{k} = {r['mean_recall@k']}   MRR = {r['mrr']}")
        # 取りこぼし（recall<1）を表示＝どの質問で根拠を引けてないか
        misses = [q for q in r["per_q"] if q.get("recall") is not None and q["recall"] < 1.0]
        if misses:
            print(f"  取りこぼし {len(misses)}問:")
            for q in misses:
                print(f"    {q['id']}({q['type']}) recall={q['recall']} gold={q['gold']} top={q['top_docs']}")
        print()

    # デモ: 1問の検索結果（根拠付き回答の材料になる）
    demo = qa[0]
    print(f"== デモ検索: {demo['id']} ==")
    print(f"  Q: {demo['question']}")
    for c, s in bm25.search(demo["question"], k=3):
        print(f"  [{s:.2f}] ({c['doc_id']}) {c['text'][:90]}...")


if __name__ == "__main__":
    main()
