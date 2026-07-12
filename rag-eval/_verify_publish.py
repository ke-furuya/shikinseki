#!/usr/bin/env python3
"""出荷前の機械verifyゲート（CLAUDE.md if-then「数値の完了条件を先に宣言」の実装）。

完了条件＝FAIL=0。FAIL>0の間は公開しない。監査の追加ラウンドは
「新しい失敗仮説を具体的に言える時だけ」＝このスクリプトにチェックを足す形で行う。
  python3 _verify_publish.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def main():
    # 1. 知識ベース
    kb_lines = [ln for ln in (HERE / "knowledge/postmortems.jsonl").read_text().splitlines() if ln.strip()]
    docs = [json.loads(ln) for ln in kb_lines]
    ids = [d["id"] for d in docs]
    check("KB: 18ドキュメント", len(docs) == 18, f"実際{len(docs)}")
    check("KB: id重複なし", len(ids) == len(set(ids)))
    check("KB: 旧'dynamodb-2020'誤記の残存なし", "pm-dynamodb-2020" not in ids)

    # 2. 質問セット
    qa = [json.loads(ln) for ln in (HERE / "qa/qa_set.jsonl").read_text().splitlines() if ln.strip()]
    check("QA: 30問", len(qa) == 30, f"実際{len(qa)}")
    orphan = [q["id"] for q in qa for g in q["gold_doc_ids"] if g not in ids]
    check("QA: gold_doc_idsが全てKBに存在", not orphan, f"孤児参照 {orphan}")
    n_trap = sum(1 for q in qa if q["type"] == "unanswerable")
    check("QA: 罠問3問以上", n_trap >= 3, f"実際{n_trap}")

    # 3. 検索の回帰ゲート
    sys.path.insert(0, str(HERE))
    from rag_min import BM25, chunk_documents, evaluate_retrieval
    bm25 = BM25(chunk_documents(docs))
    r = evaluate_retrieval(bm25, qa, k=5, docs=docs)
    check("検索: doc-dedup Recall@5 >= 0.90", r["mean_recall@k"] >= 0.90, f"実際{r['mean_recall@k']}")
    check("検索: MRR >= 0.90", r["mrr"] >= 0.90, f"実際{r['mrr']}")

    # 4. 生成・採点ファイル（存在する分だけ検査）
    for f in sorted(HERE.glob("answers_*.json")):
        data = json.loads(f.read_text())
        n = len(data["answers"])
        nonempty = sum(1 for a in data["answers"] if a["answer"].strip())
        check(f"{f.name}: 30問・空回答<=2", n == 30 and nonempty >= 28, f"{n}問/空{n-nonempty}")
    judged_ok = 0
    for f in sorted(HERE.glob("judged_*_by_*.json")):
        scores = json.loads(f.read_text())["scores"]
        n_num = sum(1 for s in scores if isinstance(s.get("faithfulness"), (int, float)))
        if "_norag_" not in f.name and n_num >= 27:
            judged_ok += 1
    check("judge: RAGモードで有効judge(27/30数値)が2つ以上", judged_ok >= 2, f"実際{judged_ok}")

    # 5. 秘密情報：pyファイルにキーのハードコードなし
    leaked = []
    for f in HERE.glob("*.py"):
        src = f.read_text()
        if re.search(r"FAL_KEY\s*=\s*['\"][^'\"]{10,}", src):
            leaked.append(f.name)
    check("秘密: FAL_KEYのハードコードなし", not leaked, str(leaked))

    print(f"\nFAIL={len(FAILS)}" + (f"  → 未達: {FAILS}" if FAILS else "  → 出荷ゲート通過（公開してよい）"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
