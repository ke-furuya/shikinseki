#!/usr/bin/env python3
"""引用 [n] の決定論チェック（LLM judge 不要の軸を機械で採点する）。

judge実験で citation 軸は κ≈0（judge間の一致がランダム同然）だった。
だが引用の正しさは、[n]→retrieved_doc_ids[n-1]→gold_doc_ids の照合で機械的に測れる。
教訓の実装＝「機械で測れる軸にLLM judgeを使わない。judgeは判断が要る軸だけに使う」。

  python3 rag_citation_check.py [answers_file.json]   # default: answers_gemini.json
出力: citation_check_<name>.json ＋ サマリ表示
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
CIT_RE = re.compile(r"\[(\d+)\]")
REFUSAL = "do not contain this information"


def check_answer(a: dict) -> dict:
    """1回答の引用を機械採点。precision=引用のうちgoldを指す割合 / recall=goldのうち引用された割合。"""
    gold = set(a.get("gold_doc_ids", []))
    retrieved = a.get("retrieved_doc_ids", [])
    cited_idx = sorted({int(n) for n in CIT_RE.findall(a.get("answer", ""))})
    valid_idx = [n for n in cited_idx if 1 <= n <= len(retrieved)]
    cited_docs = {retrieved[n - 1] for n in valid_idx}
    refused = REFUSAL in a.get("answer", "").lower()

    if a.get("type") == "unanswerable":
        # 罠問の正解＝拒否（または前提訂正）。gold_doc_idsの有無でなくtypeで判定する。
        # 引用は「してもよい」（前提訂正で関連docを引く形は正しい）が、必須ではない。
        ok = refused or (cited_docs and cited_docs <= gold)
        return {"id": a["id"], "type": a["type"], "unanswerable": True,
                "refused": refused, "n_cited": len(cited_idx), "citation_ok": bool(ok)}
    precision = (len(cited_docs & gold) / len(cited_docs)) if cited_docs else 0.0
    recall = len(cited_docs & gold) / len(gold)
    return {"id": a["id"], "type": a["type"], "unanswerable": False,
            "refused": refused, "n_cited": len(cited_idx),
            "out_of_range": len(cited_idx) - len(valid_idx),
            "cited_docs": sorted(cited_docs), "gold_docs": sorted(gold),
            "precision": round(precision, 3), "recall": round(recall, 3),
            "citation_ok": precision == 1.0 and recall > 0}


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "answers_gemini.json"
    data = json.loads((HERE / fname).read_text())
    results = [check_answer(a) for a in data["answers"]]

    scored = [r for r in results if not r["unanswerable"]]
    traps = [r for r in results if r["unanswerable"]]
    n_ok = sum(1 for r in scored if r["citation_ok"])
    mean_p = sum(r["precision"] for r in scored) / len(scored)
    mean_r = sum(r["recall"] for r in scored) / len(scored)

    print(f"== 引用の決定論チェック: {fname}  (通常{len(scored)}問+罠{len(traps)}問) ==")
    print(f"  citation_ok (precision=1.0 & recall>0): {n_ok}/{len(scored)}問")
    print(f"  mean precision = {mean_p:.3f}   mean recall = {mean_r:.3f}")
    bad = [r for r in scored if not r["citation_ok"]]
    for r in bad:
        print(f"    NG {r['id']}({r['type']}) p={r['precision']} r={r['recall']} "
              f"cited={r['cited_docs']} gold={r['gold_docs']}")
    for r in traps:
        print(f"  罠 {r['id']}: 拒否={r['refused']} 引用数={r['n_cited']} → {'OK' if r['citation_ok'] else 'NG'}")

    out = HERE / f"citation_check_{Path(fname).stem}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"→ {out.name} に保存")


if __name__ == "__main__":
    main()
