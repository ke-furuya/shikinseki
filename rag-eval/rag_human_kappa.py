#!/usr/bin/env python3
"""人手ラベル vs LLM judge の一致検証（judgeの信頼性を人間基準で測る）。

「judge同士が食い違う」の次の問い＝「じゃあどのjudgeが人間に近いのか」。
けんてぃ本人が30問を採点し、judgeごとに κ(judge, 人間) を出す。
これがある評価ハーネスと無い評価ハーネスでは、外部評価での説得力が段違い。

使い方:
  1. python3 rag_human_kappa.py --template
     → human_labels.json が出る。各問の human_* を 0/1/2 で埋める（所要20-30分）
     ※ rubricはjudgeと同じ: faithfulness(正しさ) / citation(引用) / coverage(網羅)
  2. 埋めたら: python3 rag_human_kappa.py
     → judgeごとの κ・一致率・系統差を表示
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
for _cand in (HERE.parent / "shikinseki", HERE.parent / "shikinseki-public" / "shikinseki"):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
from edge_validator import cohens_kappa, mean_ci  # noqa: E402

AXES = ("faithfulness", "citation", "coverage")
LABELS_PATH = HERE / "human_labels.json"


def is_num(x):
    return isinstance(x, (int, float))


def make_template():
    data = json.loads((HERE / "answers_gemini.json").read_text())
    rows = []
    for a in data["answers"]:
        rows.append({
            "id": a["id"], "type": a["type"],
            "question": a["question"],
            "gold_answer": a["gold_answer"],
            "answer": a["answer"],
            "retrieved_doc_ids": a["retrieved_doc_ids"],
            "human_faithfulness": None,
            "human_citation": None,
            "human_coverage": None,
        })
    LABELS_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"→ {LABELS_PATH.name} を生成（30問）。human_* を 0/1/2 で埋めて再実行。")
    print("  採点基準はjudgeと同一: 2=問題なし / 1=部分的 / 0=誤り・欠落")


def compare():
    rows = json.loads(LABELS_PATH.read_text())
    human = {r["id"]: r for r in rows}
    n_labeled = sum(1 for r in rows if is_num(r.get("human_faithfulness")))
    if n_labeled < 10:
        raise SystemExit(f"human_* が{n_labeled}問しか埋まっていない（10問以上必要）。--templateで生成→記入して再実行。")
    print(f"人手ラベル {n_labeled}問 vs 各judge:")

    for jf in sorted(HERE.glob("judged_gemini_by_*.json")):
        judge = jf.stem.replace("judged_gemini_by_", "")
        d = {s["id"]: s for s in json.loads(jf.read_text())["scores"]}
        print(f"  --- judge={judge} ---")
        for ax in AXES:
            ids = [i for i in human if is_num(human[i].get(f"human_{ax}"))
                   and i in d and is_num(d[i].get(ax))]
            if len(ids) < 10:
                print(f"    [{ax}] 共通{len(ids)}問＝比較不能")
                continue
            hu = [human[i][f"human_{ax}"] for i in ids]
            ju = [d[i][ax] for i in ids]
            kap = cohens_kappa(ju, hu)
            mc = mean_ci([{"value": d[i][ax] - human[i][f"human_{ax}"], "group": i} for i in ids])
            kstr = "degenerate" if kap.get("kappa") is None else f"{kap['kappa']:.3f}"
            bias = ("judgeが甘い" if mc["robust_positive"] else
                    "judgeが辛い" if mc["robust_negative"] else "偏りは言えない")
            print(f"    [{ax}] n={len(ids)}  κ(judge,人間)={kstr} 一致{kap['percent_agreement']*100:.0f}%  "
                  f"差{mc['mean']:+.2f} CI[{mc['ci_low']:+.2f},{mc['ci_high']:+.2f}] → {bias}")


if __name__ == "__main__":
    if "--template" in sys.argv:
        make_template()
    else:
        compare()
