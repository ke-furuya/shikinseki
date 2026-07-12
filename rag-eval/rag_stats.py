#!/usr/bin/env python3
"""judge採点の集計＝「本物か」判定（Round2仕様）。

出すもの:
  1. 各judge×各軸の平均スコア（RAG / no-RAG）
  2. judge間の一致 cohens_kappa ＋ 系統差 mean_ci（有効judge総当たり）
  3. RAG vs no-RAG の対差 mean_ci（faithfulness/coverage）＝「RAGは効いたか」
  4. 機械citationチェックとjudge citationスコアの突き合わせ
  5. 罠問(unanswerable)での忠実性
空応答judgeは除外し明示（誇張しない）。
"""
import itertools
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
GEN = "gemini"
JUDGE_ALIASES = ["gpt4o-mini", "gemini", "llama", "llama31-8b", "llama32-1b",
                 "claude-sonnet", "gpt4o", "gemini-pro", "deepseek"]


def is_num(x):
    return isinstance(x, (int, float))


def load_scores(suffix=""):
    out = {}
    for j in JUDGE_ALIASES:
        p = HERE / f"judged_{GEN}{suffix}_by_{j}.json"
        if p.exists():
            out[j] = {s["id"]: s for s in json.loads(p.read_text())["scores"]}
    return out


def usable_judges(judged):
    ok = []
    for j, d in judged.items():
        n = sum(1 for s in d.values() if is_num(s.get("faithfulness")))
        print(f"  {j:12} : {n}/{len(d)}問 数値採点 → {'使用可' if n >= 20 else '除外（空応答/失敗多数）'}")
        if n >= 20:
            ok.append(j)
    return ok


def axis_means(judged, ok):
    for j in ok:
        d = judged[j]
        row = []
        for ax in AXES:
            vals = [s[ax] for s in d.values() if is_num(s.get(ax))]
            row.append(f"{ax}={sum(vals)/len(vals):.2f}" if vals else f"{ax}=—")
        print(f"  {j:12} : " + "  ".join(row))


def pairwise(judged, ok):
    for ja, jb in itertools.combinations(ok, 2):
        da, db = judged[ja], judged[jb]
        common = [i for i in da if i in db]
        print(f"  --- {ja} vs {jb} ---")
        for ax in AXES:
            ids = [i for i in common if is_num(da[i].get(ax)) and is_num(db[i].get(ax))]
            if len(ids) < 10:
                print(f"    [{ax}] 共通数値{len(ids)}問＝比較不能")
                continue
            kap = cohens_kappa([da[i][ax] for i in ids], [db[i][ax] for i in ids])
            mc = mean_ci([{"value": da[i][ax] - db[i][ax], "group": i} for i in ids])
            kstr = "degenerate" if kap.get("kappa") is None else f"{kap['kappa']:.3f}"
            sysd = "系統差あり" if (mc["robust_positive"] or mc["robust_negative"]) else "系統差は言えない"
            print(f"    [{ax}] n={len(ids)}  κ={kstr} 一致{kap['percent_agreement']*100:.0f}%  "
                  f"差{mc['mean']:+.2f} CI[{mc['ci_low']:+.2f},{mc['ci_high']:+.2f}] → {sysd}")


def ablation(rag, norag, ok_rag, ok_norag):
    ok = [j for j in ok_rag if j in ok_norag]
    if not ok:
        print("  RAG/no-RAG両方で有効なjudgeが無い＝対照比較不能")
        return
    for j in ok:
        dr, dn = rag[j], norag[j]
        common = [i for i in dr if i in dn]
        print(f"  --- judge={j} ---")
        for ax in ("faithfulness", "coverage"):  # citationはno-RAGで定義不能＝除外
            ids = [i for i in common if is_num(dr[i].get(ax)) and is_num(dn[i].get(ax))]
            if len(ids) < 10:
                print(f"    [{ax}] 共通数値{len(ids)}問＝比較不能")
                continue
            mc = mean_ci([{"value": dr[i][ax] - dn[i][ax], "group": i} for i in ids])
            verdict = ("RAGが頑健に勝ち" if mc["robust_positive"] else
                       "no-RAGが頑健に勝ち" if mc["robust_negative"] else "差は言えない")
            print(f"    [{ax}] n={len(ids)}  RAG−noRAG {mc['mean']:+.2f} "
                  f"CI[{mc['ci_low']:+.2f},{mc['ci_high']:+.2f}] → {verdict}")


def citation_vs_machine(judged, ok):
    p = HERE / "citation_check_answers_gemini.json"
    if not p.exists():
        print("  citation_check未実行＝スキップ")
        return
    machine = {r["id"]: r for r in json.loads(p.read_text()) if not r["unanswerable"]}
    for j in ok:
        d = judged[j]
        good = [d[i]["citation"] for i in machine if machine[i]["citation_ok"]
                and i in d and is_num(d[i].get("citation"))]
        bad = [d[i]["citation"] for i in machine if not machine[i]["citation_ok"]
               and i in d and is_num(d[i].get("citation"))]
        if good and bad:
            print(f"  {j:12}: 機械OK群のjudge citation平均 {sum(good)/len(good):.2f} (n={len(good)}) "
                  f"vs 機械NG群 {sum(bad)/len(bad):.2f} (n={len(bad)})")
        elif good:
            print(f"  {j:12}: 機械OK群 {sum(good)/len(good):.2f} (n={len(good)})／NG群サンプル無し")


def traps(judged, ok):
    for j in ok:
        d = judged[j]
        vals = [(i, d[i]["faithfulness"]) for i in d
                if d[i].get("type") == "unanswerable" and is_num(d[i].get("faithfulness"))]
        if vals:
            print(f"  {j:12}: 罠{len(vals)}問 faithfulness平均 "
                  f"{sum(v for _, v in vals)/len(vals):.2f}/2  ({[v for _, v in vals]})")


def main():
    rag = load_scores("")
    norag = load_scores("_norag")

    print("=== judge採点の可否（RAGモード） ===")
    ok_rag = usable_judges(rag)
    ok_norag = []
    if norag:
        print("=== judge採点の可否（no-RAGモード） ===")
        ok_norag = usable_judges(norag)
    print()

    print("=== 各judge×各軸の平均（RAG） ===")
    axis_means(rag, ok_rag)
    if norag and ok_norag:
        print("=== 各judge×各軸の平均（no-RAG） ===")
        axis_means(norag, ok_norag)
    print()

    print("=== judge間の一致・系統差（RAG・総当たり） ===")
    pairwise(rag, ok_rag)
    print()

    if norag:
        print("=== RAG vs no-RAG（同一judge・対差のmean_ci） ===")
        ablation(rag, norag, ok_rag, ok_norag)
        print()

    print("=== 機械citationチェック vs judge citationスコア ===")
    citation_vs_machine(rag, ok_rag)
    print()

    print("=== 罠問(unanswerable)での忠実性（RAG） ===")
    traps(rag, ok_rag)


if __name__ == "__main__":
    main()
