#!/usr/bin/env python3
"""感情分析プロンプトA/B評価 — 試金石（edge-validator）の初・実ドメイン適用。

問い：「工夫したプロンプトBは、素朴なAより本当に良いのか？ それともまぐれか？」
普通は正解率の点推定だけ見て勝者を決める。試金石は『その差は測れているか』まで見る。

実APIで回す場合：predictions.json を手書きする代わりに、各 text を
prompt_A / prompt_B でモデルに投げて pos/neg を取り、同じ形で保存すればよい（seam は下記）。
"""
import json, sys, statistics
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # リポジトリルート
from shikinseki import edge_validator as ev


# ── 実APIに差し替えるならここだけ（今回は predictions.json を読むモック）──
def classify(prompt_template, text):
    raise NotImplementedError(
        "実行時は predictions.json を使用。実APIならここで "
        "anthropic等を呼んで pos/neg を返す実装に差し替える。")


def main():
    testset = {str(r["id"]): r for r in json.load(open(HERE / "testset.json"))}
    preds = json.load(open(HERE / "predictions.json"))

    from collections import defaultdict
    A_rec, B_rec, diff_rec = [], [], []
    strata = defaultdict(list)   # 層別の差分（kind / class）
    wrongA, wrongB = [], []
    for rid, item in testset.items():
        if rid not in preds:
            continue
        lab = item["label"]
        ac = 1 if preds[rid]["A"] == lab else 0
        bc = 1 if preds[rid]["B"] == lab else 0
        # 1件=1group（独立）。group単位CI＝項目ブートストラップ
        A_rec.append({"group": rid, "value": ac})
        B_rec.append({"group": rid, "value": bc})
        diff_rec.append({"group": rid, "value": bc - ac})
        strata[f"kind={item.get('kind','?')}"].append({"group": rid, "value": bc - ac})
        strata[f"class={lab}"].append({"group": rid, "value": bc - ac})
        if not ac:
            wrongA.append((rid, item["text"]))
        if not bc:
            wrongB.append((rid, item["text"]))

    N = len(A_rec)
    accA = ev.mean_ci(A_rec)
    accB = ev.mean_ci(B_rec)
    gap = ev.mean_ci(diff_rec)

    print("=" * 66)
    print(f"感情分析 プロンプトA/B 評価（N={N}件・試金石で判定）")
    print("=" * 66)
    print(f"プロンプトA(素朴)  正解率 {accA['mean']*100:5.1f}%  "
          f"90%CI {accA['ci_low']*100:.1f}〜{accA['ci_high']*100:.1f}%")
    print(f"プロンプトB(工夫)  正解率 {accB['mean']*100:5.1f}%  "
          f"90%CI {accB['ci_low']*100:.1f}〜{accB['ci_high']*100:.1f}%")
    print("-" * 66)
    g = gap["mean"] * 100
    lo, hi = gap["ci_low"] * 100, gap["ci_high"] * 100
    print(f"B−A の差          {g:+.1f}pp   90%CI {lo:+.1f}〜{hi:+.1f}pp")
    print("判定: " + ("✅ Bが頑健に良い（90%CIが0をまたがない）"
                    if gap["robust_positive"]
                    else "⚠️ まぐれと区別できない（90%CIが0をまたぐ）"))
    # CI用の回帰固定：READMEの主張「N=110の手ラベルでB−AのCIが0を除外」そのものを固定する
    assert gap["robust_positive"], f"回帰: B−AのCIが0を除外しなくなった gap={gap}"

    # 検出力：観測差を使うpost-hoc powerは避け、『事前に決めた最小効果(MDE)』で必要Nを出す
    MDE = 0.05  # 事前指定：5pp以上の改善を「意味あり」とみなす（観測差は使わない）
    pw = ev.power_required_mean([r["value"] for r in diff_rec], mde=MDE)
    print(f"\n検出力（事前指定MDE={MDE*100:.0f}pp・post-hoc powerは使わない）")
    print(f"  {MDE*100:.0f}pp差を80%検出力・両側5%で検出するのに要るN ≈ {pw['need_n']:.0f}件（現在 {N}件）")
    if pw["heavy_tailed"]:
        print(f"  ⚠️ 分布の裾が重い(歪度{pw['skewness']:.1f})→CLT近似が甘く、実際はさらに必要な可能性")

    # 逐次的に増やして何度も覗くなら：anytime-valid な confidence sequence で判定
    # （差は{-1,0,1}＝値域2なので sigma=1 の保守版＝誤り率を理論保証）
    cs = ev.confidence_sequence([r["value"] for r in diff_rec], alpha=0.10, sigma=1.0)
    csl, csh = cs["ci_low"] * 100, cs["ci_high"] * 100
    decided = cs["robust_positive"] or cs["robust_negative"]
    print(f"\n[覗き見OKな anytime CI（confidence sequence・保守版）]")
    print(f"  B−A {cs['mean']*100:+.1f}pp  CS {csl:+.1f}〜{csh:+.1f}pp"
          f"  → {'✅決着' if decided else '⚠️未決着（覗き見の自由の代償で固定Nより広い）'}")
    print("  ※Nを先に決めて1回だけ判定するなら上の90%CIで可。"
          "増やしながら何度も覗くなら、この広いCSで見る（覗き見の罠を防ぐ）。")

    # 層別レポート：集計は層を隠す。効果がどこに居るかを見る
    print(f"\n[層別 B−A（集計は層を隠す＝効果の在処を見る）]")
    for key in sorted(strata):
        s = ev.mean_ci(strata[key])
        flag = "✅" if s["robust_positive"] else ("➖" if s["robust_negative"] else "・")
        print(f"  {key:<14} n={s['n']:<3} B−A {s['mean']*100:+5.1f}pp"
              f"  90%CI {s['ci_low']*100:+.1f}〜{s['ci_high']*100:+.1f}pp {flag}")

    # 意思決定層：「本物か」→「採用すべきか／集める価値があるか」
    VALUE_PER_PP = 1.0    # 1pp精度向上の価値（用途で設定）
    SWITCH_COST = 2.0     # Bの切替コスト（追加トークン/レイテンシ等を同単位で）
    dv = ev.decision_value_ab(g, lo, VALUE_PER_PP, SWITCH_COST)
    se_pp = (hi - lo) / (2 * 1.645)
    vmd = ev.value_of_more_data(g, se_pp, VALUE_PER_PP, sampling_cost=1.0)
    print(f"\n[意思決定（価値=1/pp・切替コスト={SWITCH_COST}）]")
    print(f"  採用判断: 期待純益{dv['expected_net']:+.1f} 悲観純益{dv['worst_net']:+.1f} → {dv['recommendation']}")
    print(f"  追加収集: 情報の価値EVPI={vmd['evpi']:.2f} → {vmd['verdict']}")

    # 測定の信頼性：今回はラベル1人＝一致度が測れない（妥当性の前提）
    print("\n[妥当性の前提]")
    print("  ⚠️ ラベルは1人＝アノテータ間一致(κ)が未測定。≥2名で ev.cohens_kappa を。")
    print("  ⚠️ 分類モデル=このセッションのClaude（評価者≒モデル）。独立モデルで再実施を。")

    print("\n[Aだけ外した文（プロンプトの工夫が効いた候補）]")
    seenB = {r for r, _ in wrongB}
    for rid, t in wrongA:
        mark = "（Bも外し）" if rid in seenB else "（Bは正解）"
        print(f"  #{rid} {mark} {t}")


if __name__ == "__main__":
    main()
