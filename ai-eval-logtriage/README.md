# BGL実ログ 障害トリアージ プロンプトA/B評価 — 試金石の第2実ドメイン

[試金石](../README.md)を、**実世界の公開データ**（BlueGene/L スーパーコンピュータの実RASログ）による
障害トリアージに適用する。[ai-eval-sentiment](../ai-eval-sentiment/)（手作り110件）に残っていた
「テストセットが手作り」という弱点を、実データで潰しに行く回。

## 問い

1. SRE風に工夫したプロンプトBは、素朴なAより本当に良いのか？（感情分析では符号ごと反転した）
2. そもそもLLMは、**単純なキーワードルール（severity FATAL → alert）に勝てるのか？**
   ——「LLMを使うべきか」は、まずこのbaselineに勝ってからの話。

## データと出所

- **BGL**: Lawrence Livermore National Labs の BlueGene/L (131,072 CPU) 実システムログ。
  各行に「アラート種別タグ or `-`（非アラート）」のラベルが付いた、ログ研究の標準データセット。
  出典: Oliner & Stearley, "What Supercomputers Say: A Study of Five System Logs" (DSN 2007)。
- 取得元は [LogHub (logpai/loghub)](https://github.com/logpai/loghub) の 2,000行公開サンプル
  （研究用途に無償公開）。`testset.json` はそこから決定論抽出した201行＋出所情報。
- 再生成: `python3 prepare_testset.py`（取得→**completeness_gateで検査**→層別抽出まで自動）。

## なぜこのタスクは自明でないか

BGLの面白い構造＝**normal側にもFATALが204行ある**（2,000行中）。クラッシュ周辺の
レジスタダンプ・状態出力（`instruction address: 0x...` 等）はFATALだが「対応すべきアラート」
ではない。つまり：

- 「FATALという単語→alert」のキーワードルールは、全2,000行では **88%**。
- 本testsetは**ハードネガティブ（FATALだがnormal）を意図的に厚く**採っているので（90/201行）、
  同じルールがtestset上では **57.7%** に落ちる。⚠️ ここの正解率は実運用の分布上の値ではない
  ＝「難所でどれだけ差が出るか」を測るための層別設計（分布は `kind` で層別報告する）。

これは実務の「アラート疲れ」（怖い単語は多いが、対応すべき行は少ない）の縮図で、
LLMトリアージの価値が本当に出るかを問うのに向いている。

## 試金石の4層をそのまま通す

| 層 | この題材での役割 |
|---|---|
| ①取得 | LogHubから取得 → `completeness_gate` で必須列の欠損を使う前に検査 |
| ②防御 | **行頭のラベルトークンを本文に残すと `target_leak_scan` が分離度1.0で検出**（＝答えの混入）→剥がす。severity(FATAL)は運用者も見る正当な入力なので残す（分離不完全＝非検出も機械で確認） |
| ③検証 | `edge-validator` で判定。**group=ログテンプレ(EventId)**＝同一テンプレの行は相関するので1行1群にしない（重複だらけのログで母数を水増しして「有意」を捏造しない） |

## 測定の設計（結果を見る前に固定した事項）

- プロンプトBは**結果を見る前に1回だけ書いた**（チューニングループなし）。一般的なトリアージ原則
  （イベントか詳細ダンプか／FATAL≠alert）を素直に文章化したもの。
- 検出力の事前計算：N=201・実効group104では、**MDE=5ppの証明には約770行必要＝この題材で
  証明できるのは大きな差だけ**。小さな差は「まぐれと区別できない」に落ちる想定で読む。
- モデルは感情分析と同じ独立3系列（gemini-2.5-flash-lite / gpt-4o-mini / llama-3.2-3b）
  ＝「効果はモデル固有か」を再確認する。

## 再現方法

```bash
export FAL_KEY=<your-fal-api-key>   # https://fal.ai（201行×2プロンプト≒数円〜数十円/モデル）
python3 run_eval_logtriage.py google/gemini-2.5-flash-lite
python3 run_eval_logtriage.py openai/gpt-4o-mini
python3 run_eval_logtriage.py meta-llama/llama-3.2-3b-instruct
# predictions_real_*.json が既にあればAPIを呼ばず判定だけ再実行（APIキー不要）
```

## 結果

実測はこれから（上のコマンドで誰でも再現できる）。測定後、この節に
「A/B差・ルールとの差・層別・モデル間の一致/不一致」と、その正直な限界を追記する。

## ファイル

`prepare_testset.py`（取得→ゲート→リーク検査→層別抽出）／`testset.json`（201行・ラベル・group付き）／
`run_eval_logtriage.py`（実API測定＋edge-validator判定）／`predictions_real_*.json`（測定後に生成）
