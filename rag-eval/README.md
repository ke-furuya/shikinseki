# rag-eval — LLM-as-judgeの信頼性を測るRAG評価ハーネス

**「AIの答案をAIに採点させたとき、その採点者は信用できるのか?」を実測するための最小ハーネス。**
同じ30個のRAG回答を5つのLLM judge（GPT-4o / GPT-4o-mini / Gemini 2.5 Flash-Lite / Claude 3.5 Sonnet / Llama 3.1 8B）に同一rubricで採点させ、judge間の一致（Cohen's κ）と系統差（ブートストラップ90%CI）を測る。[試金石](../README.md)の統計エンジン（`edge_validator`）の適用例第3弾。

> 📖 物語と結果の解説 → **[AIの答案をAIに採点させたら、採点者ごとに点が違った](https://zenn.dev/ke_furuya/articles/96b0b15b84d9f0)**（Zenn）

## 主な実測結果（2026-07-12・N=30・すべて実API）

- **judge間の一致は「能力」でなく「系統」で決まった**：同族ペア（GPT-4o vs GPT-4o-mini）は忠実性・網羅性κ=0.858。実験内の上位2モデル同士（Claude 3.5 Sonnet vs GPT-4o・異族）はκ=0.27/0.005/0.34で全3軸に頑健な系統差
- **引用の正しさはjudge間κ≈0（測れない軸）→ 決定論チェッカーなら確定的に測れる**（27問中26問正解・precision 0.988）
- **RAG vs no-RAG対照**：網羅性は3judge全員が頑健プラス（＝本物）、忠実性はjudge次第（＝採点者選びで結論が変わる）
- **文書に答えが無い罠問3問は5judge全員が満点で一致**

## 構成

| ファイル | 役割 |
|---|---|
| `knowledge/postmortems.jsonl` | 知識ベース18文書（[danluu/post-mortems](https://github.com/danluu/post-mortems)の公開ポストモーテムを症状→原因→対処に要約） |
| `qa/qa_set.jsonl` | 30問（正解＋根拠文書ID付き。横断問・答えの無い罠問3問込み） |
| `rag_min.py` | BM25検索（numpyのみ）＋検索評価（Recall@k / MRR） |
| `rag_answer.py` | 根拠[n]付き回答生成（`--norag`で対照群） |
| `rag_judge.py` | LLM-as-judge採点（忠実性/引用/網羅・各0-2・全judge同一プロンプト） |
| `rag_citation_check.py` | **引用の決定論チェック**（judgeに聞かず[n]→文書照合で採点） |
| `rag_stats.py` | 集計＝judge総当たりκ・系統差CI・RAG対照・機械vs judge比較 |
| `rag_label.py` / `rag_human_kappa.py` | 人手ラベル入力（日本語対訳・チェックリスト式）とκ(judge,人間)算出 |
| `_verify_publish.py` | 出荷前verifyゲート（11チェック・完了条件FAIL=0） |
| `run_round2.py` / `run_strong_judges.py` | 一括実行（probe→生成→採点→集計→ゲート） |
| `answers_*.json` / `judged_*.json` / `citation_check_*.json` | 実測データ（再現用にそのまま同梱） |

## 再現方法

```bash
export FAL_KEY=<fal.aiのAPIキー>   # https://fal.ai
python3 run_round2.py              # 生成→3judge採点→機械チェック→統計→ゲート（約240コール）
python3 run_strong_judges.py       # 上位judge追加（約70コール）
python3 rag_stats.py               # 集計のみ（同梱の実測JSONを使う場合はキー不要）
```

- 依存：この適用例のみ検索部で `numpy` を使用（本体4層スタックは標準ライブラリのみ）。API呼び出しは標準ライブラリの`urllib`。
- 生成・採点のプロンプト全文は各スクリプトに含まれる。

## 限界（正直に）

N=30の小規模実験／生成モデルは1つ／judgeは2024-25年世代（2026年時点の各社最上位は未検証）／人間・専門家ラベルとの突き合わせは未実施（入力ツールとκ算出は実装済み＝次回）／κはunweighted／詳細は[記事の「正直な限界」](https://zenn.dev/ke_furuya/articles/96b0b15b84d9f0)参照。
