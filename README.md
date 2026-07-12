# 試金石 / shikinseki

![CI](https://github.com/ke-furuya/shikinseki/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![deps](https://img.shields.io/badge/dependencies-none%20(stdlib%20only)-brightgreen)

**「その数字、本物か? それともまぐれか?」を統計で見分ける、定量検証ツールキット。**
*A toolkit to tell whether a quantitative edge is real, luck, or simply unmeasurable.*

> 📖 統計用語が並ぶ前に、まず物語で知りたい人へ → **[AIで競馬に勝とうとして"まぐれ発見器"を作った話](ARTICLE.md)**

予測・施策・モデルを作る人は多い。でも「その改善、本当に効いてる? それとも良く見えてるだけ?」を統計で答えられる人は少ない。多くが**分散の揺れ（まぐれ）を実力と勘違いする。** 試金石は、それを構造的に防ぐ4層スタック：

| 層 | ツール | 役割 |
|---|---|---|
| ①取得 | [`data-harvester`](data-harvester/) | 壊れた/欠けたデータで分析を始めない（完全性ゲート） |
| ②防御 | [`leak-guard`](leak-guard/) | 未来情報の混入（リーク）を止める |
| ③**検証** | [`edge-validator`](edge-validator/) ⭐ | **「効いてる/まぐれ/測れない」を統計で判定（中核）** |
| ④運用 | [`action-gate`](action-gate/) | 出す結論を毎回同じ手順で機械チェックし、うっかり（重複・予算超過・矛盾）を仕組みで防ぐ |

すべて **Python標準ライブラリのみ**（numpy/pandas不要＝どこでも動く）。各ツールは独立して使えるし、繋いでも使える。ドメイン非依存＝アダプタだけ書けば何にでも適用できる。名前の由来は**試金石**＝本物の金か"愚者の金（ノイズ）"かを見分ける石。
コード本体は [`shikinseki/`](shikinseki/)（各層のディレクトリは解説＋実行例）。

### インストール / 30秒で確かめる
```bash
pip install -e .                             # 依存ゼロ（標準ライブラリのみ）
python3 -m unittest discover tests           # 回帰テスト62件（ツール自身が壊れてないかの回帰確認）
python3 edge-validator/example_advanced.py   # 検証エンジンの全機能を自己検証つきで
```
インストールなしでも動く（各スクリプトがリポジトリルートを自動でパスに追加）。古い pip で editable install が失敗する場合は最初の行を飛ばしてよい（依存ゼロなので実害なし）。

**自分のデータで試す最小コード**（A/Bテストのペア差を判定する例）：
```python
from shikinseki import edge_validator as ev

# my_ab_diffs = 対象ごとの (B成績 − A成績) のリスト。1対象=1group（独立）
records = [{"group": f"u{i}", "value": d} for i, d in enumerate(my_ab_diffs)]
print(ev.mean_ci(records))                            # 90%CIが0をまたぐか＝本物か/まぐれか
print(ev.power_required_mean(my_ab_diffs, mde=0.05))  # 5pp差の証明に何件要るか＝そもそも測れるか
```

> このツールキットは、AIで競馬予想の「勝てる買い方」を探して**勝てなかった**経験から生まれた。「公開情報で市場に勝つのは原理的に無理」を検出力計算つきで自ら証明する過程で作った検証の規律を、競馬から引き剥がして汎用化したもの。その顛末は [ARTICLE.md](ARTICLE.md) に。

## もっと動かす

```bash
# 「覗き見の罠」の実証：固定CIは偽陽性49%、confidence sequenceは0%
python3 edge-validator/example_sequential.py

# 実適用例：LLMプロンプトA/B評価（試金石で「差は本物か」を判定）
python3 ai-eval-sentiment/run_eval.py

# 実API・独立モデルで再測定（要 FAL_KEY・数円〜数十円/モデル）
FAL_KEY=<key> python3 ai-eval-sentiment/run_eval_realapi.py openai/gpt-4o-mini

# 第2の実ドメイン：実スパコンログ(BGL)の障害トリアージA/B — 実データ・測定手順つき
python3 ai-eval-logtriage/prepare_testset.py   # 取得→ゲート→リーク検査→層別抽出（結果はコミット済み）
FAL_KEY=<key> python3 ai-eval-logtriage/run_eval_logtriage.py google/gemini-2.5-flash-lite
```

## テスト（検証ツール自身を検証する）

「本物か、まぐれか」を判定するツールが壊れていては本末転倒なので、
**既知の答え・統計的性質で確認する回帰テスト62件**を用意している（標準ライブラリ `unittest` のみ・追加依存なし・CIで自動実行）。
中核の統計関数（edge_validator）を最も厚く、他の3層も**「止める」側の経路**——action-gate が不正アクションをブロックするか、
完全性ゲートが欠損で FAIL を返すか、リーク検出が H/M/L?/L を正しく分級するか——を固定している。

```bash
python3 -m unittest discover tests -v
```

例：正規CDF/PPFは数学の教科書値と照合、Cohen's κは完全一致/完全不一致で±1.0、
confidence sequence は「真値0を1000回引いて偽陽性率<5%（＝覗き見しても誤検出しない）」を実測、
PBO は純ノイズ戦略で高く・本物のエッジ1本で低く出ることを確認する（乱数ストリーム実装に依存しないよう複数seedの中央値で判定）。
回帰テストには「単一要素で誤って"有意"と言わない」「scan系が渡したデータを汚さない」
「較正のfit集合が小さくてもbinの参照がずれない」「group不足でCIが退化しても"頑健"と偽らない」など、
過去に自分で見つけたバグの再発防止テストも含む。

## 中核機能（edge-validator）が答える問い

- `roi_ci` / `mean_ci` — 価値・A/B差と、まぐれを除いた**信頼区間**（group単位ブートストラップ）
- `power_required` / `power_required_mean` — その差を**証明するのに何件要るか**＝そもそも測れるのか（事前指定MDE）
- `confidence_sequence` / `sequential_scan` — データを足しながら何度チェックしても判定がブレない信頼区間（「有意になった瞬間に止める」という自己欺瞞を防ぐ）
- `residual_scan` / `holdout_scan` / `staged_scan` — baselineを超えて予測するか、探索→検証→**確認**で再現したものだけ残す
- `pbo_cscv` / `deflated_sharpe_ratio` — 「試した戦略の中で一番良いのは、まぐれである確率何%か」を出す（クオンツ標準の過学習診断）
- `leak_check` — signalが未来情報で汚染されてないか
- `decision_value_ab` / `value_of_information` — 「本物か」→**採用すべきか／集める価値があるか**

> 「本物か測れる」の正確な意味：頻度論の枠内で「この標本・この前提のもとで効果が非ゼロか」を測る。真値そのものは誰にも測れない。前提（独立・定常・正規近似）が破れれば数字も歪む——だから本ツールは *前提を明示し、限界を返り値に載せる*（`degenerate` / `ci_caveat` / `heavy_tailed` 等）。

## 方法論：8原則

1. 点推定でなく信頼区間で判断する
2. 最良に見える結果ほど疑う（分散が一番大きい）
3. 探索と検証を必ず分ける
4. 試行回数を数える（多重比較）
5. 「測れるか」を先に問う（検出力）
6. baseline（超える相手）を必ず置く
7. リークを疑う
8. 止めどきを結果で決めない（覗き見の罠）

## 実適用の一例（[ai-eval-sentiment](ai-eval-sentiment/)）— 符号まで反転した

感情分析のプロンプトA/Bを試金石で評価する。まず対話セッション内の分類（＝手ラベル）で見ると：

```
N=30   B−A +6.7pp  90%CI +0.0〜+13.3pp → ⚠️ まぐれと区別できない
N=110  B−A +7.3pp  90%CI +2.7〜+11.8pp → ✅ 良く見える（※ただしこれは手ラベル）
層別:  kind=clear +0.0pp / kind=tricky +22.9pp ← 効果は全部"難しい文"に居た
```

手ラベルは信じない。**独立3モデルの実APIで測り直すと**（`run_eval_realapi.py`・再現可能）：

| 分類器 | B−A | 判定 |
|---|---|---|
| Claude（セッション内・手ラベル/再現不能） | +7.3pp | ⚠️ "身内採点"＝疑って独立検証した |
| gemini-2.5-flash-lite | +0.0pp | 差なし（天井） |
| gpt-4o-mini | +1.8pp | ⚠️ ノイズと区別不能 |
| Llama-3.2-3b | **−9.1pp** | ❌ 頑健に悪い（逆効果） |

**冒頭の+7.3ppは対話セッションでの手採点（＝身内採点）。それを疑って独立3モデルの実APIで測り直すと、頑健な改善はどこにも無く、唯一はっきり出たのは"弱いモデルでの悪化"だった。** 点推定は嘘をつき、集計は効き所を隠し、効果はモデル固有で符号すら保存されない。詳細と限界（Llamaの悪化は失敗4件を除くと有意ぎりぎり等）は [ai-eval-sentiment/README.md](ai-eval-sentiment/README.md)。

## 実適用の第2例（[ai-eval-logtriage](ai-eval-logtriage/)）— 今度は3モデル一貫で"本物"だった

実スパコンログ（BGL・公開実データ）の障害トリアージ。**素朴にLLMを投げると正規表現1本のキーワードルール（57.7%）に3モデルとも負け**（40〜49%）、ドメイン知識を載せたプロンプトは**3モデル全てでルールに頑健勝ち**（B−ルール +10〜+17pp・90%CIが0を除外）。効き所は「FATALと書いてあるが対応不要なダンプ行」の誤報止め（+41〜+76pp）。

感情分析（符号反転）と並べると教訓が完成する：**プロンプト改善は「効かない」でも「効く」でもなく、タスク×モデル固有——符号反転もあれば3モデル一貫の本物もある。どちらかは測るまで分からない。だから測る。** 限界（alert層の見逃し増の兆候は検出力不足で判定保留等）は [ai-eval-logtriage/README.md](ai-eval-logtriage/README.md)。

## 実適用の第3例（[rag-eval](rag-eval/)）— 今度は"採点者"自体を疑った

RAG QAの同じ30回答を**5つのLLM judge**に同一rubricで採点させ、judge間一致（Cohen's κ）と系統差（CI）を測定。**judgeの一致は能力でなく系統で決まった**（同族GPTペアκ0.86 vs 異族の上位2モデル間は全軸で頑健な系統差）。引用の正しさはjudge間κ≈0＝測れない軸だったが、決定論チェッカーなら確定的に測れる（26/27問）。RAGの効果は網羅性で3judge全員一致の本物・忠実性はjudge次第。**測る道具も測る**——シリーズ3作目の教訓。→ 解説記事: [AIの答案をAIに採点させたら、採点者ごとに点が違った](https://zenn.dev/ke_furuya/articles/96b0b15b84d9f0)

## 正直な限界（隠さないのが流儀）

**適用範囲**
- ドメイン非依存の実証＝競馬（実データ）・感情分析（手作り110件）・実スパコンログの障害トリアージ（BGL・公開実データ）。さらに異質なドメイン（数値時系列・因果推論寄りの施策効果等）は未検証。
- 感情分析実験の限界：テストセットは手作り110件で実運用分布ではない／強モデルには易しすぎて天井（Geminiの99.1%）／Llamaの−9.1ppには出力フォーマット崩れ4件を含む（詳細は [ai-eval-sentiment](ai-eval-sentiment/) に明記）。主張は個別数値でなく構造。
- ログトリアージ実験の限界：層別オーバーサンプリングのため正解率は実運用値でない／alert層（見逃し側）はテンプレ15種で判定保留（詳細は [ai-eval-logtriage](ai-eval-logtriage/) に明記）。

**統計手法の既知の限界**（＝プロなら突く点を、こちらから先に開示する）
- **percentile ブートストラップ**（roi_ci/mean_ci）は BCa 補正なし。右に歪んだ重裾（ROI等）では区間がやや狭く、名目90%が実被覆で数%下振れする（自前シミュレーションで G=30 のとき実測 ~83%）。＝下限が楽観側に寄るので、重裾データでの「頑健にプラス」は割り引いて読む。返り値に `ci_caveat` で明示。
- **confidence sequence の「anytime-valid（何度覗いてもOK）」保証は finite-sample モード限定**。既定（標本SD）は漸近版で、理論保証はつかない。`example_sequential.py` の「固定CIは偽陽性49%・CSは0%」は特定シミュレーション下の実測値であって、既定モードの数学的保証ではない。
- **PSR / DSR はリターン IID を前提**。系列相関があると有効Nが過大になり楽観化する（時系列データでは要注意）。
- 検出力式は「1標本／ペア差」用の概算。独立2群には `power_required_mean(..., two_sample=True)` が必要（＝混同すると必要Nを半分に過小評価する）。
- BCa ブートストラップ・FDR制御・連続値の t 近似は今後の課題。

## 構成

```
shikinseki/        パッケージ本体（4モジュール・pip install -e . で使える）
  edge_validator.py   ③検証（中核）   data_harvester.py  ①取得
  leak_guard.py       ②防御           action_gate.py     ④運用
tests/             回帰テスト62件（CIで自動実行）
examples は各層のディレクトリ（edge-validator/ 等）に、実適用例は ai-eval-sentiment/ に。
```

## 参考（手法の出典と、このツールの寄与）

個々の統計手法は既存研究に基づく：逆正規CDF近似＝Acklam のアルゴリズム／deflated・probabilistic Sharpe と PBO(CSCV)＝Bailey & López de Prado／anytime-valid CI＝Howard et al. の正規混合境界。
**新規なのは手法そのものではなく統合**——これらを「group/time/baseline/outcome」という1つのデータ契約に束ね、**前提の破れ（`degenerate`/`ci_caveat`/`heavy_tailed`）を返り値で警告して「黙って嘘をつかせない」**設計にしたこと。「公式を知っている」と「公式が嘘をつく条件を製品に組み込む」は別のスキルだと考えている。

**なぜ標準ライブラリ縛りか**：numpy/pandasを入れられない本番環境・エアギャップでも動くことに加え、
「ブラックボックスに頼らず、中身を自分で証明できる」ことを優先した。scipyがあるのにAcklam近似を
自前実装するリスクは承知の上で、教科書値との照合テスト（誤差~1e-9を回帰テストで固定）で担保している。

## English summary

**shikinseki** ("touchstone") is a zero-dependency Python toolkit that tells whether a quantitative
edge is **real, luck, or simply unmeasurable** — born from rigorously proving that my horse-racing
prediction AI could *not* beat the market (power analysis included).

Four stdlib-only layers: **data-harvester** (completeness gate), **leak-guard** (future-information
leak detection), **edge-validator** (group bootstrap CIs, power analysis, anytime-valid confidence
sequences, PBO/DSR overfitting diagnostics — the flagship), **action-gate** (deterministic action
generation with machine-checked invariants).

Case study 1: a "prompt improvement" that scored **+7.3pp** when self-graded turned into
**+0.0pp (Gemini), +1.8pp (GPT-4o-mini), and −9.1pp (Llama-3.2-3B)** when re-measured via
independent model APIs — the effect was model-specific and didn't even preserve its sign.

Case study 2 (real supercomputer logs, BGL): naive LLM prompting **lost to a one-line keyword
rule** on alert triage, while a domain-informed prompt **robustly beat the rule across all three
models** (+10 to +17pp, 90% CI excluding zero). Same engine, opposite verdict — you don't know
which case you're in until you measure.

```bash
pip install -e .                             # zero dependencies
python3 -m unittest discover tests           # 62 regression tests
python3 edge-validator/example_advanced.py   # self-verifying demo
```

## ライセンス

MIT License — [LICENSE](LICENSE) 参照。
