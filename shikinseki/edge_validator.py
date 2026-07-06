#!/usr/bin/env python3
"""
edge_validator.py — ドメイン非依存の「そのエッジは本物か、ただの分散か」判定エンジン。

競馬予想の検証プロジェクトで作った検証手法を、競馬から切り離して
汎用化したもの。「予想（signal）／結果（outcome）／超えるべき相手（baseline）」さえ
あれば、トレード・A/Bテスト・モデル予測・施策効果など、何にでも使える。

依存：標準ライブラリのみ（numpy/pandas不要＝どこでも動く）。

────────────────────────────────────────────────────────────
データ契約（1行＝1つの判断/観測）
  group   : str   この行が属するイベント（同一grupoは相関するので再標本化の単位）。例:レースID/実験ID/日付
  time    : str   時系列ホールドアウト分割用のタイムスタンプ（ISO文字列など、ソート可能なら何でも）
  baseline: float 超えるべき相手の予測（市場確率/対照群の率/コンセンサス）。0〜1の確率か単調スコア
  outcome : float 実際に起きたこと（1/0の的中、または数値リターン）
  cost    : float そのアクションのコスト/賭け金（ROI算出用・既定1）
  payoff  : float アクションした時の粗リターン（既定=outcome）。ROI=payoff/cost
  features: dict  残差スキャン用の任意の共変量（{特徴名: 値}）
────────────────────────────────────────────────────────────

5つの問いに答える：
  1. roi_ci()        … 価値（ROI）と、まぐれを除いた信頼区間（group単位ブートストラップ）
  2. power_required()… その差を「証明」するのに何件要るか（=測れるのか）
  3. residual_scan() … baselineを土台に、特徴量がそれを超えて結果を予測するか
  4. holdout_scan()  … 探索/検証に時系列分割し、out-of-sampleで再現したものだけ残す
  5. leak_check()    … signalが未来情報で汚染されていないかのヒューリスティック検査
"""
import math
import statistics, random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Record = Dict[str, Any]                              # データ契約の1行（group/time/baseline/outcome/…）
FeatureSpec = Tuple[str, Callable[[Record], bool]]   # スキャン用 (ラベル, 述語関数)

__all__ = [
    "roi_ci", "mean_ci", "confidence_sequence", "sequential_scan",
    "power_required", "power_required_mean", "residual_scan", "holdout_scan",
    "staged_scan", "multiple_comparison_note", "leak_check", "purged_kfold_indices",
    "sharpe", "probabilistic_sharpe_ratio", "deflated_sharpe_ratio", "pbo_cscv",
    "cohens_kappa", "decision_value_ab", "value_of_information", "value_of_more_data",
    "print_report", "MIN_GROUPS",
]

# 80%検出力・両側5%で必要Nを出すときの係数 ≈ (z_0.975 + z_0.80) = 1.95996 + 0.84162
_POWER_COEF = 2.80158


def _finite(x):
    """None/NaN/Inf を弾く（実データ由来の欠測・異常値が黙って統計に混入するのを防ぐ）。"""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ── 共通：group単位ブートストラップ（同一イベント内の相関を壊さない再標本化）──
def _bootstrap(groups, value_fn, iters=2000, seed=42):
    """groups: list[list[record]]（group単位の束）。value_fn(flat_records)->float。
    group単位で復元抽出して value_fn の分布を作り、90%区間を返す。"""
    rng = random.Random(seed)
    m = len(groups)
    out = []
    for _ in range(iters):
        sample = []
        for _ in range(m):
            sample.extend(groups[rng.randrange(m)])
        v = value_fn(sample)
        if v is not None:
            out.append(v)
    out.sort()
    if not out:
        return (None, None)
    return out[int(0.05 * len(out))], out[int(0.95 * len(out))]


def _by_group(records):
    g = defaultdict(list)
    for r in records:
        g[r.get("group", id(r))].append(r)
    return list(g.values())


MIN_GROUPS = 30  # これ未満はブートストラップCIの信頼性が低い（再標本化の空間が小さすぎる）


def _skewness(xs):
    """標本歪度。|skew|が大きい＝裾が重い＝平均のCLT近似が甘い（必要N過小評価）の合図。"""
    n = len(xs)
    if n < 3:
        return 0.0
    m = sum(xs) / n
    sd = statistics.pstdev(xs)
    if sd == 0:
        return 0.0
    return sum(((x - m) / sd) ** 3 for x in xs) / n


# ── 1. ROI と信頼区間 ──
def roi_ci(records: List[Record], iters: int = 2000, seed: int = 42) -> Optional[dict]:
    """acted行（cost>0）の ROI（払戻/賭け金×100）と group単位90%CI。
    CI下限>100かつgroup数>=30なら『頑健にプラス』。点推定だけ見ると分散に騙される。
    group数<30ではCIが退化しうるため robust_* は常にFalse（reliable=Falseで判定保留を明示）。"""
    acted = [r for r in records if r.get("cost", 1) > 0]
    if not acted:
        return None
    def roi_of(rs):
        c = sum(r.get("cost", 1) for r in rs)
        p = sum(r.get("payoff", r.get("outcome", 0)) for r in rs)
        return p / c * 100 if c else None
    point = roi_of(acted)
    groups = _by_group(acted)
    lo, hi = _bootstrap(groups, roi_of, iters, seed)
    hits = sum(1 for r in acted if r.get("payoff", r.get("outcome", 0)) > 0)
    # ⚠️ percentile直取り＝BCa補正なし。右に歪んだ重裾（ROI等）では区間がやや狭め＝下限が
    #    楽観側に寄る（名目90%が実被覆で数%下振れ）。BCa/basic補正は今後の課題（READMEの限界に明記）。
    reliable = len(groups) >= MIN_GROUPS
    # robust_* は reliable でゲート＝group不足でCIが点に退化しても「頑健」と偽らない
    # （confidence_sequence の degenerate と同じ思想）。境界は docstring 通り strict（>100）。
    return {"roi": point, "ci_low": lo, "ci_high": hi, "n": len(acted),
            "n_groups": len(groups), "reliable": reliable,
            "hits": hits, "robust_positive": (lo is not None and lo > 100 and reliable),
            "robust_negative": (hi is not None and hi < 100 and reliable),
            "ci_method": "percentile", "ci_caveat": "heavy_tailed_undercoverage"}


# ── 1b. 任意の値の平均と信頼区間（A/B差・効果量の汎用判定）──
def mean_ci(records: List[Record], value_key: str = "value", iters: int = 2000, seed: int = 42) -> Optional[dict]:
    """value_key の平均と group単位90%CI。ROIでない一般の量（正解率・A/B差・効果量）用。
    CIが0をまたがない（かつgroup数>=30）＝『頑健に非ゼロ』。ペア差を入れれば A/Bテストの判定になる。"""
    rs = [r for r in records if _finite(r.get(value_key))]  # None/NaN/Inf を除去
    if not rs:
        return None
    def mean_of(xs):
        vs = [x[value_key] for x in xs if _finite(x.get(value_key))]
        return sum(vs) / len(vs) if vs else None
    point = mean_of(rs)
    groups = _by_group(rs)
    lo, hi = _bootstrap(groups, mean_of, iters, seed)
    reliable = len(groups) >= MIN_GROUPS
    return {"mean": point, "ci_low": lo, "ci_high": hi, "n": len(rs),
            "n_groups": len(groups), "reliable": reliable,
            "robust_positive": (lo is not None and lo > 0 and reliable),
            "robust_negative": (hi is not None and hi < 0 and reliable),
            "ci_method": "percentile", "ci_caveat": "heavy_tailed_undercoverage"}


# ── 1d. 信頼区間列（逐次的に母数を増やしても妥当・覗き見OK）──
def confidence_sequence(values: Sequence[float], alpha: float = 0.10,
                        sigma: Optional[float] = None, rho: float = 0.5) -> Optional[dict]:
    """anytime-valid な信頼区間（Robbins 正規混合）。
    『データを足しながら何度も覗き、0を外れたら止める』をやっても、全体の誤り率を
    alpha 以下に保つ＝optional stopping（覗き見して都合よく止める）の罠を防ぐ。
    固定Nの roi_ci/mean_ci より広い＝"何度でも覗ける自由"の代償。
      values: 観測値の列。
      sigma : sub-Gaussianパラメータ。None＝標本SD（漸近CS・実用だがやや甘め）。
              有界データで finite-sample に厳密化したいなら sigma=(値域)/2 を渡す（保守的・誤り率を理論保証）。"""
    from math import log, sqrt
    given_sigma = sigma is not None  # σを明示→finite-sample保証 / None→標本SDで漸近のみ
    xs = [v for v in values if _finite(v)]
    n = len(xs)
    if n == 0:
        return None
    mean = sum(xs) / n
    raw_sigma = sigma if given_sigma else (statistics.pstdev(xs) if n > 1 else 0.0)
    # 退化ケース（n<2・全同値でσ≈0）は「判定不能」とする。ここで decided を返すと
    # データ不足なのに『頑健に非ゼロ』という偽陽性を出す＝検証ツールとして最悪の挙動。
    degenerate = (n < 2) or (raw_sigma <= 0)
    sigma_eff = raw_sigma if raw_sigma > 0 else 1e-9
    nr = n * rho * rho
    radius = sigma_eff * sqrt((2 * (nr + 1)) / (n * n * rho * rho) * log(sqrt(nr + 1) / alpha))
    lo, hi = mean - radius, mean + radius
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": n,
            "radius": radius, "sigma": sigma_eff, "alpha": alpha,
            "degenerate": degenerate,
            "validity": "finite-sample" if given_sigma else "asymptotic",
            "robust_positive": (lo > 0) and not degenerate,
            "robust_negative": (hi < 0) and not degenerate}


def sequential_scan(values: Sequence[float], alpha: float = 0.10, sigma: Optional[float] = None,
                    rho: float = 0.5, min_n: int = 20, step: int = 10) -> dict:
    """到着順に min_n から step ごとに CS を再計算し、初めて 0 を外した（=決着した）
    時点 crossed_at を返す。CS が誤り率を保証するので『増やしながら覗いて、外れたら止める』が安全。"""
    xs = [v for v in values if _finite(v)]
    checkpoints, crossed_at = [], None
    n = max(1, min_n)  # min_n<=0 だと空列のCSがNoneを返しクラッシュする
    while n <= len(xs):
        cs = confidence_sequence(xs[:n], alpha=alpha, sigma=sigma, rho=rho)
        decided = cs["robust_positive"] or cs["robust_negative"]
        checkpoints.append({"n": n, "mean": cs["mean"],
                            "ci_low": cs["ci_low"], "ci_high": cs["ci_high"],
                            "decided": decided})
        if decided and crossed_at is None:
            crossed_at = n
        n += step
    return {"crossed_at": crossed_at, "checkpoints": checkpoints}


# ── 2. 検出力：その差を証明するのに何件要るか ──
def power_required(records: List[Record], target_roi: float = 110.0,
                   measurable_n: int = 5000) -> Optional[dict]:
    """target_roi%（＝検出したい最小効果・*事前指定*）を80%検出力・両側5%で示すのに必要なN。
    ⚠️ target_roi に観測ROIを入れる post-hoc power は必要Nを過小評価する→使わない。
    歪度が大きい(裾が重い)とCLT近似が甘く実際はより多く要る→heavy_tailedで警告。
    ⚠️ 前提＝costがほぼ均一。不均一だと roi_ci のROI（Σpayoff/Σcost＝コスト加重比）と
    ここで使う等重み平均（per-100純益）の推定対象がずれる。sdが0（全行同値）は
    データ異常の合図なので None＝判定不能を返す。"""
    acted = [r for r in records if r.get("cost", 1) > 0]
    if len(acted) < 5:
        return None
    nets = []  # 賭け金1単位あたり純益
    for r in acted:
        c = r.get("cost", 1)
        p = r.get("payoff", r.get("outcome", 0))
        nets.append((p - c) / (c / 100.0) if c else 0.0)  # 100円基準の純益
    sd = statistics.stdev(nets)  # 標本SD（n-1）＝小Nの下方バイアスを避ける
    if sd == 0:
        return None  # 分散ゼロ＝「0件で証明可能」と偽らない
    edge = target_roi - 100.0  # 指定した最小効果（観測値でなく事前指定）
    se_per_n = lambda N: sd / (N ** 0.5)
    need = (_POWER_COEF * sd / edge) ** 2 if edge else float("inf")  # 80%検出力・両側5%の概算
    sk = _skewness(nets)
    return {"sd_per_100": sd, "se_at_n": se_per_n(len(acted)),
            "n_now": len(acted), "need_n": need,
            "measurable": need <= measurable_n, "measurable_n": measurable_n,
            "skewness": sk, "heavy_tailed": abs(sk) > 2}


def power_required_mean(values: Sequence[float], mde: float, measurable_n: int = 5000,
                        two_sample: bool = False) -> Optional[dict]:
    """平均（A/B差・効果量）を検出するのに要するN。
    mde＝『検出したい最小効果（絶対値）』を*外から指定*する。観測差を使うpost-hoc power
    は過小評価になるため避ける。歪度が大きいと実際はより多く要る→heavy_tailed。
    ⚠️ two_sample: values がペア差（同一対象をA/B両方に通した差）なら False（＝1標本）。
       独立2群の生データを入れるなら True（分散が2倍要り必要Nも約2倍）＝混同すると半分に過小評価する。"""
    xs = [v for v in values if _finite(v)]
    if len(xs) < 2 or not mde:
        return None
    sd = statistics.stdev(xs)  # 標本SD（n-1）
    if sd == 0:
        return None  # 分散ゼロ＝データ異常の合図。「0件で証明可能」と偽らない
    factor = 2.0 if two_sample else 1.0
    need = factor * (_POWER_COEF * sd / abs(mde)) ** 2  # 80%検出力・両側5%の概算
    sk = _skewness(xs)
    return {"sd": sd, "mde": mde, "need_n": need, "n_now": len(xs),
            "two_sample": two_sample, "measurable": need <= measurable_n,
            "skewness": sk, "heavy_tailed": abs(sk) > 2}


# ── ベースライン較正（baselineを束ねて実測率を出す＝市場の予測）──
def _calibrate(records, nbins=15, value_key="outcome"):
    """呼び出し側の records は汚さない（シャローコピーに _bin/_resid を付けて返す）。
    返り値=(binrate, calibrated_copies)。純粋関数＝同じ入力を別の分析に再利用しても安全。"""
    rs = [dict(r) for r in records if r.get("baseline") is not None]  # copy＝in-placeしない
    rs.sort(key=lambda r: r["baseline"])
    n = len(rs)
    if n == 0:
        return {}, []
    binrate = {}
    tmp = defaultdict(list)
    for i, r in enumerate(rs):
        b = min(nbins - 1, i * nbins // n)
        r["_bin"] = b
        tmp[b].append(r.get(value_key, 0))
    for b, vs in tmp.items():
        binrate[b] = statistics.mean(vs)
    for r in rs:
        r["_resid"] = r.get(value_key, 0) - binrate[r["_bin"]]
    return binrate, rs


# ── フォールド安全な較正（探索で作り検証へ適用＝ホールドアウト境界をまたぐリーク防止）──
def _fit_calibration(records, nbins=15, value_key="outcome"):
    """records だけで baseline→outcome の較正を作る。返り値=(cuts, binrate)。
    cuts[b]＝そのbinの最大baseline（検証データの割り当て用しきい値）。"""
    rs = sorted((r for r in records if r.get("baseline") is not None),
                key=lambda r: r["baseline"])
    n = len(rs)
    if n == 0:
        return [], {}
    tmp = defaultdict(list)
    for i, r in enumerate(rs):
        tmp[min(nbins - 1, i * nbins // n)].append((r["baseline"], r.get(value_key, 0)))
    cuts, binrate = [], {}
    # binrate のキーは cuts の位置（0..len-1）で揃える。n<nbins だと bin番号が飛ぶため、
    # bin番号をキーにすると _apply_calibration の参照がずれる（旧バグ・回帰テストあり）。
    for j, b in enumerate(sorted(tmp)):
        binrate[j] = statistics.mean(v for _, v in tmp[b])
        cuts.append(max(bl for bl, _ in tmp[b]))
    return cuts, binrate


def _apply_calibration(records, cuts, binrate, value_key="outcome"):
    """fit済みの (cuts, binrate) を適用し、_resid を付けたシャローコピーの list を返す。
    呼び出し側の records は汚さない（純粋関数）。baseline欠損の行は除外。"""
    if not cuts:
        return []
    last = len(cuts) - 1
    out = []
    for r in records:
        if r.get("baseline") is None:
            continue
        rr = dict(r)  # copy＝in-placeしない
        b = next((i for i, c in enumerate(cuts) if r["baseline"] <= c), last)
        rr["_resid"] = r.get(value_key, 0) - binrate[b]
        out.append(rr)
    return out


def _resid_stat(subset, iters=1000, seed=42):
    """残差平均の有意性。SEは『group単位ブートストラップ』で出す＝同一イベント内の相関を
    壊さない（roi_ciと同じ作法）。naïveな sd/√n は実効Nを過大評価し、zを膨らませて偽陽性を生む。
    ⚠️ z は正規近似で解釈する。group数が小さいと t寄りの重い裾になり名目誤り率が甘くなる。"""
    rs = [r for r in subset if "_resid" in r]
    groups = _by_group(rs)
    # ゲートは group数（＝再標本化の実効N）。レコード数で見ると「2群40行」が素通りし、
    # ほぼ無意味なクラスタブートストラップSEの上に z 判定が載ってしまう（roi_ci/mean_ci と同じ基準）。
    if len(groups) < MIN_GROUPS:
        return None
    m = statistics.mean(r["_resid"] for r in rs)
    rng = random.Random(seed)
    G = len(groups)
    boot = []
    for _ in range(iters):
        acc = cnt = 0
        for _ in range(G):
            for x in groups[rng.randrange(G)]:
                acc += x["_resid"]; cnt += 1
        if cnt:
            boot.append(acc / cnt)
    se = statistics.pstdev(boot) if len(boot) > 1 else 0.0
    return {"resid": m, "se": se, "z": (m / se if se else 0),
            "n": len(rs), "n_groups": G}


# ── 3. 残差スキャン：特徴量はbaselineを超えて予測するか ──
def residual_scan(records: List[Record], feature_specs: List[FeatureSpec], nbins: int = 15,
                  value_key: str = "outcome", seed: int = 42) -> List[tuple]:
    """feature_specs: list[(label, predicate)]。
    baseline較正後の残差を、各特徴量サブセットで平均→z検定。
    z>=2.5でやっと注目（同イベント相関で実効Nは小さめ＝高めの閾値）。"""
    _, cal = _calibrate(records, nbins, value_key)  # cal＝汚さないコピー
    rows = []
    for label, pred in feature_specs:
        s = _resid_stat([r for r in cal if pred(r)], seed=seed)
        rows.append((label, s))
    return rows


# ── 4. ホールドアウト：探索/検証で再現したものだけ残す ──
def holdout_scan(records: List[Record], feature_specs: List[FeatureSpec], split_time: Any = None,
                 nbins: int = 15, value_key: str = "outcome", seed: int = 42) -> dict:
    """時系列で前半(探索)/後半(検証)に分割。探索z>=2 かつ 検証で同符号かつz>=1を
    満たしたものだけ survivors。多重比較の罠（試行数だけ偽陽性が出る）を潰す。
    較正は探索データのみで作り検証へ適用＝境界をまたぐ較正リークを防ぐ。"""
    times = sorted(set(r["time"] for r in records if r.get("time") is not None))
    if split_time is None and times:
        split_time = times[len(times) // 2]
    disc0 = [r for r in records if r.get("time") is not None and r["time"] < split_time]
    val0 = [r for r in records if r.get("time") is not None and r["time"] >= split_time]
    # 探索データだけで較正を作り、探索・検証の両方に適用（検証情報を一切混ぜない）
    cuts, binrate = _fit_calibration(disc0, nbins, value_key)
    disc = _apply_calibration(disc0, cuts, binrate, value_key)  # _resid付きコピー
    val = _apply_calibration(val0, cuts, binrate, value_key)
    table, survivors = [], []
    for label, pred in feature_specs:
        d = _resid_stat([r for r in disc if pred(r)], seed=seed)
        v = _resid_stat([r for r in val if pred(r)], seed=seed)
        repl = bool(d and v and abs(d["z"]) >= 2 and (d["resid"] > 0) == (v["resid"] > 0)
                    and abs(v["resid"]) >= 0.02 and abs(v["z"]) >= 1)
        if repl:
            survivors.append(label)
        table.append((label, d, v, repl))
    return {"split_time": split_time, "n_disc": len(disc0), "n_val": len(val0),
            "table": table, "survivors": survivors, "n_tests": len(feature_specs)}


def multiple_comparison_note(n_tests: int, z: float = 2.5) -> dict:
    """n_tests回の検定で、無エッジでも偶然 |z|>=z が出る期待個数と、
    全体を5%に抑えるŠidák補正後の『1検定あたり』有意水準。"""
    from math import erfc, sqrt
    p_two = erfc(z / sqrt(2))  # 両側p値（z=2.5 で ≈0.012）
    alpha_corr = 1 - 0.95 ** (1 / n_tests) if n_tests else 0.05
    return {"n_tests": n_tests, "z": z, "p_two_sided": p_two,
            "expected_false": n_tests * p_two,
            "alpha_corrected_5pct": alpha_corr}


# ── 5. リーク検査：signalが未来情報で汚染されてないか ──
def leak_check(records: List[Record], entity_key: str = "entity",
               time_key: str = "time") -> List[dict]:
    """同一エンティティ（馬/銘柄/ユーザー等）が、ある時点より後に『良い結果で再登場』する場合、
    過去レコードのsignal付けに未来の知識が混入し得る（=リーク）候補としてフラグ。
    完全自動判定は不可。H/M候補を出して人手レビューを絞る簡易版
    （H/M/L?/L の4分級・昇格判定つきの完全版は leak_guard.reappearance_flags）。"""
    seen = defaultdict(list)
    for r in records:
        ent = r.get(entity_key)
        if ent is None:
            continue
        seen[ent].append(r)
    flags = []
    for ent, rs in seen.items():
        rs = sorted(rs, key=lambda r: r.get(time_key, ""))
        if len(rs) >= 2:
            later_good = any(x.get("outcome", 0) > 0 for x in rs[1:])
            level = "H" if later_good else "M"
            flags.append({"entity": ent, "appearances": len(rs), "level": level})
    flags.sort(key=lambda f: (f["level"] != "H", -f["appearances"]))
    return flags


# ════════════════════════════════════════════════════════════
#  拡張（2026-06-16）：過学習診断 / 選択的推論3段化 / 測定一致 / 意思決定
# ════════════════════════════════════════════════════════════
from math import erf as _erf, exp as _exp, log as _ln, pi as _pi, sqrt as _rt


def _norm_cdf(x):
    return 0.5 * (1 + _erf(x / _rt(2)))


def _norm_pdf(x):
    return _exp(-x * x / 2) / _rt(2 * _pi)


def _norm_ppf(p):
    """逆正規CDF（Peter Acklam のアルゴリズム／相対誤差~1.1e-9）。"""
    if p <= 0:
        return -float("inf")
    if p >= 1:
        return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = _rt(-2 * _ln(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = _rt(-2 * _ln(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _kurtosis(xs):
    n = len(xs)
    if n < 4:
        return 3.0
    m = sum(xs) / n
    sd = statistics.pstdev(xs)
    if sd == 0:
        return 3.0
    return sum(((x - m) / sd) ** 4 for x in xs) / n  # Pearson（正規=3）


# ── A. 過学習診断（クオンツの成熟ツール：手探りの多重比較対策の上位互換）──
def sharpe(values: Sequence[float]) -> Optional[float]:
    """標本SD（n-1）ベースのSharpe＝標準的定義。⚠️前提＝リターンがIID。
    系列相関があると有効Nが過大になりPSR/DSRが楽観化する（時系列データでは要注意）。"""
    xs = [v for v in values if _finite(v)]
    if len(xs) < 2:
        return None
    sd = statistics.stdev(xs)
    return (sum(xs) / len(xs)) / sd if sd else 0.0


def probabilistic_sharpe_ratio(values: Sequence[float], sr_benchmark: float = 0.0) -> Optional[dict]:
    """PSR：真のSharpeが sr_benchmark を超える確率（歪度・尖度・標本長で補正）。
    n<4 では尖度が正規(3.0)扱い＝高次補正は事実上かからない（黙って補正した顔をしない）。"""
    xs = [v for v in values if _finite(v)]
    n = len(xs)
    if n < 3:
        return None
    sr = sharpe(xs); g3 = _skewness(xs); g4 = _kurtosis(xs)
    denom = _rt(max(1e-12, 1 - g3 * sr + ((g4 - 1) / 4.0) * sr * sr))
    z = ((sr - sr_benchmark) * _rt(n - 1)) / denom
    return {"sharpe": sr, "n": n, "skew": g3, "kurtosis": g4,
            "psr": _norm_cdf(z), "sr_benchmark": sr_benchmark}


def deflated_sharpe_ratio(values: Sequence[float],
                          trial_sharpes: Sequence[Optional[float]]) -> Optional[dict]:
    """DSR：多数の戦略を試した中での最良が本物の確率（Bailey & López de Prado 2014）。
    trial_sharpes＝試した全戦略のSharpe一覧（＝選択バイアスの源）。試行が多いほど厳しく割り引く。"""
    xs = [v for v in values if _finite(v)]
    srs = [s for s in trial_sharpes if _finite(s)]
    N = len(srs)
    if len(xs) < 3 or N < 2:
        return None
    var_sr = statistics.pvariance(srs)
    EM = 0.5772156649015329  # Euler–Mascheroni
    sr0 = _rt(var_sr) * ((1 - EM) * _norm_ppf(1 - 1.0 / N) + EM * _norm_ppf(1 - 1.0 / (N * _exp(1))))
    psr = probabilistic_sharpe_ratio(xs, sr_benchmark=sr0)
    return {"sr0_expected_max_under_null": sr0, "n_trials": N,
            "dsr": psr["psr"] if psr else None,
            "observed_sharpe": psr["sharpe"] if psr else None}


def pbo_cscv(perf_matrix: List[List[float]], n_blocks: int = 10) -> Optional[dict]:
    """Probability of Backtest Overfitting（CSCV／Bailey & López de Prado 2015）。
    perf_matrix: list[strategy]、各々 list[期間ごとの成績]（全戦略同じ長さ）。
    pbo＝『in-sample最良がout-of-sampleで中央値以下に落ちる確率』。高いほど過学習。
    OOS成績の同値（タイ）は最下位rank＝悲観側に倒す（保守的＝過学習を見逃さない方向）。"""
    from itertools import combinations
    S = len(perf_matrix)
    if S < 2:
        return None
    T = min(len(r) for r in perf_matrix)
    M = [r[:T] for r in perf_matrix]
    nb = n_blocks if n_blocks % 2 == 0 else n_blocks - 1
    nb = max(2, min(nb, T))
    nb -= nb % 2  # min(nb,T) 後も偶数を保証（IS/OOS同数ブロック＝CSCVの定義）
    nb = max(2, nb)
    bnd = [round(i * T / nb) for i in range(nb + 1)]
    blocks = [list(range(bnd[i], bnd[i + 1])) for i in range(nb)]
    lams, below = [], 0
    for combo in combinations(range(nb), nb // 2):
        IS = set(combo)
        is_cols = [c for b in IS for c in blocks[b]]
        oos_cols = [c for b in range(nb) if b not in IS for c in blocks[b]]
        if not is_cols or not oos_cols:
            continue
        is_perf = [sum(M[s][c] for c in is_cols) / len(is_cols) for s in range(S)]
        oos_perf = [sum(M[s][c] for c in oos_cols) / len(oos_cols) for s in range(S)]
        best = max(range(S), key=lambda s: is_perf[s])
        rank = 1 + sum(1 for s in range(S) if oos_perf[s] < oos_perf[best])  # 1=最低..S=最高
        w = rank / (S + 1.0)
        lam = _ln(w / (1 - w)) if 0 < w < 1 else 0.0
        lams.append(lam)
        if lam <= 0:
            below += 1
    if not lams:
        return None
    return {"pbo": below / len(lams), "n_combos": len(lams),
            "median_logit": sorted(lams)[len(lams) // 2],
            "n_strategies": S, "n_blocks": nb}


def purged_kfold_indices(times: Sequence[Any], n_splits: int = 5,
                         embargo_frac: float = 0.01) -> List[Tuple[List[int], List[int]]]:
    """時系列の purged＋embargo K-fold。times＝各サンプルの時刻（ソート可能）。
    返り値 list[(train_idx, test_idx)]：testの時間範囲＋embargo分をtrainから除外（情報漏れ防止）。"""
    n = len(times)
    if n < n_splits or n_splits < 2:
        return []
    idx = sorted(range(n), key=lambda i: times[i])
    emb = max(0, int(n * embargo_frac))
    bnd = [round(k * n / n_splits) for k in range(n_splits + 1)]
    folds = []
    for k in range(n_splits):
        test = idx[bnd[k]:bnd[k + 1]]
        if not test:
            continue
        t_lo, t_hi = times[test[0]], times[test[-1]]
        purge = set(idx[bnd[k]:min(n, bnd[k + 1] + emb)])
        train = [i for i in idx if i not in purge and not (t_lo <= times[i] <= t_hi)]
        folds.append((train, test))
    return folds


# ── B. 選択的推論：探索→検証→確認の3段化（勝者の呪いを構造で緩和）──
def staged_scan(records: List[Record], feature_specs: List[FeatureSpec], estimand: str = "",
                nbins: int = 15, value_key: str = "outcome", seed: int = 42) -> dict:
    """時系列を3分割（探索/検証/確認）。較正は探索のみ→全体に適用。
    survivor＝探索|z|>=2 → 検証 同符号|z|>=1 → 確認 同符号|z|>=1（確認集合は選択に未使用）。
    estimand＝『何を推定しているか』の宣言を要求（予測残差か因果効果かの混同防止）。"""
    note = "" if estimand else "⚠️ estimand未宣言：何を推定しているか一文で明記を推奨"
    times = sorted(set(r["time"] for r in records if r.get("time") is not None))
    if len(times) < 3:
        return {"estimand": estimand, "note": "時点が3未満で3分割不可",
                "confirmed": [], "table": []}
    t1, t2 = times[len(times) // 3], times[2 * len(times) // 3]
    disc0 = [r for r in records if r.get("time") is not None and r["time"] < t1]
    val0 = [r for r in records if r.get("time") is not None and t1 <= r["time"] < t2]
    conf0 = [r for r in records if r.get("time") is not None and r["time"] >= t2]
    cuts, binrate = _fit_calibration(disc0, nbins, value_key)
    disc = _apply_calibration(disc0, cuts, binrate, value_key)  # _resid付きコピー（元は汚さない）
    val = _apply_calibration(val0, cuts, binrate, value_key)
    conf = _apply_calibration(conf0, cuts, binrate, value_key)
    table, confirmed = [], []
    for label, pred in feature_specs:
        dd = _resid_stat([r for r in disc if pred(r)], seed=seed)
        vv = _resid_stat([r for r in val if pred(r)], seed=seed)
        cc = _resid_stat([r for r in conf if pred(r)], seed=seed)
        ok = bool(dd and vv and cc and abs(dd["z"]) >= 2
                  and (dd["resid"] > 0) == (vv["resid"] > 0) and abs(vv["z"]) >= 1
                  and (dd["resid"] > 0) == (cc["resid"] > 0) and abs(cc["z"]) >= 1)
        if ok:
            confirmed.append(label)
        table.append((label, dd, vv, cc, ok))
    return {"estimand": estimand, "note": note, "t1": t1, "t2": t2,
            "n_disc": len(disc0), "n_val": len(val0), "n_conf": len(conf0),
            "table": table, "confirmed": confirmed, "n_tests": len(feature_specs)}


# ── C. 測定の信頼性：アノテータ間一致（ラベル品質の前提チェック）──
def cohens_kappa(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> Optional[dict]:
    """2アノテータの名義一致（Cohen's κ）。κが低い＝『正解』自体が不安定＝精度の上限。
    （≥3名・欠測ありは Fleiss/Krippendorff が一般化版＝Roadmap）。"""
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return None
    cats = set(a for a, _ in pairs) | set(b for _, b in pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum((sum(1 for a, _ in pairs if a == cat) / n) *
             (sum(1 for _, b in pairs if b == cat) / n) for cat in cats)
    if pe >= 1:
        # 両者が単一カテゴリのみ＝0/0の不定形。「一致の情報量ゼロ」をκ=1と偽らず判定不能を返す
        return {"kappa": None, "percent_agreement": po, "n": n,
                "n_categories": len(cats), "degenerate": True}
    return {"kappa": (po - pe) / (1 - pe), "percent_agreement": po, "n": n,
            "n_categories": len(cats), "degenerate": False}


# ── D. 意思決定層：「本物か」→「行動すべきか／集める価値があるか」──
def _unit_normal_loss(d):
    return _norm_pdf(d) - d * (1 - _norm_cdf(d))


def decision_value_ab(effect_mean: float, effect_lo: float, value_per_unit: float,
                      switch_cost: float = 0.0) -> dict:
    """Bを採用すべきか。効果×価値−切替コスト。期待値と悲観値(CI下限)の両方で評価。"""
    exp_net = effect_mean * value_per_unit - switch_cost
    worst_net = effect_lo * value_per_unit - switch_cost
    if worst_net > 0:
        rec = "採用（悲観ケースでも黒字＝頑健）"
    elif exp_net > 0:
        rec = "条件付き採用（期待は黒字だが下振れリスクあり）"
    else:
        rec = "見送り（期待でも黒字にならない）"
    return {"expected_net": exp_net, "worst_net": worst_net, "recommendation": rec}


def value_of_information(effect_mean: float, se: float, value_per_unit: float) -> dict:
    """EVPI＝完全情報の価値＝『確実性にいくらまで払う価値があるか』の上限。
    線形2択（効果>0なら採用）の期待機会損失。"""
    if se <= 0:
        return {"evpi": 0.0, "d": float("inf")}
    d = abs(effect_mean) / se
    return {"evpi": value_per_unit * se * _unit_normal_loss(d), "d": d}


def value_of_more_data(effect_mean: float, se: float, value_per_unit: float,
                       sampling_cost: float) -> dict:
    """追加データ収集の価値の上限(EVPI)とコストを比較。
    EVPI≤コストなら『集めても割に合わない』が確実に言える（EVPIは上限ゆえ強い否定）。"""
    evpi = value_of_information(effect_mean, se, value_per_unit)["evpi"]
    worth = evpi > sampling_cost
    return {"evpi": evpi, "sampling_cost": sampling_cost, "worth_collecting": worth,
            "verdict": ("集める価値あり（上限EVPI>コスト）" if worth
                        else "集めても割に合わない（上限EVPI≤コスト）")}


# ── 整形出力ヘルパ ──
def print_report(title: str, **sections: Any) -> None:
    """title と名前付きセクションを罫線つきで標準出力する整形ヘルパ。"""
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    for name, val in sections.items():
        print(f"\n[{name}]\n{val}")


if __name__ == "__main__":
    print(__doc__)
