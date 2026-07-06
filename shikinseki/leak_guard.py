#!/usr/bin/env python3
"""
leak_guard.py — 先読み（lookahead/リーク）汚染の検出エンジン。

競馬プロジェクトのリーク検出（古いレースの馬が後で昇格再登場→勝者逆算可をH/M/Lでフラグ）を
競馬から切り離して汎用化したもの。data-harvester（取得）と edge-validator（検証）の間＝**防御層**。

リークとは：予測/採点する時点Tのラベルやsignalが、**T以降にしか得られない情報で汚染**されること。
データが「完全」でも、リークがあれば検証は嘘になる（in-sampleで光り、実戦＝フォワードで必ず剥がれる）。

4つの検査（データ契約: group/time/entity/baseline/outcome/features を流用）：
  1. reappearance_flags() … 同じentityが後の時点で"昇格再登場"→過去の結果が逆算可（H/M/L）
  2. split_contamination() … 同じgroupが探索/検証の境界をまたぐ（ホールドアウトが壊れる）
  3. target_leak_scan()  … outcomeを出来すぎなほど当てる特徴量（答えそのものが混入）
  4. duplicate_check()   … 同じ(entity,time)の重複

依存：標準ライブラリのみ。
"""
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["reappearance_flags", "split_contamination", "target_leak_scan",
           "duplicate_check", "run_all"]


# ── 1. 再登場リーク（競馬のリーク検出を一般化）──
def reappearance_flags(records: List[dict], level_key: Optional[str] = None,
                       top_k: int = 3) -> Dict[Any, str]:
    """各group（イベント）を、所属entityが「後の時点で再登場」するかでH/M/Lフラグ。
    level_key を渡すと「昇格（後の時点でlevelが上昇）」を判定でき精度が上がる。
      H  = そのgroupの最有力(baseline最小=本命)が後で昇格再登場（=結果が逆算可）
      M  = 上位top_kが昇格 or 2頭以上が昇格
      L? = 2件以上が（昇格でなく）単に再登場（履歴で結果露出の可能性→一瞥）
      L  = 安全
    返り値: {group: 'H'|'M'|'L?'|'L'}。H/Mを人手レビューに回し、Lは安全に使える。"""
    # entity → [(time, level)] の索引
    idx = defaultdict(list)
    for r in records:
        e = r.get("entity")
        if e is None or r.get("time") is None:
            continue
        idx[e].append((r["time"], r.get(level_key) if level_key else None))

    # group → そのgroupのrecords
    groups = defaultdict(list)
    for r in records:
        groups[r.get("group")].append(r)

    flags = {}
    for g, rs in groups.items():
        t_g = min((r["time"] for r in rs if r.get("time") is not None), default=None)
        if t_g is None:
            flags[g] = "L"
            continue
        # baseline昇順=強い順。上位top_kを本命群とする
        ranked = sorted([r for r in rs if r.get("baseline") is not None],
                        key=lambda r: r["baseline"])
        topfav = ranked[0]["entity"] if ranked else None
        top_entities = set(r["entity"] for r in ranked[:top_k])

        reappear = 0
        promoted = 0
        topfav_promoted = False
        topk_promoted = 0
        for r in rs:
            e = r.get("entity")
            lvl = r.get(level_key) if level_key else None
            later = [(t, l) for (t, l) in idx.get(e, []) if t > t_g]
            if not later:
                continue
            reappear += 1
            is_promo = (level_key is not None and lvl is not None
                        and any(l is not None and l > lvl for (t, l) in later))
            # level_keyが無い場合は「後に再登場した事実」自体を弱いリスクとして扱う
            if level_key is None:
                is_promo = True
            if is_promo:
                promoted += 1
                if e == topfav:
                    topfav_promoted = True
                if e in top_entities:
                    topk_promoted += 1

        if topfav_promoted:
            flags[g] = "H"
        elif topk_promoted >= 1 or promoted >= 2:
            flags[g] = "M"
        elif reappear >= 2:
            flags[g] = "L?"
        else:
            flags[g] = "L"
    return flags


# ── 2. 探索/検証 境界またぎ（ホールドアウト汚染）──
def split_contamination(records: List[dict], split_time: Any) -> List[Any]:
    """同じgroupが split_time の前後両方に出ると、ホールドアウト検証が汚染される。
    またいでいるgroupの一覧を返す。"""
    sides = defaultdict(set)
    for r in records:
        t = r.get("time")
        if t is None:
            continue
        sides[r.get("group")].add("before" if t < split_time else "after")
    return [g for g, s in sides.items() if len(s) > 1]


# ── 3. ターゲットリーク（出来すぎる特徴量＝答えの混入）──
def target_leak_scan(records: List[dict], feature_keys: Sequence[str],
                     thresh: float = 0.95) -> List[dict]:
    """各特徴量について、上位群と下位群のoutcome率の差（分離度）を測る。
    分離がほぼ完全（=単独でoutcomeをほぼ言い当てる）なら、答えが混入したリーク候補。"""
    flags = []
    for f in feature_keys:
        vals = [(r.get(f), r.get("outcome")) for r in records
                if isinstance(r.get(f), (int, float)) and r.get("outcome") is not None]
        if len(vals) < 30:
            continue
        vals.sort(key=lambda x: x[0])
        q = len(vals) // 4
        low = [o for _, o in vals[:q]]
        high = [o for _, o in vals[-q:]]
        if not low or not high:
            continue
        sep = abs(statistics.mean(high) - statistics.mean(low))
        # outcomeを0/1とみなした分離度。1に近い=単独でほぼ完全予測=怪しい
        if sep >= thresh:
            flags.append({"feature": f, "separation": round(sep, 3),
                          "low_rate": round(statistics.mean(low), 3),
                          "high_rate": round(statistics.mean(high), 3)})
    return flags


# ── 4. 重複 ──
def duplicate_check(records: List[dict]) -> List[dict]:
    """同一(entity,time)の二重登録を検出して一覧を返す。"""
    seen = defaultdict(int)
    for r in records:
        seen[(r.get("entity"), r.get("time"))] += 1
    return [{"entity": e, "time": t, "count": c} for (e, t), c in seen.items() if c > 1 and e is not None]


# ── まとめ実行 ──
def run_all(records: List[dict], level_key: Optional[str] = None, split_time: Any = None,
            feature_keys: Optional[Sequence[str]] = None) -> dict:
    """4検査（再登場リーク/境界またぎ/ターゲットリーク/重複）を一括実行し結果を表示。"""
    print("="*60 + "\nリーク防御チェック\n" + "="*60)
    rf = reappearance_flags(records, level_key=level_key)
    from collections import Counter
    c = Counter(rf.values())
    print(f"\n[1] 再登場リーク: H={c['H']} M={c['M']} L?={c['L?']} L={c['L']}")
    hi = [g for g, v in rf.items() if v in ("H", "M")]
    if hi:
        print(f"    → 封印/除外を人手確認すべきgroup: {hi[:10]}")
    if split_time is not None:
        sc = split_contamination(records, split_time)
        print(f"\n[2] 探索/検証 境界またぎ: {len(sc)}group" + (f" {sc[:10]}" if sc else " （なし=クリーン）"))
    if feature_keys:
        tl = target_leak_scan(records, feature_keys)
        print(f"\n[3] ターゲットリーク（出来すぎる特徴量）: {len(tl)}件")
        for t in tl:
            print(f"    🔴 {t['feature']}: 分離度{t['separation']}（下位{t['low_rate']}↔上位{t['high_rate']}）=答え混入の疑い")
    dup = duplicate_check(records)
    print(f"\n[4] 重複(entity,time): {len(dup)}件" + (f" {dup[:5]}" if dup else " （なし）"))
    clean = (c['H'] == 0 and c['M'] == 0 and (split_time is None or not split_contamination(records, split_time))
             and (not feature_keys or not target_leak_scan(records, feature_keys)) and not dup)
    print("\n防御: " + ("✅クリーン（検証GO）" if clean else "⚠️リスクあり（H/M封印・特徴量除外・分割修正を）"))
    return {"reappearance": rf, "clean": clean}


if __name__ == "__main__":
    print(__doc__)
