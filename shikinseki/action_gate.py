#!/usr/bin/env python3
"""
action_gate.py — 検証済みの方針を「人為ミスなく実行」するエンジン。

競馬プロジェクトで作った「買い目を決定論生成し（手組み禁止）、機械ゲートで
買い忘れ・重複・予算超過を弾く」仕組みを、競馬から切り離して汎用化したもの。
4層スタックの最後＝**運用層**（検証済みの手順を、毎回同じく・ミスなく回す）。

中心思想：**人間が手で組むのをやめる。**
  方針（policy）＋現在の状態（state）→ アクションを決定論的にコード生成（同じ入力→必ず同じ出力）。
  そして実行する前に、機械ゲート（invariants）で不変条件を検査し、違反があれば止める。
  「買い忘れ」「重複」「予算超過」「矛盾」を、注意力でなく構造で防ぐ。

3つの責務：
  1. generate()  … policy + state → アクションを決定論生成（順序非依存・重複排除・再現可能）
  2. check()     … 生成物に機械ゲート（不変条件）を適用。errorは実行をブロック
  3. run()       … generate→check→ゲート通過時のみ commit（実行可能アクションを返す）

依存：標準ライブラリのみ。
"""
import json
from collections import defaultdict
from typing import Any, Callable, List, Sequence, Tuple

__all__ = ["generate", "check", "run", "inv_no_duplicate", "inv_budget",
           "inv_required_fields", "inv_not_contradict", "inv_expected_count"]


def _canon(action: dict) -> str:
    """アクションの正規化キー（決定論的な重複排除・ソート用）。"""
    return json.dumps(action, sort_keys=True, ensure_ascii=False)


# ── 1. 決定論生成（手組み禁止）──
def generate(policy: List[tuple], state: dict) -> Tuple[List[dict], dict]:
    """policy: list[(rule_name, condition_fn, action_fn)]。
    condition_fn(state)->bool が真のルールが action_fn(state) を寄与（dict か dictのlist）。
    同じstateなら必ず同じアクション集合を返す（ルール順に依存しない＝正規化キーでソート・重複排除）。
    返り値: (actions, audit)。audit: {canonキー: [そのアクションを生んだルール名...]}。"""
    bag = {}
    audit = defaultdict(list)
    for rule_name, cond, act in policy:
        try:
            if not cond(state):
                continue
        except Exception as e:  # noqa: BLE001
            audit["_ERROR"].append(f"{rule_name}: condition例外 {e}")
            continue
        try:
            produced = act(state)
        except Exception as e:  # noqa: BLE001 — 生成も止めず監査に残す（condと対称）
            audit["_ERROR"].append(f"{rule_name}: action例外 {e}")
            continue
        if produced is None:
            continue
        if isinstance(produced, dict):
            produced = [produced]
        for a in produced:
            k = _canon(a)
            bag[k] = a
            audit[k].append(rule_name)
    # 決定論：正規化キーでソートして返す
    actions = [bag[k] for k in sorted(bag.keys())]
    return actions, dict(audit)


# ── 2. 機械ゲート（不変条件・実行前の最後の砦）──
def check(actions: List[dict], state: dict, invariants: List[tuple]) -> dict:
    """invariants: list[(name, check_fn(actions,state)->True/False/メッセージ, severity)]。
    severity='error'=実行ブロック / 'warn'=警告のみ。
    check_fn は True=合格、False か 文字列=違反（文字列は詳細）。"""
    violations = []
    for name, fn, severity in invariants:
        try:
            r = fn(actions, state)
        except Exception as e:  # noqa: BLE001
            r = f"検査例外: {e}"
            severity = "error"
        if r is not True:
            detail = r if isinstance(r, str) else ""
            violations.append({"invariant": name, "severity": severity, "detail": detail})
    errors = [v for v in violations if v["severity"] == "error"]
    return {"passed": not errors, "violations": violations, "errors": errors}


# ── 3. 通し実行（ゲート通過時のみ commit）──
def run(policy: List[tuple], state: dict, invariants: List[tuple], verbose: bool = True) -> dict:
    """generate→check→通過時のみcommit。返り値=actions(実行可)/generated/audit/gate/committed。
    verbose=Trueで実行ログを表示。error違反があると actions は空（=実行ブロック）。"""
    actions, audit = generate(policy, state)
    gate = check(actions, state, invariants)
    if verbose:
        print(f"生成アクション {len(actions)}件（決定論）")
        for a in actions:
            who = "+".join(audit.get(_canon(a), []))
            print(f"  • {_canon(a)}  ← {who}")
        if audit.get("_ERROR"):
            for e in audit["_ERROR"]:
                print(f"  🔴 ルール例外: {e}")
        for v in gate["violations"]:
            mark = "🔴" if v["severity"] == "error" else "⚠️"
            print(f"  {mark} ゲート違反[{v['invariant']}] {v['detail']}")
        print("運用: " + ("✅commit（実行可）" if gate["passed"] else "⛔ブロック（違反を直すまで実行しない）"))
    return {"actions": actions if gate["passed"] else [],
            "generated": actions, "audit": audit, "gate": gate, "committed": gate["passed"]}


# ── よく使う不変条件のビルダー（汎用）──
def inv_no_duplicate() -> tuple:
    """不変条件: 同一アクションの重複を禁止（違反はerrorで実行ブロック）。"""
    def f(actions, state):
        keys = [_canon(a) for a in actions]
        return True if len(keys) == len(set(keys)) else "重複アクションあり"
    return ("重複なし", f, "error")

def inv_budget(cost_key: str, limit: float) -> tuple:
    """不変条件: cost_keyの合計がlimit以下（予算超過をerrorでブロック）。"""
    def f(actions, state):
        total = sum(a.get(cost_key, 0) for a in actions)
        return True if total <= limit else f"予算超過 {total}>{limit}"
    return (f"予算<= {limit}", f, "error")

def inv_required_fields(fields: Sequence[str]) -> tuple:
    """不変条件: 各アクションにfieldsが全て埋まっている（必須欄欠落をerrorでブロック）。"""
    def f(actions, state):
        for a in actions:
            for k in fields:
                if a.get(k) in (None, ""):
                    return f"必須欄欠落 {k} in {_canon(a)}"
        return True
    return ("必須欄充足", f, "error")

def inv_not_contradict(key: str, conflicting_pairs: Sequence[Tuple[Any, Any]]) -> tuple:
    """同じkeyの値で、両立してはいけない組がアクション集合に共存しないか。"""
    def f(actions, state):
        vals = set(a.get(key) for a in actions)
        for x, y in conflicting_pairs:
            if x in vals and y in vals:
                return f"矛盾アクション {x} と {y}"
        return True
    return ("矛盾なし", f, "error")

def inv_expected_count(expected_fn: Callable[[dict], int], severity: str = "error") -> tuple:
    """『出るべき件数』と一致するか（買い忘れ/出し過ぎの検出）。"""
    def f(actions, state):
        exp = expected_fn(state)
        return True if len(actions) == exp else f"件数不一致 生成{len(actions)}≠期待{exp}"
    return ("期待件数一致", f, severity)


if __name__ == "__main__":
    print(__doc__)
