#!/usr/bin/env python3
"""人手ラベルの対話入力（ターミナルで1問ずつ・3桁打つだけ）。

  python3 rag_label.py
1問ごとに 質問/正解/AIの回答 を表示。「212」のように3桁で入力
（順番＝忠実性・引用・網羅、各0/1/2）。1問ごとに自動保存＝
途中でCtrl+Cで抜けても、次回は残りから再開できる。
"""
import json
import sys
import termios
import tty
from pathlib import Path

HERE = Path(__file__).parent
PATH = HERE / "human_labels.json"


def is_num(x):
    return isinstance(x, (int, float))


def getch() -> str:
    """1キー読む（Enter不要）。IMEオフ（英数）前提。"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03":  # Ctrl+C
        raise KeyboardInterrupt
    return ch


def read_score(n_digits: int = 2) -> str:
    """0/1/2をn回押すだけ（Enter不要）。s=飛ばす q=終了 x=打ち直し。"""
    buf = ""
    while True:
        ch = getch().lower()
        if ch in ("q", "s"):
            print(ch)
            return ch
        if ch == "x" or ch == "\x7f":  # 打ち直し
            buf = ""
            print("\r入力 > " + " " * 12, end="")
            print("\r入力 > ", end="", flush=True)
            continue
        if ch in "012":
            buf += ch
            print(ch, end="", flush=True)
            if len(buf) == n_digits:
                print()
                return buf


def main():
    rows = json.loads(PATH.read_text())
    redo_ids = [a for a in sys.argv[1:] if a.startswith("q")]
    if redo_ids:
        todo = [r for r in rows if r["id"] in redo_ids]  # 指定問の再採点
    else:
        todo = [r for r in rows if not is_num(r.get("human_faithfulness"))]
    done = len(rows) - len(todo)
    if not todo:
        print("全30問 記入済み。次は python3 rag_human_kappa.py")
        return
    print(f"残り {len(todo)}問（記入済み {done}問）。チェックリスト方式＝2桁入力（Enter不要）:")
    print("  1桁目【矛盾】回答に要点と食い違う記述ある? → 2=ない / 1=微妙 / 0=明確にある")
    print("  2桁目【要点】要点リストのうち回答に入ってる数 → 2=ほぼ全部 / 1=一部 / 0=ほぼ無し")
    print("  ※引用の正しさは機械チェック済み＝君は判断しなくていい")
    print("  ※罠問は要点＝「正直に断っているか」だけ。断っていれば 22")
    print("s=この問を飛ばす / q=保存して終了 / x=打ち直し\n")

    for r in todo:
        print("=" * 72)
        print(f"◆ {r['id']} ({r['type']})")
        print(f"【質問】{r.get('question_ja') or r['question']}")
        print(f"【AIの回答(訳)】{r.get('answer_ja') or r['answer']}")
        print("【正解の要点リスト】← 回答にいくつ入ってるか数える")
        for i, kp in enumerate(r.get("keypoints_ja", []), 1):
            print(f"   {i}. {kp}")
        print("入力 2桁（矛盾・要点）> ", end="", flush=True)
        try:
            v = read_score(2)
        except KeyboardInterrupt:
            print("\n保存して終了。再開は同じコマンドで。")
            return
        if v == "q":
            print("保存して終了。再開は同じコマンドで。")
            return
        if v == "s":
            continue
        r["human_faithfulness"], r["human_coverage"] = int(v[0]), int(v[1])
        r["human_citation"] = None  # 引用は機械チェックが正解を持つ＝人手では付けない
        PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    remaining = sum(1 for r in rows if not is_num(r.get("human_faithfulness")))
    print("=" * 72)
    print(f"完了！残り{remaining}問。" + ("次は python3 rag_human_kappa.py" if remaining == 0 else "再開は同じコマンドで。"))


if __name__ == "__main__":
    main()
