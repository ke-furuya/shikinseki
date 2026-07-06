"""試金石 (shikinseki) — 「その数字は本物か、まぐれか」を統計で見分ける定量検証ツールキット。

4層スタック：
  data_harvester … 取得（壊れた/欠けたデータで分析を始めない）
  leak_guard     … 防御（未来情報リークを止める）
  edge_validator … 検証（効いてる/まぐれ/測れないを統計判定）＝看板
  action_gate    … 運用（決定論生成＋機械ゲートで人為ミスを防ぐ）

依存：標準ライブラリのみ。使い方: `from shikinseki import edge_validator as ev`
"""
from . import edge_validator, data_harvester, leak_guard, action_gate

__version__ = "0.1.0"
__all__ = ["edge_validator", "data_harvester", "leak_guard", "action_gate"]
