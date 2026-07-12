#!/usr/bin/env python3
"""fal.ai any-llm 共通呼び出しヘルパー（依存は標準ライブラリのみ）。

感情分析・logtriage と同じ経路（fal-ai/any-llm）で複数モデルを叩く。
judge も生成も同じ3モデルで揃えられる＝物語（独立3モデル）の一貫性。
  export FAL_KEY=<fal.ai のAPIキー>
"""
import json
import os
import sys
import urllib.error
import urllib.request

ENDPOINT = "https://fal.run/fal-ai/any-llm"

# 感情分析・logtriage と揃えた独立3モデル ＋ llama系フォールバック候補
MODELS = {
    "gemini": "google/gemini-2.5-flash-lite",
    "gpt4o-mini": "openai/gpt-4o-mini",
    "llama": "meta-llama/llama-3.2-3b-instruct",
    "llama31-8b": "meta-llama/llama-3.1-8b-instruct",
    "llama32-1b": "meta-llama/llama-3.2-1b-instruct",
}
# judge3（Meta枠）の探索順。3.2-3bが応答しない時は次を試す
LLAMA_CANDIDATES = ["llama", "llama31-8b", "llama32-1b"]

# フロンティア級judge候補（fal any-llmの実カタログはprobeで確定させる。
# 応答しない場合もerrorに許可モデル一覧が出ることが多い＝それ自体が情報）
STRONG_MODELS = {
    "claude-sonnet": "anthropic/claude-3.5-sonnet",
    "gpt4o": "openai/gpt-4o",
    "gemini-pro": "google/gemini-2.5-pro",
    "deepseek": "deepseek/deepseek-v3",
}
MODELS.update(STRONG_MODELS)


def get_key() -> str:
    key = os.environ.get("FAL_KEY")
    if not key:
        raise SystemExit("環境変数 FAL_KEY を設定してください（https://fal.ai のAPIキー）")
    return key


def call_llm_raw(model: str, prompt: str, system_prompt: str = "") -> dict:
    """1コール・リトライ無しで生レスポンス(dict)を返す。probe/診断用。
    errorフィールドも含めて返すので「なぜ空か」が見える。"""
    key = get_key()
    body = json.dumps({"model": model, "system_prompt": system_prompt, "prompt": prompt}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
    )
    try:
        return json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as e:
        return {"output": "", "error": f"HTTP {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:  # noqa: BLE001
        return {"output": "", "error": f"{type(e).__name__}: {e}"}


def pick_llama_judge() -> tuple:
    """応答するllama系エイリアスを探索順に1コールずつprobeして返す。
    全滅なら (None, 最後のerror)。"""
    from fal_llm import LLAMA_CANDIDATES  # 自己参照でも動く保険
    last_err = ""
    for alias in LLAMA_CANDIDATES:
        resp = call_llm_raw(MODELS[alias], 'Reply with ONLY this JSON: {"ok": 1}')
        if resp.get("output", "").strip():
            return alias, ""
        last_err = str(resp.get("error", ""))[:300]
    return None, last_err


def call_llm(model: str, prompt: str, system_prompt: str = "", retries: int = 3) -> str:
    """any-llm を叩いて output 文字列を返す。失敗時は空文字＋stderr警告。"""
    key = get_key()
    body = json.dumps({"model": model, "system_prompt": system_prompt, "prompt": prompt}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
    )
    last = None
    for _ in range(retries):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120)).get("output", "")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise SystemExit(f"認証エラー HTTP {e.code}: FAL_KEY を確認してください")
            last = e
        except Exception as e:  # noqa: BLE001 — 一時障害は再試行
            last = e
    print(f"  ⚠️ APIコール{retries}回失敗: {type(last).__name__}: {last}", file=sys.stderr)
    return ""
