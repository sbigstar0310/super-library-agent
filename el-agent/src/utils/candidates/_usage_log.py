"""Durable usage logging for aux LLM calls (nl summaries + candidate picks).

These calls (``nl_index._summarize_one`` and ``nl._llm_pick``) bypass
``utils.llm.llm_generation`` and previously discarded ``resp.usage``, so their
token cost was invisible to step-log-based cost accounting. When the env var
``AUX_USAGE_LOG`` points at a file, :func:`record_aux_usage` appends one JSON
line per call with raw token counts — both the OpenAI Responses
(``input_tokens``/``output_tokens``) and OpenAI-compatible chat.completions
(``prompt_tokens``/``completion_tokens``) usage shapes are normalized — plus a
derived cost. Raw tokens are always stored so cost can be recomputed if a price
changes. When ``AUX_USAGE_LOG`` is unset the function is a no-op, so callers
incur no behavior change unless logging is explicitly enabled.
"""

from __future__ import annotations

import json
import os
import threading

_LOCK = threading.Lock()


def _attr(obj, key):
    """Read ``key`` from a pydantic-style object or a plain dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _normalize(usage) -> dict:
    """Normalize Responses / chat.completions usage into a flat token dict."""
    if usage is None:
        return {}
    prompt = _attr(usage, "prompt_tokens")
    if prompt is None:
        prompt = _attr(usage, "input_tokens")
    completion = _attr(usage, "completion_tokens")
    if completion is None:
        completion = _attr(usage, "output_tokens")

    # DeepSeek/OpenRouter report cache hits via a top-level field; OpenAI nests
    # it under {prompt,input}_tokens_details.cached_tokens (matches aggregate_cost.py).
    cached = _attr(usage, "prompt_cache_hit_tokens")
    if cached is None:
        cached = _attr(_attr(usage, "prompt_tokens_details")
                       or _attr(usage, "input_tokens_details"), "cached_tokens") or 0
    reasoning = _attr(_attr(usage, "completion_tokens_details")
                      or _attr(usage, "output_tokens_details"), "reasoning_tokens") or 0

    if prompt is None and completion is None:
        return {}
    return {
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),   # includes reasoning tokens
        "cached_tokens": int(cached or 0),
        "reasoning_tokens": int(reasoning or 0),
    }


def _cost(model: str, tok: dict) -> float | None:
    try:
        from utils.llm import _get_pricing
        p = _get_pricing(model)
        uncached = tok["prompt_tokens"] - tok["cached_tokens"]
        return (uncached * p.get("input", 0)
                + tok["cached_tokens"] * p.get("cached_input", 0)
                + tok["completion_tokens"] * p.get("output", 0)) / 1_000_000
    except Exception:
        return None


def record_aux_usage(kind: str, model: str, usage, *, context: str = "") -> None:
    """Append one usage record to ``$AUX_USAGE_LOG``; no-op if unset/empty.

    Args:
        kind:    call category, e.g. ``"summary"`` or ``"pick:apply"``.
        model:   model id used for the call.
        usage:   the raw ``resp.usage`` object (Responses or chat shape).
        context: free-text provenance (typically the app/lib dir path, whose
                 components encode round/phase/task for post-hoc attribution).
    """
    path = os.environ.get("AUX_USAGE_LOG")
    if not path:
        return
    tok = _normalize(usage)
    if not tok:
        return
    rec = {
        "kind": kind,
        "model": model,
        "context": context,
        "round": os.environ.get("AUX_USAGE_ROUND", ""),
        "phase": os.environ.get("AUX_USAGE_PHASE", ""),
        **tok,
        "cost_usd": _cost(model, tok),
    }
    line = json.dumps(rec, ensure_ascii=False)
    try:
        with _LOCK:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass  # logging must never break the pipeline
