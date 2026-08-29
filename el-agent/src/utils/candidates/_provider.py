"""Provider routing for aux LLM calls (candidate picker + NL summaries).

Policy: **OpenAI-family models use the OpenAI Responses API; every other model is
served via OpenRouter Chat Completions.** Previously only ``deepseek-*`` was sent
to OpenRouter and everything else defaulted to OpenAI — which broke any other
OpenRouter-hosted backbone (minimax, qwen, …) used as the picker/summary model.
Now the picker follows the coding backbone verbatim and just works.
"""

from __future__ import annotations

# OpenAI-served model id prefixes. Everything not matching goes to OpenRouter.
_OPENAI_PREFIXES = (
    "gpt-", "openai/", "chatgpt",
    "o1-", "o1_", "o3-", "o3_", "o4-", "o4_",
)


def is_openai_model(model: str) -> bool:
    """True iff ``model`` is served by the OpenAI (Responses API) endpoint."""
    m = (model or "").lower()
    return any(m.startswith(p) for p in _OPENAI_PREFIXES)


def openai_model_id(model: str) -> str:
    """Model id for the OpenAI SDK (drops a redundant ``openai/`` prefix)."""
    return model.split("/", 1)[1] if model.lower().startswith("openai/") else model


def openrouter_model_id(model: str) -> str:
    """Aggregator-style ``<vendor>/<model>`` id OpenRouter expects.

    Ids that already carry a vendor prefix pass through unchanged. Bare
    ``deepseek*`` ids get a ``deepseek/`` prefix (back-compat). Any other bare id
    is returned as-is and is expected to already include its vendor prefix
    (e.g. ``minimax/minimax-m3``).
    """
    if "/" in model:
        return model
    if model.lower().startswith("deepseek"):
        return f"deepseek/{model}"
    return model


def is_deepseek_model(model: str) -> bool:
    """True for deepseek ids (which get pinned to the official OR provider)."""
    return "deepseek" in (model or "").lower()
