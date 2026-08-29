import os
import re
from typing import Literal
from openai import OpenAI, BadRequestError, RateLimitError, APIError, APITimeoutError


_clients = {}

PRICING = {
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "deepseek-v3.2": {"input": 0.25, "cached_input": 0.26, "output": 0.38},
    "deepseek-v4-flash":  {"input": 0.14, "cached_input": 0.0028, "output": 0.28},
    "deepseek-v4-pro":    {"input": 0.435, "cached_input": 0.003625, "output": 0.87},
    "anthropic/claude-opus-4.6": {"input": 5, "cached_input": 0.5, "output": 25},
    "minimax/minimax-m2.5": {"input": 0.30, "cached_input": 0.03, "output": 1.20},
    "minimax-m3": {"input": 0.30, "cached_input": 0.06, "output": 1.20},
}

# Date-suffix pattern: e.g. "gpt-5.4-mini-2026-03-17" -> "gpt-5.4-mini"
_DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _get_pricing(model: str) -> dict:
    """Look up pricing for a model, stripping provider prefix and date suffix if needed."""
    # Try exact match first
    if model in PRICING:
        return PRICING[model]
    # Strip provider prefix (e.g. "openai/gpt-5-mini" -> "gpt-5-mini")
    stripped = model.split("/", 1)[-1] if "/" in model else model
    if stripped in PRICING:
        return PRICING[stripped]
    # Strip date suffix (e.g. "gpt-5.4-mini-2026-03-17" -> "gpt-5.4-mini")
    base_model = _DATE_SUFFIX_RE.sub("", model)
    if base_model in PRICING:
        return PRICING[base_model]
    # Both: strip prefix then date suffix
    base_stripped = _DATE_SUFFIX_RE.sub("", stripped)
    if base_stripped in PRICING:
        return PRICING[base_stripped]
    return {"input": 0.03, "output": 0.12}


def get_client(provider: Literal["openai", "openrouter"] = "openai"):
    """Lazy initialization of OpenAI-compatible client per provider.

    DeepSeek models are served through the ``openrouter`` provider (lab
    policy: the direct DeepSeek API is no longer used).
    """
    if provider not in _clients:
        if provider == "openai":
            _clients[provider] = OpenAI(
                api_key=os.environ.get("OPENAILIKE_API_KEY"),
                base_url=os.environ.get("OPENAILIKE_BASE_URL"),
            )
        elif provider == "openrouter":
            _clients[provider] = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            )
        else:
            raise ValueError(f"Invalid provider: {provider}")
    return _clients[provider]


def _clean_messages(messages: list[dict]) -> list[dict]:
    """Strip non-standard fields (e.g., meta) from messages before LLM API call."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def filter_messages_by_steps(messages: list[dict], current_step: int, keep_last_k: int = 10) -> list[dict]:
    """Keep system prompt + messages from the last k steps only.

    Messages without 'meta' field (e.g., system prompt) are always kept.
    Messages with meta.step >= (current_step - keep_last_k) are kept.
    """
    cutoff_step = max(0, current_step - keep_last_k)
    filtered = []
    for m in messages:
        meta = m.get("meta")
        if meta is None:
            # System prompt (no meta) — always keep
            filtered.append(m)
        elif meta.get("step", 0) >= cutoff_step:
            filtered.append(m)
    return filtered


def filter_messages_by_turns(messages: list[dict], current_turn: int, keep_last_k: int = 10) -> list[dict]:
    """Keep system prompt + messages from the last k turns only.

    Messages without 'meta' field (e.g., system prompt) are always kept.
    Messages with meta.turn == -1 (initial user prompt) are always kept.
    Messages with meta.turn >= (current_turn - keep_last_k) are kept.
    """
    cutoff_turn = max(0, current_turn - keep_last_k)
    filtered = []
    for m in messages:
        meta = m.get("meta")
        if meta is None:
            filtered.append(m)
        elif meta.get("turn", -1) == -1:
            filtered.append(m)
        elif meta.get("turn", 0) >= cutoff_turn:
            filtered.append(m)
    return filtered


def _supports_reasoning_effort(model: str) -> bool:
    """GPT-5 family and o-series support reasoning_effort."""
    m = model.lower()
    if "/" in m:
        m = m.split("/", 1)[-1]
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def llm_generation(messages, model, max_tokens=-1, max_completion_tokens=-1, temperature=0.0, provider="openai", openrouter_reasoning=True, reasoning_effort=None) -> dict:
    """
    Generate LLM response with automatic fallback for different model APIs.
    - Tries with all parameters first
    - Falls back to different parameter combinations on BadRequestError

    Returns:
        dict: {
            "content": str,
            "prompt_tokens": int,
            "completion_tokens": int,
            "cost": float,
        }
    """
    client = get_client(provider)
    tokens = max_completion_tokens if max_completion_tokens > 0 else max_tokens

    # Add an aggregator prefix when the model has none (e.g.
    # text-embedding-3-small -> openai/text-embedding-3-small). DeepSeek models
    # are served via OpenRouter under the "deepseek/" namespace.
    if provider == "openrouter" and "/" not in model:
        model = f"deepseek/{model}" if "deepseek" in model.lower() else f"openai/{model}"

    if provider == "openai" and model.startswith("openai/"):
        model = model[len("openai/"):]

    # OpenRouter extra_body
    extra_body = None
    if provider == "openrouter":
        # Exclude quantized models
        extra_body = {
            "provider": {
                "quantizations": ["unknown", "fp32", "fp16", "bf16", "fp8"],
                "allow_fallbacks": False,
            }
        }
        if openrouter_reasoning and reasoning_effort:
            extra_body["reasoning"] = {"enabled": True, "effort": reasoning_effort}

        # Claude 4.6 - effort
        if "claude" in model.lower() and "4.6" in model:
            extra_body["verbosity"] = "high"  # "low", "medium", "high", "max"

        if "deepseek" in model.lower():
            extra_body["provider"].update({
                "only": ["deepseek"],
                "require_parameters": True,
            })
            # DeepSeek official provider doesn't tag quantization — remove filter
            extra_body["provider"].pop("quantizations", None)
        if not extra_body:
            extra_body = None

    # Native reasoning_effort for OpenAI GPT-5 / o-series
    reasoning_kwarg = {}
    if reasoning_effort and provider == "openai" and _supports_reasoning_effort(model):
        reasoning_kwarg["reasoning_effort"] = reasoning_effort
    
    # Build parameter combinations to try (in order of preference)
    param_sets = []

    # 1. Try with temperature + max_completion_tokens (newer models)
    if tokens > 0 and temperature > 0:
        param_sets.append({"temperature": temperature, "max_completion_tokens": tokens, **reasoning_kwarg})
    # 2. Try with temperature + max_tokens (older models)
    if tokens > 0 and temperature > 0:
        param_sets.append({"temperature": temperature, "max_tokens": tokens, **reasoning_kwarg})
    # 3. Try with max_completion_tokens only (models that don't support custom temperature)
    if tokens > 0:
        param_sets.append({"max_completion_tokens": tokens, **reasoning_kwarg})
    # 4. Try with max_tokens only
    if tokens > 0:
        param_sets.append({"max_tokens": tokens, **reasoning_kwarg})
    # 5. Try with temperature only
    if temperature > 0:
        param_sets.append({"temperature": temperature, **reasoning_kwarg})
    # 6. Try with flex tier
    # param_sets.append({"service_tier": "flex"})
    param_sets.append({**reasoning_kwarg})
    # 7. Try with timeout
    param_sets.append({"timeout": 300, **reasoning_kwarg})
    # 8. Final fallback without reasoning_effort (for models that reject it)
    if reasoning_kwarg:
        param_sets.append({"timeout": 300})
    
    
    last_error = None
    for i, params in enumerate(param_sets):
        try:
            request_kwargs = {
                "model": model,
                "messages": _clean_messages(messages),
                **params,
            }
            if extra_body is not None:
                request_kwargs["extra_body"] = extra_body

            chat_response = client.chat.completions.create(**request_kwargs)

            choice = chat_response.choices[0]
            content = choice.message.content
            usage = chat_response.usage

            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

            if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
            else:
                cached_tokens = 0

            price = _get_pricing(model)
            uncached_input = prompt_tokens - cached_tokens
            cost = (
                uncached_input * price.get("input", 0) / 1_000_000 +
                cached_tokens * price.get("cached_input", 0) / 1_000_000 +
                completion_tokens * price.get("output", 0) / 1_000_000
            )
            if not content:
                print(f"[LLM] Warning: Empty response from model '{model}' (finish_reason: {choice.finish_reason})")

            reasoning = getattr(choice.message, "reasoning", None)
            reasoning_details = getattr(choice.message, "reasoning_details", None)

            return {
                "content": content or "",
                "reasoning": reasoning,
                "reasoning_details": reasoning_details,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
            }
        except BadRequestError as e:
            last_error = e
            print(f"[LLM] BadRequestError (attempt {i+1}/{len(param_sets)}): {e}")
            continue
        except RateLimitError as e:
            print(f"[LLM] RateLimitError (attempt {i+1}/{len(param_sets)}): {e}")
            last_error = e
            continue
        except APIError as e:
            print(f"[LLM] APIError (attempt {i+1}/{len(param_sets)}): {e}")
            last_error = e
            continue
        except Exception as e:
            print(f"[LLM] Exception (attempt {i+1}/{len(param_sets)}): {e}")
            last_error = e
            continue
    
    # All attempts failed
    print(f"[LLM] All {len(param_sets)} attempts failed. Last error: {last_error}")
    raise last_error

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    messages = [
        {"role": "user", "content": "Solve this carefully and give the final answer."}
    ]

    result = llm_generation(
        messages=messages,
        model="deepseek/deepseek-v3.2",
        provider="openrouter",
        temperature=0.0,  # reasoning route에서는 보수적으로 0 권장
        max_completion_tokens=4000,
        openrouter_reasoning=True,
    )

    print("CONTENT:\n", result["content"])
    print("REASONING:\n", result["reasoning"])
    print("REASONING_DETAILS:\n", result["reasoning_details"])
    print("COST:\n", result["cost"])