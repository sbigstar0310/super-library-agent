"""LLM logprobs over a prompt (vLLM / OpenAI-compatible completions endpoint).

Returns *prompt* token logprobs — i.e. the model's logprob for each input
token given the preceding tokens, obtained via the well-known
``completions.create(echo=True, logprobs=1, max_tokens=1)`` trick. We trim
the first and last tokens to drop BOS/EOS artifacts.

This is the only LLM provider used in current measurements (Qwen2.5-7B
behind vLLM, see ``deploy_vllm.sh``). The Together provider that lived in
this codebase up to mid-2026 was removed during the refactor since no
production path used it.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for type hints
    pass


async def compute_prompt_logprobs(
    text: str,
    *,
    model: str,
    base_url: str,
    api_key: str = "EMPTY",
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[list[float], list[str]]:
    """Score a prompt and return ``(token_logprobs, tokens)`` for the *input*.

    BOS and EOS tokens are trimmed from both lists. Empty input → empty lists.
    """
    if not text:
        return [], []

    try:
        import openai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "The `openai` package is required for prompt-logprob scoring."
        ) from e

    if not base_url:
        raise RuntimeError("base_url is required (e.g. http://localhost:8000/v1).")

    async def _call() -> tuple[list[float], list[str]]:
        if not hasattr(openai, "AsyncOpenAI"):
            raise RuntimeError(
                "openai>=1.0 with AsyncOpenAI client is required."
            )
        client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        try:
            completion = await client.completions.create(
                model=model,
                prompt=text,
                echo=True,
                logprobs=1,
                max_tokens=1,
            )
            logprobs = completion.choices[0].logprobs
            tokens = list(logprobs.tokens or [])
            token_logprobs = list(logprobs.token_logprobs or [])
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                await close()
        # Drop BOS (index 0) and the appended max_tokens=1 generation slot.
        return (
            [float(x) for x in token_logprobs[1:-1]],
            [str(t) for t in tokens[1:-1]],
        )

    if semaphore is not None:
        async with semaphore:
            return await _call()
    return await _call()
