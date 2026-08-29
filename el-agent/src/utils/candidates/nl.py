"""NL-index based candidate retrieval (Method B port).

No embedding/cosine step: per-chunk NL summaries for each app + lib are dumped
into one LLM prompt, and the LLM picks top-K candidates directly as markdown.
Two entry points mirror ``embed.py``: ``get_apply_candidates_nl`` (lib + one
app → symbols to adopt) and ``get_extract_candidates_nl`` (all apps → cross-app
candidates; ``mode="local"`` for single-app).

Duplicate prevention is implicit: lib summaries enter the extract prompt as
"already-extracted symbols", subsuming the legacy ``Refused`` ledger. ``prep``
is empty for both functions — refused-judge wiring is deferred.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from .nl_index import ensure_nl_index, load_nl_index
from .types import (
    NO_APPLY_CANDIDATES,
    NO_EXTRACT_CANDIDATES,
    CandidateResult,
)

_DEFAULT_MODEL = "gpt-5.4-nano"             # nl_index one-line summarization
_DEFAULT_PICK_MODEL = "deepseek-v4-flash"   # candidate selection (reasoning)
_REASONING_EFFORT = "high"


class CandidateIndexError(RuntimeError):
    """The NL index could not be built or read.

    Raised instead of degrading, because `nl` is the default retrieval
    strategy: a run that continues without candidates still finishes and
    still produces metrics, they are just the metrics of a weaker method.
    That failure mode cost us a whole campaign (mm3, 2026-07-10) before
    anyone noticed. Set SLA_STRICT_CANDIDATES=0 to get best-effort back.
    """


def _strict() -> bool:
    return os.environ.get("SLA_STRICT_CANDIDATES", "1") not in ("0", "false", "False")


def _degrade(stage: str, exc: Exception, fallback: str) -> CandidateResult:
    """Fail loudly by default; warn and continue only when asked to."""
    hint = ""
    if "cocoindex sqlite missing" in str(exc):
        hint = (
            " The cocoindex daemon usually failed or its config is absent — check "
            "~/.cocoindex_code/global_settings.yml and that `ccc index` can run."
        )
    if _strict():
        raise CandidateIndexError(
            f"{stage} candidate retrieval failed: {exc}.{hint} Continuing would "
            f"silently run this phase without candidates and report the result as "
            f"if the method had been applied. Set SLA_STRICT_CANDIDATES=0 to "
            f"proceed anyway."
        ) from exc
    print(f"[candidates.nl] {stage} degraded, continuing without candidates: {exc}{hint}")
    return CandidateResult(fallback, [])

_APPLY_SYSTEM = (
    "You are a senior engineer migrating an app to use a shared utility library. "
    "Given (a) one-line summaries of every symbol in the library and (b) summaries "
    "of every chunk in ONE specific app, pick the TOP-K library symbols this app "
    "should adopt — i.e., places where the app currently has its own implementation "
    "that the library symbol could replace.\n\n"
    "For each candidate, output exactly:\n\n"
    "### A<n>. <lib symbol>\n"
    "**Library**: <lib_file>:<start>-<end>::<chunk_id=<id>> — <lib summary>\n"
    "**Replaces in this app**:\n"
    "  - <app_file>:<start>-<end>::<chunk_id=<id>> — <reason>\n"
    "  - ... (one or more places)\n\n"
    "Be conservative. If fewer than the requested number of strong matches exist, "
    "output only the strong ones. If NONE, output exactly 'NONE'."
)

_EXTRACT_SYSTEM = (
    "You are a senior library architect deciding which code patterns to extract "
    "from multiple Python codebases into a shared library. You will be given "
    "one-line summaries of every chunk across all apps, plus summaries of the "
    "EXISTING library (treat lib symbols as 'already extracted' — do NOT propose "
    "duplicates).\n\n"
    "Identify the TOP-K distinct, high-value cross-app patterns that appear in "
    "TWO OR MORE apps and would be genuinely worth extracting (not trivial "
    "helpers, not app-specific glue, not duplicates of existing lib).\n\n"
    "For each candidate, output exactly:\n\n"
    "### C<n>. <short pattern name>\n"
    "**Pattern**: <1 sentence describing the shared behavior>\n"
    "**Why extractable**: <1 sentence on why this generalizes>\n"
    "**Members**:\n"
    "  - <app>::<file_path>:<start>-<end>::<chunk_id=<id>> — <summary>\n"
    "  - ... (>=2 distinct apps required)\n\n"
    "Be conservative — false positives matter more than coverage."
)

_LOCAL_EXTRACT_SYSTEM = (
    "You are a senior engineer identifying intra-app duplicated code patterns "
    "that should be extracted into a per-app helper module (local_lib/).\n\n"
    "Given one-line summaries of every chunk in ONE app, identify the TOP-K "
    "patterns that appear in TWO OR MORE places within this app and would "
    "benefit from being unified into a single helper.\n\n"
    "For each candidate, output exactly:\n\n"
    "### C<n>. <short pattern name>\n"
    "**Pattern**: <1 sentence describing the shared behavior>\n"
    "**Members**:\n"
    "  - <file_path>:<start>-<end>::<chunk_id=<id>> — <summary>\n"
    "  - ... (>=2 distinct call sites)\n\n"
    "Be conservative — only surface patterns with clear unification value."
)


# ---- apply --------------------------------------------------------------


def get_apply_candidates_nl(
    *,
    library_dir: str,
    app_dir: str,
    top_k: int = 10,
    min_line: int = 5,
    model: str = _DEFAULT_MODEL,
    pick_model: str | None = None,
    max_tokens: int = 16000,
) -> CandidateResult:
    """NL-index Library→app retrieval. Returns ``CandidateResult``.

    Args:
        model:      Model for nl_index one-line summaries (per chunk).
        pick_model: Model for final candidate selection. Defaults to
                    ``model`` when None (backwards-compat); explicit
                    callers typically pass ``deepseek-v4-flash``.
    """
    pick = pick_model or model
    try:
        ensure_nl_index(library_dir, model=model, min_line=min_line)
        ensure_nl_index(app_dir, model=model, min_line=min_line)
        lib_idx = load_nl_index(library_dir)
        app_idx = load_nl_index(app_dir)
    except Exception as e:
        return _degrade("apply", e, NO_APPLY_CANDIDATES)

    if not lib_idx:
        return CandidateResult(NO_APPLY_CANDIDATES, [])

    user_msg = (
        f"### Library ({len(lib_idx)} symbols)\n\n"
        f"{_render_index('LIBRARY', lib_idx)}\n\n"
        f"### Target App: {Path(app_dir).name} ({len(app_idx)} chunks)\n\n"
        f"{_render_index('APP', app_idx)}\n\n"
        f"Pick the top-{top_k} migration candidates."
    )
    md = _llm_pick(_APPLY_SYSTEM, user_msg, model=pick, max_tokens=max_tokens,
                   label="apply", context=app_dir)
    if not md or md.strip().upper() == "NONE":
        return CandidateResult(NO_APPLY_CANDIDATES, [])

    if os.environ.get("WEBGEN_INJECT_NEIGHBORS", "1") not in ("0", "false", "False"):
        try:
            from .dep_neighbors import build_neighbors, inject_neighbors
            graph = build_neighbors(app_dir, library_dir, task_name="webgen")
            md = inject_neighbors(md, graph, app_dir)
        except Exception as e:
            print(f"[candidates.nl] neighbor injection skipped: {e}")

    return CandidateResult(md, [])


# ---- extract ------------------------------------------------------------


ExtractMode = Literal["global", "local"]


def get_extract_candidates_nl(
    *,
    app_dirs: dict[str, str],
    library_dir: str | None = None,
    mode: ExtractMode = "global",
    top_k: int = 10,
    min_line: int = 5,
    model: str = _DEFAULT_MODEL,
    pick_model: str | None = None,
    max_tokens: int = 8000,
) -> CandidateResult:
    """NL-index extract. Returns ``CandidateResult``.

    Args:
        library_dir: Optional. When provided, lib NL is dumped into the
            prompt as 'already-extracted symbols' so the LLM avoids
            duplicates. Ignored when ``mode='local'``.
        model:       Model for nl_index one-line summaries.
        pick_model:  Model for final candidate selection. Defaults to
                     ``model`` when None.
    """
    pick = pick_model or model
    try:
        if mode == "global":
            return _extract_global_inner(
                app_dirs=app_dirs,
                library_dir=library_dir,
                top_k=top_k,
                min_line=min_line,
                model=model,
                pick_model=pick,
                max_tokens=max_tokens,
            )
        return _extract_local_inner(
            app_dirs=app_dirs,
            top_k=top_k,
            min_line=min_line,
            model=model,
            pick_model=pick,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return _degrade("extract", e, NO_EXTRACT_CANDIDATES)


def _extract_global_inner(
    *,
    app_dirs: dict[str, str],
    library_dir: str | None,
    top_k: int,
    min_line: int,
    model: str,
    pick_model: str,
    max_tokens: int,
) -> CandidateResult:
    blocks: list[str] = []
    total = 0
    for tid, app_dir in app_dirs.items():
        try:
            ensure_nl_index(app_dir, model=model, min_line=min_line)
            idx = load_nl_index(app_dir)
        except Exception as e:
            if _strict():
                raise CandidateIndexError(
                    f"extract: no NL index for task {tid}: {e}. That app would "
                    f"contribute nothing to cross-app extraction while the run "
                    f"still reported success. Set SLA_STRICT_CANDIDATES=0 to skip it."
                ) from e
            print(f"[candidates.nl] skip {tid}: {e}")
            continue
        if not idx:
            continue
        blocks.append(_render_index(f"App: {tid}", idx))
        total += len(idx)

    if total < 2:
        return CandidateResult(NO_EXTRACT_CANDIDATES, [])

    lib_block = ""
    if library_dir:
        try:
            ensure_nl_index(library_dir, model=model, min_line=min_line)
            lib_idx = load_nl_index(library_dir)
            if lib_idx:
                lib_block = (
                    f"### EXISTING LIBRARY (already extracted — do NOT duplicate)\n\n"
                    f"{_render_index('LIBRARY', lib_idx)}\n\n"
                )
        except Exception as e:
            print(f"[candidates.nl] lib index unavailable, proceeding without: {e}")

    user_msg = (
        f"{lib_block}"
        f"### APPS ({len(app_dirs)} apps, {total} chunks total)\n\n"
        f"{chr(10).join(blocks)}\n\n"
        f"Identify the top-{top_k} cross-app extraction candidates."
    )
    md = _llm_pick(_EXTRACT_SYSTEM, user_msg, model=pick_model, max_tokens=max_tokens,
                   label="extract_global", context=(library_dir or ""))
    if not md or md.strip().upper() == "NONE":
        return CandidateResult(NO_EXTRACT_CANDIDATES, [])
    return CandidateResult(md, [])


def _extract_local_inner(
    *,
    app_dirs: dict[str, str],
    top_k: int,
    min_line: int,
    model: str,
    pick_model: str,
    max_tokens: int,
) -> CandidateResult:
    if len(app_dirs) != 1:
        raise ValueError(
            f"NL local extract expects exactly 1 app dir, got {len(app_dirs)}"
        )
    tid, app_dir = next(iter(app_dirs.items()))
    try:
        ensure_nl_index(app_dir, model=model, min_line=min_line)
        idx = load_nl_index(app_dir)
    except Exception as e:
        print(f"[candidates.nl] local extract index prep failed: {e}")
        return CandidateResult(NO_EXTRACT_CANDIDATES, [])

    if len(idx) < 2:
        return CandidateResult(NO_EXTRACT_CANDIDATES, [])

    user_msg = (
        f"### App: {tid} ({len(idx)} chunks)\n\n"
        f"{_render_index(tid.upper(), idx)}\n\n"
        f"Identify the top-{top_k} intra-app extraction candidates."
    )
    md = _llm_pick(_LOCAL_EXTRACT_SYSTEM, user_msg, model=pick_model, max_tokens=max_tokens,
                   label="extract_local", context=app_dir)
    if not md or md.strip().upper() == "NONE":
        return CandidateResult(NO_EXTRACT_CANDIDATES, [])
    return CandidateResult(md, [])


# ---- shared helpers -----------------------------------------------------


def _render_index(label: str, index: dict[str, dict]) -> str:
    lines = [f"## {label}"]
    for chunk_id, e in index.items():
        fp = e.get("file_path", "")
        lines_range = e.get("lines", [0, 0])
        n = lines_range[1] - lines_range[0] + 1
        summary = e.get("content_summary", "").strip()
        lines.append(f"- chunk_id={chunk_id} [{n}L] {fp}:{lines_range[0]} — {summary}")
    return "\n".join(lines)


def _llm_pick(system: str, user: str, *, model: str, max_tokens: int,
              label: str = "pick", context: str = "") -> str:
    from utils.llm import get_client

    from ._provider import (
        is_deepseek_model,
        is_openai_model,
        openai_model_id,
        openrouter_model_id,
    )
    from ._usage_log import record_aux_usage

    # Routing: OpenAI-family → OpenAI Responses API; everything else (deepseek,
    # minimax, qwen, …) → OpenRouter Chat Completions. The picker follows the
    # coding backbone, so any OpenRouter-hosted backbone just works.
    if is_openai_model(model):
        # gpt-5.x reasoning models — use Responses API (no temperature/max_tokens).
        # Picker benefits from `effort=high` (broad context, careful selection).
        resp = get_client("openai").responses.create(
            model=openai_model_id(model),
            reasoning={"effort": _REASONING_EFFORT},
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        record_aux_usage(f"pick:{label}", model, getattr(resp, "usage", None), context=context)
        return (resp.output_text or "").strip()

    # OpenRouter: enable reasoning effort. DeepSeek is pinned to its official
    # provider (lab policy: avoid third-party quantized serving); other backbones
    # use OpenRouter's default routing.
    extra_body: dict = {"reasoning": {"enabled": True, "effort": _REASONING_EFFORT}}
    if is_deepseek_model(model):
        extra_body["provider"] = {"only": ["deepseek"], "allow_fallbacks": False}
    resp = get_client("openrouter").chat.completions.create(
        model=openrouter_model_id(model),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    record_aux_usage(f"pick:{label}", model, getattr(resp, "usage", None), context=context)
    return (resp.choices[0].message.content or "").strip()
