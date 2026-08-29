"""Dep-aware MDL/NLL scoring for (library, app) code pairs.

Formula
-------
For a single app conditioned on a library::

    MDL  =  NLL(lib_stripped)
          + Σ_f  NLL( file_stripped | direct_deps_of_f )

- ``lib_stripped``: library source with language-appropriate comments and
  docstrings removed (see ``parsers.{js,python}.strip_comments``). Joined
  per-file into a single corpus string via ``file_io.read_dir_to_text``.
- ``file_stripped``: the app file being scored, with comments stripped.
- ``direct_deps_of_f``: the source of files ``f`` directly imports (1 level,
  not transitive). Comments are **not** stripped from dep context.
- ``NLL(x | y)``: ``-Σ log p(x_i | y, x_<i)`` via prompt-logprob trick on a
  vLLM/OpenAI-compatible endpoint (see ``llm.compute_prompt_logprobs``).

Legacy "concat" mode lives in ``concat.py`` for cross-paper sanity checks.
``MDLMetric`` itself is dep-aware only.

See ``README.md`` next to this file for a reproduction walk-through and
the JSON schema of the returned result.
"""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass

import numpy as np

from .configs import TaskConfig, load_task_config
from .file_io import format_dep_context, read_dir_to_text
from .llm import compute_prompt_logprobs


@dataclass
class FileDetail:
    rel_path: str
    nll: float
    tokens: int
    nll_per_token: float
    dep_paths: list[str] | None = None


@dataclass
class AppMDLResult:
    """MDL result for one app. ``as_dict()`` produces the JSON payload that
    ``scripts/metrics/get_mdl.py`` writes to ``mdl_results.json``.

    Fields::

        app_nll       total NLL of stripped app files conditioned on deps
        library_nll   NLL of stripped library corpus (precomputed once when
                      a single lib_dir is shared across multiple apps)
        nll_per_token (app_nll + library_nll) / (app_tokens + library_tokens)
        perplexity    exp(nll_per_token), capped at +inf
        file_details  per-file FileDetail records
    """
    model: str
    provider: str
    app_path: str
    app_nll: float
    library_nll: float
    app_tokens: int
    library_tokens: int
    nll_per_token: float
    perplexity: float
    file_details: list[FileDetail]

    @property
    def total_tokens(self) -> int:
        return self.app_tokens + self.library_tokens

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "app_path": self.app_path,
            "app_nll": round(self.app_nll, 4),
            "library_nll": round(self.library_nll, 4),
            "nll_per_token": round(self.nll_per_token, 4),
            "perplexity": round(self.perplexity, 4),
            "total_tokens": self.total_tokens,
            "library_tokens": self.library_tokens,
            "app_tokens": self.app_tokens,
            "file_details": [self._fd_dict(fd) for fd in self.file_details],
        }

    @staticmethod
    def _fd_dict(fd: FileDetail) -> dict:
        d: dict = {
            "rel_path": fd.rel_path,
            "nll": round(fd.nll, 4),
            "tokens": fd.tokens,
            "nll_per_token": round(fd.nll_per_token, 4),
        }
        if fd.dep_paths is not None:
            d["dep_paths"] = fd.dep_paths
        return d


class MDLMetric:
    """Dep-aware MDL scorer over a vLLM/OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen2.5-7B",
        base_url: str | None = None,
        api_key: str | None = None,
        joiner: str = "\n",
        max_concurrency: int = 1,
    ):
        self.model = model
        self.joiner = joiner
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._base_url = base_url or "http://127.0.0.1:8000/v1"
        self._api_key = api_key or "EMPTY"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        app_dir: str,
        lib_dir: str | None = None,
        *,
        task: TaskConfig | None = None,
        precomputed_lib: tuple[float, int] | None = None,
        strip_comments: bool = True,
    ) -> AppMDLResult:
        """Synchronous wrapper around :meth:`ascore`."""
        return asyncio.run(self.ascore(
            app_dir, lib_dir,
            task=task,
            precomputed_lib=precomputed_lib,
            strip_comments=strip_comments,
        ))

    async def ascore(
        self,
        app_dir: str,
        lib_dir: str | None = None,
        *,
        task: TaskConfig | None = None,
        precomputed_lib: tuple[float, int] | None = None,
        strip_comments: bool = True,
    ) -> AppMDLResult:
        """Compute dep-aware MDL for one app.

        Args:
            app_dir: directory of app source to score.
            lib_dir: optional library directory whose NLL contributes to MDL.
            task: ``TaskConfig`` (webgen / paperbench).
                Defaults to webgen for back-compat with pre-refactor callers.
            precomputed_lib: ``(library_nll, library_tokens)`` if you already
                scored the library elsewhere (e.g. across N apps sharing one
                lib_dir). Set by the CLI's ``LibNllCache``.
            strip_comments: when True (default), strip comments/docstrings
                from lib and app files *before* scoring. Dep context is
                **never** stripped, regardless of this flag.
        """
        if task is None:
            task = load_task_config("webgen")
        return await self._ascore_dep_aware(
            app_dir, lib_dir, precomputed_lib, strip_comments, task,
        )

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    async def _ascore_dep_aware(
        self,
        app_dir: str,
        lib_dir: str | None,
        precomputed_lib: tuple[float, int] | None,
        strip_comments: bool,
        task: TaskConfig,
    ) -> AppMDLResult:
        parser = task.parser_module
        strip_fn = parser.strip_comments if strip_comments else None

        graph = parser.build_dep_graph(app_dir, lib_dir, task)
        app_dir_prefix = os.path.abspath(app_dir) + os.sep
        app_files = {
            p: node for p, node in graph.items()
            if os.path.abspath(p).startswith(app_dir_prefix)
        }

        # --- App files, each conditioned on its deps (deps NOT stripped) ---
        total_app_nll = 0.0
        total_app_tokens = 0
        file_details: list[FileDetail] = []
        for _abs_path, node in sorted(app_files.items(), key=lambda x: x[1].rel_path):
            dep_nodes = [graph[dp] for dp in node.deps if dp in graph]
            dep_context = format_dep_context(dep_nodes)
            scored_content = strip_fn(node.content) if strip_fn else node.content
            lp, lt = await self._score_conditioned(scored_content, dep_context)

            file_nll = -1.0 * float(np.sum(lp))
            file_tokens = len(lt)
            npt = file_nll / file_tokens if file_tokens > 0 else float("nan")
            file_details.append(FileDetail(
                rel_path=node.rel_path,
                nll=file_nll,
                tokens=file_tokens,
                nll_per_token=npt,
                dep_paths=[graph[dp].rel_path for dp in node.deps if dp in graph],
            ))
            total_app_nll += file_nll
            total_app_tokens += file_tokens

        # --- Library NLL: flat-concat of stripped lib files ---
        if precomputed_lib is not None:
            library_nll, library_tokens = precomputed_lib
        else:
            library_nll, library_tokens = await self._score_library(
                lib_dir, task, strip_fn,
            )

        return self._build_result(
            app_dir, total_app_nll, total_app_tokens,
            library_nll, library_tokens, file_details,
        )

    async def _score_library(
        self,
        lib_dir: str | None,
        task: TaskConfig,
        strip_fn,
    ) -> tuple[float, int]:
        """Compute lib NLL once. Used by ``LibNllCache`` for batch CLI runs."""
        if not lib_dir:
            return 0.0, 0
        lib_text = read_dir_to_text(lib_dir, task=task, strip_fn=strip_fn)
        if not lib_text:
            return 0.0, 0
        lp, lt = await self.get_prompt_logprobs(lib_text)
        return -1.0 * float(np.sum(lp)), len(lt)

    # ------------------------------------------------------------------
    # Logprob helpers (also used by concat.py)
    # ------------------------------------------------------------------

    async def get_prompt_logprobs(self, text: str) -> tuple[list[float], list[str]]:
        """Public alias for ``llm.compute_prompt_logprobs`` bound to this metric."""
        return await compute_prompt_logprobs(
            text,
            model=self.model,
            base_url=self._base_url,
            api_key=self._api_key,
            semaphore=self._semaphore,
        )

    async def _score_conditioned(
        self, content: str, context: str,
    ) -> tuple[list[float], list[str]]:
        """Return (logprobs, tokens) for ``content`` conditioned on ``context``.

        Trick: tokenize ``context`` alone and ``context + joiner + content``
        separately, then drop the first ``len(context_tokens)`` from the
        combined run. This yields logprobs of ``content`` tokens given the
        context as prefix.
        """
        if not context:
            return await self.get_prompt_logprobs(content)

        concat_text = self._concat(context, content)
        ctx_lp, ctx_lt = await self.get_prompt_logprobs(context)
        concat_lp, concat_lt = await self.get_prompt_logprobs(concat_text)
        prefix = min(len(ctx_lt), len(concat_lt), len(concat_lp))
        return concat_lp[prefix:], concat_lt[prefix:]

    def _concat(self, prefix: str, suffix: str) -> str:
        if not prefix:
            return suffix
        if not suffix:
            return prefix
        return f"{prefix}{self.joiner}{suffix}"

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    def _build_result(
        self,
        app_path: str,
        app_nll: float,
        app_tokens: int,
        library_nll: float,
        library_tokens: int,
        file_details: list[FileDetail],
    ) -> AppMDLResult:
        total_tokens = app_tokens + library_tokens
        nll_per_token = (
            ((app_nll + library_nll) / total_tokens)
            if total_tokens > 0 and np.isfinite(app_nll)
            else float("nan")
        )
        perplexity = (
            float(math.exp(nll_per_token))
            if np.isfinite(nll_per_token) and nll_per_token < 100
            else (float("inf") if np.isfinite(nll_per_token) else float("nan"))
        )
        return AppMDLResult(
            model=self.model,
            provider="openai_completions",
            app_path=app_path,
            app_nll=app_nll,
            library_nll=library_nll,
            app_tokens=app_tokens,
            library_tokens=library_tokens,
            nll_per_token=nll_per_token,
            perplexity=perplexity,
            file_details=file_details,
        )


# ----------------------------------------------------------------------
# Convenience wrapper (production callers use this — signature is stable)
# ----------------------------------------------------------------------

def get_mdl_score(
    app_dir: str,
    library_dir: str | None = None,
    shuffle: bool = False,
    *,
    task: TaskConfig | None = None,
) -> float:
    """Compute ``app_nll`` for one app. Returns ``nan`` on error.

    Stable signature used by ``main.py``, ``coding_exp.py``, ``apply_exp.py``,
    ``feedback_agent.py``. Creates a fresh ``MDLMetric`` per call so the
    ``asyncio.Semaphore`` doesn't bind to a stale loop. ``shuffle`` is kept
    for back-compat and ignored.
    """
    try:
        metric = MDLMetric()
        result = metric.score(app_dir, library_dir, task=task)
        print(f"[MDL] score (app_nll, dep-aware): {result.app_nll:.4f}")
        return result.app_nll
    except Exception as e:
        print(f"[MDL] computation failed: {e}")
        return float("nan")
