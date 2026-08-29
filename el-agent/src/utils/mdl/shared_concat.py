"""Shared-lib-concat MDL scoring.

Definition::

    MDL  =  NLL(cat(lib_files))
          + Σ_app  NLL( cat(app_files) | cat(lib_files) )

Each app is conditioned on the *same* library prefix. The library is scored
once across all apps in the suite (no per-task mirror inflation). Files
inside an app see each other (sibling driver scripts compress).

vLLM prefix caching (server-side, ``--enable-prefix-caching``) makes the
shared library prefix free after the first forward pass.

Caveat — file-level NLL: this v1 emits a single ``<app-concat>`` FileDetail
because slicing the per-file boundary inside a concat run is tokenizer-fuzzy
(same caveat as ``concat.py``). For per-file breakdown use dep-aware.
"""

from __future__ import annotations

import hashlib
import os
from collections import deque
from dataclasses import dataclass

import numpy as np

from .configs import TaskConfig, load_task_config
from .file_io import _fence, _apply_strip, _iter_files
from .parsers._types import FileNode
from .scoring import AppMDLResult, FileDetail, MDLMetric


# ----------------------------------------------------------------------
# Public dataclass
# ----------------------------------------------------------------------

@dataclass
class SuiteSharedConcatResult:
    """One suite = one library + N apps. ``library_nll`` is counted once."""
    library_nll: float
    library_tokens: int
    lib_dir: str | None
    lib_text_hash: str
    app_results: list[AppMDLResult]  # each carries the same lib_nll value


# ----------------------------------------------------------------------
# Ordering helpers
# ----------------------------------------------------------------------

def _topo_order(nodes: dict[str, FileNode]) -> list[FileNode]:
    """Kahn's algorithm on the 1-level dep graph.

    Returns nodes in dependency order (deps first). On cycle or empty graph,
    falls back to lexicographic order on ``rel_path`` for determinism.
    """
    if not nodes:
        return []

    indeg = {p: 0 for p in nodes}
    succ: dict[str, list[str]] = {p: [] for p in nodes}
    for p, n in nodes.items():
        for d in n.deps:
            if d in nodes and d != p:
                succ[d].append(p)
                indeg[p] += 1

    queue = deque(sorted(p for p, deg in indeg.items() if deg == 0))
    ordered: list[FileNode] = []
    visited = 0
    while queue:
        p = queue.popleft()
        ordered.append(nodes[p])
        visited += 1
        for nxt in sorted(succ[p]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    if visited != len(nodes):
        return sorted(nodes.values(), key=lambda n: n.rel_path)
    return ordered


def _read_dir_topo(
    dir_path: str,
    task: TaskConfig,
    strip_fn,
) -> tuple[str, list[str]]:
    """Read all code files under ``dir_path`` → markdown-fenced concat string.

    Returns ``(text, ordered_rel_paths)``. Ordering is dep-topo when the
    parser can build a graph for the directory in isolation; lex fallback
    on cycles or unsupported languages.

    **Scope filter**: nodes outside ``dir_path`` are excluded even if the
    parser resolved them via relative imports (e.g. webgen apps that
    ``import '../lib/...'`` pull lib files into the graph — those are
    measured separately in the lib prefix and must not double-count here).
    """
    if not dir_path or not os.path.isdir(dir_path):
        return "", []

    nodes: dict[str, FileNode] = {}

    parser = getattr(task, "parser_module", None)
    if parser is not None and hasattr(parser, "build_dep_graph"):
        try:
            nodes = parser.build_dep_graph(dir_path, None, task)
        except Exception:
            nodes = {}

    if not nodes:
        nodes = _read_dir_as_flat_nodes(dir_path, task)

    dir_prefix = os.path.abspath(dir_path) + os.sep
    nodes = {
        p: n for p, n in nodes.items()
        if os.path.abspath(p).startswith(dir_prefix)
    }
    # Filter dangling deps too (deps pointing outside dir_path) so topo is
    # computed on an internally consistent subgraph.
    for n in nodes.values():
        n.deps = [d for d in n.deps if d in nodes]

    failures: dict[str, int] = {}
    parts: list[str] = []
    rel_paths: list[str] = []
    for node in _topo_order(nodes):
        content = node.content
        if strip_fn is not None:
            content = _apply_strip(content, strip_fn, failures)
        parts.append(_fence(node.rel_path, content))
        rel_paths.append(node.rel_path)

    return "\n\n".join(parts), rel_paths


def _read_dir_as_flat_nodes(dir_path: str, task: TaskConfig) -> dict[str, FileNode]:
    """Fallback when parser has no ``build_dep_graph``: flat node dict, no deps."""
    nodes: dict[str, FileNode] = {}
    for fpath in _iter_files(dir_path, task):
        abs_path = os.path.abspath(fpath)
        rel = os.path.relpath(abs_path, dir_path)
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue
        nodes[abs_path] = FileNode(abs_path=abs_path, rel_path=rel, content=content)
    return nodes


# ----------------------------------------------------------------------
# Core scoring
# ----------------------------------------------------------------------

async def score_shared_concat(
    metric: MDLMetric,
    app_dirs: list[tuple[str, str]],
    lib_dir: str | None,
    *,
    task: TaskConfig | None = None,
    strip_comments: bool = True,
    precomputed_lib: tuple[float, int, str] | None = None,
) -> SuiteSharedConcatResult:
    """Score N apps that share one library.

    Args:
        app_dirs: list of ``(label, app_dir)`` — order = scoring order.
        lib_dir: shared library directory. ``None`` → fallback to
            ``Σ NLL(cat(app_files))`` with empty prefix.
        precomputed_lib: ``(library_nll, library_tokens, lib_text_hash)`` to
            reuse a prior measurement on the same lib bytes.

    Returns:
        ``SuiteSharedConcatResult`` with one ``AppMDLResult`` per app.
        ``library_nll`` is populated identically on every record (consumers
        must count it once at suite level).
    """
    if task is None:
        task = load_task_config("paperbench")

    strip_fn = task.parser_module.strip_comments if strip_comments else None

    lib_text, _lib_rel_paths = _read_dir_topo(lib_dir, task, strip_fn) if lib_dir else ("", [])
    lib_text_hash = hashlib.sha256(lib_text.encode("utf-8")).hexdigest()

    if precomputed_lib is not None:
        cached_nll, cached_tokens, cached_hash = precomputed_lib
        if cached_hash == lib_text_hash:
            library_nll, library_tokens = cached_nll, cached_tokens
        else:
            library_nll, library_tokens = await _score_lib_text(metric, lib_text)
    else:
        library_nll, library_tokens = await _score_lib_text(metric, lib_text)

    app_results: list[AppMDLResult] = []
    for label, app_dir in app_dirs:
        app_text, _app_rel_paths = _read_dir_topo(app_dir, task, strip_fn)
        if not app_text:
            raise ValueError(f"No app code files found under {app_dir}")

        if lib_text:
            concat_text = metric._concat(lib_text, app_text)
            concat_lp, concat_lt = await metric.get_prompt_logprobs(concat_text)
            prefix = min(library_tokens, len(concat_lt), len(concat_lp))
            app_lp = concat_lp[prefix:]
            app_tokens = max(0, len(concat_lt) - prefix)
        else:
            app_lp, app_lt = await metric.get_prompt_logprobs(app_text)
            app_tokens = len(app_lt)

        app_nll = -1.0 * float(np.sum(app_lp))
        npt = app_nll / app_tokens if app_tokens > 0 else float("nan")
        file_details = [FileDetail(
            rel_path="<app-concat>",
            nll=app_nll,
            tokens=app_tokens,
            nll_per_token=npt,
        )]

        result = metric._build_result(
            os.path.abspath(app_dir),
            app_nll, app_tokens,
            library_nll, library_tokens,
            file_details,
        )
        app_results.append(result)

    return SuiteSharedConcatResult(
        library_nll=library_nll,
        library_tokens=library_tokens,
        lib_dir=lib_dir,
        lib_text_hash=lib_text_hash,
        app_results=app_results,
    )


async def _score_lib_text(metric: MDLMetric, lib_text: str) -> tuple[float, int]:
    if not lib_text:
        return 0.0, 0
    lp, lt = await metric.get_prompt_logprobs(lib_text)
    return -1.0 * float(np.sum(lp)), len(lt)
