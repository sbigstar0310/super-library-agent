"""File traversal helpers for MDL.

All functions take a ``TaskConfig`` (loaded via ``load_task_config(name)``)
that supplies the file-discovery filters: code extensions, ignore_dirs,
ignore_files, max_file_size_kb, marker_files. When ``task`` is omitted, the
``webgen`` config is used as a back-compat default — explicit task is
preferred in new code.

Three output shapes are useful in different contexts:

- ``read_files(dir, task)``       → list of raw file contents (no paths)
- ``read_files_with_paths(dir)``  → list of ``(rel_path, content)`` tuples
- ``read_dir_to_text(dir, task)`` → single markdown-fenced corpus string,
                                    same format used by ``format_dep_context``

``read_dir_to_text`` and ``read_app_code_files`` both accept an optional
``strip_fn`` that is applied **per-file before** the markdown fence is
emitted. Stripping after the fence is broken for JS/TS (the parser's
string-literal regex treats triple-backtick fences as a template literal
and swallows comments inside).
"""

from __future__ import annotations

import glob
import logging
import os
import random
from pathlib import Path

from .configs import TaskConfig, load_task_config

_log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _task(task: TaskConfig | None) -> TaskConfig:
    """Resolve ``task`` arg, falling back to webgen for back-compat."""
    return task if task is not None else load_task_config("webgen")


def _is_acceptable(
    filepath: str,
    ignore_dirs: list[str],
    ignore_files: list[str],
    max_file_size_kb: int | None,
) -> bool:
    if any(d in filepath for d in ignore_dirs):
        return False
    if os.path.basename(filepath) in ignore_files:
        return False
    if max_file_size_kb is not None:
        try:
            if os.path.getsize(filepath) > max_file_size_kb * 1024:
                return False
        except OSError:
            return False
    return True


def _iter_files(dir_path: str, task: TaskConfig, *, shuffle: bool = False):
    """Yield acceptable file paths under ``dir_path`` per the task config."""
    for ext in task.code_extensions:
        files = glob.glob(os.path.join(dir_path, "**", ext), recursive=True)
        if shuffle:
            random.shuffle(files)
        for fpath in files:
            if _is_acceptable(
                fpath, task.ignore_dirs, task.ignore_files, task.max_file_size_kb,
            ):
                yield fpath


def _apply_strip(content: str, strip_fn, failures: dict[str, int]) -> str:
    """Run ``strip_fn`` with raw-fallback; mutate ``failures`` on error."""
    if strip_fn is None:
        return content
    try:
        return strip_fn(content)
    except Exception as e:
        failures[type(e).__name__] = failures.get(type(e).__name__, 0) + 1
        return content


def _log_strip_failures(where: str, failures: dict[str, int], attempts: int) -> None:
    if not failures:
        return
    summary = ", ".join(f"{k}={v}" for k, v in failures.items())
    _log.warning(
        "%s strip_fn fell back to raw on %d/%d files (%s); "
        "those files keep comments/docstrings in the measured text",
        where, sum(failures.values()), attempts, summary,
    )


def _fence(rel_path: str, content: str) -> str:
    """The canonical markdown wrap: ``### path\\n```\\ncontent\\n``` ``."""
    return f"### {rel_path}\n```\n{content}\n```"


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def is_valid_codebase(path: str | Path, task: TaskConfig | None = None) -> bool:
    """True iff any of ``task.marker_files`` (e.g. package.json) exists at path."""
    task = _task(task)
    p = Path(path)
    return p.is_dir() and any((p / m).exists() for m in task.marker_files)


def get_code_files(
    dir_path: str,
    extensions: list[str] | None = None,
    ignore_dirs: list[str] | None = None,
    ignore_files: list[str] | None = None,
    *,
    task: TaskConfig | None = None,
) -> list[str]:
    """Return raw file contents from ``dir_path`` matching the task extensions.

    Legacy explicit filter args are honored when present (back-compat); when
    omitted, falls back to ``task`` (default: webgen).
    """
    task = _task(task)
    # Explicit-arg back-compat: temporarily override task fields if any of
    # the legacy filter args is supplied. This path is used by a few
    # callers in scripts/ that pass extensions=["*.py"] etc.
    if extensions is not None or ignore_dirs is not None or ignore_files is not None:
        task = _task_override(
            task, extensions, ignore_dirs, ignore_files,
        )
    out: list[str] = []
    for fpath in _iter_files(dir_path, task):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                out.append(f.read())
        except Exception:
            continue
    return out


def _task_override(
    task: TaskConfig,
    extensions: list[str] | None,
    ignore_dirs: list[str] | None,
    ignore_files: list[str] | None,
) -> TaskConfig:
    """Return a shallow-copied TaskConfig with optional field overrides."""
    from dataclasses import replace
    kwargs = {}
    if extensions is not None:
        kwargs["code_extensions"] = extensions
    if ignore_dirs is not None:
        kwargs["ignore_dirs"] = ignore_dirs
    if ignore_files is not None:
        kwargs["ignore_files"] = ignore_files
    return replace(task, **kwargs) if kwargs else task


def read_dir_to_text(
    dir_path: str | None,
    extensions: list[str] | None = None,
    ignore_dirs: list[str] | None = None,
    ignore_files: list[str] | None = None,
    shuffle: bool = False,
    *,
    task: TaskConfig | None = None,
    strip_fn=None,
) -> str:
    """Read all source files into a single markdown-fenced corpus string.

    Format: ``### {rel_path}\\n```\\n{content}\\n``` `` joined by ``\\n\\n``.

    If ``strip_fn`` is given, it runs per-file *before* wrapping. (Post-wrap
    stripping is broken for JS — the parser regex swallows fences as
    template literals.) Strip failures fall back to raw + warning.
    """
    if not dir_path or not os.path.exists(dir_path):
        return ""

    task = _task(task)
    if extensions is not None or ignore_dirs is not None or ignore_files is not None:
        task = _task_override(task, extensions, ignore_dirs, ignore_files)

    parts: list[str] = []
    failures: dict[str, int] = {}
    attempts = 0
    for fpath in _iter_files(dir_path, task, shuffle=shuffle):
        rel = os.path.relpath(fpath, dir_path)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue
        if strip_fn is not None:
            attempts += 1
            content = _apply_strip(content, strip_fn, failures)
        parts.append(_fence(rel, content))

    _log_strip_failures("read_dir_to_text", failures, attempts)
    return "\n\n".join(parts)


def read_app_code_files(
    app_dir: str,
    strip_fn=None,
    *,
    task: TaskConfig | None = None,
) -> list[tuple[str, str]]:
    """Read all code files as ``(rel_path, content)`` tuples (sorted by path).

    Strip failures fall back to raw + warning (matches ``read_dir_to_text``).
    Files that fail to *read* are dropped entirely with a separate warning.
    """
    task = _task(task)

    files: list[tuple[str, str]] = []
    drops: dict[str, int] = {}
    strip_failures: dict[str, int] = {}
    strip_attempts = 0
    for fpath in sorted(_iter_files(app_dir, task)):
        rel = os.path.relpath(fpath, app_dir)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            drops[type(e).__name__] = drops.get(type(e).__name__, 0) + 1
            continue
        if strip_fn is not None:
            strip_attempts += 1
            content = _apply_strip(content, strip_fn, strip_failures)
        files.append((rel, content))

    if drops:
        summary = ", ".join(f"{k}={v}" for k, v in drops.items())
        _log.warning(
            "read_app_code_files dropped %d unreadable file(s) under %s (%s)",
            sum(drops.values()), app_dir, summary,
        )
    _log_strip_failures("read_app_code_files", strip_failures, strip_attempts)
    return files


def format_dep_context(dep_nodes) -> str:
    """Format dep-graph nodes in the same markdown fence as ``read_dir_to_text``.

    Comments are NOT stripped — dep context is intentionally raw in the
    dep-aware MDL formula (see ``scoring.py``).
    """
    if not dep_nodes:
        return ""
    return "\n\n".join(_fence(node.rel_path, node.content) for node in dep_nodes)


# ----------------------------------------------------------------------
# Maintainability (LOC / tokens) — no LLM
# ----------------------------------------------------------------------

def get_maintainability_metrics(
    app_dir: str,
    library_dir: str | None = None,
    *,
    task: TaskConfig | None = None,
    strip_comments: bool = True,
) -> dict:
    """LOC + token counts for ``app_dir`` and (optionally) ``library_dir``.

    Comments and docstrings are stripped by default (matches cloc/scc/tokei
    conventions). Strip failures fall back to raw with a warning so the
    numbers aren't silently inflated.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    task = _task(task)
    strip_fn = task.parser_module.strip_comments if strip_comments else None

    def _prep(contents: list[str]) -> list[str]:
        if strip_fn is None:
            return contents
        failures: dict[str, int] = {}
        out = [_apply_strip(c, strip_fn, failures) for c in contents]
        _log_strip_failures("get_maintainability_metrics", failures, len(contents))
        return out

    def _loc(contents: list[str]) -> int:
        return sum(1 for c in contents for line in c.splitlines() if line.strip())

    def _tok(contents: list[str]) -> int:
        # disallowed_special=() — source files may legitimately contain
        # literals like '<|endoftext|>' (tokenizer code, tests, comments).
        return sum(len(enc.encode(c, disallowed_special=())) for c in contents)

    app_files = _prep(get_code_files(app_dir, task=task))
    result = {
        "app_tokens": _tok(app_files),
        "app_loc": _loc(app_files),
        "lib_tokens": 0,
        "lib_loc": 0,
    }
    if library_dir:
        lib_files = _prep(get_code_files(library_dir, task=task))
        result["lib_tokens"] = _tok(lib_files)
        result["lib_loc"] = _loc(lib_files)
    return result
