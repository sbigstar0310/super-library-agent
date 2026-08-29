"""
Verbosity metric from SlopCodeBench (arXiv:2603.24755).

    Verbosity = | AST-grep Flagged Lines ∪ Clone Lines | / SLOC

Where:
  - AST-grep Flagged Lines: 1-indexed (file, line) tuples matched by rules in
    ``el-agent/config/sgconfig.yml`` via ``ast-grep scan --json=stream``.
  - Clone Lines: 1-indexed (file, line) tuples from jscpd-detected structural
    duplicates (both firstFile and secondFile ranges count).
  - SLOC: the set of non-blank, non-comment source lines per file.
  - Both line sets are intersected with SLOC before being unioned.
  - Result is bounded in [0, 1].

Public API: ``compute_app_verbosity(app_dir, lib_dir=None)`` — measure one
program. Pass a pre-built corpus root as ``app_dir`` with ``lib_dir=None``
to score (N apps + lib) as one codebase (scripts/metrics/scb_quality.py does this
for webgen).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

# Allow running this module directly and still resolve ``constants``
sys.path.insert(0, str(Path(__file__).resolve().parent))
from constants import IGNORE_DIRS, IGNORE_FILES  # noqa: E402

# ============================================================================
# Constants
# ============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AST_GREP_BIN = _REPO_ROOT / "node_modules" / ".bin" / "ast-grep"
# sgconfig.yml points to el-agent/config/slop_rules/ — one YAML per language
# (javascript.yml, typescript.yml, tsx.yml). ast-grep loads all of them via -c.
_DEFAULT_SGCONFIG = _REPO_ROOT / "el-agent" / "config" / "sgconfig.yml"

# Only count JS/TS sources; mdl.py's broader CODE_EXTENSIONS mixes in
# .css/.json/.html which don't belong in ast-grep matching or jscpd clone detection.
JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}

# jscpd ignore pattern, plus lock files.
_JSCPD_IGNORE = (
    "**/node_modules/**,**/.vite/**,**/dist/**,"
    "**/*.md,**/package-lock.json,**/pnpm-lock.yaml,**/yarn.lock"
)


# ============================================================================
# SLOC classification
#
# Line-level state machine that walks JS/TS source and returns the set of
# 1-indexed lines that contain non-blank, non-comment code. Approximate —
# does not handle regex literals or escaped-newline continuation — but
# sufficient for aggregate comparisons.
# ============================================================================


def _classify_sloc_lines(text: str) -> set[int]:
    """Return 1-indexed line numbers that contain non-blank, non-comment code."""
    sloc: set[int] = set()
    in_block_comment = False
    for line_num, line in enumerate(text.splitlines(), start=1):
        i = 0
        n = len(line)
        found_code = False
        in_string = False
        string_char: str | None = None
        in_template = False
        while i < n:
            if in_block_comment:
                end = line.find("*/", i)
                if end == -1:
                    break  # extends to next line
                i = end + 2
                in_block_comment = False
                continue
            c = line[i]
            if in_string:
                if c == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c == string_char:
                    in_string = False
                    string_char = None
                i += 1
                continue
            if in_template:
                if c == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c == "`":
                    in_template = False
                    i += 1
                    continue
                if c == "$" and i + 1 < n and line[i + 1] == "{":
                    # Skip to matching '}' (flat scan, no nested braces)
                    depth = 1
                    i += 2
                    while i < n and depth > 0:
                        if line[i] == "{":
                            depth += 1
                        elif line[i] == "}":
                            depth -= 1
                        i += 1
                    continue
                i += 1
                continue
            if c == "/" and i + 1 < n:
                nxt = line[i + 1]
                if nxt == "/":
                    break  # line comment
                if nxt == "*":
                    in_block_comment = True
                    i += 2
                    continue
            if c in ('"', "'"):
                in_string = True
                string_char = c
                found_code = True
                i += 1
                continue
            if c == "`":
                in_template = True
                found_code = True
                i += 1
                continue
            if not c.isspace():
                found_code = True
            i += 1
        if found_code:
            sloc.add(line_num)
    return sloc


# ============================================================================
# File discovery + SLOC map
# ============================================================================


def _enumerate_files(root: Path) -> list[Path]:
    """Walk ``root`` and yield .js/.jsx/.ts/.tsx files, honoring ignore lists."""
    root = Path(root)
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for fn in filenames:
            if fn in IGNORE_FILES:
                continue
            if Path(fn).suffix not in JS_EXTENSIONS:
                continue
            out.append(Path(dirpath) / fn)
    return out


def _build_sloc_map(files: list[Path], root: Path) -> dict[str, set[int]]:
    """Return ``{rel_path_from_root: sloc_lines}``."""
    root = Path(root).resolve()
    sloc_map: dict[str, set[int]] = {}
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(fp.resolve().relative_to(root))
        sloc_map[rel] = _classify_sloc_lines(text)
    return sloc_map


def _normalize_path(p: str | Path, root: Path) -> str | None:
    """Normalize a tool-reported path to ``root``-relative string.

    Handles both absolute (ast-grep) and cwd-relative (jscpd) forms.
    Returns ``None`` if the path can't be resolved inside ``root``.
    """
    root = Path(root).resolve()
    candidate = Path(p)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        return str(candidate.relative_to(root))
    except ValueError:
        return None


# ============================================================================
# Multi-target scan context
#
# Builds (sloc_map, normalize_fn) for a list of target directories. When there
# are multiple targets, keys are prefixed with the target's directory name
# (e.g. "000056/src/App.jsx", "ui-lib/src/Foo.jsx") to avoid collisions.
# Single-target case keeps unprefixed keys for maintain_feedback compatibility.
# ============================================================================


def _build_targets_context(
    targets: list[Path],
) -> tuple[dict[str, set[int]], Callable[[str], str | None]]:
    multi = len(targets) > 1
    sloc_map: dict[str, set[int]] = {}
    for t in targets:
        files = _enumerate_files(t)
        t_sloc = _build_sloc_map(files, t)
        if multi:
            for k, v in t_sloc.items():
                sloc_map[f"{t.name}/{k}"] = v
        else:
            sloc_map.update(t_sloc)

    def normalize_fn(raw: str) -> str | None:
        for t in targets:
            rel = _normalize_path(raw, t)
            if rel is not None and rel != "" and ".." not in rel.split(os.sep):
                return f"{t.name}/{rel}" if multi else rel
        return None

    return sloc_map, normalize_fn


# ============================================================================
# Tool invocations
# ============================================================================


def _run_ast_grep(targets: list[Path], sgconfig: Path) -> list[dict]:
    """Run ``ast-grep scan --json=stream -c sgconfig`` on ``targets``."""
    if not _AST_GREP_BIN.exists():
        raise FileNotFoundError(
            f"ast-grep binary not found at {_AST_GREP_BIN}. "
            f"Run `npm install --save-dev @ast-grep/cli` at repo root."
        )
    if not Path(sgconfig).exists():
        raise FileNotFoundError(f"sgconfig not found: {sgconfig}")

    cmd = [str(_AST_GREP_BIN), "scan", "--json=stream", "-c", str(sgconfig)]
    cmd.extend(str(t) for t in targets)

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(
            f"ast-grep scan failed (exit {result.returncode}):\n{result.stderr}"
        )

    matches: list[dict] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            matches.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return matches


def _run_jscpd(targets: list[Path], min_clone_lines: int) -> list[dict]:
    """Run ``npx jscpd`` on ``targets`` and return the ``duplicates[]`` array."""
    if not targets:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "npx", "jscpd",
            "--min-lines", str(min_clone_lines),
            "--ignore", _JSCPD_IGNORE,
            "--reporters", "json",
            "--output", tmp,
            "--format", "javascript,typescript,jsx,tsx",
            "--silent",
        ]
        cmd.extend(str(t) for t in targets)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT))
        # jscpd can return non-zero when clones exceed threshold; rely on the
        # report file's existence instead of exit code.
        report = Path(tmp) / "jscpd-report.json"
        if not report.exists():
            if result.returncode != 0:
                raise RuntimeError(
                    f"jscpd did not produce a report (exit {result.returncode}):\n"
                    f"{result.stderr}"
                )
            return []
        try:
            data = json.loads(report.read_text())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"jscpd report is not valid JSON: {e}") from e
        return data.get("duplicates", [])


# ============================================================================
# AST-grep match parsing
#
# All valid match iteration goes through ``_iter_ast_grep_matches`` so the
# filtering logic (path normalization, sloc intersection, range parsing) lives
# in exactly one place. Line extraction and detail extraction are thin
# consumers of the same generator.
# ============================================================================


def _iter_ast_grep_matches(
    matches: list[dict],
    sloc_map: dict[str, set[int]],
    normalize_fn: Callable[[str], str | None],
) -> Iterator[tuple[str, int, int, str, dict]]:
    """Yield ``(key, start_1, end_1, rule_id, raw_match)`` for each valid match.

    ``start_1`` / ``end_1`` are the match's start and end line numbers, converted
    from ast-grep's 0-indexed reporting to 1-indexed inclusive bounds.
    """
    for m in matches:
        raw_file = m.get("file")
        if not raw_file:
            continue
        key = normalize_fn(raw_file)
        if key is None or key not in sloc_map:
            continue
        rng = m.get("range") or {}
        start = (rng.get("start") or {}).get("line")
        end = (rng.get("end") or {}).get("line")
        if start is None or end is None:
            continue
        yield key, start + 1, end + 1, m.get("ruleId", "<unknown>"), m


def _ast_grep_flagged_lines(
    matches: list[dict],
    sloc_map: dict[str, set[int]],
    normalize_fn: Callable[[str], str | None],
) -> tuple[set[tuple[str, int]], Counter]:
    """Return (flagged_line_set, by_rule_counter) filtered to SLOC lines."""
    flagged: set[tuple[str, int]] = set()
    by_rule: Counter = Counter()
    for key, start_1, end_1, rule_id, _ in _iter_ast_grep_matches(
        matches, sloc_map, normalize_fn
    ):
        file_sloc = sloc_map[key]
        hit = False
        for ln in range(start_1, end_1 + 1):
            if ln in file_sloc:
                flagged.add((key, ln))
                hit = True
        if hit:
            by_rule[rule_id] += 1
    return flagged, by_rule


def _ast_grep_details(
    matches: list[dict],
    sloc_map: dict[str, set[int]],
    normalize_fn: Callable[[str], str | None],
    snippet_max_chars: int = 200,
) -> list[dict]:
    """Return per-match detail list for feedback generation."""
    details: list[dict] = []
    for key, start_1, end_1, rule_id, m in _iter_ast_grep_matches(
        matches, sloc_map, normalize_fn
    ):
        snippet = (m.get("text") or "").strip()
        if len(snippet) > snippet_max_chars:
            snippet = snippet[:snippet_max_chars].rstrip() + "…"
        details.append(
            {
                "rule_id": rule_id,
                "file": key,
                "line": start_1,
                "end_line": end_1,
                "snippet": snippet,
                "message": m.get("message", ""),
            }
        )
    return details


# ============================================================================
# jscpd duplicate parsing
# ============================================================================


def _parse_clone_side(
    side: dict,
    sloc_map: dict[str, set[int]],
    normalize_fn: Callable[[str], str | None],
) -> tuple[str, int, int] | None:
    """Return ``(key, start_line, end_line)`` for one side of a clone, or None."""
    name = side.get("name") or ""
    if not name:
        return None
    key = normalize_fn(name)
    if key is None or key not in sloc_map:
        return None
    start_loc = side.get("startLoc") or {}
    end_loc = side.get("endLoc") or {}
    start = start_loc.get("line")
    end = end_loc.get("line")
    if start is None or end is None:
        # Fallback to top-level start/end (1-indexed per v1 verification)
        start = side.get("start")
        end = side.get("end")
        if start is None or end is None:
            return None
    return key, int(start), int(end)


def _iter_clone_sides(
    duplicates: list[dict],
    sloc_map: dict[str, set[int]],
    normalize_fn: Callable[[str], str | None],
) -> Iterator[tuple[dict, tuple[str, int, int], tuple[str, int, int]]]:
    """Yield ``(dup, side1, side2)`` for each duplicate whose both sides resolve."""
    for dup in duplicates:
        side1 = _parse_clone_side(dup.get("firstFile") or {}, sloc_map, normalize_fn)
        side2 = _parse_clone_side(dup.get("secondFile") or {}, sloc_map, normalize_fn)
        if side1 is None or side2 is None:
            continue
        yield dup, side1, side2


def _clone_lines(
    duplicates: list[dict],
    sloc_map: dict[str, set[int]],
    normalize_fn: Callable[[str], str | None],
) -> set[tuple[str, int]]:
    """Return the clone (file, line) set, filtered to SLOC lines."""
    out: set[tuple[str, int]] = set()
    for _, side1, side2 in _iter_clone_sides(duplicates, sloc_map, normalize_fn):
        for key, start, end in (side1, side2):
            file_sloc = sloc_map[key]
            for ln in range(start, end + 1):
                if ln in file_sloc:
                    out.add((key, ln))
    return out


def _clone_details(
    duplicates: list[dict],
    sloc_map: dict[str, set[int]],
    normalize_fn: Callable[[str], str | None],
    fragment_max_lines: int = 6,
    fragment_max_chars: int = 400,
) -> list[dict]:
    """Return per-pair clone detail list for feedback generation."""
    details: list[dict] = []
    for dup, (k1, s1, e1), (k2, s2, e2) in _iter_clone_sides(
        duplicates, sloc_map, normalize_fn
    ):
        fragment = (dup.get("fragment") or "").strip()
        lines = fragment.split("\n")
        if len(lines) > fragment_max_lines:
            fragment = "\n".join(lines[:fragment_max_lines]) + "\n…"
        if len(fragment) > fragment_max_chars:
            fragment = fragment[:fragment_max_chars].rstrip() + "…"
        details.append(
            {
                "lines": dup.get("lines", 0),
                "first_file": k1,
                "first_start": s1,
                "first_end": e1,
                "second_file": k2,
                "second_start": s2,
                "second_end": e2,
                "fragment_preview": fragment,
            }
        )
    return details


# ============================================================================
# Stats aggregation
# ============================================================================


def _compute_stats(
    flagged: set[tuple[str, int]],
    clones: set[tuple[str, int]],
    sloc_map: dict[str, set[int]],
    flagged_by_rule: Counter,
    sgconfig: Path,
) -> dict[str, Any]:
    union = flagged | clones
    total_sloc = sum(len(s) for s in sloc_map.values())
    return {
        "verbosity": (len(union) / total_sloc) if total_sloc > 0 else 0.0,
        "flagged_lines": len(flagged),
        "clone_lines": len(clones),
        "union_lines": len(union),
        "total_sloc": total_sloc,
        "files_scanned": len(sloc_map),
        "files_with_violations": len({p for p, _ in union}),
        "flagged_by_rule": dict(flagged_by_rule),
        "rules_file": str(sgconfig),
    }


# ============================================================================
# Core helper: scan a target group and return stats
# ============================================================================


def _resolve_sgconfig(sgconfig: str | Path | None) -> Path:
    return Path(sgconfig) if sgconfig is not None else _DEFAULT_SGCONFIG


def _compute_verbosity_for_targets(
    targets: list[Path],
    sgconfig_path: Path,
    min_clone_lines: int,
    include_details: bool = True,
) -> dict[str, Any]:
    """Run ast-grep + jscpd on ``targets`` and return verbosity stats.

    Multi-target calls prefix sloc_map keys with each target's directory name.
    Single-target calls keep unprefixed keys (backward-compatible with
    ``maintain_feedback``).
    """
    resolved = [Path(t).resolve() for t in targets]
    sloc_map, normalize_fn = _build_targets_context(resolved)

    matches = _run_ast_grep(resolved, sgconfig_path)
    duplicates = _run_jscpd(resolved, min_clone_lines)

    flagged, by_rule = _ast_grep_flagged_lines(matches, sloc_map, normalize_fn)
    clones = _clone_lines(duplicates, sloc_map, normalize_fn)

    result = _compute_stats(flagged, clones, sloc_map, by_rule, sgconfig_path)

    if include_details:
        result["flagged_samples"] = _ast_grep_details(matches, sloc_map, normalize_fn)
        result["clones_detail"] = _clone_details(duplicates, sloc_map, normalize_fn)

    return result


# ============================================================================
# Averaging (for experiment mode)
# ============================================================================


# ============================================================================
# Public API
# ============================================================================


def compute_app_verbosity(
    app_dir: str | Path,
    lib_dir: str | Path | None = None,
    sgconfig: str | Path | None = None,
    min_clone_lines: int = 5,
    include_details: bool = True,
) -> dict[str, Any]:
    """Compute verbosity for a single program (app, optionally + lib).

    Args:
        app_dir: Path to the app directory.
        lib_dir: Optional library directory. If provided and exists, it is
            scanned alongside the app (ast-grep + jscpd run over both).
        sgconfig: Optional override for the ast-grep sgconfig.yml. Defaults
            to ``el-agent/config/sgconfig.yml``.
        min_clone_lines: jscpd ``--min-lines`` threshold (default 5).
        include_details: If True (default), the returned dict carries
            ``flagged_samples`` and ``clones_detail``. When ``lib_dir`` is
            provided, detail file paths are prefixed with the target directory
            name; otherwise they stay unprefixed.

    Returns:
        dict with verbosity, flagged_lines, clone_lines, union_lines,
        total_sloc, files_scanned, files_with_violations, flagged_by_rule,
        rules_file (+ flagged_samples, clones_detail when include_details).
    """
    app_path = Path(app_dir).resolve()
    if not app_path.exists():
        raise FileNotFoundError(f"app_dir not found: {app_path}")

    sgconfig_path = _resolve_sgconfig(sgconfig).resolve()

    targets: list[Path] = [app_path]
    if lib_dir is not None:
        lib_path = Path(lib_dir).resolve()
        if lib_path.exists():
            targets.append(lib_path)

    return _compute_verbosity_for_targets(
        targets, sgconfig_path, min_clone_lines, include_details=include_details
    )
