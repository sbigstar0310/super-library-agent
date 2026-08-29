"""
JS/TS/JSX/TSX parser for MDL.

Provides:
- ``strip_comments(code)``  → strip JS/TS/CSS comments while preserving string literals
- ``parse_imports(content)`` → ES6 import / export-from records
- ``build_dep_graph(app_dir, lib_dir, task)`` → 1-level abs_path → FileNode

Implementation: regex-based (matches the legacy `utils/dep_graph.py` and
`utils/mdl.py:strip_js_comments` exactly so webgen results stay bit-exact).
"""

from __future__ import annotations

import glob
import os
import re

from ._types import FileNode, ImportStmt


# ------------------------------------------------------------------
# Comment stripping
# ------------------------------------------------------------------

# Group 1 = string literals (kept). Group 2 = comments (removed).
_COMMENT_RE = re.compile(
    r"""("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)"""
    r"""|(/\*[\s\S]*?\*/|//[^\n]*)""",
)
_BLANK_LINES_RE = re.compile(r"\n\s*\n+")


def strip_comments(code: str) -> str:
    """Strip JS/TS/CSS comments while preserving string literals."""
    result = _COMMENT_RE.sub(lambda m: m.group(1) or "", code)
    result = _BLANK_LINES_RE.sub("\n", result)
    return result.strip()


# Back-compat alias (some callers import strip_js_comments by name)
strip_js_comments = strip_comments


# ------------------------------------------------------------------
# Import parsing
# ------------------------------------------------------------------

def parse_imports(content: str) -> list[ImportStmt]:
    """Parse import and export-from statements from JS/TS source."""
    results: list[ImportStmt] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def _add(specifier: str, names: list[str]) -> None:
        kind = _classify_specifier(specifier)
        key = (specifier, tuple(sorted(names)))
        if key not in seen:
            seen.add(key)
            results.append(ImportStmt(specifier=specifier, names=names, kind=kind))

    for m in re.finditer(r"import\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"]", content):
        names = [
            n.strip().split(" as ")[0].strip()
            for n in m.group(1).split(",")
            if n.strip() and not n.strip().startswith("type ")
        ]
        if names:
            _add(m.group(2), names)

    for m in re.finditer(r"import\s+(\w+)\s+from\s*['\"]([^'\"]+)['\"]", content):
        _add(m.group(2), [m.group(1)])

    for m in re.finditer(r"import\s+\*\s+as\s+\w+\s+from\s*['\"]([^'\"]+)['\"]", content):
        _add(m.group(1), [])

    for m in re.finditer(r"(?:^|[;\n])\s*import\s+['\"]([^'\"]+)['\"]", content, re.MULTILINE):
        _add(m.group(1), [])

    for m in re.finditer(r"export\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"]", content):
        names = [
            n.strip().split(" as ")[0].strip()
            for n in m.group(1).split(",")
            if n.strip() and not n.strip().startswith("type ")
        ]
        if names:
            _add(m.group(2), names)

    for m in re.finditer(r"export\s+\*\s+from\s*['\"]([^'\"]+)['\"]", content):
        _add(m.group(1), [])

    return results


def _classify_specifier(specifier: str) -> str:
    if specifier.startswith("./") or specifier.startswith("../"):
        return "local"
    return "external"


# ------------------------------------------------------------------
# Local path resolution
# ------------------------------------------------------------------

_JS_EXTENSIONS = (".tsx", ".ts", ".jsx", ".js")


def resolve_local_import(specifier: str, importer_dir: str) -> str | None:
    base = os.path.normpath(os.path.join(importer_dir, specifier))
    if os.path.isfile(base):
        return base
    for ext in _JS_EXTENSIONS:
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    if os.path.isdir(base):
        for ext in _JS_EXTENSIONS:
            candidate = os.path.join(base, "index" + ext)
            if os.path.isfile(candidate):
                return candidate
    return None


def _try_resolve_file(base_path: str) -> str | None:
    if os.path.isfile(base_path):
        return base_path
    for ext in _JS_EXTENSIONS:
        candidate = base_path + ext
        if os.path.isfile(candidate):
            return candidate
    return None


# ------------------------------------------------------------------
# ui-lib barrel resolution
# ------------------------------------------------------------------

def _parse_barrel_exports(barrel_path: str) -> tuple[dict[str, str], list[str]]:
    """Parse a barrel index file (e.g. ``src/layouts/index.ts``)."""
    try:
        with open(barrel_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return {}, []

    named: dict[str, str] = {}
    star_paths: list[str] = []

    for m in re.finditer(r"export\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"]", content):
        from_path = m.group(2)
        for item in m.group(1).split(","):
            item = item.strip()
            if not item or item.startswith("type "):
                continue
            if "default as" in item:
                match = re.search(r"default\s+as\s+(\w+)", item)
                if match:
                    named[match.group(1)] = from_path
            else:
                parts = item.split(" as ")
                export_name = parts[-1].strip()
                if export_name:
                    named[export_name] = from_path

    for m in re.finditer(r"export\s+\*\s+from\s*['\"]([^'\"]+)['\"]", content):
        star_paths.append(m.group(1))

    return named, star_paths


_REEXPORT_PROBE = re.compile(r"export\s+(?:\{[^}]*\}|\*)\s+from\s*['\"]")


def _is_barrel_file(abs_path: str) -> bool:
    """A file is a barrel iff it's named ``index.{js,jsx,ts,tsx}`` and
    contains at least one re-export-from statement."""
    name = os.path.basename(abs_path)
    stem, _, ext = name.rpartition(".")
    if not stem or stem != "index" or f".{ext}" not in _JS_EXTENSIONS:
        return False
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return False
    return bool(_REEXPORT_PROBE.search(content))


def _expand_barrel(
    barrel_abs: str,
    names: list[str],
    *,
    visited: set[str] | None = None,
    depth: int = 0,
    max_depth: int = 3,
) -> list[str]:
    """Resolve a barrel file + imported names to underlying implementation
    abs-paths. Recursively follows nested barrels up to ``max_depth``.

    If ``names`` is empty (default/side-effect import) the barrel itself is
    not expanded — caller keeps barrel as the dep.
    """
    if visited is None:
        visited = set()
    if barrel_abs in visited or depth >= max_depth or not names:
        return []
    visited.add(barrel_abs)

    barrel_dir = os.path.dirname(barrel_abs)
    named_exports, star_from_paths = _parse_barrel_exports(barrel_abs)
    out: list[str] = []

    for name in names:
        from_path = named_exports.get(name)
        candidates: list[str] = []
        if from_path:
            candidates.append(from_path)
        else:
            candidates.extend(star_from_paths)
        for fp in candidates:
            resolved = _try_resolve_file(os.path.normpath(os.path.join(barrel_dir, fp)))
            if not resolved:
                continue
            resolved_abs = os.path.abspath(resolved)
            if _is_barrel_file(resolved_abs):
                nested = _expand_barrel(
                    resolved_abs, [name],
                    visited=visited, depth=depth + 1, max_depth=max_depth,
                )
                for n in nested:
                    if n not in out:
                        out.append(n)
            else:
                if resolved_abs not in out:
                    out.append(resolved_abs)
            if from_path:
                break
    return out


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------

def _should_include(file_path: str, ignore_dirs: list[str], ignore_files: list[str]) -> bool:
    return (
        not any(d in file_path for d in ignore_dirs)
        and os.path.basename(file_path) not in ignore_files
    )


def _add_file_node(nodes: dict[str, FileNode], abs_path: str, base_dir: str) -> None:
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        nodes[abs_path] = FileNode(
            abs_path=abs_path,
            rel_path=os.path.relpath(abs_path, base_dir),
            content=content,
        )
    except Exception:
        pass


def build_dep_graph(app_dir: str, lib_dir: str | None = None, task=None) -> dict[str, FileNode]:
    """Build a 1-level dependency graph for all code files in *app_dir*.

    Library source files referenced by app imports are added but their own
    imports are NOT followed.

    Args:
        task: Optional TaskConfig. When provided, overrides extensions /
              ignore_dirs / ignore_files and lib_layout.src_subdir.
              When None, falls back to webgen defaults (back-compat).
    """
    if task is None:
        # Legacy fallback (webgen defaults)
        from ..configs import load_task_config
        task = load_task_config("webgen")

    extensions = task.code_extensions
    ignore_dirs = task.ignore_dirs
    ignore_files = task.ignore_files

    nodes: dict[str, FileNode] = {}

    for ext_pattern in extensions:
        for fpath in glob.glob(os.path.join(app_dir, "**", ext_pattern), recursive=True):
            if not _should_include(fpath, ignore_dirs, ignore_files):
                continue
            abs_path = os.path.abspath(fpath)
            rel_path = os.path.relpath(abs_path, app_dir)
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue
            nodes[abs_path] = FileNode(abs_path=abs_path, rel_path=rel_path, content=content)

    app_abs_paths = list(nodes.keys())
    for abs_path in app_abs_paths:
        node = nodes[abs_path]
        importer_dir = os.path.dirname(abs_path)
        deps: list[str] = []

        for imp in parse_imports(node.content):
            if imp.kind == "external":
                continue

            if imp.kind == "local":
                resolved = resolve_local_import(imp.specifier, importer_dir)
                if not resolved:
                    continue
                resolved_abs = os.path.abspath(resolved)

                expanded: list[str] = []
                if imp.names and _is_barrel_file(resolved_abs):
                    expanded = _expand_barrel(resolved_abs, imp.names)

                if expanded:
                    targets = expanded
                else:
                    targets = [resolved_abs]

                for tgt in targets:
                    if tgt not in deps:
                        deps.append(tgt)
                    if tgt not in nodes:
                        _add_file_node(nodes, tgt, app_dir)

        node.deps = deps

    return nodes
