"""PaperBench static gate — compile + import-resolution correctness check.

Given a corpus of PaperBench submissions (``tasks_dir/<tid>/submission/``) and
an optional shared library, verifies — **without executing any submission or
library code** — that

1. every ``.py`` file parses (syntax check), and
2. every ``import`` that targets either the shared ``lib`` package or a module
   *internal* to the submission resolves: the target module file exists and, for
   ``from M import name``, ``name`` is defined at the module top level.

Why static only: PaperBench submissions are research code (torch/jax training
scripts). Importing them for real would run arbitrary top-level side effects and
require a GPU environment we don't have. The grader itself never executes the
code, so an import that resolves textually is all the pipeline needs. We instead
parse each file with :func:`ast.parse` / :func:`compile` (in memory — no ``.pyc``
files are written, so passing a read-only backup dir never mutates it).

Import convention (see ``prompts/paperbench/paperbench.py`` and
``mswe_agents/base_coding_agent.py``): the library is NOT pip-installable. It is
imported via ``PYTHONPATH`` as ``from lib.<subpkg> import X`` / ``import
lib.<subpkg>``. On disk the ``lib`` *package* lives at ``<lib_dir>/lib`` (i.e.
``lib_dir`` is the ``PYTHONPATH`` entry and ``lib`` is the top-level package
inside it). Submissions are ordinary package trees rooted at ``submission/``, so
internal imports (``from fre.rl import Actor``, ``from . import utils``) resolve
against ``submission/``.

What is intentionally NOT checked (and why it cannot produce false failures):

* **Third-party / stdlib imports** (``numpy``, ``torch``, ``os``, …) are skipped
  entirely — there is no environment to resolve them against, and their presence
  or absence is not what this gate is about. An import is treated as internal
  only when its top-level name resolves to a file/dir under ``submission/`` (or is
  the ``lib`` package when a ``lib_dir`` is given); everything else is skipped.
* **Names pulled in via ``from x import *``** — the star target's exports cannot
  be enumerated statically, so any name whose module (transitively) does a star
  import is accepted. Conservative by design.
* **Dynamically created names** (``globals()[...] = ...``, ``setattr`` on a
  module, ``exec``) are invisible to AST and would over-report; to stay on the
  safe side, namespace packages without ``__init__.py`` and modules with star
  imports pass their name checks unconditionally.

Design bias: **a false gate failure is worse than a lenient pass.** When
resolution is ambiguous the gate passes. Definitions guarded by ``try/except`` or
``if TYPE_CHECKING:`` at module top level still count (the collector descends into
control-flow bodies but not into function/class scopes).

Public API::

    from utils.gates.paperbench_static import run_paperbench_static_gate, GateResult

CLI::

    uv --project el-agent run python -m utils.gates.paperbench_static \
        --tasks-dir <dir> [--lib-dir <dir>] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

__all__ = ["GateResult", "run_paperbench_static_gate"]


# Directories never walked for source files: VCS metadata, byte-code caches, the
# cocoindex artefacts stored beside a lib, node deps, and — critically — the
# read-only ``paper/`` snapshot that sits beside a PaperBench ``submission/``
# (grader-only; not part of the submitted code).
_SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".cocoindex_code", "node_modules", "paper"}
)

# The top-level package name the shared library is imported as.
_LIB_PKG = "lib"


@dataclass
class GateResult:
    """Per-task verdict. ``ok`` is ``True`` iff ``errors`` is empty."""

    ok: bool
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def _iter_py_files(root: Path):
    """Yield every ``*.py`` under ``root``, skipping ``_SKIP_DIRS`` subtrees."""
    for path in sorted(root.rglob("*.py")):
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def _resolve_module(root: Path, parts: list[str]) -> tuple[Path | None, bool]:
    """Resolve a dotted module ``parts`` under sys.path entry ``root``.

    Returns ``(path, is_package)``:
      * regular package  -> (``.../__init__.py``, True)
      * namespace package (dir, no ``__init__``) -> (``.../<dir>``, True)
      * module           -> (``.../<name>.py``, False)
      * unresolved       -> (None, False)
    """
    if not parts:
        # The root itself, treated as a package for relative-import purposes.
        init = root / "__init__.py"
        return (init if init.exists() else root), True
    base = root.joinpath(*parts)
    if base.is_dir():
        init = base / "__init__.py"
        return (init if init.exists() else base), True
    mod = base.with_suffix(".py")
    if mod.is_file():
        return mod, False
    return None, False


# --------------------------------------------------------------------------- #
# Top-level name collection
# --------------------------------------------------------------------------- #
def _collect_target_names(target: ast.expr, out: set[str]) -> None:
    """Collect bound names from an assignment target (handles tuple/list unpack)."""
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _collect_target_names(elt, out)
    elif isinstance(target, ast.Starred):
        _collect_target_names(target.value, out)


def _collect_module_names(body: list[ast.stmt], names: set[str]) -> bool:
    """Collect module-top-level bound names from ``body``.

    Descends into control-flow compound statements (``if``/``try``/``with``/
    ``for``/``while``) so definitions guarded by ``try/except`` or
    ``if TYPE_CHECKING:`` still count, but does NOT descend into function/class
    bodies (those are nested scopes). Returns ``True`` if any ``from x import *``
    was seen (in which case the module's exported name set is unknowable and the
    caller should treat every name lookup as satisfied).

    ``__all__ = [...]`` string entries are added as names too: they express
    re-export intent, and counting them keeps the gate lenient.
    """
    has_star = False
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                _collect_target_names(tgt, names)
            if any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            ):
                _add_dunder_all(node.value, names)
        elif isinstance(node, ast.AnnAssign):
            # Count the annotated name whether or not it has a value; lenient.
            _collect_target_names(node.target, names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                # `import a.b.c`      -> binds `a`
                # `import a.b as c`   -> binds `c`
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    has_star = True
                else:
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.If):
            has_star |= _collect_module_names(node.body, names)
            has_star |= _collect_module_names(node.orelse, names)
        elif isinstance(node, ast.Try):
            has_star |= _collect_module_names(node.body, names)
            for handler in node.handlers:
                has_star |= _collect_module_names(handler.body, names)
            has_star |= _collect_module_names(node.orelse, names)
            has_star |= _collect_module_names(node.finalbody, names)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            has_star |= _collect_module_names(node.body, names)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            has_star |= _collect_module_names(node.body, names)
            has_star |= _collect_module_names(node.orelse, names)
        elif isinstance(node, ast.While):
            has_star |= _collect_module_names(node.body, names)
            has_star |= _collect_module_names(node.orelse, names)
    return has_star


def _add_dunder_all(value: ast.expr, names: set[str]) -> None:
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        for elt in value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                names.add(elt.value)


# --------------------------------------------------------------------------- #
# Parsing (cached, in-memory — never writes .pyc)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _module_names(path_str: str) -> tuple[frozenset[str], bool]:
    """Return ``(top_level_names, has_star_import)`` for a module file.

    A parse failure here means the file has a syntax error; it is reported
    separately by the syntax pass, so we return an empty, star-flagged result
    (lenient) to avoid double-reporting the same file as an import failure.
    """
    path = Path(path_str)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), path_str)
    except SyntaxError:
        return frozenset(), True
    names: set[str] = set()
    star = _collect_module_names(tree.body, names)
    return frozenset(names), star


def _name_available(root: Path, parts: list[str], name: str) -> bool:
    """Is ``name`` importable from module ``parts`` (resolved under ``root``)?"""
    mod_path, is_pkg = _resolve_module(root, parts)
    if mod_path is None:
        return False  # module itself missing — caller reports that separately
    if mod_path.suffix == ".py":  # regular module or package __init__.py
        names, star = _module_names(str(mod_path))
        if star or name in names:
            return True
    if is_pkg:
        pkg_dir = root.joinpath(*parts) if parts else root
        # A submodule/subpackage of this name satisfies `from pkg import name`.
        if (pkg_dir / f"{name}.py").is_file() or (pkg_dir / name).is_dir():
            return True
        # Namespace package (no __init__) — exports are unknowable: be lenient.
        if not (pkg_dir / "__init__.py").exists():
            return True
    return False


# --------------------------------------------------------------------------- #
# Per-file import checking
# --------------------------------------------------------------------------- #
def _package_parts(sub_dir: Path, file_path: Path) -> list[str]:
    """Dotted package parts of the package *containing* ``file_path``.

    ``fre/rl.py`` -> ``['fre']``; ``fre/__init__.py`` -> ``['fre']``;
    ``run.py`` -> ``[]``.
    """
    rel = file_path.relative_to(sub_dir)
    return list(rel.parts[:-1])


def _check_imports_in_file(
    file_path: Path,
    sub_dir: Path,
    lib_dir: Path | None,
    rel_label: str,
) -> list[str]:
    """Return a list of unresolved-import error strings for one file."""
    errors: list[str] = []
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return errors  # reported by the syntax pass

    # Is `top` a module/package directly under the submission root?
    def _is_internal(top: str) -> bool:
        return (sub_dir / f"{top}.py").is_file() or (sub_dir / top).is_dir()

    def _lib_parts(parts: list[str]) -> list[str] | None:
        # `lib`, `lib.nn`, ... -> path parts under lib_dir. lib pkg == lib_dir/lib.
        if lib_dir is not None and parts and parts[0] == _LIB_PKG:
            return parts
        return None

    pkg_parts = _package_parts(sub_dir, file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                lp = _lib_parts(parts)
                if lp is not None:
                    if _resolve_module(lib_dir, lp)[0] is None:
                        errors.append(
                            f"{rel_label}:{node.lineno}: unresolved lib module "
                            f"'{alias.name}' (import {alias.name})"
                        )
                elif _is_internal(parts[0]):
                    if _resolve_module(sub_dir, parts)[0] is None:
                        errors.append(
                            f"{rel_label}:{node.lineno}: unresolved internal module "
                            f"'{alias.name}' (import {alias.name})"
                        )
                # else: third-party / stdlib -> skip

        elif isinstance(node, ast.ImportFrom):
            level = node.level
            module = node.module  # may be None for `from . import x`

            if level > 0:
                # Relative import -> always internal to the submission.
                drop = level - 1
                if drop > len(pkg_parts):
                    continue  # points above the submission root -> unresolvable, skip
                base = pkg_parts[: len(pkg_parts) - drop]
                mod_parts = base + (module.split(".") if module else [])
                root = sub_dir
                origin = "internal"
            else:
                parts = module.split(".") if module else []
                lp = _lib_parts(parts)
                if lp is not None:
                    mod_parts, root, origin = lp, lib_dir, "lib"
                elif parts and _is_internal(parts[0]):
                    mod_parts, root, origin = parts, sub_dir, "internal"
                else:
                    continue  # third-party / stdlib -> skip

            assert root is not None
            mod_path, _is_pkg = _resolve_module(root, mod_parts)
            dotted = ".".join(mod_parts) if mod_parts else "."
            if mod_path is None:
                errors.append(
                    f"{rel_label}:{node.lineno}: unresolved {origin} module "
                    f"'{dotted}' (from {'.' * level}{module or ''} import ...)"
                )
                continue

            # Module exists; verify each imported name.
            for alias in node.names:
                if alias.name == "*":
                    continue  # module existence is all we can check for star
                if not _name_available(root, mod_parts, alias.name):
                    errors.append(
                        f"{rel_label}:{node.lineno}: name '{alias.name}' not found "
                        f"in {origin} module '{dotted}'"
                    )
    return errors


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _syntax_errors(root: Path, label_root: Path) -> list[str]:
    """Compile every ``.py`` under ``root`` in memory; collect SyntaxErrors."""
    errors: list[str] = []
    for path in _iter_py_files(root):
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
            compile(src, str(path), "exec")  # syntax check; does NOT run the code
        except SyntaxError as e:
            try:
                rel = path.relative_to(label_root)
            except ValueError:
                rel = path
            errors.append(f"SyntaxError in {rel}:{e.lineno}: {e.msg}")
        except ValueError as e:
            # e.g. source containing null bytes — treat as a syntax-level defect.
            errors.append(f"CompileError in {path.relative_to(label_root)}: {e}")
    return errors


def run_paperbench_static_gate(
    tasks_dir: str,
    lib_dir: str | None,
    *,
    only: set[str] | None = None,
) -> dict[str, GateResult]:
    """Statically gate every PaperBench submission under ``tasks_dir``.

    Args:
        tasks_dir: dir containing ``<tid>/submission/`` subdirs.
        lib_dir: dir whose ``lib/`` subdir is the shared library package, or
            ``None`` for a zero-shot corpus with no shared library.
        only: if given, only these task ids are gated (the rest are omitted from
            the result — used for repair re-gating).

    Returns:
        ``{task_id: GateResult}``. A task is ``ok`` iff it has no syntax errors
        and no unresolved lib/internal import.
    """
    tasks_path = Path(tasks_dir)
    lib_path = Path(lib_dir) if lib_dir else None

    # The shared lib is compiled once; a syntax error there breaks every app.
    lib_syntax_errors: list[str] = []
    if lib_path is not None:
        pkg_root = lib_path / _LIB_PKG
        if pkg_root.exists():
            lib_syntax_errors = _syntax_errors(pkg_root, lib_path)
        else:
            # Fall back to scanning lib_path directly (defensive; keeps working if
            # a caller passes the package dir itself).
            lib_syntax_errors = _syntax_errors(lib_path, lib_path)

    results: dict[str, GateResult] = {}
    for task_path in sorted(p for p in tasks_path.iterdir() if p.is_dir()):
        tid = task_path.name
        if only is not None and tid not in only:
            continue
        sub_dir = task_path / "submission"
        if not sub_dir.is_dir():
            results[tid] = GateResult(
                ok=False, errors=[f"missing submission dir: {sub_dir}"]
            )
            continue

        errors: list[str] = []
        errors.extend(_syntax_errors(sub_dir, sub_dir))
        # Lib syntax errors are only relevant if this app actually imports lib;
        # but a broken lib is a genuine, unambiguous defect, so surface it on any
        # task once a lib is present. (Real graded libs compile clean.)
        errors.extend(f"[lib] {e}" for e in lib_syntax_errors)

        for py in _iter_py_files(sub_dir):
            rel_label = str(py.relative_to(sub_dir))
            errors.extend(_check_imports_in_file(py, sub_dir, lib_path, rel_label))

        results[tid] = GateResult(ok=not errors, errors=errors)

    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main() -> int:
    parser = argparse.ArgumentParser(
        description="PaperBench static correctness gate (compile + import "
        "resolution; no code execution)."
    )
    parser.add_argument(
        "--tasks-dir",
        required=True,
        help="dir containing <tid>/submission/ subdirs",
    )
    parser.add_argument(
        "--lib-dir",
        default=None,
        help="dir whose lib/ subdir is the shared library (omit for zero-shot)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="write the full {tid: {ok, errors}} result as JSON here",
    )
    args = parser.parse_args()

    results = run_paperbench_static_gate(args.tasks_dir, args.lib_dir)

    n_ok = sum(1 for r in results.values() if r.ok)
    for tid in sorted(results):
        r = results[tid]
        status = "ok" if r.ok else "FAIL"
        print(f"[{status}] {tid}")
        if not r.ok:
            for err in r.errors:
                print(f"    {err}")
    print(f"\n{n_ok}/{len(results)} tasks passed the static gate.")

    if args.json_out:
        payload = {tid: asdict(r) for tid, r in results.items()}
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")

    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
