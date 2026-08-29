"""Symbol-level usage counting against a Python lib package.

Pipeline:
    collect_lib_definitions(lib_dir)                       # what's defined
    count_symbol_usage(lib_dir, consumer_dirs, cfg)        # hybrid scan
    aggregate(per_consumer)                                 # collapse to {symbol: total}
    compute_dead_symbols(snapshots, current_round, grace)  # decide what's dead

Detection model — hybrid (AST for apps, textual for lib-internal):

  Consumer dir == lib_dir → textual ``\\bNAME\\b`` scan
    For lib's own modules (excluding ``__init__.py`` barrels), every
    occurrence of a defined bare name credits all canonicals sharing
    that name (def-site line excluded). This is the only reliable way
    to detect intra-file references like ``ISO8601_RE`` used by
    ``parse_iso8601`` defined in the same file — Python looks those up
    in the module namespace, so there is no import statement to track.

  Consumer dir != lib_dir → AST explicit-import scan
    Walks every ``ImportFrom``/``Import`` node and resolves to canonical
    names:
      - ``from lib.<mod> import X``  →  ``lib.<mod>.X``
      - ``from lib.<mod> import *``  →  every canonical in ``lib.<mod>``
      - ``from lib import X``        →  resolved via ``lib/__init__.py``
                                         re-export table; if X is itself
                                         a sub-module, every canonical
                                         in that module
      - ``import lib.<mod>``         →  every canonical in ``lib.<mod>``
                                         (``import lib.<mod> as Y`` too)
      - ``import lib``               →  conservatively marks every
                                         canonical alive (we can't tell
                                         which attribute the app reads)
    String/comment occurrences and bare-name collisions inside the app
    do NOT count — only real import statements do. This eliminates the
    false-alive failure mode where an app has its own ``lib/`` package
    (e.g. sqlmap) whose symbols share names with our extracted lib.

Why this split:

  - Apps frequently vendor their own packages and reuse common names
    (``cache``, ``conf``, ``util.get_bool_opt``). Textual matching on
    apps mis-identifies those as live references → real dead symbols
    stay protected and the deletion feedback never fires. AST forces
    the import to actually point at our lib.
  - Lib internals do NOT have name collisions (a lib module is its own
    namespace). Same-file references — the ``ISO8601_RE`` pattern —
    require textual scanning because they aren't imports at all. This
    is also the conservative direction: if anyone in the lib touches
    a symbol, we keep it alive, otherwise the deletion would break a
    helper used by another exported symbol.

Lib's own ``__init__.py`` is skipped during the scan; otherwise the
barrel re-exports would keep every symbol artificially alive. The
re-export table IS read separately to resolve ``from lib import X``
imports from app code (X gets mapped back to its canonical).

Canonical symbol form: ``"<pkg>.<rel_module>.<name>"`` where
``rel_module`` is the dotted path of the defining ``.py`` file relative
to the lib root (e.g. ``"text"`` or ``"sub.helpers"``).
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------- types ---


@dataclass(frozen=True)
class LibraryUsageConfig:
    """Per-task knobs for the usage counter."""

    lib_package_name: str = "lib"
    code_extensions: tuple[str, ...] = ("*.py",)
    ignore_dirs: tuple[str, ...] = (
        "__pycache__",
        ".venv",
        "venv",
        ".git",
        "build",
        "dist",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        ".ruff_cache",
        "embeddings",
        "tests",
    )


@dataclass(frozen=True)
class _Definition:
    """One top-level name defined in a lib module."""

    canonical: str   # "lib.<rel_module>.<name>"
    bare_name: str   # the textual name to grep for
    file: Path       # absolute path of the defining .py file
    def_lineno: int  # the def/class/Assign line number (1-based)


@dataclass
class LibraryUsageSnapshot:
    """Frozen-in-time record of which symbols a library exposed and who
    referenced each one.
    """

    lib_version_round: int
    exported_symbols: list[str]
    per_consumer_usage: dict[str, dict[str, int]]
    aggregate: dict[str, int]

    def to_json(self) -> dict:
        return {
            "lib_version_round": self.lib_version_round,
            "exported_symbols": list(self.exported_symbols),
            "per_consumer_usage": {
                k: dict(v) for k, v in self.per_consumer_usage.items()
            },
            "aggregate": dict(self.aggregate),
        }

    @classmethod
    def from_json(cls, data: dict) -> "LibraryUsageSnapshot":
        return cls(
            lib_version_round=int(data["lib_version_round"]),
            exported_symbols=list(data.get("exported_symbols") or []),
            per_consumer_usage={
                k: dict(v) for k, v in (data.get("per_consumer_usage") or {}).items()
            },
            aggregate=dict(data.get("aggregate") or {}),
        )


# ------------------------------------------------------- definition scan ---


def _module_dotted_path(lib_dir: Path, file_path: Path) -> str:
    """Return the dotted submodule path relative to ``lib_dir``."""
    rel = file_path.relative_to(lib_dir).with_suffix("")
    return ".".join(rel.parts)


def collect_lib_definitions(
    lib_dir: Path,
    config: LibraryUsageConfig | None = None,
) -> list[_Definition]:
    """Walk the lib package and return every top-level name defined in
    its non-``__init__`` modules.

    ``__init__.py`` files are skipped: their top-level statements are
    typically re-exports (``from lib.text import str_to_list``) that
    rebind names already defined elsewhere — they do not define new
    symbols and counting them as a separate canonical would inflate the
    candidate set.
    """
    cfg = config or LibraryUsageConfig(lib_package_name=lib_dir.name)
    pkg = cfg.lib_package_name
    out: list[_Definition] = []

    for fpath in _iter_python_files(Path(lib_dir), cfg, exclude_init_under=None):
        if fpath.name == "__init__.py":
            continue
        try:
            src = fpath.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=str(fpath))
        except (OSError, SyntaxError):
            continue
        modname = _module_dotted_path(Path(lib_dir), fpath)

        for node in tree.body:
            names_lines: list[tuple[str, int]] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names_lines.append((node.name, node.lineno))
            elif isinstance(node, ast.Assign):
                line = node.lineno
                for tgt in node.targets:
                    names_lines.extend(_extract_target_names(tgt, line))
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    names_lines.append((node.target.id, node.lineno))
            for name, lineno in names_lines:
                out.append(_Definition(
                    canonical=f"{pkg}.{modname}.{name}",
                    bare_name=name,
                    file=fpath,
                    def_lineno=lineno,
                ))
    return out


def _extract_target_names(node: ast.expr, lineno: int) -> list[tuple[str, int]]:
    """Recursively collect ``Name.id`` from an Assign target (handles
    nested tuple/list unpacking).
    """
    if isinstance(node, ast.Name):
        return [(node.id, lineno)]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[tuple[str, int]] = []
        for elt in node.elts:
            out.extend(_extract_target_names(elt, lineno))
        return out
    return []


def read_lib_exports(
    lib_dir: Path,
    config: LibraryUsageConfig | None = None,
) -> list[str]:
    """Return the canonical names of every symbol the lib defines.

    Under the textual-search model, "exports" == "definitions" — there
    is no ``__init__.py`` re-export contract being honored. Names with
    leading ``_`` are included; aliases are irrelevant (the textual
    name is what gets searched).
    """
    cfg = config or LibraryUsageConfig(lib_package_name=Path(lib_dir).name)
    return sorted({d.canonical for d in collect_lib_definitions(Path(lib_dir), cfg)})


# ------------------------------------------------------- consumer scan ---


def _iter_python_files(
    root: Path,
    cfg: LibraryUsageConfig,
    *,
    exclude_init_under: Path | None = None,
) -> Iterable[Path]:
    """Yield ``.py`` files under ``root``, honoring ``ignore_dirs``.

    When ``exclude_init_under`` is supplied, every ``__init__.py``
    located under that path is skipped — used to keep lib's own barrel
    files out of the consumer scan (they would always keep every
    re-exported name alive).
    """
    exclude_under = (
        exclude_init_under.resolve() if exclude_init_under else None
    )
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in cfg.ignore_dirs]
        for fn in filenames:
            if not any(Path(fn).match(pat) for pat in cfg.code_extensions):
                continue
            fpath = Path(dirpath) / fn
            if (
                exclude_under is not None
                and fn == "__init__.py"
                and (
                    fpath.resolve() == exclude_under / "__init__.py"
                    or exclude_under in fpath.resolve().parents
                )
            ):
                continue
            yield fpath


def count_symbol_usage(
    lib_dir: Path | str,
    consumer_dirs: list[Path | str],
    config: LibraryUsageConfig | None = None,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Hybrid usage scan — textual for ``lib_dir``, AST for app dirs.

    For ``consumer_dir == lib_dir`` the scan is textual ``\\bNAME\\b`` so
    intra-file references (a regex referenced by a function in the same
    module) count toward alive — those references aren't imports at all.

    For every other consumer dir, only real ``ImportFrom`` / ``Import``
    nodes that resolve to a lib canonical are counted. ``import lib.X``
    or ``from lib.X import *`` conservatively marks every symbol in
    ``lib.X`` alive; bare ``import lib`` marks every canonical alive
    because we can't know which attribute the app reads.

    ``consumer_dirs`` should include ``lib_dir`` so symbols used only
    inside the lib stay alive. Lib's own ``__init__.py`` is skipped
    during the textual scan (barrel re-exports are not real consumers)
    but its ``from .X import Y`` lines ARE read once to build the
    ``from lib import Y`` re-export table for the AST scan.
    """
    lib_dir = Path(lib_dir)
    cfg = config or LibraryUsageConfig(lib_package_name=lib_dir.name)
    defs = collect_lib_definitions(lib_dir, cfg)
    canonicals = sorted({d.canonical for d in defs})

    by_bare: dict[str, list[_Definition]] = {}
    for d in defs:
        by_bare.setdefault(d.bare_name, []).append(d)

    if not defs:
        return canonicals, {}

    # Module-level lookup table: "lib.<rel_module>" -> {canonicals}
    by_module: dict[str, set[str]] = {}
    for d in defs:
        modkey = ".".join(d.canonical.split(".")[:-1])
        by_module.setdefault(modkey, set()).add(d.canonical)

    canonical_set = set(canonicals)
    init_reexports = _read_init_reexports(lib_dir, canonical_set)
    lib_dir_resolved = lib_dir.resolve()

    # One regex per textual scan — alternation of every defined bare name.
    pattern = r"\b(" + "|".join(re.escape(n) for n in by_bare) + r")\b"
    big_rx = re.compile(pattern)

    per_consumer: dict[str, dict[str, int]] = {}
    seen: set[Path] = set()
    for raw_root in consumer_dirs:
        root = Path(raw_root)
        if not root.is_dir():
            continue
        is_lib_root = root.resolve() == lib_dir.resolve()
        excl = lib_dir if is_lib_root else None
        for fpath in _iter_python_files(root, cfg, exclude_init_under=excl):
            resolved = fpath.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if is_lib_root:
                counts = _count_in_file(fpath, by_bare, big_rx)
            else:
                counts = _ast_imports_in_file(
                    fpath, canonical_set, by_module, init_reexports,
                    lib_dir_resolved,
                )
            if counts:
                per_consumer[str(fpath)] = counts
    return canonicals, per_consumer


def _count_in_file(
    file_path: Path,
    by_bare: dict[str, list[_Definition]],
    big_rx: re.Pattern[str],
) -> dict[str, int]:
    """Textual ``\\bNAME\\b`` count for one lib-internal file.

    Skips only the *def line* of any name defined in this file (so
    same-file in-body references count toward alive).
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}

    file_resolved = file_path.resolve()
    local_def_lines: dict[str, set[int]] = {}
    for defs in by_bare.values():
        for d in defs:
            if d.file.resolve() == file_resolved:
                local_def_lines.setdefault(d.bare_name, set()).add(d.def_lineno)

    counts: dict[str, int] = {}
    for line_idx, line in enumerate(content.splitlines(), start=1):
        for m in big_rx.finditer(line):
            name = m.group(1)
            if line_idx in local_def_lines.get(name, ()):
                continue
            for d in by_bare[name]:
                counts[d.canonical] = counts.get(d.canonical, 0) + 1
    return counts


def _read_init_reexports(
    lib_dir: Path,
    canonical_set: set[str],
) -> dict[str, set[str]]:
    """Read ``lib/__init__.py``'s re-exports.

    Returns ``exposed_name -> {canonical}``. ``ImportFrom`` lines are
    honored in three forms:
      - ``from lib.<sub> import X``   (absolute)
      - ``from .<sub> import X``      (relative; resolved against
                                       ``lib_dir.name``)
      - ``from . import <sub>``       (re-exports a sub-module)

    The exposed name is ``alias.asname or alias.name``; the resolved
    canonical is ``<lib.subpath>.<alias.name>`` and must exist in
    ``canonical_set`` (otherwise the alias points at a sub-module rather
    than a single symbol — handled separately by the caller).

    Used to translate app code's ``from lib import X`` into the original
    canonical defined deep in the lib.
    """
    init = lib_dir / "__init__.py"
    if not init.is_file():
        return {}
    try:
        tree = ast.parse(init.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return {}
    pkg = lib_dir.name
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        # Resolve "from .X import Y" / "from . import Y" against the lib
        # package; "from lib.X import Y" is taken at face value.
        if node.level and node.level >= 1:
            if node.module:
                base = f"{pkg}.{node.module}"
            else:
                base = pkg
        elif node.module:
            base = node.module
        else:
            continue
        for alias in node.names or []:
            if alias.name == "*":
                continue
            exposed = alias.asname or alias.name
            canon = f"{base}.{alias.name}"
            if canon in canonical_set:
                out.setdefault(exposed, set()).add(canon)
    return out


def _is_inside_vendored_lib(
    file_path: Path,
    lib_dir_resolved: Path,
) -> bool:
    """True if the file lives under a directory that vendors its own
    ``lib/`` package (one whose ``__init__.py`` is not our ``lib_dir``).

    Such a file's ``from lib import X`` and ``import lib`` statements
    bind to the app's vendored ``lib`` (sqlmap's pattern), not to ours.
    Resolving them against our re-export table would produce false-alive
    counts — every sqlmap-internal ``from lib import decode_base64`` would
    keep our ``lib.convert.decode_base64`` artificially live.

    Sub-module imports like ``from lib.convert import decode_base64`` are
    still resolved (sqlmap doesn't have a sub-module named ``convert``).
    Only the ambiguous bare ``from lib import …`` / ``import lib`` forms
    are skipped for vendored-lib files.
    """
    try:
        for parent in file_path.resolve().parents:
            if parent.name != "lib":
                continue
            init = parent / "__init__.py"
            if init.is_file() and parent.resolve() != lib_dir_resolved:
                return True
    except OSError:
        return False
    return False


def _ast_imports_in_file(
    file_path: Path,
    canonical_set: set[str],
    by_module: dict[str, set[str]],
    init_reexports: dict[str, set[str]],
    lib_dir_resolved: Path,
) -> dict[str, int]:
    """AST scan: which lib canonicals does this app file explicitly import?

    Returns ``{canonical: count}`` where each count is the number of import
    statements that resolve to that canonical (typically 1, but multiple
    imports of the same symbol from different statements add up).

    Wildcards (``from lib.X import *``, ``import lib.X``) credit every
    canonical in the targeted module. A bare ``import lib`` credits every
    canonical — we can't tell statically which attribute is read.

    Files inside a vendored ``lib/`` package (see
    ``_is_inside_vendored_lib``) skip ``from lib import …`` and bare
    ``import lib`` resolution — those forms bind to the app's own
    package, not ours.
    """
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return {}

    skip_bare_lib = _is_inside_vendored_lib(file_path, lib_dir_resolved)
    counts: dict[str, int] = {}

    def _bump(canon: str) -> None:
        counts[canon] = counts.get(canon, 0) + 1

    def _bump_module(modkey: str) -> None:
        for c in by_module.get(modkey, ()):
            _bump(c)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if not (mod == "lib" or mod.startswith("lib.")):
                continue
            if mod == "lib" and skip_bare_lib:
                continue
            names = node.names or []
            if names and names[0].name == "*":
                _bump_module(mod)
                continue
            for alias in names:
                # 1) "from lib.<sub> import X" → lib.<sub>.X
                canon = f"{mod}.{alias.name}"
                if canon in canonical_set:
                    _bump(canon)
                # 2) "from lib.<sub> import <subsub>" — alias is itself a
                # sub-module under lib.<sub>. Bump everything in it.
                _bump_module(f"{mod}.{alias.name}")
                # 3) "from lib import X" — resolve via __init__ re-exports
                if mod == "lib":
                    for c in init_reexports.get(alias.name, ()):
                        _bump(c)
                    # X may also itself be a top-level submodule
                    _bump_module(f"lib.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names or []:
                name = alias.name
                if name == "lib":
                    if skip_bare_lib:
                        continue
                    # Bare ``import lib`` exposes the entire surface area —
                    # any ``lib.X.Y`` access is possible. Mark every
                    # canonical alive (conservative, rare in practice).
                    for c in canonical_set:
                        _bump(c)
                elif name.startswith("lib."):
                    _bump_module(name)

    return counts


def aggregate(per_consumer: dict[str, dict[str, int]]) -> dict[str, int]:
    """Sum per-symbol counts across all consumer files."""
    out: dict[str, int] = {}
    for counts in per_consumer.values():
        for sym, n in counts.items():
            out[sym] = out.get(sym, 0) + n
    return out


# ------------------------------------------------------- dead inference ---


def is_dead(
    symbol: str,
    snapshots_by_round: dict[int, "LibraryUsageSnapshot"],
    current_round: int,
    grace_rounds: int,
) -> bool:
    """True iff ``symbol`` was exported AND had aggregate==0 in every one of
    the last ``grace_rounds`` snapshots strictly before ``current_round``.
    """
    rounds = sorted(r for r in snapshots_by_round if r < current_round)[-grace_rounds:]
    if len(rounds) < grace_rounds:
        return False
    for r in rounds:
        snap = snapshots_by_round[r]
        if symbol not in snap.exported_symbols:
            return False
        if snap.aggregate.get(symbol, 0) != 0:
            return False
    return True


def compute_dead_symbols(
    snapshots_by_round: dict[int, "LibraryUsageSnapshot"],
    current_round: int,
    grace_rounds: int,
) -> list[str]:
    """Return all dead symbols sorted by canonical path."""
    rounds = sorted(r for r in snapshots_by_round if r < current_round)[-grace_rounds:]
    if len(rounds) < grace_rounds:
        return []
    candidates = set(snapshots_by_round[rounds[-1]].exported_symbols)
    return sorted(
        s for s in candidates
        if is_dead(s, snapshots_by_round, current_round, grace_rounds)
    )
