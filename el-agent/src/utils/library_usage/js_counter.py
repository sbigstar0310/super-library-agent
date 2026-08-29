"""JS/JSX library-symbol usage counting.

Mirrors the public API of ``counter.py`` (Python) but resolves imports
the JavaScript way: explicit ``import`` statements with a source path
that must resolve into ``lib_dir`` to count.

Canonical naming
----------------
A canonical symbol is the **exposed name** in the lib's barrel
(``lib/src/index.js`` or ``lib/index.js``). For webgen the barrel
re-exports everything the public API offers, e.g. ::

    export { useLocalStorage } from './hooks/useLocalStorage.js';
    export { default as Footer } from './components/Footer.jsx';

so the canonicals are ``useLocalStorage``, ``Footer``, etc. Symbols
defined in a lib module but NOT re-exported by the barrel are not part
of the public API and are intentionally excluded.

Resolution model
----------------
For every consumer ``.js``/``.jsx`` file (apps + lib internals minus
the barrel itself) every ``import … from '<src>'`` is checked:

  1. Resolve ``<src>`` against the importing file's directory.
  2. If the resolved path is the barrel file → look up each named
     specifier in the re-export table → bump that canonical.
  3. If the resolved path is a non-barrel lib module file → reverse
     lookup the barrel table to find which canonical(s) come from
     that file:
       * default import → barrel entry with ``default as <name>``
       * named import   → barrel entry with the matching original name
  4. ``import * as ns from '<barrel>'`` → bump every canonical.

Imports to paths outside ``lib_dir`` (``react``, ``zlib``, sibling app
files) are silently ignored.

Lib-internal cross-file imports
-------------------------------
``hooks/useLocalStorageState.js`` may ``import { saveJSON } from
'../utils/localStorage.js'``. We count those too, so symbols used only
by other lib modules stay alive even when no app imports them. The
barrel file itself is excluded from the consumer scan — including it
would keep every re-exported symbol artificially alive.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from .counter import LibraryUsageConfig  # reuse the dataclass


_DEFAULT_JS_EXTS = ("*.js", "*.jsx", "*.ts", "*.tsx", "*.mjs", "*.cjs")
_RESOLVE_EXTS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
_INDEX_BASENAMES = tuple(f"index{e}" for e in _RESOLVE_EXTS)


# --------------------------------------------------------------- helpers ---


def _js_config(lib_dir: Path, cfg: LibraryUsageConfig | None) -> LibraryUsageConfig:
    if cfg is not None:
        return cfg
    return LibraryUsageConfig(
        lib_package_name=lib_dir.name,
        code_extensions=_DEFAULT_JS_EXTS,
    )


def _iter_js_files(root: Path, cfg: LibraryUsageConfig) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in cfg.ignore_dirs]
        for fn in filenames:
            if not any(Path(fn).match(pat) for pat in cfg.code_extensions):
                continue
            yield Path(dirpath) / fn


def _find_barrel(lib_dir: Path) -> Path | None:
    """Locate the lib's public barrel.

    Conventional layout for webgen is ``<lib_dir>/src/index.{js,jsx,ts,tsx}``;
    fall back to ``<lib_dir>/index.*`` if absent.
    """
    candidates = []
    for sub in ("src", ""):
        base = lib_dir / sub if sub else lib_dir
        for ext in _RESOLVE_EXTS:
            cand = base / f"index{ext}"
            if cand.is_file():
                candidates.append(cand)
    return candidates[0] if candidates else None


def _resolve_import_path(importer: Path, src: str) -> Path | None:
    """Resolve ``src`` (as written in JS) against ``importer``'s dir.

    Non-relative bare specifiers (``'react'``, ``'ui-lib'``) return None
    — caller decides whether to count them via a different mechanism.
    Relative paths get extension/index fall-through resolution.
    """
    if not (src.startswith("./") or src.startswith("../") or src.startswith("/")):
        return None
    raw = (importer.parent / src).resolve()

    # Exact file (extension included)
    if raw.is_file():
        return raw

    # Append extensions
    for ext in _RESOLVE_EXTS:
        cand = raw.with_suffix(ext) if raw.suffix else Path(str(raw) + ext)
        if cand.is_file():
            return cand

    # Directory with index.*
    if raw.is_dir():
        for base in _INDEX_BASENAMES:
            cand = raw / base
            if cand.is_file():
                return cand

    return None


# --------------------------------------------------------- import parsing ---


# Captures three import shapes in one regex; group meanings below.
_IMPORT_RX = re.compile(
    r"""
    ^[\t ]*import
    (?:                                # optional clause group
        \s+(?:type\s+)?                # 'type' keyword (TS)
        (
            \*\s+as\s+[A-Za-z_$][\w$]*      # group 1: namespace 'as NS'
          | \{[^}]*\}                       # group 1: named { ... }
          | [A-Za-z_$][\w$]*               # group 1: default
            (?:\s*,\s*\{[^}]*\})?            #   optional ', { ... }' tail
          | [A-Za-z_$][\w$]*\s*,\s*\*\s+as\s+[A-Za-z_$][\w$]*  # default + ns
        )
        \s+from
    )?
    \s+['"]([^'"]+)['"]                # group 2: source
    """,
    re.MULTILINE | re.VERBOSE | re.DOTALL,
)


def _parse_named_clause(clause: str) -> list[tuple[str, str]]:
    """Parse ``{ A, B as C, default as D }`` → ``[(orig, local), ...]``.

    ``orig`` is the name as exported by the target file, ``local`` is the
    name the importer binds it to. For our counting purposes only ``orig``
    matters (it maps back to the barrel canonical), but ``local`` is kept
    for completeness.
    """
    inner = clause.strip()
    if inner.startswith("{"):
        inner = inner[1:]
    if inner.endswith("}"):
        inner = inner[:-1]
    out: list[tuple[str, str]] = []
    for part in inner.split(","):
        s = part.strip()
        if not s:
            continue
        # strip TS `type` keyword on individual specifier
        if s.startswith("type "):
            s = s[5:].strip()
        m = re.match(r"([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)", s)
        if m:
            out.append((m.group(1), m.group(2)))
        else:
            m2 = re.match(r"([A-Za-z_$][\w$]*)", s)
            if m2:
                out.append((m2.group(1), m2.group(1)))
    return out


def _parse_clause(clause: str) -> dict:
    """Classify an import clause.

    Returns one of:
      {"kind": "default", "name": str}
      {"kind": "named", "specifiers": [(orig, local), ...]}
      {"kind": "namespace", "local": str}
      {"kind": "default_named", "default": str,
       "specifiers": [(orig, local), ...]}
    """
    c = clause.strip()
    if c.startswith("*"):
        m = re.match(r"\*\s+as\s+([A-Za-z_$][\w$]*)", c)
        return {"kind": "namespace", "local": m.group(1) if m else ""}
    if c.startswith("{"):
        return {"kind": "named", "specifiers": _parse_named_clause(c)}
    # default OR default + named
    m_def = re.match(r"([A-Za-z_$][\w$]*)", c)
    default_name = m_def.group(1) if m_def else ""
    rest = c[m_def.end():] if m_def else c
    rest = rest.strip()
    if rest.startswith(","):
        rest = rest[1:].strip()
        if rest.startswith("{"):
            return {
                "kind": "default_named",
                "default": default_name,
                "specifiers": _parse_named_clause(rest),
            }
    return {"kind": "default", "name": default_name}


def _imports_in_file(path: Path) -> list[tuple[dict | None, str]]:
    """Return ``[(clause_dict_or_None, source_path), …]`` for one file.

    ``clause_dict_or_None`` is None for side-effect imports
    (``import './style.css'``).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: list[tuple[dict | None, str]] = []
    for m in _IMPORT_RX.finditer(text):
        clause_raw = m.group(1)
        src = m.group(2)
        if clause_raw is None:
            out.append((None, src))
        else:
            out.append((_parse_clause(clause_raw), src))
    return out


# ---------------------------------------------------------- barrel parse ---


# Captures ``export { X } from './path'`` and ``export * from './path'``.
_REEXPORT_RX = re.compile(
    r"""
    ^[\t ]*export
    \s+
    (?:
        (\*)                                   # group 1: star
      | (\{[^}]*\})                            # group 2: named clause
    )
    \s+from\s+['"]([^'"]+)['"]                 # group 3: source
    """,
    re.MULTILINE | re.VERBOSE | re.DOTALL,
)


def _parse_barrel(barrel: Path) -> tuple[dict[str, tuple[Path, str]], list[tuple[Path, str]]]:
    """Walk barrel re-exports.

    Returns:
      named: exposed_name -> (source_file, original_name)
             - ``export { X } from './a.js'``         → X -> (a.js, X)
             - ``export { default as Foo } from`` ... → Foo -> (file, 'default')
             - ``export { X as Y } from ...``         → Y -> (file, X)
      stars: list of (source_file, 'star') for ``export * from …``;
             callers should treat as: every symbol the source defines
             is exposed under its own name.
    """
    named: dict[str, tuple[Path, str]] = {}
    stars: list[tuple[Path, str]] = []

    try:
        text = barrel.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return named, stars

    for m in _REEXPORT_RX.finditer(text):
        star = m.group(1)
        clause = m.group(2)
        src = m.group(3)
        resolved = _resolve_import_path(barrel, src)
        if resolved is None:
            continue
        if star:
            stars.append((resolved, "star"))
            continue
        for orig, local in _parse_named_clause(clause):
            # `default as Foo`  → orig='default', local='Foo'
            # `Foo`              → orig='Foo',     local='Foo'
            # `Foo as Bar`       → orig='Foo',     local='Bar'
            named[local] = (resolved, orig)
    return named, stars


def _scan_module_exports(path: Path) -> list[str]:
    """Cheap regex over a lib module to enumerate names exported via
    ``export {function,const,class,let,var} NAME`` plus a single
    ``default``. Used only to expand ``export * from …`` in the barrel.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    names: set[str] = set()
    for m in re.finditer(
        r"^[\t ]*export\s+(?:default\s+)?(?:async\s+)?"
        r"(?:function|class|const|let|var)\s+([A-Za-z_$][\w$]*)",
        text, re.MULTILINE,
    ):
        names.add(m.group(1))
    if re.search(r"^[\t ]*export\s+default\b", text, re.MULTILINE):
        names.add("default")
    # `export { A, B as C }` directly inside the module
    for m in re.finditer(r"^[\t ]*export\s+(\{[^}]*\})", text, re.MULTILINE):
        for orig, local in _parse_named_clause(m.group(1)):
            names.add(local)
    return sorted(names)


# ------------------------------------------------------------- public API ---


def read_lib_exports(
    lib_dir: Path,
    config: LibraryUsageConfig | None = None,
) -> list[str]:
    """Return the lib's public canonicals (barrel-exposed names)."""
    lib_dir = Path(lib_dir)
    barrel = _find_barrel(lib_dir)
    if barrel is None:
        return []
    named, stars = _parse_barrel(barrel)
    out = set(named)
    for src_file, _ in stars:
        for n in _scan_module_exports(src_file):
            if n == "default":
                continue  # `default` can't be re-exposed via `export * from`
            out.add(n)
    return sorted(out)


def count_symbol_usage(
    lib_dir: Path | str,
    consumer_dirs: list[Path | str],
    config: LibraryUsageConfig | None = None,
    *,
    extra_lib_roots: list[Path | str] | None = None,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """JS/JSX usage scan — explicit-import resolution only.

    Returns ``(canonicals, per_consumer)`` where ``per_consumer`` is
    ``{consumer_file_path: {canonical: count}}``. ``count`` is the number
    of import statements that resolve to that canonical (deduped per
    statement — ``import { X, X } from`` would be 1 if the parser saw it,
    but real JS code never writes that).

    ``extra_lib_roots``: when set, every path under any of those roots is
    treated as equivalent to ``lib_dir`` for import-target resolution.
    Use this when each task carries its own identical mirror of the lib
    (webgen apply-phase layout: ``tasks/<id>/lib/``). The canonical
    ``lib_dir`` is still the single source of truth for exports.
    """
    lib_dir = Path(lib_dir).resolve()
    cfg = _js_config(lib_dir, config)
    barrel = _find_barrel(lib_dir)
    if barrel is None:
        return [], {}
    barrel = barrel.resolve()

    all_lib_roots = [lib_dir] + [Path(r).resolve() for r in (extra_lib_roots or [])]

    def _canonicalize(p: Path) -> Path | None:
        """Map any per-task lib path back to the canonical lib_dir path."""
        for root in all_lib_roots:
            try:
                rel = p.relative_to(root)
                return (lib_dir / rel).resolve()
            except ValueError:
                continue
        return None

    named_reexports, stars = _parse_barrel(barrel)

    # Canonical name set: barrel-exposed names + star-expanded module symbols
    canonical_set: set[str] = set(named_reexports.keys())
    # Reverse: source_file → {canonical_name: original_name_in_file}
    by_source: dict[Path, dict[str, str]] = {}
    for exposed, (src_file, orig) in named_reexports.items():
        by_source.setdefault(src_file.resolve(), {})[exposed] = orig
    for src_file, _ in stars:
        for n in _scan_module_exports(src_file):
            if n == "default":
                continue
            canonical_set.add(n)
            by_source.setdefault(src_file.resolve(), {})[n] = n

    canonicals = sorted(canonical_set)
    per_consumer: dict[str, dict[str, int]] = {}
    seen: set[Path] = set()

    def _bump(table: dict[str, int], name: str) -> None:
        if name in canonical_set:
            table[name] = table.get(name, 0) + 1

    for raw_root in consumer_dirs:
        root = Path(raw_root)
        if not root.is_dir():
            continue
        for fpath in _iter_js_files(root, cfg):
            resolved = fpath.resolve()
            if resolved in seen:
                continue
            if resolved == barrel:
                continue  # barrel itself doesn't consume anything
            seen.add(resolved)

            counts: dict[str, int] = {}
            for clause, src in _imports_in_file(fpath):
                target = _resolve_import_path(fpath, src)
                if target is None:
                    continue
                target = target.resolve()
                # Map per-task lib mirror → canonical lib path
                canon_target = _canonicalize(target)
                if canon_target is None:
                    continue
                target = canon_target
                if clause is None:
                    continue  # side-effect import: no symbol bumped

                if target == barrel:
                    # Bump per the barrel's exposed name
                    if clause["kind"] == "named":
                        for orig, _local in clause["specifiers"]:
                            _bump(counts, orig)
                    elif clause["kind"] == "default_named":
                        # 'default' from barrel is unusual; treat default
                        # name as a canonical only if barrel exposes it.
                        _bump(counts, clause["default"])
                        for orig, _local in clause["specifiers"]:
                            _bump(counts, orig)
                    elif clause["kind"] == "default":
                        _bump(counts, clause["name"])
                    elif clause["kind"] == "namespace":
                        for n in canonical_set:
                            _bump(counts, n)
                else:
                    # Deep import — reverse lookup by source file
                    by_local = by_source.get(target, {})
                    if not by_local:
                        continue
                    if clause["kind"] == "named":
                        for orig, _local in clause["specifiers"]:
                            for exp, original in by_local.items():
                                if original == orig:
                                    _bump(counts, exp)
                    elif clause["kind"] == "default":
                        for exp, original in by_local.items():
                            if original == "default":
                                _bump(counts, exp)
                    elif clause["kind"] == "default_named":
                        for exp, original in by_local.items():
                            if original == "default":
                                _bump(counts, exp)
                        for orig, _local in clause["specifiers"]:
                            for exp, original in by_local.items():
                                if original == orig:
                                    _bump(counts, exp)
                    elif clause["kind"] == "namespace":
                        for exp in by_local:
                            _bump(counts, exp)

            if counts:
                per_consumer[str(fpath)] = counts

    return canonicals, per_consumer


def aggregate(per_consumer: dict[str, dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for counts in per_consumer.values():
        for sym, n in counts.items():
            out[sym] = out.get(sym, 0) + n
    return out
