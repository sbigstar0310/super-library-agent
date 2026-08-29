"""Shared types for MDL parsers (language-independent)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FileNode:
    abs_path: str
    rel_path: str
    content: str
    deps: list[str] = field(default_factory=list)


@dataclass
class ImportStmt:
    """Generic import record. Languages may store extra fields in `extra`."""
    specifier: str          # raw module spec (`./x`, `ui-lib/foo`, `a.b.c`, ...)
    names: list[str]        # imported identifiers (empty for side-effect / star)
    kind: str               # 'local' | 'library' | 'external'
    level: int = 0          # Python relative-import depth (0=absolute, 1=`from .`, 2=`from ..`)
