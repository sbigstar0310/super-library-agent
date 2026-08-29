"""Task configuration loader for MDL.

Each task (webgen, paperbench, ...) has a YAML config in this directory that
parameterizes file traversal (extensions, ignore patterns), the parser
language, and library-layout details.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).parent


@dataclass(frozen=True)
class LibLayout:
    src_subdir: str | None
    barrel_specifier: str | None


@dataclass(frozen=True)
class TaskConfig:
    name: str
    language: str
    code_extensions: list[str]
    ignore_dirs: list[str]
    ignore_files: list[str]
    marker_files: list[str]
    dep_strategy: str
    lib_layout: LibLayout
    joiner: str
    max_file_size_kb: int | None

    @property
    def parser_module(self):
        """Dispatch to parsers.<language>. Imported lazily to avoid cycles."""
        from .. import parsers
        mod = getattr(parsers, self.language, None)
        if mod is None:
            raise ValueError(
                f"No parser registered for language={self.language!r}. "
                f"Available: {[a for a in dir(parsers) if not a.startswith('_')]}"
            )
        return mod


@lru_cache(maxsize=None)
def load_task_config(name: str) -> TaskConfig:
    """Load a task config by name (filename without `.yaml`)."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"Task config {name!r} not found at {path}. Available: {available}"
        )
    data = yaml.safe_load(path.read_text())
    lib = data.get("lib_layout") or {}
    return TaskConfig(
        name=data["name"],
        language=data["language"],
        code_extensions=list(data["code_extensions"]),
        ignore_dirs=list(data.get("ignore_dirs") or []),
        ignore_files=list(data.get("ignore_files") or []),
        marker_files=list(data.get("marker_files") or []),
        dep_strategy=data.get("dep_strategy", ""),
        lib_layout=LibLayout(
            src_subdir=lib.get("src_subdir"),
            barrel_specifier=lib.get("barrel_specifier"),
        ),
        joiner=data.get("joiner", "\n"),
        max_file_size_kb=data.get("max_file_size_kb"),
    )


__all__ = ["TaskConfig", "LibLayout", "load_task_config", "CONFIG_DIR"]
