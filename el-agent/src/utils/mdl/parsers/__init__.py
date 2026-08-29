"""Language-specific parsers for MDL.

Each language module exports the same surface so `TaskConfig.parser_module`
can dispatch uniformly:
    - `strip_comments(code: str) -> str`
    - `build_dep_graph(app_dir, lib_dir, task) -> dict[abs_path, FileNode]`

Step A: ``js``. Step B: ``python``.
"""

from . import js, python
from ._types import FileNode, ImportStmt

__all__ = ["js", "python", "FileNode", "ImportStmt"]
