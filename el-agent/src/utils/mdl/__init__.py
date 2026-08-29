"""Public API for ``utils.mdl``.

See ``README.md`` next to this file for the MDL formula and the
reproduction guide.

Stable production API (unchanged across the 2026-05 refactor)::

    from utils.mdl import get_mdl_score, get_maintainability_metrics, read_dir_to_text

Programmatic MDL (research / CLI)::

    from utils.mdl import MDLMetric, AppMDLResult, FileDetail
    from utils.mdl import load_task_config

Legacy concat-mode (cross-paper sanity check)::

    from utils.mdl.concat import score_concat
"""

from .configs import LibLayout, TaskConfig, load_task_config
from .file_io import (
    format_dep_context,
    get_code_files,
    get_maintainability_metrics,
    is_valid_codebase,
    read_app_code_files,
    read_dir_to_text,
)
from .llm import compute_prompt_logprobs
from .parsers._types import FileNode, ImportStmt
from .parsers.js import strip_comments as strip_js_comments
from .parsers.python import strip_comments as strip_py_comments
from .scoring import (
    AppMDLResult,
    FileDetail,
    MDLMetric,
    get_mdl_score,
)

__all__ = [
    # scoring (dep-aware MDL — the default)
    "MDLMetric",
    "AppMDLResult",
    "FileDetail",
    "get_mdl_score",
    # llm
    "compute_prompt_logprobs",
    # file_io
    "get_code_files",
    "read_dir_to_text",
    "get_maintainability_metrics",
    "is_valid_codebase",
    "read_app_code_files",
    "format_dep_context",
    # parsers
    "FileNode",
    "ImportStmt",
    "strip_js_comments",
    "strip_py_comments",
    # configs
    "TaskConfig",
    "LibLayout",
    "load_task_config",
]
