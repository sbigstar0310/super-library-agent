"""Shared constants for code file discovery (MDL, embedding, etc.).

These mirror the ``webgen`` task config (``utils/mdl/configs/webgen.yaml``).
Loaded lazily so import never fails if the YAML is being edited.

Prefer ``utils.mdl.load_task_config(...)`` for new code — these globals exist
only for legacy callers that read them as plain lists.
"""

from __future__ import annotations

try:
    from .mdl.configs import load_task_config  # when imported as utils.constants
except ImportError:
    from utils.mdl.configs import load_task_config  # bare `import constants` fallback

_TASK = load_task_config("webgen")

# Glob patterns for source files to include
CODE_EXTENSIONS: list[str] = list(_TASK.code_extensions)

# Directories to skip when traversing
IGNORE_DIRS: list[str] = list(_TASK.ignore_dirs)

# Individual files to skip
IGNORE_FILES: list[str] = list(_TASK.ignore_files)
