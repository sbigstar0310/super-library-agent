"""Prompt for the WebgenLocalExtractAgent (sla_ours) — v3.

Per-task intra-app deduplication. v3 drops v1's catch-all
``src/local_lib/`` directory and mandatory barrel; the agent integrates
each helper into the app's existing layout.

Why: v1's catch-all dir cost +8.3% NLL under dep-aware MDL (each tiny
file inherits the same ~1.9 NLL/token penalty as its dependents). v3 on
3-trial smoke vs v1: NLL +3.8%, erosion −55%, zero ``src/local_lib/``.
"""

from __future__ import annotations
from typing import Any

from prompts.common import LOCAL_EXTRACT_SYSTEM_PROMPT
from prompts.webgen.common import format_task_body
from prompts.webgen.webgen import APP_RULES


__all__ = [
    "LOCAL_EXTRACT_SYSTEM_PROMPT",
    "build_local_extract_user_prompt",
]

# System prompt sourced from prompts.common.local_extract.


_WEBGEN_LOCAL_EXTRACT_OUTPUT = """\
[Output — adapt to the existing source layout]
Inspect codebase first and integrate each extracted helper into the
app's existing structure. Guidance:

  - Reuse the layout the app already has where it makes sense.
    If a thematically related file already exists, APPEND to it
    rather than create a new file.
  - If the existing layout doesn't have a natural home for a given
    helper, create whatever directory or file feels appropriate at
    the level of its callers.
  - `lib/` is reserved for the cross-task shared library. Do NOT
    create `src/lib/` (or `src/local_lib/`) for app-local helpers.
  - Rewrite call sites with relative imports against whatever
    paths you actually chose. No barrel files unless the app
    already uses one.
"""


def build_local_extract_user_prompt(
    task: dict[str, Any],
    *,
    task_id: str,
    workspace_dir: str,
    local_extract_candidates: str,
    existing_global_lib_summary: str,
) -> str:
    """User prompt for one WebgenLocalExtractAgent invocation (v3).

    Args:
        task: WebGen JSONL row (`ui_instruct` stripped by `format_task_body`).
        task_id: the WebGen id (printed for context only).
        workspace_dir: absolute path of the app to refactor in place.
        local_extract_candidates: markdown from
            ``utils.candidates.get_extract_candidates(strategy="nl",
            app_dirs={task_id: workspace_dir}, mode="local", library_dir=...)``.
        existing_global_lib_summary: short listing of the carry-forward
            global ui-lib (so the agent avoids duplicating it locally).
    """
    return f"""\
You are deduplicating intra-app code in WebGen-Bench task `{task_id}`.

{format_task_body(task)}

{APP_RULES}

[Workspace] (READ and EDIT in place)
{workspace_dir}

{_WEBGEN_LOCAL_EXTRACT_OUTPUT}

[Existing global library] (read-only — do not recreate these locally)
{existing_global_lib_summary}

[Local extract candidates]
{local_extract_candidates}
"""
