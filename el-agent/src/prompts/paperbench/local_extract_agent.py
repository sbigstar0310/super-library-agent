"""Prompt for the PaperbenchLocalExtractAgent (sla_ours, intra-paper).

Per-paper deduplication. Mirrors `prompts/webgen/local_extract_agent.py`'s
v3 design: no forced `local_lib/` directory, no mandatory barrel — the
agent reads the submission's existing layout and integrates each
extracted helper where it naturally fits (existing utility module,
package subdir, or a new module at the level of its callers).

Rationale (per webgen v3 study): forcing a catch-all dir for tiny
symbols inflates dep-aware NLL because every fragment inherits the same
~1.9 NLL/token penalty as its callers. Letting placement follow the
existing layout keeps the dependency graph natural.
"""

from __future__ import annotations
from typing import Any

from prompts.common import LOCAL_EXTRACT_SYSTEM_PROMPT
from prompts.paperbench.common import format_paper_body
from prompts.paperbench.paperbench import APP_RULES


__all__ = [
    "LOCAL_EXTRACT_SYSTEM_PROMPT",
    "build_local_extract_user_prompt",
]

# System prompt sourced from prompts.common.local_extract (byte-equal
# across webgen/paperbench post-v3).


_PAPERBENCH_LOCAL_EXTRACT_OUTPUT = """\
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
  - Rewrite call sites with absolute or package-relative Python imports
    against whatever paths you actually chose.
"""


def build_local_extract_user_prompt(
    paper: dict[str, Any],
    *,
    task_id: str,
    workspace_dir: str,
    local_extract_candidates: str,
    existing_global_lib_summary: str,
) -> str:
    """User prompt for one PaperbenchLocalExtractAgent invocation.

    Args:
        paper: task entry from `task_lookup(task_id)` — used by
            `format_paper_body` to inject the [Paper] context block.
        task_id: printed for context.
        workspace_dir: absolute path of the submission to refactor in place.
        local_extract_candidates: markdown from
            ``utils.candidates.get_extract_candidates(strategy="nl",
            app_dirs={task_id: workspace_dir}, mode="local",
            library_dir=...)``.
        existing_global_lib_summary: short listing of the carry-forward
            global ``lib/`` (so the agent avoids duplicating it locally).
    """
    return f"""\
You are deduplicating intra-submission code in paperbench task `{task_id}`.

{format_paper_body(paper)}

{APP_RULES}

[Workspace] (READ and EDIT in place)
{workspace_dir}

{_PAPERBENCH_LOCAL_EXTRACT_OUTPUT}

[Existing global library] (read-only — do not recreate these locally)
{existing_global_lib_summary}

[Local extract candidates]
{local_extract_candidates}
"""
