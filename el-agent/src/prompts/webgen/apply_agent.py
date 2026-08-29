"""Prompt for the WebgenApplyAgent (sla_ours post-Extract).

One sub-agent per app. Consumes the cumulative `ui-lib` + `extract_map.md`
+ per-app NL apply candidates and rewrites the app to use the library.

System prompt + user-prompt template now live in
:mod:`prompts.common.apply` (byte-equal across webgen and paperbench).
"""

from __future__ import annotations
from typing import Any

from prompts.common import APPLY_SYSTEM_PROMPT, APPLY_USER_TEMPLATE, WORKSPACE_BLOCK
from prompts.webgen.common import LIBRARY_BLOCK, format_task_body
from prompts.webgen.webgen import APP_RULES


__all__ = [
    "APPLY_SYSTEM_PROMPT",
    "build_apply_user_prompt",
]


def build_apply_user_prompt(
    task: dict[str, Any],
    *,
    task_id: str,
    workspace_dir: str,
    library_dir: str,
    extract_map_block: str,
    apply_candidates: str,
) -> str:
    """User prompt for one WebgenApplyAgent invocation.

    Args:
        task: WebGen JSONL row (`ui_instruct` stripped by `format_task_body`).
        task_id: WebGen id.
        workspace_dir: absolute path of this app's submission (READ+EDIT).
        library_dir: absolute path of the cumulative `ui-lib`.
        extract_map_block: full text of `<library_dir>/extract_map.md`
            (per-app slicing is optional — the file is short).
        apply_candidates: markdown from
            `utils.candidates.get_apply_candidates(strategy="nl",
            library_dir=..., app_dir=workspace_dir)`.
    """
    return APPLY_USER_TEMPLATE.format(
        task_id=task_id,
        task_body=format_task_body(task),
        app_rules=APP_RULES,
        workspace_dir=workspace_dir,
        library_dir=library_dir,
        library_block=LIBRARY_BLOCK.format(library_dir=library_dir),
        extract_map_block=extract_map_block,
        apply_candidates=apply_candidates,
        workspace_block=WORKSPACE_BLOCK.format(workspace_dir=workspace_dir),
    )
