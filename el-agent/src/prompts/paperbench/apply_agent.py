"""Prompt for the PaperbenchApplyAgent (sla_ours post-Extract).

One sub-agent per paper. Consumes the cumulative `lib/` + `extract_map.md`
+ per-paper apply candidates (NL default, embed optional) and rewrites
the submission to use the library. Mirrors
`prompts/webgen/apply_agent.py`.

System prompt + user-prompt template now live in
:mod:`prompts.common.apply` (byte-equal across webgen and paperbench).
"""

from __future__ import annotations
from typing import Any

from prompts.common import APPLY_SYSTEM_PROMPT, APPLY_USER_TEMPLATE, WORKSPACE_BLOCK
from prompts.paperbench.common import LIBRARY_BLOCK, format_paper_body
from prompts.paperbench.paperbench import APP_RULES


__all__ = [
    "APPLY_SYSTEM_PROMPT",
    "build_apply_user_prompt",
]


def build_apply_user_prompt(
    paper: dict[str, Any],
    *,
    task_id: str,
    workspace_dir: str,
    library_dir: str,
    extract_map_block: str,
    apply_candidates: str,
) -> str:
    """User prompt for one PaperbenchApplyAgent invocation.

    Args:
        paper: task entry from `task_lookup(task_id)`.
        task_id: paperbench paper id.
        workspace_dir: absolute path of this submission (READ+EDIT).
        library_dir: absolute path of the cumulative `lib/`.
        extract_map_block: full text of `<library_dir>/extract_map.md`
            (per-paper slicing is optional — the file is short).
        apply_candidates: markdown from
            ``utils.candidates.get_apply_candidates(strategy="nl"|"embed",
            library_dir=..., app_dir=workspace_dir)``.
    """
    return APPLY_USER_TEMPLATE.format(
        task_id=task_id,
        task_body=format_paper_body(paper),
        app_rules=APP_RULES,
        workspace_dir=workspace_dir,
        library_dir=library_dir,
        library_block=LIBRARY_BLOCK.format(library_dir=library_dir),
        extract_map_block=extract_map_block,
        apply_candidates=apply_candidates,
        workspace_block=WORKSPACE_BLOCK.format(workspace_dir=workspace_dir),
    )
