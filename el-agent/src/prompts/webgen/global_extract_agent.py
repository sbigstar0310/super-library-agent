"""Prompt for the WebgenGlobalExtractAgent.

Cross-app extraction (sla_ours). Writes new symbols into `ui-lib/` and an
explicit `extract_map.md` documenting which baseline patterns each new
symbol replaces.

System prompt + extract_map instruction now live in
:mod:`prompts.common.extract` (byte-equal across webgen and paperbench).
"""

from __future__ import annotations
from typing import Any

from prompts.common import EXTRACT_MAP_INSTRUCTION, EXTRACT_SYSTEM_PROMPT
from prompts.webgen.common import format_task_body
from prompts.webgen.webgen import LIB_RULES


__all__ = [
    "EXTRACT_SYSTEM_PROMPT",
    "EXTRACT_MAP_INSTRUCTION",
    "build_extract_user_prompt",
]


def build_extract_user_prompt(
    *,
    tasks: dict[str, dict[str, Any]],
    apps: dict[str, str],
    library_dir: str,
    extract_candidates: str,
    existing_lib_summary: str,
) -> str:
    """User prompt for one WebgenGlobalExtractAgent invocation.

    The `extract_map.md` writing instruction is NOT included here. It is
    delivered exclusively in the 2nd resume turn (see
    ``WebgenFullRun._run_extract_map_turn``) so the 1st-turn prompt stays
    focused on extraction.

    Args:
        tasks: {task_id: webgen JSONL row}.
        apps: {task_id: absolute path to that app's submission}. READ-ONLY.
        library_dir: absolute path of `ui-lib`. Seeded with previous round's
            lib in `r ≥ 2`.
        extract_candidates: markdown from
            `utils.candidates.get_extract_candidates(strategy="nl",
            app_dirs=apps, library_dir=library_dir, mode="global")`.
        existing_lib_summary: short directory listing of the seeded library.
    """
    apps_block = "\n".join(f"- `{tid}` → `{path}`" for tid, path in apps.items())

    task_blocks: list[str] = []
    for tid, task in tasks.items():
        task_blocks.append(f"### {tid}\n{format_task_body(task)}")
    tasks_block = "\n\n".join(task_blocks) if task_blocks else "(none)"

    return f"""\
Extract a cross-task shared library from {len(apps)} tasks.

[Source tasks] (READ-ONLY)
{apps_block}

[Task descriptions]
{tasks_block}

[Library output]
Target directory: {library_dir}

[Existing library]
{existing_lib_summary}

{LIB_RULES}

[Extract candidates]
{extract_candidates}
"""
