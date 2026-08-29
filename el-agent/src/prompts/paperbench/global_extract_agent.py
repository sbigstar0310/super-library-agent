"""Prompt for the PaperbenchGlobalExtractAgent.

Cross-paper extraction (sla_ours). Writes new symbols into the shared
Python `lib/` and an explicit `extract_map.md` documenting which
baseline patterns each new symbol replaces. Mirrors
`prompts/webgen/global_extract_agent.py`.

System prompt + extract_map instruction now live in
:mod:`prompts.common.extract` (byte-equal across webgen and paperbench).
"""

from __future__ import annotations
from typing import Any

from prompts.common import EXTRACT_MAP_INSTRUCTION, EXTRACT_SYSTEM_PROMPT
from prompts.paperbench.common import format_paper_body
from prompts.paperbench.paperbench import LIB_RULES


__all__ = [
    "EXTRACT_SYSTEM_PROMPT",
    "EXTRACT_MAP_INSTRUCTION",
    "build_extract_user_prompt",
]


def build_extract_user_prompt(
    *,
    papers: dict[str, dict[str, Any]],
    apps: dict[str, str],
    library_dir: str,
    extract_candidates: str,
    existing_lib_summary: str,
) -> str:
    """User prompt for one PaperbenchGlobalExtractAgent invocation.

    The `extract_map.md` writing instruction is NOT included here. It is
    delivered in a follow-up resume turn (see
    ``PaperbenchFullRun._run_extract_map_turn``) so the 1st-turn prompt
    stays focused on extraction.

    Args:
        papers: {paper_id: task entry from `task_lookup`}.
        apps: {paper_id: absolute path to that submission}. READ-ONLY.
        library_dir: absolute path of `lib/`. Seeded with previous round's
            lib in `r ≥ 2`.
        extract_candidates: markdown from
            ``utils.candidates.get_extract_candidates(strategy="nl",
            app_dirs=apps, library_dir=library_dir, mode="global")``.
        existing_lib_summary: short directory listing of the seeded library.
    """
    apps_block = "\n".join(f"- `{pid}` → `{path}`" for pid, path in apps.items())

    paper_blocks: list[str] = []
    for pid, paper in papers.items():
        paper_blocks.append(f"### {pid}\n{format_paper_body(paper)}")
    papers_block = "\n\n".join(paper_blocks) if paper_blocks else "(none)"

    return f"""\
Extract a cross-paper shared Python library from {len(apps)} paper
submissions.

[Source submissions] (READ-ONLY)
{apps_block}

[Paper descriptions]
{papers_block}

[Library output]
Target directory: {library_dir}

[Existing library]
{existing_lib_summary}

{LIB_RULES}

[Extract candidates]
{extract_candidates}
"""
