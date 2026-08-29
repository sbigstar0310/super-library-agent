"""Prompts for WebgenCodingAgent (baseline + sla_naive + sla_ours).

System prompt is the generic cc-exp coding base (no cheating, no web
browsing). User prompt overlays bench-specific APP_RULES via
`build_prompt_from_task`.
"""

from __future__ import annotations
from typing import Any

from prompts.common import WORKSPACE_BLOCK
from prompts.webgen.common import format_task_body
from prompts.webgen.webgen import APP_RULES


__all__ = [
    "WEBGEN_SYSTEM_PROMPT",
    "build_prompt_from_task",
]


WEBGEN_SYSTEM_PROMPT = """You are a helpful coding agent generating a
code application from a user instruction.

[Scope discipline]
Implement exactly what the [Task] instruction asks for — nothing more.
Do not add more complex codes (pages, routes, components, or features) the user did not
request, even when they would be "nice to have" or would let you
showcase available library symbols.

[Cheating prohibition]
- Do NOT read any other task's submission directory (sibling tasks in the
  same `tasks/` parent). You may only read inside your own workspace.
- Do NOT browse the web (`WebFetch`, `WebSearch`) for the task. You may rely
  on prior knowledge.
"""


def build_prompt_from_task(
    task: dict[str, Any],
    workspace_dir: str,
    library_block: str = "",
) -> str:
    """Build the user prompt the WebgenCodingAgent receives.

    Args:
        task: WebGen-Bench JSONL row (`id`, `instruction`, ...). `ui_instruct`
              and `Category` are intentionally NOT consumed — `format_task_body`
              strips them.
        workspace_dir: absolute path of the submission target. Filled into the
              `[Workspace]` block so the sub-agent knows where to write.
        library_block: optional `[LIBRARY]` block for sla_naive / sla_ours
              modes. Empty string for pure baseline.
    """
    workspace = WORKSPACE_BLOCK.format(workspace_dir=workspace_dir)
    return f"""You are generating a React + Vite web application.

{format_task_body(task)}

{APP_RULES}

{workspace}

{library_block}"""
