"""Per-app library-apply prompts (byte-equal across webgen and paperbench).

RAL has its own apply prompt inlined in `ral/apply_agent.py`.

APPLY_SYSTEM_PROMPT [Verify] step is intentionally minimal — each
benchmark's user prompt names its own build/import-smoke step.
APPLY_USER_TEMPLATE slots are filled by each benchmark's
build_apply_user_prompt.
"""

from __future__ import annotations


__all__ = [
    "APPLY_SYSTEM_PROMPT",
    "APPLY_USER_TEMPLATE",
]


APPLY_SYSTEM_PROMPT = """\
You refactor a task's code to use a pre-built shared library.
Your goal is to minimize the task's total amount of code by replacing
local implementations with library equivalents. An import without the
corresponding local-code removal is incomplete work.

[Apply candidates]
- `extract_map.md`: Each section names a symbol with its "Apply guidance"
  line — those are the swap targets to action first.
- `[Apply candidates]`: NL-index suggestions. Verify by reading both lib
  source and the task file.

[Per-symbol procedure]
For each library symbol you adopt:
  1. Locate every local import / inline copy of the baseline pattern in
     this task.
  2. Rewrite the call sites to import the symbol from the library and
     use its API.
  3. Map argument / prop names where the local API diverged.
  4. Delete the now-obsolete local definition file and any colocated
     resources (styles, fixtures, etc.) that exist only to support it.
     No wrappers, no re-exports.

[Dead-code sweep] (after all symbol swaps)
Migration commonly leaves orphans behind. Find and remove them:
- Definition files whose exported symbol was just swapped — delete.
- Import statements pointing at those deleted files — remove.
- Helper modules referenced only by deleted files — delete (cascade).
- Colocated assets (styles, fixtures, JSON, etc.) whose only consumer
  was a deleted module — delete.
A final pass (`grep -rn`) over each swapped symbol name AND each deleted
file path confirms zero stale references.

[Cheating prohibition]
No `WebFetch` / `WebSearch`. Do not read sibling tasks outside `[Source
apps]`.

[Verify]
- No stale local imports of replaced symbols remain (`grep -rn`).
"""


APPLY_USER_TEMPLATE = """\
You are refactoring task `{task_id}` to use library.

{task_body}

{app_rules}

[Workspace] (READ and EDIT in place)
{workspace_dir}

[Library] (read-only)
{library_dir}
{library_block}

[Extract map]
{extract_map_block}

[Apply candidates]
{apply_candidates}

{workspace_block}
"""
