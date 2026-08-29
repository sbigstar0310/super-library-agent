"""Prompt for the WebgenLibraryAgent (sla_naive — extract + apply in one).

A single sub-agent walks every cumulative app, builds `ui-lib`, then
rewrites the apps in place so they import from `ui-lib`. No candidate
list; the agent explores via bash. No `extract_map.md` output.
"""

from __future__ import annotations
from typing import Any

from prompts.webgen.common import format_task_body
from prompts.webgen.webgen import APP_RULES, LIB_RULES


__all__ = [
    "WEBGEN_LIBRARY_SYSTEM_PROMPT",
    "build_library_user_prompt",
]


WEBGEN_LIBRARY_SYSTEM_PROMPT = """
You are a library manager. In one run you build a shared library from N codebases
AND refactor those codebases in place to import from it.

[Dual goals]
1. EXTRACT: identify cross-app primitives (≥2 apps, behavior comparable)
   and write them into library.
2. APPLY: rewrite or refactor each codebase to reduce its code size by utilizing library,
   delete the obsolete codes and local files.

Extraction without application is incomplete. Both are mandatory.

[Discovery]
Promote only patterns shared by **2+ tasks** with comparable behavior
(API shape, side effects, observable outputs).

[Reject]
- Single-task helpers (used in only one task).
- Trivial <10 LOC patterns.
- Task-specific domain logic.
- Surface-similar code with diverging behavior.

[Per-symbol procedure]
For each library symbol you adopt:
  1. Locate every local import / inline copy of the baseline pattern in
     this task (e.g., `grep -rn`).
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

[Workflow]
1. Investigate every app under `[Source apps]` — if `[Library candidates]` is
   provided, use those clusters as the starting point; otherwise explore freely.
2. Briefly plan (5-8 lines): which primitives, from which apps, at what
   abstraction, which local files will be deleted.
3. Write the library at `<library_dir>`.
4. For each app: rewrite imports to relative paths into
   `<library_dir>/src/` → delete obsolete files → `grep -rn` to confirm
   no stale local imports remain.
"""


def build_library_user_prompt(
    *,
    tasks: dict[str, dict[str, Any]],
    apps: dict[str, str],
    library_dir: str,
    existing_lib_summary: str,
    library_candidates: str = "",
) -> str:
    """Assemble the WebgenLibraryAgent user prompt.

    Args:
        tasks: {task_id: webgen JSONL row}. `format_task_body` strips
            `ui_instruct` to keep the eval rubric out of the prompt.
        apps: {task_id: absolute path to that app's working submission} —
            the directories the agent reads AND edits in place.
        library_dir: absolute path of the `ui-lib` target. Empty for round 1,
            pre-seeded for upgrade rounds.
        existing_lib_summary: result of summarizing the library directory
            (`"(empty — building from scratch)"` when fresh).
        library_candidates: optional ward-clustering candidate markdown
            (empty string when no candidates are provided).
    """
    apps_block = "\n".join(f"- `{tid}` → `{path}`" for tid, path in apps.items())

    task_blocks: list[str] = []
    for tid, task in tasks.items():
        task_blocks.append(f"### {tid}\n{format_task_body(task)}")
    tasks_block = "\n\n".join(task_blocks) if task_blocks else "(none)"

    candidates_section = ""
    if library_candidates.strip():
        candidates_section = f"\n[Library candidates]\n{library_candidates}\n"

    return f"""\
Build a shared library from {len(apps)} WebGen-Bench apps and refactor them
to use it — in a single run.

[Library output]
Write at `{library_dir}`. Apps import via a relative path from each
source file to `{library_dir}/src/index.js` (barrel) or to a subpath
file under `{library_dir}/src/`. The library is not an npm package —
do not declare it in `package.json` and do not add a `resolve.alias`
entry for it in `vite.config.js`.

[Source apps] (READ and EDIT — in-place refactor)
{apps_block}

[App rules]
{APP_RULES}

[Existing library]
{existing_lib_summary}

{LIB_RULES}

[Task descriptions]
{tasks_block}
{candidates_section}"""
