"""Prompt for the PaperbenchLibraryAgent (sla_naive — extract + apply in one).

A single sub-agent walks every cumulative paper submission, builds the
shared Python `lib/`, then rewrites the submissions in place so they
import from `lib`. No candidate list (optional), no `extract_map.md`.
Mirrors `prompts/webgen/library_agent.py`.
"""

from __future__ import annotations
from typing import Any

from prompts.paperbench.common import format_paper_body
from prompts.paperbench.paperbench import APP_RULES, LIB_RULES


__all__ = [
    "PAPERBENCH_LIBRARY_SYSTEM_PROMPT",
    "build_library_user_prompt",
]


PAPERBENCH_LIBRARY_SYSTEM_PROMPT = """
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
2. Briefly plan (5-8 lines): which primitives, from which papers, at
   what abstraction, which local files will be deleted.
3. Write the library at `<library_dir>` (see `[LIB_RULES]`).
4. For each submission: rewrite imports to `from lib.<subpkg> import
   <Symbol>` → delete obsolete files → `grep -rn` to confirm no stale
   local imports remain.
"""


def build_library_user_prompt(
    *,
    papers: dict[str, dict[str, Any]],
    apps: dict[str, str],
    library_dir: str,
    existing_lib_summary: str,
    library_candidates: str = "",
) -> str:
    """Assemble the PaperbenchLibraryAgent user prompt.

    Args:
        papers: {paper_id: task entry from `task_lookup`}.
        apps: {paper_id: absolute path to that submission} — the
            directories the agent reads AND edits in place.
        library_dir: absolute path of the `lib/` target. Empty for
            round 1, pre-seeded for upgrade rounds.
        existing_lib_summary: short summary of the library directory
            (`"(empty — building from scratch)"` when fresh).
        library_candidates: optional candidate markdown
            (empty string when no candidates are provided).
    """
    apps_block = "\n".join(f"- `{pid}` → `{path}`" for pid, path in apps.items())

    paper_blocks: list[str] = []
    for pid, paper in papers.items():
        paper_blocks.append(f"### {pid}\n{format_paper_body(paper)}")
    papers_block = "\n\n".join(paper_blocks) if paper_blocks else "(none)"

    candidates_section = ""
    if library_candidates.strip():
        candidates_section = f"\n[Library candidates]\n{library_candidates}\n"

    return f"""\
Build a shared Python library from {len(apps)} paperbench submissions and
refactor them to use it — in a single run.

[Library output]
Write at `{library_dir}`. Submissions import as
`from lib.<subpkg> import <Symbol>` — `PYTHONPATH` is configured by the
runner so `import lib` resolves. The library is NOT a pip-installable
package; do not declare it in any submission's requirements file.

[Source submissions] (READ and EDIT — in-place refactor)
{apps_block}

[App rules]
{APP_RULES}

[Existing library]
{existing_lib_summary}

{LIB_RULES}

[Paper descriptions]
{papers_block}
{candidates_section}"""
