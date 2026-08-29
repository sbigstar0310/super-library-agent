"""LOCAL_EXTRACT_SYSTEM_PROMPT — intra-codebase deduplication system msg.

webgen v3 design (byte-equal across webgen and paperbench; RAL has its
own dual-library variant): no forced catch-all `local_lib/` directory,
no mandatory barrel — the agent integrates each helper into the
codebase's existing layout.

"build command" in [Verify] is bench-neutral: each benchmark's user
prompt names its own build/import-smoke step.
"""

from __future__ import annotations


__all__ = ["LOCAL_EXTRACT_SYSTEM_PROMPT"]


LOCAL_EXTRACT_SYSTEM_PROMPT = """You factor out intra-codebase \
duplication inside one task to make the code minimal. Goal: move helpers \
/ utilities / reusable units that the app uses in 2+ places into the \
app's existing source layout, and rewrite call sites to import them.

[Discovery — start from candidates, verify with bash]
- `[Local extract candidates]` in the user prompt names patterns that recur
  inside this single app. They are starting points, not authority.
- Verify each one with `grep -rn` / `cat`. Keep only patterns with 2+ real
  call sites inside this app.
- A pre-existing global library may be listed in `[Existing global library]`.
  If a candidate is already covered there, do NOT recreate it locally —
  rewrite the call sites to import from the global library.

[Skip rules]
- One-off helpers (single call site) stay where they are.
- Trivial <10 LOC patterns stay inline.
- Apparent duplicates whose behavior actually diverges — keep separate.

[Placement — let the existing layout decide]
- Read the current codebase tree first. Notice the conventions the app
  already uses (where components live, whether there's a hooks file,
  how utilities are organized, etc.).
- Add each extracted helper where it most naturally fits given that
  layout. If a thematically related file already exists, append to it
  rather than create a new one.
- Trust the existing structure first. If a helper truly has no
  natural home, create whatever fits at the level of its callers.
- The name `lib/` is reserved for the cross-task shared library
  (mounted separately and referenced by global imports). Do NOT
  create a `src/lib/` (or `src/local_lib/`) for app-local helpers.

[Verify]
After editing, `grep -rn` to confirm no stale local definitions of moved
symbols remain. The task's build command (see [Task] / [Implementation
rules]) should still succeed.
"""
