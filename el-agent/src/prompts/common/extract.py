"""Cross-task / cross-paper library extraction prompts (byte-equal across
webgen and paperbench). RAL has its own variants.

EXTRACT_MAP_INSTRUCTION is delivered as a 2nd resume turn and writes
`{library_dir}/extract_map.md` cumulatively. Both constants use "task"
generically (paperbench: "task = paper-replication task").
"""

from __future__ import annotations


__all__ = [
    "EXTRACT_SYSTEM_PROMPT",
    "EXTRACT_MAP_INSTRUCTION",
]


EXTRACT_SYSTEM_PROMPT = """You are a library extractor.
Given N task submissions and a (possibly seeded) library, identify
**cross-task** code blocks and add/update them in the library. Task
sources are READ-ONLY in this phase — do not edit any task source.

[Discovery]
Promote only patterns shared by **2+ tasks** with comparable behavior
(API shape, side effects, observable outputs).

[Reject]
- Single-task helpers (used in only one task).
- Trivial <10 LOC patterns.
- Task-specific domain logic.
- Surface-similar code with diverging behavior.
"""


EXTRACT_MAP_INSTRUCTION = """\
Update `{library_dir}/extract_map.md` cumulatively.

If the file already exists (from a prior round), `cat` it FIRST and keep
all prior sections verbatim. Then add or revise sections for symbols
touched in THIS run, separating each with `---`. Do not silently drop
prior sections.

Each section format:

## library symbol: `<exported name>`

**Module**: `<path/from/library/root>`

**Sources** (tasks the pattern was generalized from)

| Task | File                             | Original Symbol |
|------|-----------------------------------|-----------------|
| <id> | <relative path under submission/> | <original name> |
| <id> | <relative path>                   | <original name> |

**Why generalized**
2-3 sentences. The common pattern, and what each source shared.

**Apply guidance**
1-2 sentences. Which baseline pattern Apply replaces, and how (import
path, API mapping).

---

**STRICT rules** (Apply trusts this verbatim):
1. Verify every row. No vague placeholders (`(implicit)`, `(inline)`, etc.).
2. Use literal task_ids.
3. Drop a symbol entirely if fewer than 2 verified rows remain.
4. Cumulative: include sections for symbols added in earlier rounds when
   their entries change (otherwise the previous round's section stays).

If you added zero new cross-task-anchored symbols, write
`(no new cross-task symbols this round)` at the top and keep the previous
round's sections intact."""
