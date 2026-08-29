"""WebGen-Bench specific prompt fragments.

  - LIBRARY_BLOCK    — pointer to `ui-lib` for sla_naive / sla_ours
                       (React-specific: barrel import, no npm package)
  - format_task_body — [Task] block built from a WebGen-Bench JSONL row

`WORKSPACE_BLOCK` lives in :mod:`prompts.common.workspace` (byte-equal
across webgen and paperbench).

A WebGen-Bench task row schema (`data/WebGen-Bench/data/test.jsonl`):

  {
    "id": "000027",
    "instruction": "Please implement a website ...",
    "Category": {...},                # not exposed to coding agent
    "application_type": "...",        # not exposed to coding agent
    "ui_instruct": [...]              # the *test* checklist — agent must be blind
  }

`ui_instruct` is the evaluation rubric. It MUST be stripped before the prompt is
shown to the coding sub-agent (otherwise the agent overfits the checklist).
"""

from __future__ import annotations
from typing import Any


__all__ = [
    "LIBRARY_BLOCK",
    "format_task_body",
]


LIBRARY_BLOCK = """
[LIBRARY]
A useful library is available at `{library_dir}`.
This library consists of several useful modules that recur frequently when writing app code.
Use this library to keep your app code minimal.

**Library usage rules:**
  (1) Use library if possible: Do not re-implement code that already exists there.
  (2) Library Investigation: Before using a library symbol, read its
      source under `{library_dir}` to learn its parameters and behavior.
  (3) Use the library solely for code reuse: Do not imitate its
      internal patterns. Your code structure should follow the task's
      requirements, not the library.
  (4) If a building block is not in the library, implement it yourself.
  (5) Do not modify the library (read-only).
  (6) Import via relative path from the importing file to
      `{library_dir}/src/index.js` (barrel) or to a subpath file. The
      library is not an npm package — do not add it to `package.json`,
      and do not add a `resolve.alias` entry for it in `vite.config.js`.
"""


def format_task_body(task: dict[str, Any]) -> str:
    """Return the [Task] block from a WebGen-Bench JSONL row.

    Strips `ui_instruct` (the evaluation checklist) and the Category metadata
    so the coding sub-agent only sees the user-facing instruction.

    If the row carries an optional ``layout_spec`` key (string, injected by
    the agent after looking up ``data/augments/webgen/layout_specs/<id>.md``),
    that content is appended under a `[Visual & functional layout reference]`
    sub-block so the coder treats it as part of the task instead of as
    aesthetic guidance. Rows without ``layout_spec`` are unaffected
    (vanilla baseline behavior).
    """
    instruction = (task.get("instruction") or "").strip()
    layout_spec = (task.get("layout_spec") or "").strip()
    if layout_spec:
        instruction = (
            f"{instruction}\n\n"
            f"[Visual & functional layout reference]\n"
            f"The following describes the user-facing screens only. "
            f"Code organization (file structure, component boundaries, "
            f"panel groupings, tab-vs-route logic) is YOUR choice — match "
            f"the visible structure, not the section partitioning of this "
            f"document.\n\n"
            f"{layout_spec}"
        )
    return f"""[Task]
{instruction}"""
