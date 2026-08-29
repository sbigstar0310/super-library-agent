"""Paperbench-specific prompt fragments.

  - LIBRARY_BLOCK     — pointer to the shared Python `lib/` package
                        (Python-specific: PYTHONPATH-based import)
  - format_paper_body — [Paper] pointer block built from a paperbench
                        task entry

`WORKSPACE_BLOCK` lives in :mod:`prompts.common.workspace` (byte-equal
across webgen and paperbench).

A paperbench task entry (what the runner passes to `task_lookup(task_id)`):

  {
    "task_id": "fre",
    "paper_dir": "/abs/path/to/runs/<tag>/round_<N>/<phase>/tasks/fre/paper",
  }

The paper itself (paper.md / paper.pdf / addendum.md / blacklist.txt /
assets/) lives at `paper_dir` — agents read those via bash file tools
(paginated). There is no inline instruction string.
"""

from __future__ import annotations
from typing import Any


__all__ = [
    "LIBRARY_BLOCK",
    "format_paper_body",
]


LIBRARY_BLOCK = """
[LIBRARY]
A useful library is available at `{library_dir}`.
This library consists of several useful modules that recur frequently when writing submission code.
Use this library to keep your submission code minimal.

**Library usage rules:**
  (1) Use library if possible: Do not re-implement code that already exists there.
  (2) Library Investigation: Before using a library symbol, read its
      source under `{library_dir}` to learn its parameters and behavior.
  (3) Use the library solely for code reuse: Do not imitate its
      internal patterns. Your code structure should follow the task's
      requirements, not the library.
  (4) The submission must still implement ALL paper-specific algorithms, baselines, and experiments.
  (5) Do not modify the library (read-only).
  (6) Import as `from lib.<subpkg> import <Symbol>` (or
      `from lib.<subpkg>.<module> import <Symbol>` for less-common ones).
      Do NOT vendor or copy library code into the submission.
"""


def format_paper_body(paper: dict[str, Any]) -> str:
    """Return the [Paper] block for a paperbench task entry.

    Args:
        paper: dict with at least ``task_id`` and ``paper_dir`` keys.

    The body is a pointer block — actual paper text is read by the agent
    via the file tool (paginated). The reading discipline ("read paper.md
    and addendum.md FIRST") is enforced by the upstream coding-agent
    prompt and the apply/extract prompts that import this helper.
    """
    task_id = paper.get("task_id", "(unknown)")
    paper_dir = paper.get("paper_dir", "(unknown)")
    return f"""[Paper]
Reproducing paper `{task_id}`. Source materials (READ-ONLY):
- `{paper_dir}/paper.md`        full text in markdown
- `{paper_dir}/paper.pdf`       same paper as PDF (optional, prefer .md)
- `{paper_dir}/addendum.md`     clarifications + scope notes from the
                                paperbench authors — read this FIRST
                                alongside paper.md
- `{paper_dir}/blacklist.txt`   resources you MUST NOT consult
- `{paper_dir}/assets/`         figures and supplementary files

Read paper.md and addendum.md (paginated) before editing code."""
