"""Librarian repair-turn prompt (byte-equal across webgen and paperbench).

Delivered as a single resume turn after the correctness gate flags apps that
stopped building/compiling once the sampled library was applied. The agent has
already produced a full library + refactor in the prior conversation; this turn
asks it to make the flagged apps pass again with **minimal, local** edits —
crucially NOT to redesign the library API (other apps already build against it,
and a redesign would regress them).
"""

from __future__ import annotations


__all__ = ["build_librarian_repair_prompt"]


_HEADER = """\
The correctness gate ({gate_name}) ran on your refactored corpus. The apps
below FAILED — they built/resolved before your refactor but not after it. Fix
them.

[Rules]
- Do NOT redesign or rename the library's public API. Other apps in this corpus
  already build against it; changing exported names/signatures would break them.
- Prefer fixing the FAILING APP: correct its imports, restore any symbol the
  refactor dropped, fix a bad relative path to the shared library, or re-add
  glue the app still needs. Edit the library implementation ONLY if the failure
  is a genuine bug inside a shared symbol (never its signature).
- Keep every currently-passing app working. Re-run the app's build/import smoke
  after each fix to confirm.
- Touch only what the errors below point at. Do not start a new refactor.

[Failing apps]
"""

_APP_BLOCK = """\
### task {tid}
```
{error_tail}
```
"""


def build_librarian_repair_prompt(
    failures: dict[str, str],
    *,
    gate_name: str = "npm install && npx vite build",
) -> str:
    """Build the repair feedback.

    Args:
        failures: ``{task_id: error_tail}`` — the gate's captured error output
            (build stderr tail for webgen; compile/import errors for paperbench).
        gate_name: human-readable gate description injected into the header.
    """
    parts = [_HEADER.format(gate_name=gate_name)]
    for tid in sorted(failures):
        tail = (failures[tid] or "(no error output captured)").rstrip()
        parts.append(_APP_BLOCK.format(tid=tid, error_tail=tail))
    return "\n".join(parts)
