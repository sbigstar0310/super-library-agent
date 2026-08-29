"""Shared lib/ directory summarizer for agent prompts.

Used by extract / library / local_extract agents (webgen, paperbench) to
surface a library directory's current state to the LLM, which decides which
symbols to add vs reuse. Parameterized over `globs` (lib file extensions),
`label` (header prefix), and `missing_msg` / `empty_msg`.
"""

from __future__ import annotations
from pathlib import Path


__all__ = ["summarize_lib_dir"]


def summarize_lib_dir(
    lib_dir: Path | None,
    *,
    globs: tuple[str, ...],
    label: str = "Existing library",
    max_items: int = 25,
    show_line_counts: bool = True,
    missing_msg: str = "(directory missing — will be created on first write)",
    empty_msg: str = "(empty — building from scratch)",
) -> str:
    """Return a short listing of files matching `globs` under `lib_dir`."""
    if lib_dir is None or not lib_dir.is_dir():
        return missing_msg
    files: list[Path] = []
    for pat in globs:
        files.extend(p for p in lib_dir.rglob(pat) if p.is_file())
    files.sort()
    if not files:
        return empty_msg
    lines = [f"{label} has {len(files)} file(s):"]
    for p in files[:max_items]:
        rel = p.relative_to(lib_dir)
        if show_line_counts:
            try:
                n_lines = sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
            except OSError:
                n_lines = 0
            lines.append(f"- {rel}  ({n_lines} lines)")
        else:
            lines.append(f"- {rel}")
    if len(files) > max_items:
        lines.append(f"- ... and {len(files) - max_items} more")
    return "\n".join(lines)
