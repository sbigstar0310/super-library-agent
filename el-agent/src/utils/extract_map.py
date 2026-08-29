"""Helpers for parsing `lib/extract_map.md`, the per-round provenance doc
written by the global-extract agents at the end of an extract turn.

Format (one section per cluster-derived symbol):

    ## `lib.<module>.<name>`

    **Sources** (apps the pattern was generalized from)
    | App | File | Function/Symbol |
    |---|---|---|
    | <app_id> | <relative path> | <original name> |
    ...

    **Why generalized**
    ...

    **Potential uses**
    ...

    ---

`filter_for_app` pulls just the sections whose Sources block names a given
app_id, used to inject a per-app slice into the apply prompt. Autonomous
additions (no cross-app source) are excluded by construction since they
have no Sources rows mentioning any app.
"""

from __future__ import annotations
import re
from pathlib import Path

__all__ = ["read_extract_map", "filter_for_app"]


_SECTION_DELIM = re.compile(r"^##\s+", flags=re.MULTILINE)


def read_extract_map(lib_dir: str | Path) -> str | None:
    """Return contents of `<lib_dir>/extract_map.md` or None if missing/empty."""
    p = Path(lib_dir) / "extract_map.md"
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return text or None


def _split_sections(map_text: str) -> list[str]:
    """Split markdown into per-symbol sections by top-level `## ` headers.

    Anything before the first `## ` is dropped (e.g. an intro paragraph).
    """
    parts = _SECTION_DELIM.split(map_text)
    if not parts:
        return []
    # parts[0] is the preamble before the first `## ` header.
    return [f"## {p.rstrip()}" for p in parts[1:] if p.strip()]


def _section_mentions_app(section: str, app_id: str) -> bool:
    """Return True if the section's Sources block mentions `app_id`.

    Heuristic: scan only the lines after a `**Sources**` line (case-
    insensitive) until the next bold-header marker (`**...**`) or end of
    section, and look for `app_id` as a whole token. This avoids matching
    incidental occurrences in the prose / Potential uses paragraph (a
    weaker signal of "this app was a source").
    """
    if not app_id:
        return False
    lines = section.splitlines()
    in_sources = False
    pat = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(app_id)}(?![A-Za-z0-9_-])")
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\*\*\s*Sources\b", stripped, flags=re.IGNORECASE):
            in_sources = True
            continue
        if in_sources:
            # End of Sources block: another bold header (**Why...**, **Potential...**).
            if re.match(r"^\*\*[^*]", stripped):
                in_sources = False
                continue
            if pat.search(line):
                return True
    return False


def filter_for_app(map_text: str | None, app_id: str) -> str:
    """Return concatenated sections of `map_text` whose Sources mention `app_id`.

    When the input is empty/None or no sections match, returns a fallback
    sentinel string the prompt can render as-is.
    """
    if not map_text:
        return "(extract_map.md not produced this round)"
    sections = _split_sections(map_text)
    if not sections:
        return "(extract_map.md is empty or malformed)"
    matched = [s for s in sections if _section_mentions_app(s, app_id)]
    if not matched:
        return f"(no extract_map entries name `{app_id}` as a source)"
    return "\n\n".join(matched)
