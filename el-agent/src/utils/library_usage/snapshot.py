"""JSON snapshot I/O for ``LibraryUsageSnapshot``.

Snapshot path convention (RAL full-mode):

    backups/<bench>/<tag>/final/round_<N>/extract/library_usage.json

Snapshot for round N measures lib **v_{N-1}** — written at Extract pre-run
time, after round N-1's apply has settled the cumulative app set and
round N's coding has produced fresh consumers, but before round N's
extract main action mutates the library.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .counter import LibraryUsageSnapshot

_ROUND_RE = re.compile(r"round_(\d+)$")


def save_snapshot(path: Path | str, snapshot: LibraryUsageSnapshot) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(snapshot.to_json(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_snapshot(path: Path | str) -> LibraryUsageSnapshot | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return LibraryUsageSnapshot.from_json(json.loads(p.read_text("utf-8")))
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def load_past_snapshots(
    backup_root: Path | str,
    current_round: int,
    grace_rounds: int,
    *,
    filename: str = "library_usage.json",
) -> dict[int, LibraryUsageSnapshot]:
    """Walk ``<backup_root>/final/round_<r>/extract/<filename>`` and return
    the snapshots for ``r`` in ``[current_round - grace_rounds,
    current_round - 1]``.

    Missing files are silently skipped (cold start, pruned rounds).
    """
    out: dict[int, LibraryUsageSnapshot] = {}
    final_dir = Path(backup_root) / "final"
    if not final_dir.is_dir():
        return out
    lo = current_round - grace_rounds
    hi = current_round - 1
    for child in final_dir.iterdir():
        m = _ROUND_RE.match(child.name)
        if not m:
            continue
        r = int(m.group(1))
        if r < lo or r > hi:
            continue
        snap = load_snapshot(child / "extract" / filename)
        if snap is not None:
            out[r] = snap
    return out
