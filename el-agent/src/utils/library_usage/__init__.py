"""Library-symbol usage counting and dead-code tracking.

Counts per-symbol imports of an extracted library across all consuming
apps (and the library's own non-barrel modules), persists the result as a
JSON snapshot per round, and infers symbols that have stayed at zero
imports for a configurable number of consecutive rounds.

Used by the library-usage metric (``scripts/metrics/get_lib_usage.py``).
"""

from .counter import (
    LibraryUsageConfig,
    LibraryUsageSnapshot,
    aggregate,
    collect_lib_definitions,
    compute_dead_symbols,
    count_symbol_usage,
    is_dead,
    read_lib_exports,
)
from .snapshot import load_past_snapshots, load_snapshot, save_snapshot
from .configs import load_usage_config

__all__ = [
    "LibraryUsageConfig",
    "LibraryUsageSnapshot",
    "read_lib_exports",
    "count_symbol_usage",
    "aggregate",
    "is_dead",
    "compute_dead_symbols",
    "save_snapshot",
    "load_snapshot",
    "load_past_snapshots",
    "load_usage_config",
]
