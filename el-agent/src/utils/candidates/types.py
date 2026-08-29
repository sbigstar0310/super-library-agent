"""Shared types for the candidates package."""

from __future__ import annotations

from dataclasses import dataclass, field

# (qnames, hashes) — parallel lists, len(qnames) == len(hashes).
#   - apply  : len == 2, qnames[0]=lib_qname, qnames[1]=app_qname.
#   - extract: len >= 2, all members of the cluster/group.
PrepEntry = tuple[list[str], list[str]]


@dataclass
class CandidateResult:
    """Unified return type for apply/extract candidate retrieval.

    ``markdown`` is the prompt-ready block (always non-empty; soft-fail paths
    emit a "no candidates" sentinel). ``prep`` carries (qnames, hashes) for
    downstream judges — empty when no candidates survive or a strategy tracks
    no hash IDs.
    """

    markdown: str
    prep: list[PrepEntry] = field(default_factory=list)

    # Tuple-unpack compat: `md, prep = result`.
    def __iter__(self):
        return iter((self.markdown, self.prep))


NO_APPLY_CANDIDATES = "No migration candidates available."
NO_EXTRACT_CANDIDATES = "No cross-app primitives detected."
