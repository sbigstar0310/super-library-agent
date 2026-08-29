"""Apply / extract candidate retrieval — strategy-pluggable.

    result = get_apply_candidates("embed", library_dir=..., app_dir=...)
    md, prep = result          # tuple-unpack, or result.markdown / result.prep

Strategies live in sibling modules:
    embed.py — cosine retrieval over cocoindex vec0 embeddings
    nl.py    — cocoindex chunks + LLM NL summaries
"""

from .dispatch import Strategy, get_apply_candidates, get_extract_candidates
from .types import (
    NO_APPLY_CANDIDATES,
    NO_EXTRACT_CANDIDATES,
    CandidateResult,
    PrepEntry,
)

__all__ = [
    "CandidateResult",
    "NO_APPLY_CANDIDATES",
    "NO_EXTRACT_CANDIDATES",
    "PrepEntry",
    "Strategy",
    "get_apply_candidates",
    "get_extract_candidates",
]
