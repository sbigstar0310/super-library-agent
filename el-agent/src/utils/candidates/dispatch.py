"""Strategy dispatch for apply/extract candidate retrieval.

``strategy`` selects the underlying retrieval method; apply and extract share
the strategy name so one env var / CLI flag controls the whole pipeline. A new
strategy = one branch here + a sibling module under ``utils.candidates``.
"""

from __future__ import annotations

from typing import Literal

from . import embed, nl
from .types import CandidateResult

Strategy = Literal["embed", "nl"]


def get_apply_candidates(
    strategy: Strategy,
    *,
    library_dir: str,
    app_dir: str,
    top_k: int = 10,
    min_line: int = 5,
    # embed-only knob (NL ignores)
    min_similarity: float = 0.7,
    # nl-only knobs (embed ignores)
    nl_model: str = "gpt-5.4-nano",
    nl_pick_model: str | None = None,
) -> CandidateResult:
    if strategy == "embed":
        return embed.get_apply_candidates_embed(
            library_dir=library_dir,
            app_dir=app_dir,
            top_k=top_k,
            min_line=min_line,
            min_similarity=min_similarity,
        )
    if strategy == "nl":
        return nl.get_apply_candidates_nl(
            library_dir=library_dir,
            app_dir=app_dir,
            top_k=top_k,
            min_line=min_line,
            model=nl_model,
            pick_model=nl_pick_model,
        )
    raise ValueError(f"Unknown candidate strategy: {strategy!r}")


def get_extract_candidates(
    strategy: Strategy,
    *,
    app_dirs: dict[str, str],
    mode: Literal["global", "local"] = "global",
    top_k: int = 10,
    min_line: int = 5,
    # embed-only knobs
    min_mean_sim: float = 0.55,
    distance_threshold: float = 1.0,
    snippet_lines: int = 15,
    # nl-only knobs
    library_dir: str | None = None,
    nl_model: str = "gpt-5.4-nano",
    nl_pick_model: str | None = None,
) -> CandidateResult:
    if strategy == "embed":
        return embed.get_extract_candidates_embed(
            app_dirs=app_dirs,
            mode=mode,
            top_k=top_k,
            min_line=min_line,
            min_mean_sim=min_mean_sim,
            distance_threshold=distance_threshold,
            snippet_lines=snippet_lines,
        )
    if strategy == "nl":
        return nl.get_extract_candidates_nl(
            app_dirs=app_dirs,
            library_dir=library_dir,
            mode=mode,
            top_k=top_k,
            min_line=min_line,
            model=nl_model,
            pick_model=nl_pick_model,
        )
    raise ValueError(f"Unknown candidate strategy: {strategy!r}")
