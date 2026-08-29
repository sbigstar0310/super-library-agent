"""Strategy-aware code-index builder.

One entry point — ``index_app(code_dir, strategy=...)`` — that builds
the cocoindex sqlite once and optionally layers the NL summary index on
top:

  - ``strategy='embed'``: cocoindex sqlite only (vec0 BLOBs read directly
    by :mod:`utils.candidates.vec_index` and
    :mod:`utils.candidates.embed`).
  - ``strategy='nl'``: cocoindex sqlite + ``index.json`` with LLM
    summaries (:mod:`utils.candidates.nl_index`).

Idempotent — ``cocoindex_app`` skips work when the sqlite is up to date,
and the NL indexer reuses prior summaries by ``content_hash``.

The legacy ``utils.embedding`` pipeline (separate ``embeddings.npz`` per
app) is no longer reachable from this dispatcher. Callers that need it
(``main.py`` baseline path) import ``embed_app`` directly.
"""

from __future__ import annotations

from pathlib import Path

from utils.candidates.cocoindex_runner import cocoindex_app
from utils.candidates.dispatch import Strategy
from utils.candidates.nl_index import ensure_nl_index


def index_app(
    code_dir: str | Path,
    *,
    strategy: Strategy,
    app_id: str | None = None,  # back-compat no-op
    nl_model: str = "gpt-5.4-nano",
) -> None:
    """Build the cocoindex index for ``code_dir`` (and the NL layer on top
    when ``strategy='nl'``).

    Soft-fails per-stage (prints + continues) so downstream candidate
    retrieval surfaces "no candidates" rather than crashing on a missing
    artifact.

    Args:
        code_dir: Directory to index.
        strategy: ``'embed'`` (vec0 only) or ``'nl'`` (vec0 + summaries).
        app_id: Accepted for back-compat with the legacy signature; no
            effect (cocoindex scopes per-directory).
        nl_model: LLM used for chunk summarization (NL strategy only).
    """
    code_dir = Path(code_dir).resolve()
    try:
        cocoindex_app(code_dir)
    except Exception as e:
        print(f"[index_app:cocoindex] {code_dir}: {e}")
        return

    if strategy == "nl":
        try:
            ensure_nl_index(code_dir, model=nl_model, skip_cocoindex=True)
        except Exception as e:
            print(f"[index_app:nl] {code_dir}: {e}")
