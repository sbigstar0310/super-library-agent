"""Embedding-based candidate retrieval — cocoindex backend.

Apply and extract share one chunk universe with ``nl.py``: cocoindex's
tree-sitter chunker (``.cocoindex_code/target_sqlite.db``), with embedding
vectors read from the vec0 BLOBs via :mod:`utils.candidates.vec_index`.

Differences from the legacy ``utils.embedding`` / ``retrieve`` / ``clustering``
path (the pre-cocoindex retrieval, not used by webgen):

  - chunk granularity differs: tree-sitter token-windows, not AST
    function/class nodes — cosine top-K lists are not directly comparable,
    though the (qnames/hashes + markdown) contract is preserved.
  - ``hash_id`` is the full 64-char content SHA-256 (vs legacy 12-char over
    canonicalized AST), so legacy Refused-ledger entries look "fresh" one round.
  - ``_is_dunder`` filter dropped — cocoindex chunks carry no symbol name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .cocoindex_runner import cocoindex_app
from .types import (
    NO_APPLY_CANDIDATES,
    NO_EXTRACT_CANDIDATES,
    CandidateResult,
    PrepEntry,
)
from .vec_index import ChunkVec, load_vec_index, vec_dim


# ---- apply --------------------------------------------------------------


def get_apply_candidates_embed(
    *,
    library_dir: str,
    app_dir: str,
    top_k: int = 10,
    min_line: int = 5,
    min_similarity: float = 0.7,
) -> CandidateResult:
    """Library→app retrieval via cocoindex vec0 cosine similarity.

    Two-stage: per library chunk keep top-K app chunks above
    ``min_similarity``, then globally rank survivors and keep top-K.
    """
    try:
        return _apply_inner(
            library_dir=library_dir,
            app_dir=app_dir,
            top_k=top_k,
            min_line=min_line,
            min_similarity=min_similarity,
        )
    except Exception as e:
        print(f"[candidates.embed] apply retrieval failed: {e}")
        return CandidateResult(NO_APPLY_CANDIDATES, [])


def _apply_inner(
    *,
    library_dir: str,
    app_dir: str,
    top_k: int,
    min_line: int,
    min_similarity: float,
) -> CandidateResult:
    lib_path = Path(library_dir).resolve()
    app_path = Path(app_dir).resolve()
    cocoindex_app(lib_path)
    cocoindex_app(app_path)

    lib_chunks = load_vec_index(lib_path, min_line=min_line)
    app_chunks = load_vec_index(app_path, min_line=min_line)
    if not lib_chunks or not app_chunks:
        return CandidateResult(NO_APPLY_CANDIDATES, [])

    d_lib = vec_dim(lib_path)
    d_app = vec_dim(app_path)
    if d_lib != d_app:
        print(
            f"[candidates.embed] dim mismatch: lib D={d_lib} vs app D={d_app}; "
            f"skipping apply retrieval"
        )
        return CandidateResult(NO_APPLY_CANDIDATES, [])

    Q = np.vstack([c.vector for c in lib_chunks])
    T = np.vstack([c.vector for c in app_chunks])
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    sim = Q @ T.T  # (n_lib, n_app)

    # Stage 1: per-query top-K above threshold.
    pairs: list[tuple[int, int, float]] = []
    for q_idx in range(sim.shape[0]):
        for t_idx in np.argsort(sim[q_idx])[::-1][:top_k]:
            score = float(sim[q_idx, t_idx])
            if score < min_similarity:
                continue
            pairs.append((q_idx, int(t_idx), score))

    if not pairs:
        return CandidateResult(NO_APPLY_CANDIDATES, [])

    # Stage 2: global top-K.
    pairs.sort(key=lambda p: p[2], reverse=True)
    pairs = pairs[:top_k]

    # Re-group by library query for markdown.
    grouped: dict[int, list[tuple[int, float]]] = {}
    for q_idx, t_idx, score in pairs:
        grouped.setdefault(q_idx, []).append((t_idx, score))

    # Render markdown matching the NL strategy's apply shape, minus the
    # LLM-only fields (symbol name, summaries, reasons) which the embed path
    # has no LLM to produce. Sorted by best-per-query similarity desc.
    sorted_queries = sorted(
        grouped.items(),
        key=lambda kv: max(s for _, s in kv[1]),
        reverse=True,
    )

    lines: list[str] = []
    prep: list[PrepEntry] = []
    for idx, (q_idx, matches) in enumerate(sorted_queries, start=1):
        q = lib_chunks[q_idx]
        lines.append(f"### A{idx}.")
        lines.append(
            f"**Library**: {q.file_path}:{q.start_line}-{q.end_line}"
            f"::chunk_id={q.chunk_id}"
        )
        lines.append("**Replaces in this app**:")
        matches.sort(key=lambda p: p[1], reverse=True)
        for t_idx, _score in matches:
            r = app_chunks[t_idx]
            lines.append(
                f"  - {r.file_path}:{r.start_line}-{r.end_line}"
                f"::chunk_id={r.chunk_id}"
            )
            lib_qname = f"{q.file_path}:{q.start_line}-{q.end_line}"
            app_qname = f"{r.file_path}:{r.start_line}-{r.end_line}"
            prep.append(([lib_qname, app_qname], [q.content_hash, r.content_hash]))
        lines.append("")

    return CandidateResult("\n".join(lines).rstrip(), prep)


# ---- extract (cluster) --------------------------------------------------


ExtractMode = Literal["global", "local"]


def get_extract_candidates_embed(
    *,
    app_dirs: dict[str, str],
    mode: ExtractMode = "global",
    top_k: int = 10,
    min_line: int = 5,
    min_mean_sim: float = 0.55,
    distance_threshold: float = 1.0,
    snippet_lines: int = 15,
) -> CandidateResult:
    """Ward-clustering candidates across app chunks.

    ``mode='global'`` keeps only clusters spanning ≥2 apps (cross-app
    primitives); ``mode='local'`` also keeps single-app clusters (in-app
    duplicates → ``local_lib/``).
    """
    min_distinct_apps = 2 if mode == "global" else 1
    try:
        return _extract_inner(
            app_dirs=app_dirs,
            distance_threshold=distance_threshold,
            min_line=min_line,
            min_mean_sim=min_mean_sim,
            min_distinct_apps=min_distinct_apps,
            top_k=top_k,
            snippet_lines=snippet_lines,
        )
    except Exception as e:
        print(f"[candidates.embed] extract retrieval failed: {e}")
        return CandidateResult(NO_EXTRACT_CANDIDATES, [])


def _extract_inner(
    *,
    app_dirs: dict[str, str],
    distance_threshold: float,
    min_line: int,
    min_mean_sim: float,
    min_distinct_apps: int,
    top_k: int,
    snippet_lines: int,
) -> CandidateResult:
    # Lazy import — sklearn pulls scipy and adds ~0.5s to startup.
    from sklearn.cluster import AgglomerativeClustering

    pooled_vecs: list[np.ndarray] = []
    pooled_chunks: list[ChunkVec] = []
    pooled_app_ids: list[str] = []
    common_D: int | None = None

    for tid, app_dir in app_dirs.items():
        app_path = Path(app_dir).resolve()
        try:
            cocoindex_app(app_path)
        except Exception as e:
            print(f"[candidates.embed] skip {tid}: cocoindex_app failed: {e}")
            continue
        d_here = vec_dim(app_path)
        if d_here == 0:
            print(f"[candidates.embed] skip {tid}: no indexed chunks")
            continue
        if common_D is None:
            common_D = d_here
        elif d_here != common_D:
            print(
                f"[candidates.embed] skip {tid}: D={d_here} vs pool D={common_D}"
            )
            continue
        chunks = load_vec_index(app_path, min_line=min_line)
        if not chunks:
            continue
        pooled_vecs.append(np.vstack([c.vector for c in chunks]))
        pooled_chunks.extend(chunks)
        pooled_app_ids.extend([tid] * len(chunks))

    if not pooled_vecs or len(pooled_chunks) < 2:
        return CandidateResult(NO_EXTRACT_CANDIDATES, [])

    # L2-normalize: with unit vectors euclidean^2 = 2(1 - cos), so ward's
    # variance distance lands in a cosine-equivalent space.
    E = np.vstack(pooled_vecs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="ward",
        metric="euclidean",
    )
    labels = clusterer.fit_predict(E)

    by_label: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        by_label.setdefault(int(lab), []).append(idx)

    clusters: list[dict] = []
    for members in by_label.values():
        if len(members) < 2:
            continue
        apps = {pooled_app_ids[i] for i in members}
        if len(apps) < min_distinct_apps:
            continue
        sub = E[members]
        S = sub @ sub.T
        iu = np.triu_indices(len(members), k=1)
        mean_sim = float(S[iu].mean()) if iu[0].size else 1.0
        if mean_sim < min_mean_sim:
            continue
        clusters.append(
            {
                "members": sorted(members),
                "size": len(members),
                "distinct_apps": len(apps),
                "mean_sim": mean_sim,
            }
        )

    if not clusters:
        return CandidateResult(NO_EXTRACT_CANDIDATES, [])

    clusters.sort(
        key=lambda c: (c["mean_sim"], c["size"], c["distinct_apps"]),
        reverse=True,
    )
    clusters = clusters[:top_k]

    # Matches the NL strategy's extract shape, minus the LLM-only fields
    # (Pattern / Why extractable / per-member summary). ``snippet_lines`` is
    # now unused — kept in the signature for back-compat.
    lines: list[str] = []
    for k, c in enumerate(clusters, start=1):
        members = c["members"]
        lines.append(f"### C{k}.")
        lines.append("**Members**:")
        for i in members:
            m = pooled_chunks[i]
            lines.append(
                f"  - {pooled_app_ids[i]}::{m.file_path}:"
                f"{m.start_line}-{m.end_line}::chunk_id={m.chunk_id}"
            )
        lines.append("")

    prep: list[PrepEntry] = []
    for c in clusters:
        qnames = [
            f"{pooled_app_ids[i]}::{pooled_chunks[i].file_path}"
            f":{pooled_chunks[i].start_line}-{pooled_chunks[i].end_line}"
            for i in c["members"]
        ]
        hashes = [pooled_chunks[i].content_hash for i in c["members"]]
        prep.append((qnames, hashes))

    return CandidateResult("\n".join(lines).rstrip(), prep)
