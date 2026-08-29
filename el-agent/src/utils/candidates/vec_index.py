"""Read cocoindex vec0 embeddings + chunk metadata from target_sqlite.db.

Companion to ``nl_index.py`` over the same chunk universe (same ``chunk_id``
keys, same ``content_hash``), returning raw embedding vectors instead of LLM
summaries.

sqlite-vec vec0 schema:

  - ``code_chunks_vec_auxiliary(rowid, value00=file_path, value01=content,
    value02=start_line, value03=end_line)`` — one row/chunk; rowid = global
    chunk id (matches ``nl_index`` keys).
  - ``code_chunks_vec_rowids(rowid, id, chunk_id, chunk_offset)`` — maps
    aux.rowid → partition ``chunk_id`` + slot index.
  - ``code_chunks_vec_chunks(chunk_id, …, partition00, …)`` — one row/partition;
    ``partition00`` is the language label.
  - ``code_chunks_vec_vector_chunks00(rowid, vectors BLOB)`` — rowid matches
    partition chunk_id; ``vectors`` is a packed (1024, D) float32 matrix, slot
    index = ``chunk_offset``.

Vectors are model-side L2-normalized (text-embedding-3-small), so cosine ≡ dot
product; we do not re-normalize.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cocoindex_runner import cocoindex_app
from .nl_index import _content_hash, index_paths

# vec0 packs 1024 slots per partition matrix row.
_SLOTS_PER_PARTITION_CHUNK = 1024


@dataclass(frozen=True)
class ChunkVec:
    """One chunk's metadata + embedding vector.

    ``chunk_id`` and ``content_hash`` match ``nl_index``'s keys, so the two
    indices join on either field.
    """

    chunk_id: int
    language: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    content_hash: str
    vector: np.ndarray  # shape=(D,), float32, L2-normalized


def ensure_vec_index(
    code_dir: str | Path,
    *,
    skip_cocoindex: bool = False,
) -> Path:
    """Build the cocoindex sqlite if missing. Returns the db path.

    Mirrors ``ensure_nl_index`` for API symmetry.
    """
    code_dir = Path(code_dir).resolve()
    if not skip_cocoindex:
        cocoindex_app(code_dir)
    db_path, _ = index_paths(code_dir)
    return db_path


def vec_dim(code_dir: str | Path) -> int:
    """Embedding dimensionality D, inferred from the partition blob size.

    Raises ``FileNotFoundError`` if no sqlite exists. Returns 0 if the db
    has no partitions yet (empty code dir).
    """
    db_path, _ = index_paths(Path(code_dir).resolve())
    if not db_path.exists():
        raise FileNotFoundError(f"cocoindex sqlite missing: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT length(vectors) FROM code_chunks_vec_vector_chunks00 LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return 0
    return int(row[0]) // 4 // _SLOTS_PER_PARTITION_CHUNK


def load_vec_index(
    code_dir: str | Path,
    *,
    min_line: int = 0,
    languages: set[str] | None = None,
) -> list[ChunkVec]:
    """Read all chunks + vectors from ``code_dir``'s cocoindex sqlite.

    Args:
        code_dir: Directory whose ``.cocoindex_code/target_sqlite.db`` is
            read.
        min_line: Drop chunks shorter than this many lines.
        languages: Optional whitelist of partition labels (e.g.
            ``{"python", "javascript"}``). ``None`` keeps all.

    Returns:
        ChunkVec list ordered by ``chunk_id`` ascending. Empty list when
        the db has no chunks (empty dir).
    """
    code_dir = Path(code_dir).resolve()
    db_path, _ = index_paths(code_dir)
    if not db_path.exists():
        raise FileNotFoundError(
            f"cocoindex sqlite missing: {db_path}. Run cocoindex_app first."
        )

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        matrices: dict[int, np.ndarray] = {}  # partition rowid → (1024, D)
        D: int | None = None
        for row in conn.execute(
            "SELECT rowid, vectors FROM code_chunks_vec_vector_chunks00"
        ):
            blob = row["vectors"]
            mat = np.frombuffer(blob, dtype="<f4").copy()
            d_here = mat.size // _SLOTS_PER_PARTITION_CHUNK
            if D is None:
                D = d_here
            elif d_here != D:
                raise ValueError(
                    f"vec dim mismatch in {db_path}: "
                    f"partition rowid={row['rowid']} D={d_here}, expected {D}"
                )
            matrices[int(row["rowid"])] = mat.reshape(
                _SLOTS_PER_PARTITION_CHUNK, D
            )

        if D is None:
            return []

        lang_for_partition: dict[int, str] = {  # partition chunk_id → language
            int(r["chunk_id"]): str(r["partition00"])
            for r in conn.execute(
                "SELECT chunk_id, partition00 FROM code_chunks_vec_chunks"
            )
        }

        slot_for_aux: dict[int, tuple[int, int]] = {  # aux rowid → (partition chunk_id, slot)
            int(r["rowid"]): (int(r["chunk_id"]), int(r["chunk_offset"]))
            for r in conn.execute(
                "SELECT rowid, chunk_id, chunk_offset FROM code_chunks_vec_rowids"
            )
        }

        out: list[ChunkVec] = []
        for r in conn.execute(
            "SELECT rowid, value00, value01, value02, value03 "
            "FROM code_chunks_vec_auxiliary ORDER BY rowid"
        ):
            aux_rowid = int(r["rowid"])
            slot = slot_for_aux.get(aux_rowid)
            if slot is None:
                # vec0 invariant violation — skip rather than raise so a
                # partially-built db stays usable.
                continue
            partition_id, slot_offset = slot
            mat = matrices.get(partition_id)
            if mat is None:
                continue
            language = lang_for_partition.get(partition_id, "")
            if languages is not None and language not in languages:
                continue
            start_line = int(r["value02"])
            end_line = int(r["value03"])
            if (end_line - start_line + 1) < min_line:
                continue
            content = str(r["value01"])
            out.append(
                ChunkVec(
                    chunk_id=aux_rowid,
                    language=language,
                    file_path=str(r["value00"]),
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                    content_hash=_content_hash(content),
                    vector=mat[slot_offset].copy(),
                )
            )
    finally:
        conn.close()

    out.sort(key=lambda c: c.chunk_id)
    return out
