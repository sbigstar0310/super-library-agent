"""Replacement chunker for cocoindex-code, registered only for non-default
chunk params. Reads ``CCC_CHUNK_SIZE / CCC_MIN_CHUNK_SIZE / CCC_CHUNK_OVERLAP``
at import time, falling back to cocoindex's defaults (1000/250/150).

Gotcha: the cocoindex daemon caches loaded chunker modules (registry is
``tracked=False``), so the first import wins — changing env mid-daemon has no
effect. Restart the daemon or point ``COCOINDEX_CODE_DIR`` elsewhere to switch
params. Consumed by cocoindex only via the module-path string
``utils.candidates._chunker:chunker``; not for direct import.
"""
from __future__ import annotations

import os
from pathlib import Path

from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex_code.chunking import Chunk

_CHUNK_SIZE = int(os.environ.get("CCC_CHUNK_SIZE", "1000"))
_MIN_CHUNK_SIZE = int(os.environ.get("CCC_MIN_CHUNK_SIZE", "250"))
_CHUNK_OVERLAP = int(os.environ.get("CCC_CHUNK_OVERLAP", "150"))

_splitter = RecursiveSplitter()


def chunker(path: Path, content: str) -> tuple[str | None, list[Chunk]]:
    """cocoindex's default chunker with overridable params."""
    language = detect_code_language(filename=path.name) or "text"
    chunks = _splitter.split(
        content,
        chunk_size=_CHUNK_SIZE,
        min_chunk_size=_MIN_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        language=language,
    )
    return None, chunks
