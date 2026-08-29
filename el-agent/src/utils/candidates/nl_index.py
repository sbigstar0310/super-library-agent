"""Build & load NL summary index on top of cocoindex chunks.

``ensure_nl_index(code_dir)`` reads chunks from the cocoindex sqlite, gets
one-line LLM summaries, and writes ``.cocoindex_code/index.json`` keyed by
chunk_id; ``load_nl_index`` returns it as-is. Schema per entry:

    {"language", "file_path", "lines": [start, end], "content_hash", "content_summary"}

Incremental: entries whose ``content_hash`` is unchanged skip the LLM call;
chunks no longer in sqlite are dropped.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .cocoindex_runner import cocoindex_app

_DEFAULT_MODEL = "gpt-5.4-nano"
_REASONING_EFFORT = "medium"
_SUMMARY_SYSTEM = (
    "You summarize a code chunk in ONE concise English sentence (<=160 chars). "
    "Describe what the chunk does — its purpose and notable mechanism if non-obvious. "
    "No quotes, no preamble like 'This chunk'. Output only the summary."
)

_CHUNKS_QUERY = """
SELECT
    r.rowid          AS chunk_id,
    c.partition00    AS language,
    a.value00        AS file_path,
    a.value01        AS content,
    a.value02        AS start_line,
    a.value03        AS end_line
FROM code_chunks_vec_rowids   r
JOIN code_chunks_vec_auxiliary a ON a.rowid    = r.rowid
JOIN code_chunks_vec_chunks    c ON c.chunk_id = r.chunk_id
ORDER BY r.rowid
"""


def index_paths(code_dir: str | Path) -> tuple[Path, Path]:
    """Returns (sqlite_db_path, index_json_path) for ``code_dir``."""
    base = Path(code_dir).resolve() / ".cocoindex_code"
    return base / "target_sqlite.db", base / "index.json"


def ensure_nl_index(
    code_dir: str | Path,
    *,
    model: str = _DEFAULT_MODEL,
    workers: int = 16,
    min_line: int = 5,
    skip_cocoindex: bool = False,
) -> Path:
    """Build/refresh ``index.json``. Returns its path.

    Args:
        skip_cocoindex: When True, assume ``cocoindex_app`` has already
            been run by the caller (avoids redundant ``ccc index``).
    """
    code_dir = Path(code_dir).resolve()
    if not skip_cocoindex:
        cocoindex_app(code_dir)

    db_path, out_path = index_paths(code_dir)
    if not db_path.exists():
        raise FileNotFoundError(
            f"cocoindex sqlite missing: {db_path}. Run cocoindex_app first."
        )

    chunks = _load_chunks(db_path, min_line=min_line)
    existing = _load_existing(out_path)

    index: dict[str, dict] = {}
    to_summarize: list[tuple[dict, str]] = []
    reused = 0
    for ch in chunks:
        key = str(ch["chunk_id"])
        h = _content_hash(ch["content"])
        prev = existing.get(key)
        if prev and prev.get("content_hash") == h and "content_summary" in prev:
            index[key] = _entry(ch, h, prev["content_summary"])
            reused += 1
        else:
            to_summarize.append((ch, h))

    print(
        f"[nl_index] {code_dir.name}: chunks={len(chunks)} "
        f"reused={reused} new={len(to_summarize)}",
        flush=True,
    )

    if to_summarize:
        _summarize_batch(to_summarize, index, model=model, workers=workers)

    ordered = {k: index[k] for k in sorted(index, key=int)}
    out_path.write_text(
        json.dumps(ordered, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def load_nl_index(code_dir: str | Path) -> dict[str, dict]:
    """Read ``index.json`` for ``code_dir``. Raises if missing."""
    _, out_path = index_paths(code_dir)
    if not out_path.exists():
        raise FileNotFoundError(
            f"NL index missing: {out_path}. Run ensure_nl_index first."
        )
    return json.loads(out_path.read_text(encoding="utf-8"))


# ---- internals ----------------------------------------------------------


def _load_chunks(db_path: Path, *, min_line: int) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(_CHUNKS_QUERY).fetchall()]
    finally:
        conn.close()
    return [r for r in rows if (r["end_line"] - r["start_line"] + 1) >= min_line]


def _load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[nl_index] {path} not valid JSON; ignoring cache", file=sys.stderr)
        return {}


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _entry(ch: dict, h: str, summary: str) -> dict:
    return {
        "language": ch["language"],
        "file_path": ch["file_path"],
        "lines": [ch["start_line"], ch["end_line"]],
        "content_hash": h,
        "content_summary": summary,
    }


def _summarize_batch(
    to_summarize: list[tuple[dict, str]],
    index: dict[str, dict],
    *,
    model: str,
    workers: int,
) -> None:
    # Lazy import — utils.llm pulls openai client.
    from utils.llm import get_client

    from ._provider import is_openai_model

    # OpenAI-family → OpenAI Responses API; everything else (deepseek, minimax,
    # qwen, …) → OpenRouter Chat Completions. (Summary model defaults to
    # gpt-5.4-nano, i.e. OpenAI, but any backbone is supported.)
    provider = "openai" if is_openai_model(model) else "openrouter"
    client = get_client(provider)
    use_responses_api = provider == "openai"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_summarize_one, client, model, ch, use_responses_api): (ch, h)
            for ch, h in to_summarize
        }
        for fut in as_completed(futures):
            ch, h = futures[fut]
            try:
                summary = fut.result()
            except Exception as e:
                print(
                    f"[nl_index] fail chunk {ch['chunk_id']} "
                    f"{ch['file_path']}: {e}",
                    file=sys.stderr,
                )
                continue
            index[str(ch["chunk_id"])] = _entry(ch, h, summary)


def _summarize_one(
    client,
    model: str,
    ch: dict,
    use_responses_api: bool,
    retries: int = 2,
) -> str:
    user_msg = (
        f"File: {ch['file_path']} (lines {ch['start_line']}-{ch['end_line']}, "
        f"language: {ch['language']})\n\n"
        f"```{ch['language']}\n{ch['content']}\n```"
    )
    from ._usage_log import record_aux_usage

    last_exc: Exception | None = None
    for _ in range(retries + 1):
        try:
            if use_responses_api:
                # gpt-5.x reasoning models — temperature/max_tokens unsupported.
                resp = client.responses.create(
                    model=model,
                    reasoning={"effort": _REASONING_EFFORT},
                    input=[
                        {"role": "system", "content": _SUMMARY_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                )
                record_aux_usage("summary", model, getattr(resp, "usage", None),
                                 context=ch.get("file_path", ""))
                summary = (resp.output_text or "").strip().replace("\n", " ")
            else:
                # OpenRouter: deepseek is pinned to its official provider (lab
                # policy); other backbones use OpenRouter's default routing.
                from ._provider import is_deepseek_model, openrouter_model_id
                extra_body: dict = {}
                if is_deepseek_model(model):
                    extra_body["provider"] = {"only": ["deepseek"], "allow_fallbacks": False}
                resp = client.chat.completions.create(
                    model=openrouter_model_id(model),
                    messages=[
                        {"role": "system", "content": _SUMMARY_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0,
                    max_tokens=400,
                    extra_body=extra_body,
                )
                record_aux_usage("summary", model, getattr(resp, "usage", None),
                                 context=ch.get("file_path", ""))
                summary = (resp.choices[0].message.content or "").strip().replace("\n", " ")
            if summary:
                return summary
        except Exception as e:
            last_exc = e
    # Fallback file-path stub: downstream prompts iterate over all chunk_ids,
    # so a stub is better than dropping the row.
    if last_exc is not None:
        print(f"[nl_index] retries exhausted for {ch['file_path']}: {last_exc}", file=sys.stderr)
    return f"chunk in {ch['file_path']} (lines {ch['start_line']}-{ch['end_line']})"
