"""Run `ccc init` + `ccc index` for a code dir.

``ccc index`` is incremental (file scan only on an unchanged dir). Output:
``<code_dir>/.cocoindex_code/target_sqlite.db`` plus the ``cocoindex.db/`` lmdb
cache, both regenerable. Invoked via ``uv run --project <el-agent root> ccc``
so it uses the project's locked cocoindex-code version regardless of active venv.

Chunk params default to cocoindex's hardcoded 1000/250/150; override via
:func:`cocoindex_app` kwargs or ``CCC_CHUNK_SIZE / CCC_MIN_CHUNK_SIZE /
CCC_CHUNK_OVERLAP``. Any non-default value registers a custom chunker
(``utils.candidates._chunker:chunker``) in ``settings.yml``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

# el-agent/ root — holds the pyproject.toml that pins cocoindex-code.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
# el-agent/src — added to subprocess PYTHONPATH so the daemon can import
# our ``_chunker`` module when a custom chunker is registered.
_AGENT_SRC = str(Path(__file__).resolve().parents[2])

# Doc/config extensions filtered out of include_patterns — we index only
# source code, not the docs/configs cocoindex treats as indexable by default.
_NON_CODE_PATTERNS: frozenset[str] = frozenset({
    "**/*.md", "**/*.mdx", "**/*.txt", "**/*.rst",
    "**/*.json", "**/*.xml",
    "**/*.yaml", "**/*.yml", "**/*.toml",
})

# Cocoindex's hardcoded defaults — matching these skips chunker registration
# and lets cocoindex's vanilla splitter run.
_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_MIN_CHUNK_SIZE = 250
_DEFAULT_CHUNK_OVERLAP = 150

# Code extensions the custom chunker binds to when active (source-code subset
# of cocoindex's DEFAULT_INCLUDED_PATTERNS).
_CUSTOM_CHUNKER_EXTS: tuple[str, ...] = (
    "py", "pyi", "js", "jsx", "ts", "tsx", "mjs", "cjs",
    "rs", "go", "java",
    "c", "h", "cpp", "hpp", "cc", "cxx", "hxx", "hh", "cs",
    "sh", "bash", "zsh",
    "php", "lua", "rb", "swift", "kt", "kts", "scala", "r",
    "html", "htm", "svelte", "vue", "css", "scss",
)
_CHUNKER_MODULE = "utils.candidates._chunker:chunker"


def cocoindex_app(
    code_dir: str | Path,
    *,
    force_reinit: bool = False,
    quiet: bool = True,
    chunk_size: int | None = None,
    min_chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Path:
    """Create or update the cocoindex-code index for ``code_dir``.

    Idempotent. Returns the path to ``target_sqlite.db``.

    Args:
        code_dir: Directory to index (app submission, lib, anything).
        force_reinit: When True, wipe ``.cocoindex_code/`` first (e.g. after
            settings drift or schema mismatch).
        quiet: Suppress CLI stdout (errors still printed).
        chunk_size / min_chunk_size / chunk_overlap: Override cocoindex's
            1000/250/150 defaults. Precedence ``kwarg > env var > default``.
            Any non-default value registers a custom chunker and injects params
            into the ccc subprocess. Daemon caches the chunker module, so a
            mid-run change needs a daemon restart (see :mod:`._chunker`).
    """
    code_dir = Path(code_dir).resolve()
    if not code_dir.is_dir():
        raise FileNotFoundError(f"cocoindex_app: not a dir: {code_dir}")

    size = _resolve_chunk_param(chunk_size, "CCC_CHUNK_SIZE", _DEFAULT_CHUNK_SIZE)
    minsize = _resolve_chunk_param(min_chunk_size, "CCC_MIN_CHUNK_SIZE", _DEFAULT_MIN_CHUNK_SIZE)
    overlap = _resolve_chunk_param(chunk_overlap, "CCC_CHUNK_OVERLAP", _DEFAULT_CHUNK_OVERLAP)
    use_custom = (size, minsize, overlap) != (
        _DEFAULT_CHUNK_SIZE, _DEFAULT_MIN_CHUNK_SIZE, _DEFAULT_CHUNK_OVERLAP,
    )

    cci_dir = code_dir / ".cocoindex_code"
    if force_reinit and cci_dir.exists():
        shutil.rmtree(cci_dir)

    settings_path = cci_dir / "settings.yml"
    if not settings_path.exists():
        _run_ccc(["init", "-f"], cwd=code_dir, quiet=quiet)

    _filter_non_code_patterns(settings_path)
    if use_custom:
        _register_custom_chunker(settings_path)

    extra_env = _chunker_env(size, minsize, overlap) if use_custom else None
    _run_ccc(["index"], cwd=code_dir, quiet=quiet, extra_env=extra_env)
    # `ccc index` exits 0 even when it fails (e.g. "Failed to connect to
    # daemon after starting it" under concurrent-index load), so verify the
    # artifact and retry with backoff — otherwise ensure_nl_index raises
    # "cocoindex sqlite missing", is soft-caught, and extract/apply silently
    # run WITHOUT NL candidates (observed on the mm3 campaign, 2026-07-10).
    db_path = cci_dir / "target_sqlite.db"
    for delay in (5, 15):
        if db_path.exists():
            break
        print(
            f"[cocoindex_app] index produced no db, retry in {delay}s: {code_dir}",
            flush=True,
        )
        time.sleep(delay)
        _run_ccc(["index"], cwd=code_dir, quiet=quiet, extra_env=extra_env)
    # Do NOT `ccc daemon stop` here: the daemon is global (one socket per user)
    # and shared by concurrent callers — stopping it kills another trial's
    # in-flight index and spawns orphan daemons as survivors race to respawn.
    return db_path


def _resolve_chunk_param(explicit: int | None, env_key: str, default: int) -> int:
    """kwarg > env var > default."""
    if explicit is not None:
        return explicit
    val = os.environ.get(env_key)
    return int(val) if val else default


def _register_custom_chunker(settings_path: Path) -> None:
    """Add our chunker mapping to ``chunkers:`` for every code extension.

    Idempotent — extensions already mapped are skipped, so a user-defined
    chunker for a specific extension takes precedence.
    """
    if not settings_path.is_file():
        return
    with open(settings_path) as f:
        data = yaml.safe_load(f) or {}
    existing = {c.get("ext") for c in (data.get("chunkers") or [])}
    add = [
        {"ext": ext, "module": _CHUNKER_MODULE}
        for ext in _CUSTOM_CHUNKER_EXTS if ext not in existing
    ]
    if not add:
        return
    data["chunkers"] = (data.get("chunkers") or []) + add
    with open(settings_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def _chunker_env(size: int, minsize: int, overlap: int) -> dict[str, str]:
    """Env for the ccc subprocess. The chunker reads ``CCC_*`` at import time;
    ``PYTHONPATH`` must include ``el-agent/src`` so the daemon can import it.
    """
    return {
        "CCC_CHUNK_SIZE": str(size),
        "CCC_MIN_CHUNK_SIZE": str(minsize),
        "CCC_CHUNK_OVERLAP": str(overlap),
        "PYTHONPATH": _AGENT_SRC + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }


def _filter_non_code_patterns(settings_path: Path) -> None:
    """Drop doc/config globs from ``include_patterns``.

    Rewrites only when needed, preserving settings.yml mtime so ``ccc index``
    stays incremental.
    """
    if not settings_path.is_file():
        return
    with open(settings_path) as f:
        data = yaml.safe_load(f) or {}
    includes = data.get("include_patterns") or []
    filtered = [p for p in includes if p not in _NON_CODE_PATTERNS]
    if filtered == includes:
        return
    data["include_patterns"] = filtered
    with open(settings_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def _run_ccc(
    args: list[str], *, cwd: Path, quiet: bool, extra_env: dict[str, str] | None = None,
) -> None:
    cmd = ["uv", "run", "--project", _PROJECT_ROOT, "ccc", *args]
    # Cap thread pools — OpenBLAS's default per-process pool (=CPU count) across
    # accumulating ccc daemons exceeds the shared cgroup pids.max=2048, killing
    # daemons at import with EAGAIN.
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4",
           "MKL_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4"}
    if extra_env:
        env.update(extra_env)
    kwargs: dict = {"cwd": str(cwd), "check": True, "env": env}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.PIPE
    try:
        subprocess.run(cmd, **kwargs)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        raise RuntimeError(
            f"ccc {' '.join(args)} failed in {cwd} (exit {e.returncode}):\n{err}"
        ) from e


