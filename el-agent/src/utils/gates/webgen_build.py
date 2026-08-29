"""WebGen build gate — `npm install && npx vite build` correctness check.

Given a corpus of WebGen apps (``tasks_dir/<tid>/submission/``) and an optional
shared library, verifies that every app still builds. Each app is staged into a
scratch work dir (the input dirs are never mutated — they are often backups) and
built inside the ``sla-base`` docker image using the **same mount layout the
`WebgenLibraryAgent` uses**:

    /home/apps/<tid>/   ← this app's submission CONTENTS (no `submission/` layer)
    /home/apps/lib/     ← the shared library

That layout is load-bearing: LibraryAgent-refactored apps import the lib via
relative paths shaped like ``../../lib/src/index.js`` computed from exactly this
arrangement, so any mismatch produces FALSE build failures (see the README
gotchas). A shared host npm cache is bind-mounted so repeated installs are fast.

Public API::

    from utils.gates import run_webgen_build_gate, GateResult

CLI::

    uv --project el-agent run python -m utils.gates.webgen_build \
        --tasks-dir <dir> [--lib-dir <dir>] [--parallel N] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["GateResult", "run_webgen_build_gate"]


# Mirror run/base_full_run.py:_STAGE_IGNORE, plus build/eval artefacts that must
# not leak into a fresh build (dist/, shots/). node_modules is excluded so npm
# install runs clean rather than trusting a stale tree from the backup.
_STAGE_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "embeddings", ".git",
    "node_modules", "dist", "shots",
)

_DEFAULT_IMAGE = "sla-base"
_BUILD_CMD = "npm install --no-audit --no-fund && npx vite build"
_ERROR_TAIL_LINES = 40


@dataclass
class GateResult:
    """Outcome of building a single app.

    ``warnings`` surfaces non-fatal red flags (currently: absolute ``/home/...``
    import specifiers that build under the gate's dual-mount layout but can
    break under eval staging). They never fail the gate — callers may use them
    for tie-breaking or reporting.
    """

    ok: bool
    error_tail: str
    duration_s: float
    warnings: list[str] = field(default_factory=list)


# Import/require specifiers pointing at an absolute /home/... path. These build
# under the gate (lib is dual-mounted at /home/lib and /home/apps/lib) but are
# fragile under eval, which stages the lib elsewhere.
_ABS_IMPORT_RE = re.compile(
    r"""(?:from|import|require\(\s*)\s*['"](/home/[^'"]+)['"]"""
)
_SCAN_EXTS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".css")


def _scan_abs_imports(app_dir: str) -> list[str]:
    """Report ``<rel_path>: <specifier>`` for absolute /home/... imports."""
    found: list[str] = []
    for root, dirs, files in os.walk(app_dir):
        dirs[:] = [d for d in dirs if d not in ("node_modules", "dist", ".git")]
        for name in files:
            if not name.endswith(_SCAN_EXTS):
                continue
            path = os.path.join(root, name)
            try:
                text = Path(path).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for spec in _ABS_IMPORT_RE.findall(text):
                rel = os.path.relpath(path, app_dir)
                found.append(f"{rel}: {spec}")
    return found


def _tail(text: str, n: int = _ERROR_TAIL_LINES) -> str:
    lines = text.rstrip("\n").splitlines()
    return "\n".join(lines[-n:])


def _discover_task_ids(tasks_dir: str) -> list[str]:
    tids: list[str] = []
    for name in sorted(os.listdir(tasks_dir)):
        if os.path.isdir(os.path.join(tasks_dir, name, "submission")):
            tids.append(name)
    return tids


def _stage_app(tasks_dir: str, tid: str, dest: str) -> None:
    src = os.path.join(tasks_dir, tid, "submission")
    shutil.copytree(src, dest, ignore=_STAGE_IGNORE, dirs_exist_ok=True)


def _chown_to_host(path: str, image: str) -> None:
    """Reclaim ownership of files docker (running as root) wrote under ``path``.

    npm install writes ``node_modules`` and the shared cache as root; a plain
    host-side ``shutil.rmtree`` then fails with EPERM. Mirror eval_webgen.sh:
    chown the tree back to the invoking uid:gid via a throwaway container.
    """
    if not os.path.isdir(path):
        return
    subprocess.run(
        ["docker", "run", "--rm", "-v", f"{path}:/target", image,
         "chown", "-R", f"{os.getuid()}:{os.getgid()}", "/target"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _build_one(
    tid: str,
    app_dir: str,
    lib_dir: str | None,
    npm_cache: str,
    image: str,
    timeout_s: int,
) -> GateResult:
    """Run `npm install && npx vite build` for one staged app in docker."""
    warnings = _scan_abs_imports(app_dir)
    # A short uuid keeps the container name unique even when several gate
    # invocations build the same tid concurrently (librarian samples run
    # gates in parallel — a bare pid+tid name would collide across threads
    # and docker would refuse the second `run`).
    cname = f"webgen-gate-{os.getpid()}-{uuid.uuid4().hex[:8]}-{tid}"
    run_args = [
        "docker", "run", "--rm", "--name", cname,
        "-v", f"{app_dir}:/home/apps/{tid}",
    ]
    if lib_dir is not None:
        # Expose the same lib at BOTH mount points so the gate agrees with all
        # three corpus layouts (plan §7 checklist): the LibraryAgent/sla_naive
        # layout resolves lib at /home/apps/lib (relative `../../lib/...`), while
        # the sla-ours ApplyAgent and eval_webgen.sh put it at /home/lib (some
        # apps hardcode absolute `/home/lib/src/index.js`). Relative imports are
        # portable across both (lib is a sibling of the app root either way);
        # only absolute `/home/lib` imports need the alias. Both point at the
        # same host dir, so breaking the lib still breaks every app (sensitivity
        # preserved).
        run_args += [
            "-v", f"{lib_dir}:/home/apps/lib",
            "-v", f"{lib_dir}:/home/lib",
        ]
    run_args += [
        "-v", f"{npm_cache}:/npm-cache",
        "-e", "npm_config_cache=/npm-cache",
        "-w", f"/home/apps/{tid}",
        image, "bash", "-lc", _BUILD_CMD,
    ]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            run_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
        duration = time.monotonic() - start
        if proc.returncode == 0:
            return GateResult(
                ok=True, error_tail="", duration_s=duration, warnings=warnings,
            )
        return GateResult(
            ok=False,
            error_tail=_tail(proc.stdout or ""),
            duration_s=duration,
            warnings=warnings,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        # --rm won't fire on a killed-by-timeout run; force-remove the container.
        subprocess.run(
            ["docker", "rm", "-f", cname],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        partial = exc.output or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        tail = _tail(partial) if partial else ""
        return GateResult(
            ok=False,
            error_tail=f"TIMEOUT after {timeout_s}s\n{tail}".rstrip(),
            duration_s=duration,
            warnings=warnings,
        )


def run_webgen_build_gate(
    tasks_dir: str,
    lib_dir: str | None,
    *,
    only: set[str] | None = None,
    parallel: int = 4,
    timeout_s: int = 300,
    work_dir: str | None = None,
    image: str = _DEFAULT_IMAGE,
    keep_work: bool = False,
) -> dict[str, GateResult]:
    """Build every app under ``tasks_dir`` and report per-app pass/fail.

    Args:
        tasks_dir: dir containing ``<tid>/submission/`` app subdirs.
        lib_dir: shared library dir mounted at ``/home/apps/lib``; ``None`` for
            a zero-shot corpus that has no lib.
        only: if given, restrict to this set of task ids.
        parallel: max concurrent app builds (builds bind no ports — safe).
        timeout_s: per-app wall-clock budget for install+build.
        work_dir: scratch dir for staging + npm cache. Created if missing; a
            fresh tempdir under ``$TMPDIR`` is used when omitted.
        image: docker image to build in (default ``sla-base``).
        keep_work: keep the staging dir instead of deleting it on exit.

    Returns:
        ``{tid: GateResult}``. The input dirs are never mutated.
    """
    tasks_dir = os.path.abspath(tasks_dir)
    if not os.path.isdir(tasks_dir):
        raise FileNotFoundError(f"tasks_dir does not exist: {tasks_dir}")

    tids = _discover_task_ids(tasks_dir)
    if only is not None:
        tids = [t for t in tids if t in only]
    if not tids:
        return {}

    made_work = False
    if work_dir is None:
        import tempfile

        work_dir = tempfile.mkdtemp(prefix="webgen-gate-")
        made_work = True
    else:
        work_dir = os.path.abspath(work_dir)
        os.makedirs(work_dir, exist_ok=True)

    apps_root = os.path.join(work_dir, "apps")
    npm_cache = os.path.join(work_dir, ".npm-cache")
    # A reused work_dir may hold root-owned node_modules/cache from a prior run;
    # reclaim ownership up front so staging's rmtree + copytree don't hit EPERM.
    if os.path.isdir(apps_root) or os.path.isdir(npm_cache):
        _chown_to_host(work_dir, image)
    os.makedirs(apps_root, exist_ok=True)
    os.makedirs(npm_cache, exist_ok=True)

    staged_lib: str | None = None
    if lib_dir is not None:
        lib_dir = os.path.abspath(lib_dir)
        if not os.path.isdir(lib_dir):
            raise FileNotFoundError(f"lib_dir does not exist: {lib_dir}")
        staged_lib = os.path.join(work_dir, "lib")
        if os.path.isdir(staged_lib):
            shutil.rmtree(staged_lib)
        shutil.copytree(lib_dir, staged_lib, ignore=_STAGE_IGNORE)

    try:
        # Stage all apps first (cheap, sequential) so builds only do docker I/O.
        app_dirs: dict[str, str] = {}
        for tid in tids:
            dest = os.path.join(apps_root, tid)
            if os.path.isdir(dest):
                shutil.rmtree(dest)
            _stage_app(tasks_dir, tid, dest)
            app_dirs[tid] = dest

        results: dict[str, GateResult] = {}
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            futs = {
                pool.submit(
                    _build_one, tid, app_dirs[tid], staged_lib,
                    npm_cache, image, timeout_s,
                ): tid
                for tid in tids
            }
            for fut in as_completed(futs):
                tid = futs[fut]
                results[tid] = fut.result()
        return {tid: results[tid] for tid in tids}
    finally:
        # docker wrote node_modules/cache as root; reclaim so the caller (or the
        # rmtree below) can manage the tree without EPERM.
        _chown_to_host(work_dir, image)
        if made_work and not keep_work:
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table(results: dict[str, GateResult]) -> None:
    if not results:
        print("(no apps)")
        return
    n_ok = sum(1 for r in results.values() if r.ok)
    width = max(len(t) for t in results)
    print(f"{'task':<{width}}  {'result':<6}  {'dur(s)':>7}")
    print("-" * (width + 18))
    for tid, r in results.items():
        status = "ok" if r.ok else "FAIL"
        print(f"{tid:<{width}}  {status:<6}  {r.duration_s:>7.1f}")
    print("-" * (width + 18))
    print(f"{n_ok}/{len(results)} ok")
    for tid, r in results.items():
        if r.warnings:
            print(f"\n=== {tid} warnings (non-fatal) ===")
            for w in r.warnings:
                print(f"  {w}")
    for tid, r in results.items():
        if not r.ok:
            print(f"\n=== {tid} error_tail ===")
            print(r.error_tail)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WebGen build gate: npm install && npx vite build per app.",
    )
    parser.add_argument("--tasks-dir", required=True,
                        help="dir containing <tid>/submission/ subdirs")
    parser.add_argument("--lib-dir", default=None,
                        help="shared library dir (mounted at /home/apps/lib)")
    parser.add_argument("--only", default=None,
                        help="comma-separated task-id subset")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--work-dir", default=None,
                        help="scratch dir for staging + npm cache")
    parser.add_argument("--image", default=_DEFAULT_IMAGE)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--json-out", default=None,
                        help="write per-app results as JSON to this path")
    args = parser.parse_args()

    only = (
        {t.strip() for t in args.only.split(",") if t.strip()}
        if args.only else None
    )

    results = run_webgen_build_gate(
        args.tasks_dir,
        args.lib_dir,
        only=only,
        parallel=args.parallel,
        timeout_s=args.timeout_s,
        work_dir=args.work_dir,
        image=args.image,
        keep_work=args.keep_work,
    )

    _print_table(results)

    if args.json_out:
        payload = {tid: asdict(r) for tid, r in results.items()}
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\n[json] wrote {args.json_out}")


if __name__ == "__main__":
    main()
