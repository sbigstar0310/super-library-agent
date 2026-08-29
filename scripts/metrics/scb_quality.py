#!/usr/bin/env python3
"""SCB erosion + verbosity scoring for SLA agent outputs.

  verbosity = (clones ∪ ast-grep verbosity hits) / lines.loc
  erosion   = mass.high_cc_pct  (Σ cc·√sloc for cc>10) / Σ cc·√sloc

Modes:
  single (python only):  --snapshot-dir <path>
  batch (unified):       --task <paperbench|webgen> --backup-tag <tag>
                         --phase <phase> [--round N] [--task-ids id1,id2,...]
    ^ builds a single corpus (all submissions + lib once) and scores it
      once. Lib is counted once, not N times — matches SCB's metric definition.

Per-task language is auto-derived from --task:
  - paperbench       → Python (uses slop_code/SCB)
  - webgen           → JS/JSX/TS/TSX (uses utils.erosion + utils.verbosity
                       with ast-grep + Babel + jscpd; no SCB Python deps)

Re-execs via `uv run` inside SCB only when the Python scoring path is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCB_DIR = PROJECT_DIR / "data" / "slop-code-bench"
TASK_CONFIG_DIR = PROJECT_DIR / "el-agent" / "src" / "utils" / "mdl" / "configs"
EL_AGENT_SRC = PROJECT_DIR / "el-agent" / "src"

# Languages handled via SCB Python pipeline vs. the legacy JS pipeline that
# wraps utils/erosion (ESLint + Babel) + ast-grep + jscpd. The split is along
# the lines of which scoring stack each codebase needs — not which CLI to call.
PY_TASKS = {"paperbench"}
JS_TASKS = {"webgen"}

# Fallback when no task config is loaded (e.g. --snapshot-dir mode). SCB's
# hardcoded measure_snapshot_quality exclude list (driver.py:539) misses
# tests/build/dist/etc, so we always pre-filter at corpus-build time.
DEFAULT_SKIP_DIRS = {
    "__pycache__", ".venv", "venv", ".git", "build", "dist",
    ".pytest_cache", ".mypy_cache", "node_modules", ".ruff_cache",
}

# Source extensions used to tally file_count per task family.
PY_EXTS = {".py"}
JS_EXTS = {".js", ".jsx", ".ts", ".tsx"}


def load_skip_dirs(task: str | None) -> set[str]:
    """Read `ignore_dirs` from el-agent/src/utils/mdl/configs/<task>.yaml.

    Single source of truth shared with get_loc.py / get_mdl.py. Falls back to
    DEFAULT_SKIP_DIRS if the config can't be loaded.
    """
    if not task:
        return set(DEFAULT_SKIP_DIRS)
    cfg_path = TASK_CONFIG_DIR / f"{task}.yaml"
    if not cfg_path.is_file():
        print(f"[warn] task config not found: {cfg_path}; using defaults")
        return set(DEFAULT_SKIP_DIRS)
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception as e:
        print(f"[warn] failed to read {cfg_path}: {e}; using defaults")
        return set(DEFAULT_SKIP_DIRS)
    return (
        set(DEFAULT_SKIP_DIRS)
        | set(cfg.get("ignore_dirs", []))
        | set(cfg.get("ignore_files", []))   # fnmatch-eligible basenames
    )


# Populated per-run in main(); module-level default for any helper that
# imports SKIP_DIRS at import time.
SKIP_DIRS: set[str] = set(DEFAULT_SKIP_DIRS)


# ---------------------------------------------------------------------------
# Python (SCB) scoring path — re-exec into SCB venv if slop_code is missing.
# ---------------------------------------------------------------------------


def _bootstrap_scb() -> None:
    """Ensure slop_code is importable; re-exec via SCB's uv venv if not.

    Only called when --task paperbench triggers the Python scoring path.
    For --task webgen this is a no-op — we don't need SCB Python deps.
    """
    try:
        import slop_code  # noqa: F401
    except ImportError:
        if os.environ.get("_SCB_QUALITY_REEXEC") == "1":
            sys.exit(
                f"slop_code still missing after re-exec; run `uv sync` in {SCB_DIR}."
            )
        uv = shutil.which("uv") or sys.exit("`uv` not on PATH.")
        env = {**os.environ, "_SCB_QUALITY_REEXEC": "1"}
        os.execvpe(
            uv, [uv, "run", "--directory", str(SCB_DIR),
                 "python", __file__, *sys.argv[1:]], env,
        )
    # /usr/bin/sg (setgroups) shadows venv's `sg` (ast-grep) → verbosity falls
    # back to clones-only without this PATH prepend.
    venv_bin = str(Path(sys.executable).parent)
    if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")


def _score_python_corpus(snapshot_dir: Path, save_dir: Path | None = None) -> dict:
    """Score a corpus of .py files (paperbench)."""
    from slop_code.metrics import measure_snapshot_quality
    from slop_code.metrics.checkpoint.mass import compute_mass_metrics
    from slop_code.metrics.quality_io import save_quality_metrics

    snapshot_dir = snapshot_dir.resolve()
    entry = next(
        (p for p in sorted(snapshot_dir.rglob("*.py"))
         if not (set(p.parts) & SKIP_DIRS)),
        None,
    )
    if entry is None:
        return {"snapshot_dir": str(snapshot_dir), "error": "no .py files"}

    snap, files = measure_snapshot_quality(entry, snapshot_dir)
    loc = snap.lines.loc
    flagged = snap.verbosity_flagged_sloc_lines
    symbols = (sym.model_dump(mode="json") for fm in files for sym in fm.symbols)
    mass = compute_mass_metrics(symbols)

    if save_dir is not None:
        save_quality_metrics(save_dir, snap, files)

    return {
        "language": "python",
        "total_loc": loc,
        "file_count": snap.file_count,
        "verbosity": (flagged / loc) if loc else 0.0,
        "erosion": mass["mass.high_cc_pct"],
        "mass.cc": mass["mass.cc"],
    }


# ---------------------------------------------------------------------------
# JS (legacy webgen) scoring path — no SCB Python deps. Uses utils.erosion
# + utils.verbosity (which wrap node/ast-grep/jscpd as subprocesses).
# ---------------------------------------------------------------------------


def _ensure_el_agent_on_path() -> None:
    if str(EL_AGENT_SRC) not in sys.path:
        sys.path.insert(0, str(EL_AGENT_SRC))


def _score_js_corpus(snapshot_dir: Path, save_dir: Path | None = None) -> dict:
    """Score a corpus of JS/JSX/TS/TSX files (webgen).

    Calls compute_app_{erosion,verbosity} on the corpus root with lib_dir=None
    — the corpus already contains all apps + lib as a single tree, so this
    gives one measurement (not N).
    """
    _ensure_el_agent_on_path()
    from utils.erosion import compute_app_erosion
    from utils.verbosity import compute_app_verbosity

    snapshot_dir = snapshot_dir.resolve()
    erosion = compute_app_erosion(snapshot_dir, lib_dir=None, include_details=False)
    verbosity = compute_app_verbosity(
        snapshot_dir, lib_dir=None, include_details=False,
    )

    out = {
        "language": "javascript",
        "total_loc": verbosity["total_sloc"],
        "file_count": verbosity["files_scanned"],
        "verbosity": verbosity["verbosity"],
        "erosion": erosion["erosion"],
        "mass.cc": erosion["total_mass"],
        "cc_threshold": erosion["cc_threshold"],
        "rules_file": verbosity.get("rules_file"),
    }
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "verbosity_detail.json").write_text(
            json.dumps(
                {k: v for k, v in verbosity.items() if k != "flagged_samples"},
                indent=2, ensure_ascii=False,
            )
        )
        (save_dir / "erosion_detail.json").write_text(
            json.dumps(
                {k: v for k, v in erosion.items()
                 if k != "high_complexity_functions_detail"},
                indent=2, ensure_ascii=False,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Corpus construction (unified)
# ---------------------------------------------------------------------------


def find_round(bench: str, tag: str, phase: str) -> int | None:
    base = PROJECT_DIR / "backups" / bench / tag / "final"
    if not base.is_dir():
        return None
    rounds = [
        int(p.name.split("_")[1]) for p in base.iterdir()
        if p.name.startswith("round_") and (p / phase).is_dir()
    ]
    return max(rounds) if rounds else None


def collect_tasks(
    bench: str, tag: str, phase: str, round_num: int,
    task_ids: list[str] | None,
) -> tuple[list[dict], Path]:
    final_dir = (
        PROJECT_DIR / "backups" / bench / tag / "final"
        / f"round_{round_num}" / phase
    )
    tasks_dir = final_dir / "tasks"
    eval_phase = (
        PROJECT_DIR / "backups" / bench / tag / "eval_results"
        / f"round_{round_num}" / phase
    )
    if not tasks_dir.is_dir():
        sys.exit(f"tasks/ not found under {final_dir}")

    ids = task_ids or sorted(d.name for d in tasks_dir.iterdir() if d.is_dir())
    entries: list[dict] = []
    for tid in ids:
        sub_dir = tasks_dir / tid / "submission"
        if not sub_dir.is_dir():
            print(f"[skip] {tid}: no submission/ dir")
            continue
        lib_dir = tasks_dir / tid / "lib"
        entries.append({
            "label": tid,
            "submission_dir": sub_dir,
            "lib_dir": lib_dir if lib_dir.is_dir() else None,
        })

    # Phase-level lib (extract) sits at round_<N>/<phase>/lib, parallel to tasks/.
    phase_lib = final_dir / "lib"
    if phase_lib.is_dir():
        for e in entries:
            if e["lib_dir"] is None:
                e["lib_dir"] = phase_lib

    return entries, eval_phase


def build_corpus(
    entries: list[dict], target_dir: Path, count_exts: set[str],
) -> int:
    """Copy each task's submission/ + the shared lib (once) into target_dir.

    Per-task `lib/` directories under apply phase are byte-equivalent copies
    of the same shared lib. Picking the first one available is sufficient for
    corpus-level measurement.
    """
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    ignore = shutil.ignore_patterns(*SKIP_DIRS)
    lib_source = None
    for e in entries:
        dst = target_dir / e["label"]
        # ignore_dangling_symlinks: some runs leave a stray broken app-side
        # `lib -> /home/lib` symlink that would otherwise abort the whole corpus.
        shutil.copytree(e["submission_dir"], dst, ignore=ignore,
                        ignore_dangling_symlinks=True)
        if lib_source is None and e["lib_dir"] is not None:
            lib_source = e["lib_dir"]

    if lib_source is not None:
        lib_dst = target_dir / "lib"
        shutil.copytree(lib_source, lib_dst, ignore=ignore,
                        ignore_dangling_symlinks=True)

    file_count = 0
    for p in target_dir.rglob("*"):
        if p.is_file() and p.suffix in count_exts:
            file_count += 1
    return file_count


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Run paths
# ---------------------------------------------------------------------------


def run_single(snapshot_dir: str) -> None:
    """--snapshot-dir mode (python only; SCB pipeline)."""
    _bootstrap_scb()
    out = _score_python_corpus(Path(snapshot_dir))
    print(json.dumps(out, indent=2, ensure_ascii=False))


def run_batch(args: argparse.Namespace) -> None:
    if args.task in PY_TASKS:
        _bootstrap_scb()
        count_exts = PY_EXTS
        scorer = _score_python_corpus
    elif args.task in JS_TASKS:
        count_exts = JS_EXTS
        scorer = _score_js_corpus
    else:
        sys.exit(f"Unknown task: {args.task}")

    bench = args.bench or args.task
    round_num = args.round if args.round is not None else find_round(
        bench, args.backup_tag, args.phase,
    )
    if round_num is None:
        sys.exit(
            f"No round_N found for {bench}/{args.backup_tag}/{args.phase}"
        )

    task_ids = (
        [t.strip() for t in args.task_ids.split(",") if t.strip()]
        if args.task_ids else None
    )
    entries, eval_phase = collect_tasks(
        bench, args.backup_tag, args.phase, round_num, task_ids,
    )
    if not entries:
        sys.exit("No tasks resolved.")

    has_lib = any(e["lib_dir"] for e in entries)
    eval_root = (
        PROJECT_DIR / "eval" / bench / args.backup_tag
        / f"round_{round_num}" / args.phase
    )
    corpus_dir = eval_root / "scb_corpus"
    per_app_root = eval_root / "scb_corpus_per_app"
    summary_path = eval_phase / "scb_quality_summary.json"

    print(f"Building corpus at {corpus_dir} ({len(entries)} task(s))...")
    src_files = build_corpus(entries, corpus_dir, count_exts)
    print(f"  {src_files} source files (lib included: {has_lib})")
    print("Scoring corpus (serial; one tool invocation per scanner)...")

    try:
        result = scorer(corpus_dir, save_dir=eval_phase)
        per_app = _score_per_app(scorer, entries, per_app_root, count_exts)
        lib_only = _score_lib_only(scorer, entries, per_app_root, count_exts)
        summary = {
            "task": args.task,
            "tag": args.backup_tag,
            "phase": args.phase,
            "round": round_num,
            "language": result.get("language"),
            "total_apps": len(entries),
            "task_ids": [e["label"] for e in entries],
            "lib_included": has_lib,
            "total_loc": result.get("total_loc", 0),
            "file_count": result.get("file_count", 0),
            "verbosity": result.get("verbosity", 0.0),
            "erosion": result.get("erosion", 0.0),
            "mass.cc": result.get("mass.cc", 0.0),
            "per_app": per_app,
            "lib_only": lib_only,
        }
        if "cc_threshold" in result:
            summary["cc_threshold"] = result["cc_threshold"]
        if "error" in result:
            summary["error"] = result["error"]
        save_json(summary_path, {"summary": summary})
        print(f"\nSaved → {summary_path}")
        print(json.dumps(summary, indent=2))
    finally:
        shutil.rmtree(corpus_dir, ignore_errors=True)
        shutil.rmtree(per_app_root, ignore_errors=True)
        # Clean up empty parents up to eval/<task>/.
        for parent in corpus_dir.parents:
            if parent.name == args.task or not parent.is_dir():
                break
            try:
                parent.rmdir()
            except OSError:
                break


def _score_per_app(
    scorer, entries: list[dict], per_app_root: Path, count_exts: set[str],
) -> list[dict]:
    """Score each app individually as (app + its lib). One corpus per app.

    Reuses build_corpus with a 1-entry list so the corpus shape mirrors the
    overall run (lib mounted at <root>/lib). Per-app corpora are torn down
    one-at-a-time to bound disk use.
    """
    results: list[dict] = []
    print(f"Scoring per-app ({len(entries)} corpora; serial)...")
    for e in entries:
        app_corpus = per_app_root / e["label"]
        try:
            build_corpus([e], app_corpus, count_exts)
            r = scorer(app_corpus, save_dir=None)
            entry = {
                "label": e["label"],
                "lib_included": e["lib_dir"] is not None,
                "total_loc": r.get("total_loc", 0),
                "file_count": r.get("file_count", 0),
                "verbosity": r.get("verbosity", 0.0),
                "erosion": r.get("erosion", 0.0),
                "mass.cc": r.get("mass.cc", 0.0),
            }
            if "cc_threshold" in r:
                entry["cc_threshold"] = r["cc_threshold"]
            if "error" in r:
                entry["error"] = r["error"]
            results.append(entry)
            print(
                f"  [{e['label']}] loc={entry['total_loc']} "
                f"verb={entry['verbosity']:.4f} ero={entry['erosion']:.4f}"
            )
        finally:
            shutil.rmtree(app_corpus, ignore_errors=True)
    return results


def _score_lib_only(
    scorer, entries: list[dict], per_app_root: Path, count_exts: set[str],
) -> dict | None:
    """Score the shared library by itself — no apps mixed in.

    Returns None if no entry exposes a lib_dir (e.g. coding phase before the
    first extract). The lib content is identical across entries within a
    round, so picking the first available source is sufficient.
    """
    lib_source = next((e["lib_dir"] for e in entries if e["lib_dir"]), None)
    if lib_source is None:
        return None
    lib_corpus = per_app_root / "__lib_only__"
    try:
        if lib_corpus.exists():
            shutil.rmtree(lib_corpus)
        lib_corpus.mkdir(parents=True)
        ignore = shutil.ignore_patterns(*SKIP_DIRS)
        shutil.copytree(lib_source, lib_corpus / "lib", ignore=ignore)
        r = scorer(lib_corpus, save_dir=None)
        out = {
            "total_loc": r.get("total_loc", 0),
            "file_count": r.get("file_count", 0),
            "verbosity": r.get("verbosity", 0.0),
            "erosion": r.get("erosion", 0.0),
            "mass.cc": r.get("mass.cc", 0.0),
        }
        if "cc_threshold" in r:
            out["cc_threshold"] = r["cc_threshold"]
        if "error" in r:
            out["error"] = r["error"]
        print(
            f"  [lib_only] loc={out['total_loc']} "
            f"verb={out['verbosity']:.4f} ero={out['erosion']:.4f}"
        )
        return out
    finally:
        shutil.rmtree(lib_corpus, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Unified erosion + verbosity scoring. "
            "Python pipeline for paperbench, JS pipeline for webgen — "
            "auto-dispatched by --task."
        ),
    )
    p.add_argument("--snapshot-dir",
                   help="Single mode (python only): score one directory.")
    p.add_argument("--task", choices=sorted(PY_TASKS | JS_TASKS),
                   help="Batch mode: benchmark dispatcher.")
    p.add_argument("--bench", default=None,
                   help="Backup root override under backups/ (default: same "
                        "as --task; e.g. webgen-maint for maintenance runs).")
    p.add_argument("--backup-tag", dest="backup_tag")
    p.add_argument("--phase", help="baseline | coding | apply | extract")
    p.add_argument("--round", type=int)
    p.add_argument("--task-ids", dest="task_ids",
                   help="Comma-separated subset (default: all task dirs).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global SKIP_DIRS
    SKIP_DIRS = load_skip_dirs(args.task)
    if args.snapshot_dir:
        run_single(args.snapshot_dir)
    elif args.task and args.backup_tag and args.phase:
        run_batch(args)
    else:
        sys.exit("Either --snapshot-dir, or --task --backup-tag --phase [--round].")


if __name__ == "__main__":
    main()
