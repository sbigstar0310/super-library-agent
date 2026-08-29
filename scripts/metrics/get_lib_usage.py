"""
Library symbol usage stats — avg imports per extracted lib symbol across apps.

Mirrors the CLI conventions of ``scripts/metrics/get_loc.py`` and ``scripts/metrics/get_mdl.py``:
``--task``, ``--backup-tag``, ``--round``, ``--phase``, ``--task-ids``.

Task modes (``--task``):
    - ``webgen``    : JS/JSX lib (``lib/src/index.js`` barrel), regex import
                      resolution. Reads ``backups/webgen/<tag>/final/...``.
                      Phases: ``apply`` (only) — extract snapshot not produced
                      for webgen.
    - ``paperbench``: Python lib, AST import resolution. Reads
                      ``backups/paperbench/<tag>/final/...``. The paperbench
                      lib lives at ``tasks/<id>/lib/lib/`` (one level deeper,
                      alongside ``extract_map.md``). Phases:
                      ``apply`` only — extract snapshot not produced.

Usage:
    # webgen, apply phase, latest round
    python scripts/metrics/get_lib_usage.py --task webgen --backup-tag sla-ours-v3-t1

    # specific round + task subset
    python scripts/metrics/get_lib_usage.py --task webgen --backup-tag sla-ours-v3-t1 \\
        --round 4 --task-ids 000027,000051

    # multi-trial aggregation (backup_tag is treated as prefix)
    python scripts/metrics/get_lib_usage.py --task webgen --backup-tag sla-ours-v3 \\
        --trials t1,t2,t3

    # JSON output to disk
    python scripts/metrics/get_lib_usage.py --task webgen --backup-tag sla-ours-v3-t1 \\
        --save_dir analysis/lib_usage/

Metric definitions:
    - exports         : total symbols defined in lib (barrel-exposed for webgen)
    - app_imports     : sum of per-symbol import counts from app files only
                        (lib-internal references inside lib_dir are excluded)
    - live_symbols    : symbols imported by ≥1 app file
    - dead_symbols    : exports - live_symbols
    - apps_used       : distinct app/task dirs that import any lib symbol
    - avg_per_sym_all : app_imports / exports
    - avg_per_sym_live: app_imports / live_symbols

Note: counts are ``import`` occurrences, not call sites. ``import { X }``
once in a file = 1, regardless of how many times X is called.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BACKUPS_DIR = ROOT / "backups"

# Make utils.library_usage importable for --phase apply
_agent_src = ROOT / "el-agent" / "src"
if str(_agent_src) not in sys.path:
    sys.path.insert(0, str(_agent_src))


def _backups_for_task(task_kind: str) -> Path:
    return BACKUPS_DIR / task_kind


def _detect_last_round(tag_dir: Path) -> int | None:
    final = tag_dir / "final"
    if not final.is_dir():
        return None
    rounds = []
    for p in final.iterdir():
        if p.is_dir() and p.name.startswith("round_"):
            try:
                rounds.append(int(p.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return max(rounds) if rounds else None


def _resolve_round_dir(tag: str, round_num: int | None, task_kind: str) -> Path:
    tag_dir = _backups_for_task(task_kind) / tag
    if not tag_dir.is_dir():
        raise FileNotFoundError(f"No tag dir: {tag_dir}")
    if round_num is None:
        round_num = _detect_last_round(tag_dir)
        if round_num is None:
            raise FileNotFoundError(f"No round_* under {tag_dir}/final")
    p = tag_dir / "final" / f"round_{round_num}"
    if not p.is_dir():
        raise FileNotFoundError(f"No round dir: {p}")
    return p


def _finalize(label: str, source: str, n_exp: int, sum_imp: int,
              live: int, apps_used: int, total_tasks: int) -> dict:
    dead = n_exp - live
    return {
        "source": source,
        "label": label,
        "exports": n_exp,
        "app_imports": sum_imp,
        "live_symbols": live,
        "dead_symbols": dead,
        "dead_pct": (100.0 * dead / n_exp) if n_exp else 0.0,
        "apps_used": apps_used,
        "total_tasks": total_tasks,
        "apps_coverage_pct": (100.0 * apps_used / total_tasks) if total_tasks else 0.0,
        "avg_per_sym_all": (sum_imp / n_exp) if n_exp else 0.0,
        "avg_per_sym_live": (sum_imp / live) if live else 0.0,
    }


def compute_stats_apply(
    round_dir: Path,
    label: str,
    task_kind: str,
    task_ids: list[str] | None = None,
    phase: str = "apply",
) -> dict:
    """Rescan post-phase state under round_dir/<phase>/tasks/*/{lib,submission}.

    Works for any phase that produces per-task ``lib/`` + ``submission/``
    (typically ``coding`` and ``apply``). Per-task lib copies are identical
    mirrors. We use the first task's lib/ as the canonical reference, scan
    every task's submission/ as consumers, and treat sibling task libs as
    alias roots so submissions that import from their own per-task lib
    still resolve. lib-internal refs are dropped from the app-only aggregate.

    Dispatches the counter by ``task_kind``:
      - paperbench → Python AST counter (utils.library_usage.counter)
      - webgen     → JS/JSX regex counter (utils.library_usage.js_counter)

    ``task_ids``: optional filter; only the listed task subdirs participate.
    """
    tasks_dir = round_dir / phase / "tasks"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"No {phase}/tasks dir: {tasks_dir}")

    # Skip pipeline-placeholder dirs (e.g. webgen-naive's empty `__library__`)
    task_paths = sorted(
        p for p in tasks_dir.iterdir()
        if p.is_dir() and not p.name.startswith("__")
    )
    if task_ids:
        wanted = set(task_ids)
        task_paths = [p for p in task_paths if p.name in wanted]
        missing = wanted - {p.name for p in task_paths}
        if missing:
            print(f"  warn: task-ids not found: {sorted(missing)}", file=sys.stderr)
    if not task_paths:
        raise FileNotFoundError(f"No tasks resolved under {tasks_dir}")

    # paperbench wraps the lib package one directory deeper:
    #   tasks/<id>/lib/{extract_map.md, lib/...}
    def _resolve_lib_root(tp: Path) -> Path | None:
        cand = tp / "lib"
        if not cand.is_dir():
            return None
        if task_kind == "paperbench":
            inner = cand / "lib"
            return inner if inner.is_dir() else None
        return cand

    lib_dir = None
    for tp in task_paths:
        cand = _resolve_lib_root(tp)
        if cand is not None:
            lib_dir = cand
            break
    if lib_dir is None:
        raise FileNotFoundError(f"No lib/ found under {tasks_dir}")

    submission_dirs = [tp / "submission" for tp in task_paths if (tp / "submission").is_dir()]
    extra_libs = [
        r for r in (_resolve_lib_root(tp) for tp in task_paths)
        if r is not None and r != lib_dir
    ]

    if task_kind == "paperbench":
        from utils.library_usage import count_symbol_usage, load_usage_config
        cfg = load_usage_config(task_kind)
        consumer_dirs = [lib_dir] + submission_dirs
        canonicals, per_consumer = count_symbol_usage(lib_dir, consumer_dirs, cfg)
    elif task_kind == "webgen":
        from utils.library_usage.js_counter import count_symbol_usage as js_count
        consumer_dirs = [lib_dir] + submission_dirs
        canonicals, per_consumer = js_count(
            lib_dir, consumer_dirs, extra_lib_roots=extra_libs,
        )
    else:
        raise ValueError(f"--task {task_kind} not supported for lib usage")

    # App-only aggregate: drop consumer files that live under any task's lib/
    lib_resolved_prefixes = {str((tp / "lib").resolve()) for tp in task_paths if (tp / "lib").is_dir()}
    app_only_agg: dict[str, int] = {}
    apps_used: set[str] = set()
    for fp, counts in per_consumer.items():
        resolved = str(Path(fp).resolve())
        in_lib = any(resolved.startswith(pref + os.sep) or resolved == pref
                     for pref in lib_resolved_prefixes)
        if in_lib:
            continue
        for sym, c in counts.items():
            app_only_agg[sym] = app_only_agg.get(sym, 0) + c
        parts = resolved.split(os.sep)
        if "tasks" in parts:
            i = parts.index("tasks")
            if i + 1 < len(parts):
                apps_used.add(parts[i + 1])

    sum_imp = sum(app_only_agg.values())
    live = sum(1 for v in app_only_agg.values() if v > 0)
    return _finalize(label, str(round_dir / "apply"), len(canonicals), sum_imp,
                     live, len(apps_used), len(task_paths))


def compute_for_tag(
    tag: str,
    round_num: int | None,
    phase: str,
    task_kind: str,
    task_ids: list[str] | None = None,
) -> tuple[dict, int]:
    """Return (stats_row, resolved_round_num)."""
    round_dir = _resolve_round_dir(tag, round_num, task_kind)
    resolved_round = int(round_dir.name.split("_")[1])
    if phase in {"apply", "coding"}:
        return (
            compute_stats_apply(round_dir, tag, task_kind, task_ids=task_ids, phase=phase),
            resolved_round,
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")


def _print_table(label: str, row: dict) -> None:
    print(f"\n=== {label} ===")
    print(
        f"  {'exp':>4s} {'imp':>5s} {'live':>5s} {'dead':>5s} {'dead%':>6s} "
        f"{'apps':>8s} {'cov%':>6s} {'avg/all':>8s} {'avg/live':>9s}"
    )
    print(
        f"  {row['exports']:>4d} {row['app_imports']:>5d} "
        f"{row['live_symbols']:>5d} {row['dead_symbols']:>5d} "
        f"{row['dead_pct']:>5.1f}% "
        f"{row['apps_used']:>3d}/{row['total_tasks']:<3d} "
        f"{row['apps_coverage_pct']:>5.1f}% "
        f"{row['avg_per_sym_all']:>8.2f} {row['avg_per_sym_live']:>9.2f}"
    )


def _default_save_dir(task_kind: str, tag: str, round_num: int, phase: str) -> Path:
    """Mirror get_loc/get_mdl: backups/<task>/<tag>/eval_results/round_N/<phase>/."""
    return (
        BACKUPS_DIR / task_kind / tag / "eval_results"
        / f"round_{round_num}" / phase
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--task", choices=["webgen", "paperbench"], default="webgen",
                    help="Task / bench layout (default: webgen).")
    ap.add_argument("--backup-tag", dest="backup_tag", required=True,
                    help="Backup tag under backups/<task>/.")
    ap.add_argument("--round", type=int, default=None,
                    help="Round number (default: auto-detect last round).")
    ap.add_argument("--phase", choices=["apply", "coding"], default="apply",
                    help="apply (default) / coding = per-task lib+submission rescan.")
    ap.add_argument("--task-ids", dest="task_ids", default=None,
                    help="Comma-separated task IDs to include (default: all).")
    ap.add_argument("--save_dir", default=None,
                    help="Override output directory. Default: "
                         "backups/<task>/<tag>/eval_results/round_N/<phase>/.")
    ap.add_argument("--no-save", action="store_true",
                    help="Print only; do not write JSON.")
    args = ap.parse_args()

    task_ids = None
    if args.task_ids:
        task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]

    row, resolved_round = compute_for_tag(
        args.backup_tag, args.round, args.phase, args.task, task_ids=task_ids,
    )

    label = f"{args.backup_tag} [{args.task}/round_{resolved_round}/{args.phase}]"
    _print_table(label, row)

    if not args.no_save:
        save_dir = (
            Path(args.save_dir) if args.save_dir
            else _default_save_dir(args.task, args.backup_tag, resolved_round, args.phase)
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / "lib_usage_results.json"
        payload = {
            "backup_tag": args.backup_tag,
            "task": args.task,
            "round": resolved_round,
            "phase": args.phase,
            "task_ids": task_ids,
            "result": row,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
