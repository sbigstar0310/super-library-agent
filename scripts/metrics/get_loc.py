"""
CLI for LOC / token maintainability metric.

LLM-free alternative to scripts/metrics/get_mdl.py for cases where the vLLM scoring
server isn't running (or isn't worth spinning up). Counts non-blank lines and
tiktoken `cl100k_base` tokens per app/library, using the same task configs and
file filters as the MDL pipeline.

Two task modes (``--task``):
  - ``webgen``: React/JS apps with optional ui-lib (one combined file)
  - ``paperbench``: Python paper submissions (per-paper files + aggregate)

Usage:
    # paperbench
    python scripts/metrics/get_loc.py \\
        --task paperbench \\
        --backup-tag paperbench-mswe-baseline-rl4-t1

    # webgen (explicit)
    python scripts/metrics/get_loc.py \\
        --app_root_dir backups/poc-cc-ver4-t1/final/apps \\
        --lib_dir backups/poc-cc-ver4-t1/final/ui-lib
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Add el-agent/src to path so we can import utils.*
_agent_src = os.path.join(os.path.dirname(__file__), "..", "..", "el-agent", "src")
if _agent_src not in sys.path:
    sys.path.insert(0, os.path.abspath(_agent_src))

from utils.mdl import (  # noqa: E402
    get_maintainability_metrics,
    is_valid_codebase,
    load_task_config,
)


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _summarize(results: list[dict]) -> dict:
    if not results:
        return {
            "sum_app_loc": 0,
            "sum_app_tokens": 0,
            "avg_lib_loc": 0.0,
            "avg_lib_tokens": 0.0,
            "loc_total": 0,
            "tokens_total": 0,
            "total_apps": 0,
        }
    sum_app_loc = sum(int(r["app_loc"]) for r in results)
    sum_app_tok = sum(int(r["app_tokens"]) for r in results)
    avg_lib_loc = sum(int(r["lib_loc"]) for r in results) / len(results)
    avg_lib_tok = sum(int(r["lib_tokens"]) for r in results) / len(results)
    return {
        "sum_app_loc": sum_app_loc,
        "sum_app_tokens": sum_app_tok,
        "avg_lib_loc": round(avg_lib_loc, 2),
        "avg_lib_tokens": round(avg_lib_tok, 2),
        "loc_total": int(sum_app_loc + avg_lib_loc),
        "tokens_total": int(sum_app_tok + avg_lib_tok),
        "total_apps": len(results),
    }


def _latest_round(bench: str, tag: str, phase: str) -> int | None:
    """Return highest round_N under backups/<bench>/<tag>/final/ that has the
    requested phase, or None if nothing matches. Used as default for --round."""
    pattern = os.path.join(
        PROJECT_DIR, "backups", bench, tag, "final", "round_*", phase,
    )
    import glob, re
    rx = re.compile(r"round_(\d+)$")
    rounds = []
    for p in glob.glob(pattern):
        if not os.path.isdir(p):
            continue
        m = rx.match(os.path.basename(os.path.dirname(p)))
        if m:
            rounds.append(int(m.group(1)))
    return max(rounds) if rounds else None


def _resolve_paperbench_paths(
    args: argparse.Namespace,
) -> tuple[list[dict], str | None]:
    """paperbench unified layout:
        backups/paperbench/<tag>/final/round_<N>/<phase>/tasks/<id>/{submission,lib}/
    """
    if not args.backup_tag:
        raise SystemExit("--task paperbench requires --backup-tag <tag>.")
    if not args.phase:
        raise SystemExit("--task paperbench requires --phase <baseline|coding|apply|extract>.")
    bench = args.bench or "paperbench"
    round_num = args.round if args.round is not None else _latest_round(
        bench, args.backup_tag, args.phase,
    )
    if round_num is None:
        raise SystemExit(
            f"No round_N found for phase={args.phase} under "
            f"backups/{bench}/{args.backup_tag}/final/"
        )
    final_dir = os.path.join(
        PROJECT_DIR, "backups", bench, args.backup_tag, "final",
        f"round_{round_num}", args.phase,
    )
    if not os.path.isdir(final_dir):
        raise SystemExit(f"Backup not finalized: {final_dir}")
    tasks_dir = os.path.join(final_dir, "tasks")
    eval_root = os.path.join(
        PROJECT_DIR, "backups", bench, args.backup_tag,
        "eval_results", f"round_{round_num}", args.phase,
    )

    id_src = args.paper_ids or args.task_ids
    if id_src:
        ids = [p.strip() for p in id_src.split(",") if p.strip()]
    else:
        if not os.path.isdir(tasks_dir):
            raise SystemExit(f"tasks/ not found under {final_dir}")
        ids = sorted(
            d for d in os.listdir(tasks_dir)
            if os.path.isdir(os.path.join(tasks_dir, d))
        )

    entries: list[dict] = []
    for pid in ids:
        submission = os.path.join(tasks_dir, pid, "submission")
        if not os.path.isdir(submission):
            print(f"[skip] {pid}: submission dir not found at {submission}")
            continue
        lib_dir = args.lib_dir
        if lib_dir is None:
            per_task_lib = os.path.join(tasks_dir, pid, "lib")
            if os.path.isdir(per_task_lib):
                lib_dir = per_task_lib
        entries.append(
            {
                "label": pid,
                "app_dir": submission,
                "lib_dir": lib_dir,
                "save_dir": os.path.join(eval_root, "tasks", pid),
            }
        )
    return entries, eval_root


def _resolve_webgen_paths(args: argparse.Namespace, task) -> list[dict]:
    if not args.app_root_dir:
        raise SystemExit("--task webgen (legacy) requires --app_root_dir.")
    app_ids = set(args.app_ids.split(",")) if args.app_ids else None
    entries: list[dict] = []
    for app in sorted(os.listdir(args.app_root_dir)):
        app_dir = os.path.join(args.app_root_dir, app)
        if not is_valid_codebase(app_dir, task=task):
            continue
        if app_ids and app not in app_ids:
            continue
        entries.append(
            {
                "label": app,
                "app_dir": app_dir,
                "lib_dir": args.lib_dir,
                "save_dir": None,
            }
        )
    return entries


def _resolve_webgen_unified_paths(
    args: argparse.Namespace,
) -> tuple[list[dict], str | None]:
    """webgen unified layout:
        backups/webgen/<tag>/final/round_<N>/<phase>/tasks/<id>/submission/
    Library lookup:
      - per-task at tasks/<id>/lib/ if present (coding/apply phases)
      - --lib_dir override always wins
    """
    if not args.phase:
        raise SystemExit(
            "--task webgen --backup-tag requires --phase <baseline|coding|apply|extract>."
        )
    bench = args.bench or "webgen"
    round_num = args.round if args.round is not None else _latest_round(
        bench, args.backup_tag, args.phase,
    )
    if round_num is None:
        raise SystemExit(
            f"No round_N found for phase={args.phase} under "
            f"backups/{bench}/{args.backup_tag}/final/"
        )
    final_dir = os.path.join(
        PROJECT_DIR, "backups", bench, args.backup_tag, "final",
        f"round_{round_num}", args.phase,
    )
    if not os.path.isdir(final_dir):
        raise SystemExit(f"Backup not finalized: {final_dir}")
    tasks_dir = os.path.join(final_dir, "tasks")
    eval_root = os.path.join(
        PROJECT_DIR, "backups", bench, args.backup_tag,
        "eval_results", f"round_{round_num}", args.phase,
    )

    raw_ids = args.task_ids or args.app_ids
    if raw_ids:
        ids = [t.strip() for t in raw_ids.split(",") if t.strip()]
    else:
        if not os.path.isdir(tasks_dir):
            raise SystemExit(f"tasks/ not found under {final_dir}")
        ids = sorted(
            d for d in os.listdir(tasks_dir)
            if os.path.isdir(os.path.join(tasks_dir, d))
        )

    entries: list[dict] = []
    for tid in ids:
        submission = os.path.join(tasks_dir, tid, "submission")
        if not os.path.isdir(submission):
            print(f"[skip] {tid}: submission dir not found at {submission}")
            continue
        lib_dir = args.lib_dir
        if lib_dir is None:
            per_task_lib = os.path.join(tasks_dir, tid, "lib")
            if os.path.isdir(per_task_lib):
                lib_dir = per_task_lib
        entries.append(
            {
                "label": tid,
                "app_dir": submission,
                "lib_dir": lib_dir,
                "save_dir": os.path.join(eval_root, "tasks", tid),
            }
        )
    return entries, eval_root


def run(args: argparse.Namespace) -> None:
    # Each task loads its own config — comment-stripping comes from the
    # `language: python` parser_module used by the paperbench config.
    task = load_task_config(args.task)

    if args.task == "paperbench":
        entries, eval_root = _resolve_paperbench_paths(args)
    elif args.task == "webgen" and args.backup_tag:
        entries, eval_root = _resolve_webgen_unified_paths(args)
    else:
        entries = _resolve_webgen_paths(args, task)
        eval_root = None

    results: list[dict] = []
    for entry in entries:
        m = get_maintainability_metrics(
            entry["app_dir"],
            entry.get("lib_dir"),
            task=task,
        )
        rec = {
            "label": entry["label"],
            "app_loc": m["app_loc"],
            "app_tokens": m["app_tokens"],
            "lib_loc": m["lib_loc"],
            "lib_tokens": m["lib_tokens"],
            "save_dir": entry.get("save_dir"),
        }
        results.append(rec)
        print(
            f"  {entry['label']:42s}  app_loc={m['app_loc']:>6d}  "
            f"app_tok={m['app_tokens']:>7d}  "
            f"lib_loc={m['lib_loc']:>6d}  lib_tok={m['lib_tokens']:>7d}"
        )

    if args.task in ("paperbench", "webgen") and eval_root is not None:
        for rec in results:
            sd = rec["save_dir"]
            os.makedirs(sd, exist_ok=True)
            single = {
                "apps": [{k: v for k, v in rec.items() if k != "save_dir"}],
                "summary": _summarize([rec]),
            }
            out_path = os.path.join(sd, "loc_results.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(single, f, indent=2, ensure_ascii=False)
            print(f"Saved: {out_path}")
        if eval_root and results:
            os.makedirs(eval_root, exist_ok=True)
            agg = {
                "apps": [
                    {k: v for k, v in r.items() if k != "save_dir"} for r in results
                ],
                "summary": _summarize(results),
            }
            agg_path = os.path.join(eval_root, "loc_summary.json")
            with open(agg_path, "w", encoding="utf-8") as f:
                json.dump(agg, f, indent=2, ensure_ascii=False)
            print(f"Aggregated: {agg_path}")
        return

    summary = _summarize(results)
    out = {
        "apps": [{k: v for k, v in r.items() if k != "save_dir"} for r in results],
        "summary": summary,
    }
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, "loc_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved: {out_path}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(
        description="LOC / token maintainability metric (LLM-free)."
    )
    p.add_argument("--bench", default=None,
                   help="Backup root override under backups/ (default: same "
                        "as --task; e.g. webgen-maint for maintenance runs).")
    p.add_argument("--task", default="webgen",
                   choices=["webgen", "paperbench"])
    p.add_argument("--app_root_dir", default=None)
    p.add_argument("--lib_dir", default=None)
    p.add_argument("--app-ids", default=None, dest="app_ids")
    p.add_argument("--backup-tag", default=None, dest="backup_tag")
    p.add_argument("--phase", default=None,
                   help="[paperbench/webgen] phase under final/round_N/<phase>/tasks/ "
                        "(baseline | coding | apply | extract).")
    p.add_argument("--round", default=None, type=int,
                   help="[paperbench/webgen] round number (default: latest under final/).")
    p.add_argument("--paper-ids", default=None, dest="paper_ids")
    p.add_argument("--task-ids", default=None, dest="task_ids",
                   help="Comma-separated task IDs to evaluate.")
    p.add_argument(
        "--save_dir",
        default=None,
        help="[webgen] Output directory; ignored for paperbench.",
    )
    args = p.parse_args()

    if isinstance(args.lib_dir, str) and args.lib_dir.strip().lower() in {
        "none",
        "null",
        "",
    }:
        args.lib_dir = None

    run(args)


if __name__ == "__main__":
    main()
