"""Unified backup/runs layout helpers.

Path spec (cross-benchmark, all modes):

    runs/<bench>/<tag>/
        round_<N>/
            <phase>/                # baseline | coding | apply | extract
                tasks/<task_id>/
                    submission/
                    lib/            # baseline 제외; extract phase 제외
                lib/                # extract phase 전용
        logs/round_<N>/<phase>/tasks/<task_id>/...

    backups/<bench>/<tag>/
        run.sh
        final/round_<N>/<phase>/...      # mirrors runs/.../round_<N>/<phase>/
        logs/round_<N>/<phase>/...       # top-level sibling
        eval_results/round_<N>/<phase>/
            tasks/<task_id>/
            <aggregate>.json

Conventions:
- non-full mode = single round_0
- full mode = round_1, round_2, ...
- task_id used verbatim (no normalization)
- extract phase: lib at phase directory level (not per-task)
"""

from __future__ import annotations
import glob
import os
import re


# Canonical phase names (CLI --mode values), used directly as path components.
# `local_extract` (RAL-only, runs between coding and extract) was added with the
# 2-Level Extract migration; the "extract" string is kept for backward compat
# with pre-existing backups under round_<N>/extract/.
PHASES = ("baseline", "coding", "apply", "extract", "local_extract")


def runs_root(project_dir: str, bench: str, tag: str) -> str:
    return os.path.join(project_dir, "runs", bench, tag)


def backups_root(project_dir: str, bench: str, tag: str) -> str:
    return os.path.join(project_dir, "backups", bench, tag)


def runs_phase_dir(project_dir: str, bench: str, tag: str,
                   round_num: int, phase: str) -> str:
    return os.path.join(runs_root(project_dir, bench, tag),
                        f"round_{round_num}", phase)


def runs_logs_dir(project_dir: str, bench: str, tag: str,
                  round_num: int, phase: str) -> str:
    return os.path.join(runs_root(project_dir, bench, tag), "logs",
                        f"round_{round_num}", phase)


def backup_final_phase_dir(project_dir: str, bench: str, tag: str,
                           round_num: int, phase: str) -> str:
    return os.path.join(backups_root(project_dir, bench, tag), "final",
                        f"round_{round_num}", phase)


def backup_logs_phase_dir(project_dir: str, bench: str, tag: str,
                          round_num: int, phase: str) -> str:
    return os.path.join(backups_root(project_dir, bench, tag), "logs",
                        f"round_{round_num}", phase)


def backup_eval_phase_dir(project_dir: str, bench: str, tag: str,
                          round_num: int, phase: str) -> str:
    return os.path.join(backups_root(project_dir, bench, tag), "eval_results",
                        f"round_{round_num}", phase)


def task_dir_in_runs(project_dir: str, bench: str, tag: str,
                     round_num: int, phase: str, task_id: str) -> str:
    return os.path.join(runs_phase_dir(project_dir, bench, tag, round_num, phase),
                        "tasks", task_id)


def task_dir_in_backup(project_dir: str, bench: str, tag: str,
                       round_num: int, phase: str, task_id: str) -> str:
    return os.path.join(backup_final_phase_dir(project_dir, bench, tag,
                                               round_num, phase),
                        "tasks", task_id)


def extract_lib_dir_in_runs(project_dir: str, bench: str, tag: str,
                            round_num: int) -> str:
    """Cross-task library output dir for extract phase (working copy)."""
    return os.path.join(runs_phase_dir(project_dir, bench, tag,
                                       round_num, "extract"), "lib")


def extract_lib_dir_in_backup(project_dir: str, bench: str, tag: str,
                              round_num: int) -> str:
    return os.path.join(backup_final_phase_dir(project_dir, bench, tag,
                                               round_num, "extract"), "lib")


_ROUND_RE = re.compile(r"round_(\d+)$")


def latest_round_in_backup(project_dir: str, bench: str, tag: str,
                           phase: str) -> int | None:
    """Find the highest round_N under backups/<bench>/<tag>/final/ that has
    the given phase directory. Returns None if no rounds exist."""
    pattern = os.path.join(backups_root(project_dir, bench, tag), "final",
                           "round_*", phase)
    matches = []
    for path in glob.glob(pattern):
        if not os.path.isdir(path):
            continue
        parent = os.path.basename(os.path.dirname(path))
        m = _ROUND_RE.match(parent)
        if m:
            matches.append(int(m.group(1)))
    return max(matches) if matches else None


def list_rounds_in_backup(project_dir: str, bench: str, tag: str,
                          phase: str) -> list[int]:
    """Return sorted list of round numbers that have the given phase."""
    pattern = os.path.join(backups_root(project_dir, bench, tag), "final",
                           "round_*", phase)
    out = []
    for path in glob.glob(pattern):
        if not os.path.isdir(path):
            continue
        parent = os.path.basename(os.path.dirname(path))
        m = _ROUND_RE.match(parent)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)
