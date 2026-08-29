#!/usr/bin/env python3
"""Repopulate `results/` from `backups/`.

Copies only the small metric JSONs the paper's tables consume — never `logs/**`
(training transcripts contain a plaintext API key) and never submissions.
`rows.py` is the source of truth for which tags feed which row.

    python3 scripts/paper/build_results_tree.py

Requires the full `backups/` tree, so it only runs on the machine that holds
the campaign snapshots; `results/` is the tracked output.
"""
import json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from rows import MAINT_ROWS, ROWS  # noqa: E402

FILES = ["loc_summary.json", "mdl_summary.json", "mdl_summary_shared.json",
         "scb_quality_summary.json", "lib_usage_results.json",
         "ui_test_results.json", "appearance_grade.json", "summary.json"]
# PaperBench per-paper artifacts (phase/<paper_id>/<file>)
SUB_FILES = ["usage_summary.json"]


def slim_graded_tree(src, dst):
    """graded_tree.json is 0.3-4 MB of per-leaf judge explanations; the table
    needs only the root score. Write a root-only digest under a *different*
    name so nothing here masquerades as the full tree."""
    with open(src) as fh:
        d = json.load(fh)

    def count(node):
        subs = node.get("sub_tasks") or []
        n = 1 + sum(count(s) for s in subs)
        return n

    def leaves(node):
        subs = node.get("sub_tasks") or []
        if not subs:
            return 1, (0 if node.get("valid_score", True) else 1)
        tot = inv = 0
        for s in subs:
            a, b = leaves(s)
            tot += a
            inv += b
        return tot, inv

    n_leaves, n_invalid = leaves(d)
    out = {"id": d.get("id"), "score": d.get("score"),
           "valid_score": d.get("valid_score"), "weight": d.get("weight"),
           "task_category": d.get("task_category"),
           "n_nodes": count(d), "n_leaves": n_leaves,
           "n_invalid_leaves": n_invalid,
           "_note": "root-only digest of graded_tree.json; full rubric tree "
                    "(with per-leaf judge explanations) stays in backups/"}
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)

CAL = json.load(open(f"{ROOT}/analysis/outputs/librarian_baseline/"
                    "librarian_mdl_calibrated.json"))
CAL_TAGS = {}
for sec, bench in (("webgen_main", "webgen"), ("paperbench", "paperbench")):
    for tag, payload in CAL[sec].items():
        CAL_TAGS[(bench, tag)] = payload

copied = skipped = 0

for row in ROWS:
    src_bench = row.source
    for tag, backup_tag, _c, _t in row.backup_tags():
        src = (f"{ROOT}/backups/{src_bench}/{backup_tag}"
               f"/eval_results/round_{row.round}/{row.phase}")
        dst = f"{ROOT}/results/{row.bench}/{tag}/eval_results/round_{row.round}/{row.phase}"
        if not os.path.isdir(src):
            print(f"[MISSING] {src}")
            skipped += 1
            continue
        os.makedirs(dst, exist_ok=True)
        for f in FILES:
            if os.path.exists(f"{src}/{f}"):
                shutil.copy2(f"{src}/{f}", f"{dst}/{f}")
                copied += 1
        # PaperBench rubric grades sit one level deeper, per paper id
        for entry in sorted(os.listdir(src)):
            sub = f"{src}/{entry}"
            if not os.path.isdir(sub) or entry in ("tasks", "quality_analysis",
                                                   "ui_test_results"):
                continue
            for f in SUB_FILES:
                if os.path.exists(f"{sub}/{f}"):
                    os.makedirs(f"{dst}/{entry}", exist_ok=True)
                    shutil.copy2(f"{sub}/{f}", f"{dst}/{entry}/{f}")
                    copied += 1
            if os.path.exists(f"{sub}/graded_tree.json"):
                os.makedirs(f"{dst}/{entry}", exist_ok=True)
                slim_graded_tree(f"{sub}/graded_tree.json",
                                 f"{dst}/{entry}/graded_tree_score.json")
                copied += 1
        cal = CAL_TAGS.get((row.bench, backup_tag))
        if cal is not None:
            with open(f"{dst}/mdl_summary_calibrated.json", "w") as fh:
                json.dump(cal, fh, indent=1)
            copied += 1
# Post-construction maintenance (paper Table 2 and Appendix B.2). Separate
# campaign, separate backup tree, so it gets its own pass. `mdl_summary.json`
# and the two code-metric summaries are here because Appendix B.2 reports the
# post-patch minus pre-patch deltas, not just the patch size.
MAINT_FILES = ["maintenance_metrics.json",
               "ui_test_results_regression.json", "ui_test_results_change.json",
               "appearance_grade.json", "loc_summary.json",
               "scb_quality_summary.json", "mdl_summary.json"]

for row in MAINT_ROWS:
    for tag, backup_tag, _c, _t in row.backup_tags():
        src = (f"{ROOT}/backups/{row.source}/{backup_tag}"
               f"/eval_results/round_{row.round}/{row.phase}")
        dst = (f"{ROOT}/results/{row.bench}/{tag}"
               f"/eval_results/round_{row.round}/{row.phase}")
        if not os.path.isdir(src):
            print(f"[MISSING] {src}")
            skipped += 1
            continue
        os.makedirs(dst, exist_ok=True)
        for f in MAINT_FILES:
            if os.path.exists(f"{src}/{f}"):
                shutil.copy2(f"{src}/{f}", f"{dst}/{f}")
                copied += 1

print(f"copied {copied} files, {skipped} missing phase dirs")
