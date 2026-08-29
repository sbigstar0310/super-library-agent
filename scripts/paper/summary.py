#!/usr/bin/env python3
"""Print mean ± SD across trials for one method/suite combination.

Tracked copy of `scripts/local/summary.py` (which is gitignored), rebased on
the shared reader in `aggregate.py` and defaulted to the tracked `results/`
tree instead of a machine-local backups path.

    # one suite
    python3 scripts/paper/summary.py --task webgen \
        --backup-tag-base sla-ours-c2 --round 4 --phase apply

    # pooled across suites ({c} placeholder)
    python3 scripts/paper/summary.py --task webgen \
        --backup-tag-base 'sla-ours-c{c}' --clusters 2,5,13 \
        --round 4 --phase apply

    # against the full backups tree instead of results/
    python3 scripts/paper/summary.py --base-dir backups --bench webgen-rb \
        --task webgen --backup-tag-base 'librarian-c{c}' --clusters 2,5,13 \
        --round 1 --phase apply
"""
import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import read_cell  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def stats(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    p = argparse.ArgumentParser(
        description="(AVG ± SD) across trials, pooled over suites.")
    p.add_argument("--backup-tag-base", required=True,
                   help="tag without the -t<i> suffix; may contain {c}")
    p.add_argument("--round", required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--task", default="webgen",
                   choices=["webgen", "paperbench"],
                   help="which functionality metric to report")
    p.add_argument("--bench", default=None,
                   help="directory under --base-dir (default: same as --task). "
                        "Use e.g. webgen-rb when reading the backups tree.")
    p.add_argument("--t", type=int, default=3, help="number of trials")
    p.add_argument("--base-dir", default="results",
                   help="root holding <bench>/<tag>/eval_results/… "
                        "(default: results)")
    p.add_argument("--suffix", default="",
                   help="tag suffix after -t<i>, e.g. -mm3")
    p.add_argument("--clusters", default="",
                   help="comma-separated suite ids to pool; requires {c} in "
                        "--backup-tag-base")
    p.add_argument("--no-calibrated", action="store_true",
                   help="ignore mdl_summary_calibrated.json")
    args = p.parse_args()

    bench = args.bench or args.task
    base = args.base_dir if os.path.isabs(args.base_dir) \
        else os.path.join(REPO, args.base_dir)

    clusters = [c.strip() for c in args.clusters.split(",")] if args.clusters \
        else [None]
    tags = []
    for cl in clusters:
        stem = args.backup_tag_base.format(c=cl) if cl is not None \
            else args.backup_tag_base
        tags += [f"{stem}-t{i}{args.suffix}" for i in range(1, args.t + 1)]

    print(f"\n=== {bench} · round {args.round} · phase {args.phase} ===")
    print(f"tags: {tags}\n")

    cols = {}
    for tag in tags:
        cell = read_cell(base, bench, tag, args.round, args.phase,
                         use_calibrated=not args.no_calibrated)
        if not cell:
            print(f" - {tag}: NO DATA at "
                  f"{os.path.join(args.base_dir, bench, tag)}")
            continue
        for k, v in cell.items():
            if not k.startswith("_"):
                cols.setdefault(k, []).append(v)
        cal = " (calibrated MDL)" if cell.get("_calibrated") else ""
        print(f" - {tag}: ok{cal}")

    def show(label, key, nd=4):
        vals = cols.get(key)
        if not vals:
            print(f"{label:<25}: no data")
            return
        avg, sd = stats(vals)
        print(f"{label:<25}: {avg:.{nd}f} ± {sd:.{nd}f}  (N={len(vals)})")

    print(f"\n=== pooled over {len(tags)} runs ===")
    if cols.get("scbLOC") and cols.get("LOC"):
        a, b = statistics.mean(cols["scbLOC"]), statistics.mean(cols["LOC"])
        if not math.isclose(a, b, rel_tol=0.01):
            print(f"[warn] LOC disagreement: scb_quality.total_loc={a:.0f} "
                  f"vs loc_summary.loc_total={b:.0f} (diff={a - b:+.0f})")
    print("-" * 65)
    if args.task == "webgen":
        show("UI Test Accuracy (%)", "Acc", 2)
        show("Appearance Grade", "Appear")
    else:
        show("Paperbench Grade", "Acc")
    show("Verbosity", "Verb")
    show("Erosion", "Eros")
    show("LOC (total_loc)", "LOC", 0)
    show("TOK (total_tok)", "Tok", 0)
    show("MDL (total_mdl)", "MDL", 1)
    print("-" * 65)
    show("LOC (app_loc)", "appLOC", 0)
    show("LOC (lib_loc)", "libLOC", 0)
    show("TOK (app_tok)", "appTok", 0)
    show("TOK (lib_tok)", "libTok", 0)
    show("MDL (app_mdl)", "appMDL", 1)
    show("MDL (lib_mdl)", "libMDL", 1)
    print("-" * 65)
    show("Lib Exports", "lib_exports", 1)
    show("Lib Dead Symbols", "lib_dead_symbols", 1)
    show("Lib Dead %", "lib_dead_pct", 2)
    show("Lib Avg/Sym (all)", "lib_avg_per_sym_all", 2)
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
