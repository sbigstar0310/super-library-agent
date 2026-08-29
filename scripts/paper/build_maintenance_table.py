#!/usr/bin/env python3
"""Regenerate the numbers in `tables/maintenance-posthoc.tex` from `results/`.

    # emit the LaTeX body rows
    python3 scripts/paper/build_maintenance_table.py

    # ACCEPTANCE TEST: compare every cell against the committed .tex
    python3 scripts/paper/build_maintenance_table.py --diff

Each row is the mean over nine runs (three WebGen suites x three trials).

    Total / App / Lib   added_loc_by_location in maintenance_metrics.json,
                        with App = apps + other. Printed as integers.
    Orig.               summary.overall_accuracy of ui_test_results_regression.json,
                        the post-patch pass rate on behavior that already existed.
    Req.                the same field of ui_test_results_change.json, the pass
                        rate on the behavior the policy update asked for.
    Appr.               average of appearance_grade.json.

`--diff` parses the numeric cells out of the committed table and reports each
as MATCH / MISMATCH / NO-DATA. Exit status is 0 only when every cell matches.
"""
import argparse
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rows import MAINT_ROWS, MAINT_ROW_ORDER  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TEX = "paper/Super-library-agent-paper/tables/maintenance-posthoc.tex"

# column key -> (source file, how to pull the number, decimal places)
COLUMNS = ["Total", "App", "Lib", "Orig", "Req", "Appr"]
PLACES = {"Total": 0, "App": 0, "Lib": 0, "Orig": 1, "Req": 1, "Appr": 2}


def phase_dir(row, tag):
    return os.path.join(REPO, "results", row.bench, tag,
                        "eval_results", f"round_{row.round}", row.phase)


def read(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def run_values(row, tag):
    """The six cells for one run, or None for any the files do not cover."""
    d = phase_dir(row, tag)
    out = dict.fromkeys(COLUMNS)

    m = read(os.path.join(d, "maintenance_metrics.json"))
    if m is not None:
        loc = m["added_loc_by_location"]
        out["Lib"] = loc["lib"]
        out["App"] = loc["apps"] + loc["other"]
        out["Total"] = loc["lib"] + loc["apps"] + loc["other"]

    for col, fname in (("Orig", "ui_test_results_regression.json"),
                       ("Req", "ui_test_results_change.json")):
        u = read(os.path.join(d, fname))
        if u is not None:
            out[col] = u["summary"]["overall_accuracy"]

    a = read(os.path.join(d, "appearance_grade.json"))
    if a is not None:
        out["Appr"] = a["average"]
    return out


def aggregate():
    """{row label: {column: mean or None}}, plus the per-run counts."""
    table, counts = {}, {}
    for row in MAINT_ROWS:
        acc = {c: [] for c in COLUMNS}
        missing = []
        for tag, _src in row.tags():
            vals = run_values(row, tag)
            if all(v is None for v in vals.values()):
                missing.append(tag)
                continue
            for c in COLUMNS:
                if vals[c] is not None:
                    acc[c].append(vals[c])
        table[row.label] = {c: (statistics.mean(v) if v else None)
                            for c, v in acc.items()}
        counts[row.label] = {"runs": len(row.tags()), "missing": missing,
                             "per_column": {c: len(v) for c, v in acc.items()}}
    return table, counts


def fmt(col, value):
    return "--" if value is None else f"{value:.{PLACES[col]}f}"


def emit(table):
    for label in MAINT_ROW_ORDER:
        cells = table[label]
        body = " & ".join(fmt(c, cells[c]) for c in COLUMNS)
        print(f"{label:16s} & {body} \\\\")


CELL = re.compile(r"\\textbf\{([^}]*)\}|(-?\d+(?:\.\d+)?)")


def parse_tex(path):
    """{row label: [six numbers]} from the committed table body."""
    with open(path) as fh:
        text = fh.read()
    text = text.replace("\\textsc{", "").replace("\\textbf{", "")
    found = {}
    for label in MAINT_ROW_ORDER:
        # the row starts at its label and runs to the next \\
        i = text.find(label)
        if i < 0:
            continue
        j = text.find("\\\\", i)
        nums = re.findall(r"-?\d+(?:\.\d+)?", text[i + len(label):j])
        if len(nums) >= 6:
            found[label] = [float(x) for x in nums[:6]]
    return found


def diff(table, counts, tex_path):
    want = parse_tex(tex_path)
    ok = bad = nodata = 0
    for label in MAINT_ROW_ORDER:
        if label not in want:
            print(f"{label:16s}  ROW NOT FOUND IN TEX")
            bad += 6
            continue
        for col, target in zip(COLUMNS, want[label]):
            got = table[label][col]
            if got is None:
                print(f"{label:16s} {col:6s} NO-DATA   paper={target}")
                nodata += 1
                continue
            shown = round(got, PLACES[col])
            if abs(shown - target) < 10 ** -PLACES[col] / 2:
                ok += 1
            else:
                bad += 1
                print(f"{label:16s} {col:6s} MISMATCH  "
                      f"recomputed={fmt(col, got)}  paper={target}")
    for label, c in counts.items():
        if c["missing"]:
            print(f"{label:16s} missing runs: {', '.join(c['missing'])}")
    total = ok + bad + nodata
    print(f"\n{ok}/{total} cells match, {bad} mismatch, {nodata} without data")
    return 0 if bad == 0 and nodata == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff", action="store_true",
                    help="compare against the committed .tex instead of emitting it")
    ap.add_argument("--tex", default=DEFAULT_TEX,
                    help=f"table to compare against (default: {DEFAULT_TEX})")
    args = ap.parse_args()

    table, counts = aggregate()
    if not args.diff:
        emit(table)
        return 0
    path = args.tex if os.path.isabs(args.tex) else os.path.join(REPO, args.tex)
    if not os.path.exists(path):
        print(f"no such table: {path}", file=sys.stderr)
        return 2
    return diff(table, counts, path)


if __name__ == "__main__":
    sys.exit(main())
