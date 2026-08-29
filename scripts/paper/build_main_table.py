#!/usr/bin/env python3
"""Regenerate the numbers in `tables/main-combined.tex` from `results/`.

    # emit the LaTeX body rows
    python3 scripts/paper/build_main_table.py

    # ACCEPTANCE TEST: compare every cell against the committed .tex
    python3 scripts/paper/build_main_table.py --diff

`--diff` parses the numeric cells out of the committed table and reports each
cell as MATCH / MISMATCH / NO-DATA. Exit status is 0 only when every cell with
data matches.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import METRIC_ND, PAPER_KEYS, aggregate, fmt  # noqa: E402
from rows import ROWS, ROW_ORDER, UNCHECKED_ROWS  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TEX = "paper/Super-library-agent-paper/tables/main-combined.tex"

BENCHES = [("webgen", "WebGen-Bench"), ("paperbench", "PaperBench")]
# Row label -> the LaTeX line that introduces it in main-combined.tex
TEX_LABEL = {
    "Zero-Shot": "Zero-Shot",
    "Librarian K=1": r"\textsc{Librarian} ($K{=}1$)",
    "Librarian K=8": r"\textsc{Librarian} ($K{=}8$)",
    "Naive-Implicit": r"\textsc{Naive-Implicit}",
    "Naive-Ward": r"\textsc{Naive-Ward}",
    "SLA-Full": r"\textsc{SLA-Full}",
}
# Acc is printed with 2 decimals on WebGen (a percentage) and 4 on PaperBench
# (a 0-1 rubric score).
ND_OVERRIDE = {("paperbench", "Acc"): 4}


def nd_for(bench, key):
    return ND_OVERRIDE.get((bench, key), METRIC_ND[key])


def computed_table(base_dir, use_calibrated=True):
    """{(bench, row_label): {metric: formatted_string or None}}"""
    out, notes = {}, []
    for row in ROWS:
        if not row.clusters:
            out[(row.bench, row.label)] = {k: None for k in PAPER_KEYS}
            notes.append(f"{row.bench}/{row.label}: no tags mapped")
            continue
        res, missing = aggregate(base_dir, row, use_calibrated)
        if missing:
            notes.append(f"{row.bench}/{row.label}: missing tags {missing}")
        cells = {}
        for k in PAPER_KEYS:
            if k in res:
                cells[k] = fmt(res[k][0], nd_for(row.bench, k))
            else:
                cells[k] = None
        cells["_n"] = res.get("LOC", (0, 0, 0))[2]
        out[(row.bench, row.label)] = cells
        # Zero-Shot has no library; the paper prints "--" there.
        if row.label == "Zero-Shot":
            cells["libLOC"] = cells["libMDL"] = None
    return out, notes


def emit_rows(table):
    """Render the numeric body rows the way main-combined.tex writes them."""
    lines = []
    for bench, title in BENCHES:
        lines.append(rf"\multicolumn{{10}}{{l}}{{\textbf{{{title}}}}} \\")
        for label in ROW_ORDER:
            c = table.get((bench, label))
            if c is None:
                continue
            g = lambda k: c[k] if c.get(k) is not None else "--"  # noqa: E731
            lines.append(TEX_LABEL[label])
            lines.append(f"& {g('Acc')} & {g('Appear')}")
            lines.append(f"& {g('LOC')} & {g('Tok')} & {g('MDL')} "
                         f"& {g('Eros')} & {g('Verb')}")
            lines.append(rf"& {g('libLOC')} & {g('libMDL')}\\")
        lines.append(r"\midrule")
    return "\n".join(lines)


NUM = r"(?:\\textbf\{)?(-?[\d.]+)\}?"


def parse_tex(path):
    """Pull the numeric cells out of the committed main-combined.tex."""
    with open(path) as fh:
        text = fh.read()
    # Strip formatting that wraps individual numbers.
    flat = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    flat = flat.replace(r"\rowcolor{gray!12}", "")
    bench = None
    parsed = {}
    lines = [ln.strip() for ln in flat.splitlines()]
    i = 0
    label_by_tex = {v.replace(r"\textbf{", "").replace("}", "}"): k
                    for k, v in TEX_LABEL.items()}
    while i < len(lines):
        ln = lines[i]
        for b, title in BENCHES:
            if rf"\textbf{{{title}}}" in ln or f"{{{title}}}" in ln:
                bench = b
        hit = None
        for label, tex in TEX_LABEL.items():
            if ln == tex or ln == re.sub(r"\\textbf\{([^}]*)\}", r"\1", tex):
                hit = label
        if hit and bench:
            body = " ".join(lines[i + 1:i + 4])
            nums = re.findall(r"(-?\d+\.?\d*|--)", body.replace("$", "")
                              .replace("{=}", "").replace(r"\\", " "))
            # 9 cells: Acc Appear LOC Tok MDL Eros Verb libLOC libMDL
            nums = [n for n in nums if n != ""]
            if len(nums) >= 9:
                parsed[(bench, hit)] = dict(zip(PAPER_KEYS, nums[:9]))
            i += 3
        i += 1
    return parsed


def per_suite(base_dir, use_calibrated=True):
    """Per-suite means + the pooled Avg., matching the per-suite tables."""
    import copy
    for bench, title in BENCHES:
        print(f"\n[{title}]  (S1..Sn in cluster order, then pooled Avg.)")
        for row in ROWS:
            if row.bench != bench or not row.clusters:
                continue
            print(f"  {row.label}")
            for k in PAPER_KEYS:
                cells = []
                for i, c in enumerate(row.clusters, start=1):
                    sub = copy.copy(row)
                    sub.clusters = [c]
                    res, _ = aggregate(base_dir, sub, use_calibrated)
                    cells.append(fmt(res[k][0], nd_for(bench, k))
                                 if k in res else "--")
                res, _ = aggregate(base_dir, row, use_calibrated)
                avg = fmt(res[k][0], nd_for(bench, k)) if k in res else "--"
                print(f"    {k:<7} " + "  ".join(f"S{i}={v}" for i, v
                                                 in enumerate(cells, 1))
                      + f"   Avg={avg}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="results",
                    help="root holding <bench>/<tag>/eval_results/… "
                         "(default: results)")
    ap.add_argument("--tex", default=DEFAULT_TEX,
                    help=f"table to diff against (default: {DEFAULT_TEX})")
    ap.add_argument("--diff", action="store_true",
                    help="compare every cell against --tex and exit non-zero "
                         "on any mismatch")
    ap.add_argument("--no-calibrated", action="store_true",
                    help="ignore mdl_summary_calibrated.json and use the raw "
                         "pipeline MDL for the Librarian rows")
    ap.add_argument("--per-suite", action="store_true",
                    help="print per-suite means for diffing against "
                         "tables/per-suite-{webgenbench,paperbench}__results.tex")
    args = ap.parse_args()

    base = args.base_dir if os.path.isabs(args.base_dir) \
        else os.path.join(REPO, args.base_dir)
    tex = args.tex if os.path.isabs(args.tex) else os.path.join(REPO, args.tex)

    if args.per_suite:
        return per_suite(base, use_calibrated=not args.no_calibrated)

    table, notes = computed_table(base, use_calibrated=not args.no_calibrated)

    if not args.diff:
        print(emit_rows(table))
        for n in notes:
            print(f"% NOTE {n}")
        return 0

    published = parse_tex(tex)
    n_match = n_round = n_mis = n_nodata = 0
    mismatches = []
    print(f"# diff: computed({args.base_dir}) vs {os.path.relpath(tex, REPO)}\n")
    hdr = f"{'row':<28}{'metric':<8}{'computed':>12}{'published':>12}   status"
    print(hdr)
    print("-" * len(hdr))
    for bench, title in BENCHES:
        print(f"[{title}]")
        for label in ROW_ORDER:
            if label in UNCHECKED_ROWS:
                continue
            comp = table.get((bench, label), {})
            pub = published.get((bench, label), {})
            for k in PAPER_KEYS:
                c, p = comp.get(k), pub.get(k)
                if p in (None, "--") and c is None:
                    continue
                if c is None:
                    status, n_nodata = "NO-DATA", n_nodata + 1
                elif p is None:
                    status, n_nodata = "NOT-IN-TEX", n_nodata + 1
                elif _f(c) is not None and _f(c) == _f(p):
                    n_match += 1
                    continue  # exact: nothing to show
                elif within_ulp(c, p):
                    status, n_round = "MATCH (rounding)", n_round + 1
                else:
                    status, n_mis = "MISMATCH", n_mis + 1
                    mismatches.append((bench, label, k, c, p))
                print(f"{label:<28}{k:<8}{str(c):>12}{str(p):>12}   {status}")
    print("\n" + "=" * len(hdr))
    print(f"{n_match} exact, {n_round} within rounding, "
          f"{n_mis} mismatch, {n_nodata} no data")
    if UNCHECKED_ROWS:
        print("not checked: " + ", ".join(sorted(UNCHECKED_ROWS)))
    if n_round:
        print("`MATCH (rounding)` = within +/-1 unit in the published cell's "
              "last displayed place; see results/README.md for why.")
    for n in notes:
        print(f"NOTE {n}")
    return 1 if n_mis else 0


def ulp(printed: str) -> float:
    """One unit in the last displayed place of `printed`, from its own
    formatting: '9393' -> 1, '76.95' -> 0.01, '0.0987' -> 0.0001."""
    s = str(printed).strip().lstrip("+-")
    decimals = len(s.split(".", 1)[1]) if "." in s else 0
    return 10.0 ** (-decimals)


def within_ulp(computed, printed) -> bool:
    """True when the two agree to within one unit in the last place the paper
    actually prints. The tolerance is derived from the published cell, never
    hardcoded per metric, so a value shown to 4 dp gets a 4 dp tolerance."""
    a, b = _f(computed), _f(printed)
    if a is None or b is None:
        return False
    # 1e-9 absorbs binary-float representation error in the comparison itself
    return abs(a - b) <= ulp(printed) + 1e-9


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
