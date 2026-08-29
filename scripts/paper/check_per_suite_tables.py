#!/usr/bin/env python3
"""Diff results/ against the paper's two per-suite tables, section-aware.

    python3 scripts/paper/check_per_suite_tables.py

Companion to `build_main_table.py --diff`: that one checks the pooled main
table, this one checks every per-suite cell of
`tables/per-suite-{webgenbench,paperbench}__results.tex`.
"""
import re, sys, os, copy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from rows import ROWS, UNCHECKED_ROWS
from aggregate import aggregate, fmt
COLS = ["Zero-Shot", "Librarian K=8", "Naive-Implicit", "Naive-Ward", "SLA-Full"]
# (section title, latex metric label) -> (metric key, decimals)
KEYMAP = {
    ("Functionality", "Acc."): ("Acc", 2),
    ("Functionality", "Appear."): ("Appear", 2),
    ("Functionality", "Score"): ("Acc", 4),
    ("Maintainability", "LOC"): ("LOC", 0),
    ("Maintainability", "Tok."): ("Tok", 0),
    ("Maintainability", "MDL"): ("MDL", 0),
    ("Maintainability", "Eros."): ("Eros", 4),
    ("Maintainability", "Verb."): ("Verb", 4),
    ("Library Size", "LOC"): ("libLOC", 0),
    ("Library Size", "Tok."): ("libTok", 0),
    ("Library Size", "MDL"): ("libMDL", 0),
}
SPEC = [("webgen", "tables/per-suite-webgenbench__results.tex", 3),
        ("paperbench", "tables/per-suite-paperbench__results.tex", 5)]
num = lambda s: re.search(r"-?\d+\.?\d*", s)

tot = mis = rnd = 0
tally = {}
problems = []
for bench, rel, nsuite in SPEC:
    tally[bench] = [0, 0, 0]
    txt = open(f"{REPO}/paper/Super-library-agent-paper/{rel}").read()
    txt = re.sub(r"\\textbf\{([^}]*)\}", r"\1", txt)
    txt = re.sub(r"\{\\scriptsize\$\\pm\$[^}]*\}", "", txt)
    lines = [l.strip() for l in txt.splitlines()]
    rowsb = {r.label: r for r in ROWS
             if r.bench == bench and r.label not in UNCHECKED_ROWS}
    section = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"\\multicolumn\{7\}\{l\}\{(.+)\}\s*\\\\", ln)
        if m:
            section = m.group(1).strip()
        if ln.startswith("\\multirow"):
            label = ln.split("}")[-1].strip() or ln.rsplit("{", 1)[-1].rstrip("}")
            label = re.sub(r"\$\\(down|up)arrow\$", "", label).strip()
            spec = KEYMAP.get((section, label))
            if spec:
                key, nd = spec
                block = [l for l in lines[i:i + nsuite + 5] if l.startswith("&")][:nsuite + 1]
                for si, line in enumerate(block):
                    cells = [c.strip() for c in line.split("&")[2:]]
                    for ci, lab in enumerate(COLS):
                        if ci >= len(cells):
                            continue
                        mm = num(cells[ci])
                        if not mm:
                            continue
                        pub = float(mm.group())
                        r = copy.copy(rowsb[lab])
                        if si < nsuite:
                            r.clusters = [r.clusters[si]]
                        res, _ = aggregate(os.path.join(REPO, "results"), r)
                        if key not in res:
                            continue
                        comp = float(fmt(res[key][0], nd))
                        tot += 1
                        tally[bench][0] += 1
                        tag = f"S{si+1}" if si < nsuite else "Avg"
                        if comp == pub:
                            continue
                        # same +/-1-ulp band as build_main_table.py --diff,
                        # derived from the precision the paper prints
                        if abs(comp - pub) <= 10.0 ** (-nd) + 1e-9:
                            rnd += 1
                            tally[bench][2] += 1
                            problems.append(f"  {bench:11s} {lab:15s} {key:7s} {tag:4s} computed={comp} published={pub}   MATCH (rounding)")
                        else:
                            mis += 1
                            tally[bench][1] += 1
                            problems.append(f"  {bench:11s} {lab:15s} {key:7s} {tag:4s} computed={comp} published={pub}   MISMATCH")
            else:
                print(f"  [unmapped] {bench} section={section!r} label={label!r}")
        i += 1
print("\n".join(problems))
for b, (n, m, r) in tally.items():
    print(f"  {b:11s} {n:4d} cells checked, {n - m - r} exact, {r} within rounding, {m} mismatch")
print(f"\nper-suite tables: {tot} checked, {tot - mis - rnd} exact, "
      f"{rnd} within rounding, {mis} mismatch")
