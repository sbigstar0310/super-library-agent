"""Read the metric JSONs under `results/` and aggregate them per paper row.

Layout consumed (identical to `backups/<bench>/<tag>/…`, so this also works
with `--base-dir backups`):

    <base>/<bench>/<tag>/eval_results/round_<N>/<phase>/
        loc_summary.json  mdl_summary.json  mdl_summary_shared.json
        scb_quality_summary.json  lib_usage_results.json
        ui_test_results.json  appearance_grade.json           (WebGen)
        <paper_id>/graded_tree.json | graded_tree_score.json   (PaperBench)

`mdl_summary_calibrated.json`, when present, overrides `mdl_summary.json`.
Only the Librarian tags have one: their published MDL was re-measured on a
matched vLLM server after a cross-server drift was found. See results/README.md.
"""
import glob
import json
import os
import statistics

# Metric key -> (display label, decimals used in the paper table)
METRIC_ND = {
    "Acc": 2, "Appear": 2,
    "LOC": 0, "Tok": 0, "MDL": 0, "Eros": 4, "Verb": 4,
    "libLOC": 0, "libTok": 0, "libMDL": 0,
}
PAPER_KEYS = ["Acc", "Appear", "LOC", "Tok", "MDL", "Eros", "Verb",
              "libLOC", "libMDL"]


def _read(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def phase_dir(base_dir, bench, tag, rnd, phase):
    return os.path.join(base_dir, bench, tag, "eval_results",
                        f"round_{rnd}", phase)


def read_cell(base_dir, bench, tag, rnd, phase, use_calibrated=True):
    """All metrics for one (tag, round, phase). Missing metrics are absent."""
    d = phase_dir(base_dir, bench, tag, rnd, phase)
    out = {}

    loc = _read(f"{d}/loc_summary.json")
    if loc and "summary" in loc:
        out["LOC"] = loc["summary"]["loc_total"]
        out["appLOC"] = loc["summary"]["sum_app_loc"]
        out["libLOC"] = loc["summary"]["avg_lib_loc"]

    mdl = None
    if use_calibrated:
        mdl = _read(f"{d}/mdl_summary_calibrated.json")
        out["_calibrated"] = mdl is not None
    if mdl is None:
        mdl = _read(f"{d}/mdl_summary.json")
    if mdl and "summary" in mdl:
        out["MDL"] = mdl["summary"]["mdl_nll"]
        out["appMDL"] = mdl["summary"]["sum_app_nll"]
        out["libMDL"] = mdl["summary"]["avg_library_nll"]
        # The calibrated files carry `per_app`; the pipeline files carry `apps`.
        apps = mdl.get("apps")
        if apps is None and mdl.get("per_app"):
            apps = list(mdl["per_app"].values())
        if apps:
            app_tok = sum(a["app_tokens"] for a in apps)
            # library is flat-concatenated identically into every app
            lib_tok = apps[0]["library_tokens"]
            out["Tok"] = app_tok + lib_tok
            out["appTok"] = app_tok
            out["libTok"] = lib_tok

    scb = _read(f"{d}/scb_quality_summary.json")
    if scb and "summary" in scb:
        s = scb["summary"]
        out["Eros"] = s.get("erosion", s.get("erosion_mean"))
        out["Verb"] = s.get("verbosity", s.get("verbosity_mean"))
        out["scbLOC"] = s.get("total_loc")

    usage = _read(f"{d}/lib_usage_results.json")
    if usage and "result" in usage:
        for k in ("exports", "dead_symbols", "dead_pct", "avg_per_sym_all"):
            if k in usage["result"]:
                out["lib_" + k] = usage["result"][k]

    # WebGen functionality
    ui = _read(f"{d}/ui_test_results.json")
    if ui and "summary" in ui:
        out["Acc"] = ui["summary"]["overall_accuracy"]
    ap = _read(f"{d}/appearance_grade.json")
    if ap and "average" in ap:
        out["Appear"] = ap["average"]

    # PaperBench functionality: mean of the per-paper rubric root scores
    scores = []
    for sub in sorted(glob.glob(f"{d}/*/")):
        g = _read(f"{sub}graded_tree.json") or _read(f"{sub}graded_tree_score.json")
        if g and g.get("score") is not None:
            scores.append(g["score"])
    if scores:
        out["Acc"] = statistics.mean(scores)
        out["_n_papers"] = len(scores)

    return out


def aggregate(base_dir, row, use_calibrated=True):
    """Pool a Row's tags. Returns (per_metric{key: (mean, sd, n)}, missing_tags).

    Pooling is a flat mean over all tags. With an equal number of trials per
    suite this equals the mean of the per-suite means, which is what the
    paper's per-suite tables label `Avg.`.
    """
    cols, missing = {}, []
    for tag, _src in row.tags():
        cell = read_cell(base_dir, row.bench, tag, row.round, row.phase,
                         use_calibrated)
        if not any(k in cell for k in PAPER_KEYS):
            missing.append(tag)
            continue
        for k, v in cell.items():
            if not k.startswith("_"):
                cols.setdefault(k, []).append(v)
    res = {}
    for k, vals in cols.items():
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        res[k] = (statistics.mean(vals), sd, len(vals))
    return res, missing


def fmt(value, nd):
    """Format exactly as the paper table does (Python round-half-to-even)."""
    return f"{value:.{nd}f}"
