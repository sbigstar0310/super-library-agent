"""
Structural Erosion metric from SlopCodeBench (arXiv:2603.24755).

    erosion = Σ(mass(f) | CC(f) > threshold) / Σ(mass(f))
    mass(f) = CC(f) × √SLOC(f)

**Scope: JavaScript family only (`.js`/`.jsx`/`.ts`/`.tsx`).** Python
codebases (paperbench in scb_quality.py) measure CC through the
SCB ``slop_code.metrics`` Python pipeline — this module is not used for
them. Per ``scripts/metrics/scb_quality.py`` ``JS_TASKS = {"webgen"}``.

Per-function CC comes from ESLint's `complexity` rule (max=0, every
function reported); name and SLOC span come from @babel/parser. The
hybrid lives in ``measure_eslint.js`` next to this module, with the
ESLint flat config at ``eslint.config.mjs``.

Public API: ``compute_app_erosion(app_dir, lib_dir=None)`` — measure one
program. Pass a pre-built corpus root as ``app_dir`` with ``lib_dir=None``
to score (N apps + lib) as one codebase (scripts/metrics/scb_quality.py does this
for webgen).
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_METRICS_SCRIPT = Path(__file__).resolve().parent / "measure_eslint.js"


@dataclass
class FunctionMetric:
    file: str
    name: str
    line: int
    end_line: int
    sloc: int
    cc: int


def _run_node(target_dir: Path) -> list[FunctionMetric]:
    """Invoke measure_eslint.js on a directory, return per-function metrics."""
    if not _METRICS_SCRIPT.exists():
        raise FileNotFoundError(f"metrics script not found: {_METRICS_SCRIPT}")
    target_dir = Path(target_dir)
    if not target_dir.exists():
        return []

    result = subprocess.run(
        ["node", str(_METRICS_SCRIPT), str(target_dir)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"measure_eslint.js failed (exit {result.returncode}):\n{result.stderr}"
        )
    data = json.loads(result.stdout) if result.stdout.strip() else []
    return [
        FunctionMetric(
            file=d["file"],
            name=d["name"],
            line=d["line"],
            end_line=d["endLine"],
            sloc=d["sloc"],
            cc=d["cc"],
        )
        for d in data
    ]


def _mass_stats(
    functions: list[FunctionMetric],
    cc_threshold: int,
    include_details: bool = True,
) -> dict:
    """Aggregate mass = CC × √SLOC, return erosion + breakdown.

    When ``include_details=True`` the result also carries
    ``high_complexity_functions_detail`` — a list of dicts describing each
    function whose CC exceeds ``cc_threshold``, sorted by mass descending.
    This is useful for downstream feedback generation (see ``maintain_feedback``).
    """
    total_mass = 0.0
    high_mass = 0.0
    high_count = 0
    high_details: list[dict] = []
    for fn in functions:
        m = fn.cc * math.sqrt(fn.sloc)
        total_mass += m
        if fn.cc > cc_threshold:
            high_mass += m
            high_count += 1
            if include_details:
                high_details.append(
                    {
                        "file": fn.file,
                        "name": fn.name,
                        "start_line": fn.line,
                        "end_line": fn.end_line,
                        "cc": fn.cc,
                        "sloc": fn.sloc,
                        "mass": m,
                    }
                )
    result = {
        "erosion": high_mass / total_mass if total_mass > 0 else 0.0,
        "total_functions": len(functions),
        "high_complexity_functions": high_count,
        "total_mass": total_mass,
        "high_mass": high_mass,
        "cc_threshold": cc_threshold,
    }
    if include_details:
        high_details.sort(key=lambda d: d["mass"], reverse=True)
        result["high_complexity_functions_detail"] = high_details
    return result


def compute_app_erosion(
    app_dir: str | Path,
    lib_dir: str | Path | None = None,
    cc_threshold: int = 10,
    include_details: bool = True,
) -> dict:
    """Compute structural erosion for a single program (app, optionally + lib).

    "Single program" means (app + library): the library is included in scope
    whenever ``lib_dir`` is provided and exists. For experiments with no
    library, pass ``lib_dir=None`` (default) and scope collapses to the app
    alone — matching the pre-refactor single-target behavior used by
    ``maintain_feedback.py``.

    Args:
        app_dir: Path to the app directory.
        lib_dir: Optional library directory. If provided and exists, its
            functions are included in the measurement alongside the app's.
        cc_threshold: Functions with CC strictly greater than this contribute
            to the high-complexity numerator. Default 10 matches the paper.
        include_details: If True (default), the returned dict additionally
            carries ``high_complexity_functions_detail`` with per-function
            location, CC, SLOC, and mass (sorted by mass desc).

    Returns:
        dict with erosion, total_functions, high_complexity_functions,
        total_mass, high_mass, cc_threshold
        (+ high_complexity_functions_detail when include_details=True).
    """
    app_path = Path(app_dir)
    has_lib = lib_dir is not None and Path(lib_dir).exists()

    functions: list[FunctionMetric] = []
    for fn in _run_node(app_path):
        if has_lib:
            # Prefix to distinguish from lib files; single-target callers
            # keep unprefixed paths for clarity.
            fn.file = f"{app_path.name}/{fn.file}"
        functions.append(fn)

    if has_lib:
        lib_path = Path(lib_dir)
        for fn in _run_node(lib_path):
            fn.file = f"{lib_path.name}/{fn.file}"
            functions.append(fn)

    return _mass_stats(functions, cc_threshold, include_details=include_details)


def get_function_metrics(target_dir: str | Path) -> list[FunctionMetric]:
    """Return raw per-function metrics for a directory. Useful for debugging."""
    return _run_node(Path(target_dir))
