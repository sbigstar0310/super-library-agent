"""Librarian rerank — MDL-based winner selection over K sampled candidates.

Bench-agnostic. Given K candidate corpora (each = refactored apps + a shared
library) plus their correctness-gate verdicts, pick the winner the way the
Librarian paper does:

    1. Candidates whose gate-passes ⊇ ``gate_before`` (they preserved every app
       that built before refactoring) rank first, by total MDL ascending.
    2. Fallback (no candidate is a superset): rank by (#gate_before apps passing
       desc, total MDL asc) — the paper's lowest-loss fallback.

MDL of a candidate::

    total = library_nll + Σ_app  app_nll(app | library)

computed by :class:`utils.mdl.MDLMetric` against a local vLLM endpoint. A NaN
MDL (scoring failed) sinks the candidate to the bottom of its group.

The produced :class:`Report` is JSON-serializable and self-contained: a saved
report supports **post-hoc K'-subset re-selection with zero re-scoring** via
:func:`select_from_report` (the sample i.i.d. assumption — see the K-sweep note
in the paper appendix).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


__all__ = [
    "SamplePaths",
    "CandidateReport",
    "Report",
    "select_winner",
    "select_from_report",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SamplePaths:
    """Location of one sampled candidate corpus.

    ``order`` is the draw index (1-based sampling order); it defaults to ``k``.
    A ``lib_dir`` of ``None`` (or an empty dir) means the candidate produced no
    library — MDL then scores the bare apps and ``library_nll`` is 0.
    """

    k: int
    tasks_dir: str
    lib_dir: str | None
    order: int | None = None

    @property
    def draw_order(self) -> int:
        return self.order if self.order is not None else self.k


@dataclass
class CandidateReport:
    k: int
    order: int
    gate: dict[str, bool]            # per-app pass/fail (ALL gated apps)
    per_app_nll: dict[str, float]    # tid -> app_nll (given lib)
    lib_nll: float
    mdl_total: float                 # lib_nll + Σ per_app_nll (NaN if any NaN)
    pass_count: int                  # #gate_before apps passing in this cand
    superset: bool                   # gate-passes ⊇ gate_before
    rank: int = 0                    # 1-based, assigned during ranking


@dataclass
class Report:
    task: str
    winner_k: int | None
    subset: list[int]
    gate_before: list[str]
    candidates: list[CandidateReport] = field(default_factory=list)

    # ---- serialization -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "winner_k": self.winner_k,
            "subset": list(self.subset),
            "gate_before": list(self.gate_before),
            "candidates": [_candidate_to_dict(c) for c in self.candidates],
        }

    def save(self, path: str | os.PathLike) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict) -> "Report":
        return cls(
            task=data["task"],
            winner_k=data.get("winner_k"),
            subset=list(data.get("subset") or []),
            gate_before=list(data.get("gate_before") or []),
            candidates=[_candidate_from_dict(c) for c in data.get("candidates", [])],
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Report":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _candidate_to_dict(c: CandidateReport) -> dict:
    return {
        "k": c.k,
        "order": c.order,
        "gate": c.gate,
        "per_app_nll": {t: _num_or_null(v) for t, v in c.per_app_nll.items()},
        "lib_nll": _num_or_null(c.lib_nll),
        "mdl_total": _num_or_null(c.mdl_total),
        "pass_count": c.pass_count,
        "superset": c.superset,
        "rank": c.rank,
    }


def _candidate_from_dict(d: dict) -> CandidateReport:
    return CandidateReport(
        k=d["k"],
        order=d["order"],
        gate=dict(d.get("gate") or {}),
        per_app_nll={t: _null_to_nan(v) for t, v in (d.get("per_app_nll") or {}).items()},
        lib_nll=_null_to_nan(d.get("lib_nll")),
        mdl_total=_null_to_nan(d.get("mdl_total")),
        pass_count=int(d.get("pass_count", 0)),
        superset=bool(d.get("superset", False)),
        rank=int(d.get("rank", 0)),
    )


def _num_or_null(x: float) -> float | None:
    """NaN/inf → null so the JSON is valid for strict external tools."""
    return x if (x is not None and math.isfinite(x)) else None


def _null_to_nan(x) -> float:
    return float("nan") if x is None else float(x)


# ---------------------------------------------------------------------------
# Ranking (pure — shared by select_winner and select_from_report)
# ---------------------------------------------------------------------------

def _mdl_key(mdl: float) -> float:
    return mdl if (mdl is not None and math.isfinite(mdl)) else float("inf")


def _rank_candidates(
    candidates: list[CandidateReport],
    gate_before: set[str],
    subset: set[int] | None,
) -> tuple[list[CandidateReport], int | None]:
    """Order candidates and assign 1-based ranks. Returns (ordered, winner_k).

    Sort key ``(0 if superset else 1, -pass_count, mdl)`` covers both regimes:
    within the superset group ``pass_count`` is constant so it reduces to MDL
    asc; in the fallback group more gate_before passes win, ties broken by MDL.
    """
    active = [
        c for c in candidates
        if subset is None or c.k in subset
    ]
    for c in active:
        passes = {t for t, ok in c.gate.items() if ok}
        c.superset = gate_before <= passes
        c.pass_count = len(gate_before & passes)

    ordered = sorted(
        active,
        key=lambda c: (0 if c.superset else 1, -c.pass_count, _mdl_key(c.mdl_total)),
    )
    for i, c in enumerate(ordered, start=1):
        c.rank = i
    winner_k = ordered[0].k if ordered else None
    return ordered, winner_k


# ---------------------------------------------------------------------------
# Scoring + selection
# ---------------------------------------------------------------------------

def _discover_tids(tasks_dir: str) -> list[str]:
    if not os.path.isdir(tasks_dir):
        return []
    return [
        name for name in sorted(os.listdir(tasks_dir))
        if os.path.isdir(os.path.join(tasks_dir, name, "submission"))
    ]


def _lib_dir_if_present(lib_dir: str | None) -> str | None:
    if lib_dir and os.path.isdir(lib_dir) and os.listdir(lib_dir):
        return lib_dir
    return None


def _score_candidate(metric, sample: SamplePaths, task_cfg) -> tuple[dict[str, float], float]:
    """Return (per_app_nll, lib_nll). Library scored once, reused across apps."""
    tids = _discover_tids(sample.tasks_dir)
    lib_dir = _lib_dir_if_present(sample.lib_dir)
    per_app: dict[str, float] = {}
    lib_nll = 0.0
    lib_cache: tuple[float, int] | None = None

    for tid in tids:
        app_dir = os.path.join(sample.tasks_dir, tid, "submission")
        try:
            res = metric.score(
                app_dir, lib_dir, task=task_cfg, precomputed_lib=lib_cache,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[librarian_select] MDL score failed for sample "
                  f"{sample.k} / {tid}: {exc}")
            per_app[tid] = float("nan")
            continue
        if lib_cache is None and lib_dir is not None:
            lib_nll = res.library_nll
            lib_cache = (res.library_nll, res.library_tokens)
        per_app[tid] = res.app_nll
    return per_app, lib_nll


def select_winner(
    samples: dict[int, SamplePaths],
    gate_results: dict[int, dict[str, bool]],
    gate_before: set[str],
    *,
    task: str,
    base_url: str | None = None,
    subset: set[int] | None = None,
) -> Report:
    """Score every candidate's MDL and pick the winner.

    Args:
        samples: ``{k: SamplePaths}`` — one per sampled candidate.
        gate_results: ``{k: {tid: passed_bool}}`` — per-app gate verdicts.
        gate_before: task ids that built as bare ZS originals (the ⊆ baseline).
        task: ``"webgen"`` or ``"paperbench"`` (MDL task config).
        base_url: vLLM endpoint; ``None`` → MDLMetric default (127.0.0.1:8000).
        subset: restrict ranking to these k (MDL still scored for all present).
    """
    from utils.mdl import MDLMetric, load_task_config

    task_cfg = load_task_config(task)
    metric = MDLMetric(base_url=base_url) if base_url else MDLMetric()

    candidates: list[CandidateReport] = []
    for k in sorted(samples):
        sample = samples[k]
        gate = dict(gate_results.get(k, {}))
        per_app, lib_nll = _score_candidate(metric, sample, task_cfg)
        app_sum = sum(per_app.values()) if per_app else 0.0
        if any(math.isnan(v) for v in per_app.values()) or math.isnan(lib_nll):
            mdl_total = float("nan")
        else:
            mdl_total = lib_nll + app_sum
        candidates.append(CandidateReport(
            k=k,
            order=sample.draw_order,
            gate=gate,
            per_app_nll=per_app,
            lib_nll=lib_nll,
            mdl_total=mdl_total,
            pass_count=0,
            superset=False,
        ))

    _, winner_k = _rank_candidates(candidates, gate_before, subset)
    used_subset = sorted(subset) if subset is not None else sorted(samples)
    return Report(
        task=task,
        winner_k=winner_k,
        subset=used_subset,
        gate_before=sorted(gate_before),
        candidates=sorted(candidates, key=lambda c: c.rank),
    )


def select_from_report(
    report_path: str | os.PathLike,
    subset: set[int] | None = None,
) -> Report:
    """Re-select a winner from a saved report over a K'-subset — no re-scoring.

    Pure post-hoc computation: reuses each candidate's stored gate verdicts and
    MDL total. Returns a fresh :class:`Report` with re-assigned ranks, the new
    ``winner_k`` and the ``subset`` actually used.
    """
    base = Report.load(report_path)
    gate_before = set(base.gate_before)
    _, winner_k = _rank_candidates(base.candidates, gate_before, subset)
    used_subset = (
        sorted(subset) if subset is not None
        else sorted(c.k for c in base.candidates)
    )
    return Report(
        task=base.task,
        winner_k=winner_k,
        subset=used_subset,
        gate_before=base.gate_before,
        candidates=sorted(base.candidates, key=lambda c: c.rank),
    )
