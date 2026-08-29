"""Correctness gates for the Librarian baseline.

Each gate takes one candidate corpus (apps + optional shared lib) and returns a
per-app pass/fail verdict, without mutating the input. WebGen builds each app
(`npm install && npx vite build`) in docker; PaperBench runs a static
compile + import-resolution check (added separately).

Public API::

    from utils.gates import run_webgen_build_gate, GateResult
"""

from __future__ import annotations

from utils.gates.webgen_build import GateResult, run_webgen_build_gate

__all__ = ["GateResult", "run_webgen_build_gate"]
