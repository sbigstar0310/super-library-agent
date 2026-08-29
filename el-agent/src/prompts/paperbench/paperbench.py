"""PaperBench bench-level rules (paper-replication submission + library scaffold).

Two constants:
- APP_RULES : what a paper-replication submission must look like (stack,
              compute, required files). Inject in agents that EDIT
              submission code (coding / apply / local_extract).
              Scope / partial-credit / appendix-exclusion / paper-codebase
              guidance is NOT duplicated here — the paperbench upstream
              coding user prompt covers it verbatim
              (see `prompts/paperbench/coding_agent.py`).
- LIB_RULES : what the shared Python `lib/` package must look like
              (directory layout, public-API discipline, import path).
              Inject in agents that EDIT lib code
              (global_extract / library).

Mirrors `prompts/webgen/webgen.py`.
"""

from __future__ import annotations


__all__ = ["APP_RULES", "LIB_RULES"]


APP_RULES = """\
[Stack]
- Python 3.11+ project, importable as a normal package tree (no monorepo,
  no submodule of an external framework). Use plain `pip`/`uv`-installable
  third-party libraries; do NOT vendor large dependencies.
- The grader reads the source code — it does NOT execute it. Do not spend
  time on full training runs to "validate" correctness.

[Compute]
- No GPU. Do not assume CUDA. Where the paper specifies GPU training,
  implement the same code path with smaller scales / fewer epochs so the
  shape is preserved for code-only grading.

[Required files at submission root]
  README.md            describes what was reproduced and how the layout
                       maps to the paper's sections / contributions.
  <code tree>          arranged however the implementation needs —
                       prefer a clean package layout over loose scripts.
"""


LIB_RULES = """\
[Directory layout] — plain Python package at `<library_dir>`:
  lib/__init__.py                  package marker (may re-export top
                                   symbols for ergonomic imports)
  lib/<subpkg>/__init__.py         subpackage by domain (e.g. `nn/`,
                                   `buffers/`, `algo_utils/`, `eval/`,
                                   `utils/`)
  lib/<subpkg>/<module>.py         actual helpers

NO `setup.py` / `pyproject.toml` / `setup.cfg`. The library is NOT a
pip-installable package — submissions import it via PYTHONPATH
(injected by the runner). Do not declare it as a dependency in any
paper's requirements file.

[Module conventions]
- Public functions/classes use clear names; private helpers are
  underscore-prefixed and not re-exported.
- Type hints on every public signature. Short docstring per public
  symbol (1-3 lines).

[Public-API discipline]
- Subpackage `__init__.py` files MAY re-export the subpackage's stable
  public symbols. Do not re-export private helpers.
- Importers reach symbols as `from lib.<subpkg> import <Symbol>` (or
  `from lib.<subpkg>.<module> import <Symbol>` for less-common ones).

[Reuse discipline]
Do not duplicate symbols already present in the seeded library. Before
adding a new symbol, scan existing subpackages for an equivalent — extend
or parameterize the existing one if possible.
"""
