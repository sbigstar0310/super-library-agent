"""1-hop file-level neighbors for apply-candidate markdown enrichment.

Builds forward (deps) and reverse (dependents) indices from one
``build_dep_graph`` call and renders an "Also review" block per app file cited
in the LLM-generated apply-candidate markdown. Scope: depth=1 only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from utils.mdl.configs import load_task_config
from utils.mdl.parsers._types import FileNode
from utils.mdl.parsers.js import build_dep_graph

_BARREL_INDEX_NAMES = ("index.js", "index.ts", "index.jsx", "index.tsx")

_BULLET_RE = re.compile(r"^\s*-\s+([^\s][^:]+?):\d+(?:-\d+)?")


@dataclass(frozen=True)
class NeighborSets:
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NeighborGraph:
    app_dir: str
    lib_dir: str | None
    forward: dict[str, list[str]]
    reverse: dict[str, list[str]]
    nodes: dict[str, FileNode]
    barrel_paths: frozenset[str]


def build_neighbors(
    app_dir: str,
    lib_dir: str | None,
    task_name: str = "webgen",
) -> NeighborGraph:
    """Wrap build_dep_graph and add a reverse index + barrel paths."""
    task = load_task_config(task_name)
    nodes = build_dep_graph(app_dir, lib_dir, task)

    forward: dict[str, list[str]] = {p: list(n.deps) for p, n in nodes.items()}
    reverse: dict[str, list[str]] = {}
    for path, deps in forward.items():
        for d in deps:
            reverse.setdefault(d, []).append(path)

    return NeighborGraph(
        app_dir=os.path.abspath(app_dir),
        lib_dir=os.path.abspath(lib_dir) if lib_dir else None,
        forward=forward,
        reverse=reverse,
        nodes=nodes,
        barrel_paths=_barrel_paths(lib_dir, task),
    )


def neighbors_of(
    abs_path: str,
    graph: NeighborGraph,
    *,
    depth: int = 1,
) -> NeighborSets:
    """1-hop parents (callers) + children (deps) of abs_path."""
    assert depth == 1, "depth > 1 is not implemented yet"
    abs_path = os.path.abspath(abs_path)

    parents: list[str] = []
    for p in graph.reverse.get(abs_path, []):
        if p in graph.barrel_paths:
            continue
        parents.append(os.path.relpath(p, graph.app_dir))

    children: list[str] = []
    for c in graph.forward.get(abs_path, []):
        children.append(os.path.relpath(c, graph.app_dir))

    return NeighborSets(
        parents=sorted(set(parents)),
        children=sorted(set(children)),
    )


def format_neighbors_block(
    rel_path: str,
    n: NeighborSets,
    *,
    max_per_side: int = 8,
) -> str:
    """Render markdown lines to inject under '**Replaces in this app**:'.

    Returns empty string when both parents and children are empty.
    """
    if not n.parents and not n.children:
        return ""

    lines = [f"**Also refactor** (neighbors of {rel_path}):"]
    if n.parents:
        lines.append(f"  ↑ Callers (import this file)  : {_truncate(n.parents, max_per_side)}")
    if n.children:
        lines.append(f"  ↓ Imports (used by this file) : {_truncate(n.children, max_per_side)}")
    return "\n".join(lines)


def inject_neighbors(md: str, graph: NeighborGraph, app_dir: str) -> str:
    """For each '### A<n>' block, append an 'Also review' block per unique app
    file cited under '**Replaces in this app**:'.
    """
    app_dir_abs = os.path.abspath(app_dir)
    in_section = "outside"  # "outside" | "header_done" | "in_replaces"
    cited: list[str] = []
    out: list[str] = []

    def emit() -> None:
        if not cited:
            return
        seen: set[str] = set()
        for rel in cited:
            if rel in seen:
                continue
            seen.add(rel)
            abs_p = os.path.abspath(os.path.join(app_dir_abs, rel))
            ns = neighbors_of(abs_p, graph)
            block = format_neighbors_block(rel, ns)
            if block:
                out.append(block)
        cited.clear()

    for line in md.split("\n"):
        if line.startswith("### A"):
            if in_section == "in_replaces":
                emit()
            in_section = "header_done"
            out.append(line)
            continue

        if line.lstrip().startswith("**Replaces in this app**"):
            in_section = "in_replaces"
            out.append(line)
            continue

        if in_section == "in_replaces":
            m = _BULLET_RE.match(line)
            if m:
                cited.append(m.group(1).strip())
                out.append(line)
                continue
            # left the bullet list — flush before this line
            emit()
            in_section = "header_done"

        out.append(line)

    if in_section == "in_replaces":
        emit()

    return "\n".join(out)


# ---- internals -------------------------------------------------------------

def _truncate(items: list[str], max_n: int) -> str:
    if len(items) <= max_n:
        return ", ".join(items)
    head = ", ".join(items[:max_n])
    more = len(items) - max_n
    return f"{head} ...(+{more} more)"


def _barrel_paths(lib_dir: str | None, task) -> frozenset[str]:
    if not lib_dir:
        return frozenset()
    subdir = ""
    if getattr(task, "lib_layout", None) is not None:
        subdir = getattr(task.lib_layout, "src_subdir", "") or ""
    base = os.path.join(lib_dir, subdir) if subdir else lib_dir
    return frozenset(
        os.path.abspath(os.path.join(base, name))
        for name in _BARREL_INDEX_NAMES
    )
