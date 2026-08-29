"""
Python parser for MDL (paperbench task).

Provides:
- ``strip_comments(code)``  → strip Python comments + module/function/class docstrings
- ``parse_imports(content)`` → ``import``/``from … import`` records (level for relative)
- ``build_dep_graph(app_dir, lib_dir, task)`` → 1-level abs_path → FileNode

Implementation: tree-sitter (`tree_sitter_python`).

Resolution rules:
- ``import a.b.c``                → ``a/b/c.py`` then ``a/b/c/__init__.py``
- ``from a.b import x``           → ``a/b.py``  then ``a/b/__init__.py`` (+ submodule
                                    ``a/b/x.py`` only when the package init is matched)
- ``from . import x`` (level=1)   → ``importer_dir/x.py`` or ``importer_dir/x/__init__.py``
- ``from ..pkg.m import x`` (=2)  → ``parent(importer_dir)/pkg/m.py`` (or pkg)
- External (resolution misses)    → skipped
- ``__init__.py`` re-exports are NEVER followed (per design decision).
"""

from __future__ import annotations

import glob
import os
import re

from ._types import FileNode, ImportStmt


# ------------------------------------------------------------------
# Tree-sitter parser singleton
# ------------------------------------------------------------------

_PY_PARSER = None


def _get_py_parser():
    """Lazy tree-sitter Python parser. Cached at module scope."""
    global _PY_PARSER
    if _PY_PARSER is None:
        from tree_sitter import Language, Parser
        import tree_sitter_python as ts_python
        _PY_PARSER = Parser(Language(ts_python.language()))
    return _PY_PARSER


# ------------------------------------------------------------------
# Comment / docstring stripping
# ------------------------------------------------------------------

_BLANK_LINES_RE = re.compile(r"\n\s*\n+")


def _is_string_only_expr_stmt(node) -> bool:
    """True if node is `expression_statement` whose only named child is a string."""
    if node.type != "expression_statement":
        return False
    named = [c for c in node.children if c.is_named]
    return len(named) == 1 and named[0].type == "string"


def strip_comments(code: str) -> str:
    """Strip ``#`` comments and module/function/class docstrings.

    Preserves f-strings and any string used in expression context (assignment
    RHS, function args, etc.) — only the first ``expression_statement(string)``
    of a module/function/class body is treated as a docstring.
    """
    if not code:
        return ""

    parser = _get_py_parser()
    src_bytes = code.encode("utf-8")
    tree = parser.parse(src_bytes)
    root = tree.root_node

    cuts: list[tuple[int, int]] = []

    def collect_docstring(body_or_module) -> None:
        # First named child of body — if expression_statement(string), it's a docstring.
        for child in body_or_module.children:
            if not child.is_named:
                continue
            if _is_string_only_expr_stmt(child):
                cuts.append((child.start_byte, child.end_byte))
            return  # only check the first statement

    def visit(node) -> None:
        if node.type == "comment":
            cuts.append((node.start_byte, node.end_byte))
            return
        if node.type == "module":
            collect_docstring(node)
        elif node.type in ("function_definition", "class_definition"):
            body = node.child_by_field_name("body")
            if body is not None:
                collect_docstring(body)
        for child in node.children:
            visit(child)

    visit(root)

    if not cuts:
        return _BLANK_LINES_RE.sub("\n", code).strip()

    out = bytearray(src_bytes)
    for start, end in sorted(cuts, reverse=True):
        del out[start:end]

    result = out.decode("utf-8", errors="replace")
    result = _BLANK_LINES_RE.sub("\n", result)
    return result.strip()


# ------------------------------------------------------------------
# Import parsing
# ------------------------------------------------------------------

def _node_text(src_bytes: bytes, node) -> str:
    """Slice bytes — tree-sitter offsets are byte-based, not str-based."""
    return src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def parse_imports(content: str) -> list[ImportStmt]:
    """Parse ``import`` / ``from … import`` statements from a Python source.

    `kind` is set to "local" for all entries (Python has no equivalent of the
    JS `ui-lib` library specifier — `build_dep_graph` decides resolvability).
    """
    if not content:
        return []

    parser = _get_py_parser()
    src_bytes = content.encode("utf-8")
    tree = parser.parse(src_bytes)
    root = tree.root_node

    results: list[ImportStmt] = []
    seen: set[tuple[str, tuple[str, ...], int]] = set()

    def add(specifier: str, names: list[str], level: int) -> None:
        key = (specifier, tuple(sorted(names)), level)
        if key in seen:
            return
        seen.add(key)
        results.append(ImportStmt(
            specifier=specifier,
            names=names,
            kind="local",
            level=level,
        ))

    def handle_import_statement(node) -> None:
        # `import a, b.c, d as e` — specifier is the dotted_name, no `names`.
        for child in node.named_children:
            if child.type == "dotted_name":
                add(_node_text(src_bytes,child), [], 0)
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                if name_node is not None and name_node.type == "dotted_name":
                    add(_node_text(src_bytes, name_node), [], 0)

    def handle_import_from(node) -> None:
        # Field "module_name" is dotted_name OR relative_import OR missing (`from . import x`).
        module_field = node.child_by_field_name("module_name")
        level = 0
        module_path = ""

        # Some grammars expose relative_import as the module_name field; others
        # leave the field empty and place a sibling relative_import node.
        if module_field is None:
            for c in node.children:
                if c.type == "relative_import":
                    module_field = c
                    break

        if module_field is not None:
            if module_field.type == "relative_import":
                # import_prefix carries the leading dots
                prefix = next(
                    (c for c in module_field.children if c.type == "import_prefix"),
                    None,
                )
                if prefix is not None:
                    level = len(_node_text(src_bytes, prefix))
                dn = next(
                    (c for c in module_field.children if c.type == "dotted_name"),
                    None,
                )
                if dn is not None:
                    module_path = _node_text(src_bytes, dn)
            elif module_field.type == "dotted_name":
                module_path = _node_text(src_bytes, module_field)

        # Imported names: dotted_name | aliased_import | wildcard_import.
        # Some grammars expose names under field "name"; wildcard_import in
        # tree-sitter-python is a sibling node without a field tag, so we also
        # scan all named children for it.
        names: list[str] = []
        wildcard = any(c.type == "wildcard_import" for c in node.named_children)
        if wildcard:
            names = ["*"]
        else:
            for c in node.children_by_field_name("name"):
                if c.type == "dotted_name":
                    names.append(_node_text(src_bytes, c))
                elif c.type == "aliased_import":
                    name_node = c.child_by_field_name("name")
                    if name_node is not None:
                        names.append(_node_text(src_bytes, name_node))

        add(module_path, names, level)

    def walk(node) -> None:
        if node.type == "import_statement":
            handle_import_statement(node)
            return
        if node.type == "import_from_statement":
            handle_import_from(node)
            return
        for child in node.children:
            walk(child)

    walk(root)
    return results


# ------------------------------------------------------------------
# Dep resolution
# ------------------------------------------------------------------

def _resolve_module(specifier: str, names: list[str], search_roots: list[str]) -> list[str]:
    """Resolve a dotted module path against search roots.

    Returns a list of absolute paths. Submodule resolution (e.g. ``from pkg
    import sub`` → ``pkg/sub.py``) is attempted ONLY when the resolved module
    is a package (``__init__.py``).
    """
    out: list[str] = []

    if specifier:
        parts = specifier.split(".")
        rel_path = os.path.join(*parts)
        resolved_pkg_dir: str | None = None

        for root in search_roots:
            candidate = os.path.join(root, rel_path + ".py")
            if os.path.isfile(candidate):
                out.append(candidate)
                resolved_pkg_dir = None
                break
            init = os.path.join(root, rel_path, "__init__.py")
            if os.path.isfile(init):
                out.append(init)
                resolved_pkg_dir = os.path.join(root, rel_path)
                break

        if resolved_pkg_dir and names and names != ["*"]:
            for name in names:
                if name == "*":
                    continue
                sub = os.path.join(resolved_pkg_dir, name + ".py")
                if os.path.isfile(sub) and sub not in out:
                    out.append(sub)
                    continue
                sub_init = os.path.join(resolved_pkg_dir, name, "__init__.py")
                if os.path.isfile(sub_init) and sub_init not in out:
                    out.append(sub_init)
    elif names and names != ["*"]:
        # `from . import x, y` — names are submodules of the relative root.
        for name in names:
            for root in search_roots:
                candidate = os.path.join(root, name + ".py")
                if os.path.isfile(candidate):
                    if candidate not in out:
                        out.append(candidate)
                    break
                init = os.path.join(root, name, "__init__.py")
                if os.path.isfile(init):
                    if init not in out:
                        out.append(init)
                    break

    return out


def _resolve_python_import(
    imp: ImportStmt,
    importer_dir: str,
    app_dir: str,
    lib_dir: str | None,
) -> list[str]:
    """Resolve a single ImportStmt to a list of file abs paths (or [])."""
    if imp.level > 0:
        # Walk up `level - 1` directories to reach the relative root.
        base = os.path.abspath(importer_dir)
        for _ in range(imp.level - 1):
            base = os.path.dirname(base)
        return _resolve_module(imp.specifier, imp.names, [base])

    search_roots = [os.path.abspath(app_dir)]
    if lib_dir:
        # Two layouts in the wild:
        #   (a) "flat": lib_dir is the package dir; importer writes
        #       `from <basename(lib_dir)>.X import …` → resolve against parent.
        #   (b) "nested": lib_dir contains a sub-package whose name appears
        #       as the first specifier segment (e.g. lib_dir/lib/nn/…). The
        #       importer still writes `from lib.X import …`. Resolve against
        #       lib_dir itself so the first segment matches the inner dir.
        search_roots.append(os.path.abspath(os.path.dirname(lib_dir)))
        search_roots.append(os.path.abspath(lib_dir))
    return _resolve_module(imp.specifier, imp.names, search_roots)


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------

def _should_include(file_path: str, ignore_dirs: list[str], ignore_files: list[str]) -> bool:
    return (
        not any(d in file_path for d in ignore_dirs)
        and os.path.basename(file_path) not in ignore_files
    )


def _add_file_node(nodes: dict[str, FileNode], abs_path: str, base_dir: str) -> None:
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        nodes[abs_path] = FileNode(
            abs_path=abs_path,
            rel_path=os.path.relpath(abs_path, base_dir),
            content=content,
        )
    except Exception:
        pass


def build_dep_graph(app_dir: str, lib_dir: str | None = None, task=None) -> dict[str, FileNode]:
    """Build a 1-level dependency graph for all Python files in *app_dir*.

    Library files referenced by app imports are added but their own imports
    are NOT followed.
    """
    if task is None:
        # Fallback to paperbench (the only Python task at present)
        from ..configs import load_task_config
        task = load_task_config("paperbench")

    extensions = task.code_extensions
    ignore_dirs = task.ignore_dirs
    ignore_files = task.ignore_files

    nodes: dict[str, FileNode] = {}

    for ext_pattern in extensions:
        for fpath in glob.glob(os.path.join(app_dir, "**", ext_pattern), recursive=True):
            if not _should_include(fpath, ignore_dirs, ignore_files):
                continue
            abs_path = os.path.abspath(fpath)
            rel_path = os.path.relpath(abs_path, app_dir)
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue
            nodes[abs_path] = FileNode(abs_path=abs_path, rel_path=rel_path, content=content)

    app_abs_paths = list(nodes.keys())
    app_dir_abs = os.path.abspath(app_dir)
    lib_dir_abs = os.path.abspath(lib_dir) if lib_dir else None

    for abs_path in app_abs_paths:
        node = nodes[abs_path]
        importer_dir = os.path.dirname(abs_path)
        deps: list[str] = []

        for imp in parse_imports(node.content):
            for resolved in _resolve_python_import(imp, importer_dir, app_dir, lib_dir):
                rabs = os.path.abspath(resolved)
                if not _should_include(rabs, ignore_dirs, ignore_files):
                    continue
                if rabs not in deps:
                    deps.append(rabs)
                if rabs not in nodes:
                    base = (
                        lib_dir_abs
                        if (lib_dir_abs and rabs.startswith(lib_dir_abs + os.sep))
                        else app_dir_abs
                    )
                    _add_file_node(nodes, rabs, base)

        node.deps = deps

    return nodes
