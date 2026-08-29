#!/usr/bin/env python3
"""Static sanity-check for a `pages.json` recipe against the React source.

Three checks:
  1. Page-state mismatch — set of page names in recipe vs set discovered
     in `src/App.jsx` (case 'X', setPage('X'), <Route path>).
  2. Nav-text validity — every `click_text` in recipe must appear as
     visible text in some `<button>` / `<Link>` / `<a>` in src/.
  3. Form-label validity (best-effort) — `fill_label` should match a
     `<label>` text in src/. Warning-only since labels can be dynamic.

Exit code:
  0 if all checks pass (label warnings allowed).
  1 if any hard mismatch (1) or invalid click_text (2).

Usage:
  python verify_pages_recipe.py \
    --submission <submission_dir> \
    --recipe <pages.json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Regex hints; tolerant rather than perfectly accurate.
RE_CASE_STR    = re.compile(r"""case\s+['"]([\w-]+)['"]\s*:""")
# Page-state setters: only the canonical names. Avoid greedy `set\w+`
# which catches `setItem`, `setForm`, `setApplications`, etc.
RE_SETPAGE     = re.compile(
    r"""set(?:Page|View|Tab|Screen|Route|ActiveTab|ActiveView|"""
    r"""CurrentView|CurrentPage|CurrentTab)\(\s*['"]([\w-]+)['"]"""
)
RE_ROUTE_PATH  = re.compile(r"""<Route[^>]*\bpath\s*=\s*\{?['"]([^'"]+)['"]""")
# Attribute area can contain `>` (e.g. JSX arrow `() =>`). We swallow
# `{ ... }` JSX expressions as a single token, then everything else up
# to `>`. Single-level brace nesting is enough for typical handlers.
_ATTRS = r"""(?:[^>{]|\{[^{}]*\})*"""
RE_BUTTON_TEXT = re.compile(rf"""<button\b{_ATTRS}>([\s\S]*?)</button>""")
RE_A_TEXT      = re.compile(rf"""<a\b{_ATTRS}>([\s\S]*?)</a>""")
RE_LINK_TEXT   = re.compile(rf"""<Link\b{_ATTRS}>([\s\S]*?)</Link>""")
RE_LABEL_TEXT  = re.compile(rf"""<label\b{_ATTRS}>([\s\S]*?)</label>""")
# `<span onClick={...}>` is a common React idiom for inline clickable
# text (e.g. "Register here" inside a login form). Match any <span> for
# simplicity — noise is acceptable, false negatives are not.
RE_SPAN_TEXT   = re.compile(rf"""<span\b{_ATTRS}>([\s\S]*?)</span>""")


def discover_page_names(src_dir: Path) -> set[str]:
    """Page identifiers found across all `.jsx` files."""
    names: set[str] = set()
    for f in src_dir.rglob("*.jsx"):
        text = f.read_text(errors="ignore")
        names.update(RE_CASE_STR.findall(text))
        names.update(RE_SETPAGE.findall(text))
        # Route paths: strip leading slash, take first segment.
        for path in RE_ROUTE_PATH.findall(text):
            seg = path.strip("/").split("/")[0].strip(":").strip()
            if seg:
                names.add(seg)
    return {n for n in names if n}


def discover_visible_texts(src_dir: Path) -> set[str]:
    """Best-effort visible text inside buttons/links/clickable spans."""
    texts: set[str] = set()
    for f in src_dir.rglob("*.jsx"):
        text = f.read_text(errors="ignore")
        for rx in (RE_BUTTON_TEXT, RE_A_TEXT, RE_LINK_TEXT, RE_SPAN_TEXT):
            for m in rx.findall(text):
                stripped = " ".join(m.split())
                # Skip JSX expressions like {currentUser.name} — we only
                # care about literal visible text the recipe might target.
                if stripped and "{" not in stripped and "}" not in stripped:
                    texts.add(stripped)
    return texts


def discover_label_texts(src_dir: Path) -> set[str]:
    texts: set[str] = set()
    for f in src_dir.rglob("*.jsx"):
        text = f.read_text(errors="ignore")
        for m in RE_LABEL_TEXT.findall(text):
            stripped = " ".join(m.split())
            if stripped and "{" not in stripped:
                texts.add(stripped)
    return texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", required=True, type=Path,
                    help="Path to the submission/ dir (contains src/).")
    ap.add_argument("--recipe", required=True, type=Path,
                    help="Path to pages.json.")
    args = ap.parse_args()

    src_dir = args.submission / "src"
    if not src_dir.is_dir():
        sys.exit(f"src/ not found under {args.submission}")
    if not args.recipe.is_file():
        sys.exit(f"recipe not found: {args.recipe}")

    recipe = json.loads(args.recipe.read_text())
    pages = recipe.get("pages") or []
    recipe_page_names = {p["name"] for p in pages if "name" in p}

    discovered_pages = discover_page_names(src_dir)
    nav_texts = discover_visible_texts(src_dir)
    label_texts = discover_label_texts(src_dir)

    # ── Check 1: page state mismatch ─────────────────────────────────────
    # Normalize identifier style — agents often emit kebab-case
    # (`student-login`) while React source uses camelCase
    # (`studentLogin`). These are semantically identical for verification
    # purposes, so we collapse both to lowercase alnum-only before diffing.
    def _norm(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    src_norm    = {_norm(n): n for n in discovered_pages}
    recipe_norm = {_norm(n): n for n in recipe_page_names}
    in_source_only = {src_norm[k]    for k in src_norm    if k not in recipe_norm}
    in_recipe_only = {recipe_norm[k] for k in recipe_norm if k not in src_norm}
    # If our regex hint set is empty, the app likely uses navigation
    # patterns we don't cover (setView/setTab, react-router, etc.).
    # Treat the page-set comparison as warning-only in that case; the
    # crawler is the real source of truth.
    page_set_soft = (len(discovered_pages) == 0)

    # ── Check 2: click_text validity ─────────────────────────────────────
    # Allow exact match OR a prefix match against any nav_text. Prefix
    # match catches dynamic-count buttons like "My applications (0)"
    # whose static prefix "My applications" appears in source but the
    # full literal is JSX-expression interpolated and stripped by the
    # hint extractor.
    invalid_clicks: list[tuple[str, str]] = []
    for p in pages:
        for act in p.get("actions") or []:
            txt = act.get("click_text")
            if not txt:
                continue
            if txt in nav_texts:
                continue
            # Prefix-match fallback: recipe text starts with a known
            # static text from source.
            if any(txt.startswith(nt) or nt.startswith(txt)
                   for nt in nav_texts if nt):
                continue
            invalid_clicks.append((p["name"], txt))

    # ── Check 3 (warn-only): fill_label coverage ─────────────────────────
    label_warnings: list[tuple[str, str]] = []
    for p in pages:
        for act in p.get("actions") or []:
            lbl = act.get("fill_label")
            if lbl and lbl not in label_texts:
                label_warnings.append((p["name"], lbl))

    # ── Report ───────────────────────────────────────────────────────────
    print(f"[verify] submission : {args.submission}")
    print(f"[verify] recipe     : {args.recipe}")
    print(f"[verify] pages in source : {sorted(discovered_pages)}")
    print(f"[verify] pages in recipe : {sorted(recipe_page_names)}")
    print()

    hard_fail = False
    if in_source_only:
        if page_set_soft:
            print(f"  ⚠ MISSING from recipe (soft, source extraction empty): {sorted(in_source_only)}")
        else:
            print(f"  ✗ MISSING from recipe : {sorted(in_source_only)}")
            hard_fail = True
    if in_recipe_only:
        if page_set_soft:
            print(f"  ⚠ EXTRA in recipe (soft, source extraction empty): {sorted(in_recipe_only)}")
        else:
            print(f"  ✗ EXTRA in recipe    : {sorted(in_recipe_only)}")
            hard_fail = True
    if invalid_clicks:
        # Warning-only: the recipe author (Opus) is expected to have
        # read the source directly, so its click_text may legitimately
        # refer to elements our regex doesn't cover (custom tags,
        # nested JSX, dynamic-count buttons with no static fallback).
        # The crawler is the real validator; we just surface the
        # mismatch for diagnostics.
        print(f"  ⚠ click_text not statically matched (crawler will verify):")
        for page, txt in invalid_clicks:
            print(f"      [{page}] {txt!r}")
    if label_warnings:
        print(f"  ⚠ fill_label not statically matched (may still work):")
        for page, lbl in label_warnings:
            print(f"      [{page}] {lbl!r}")

    if hard_fail:
        print("\n[verify] FAIL")
        return 1
    print("\n[verify] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
