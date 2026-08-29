#!/usr/bin/env python3
"""Deterministic page crawler driven by a `pages.json` recipe.

Boots a dev server on the given submission (npm install + vite dev),
then for each page entry:
  1. opens a fresh browser context (no cookies, no storage)
  2. navigates to '/'
  3. applies localStorage_seed (recipe-level)
  4. reloads so the seed takes effect
  5. runs the actions in order
  6. captures a full-page screenshot to `<out_dir>/<name>.png`

Action vocabulary (matches verify_pages_recipe.py):
  {"click_text":    "<exact visible text>"}
  {"fill_label":    "<label text>", "value": "<string>"}
  {"wait_for_text": "<text>"}

The dev server is the caller's responsibility — the script expects it
already running and reachable at the given URL. (build_layout_spec.sh handles
the start/stop.)

Exit code:
  0 if every page screenshot was captured.
  1 if any selector failed (silent skip is forbidden; the run aborts so
    the recipe can be fixed).

Usage:
  python crawl_pages.py \
    --recipe <pages.json> \
    --url http://localhost:5173 \
    --out <screenshots_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


def json_quote(s: str) -> str:
    """Return an XPath string-literal that safely encodes any quote chars.

    XPath has no string-escape syntax — for labels containing both ' and "
    you must use concat(). Most labels are plain ASCII, so we take a
    fast path for those.
    """
    if "'" not in s:
        return f"'{s}'"
    if '"' not in s:
        return f'"{s}"'
    parts = s.split("'")
    return "concat(" + ", \"'\", ".join(f"'{p}'" for p in parts) + ")"


VIEWPORT = {"width": 1280, "height": 800}
SHORT_TIMEOUT_MS = 8000   # per-selector (was 5000 — give SPA late-attach a beat)
NAV_TIMEOUT_MS   = 30000
SETTLE_MS        = 500    # between actions, allow React state flush
SUBMIT_BUTTONS = {"Log In", "Sign In", "Login", "Sign in",
                  "Register", "Sign Up", "Submit", "Create Account"}


def apply_action(page, act: dict, prev_act: dict | None = None) -> None:
    """Execute one recipe action against the current page.

    ``prev_act`` carries the previous recipe action so we can pick a
    better selector when a click follows a form fill — see the
    submit-button priority below.
    """
    if "click_text" in act:
        txt = act["click_text"]
        # When the previous action filled a form input AND the current
        # click_text matches a submit-button word, prefer the form's
        # `<button type="submit">`. Without this, pages that render
        # both a mode-toggle "Login" button (header) AND a form-submit
        # "Login" button (inside <form>) yield first-match = the
        # mode-toggle (no-op), and the form is never submitted.
        prev_was_fill = prev_act is not None and "fill_label" in prev_act
        if prev_was_fill and txt in SUBMIT_BUTTONS:
            try:
                page.locator('button[type="submit"]').get_by_text(
                    txt, exact=True
                ).first.click(timeout=2500)
                try:
                    page.wait_for_load_state('networkidle', timeout=2000)
                except PWTimeout:
                    pass
                return
            except PWTimeout:
                pass
        # Priority:
        #   1. button/a elements with the exact text — skips <option>
        #      entries that share text with a tab/nav button (e.g. a
        #      "Technology" filter rendered as both a <select><option>
        #      and a <button> — first text-match would grab the option).
        #   2. any element with exact text.
        #   3. substring text (for STATIC PREFIX recipes like "My
        #      applications ({n})").
        try:
            page.locator("button, a").get_by_text(txt, exact=True).first.click(timeout=2500)
        except PWTimeout:
            try:
                page.get_by_text(txt, exact=True).first.click(timeout=2000)
            except PWTimeout:
                page.get_by_text(txt, exact=False).first.click(timeout=SHORT_TIMEOUT_MS)
        # Login/submit-like buttons trigger a full view swap. Wait for
        # the network to go idle so the post-login view's controls are
        # mounted before the next action's click_text fires.
        if txt in SUBMIT_BUTTONS:
            try:
                page.wait_for_load_state('networkidle', timeout=2000)
            except PWTimeout:
                pass
    elif "fill_label" in act:
        lbl = act["fill_label"]
        value = act.get("value", "")
        # Playwright's get_by_label requires explicit association
        # (`<label for>`, nesting, or aria-label). Many React forms write
        #   <label>X</label><input/>
        # with neither — get_by_label times out. Fall back to "find a
        # <label> whose text is X, then the next input/textarea/select
        # in document order" before giving up.
        #
        # .first avoids strict-mode violation when the same label text
        # appears more than once on the page (e.g. nav "Search" link +
        # filter "Search" input). The xpath fallback catches both the
        # PWTimeout (no association) and any other resolution error.
        try:
            page.get_by_label(lbl).first.fill(value, timeout=2000)
        except Exception:
            # XPath union — match input either INSIDE the label (the
            # `<label>X<input/></label>` nested form) or FOLLOWING it
            # (the `<label>X</label><div><input/></div>` sibling form).
            # `following::` does not descend into the label itself, so
            # nested-input forms must be covered by the descendant arm.
            quoted = json_quote(lbl)
            xp = (
                f"xpath=(//label[normalize-space(.)={quoted}]"
                "//input | "
                f"//label[normalize-space(.)={quoted}]"
                "/following::*[self::input or self::textarea or self::select][1])[1]"
            )
            page.locator(xp).first.fill(value, timeout=SHORT_TIMEOUT_MS)
    elif "wait_for_text" in act:
        # .first because the same string often appears in nav AND header
        # (e.g. nav button "Student Login" + page heading "Student Login").
        # We only need ANY matching element to confirm navigation arrived.
        # Soft: a missing text is often a recipe-side issue (dynamic
        # content, empty-state copy) — log it and capture the screen
        # anyway. The describer can interpret empty states correctly.
        try:
            page.get_by_text(act["wait_for_text"]).first.wait_for(
                timeout=SHORT_TIMEOUT_MS
            )
        except PWTimeout:
            print(f"    ⚠ wait_for_text {act['wait_for_text']!r} not found"
                  f" — capturing screen anyway")
    else:
        raise ValueError(f"Unknown action: {act!r}")


def run(recipe_path: Path, base_url: str, out_dir: Path) -> int:
    recipe = json.loads(recipe_path.read_text())
    pages = recipe.get("pages") or []
    seed = recipe.get("localStorage_seed") or {}
    if not pages:
        sys.exit("[crawl] recipe has no `pages` entries")

    out_dir.mkdir(parents=True, exist_ok=True)

    failed: list[tuple[str, str]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for entry in pages:
            name = entry["name"]
            actions = entry.get("actions") or []
            ctx = browser.new_context(viewport=VIEWPORT)
            page = ctx.new_page()
            try:
                # One retry: transient vite/network spikes occasionally
                # blow past NAV_TIMEOUT_MS on a single goto even when the
                # server is healthy (observed once mid-batch on ai5).
                try:
                    page.goto(base_url, timeout=NAV_TIMEOUT_MS)
                except PWTimeout:
                    page.goto(base_url, timeout=NAV_TIMEOUT_MS)
                # Per-entry seed override (merged on top of recipe-level
                # seed). Lets a single recipe carry a default logged-in
                # state while a few entries (auth-login, auth-register)
                # pin themselves to logged-out by overriding the session
                # key with null.
                entry_seed = {**seed, **(entry.get("seed_override") or {})}
                # Seed BEFORE actions but after first navigation so the
                # origin is established (localStorage is per-origin).
                #
                # Some apps store raw strings (e.g. user id) under their
                # session key; others store JSON objects. Detect by
                # value type:
                #   - str  → write the string verbatim (matches
                #            `localStorage.setItem(k, "<id>")`)
                #   - None → removeItem (clears the key entirely)
                #   - else → JSON.stringify (object/array/number/bool)
                for k, v in entry_seed.items():
                    if v is None:
                        page.evaluate(
                            "(k) => localStorage.removeItem(k)", k,
                        )
                    elif isinstance(v, str):
                        page.evaluate(
                            "([k, v]) => localStorage.setItem(k, v)",
                            [k, v],
                        )
                    else:
                        page.evaluate(
                            "([k, v]) => localStorage.setItem(k, JSON.stringify(v))",
                            [k, v],
                        )
                if entry_seed:
                    page.reload(timeout=NAV_TIMEOUT_MS)

                prev_act = None
                for act in actions:
                    apply_action(page, act, prev_act=prev_act)
                    prev_act = act
                    # settle delay; React re-render after a click can lag
                    # by 100–400ms even after networkidle.
                    page.wait_for_timeout(SETTLE_MS)

                out_path = out_dir / f"{name}.png"
                page.screenshot(path=str(out_path), full_page=True)
                print(f"  ✓ {name}  -> {out_path.name}")
            except (PWTimeout, Exception) as exc:
                failed.append((name, str(exc).splitlines()[0][:200]))
                print(f"  ✗ {name}  {exc.__class__.__name__}: "
                      f"{str(exc).splitlines()[0][:200]}")
            finally:
                ctx.close()
        browser.close()

    print(f"\n[crawl] {len(pages) - len(failed)}/{len(pages)} OK")
    if failed:
        print("[crawl] FAIL")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", required=True, type=Path)
    ap.add_argument("--url", required=True,
                    help="Base URL of the running dev server, e.g. "
                         "http://localhost:5173")
    ap.add_argument("--out", required=True, type=Path,
                    help="Directory for `<name>.png` screenshots.")
    args = ap.parse_args()
    if not args.recipe.is_file():
        sys.exit(f"recipe not found: {args.recipe}")
    return run(args.recipe, args.url.rstrip("/"), args.out)


if __name__ == "__main__":
    sys.exit(main())
