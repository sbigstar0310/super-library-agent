#!/usr/bin/env python3
"""Generate `pages.json` for an existing React submission.

Pipeline:
  1. Collect static hints from src/ (page state names, nav text, labels).
  2. Build a focused prompt: instruction + App.jsx + hints + schema.
  3. Launch Claude Opus 4.7 inside cc-sandbox to write `pages.json`
     into the submission directory.

The recipe is *post-hoc*: it describes how to walk the already-finished
app, not how to build it. The agent reads the source to figure out the
shortest deterministic action sequence to reach each page/screen.

Usage:
  python generate_pages_recipe.py \
    --submission <submission_dir> \
    --task-file <test.jsonl> \
    --task-id 000053 \
    --out <pages.json> \
    [--log-dir <logs>] \
    [--model claude-opus-4-7]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Reuse the verify hints so the agent sees the same `discovered_pages`
# the verifier will later check against — keeps both ends consistent.
sys.path.insert(0, str(Path(__file__).parent))
from _sandbox import run_agent, DEFAULT_MODEL  # noqa: E402
from verify_pages_recipe import (  # noqa: E402
    discover_label_texts,
    discover_page_names,
    discover_visible_texts,
)


SCHEMA = """
{
  "localStorage_seed": { "<key>": <json-value>, ... },
  "pages": [
    {
      "name": "<unique-slug-for-this-page>",
      "actions": [
        {"click_text":    "<exact visible text of a button/link>"} |
        {"fill_label":    "<exact <label> text>", "value": "<string>"} |
        {"wait_for_text": "<text>"}
      ]
    }, ...
  ]
}
""".strip()


SYSTEM_PROMPT = """You are a Playwright recipe authoring agent. Your job
is to write a `pages.json` file that lets a deterministic crawler walk
every distinct screen of an already-finished React + Vite SPA, capture
one screenshot per screen, and reproduce the same screens on every run.

Source of truth = the actual source code. The user prompt gives you a
small set of static hints, but those hints are LOSSY (regex-extracted
from a single nav pattern). For any non-trivial app you MUST read the
source yourself before authoring the recipe.

Required workflow:
  1. Read `src/App.jsx` to identify the top-level navigation pattern
     (state hook name, route library, prop drilling, etc.).
  2. Recursively read every component file that participates in
     navigation (anything that calls the page-state setter, mounts
     subviews, or holds a `useState` for "view"/"tab"/"page"/"screen").
  3. From the source, enumerate EVERY distinct screen the user can
     reach (top-level views, sub-tabs, modals/drawers with distinct
     content).
  4. For each screen, find the EXACT visible button/link/anchor text
     in the JSX that triggers that navigation. Copy the literal text
     between the tags. Class names, ids, and prop names are NOT valid.
  5. Only then write the recipe.

Hard rules:
- The recipe is consumed by Playwright. Selectors must resolve.
- `click_text` matches `page.get_by_text(<text>, exact=True).first.click()`
  (with an `exact=False` fallback). Use the EXACT visible text that the
  user reads on screen — never paraphrase from the task instruction.
- For dynamic-count buttons (e.g. JSX `My applications ({count})`
  rendering as `My applications (3)`), emit the STATIC PREFIX
  `My applications` as `click_text`. Do NOT invent a concrete count.
- `fill_label` matches `page.get_by_label(<label>).fill(<value>)`. Use
  the exact text inside the `<label>` element.
- Every page entry starts from the ROOT URL with cleared state; the
  crawler reloads between pages. So actions must be self-sufficient.
- `localStorage_seed` is applied AFTER the first navigation and BEFORE
  actions, then the page is reloaded. Use it ONLY to pre-seed users so
  login flows are reachable without a real signup. Store JSON values.
- Cover every reachable distinct screen: home, login(s), register(s),
  dashboard(s), detail/edit views, and *modals/drawers* if they have
  visibly different content.
- Pick short stable slugs for `name` (kebab-case is fine).
- Do NOT invent buttons. If a flow needs a button that doesn't exist,
  skip the page rather than fabricate a selector.

Output discipline:
- Write the file with the `Write` tool. Path will be given in the user
  prompt. The file must be valid JSON, no trailing commas, no comments.
- After writing, end the turn. Do not modify any other file.
"""


def collect_hints(src_dir: Path) -> dict:
    return {
        "page_states": sorted(discover_page_names(src_dir)),
        "nav_texts":   sorted(discover_visible_texts(src_dir)),
        "label_texts": sorted(discover_label_texts(src_dir)),
    }


def list_src_files(src_dir: Path, max_files: int = 60) -> list[Path]:
    """Sorted list of every .jsx / .js / .css under src/, capped."""
    out = sorted(
        f for f in src_dir.rglob("*")
        if f.is_file() and f.suffix in {".jsx", ".js", ".css"}
    )
    return out[:max_files]


def read_safe(path: Path, max_chars: int = 8000) -> str:
    try:
        text = path.read_text(errors="ignore")
    except OSError as exc:
        return f"<read error: {exc}>"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"
    return text


def build_user_prompt(
    task_instruction: str,
    submission_dir: Path,
    src_dir: Path,
    hints: dict,
    out_path: Path,
) -> str:
    file_list = list_src_files(src_dir)
    rel_listing = "\n".join(f"  {p.relative_to(submission_dir)}" for p in file_list)

    return f"""Generate a Playwright `pages.json` recipe for the React app at
`{submission_dir}` and write it to `{out_path}`.

The app source lives at `{src_dir}` and is mounted READ-ONLY in your
sandbox. Use the Read tool to inspect it — start with `src/App.jsx`,
then follow component imports. The static hints below are a fallback
quick-reference, not a substitute for reading the source.

[Original task instruction the app was built from]
{task_instruction}

[Source-file index — Read tool any of these]
{rel_listing}

[Static hints (lossy — may miss dynamic text and uncommon nav patterns)]
- Page-state slugs picked up by regex (`case 'X'`, `setPage('X')`, `<Route path>`):
    {hints['page_states']}
- Visible button/link/anchor/clickable-span texts:
    {hints['nav_texts']}
- `<label>` texts (for forms):
    {hints['label_texts']}

If the page-state slug list is empty or sparse, the app uses a nav
pattern outside the regex (e.g. `setView('student')`, `setTab(...)`,
`react-router`, prop-drilled view names). Discover those by reading
`src/App.jsx` and the imported components.

[Required output]
Write the recipe to: {out_path}

Schema (exact keys, exact action vocabulary):
{SCHEMA}

Coverage requirement: one entry per distinct screen reachable in the
running app — top-level views, sub-tabs, modals/drawers with
distinguishable content. If the screen needs a logged-in user, prime
`localStorage_seed` with the exact key(s) the app reads from. For
tab-driven screens, emit one entry per tab.

End the turn after writing the file.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--task-file", required=True, type=Path,
                    help="WebGen-Bench JSONL")
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--out", required=True, type=Path,
                    help="Where to write pages.json (e.g. "
                         "<submission>/pages.json)")
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="Where to drop the prompt + agent log "
                         "(default: <out>.parent/.agent_logs)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    src_dir = args.submission / "src"
    if not src_dir.is_dir():
        sys.exit(f"src/ not found under {args.submission}")

    # Find task instruction by id.
    instruction = None
    for line in args.task_file.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("id")) == args.task_id:
            instruction = (row.get("instruction") or "").strip()
            break
    if not instruction:
        sys.exit(f"task id {args.task_id!r} not found in {args.task_file}")

    hints = collect_hints(src_dir)
    user_text = build_user_prompt(
        instruction, args.submission, src_dir, hints, args.out,
    )

    log_dir = args.log_dir or (args.out.parent / ".agent_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = log_dir / f"generate_pages_{args.task_id}.txt"
    prompt_file.write_text(user_text)

    # rw: out file parent + submission (agent may need to Read src/)
    # ro: nothing extra — submission is already RW
    rw_dirs = [args.out.parent, args.submission]
    rc = run_agent(
        prompt_file=prompt_file,
        rw_dirs=rw_dirs,
        ro_dirs=[],
        description=f"generate_pages_recipe {args.task_id}",
        model=args.model,
        system_text=SYSTEM_PROMPT,
    )
    if rc != 0:
        print(f"[generate] agent failed (rc={rc}). Log: "
              f"{prompt_file.with_suffix('.agent.jsonl')}")
        return rc
    if not args.out.is_file():
        print(f"[generate] agent did not write {args.out}")
        return 1
    try:
        recipe = json.loads(args.out.read_text())
    except json.JSONDecodeError as exc:
        print(f"[generate] {args.out} is not valid JSON: {exc}")
        return 1
    n_pages = len(recipe.get("pages") or [])
    print(f"[generate] wrote {args.out} ({n_pages} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
