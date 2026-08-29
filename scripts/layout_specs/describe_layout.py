#!/usr/bin/env python3
"""Generate `layout_spec.md` from per-page screenshots.

Single batched call: the agent receives ALL pages/*.png at once,
together with their page names, and produces one Markdown file
describing the visual layout of every page. The intent is to fix the
visual/functional spec so that re-runs (baseline-v3, sla-naive-v3,
sla-ours-v3) face identical detailed requirements instead of inflating
scope to consume library symbols.

Discipline:
  * Screenshots only — do NOT pass source code to this step. The agent
    must describe what is visible on each screen, not what state shape
    or handlers exist behind it. (Reason: code-aware descriptions leak
    architectural choices back into the v3 instruction.)
  * Negative spec — the agent is asked to list components that are
    deliberately ABSENT on each page, to reduce later "symbol-driven
    feature contagion".

Runs inside cc-sandbox just like generate_pages_recipe.py.

Usage:
  python describe_layout.py \
    --pages <screenshots_dir> \
    --out <layout_spec.md> \
    [--task-file <test.jsonl> --task-id 000053] \
    [--log-dir <logs>] \
    [--model claude-opus-4-7]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _sandbox import run_agent, DEFAULT_MODEL  # noqa: E402


SYSTEM_PROMPT = """You are a visual-layout describer. You are given a
set of screenshots, each tagged with a screen name, from a small React
SPA. Produce ONE Markdown file with a section per screen, describing
the visible layout precisely enough that a different coding agent —
without seeing the screenshots — could rebuild the same visible
structure.

Each section describes ONE user-facing SCREEN the user can navigate to.
This is NOT a prescription of code organization. The downstream coder
may implement each screen as a separate file, as a tab inside one
component, as a routed page, or any other layout — your job is only
to describe what is visible. Do not group screens, do not infer file
boundaries, do not label them as "page files".

Strict rules:
- Use ONLY the screenshots. Do NOT read or speculate about source code,
  state shape, handler names, file structure, or component boundaries.
- Describe LAYOUT and CONTENT, not implementation: regions (header,
  sidebar, main, footer), grouping, alignment, key visible text, form
  fields with labels, button labels and their visible affordances,
  empty states, badges/tags/avatars when actually visible.
- Do NOT include design/marketing language ("modern", "clean", "user
  friendly"). Describe geometry and visible affordances, period.
- Do NOT prescribe colors, font sizes, or pixel values; describe the
  visible hierarchy (e.g. "primary CTA", "secondary link") instead.

Position vocabulary (be specific — never just "left" / "right"):
- Use 2-axis terms: `top-left`, `top-right`, `top-center`,
  `bottom-left`, `bottom-right`, `bottom-center`, `middle-left`,
  `middle-right`, `center`.
- For elements inside a region (header, sidebar, card, etc.), describe
  the position relative to THAT region using the same vocabulary.
  E.g. "Header (top of viewport): wordmark at header-left, nav buttons
  at header-right." or "Card (page-center): title at card-top, two
  buttons in a row at card-bottom."
- For lists/grids, use ordinal terms (first row, second row, third
  column) plus the 2-axis term for the group as a whole.
- For modals/drawers, name the anchoring edge (e.g. "drawer anchored
  to viewport-right, full height").

Output format (Markdown):

# Layout spec

## Screen: <name>
- Header: <position-of-header> — <items with their positions inside the header>
- Main content: <region position> — describe sub-regions with explicit
  positions (top, middle, bottom) and how they stack/align.
- Forms (if any): label → input type → submit label, with the form's
  position on the screen.
- Interactive elements: button labels, link labels, each with its
  position.
- Empty state / loading text (if visible), with its position.

Repeat per screen. End with a single empty newline.
"""


MEDIUM_SYSTEM_PROMPT = """You are a visual-layout describer. You are given a
set of screenshots, each tagged with a screen name, from a small React
SPA. Produce ONE Markdown file with a section per screen, describing
the visible layout at a MEDIUM level of detail — enough that a different
coding agent (without seeing the screenshots) can rebuild the same
visible structure, but WITHOUT transcribing seed data or specific text.

Each section describes ONE user-facing SCREEN the user can navigate to.
This is NOT a prescription of code organization. The downstream coder
may implement each screen as a separate file, as a tab inside one
component, as a routed page, or any other layout — your job is only
to describe what is visible. Do not group screens, do not infer file
boundaries, do not label them as "page files".

Strict rules:
- Use ONLY the screenshots. Do NOT read or speculate about source code,
  state shape, handler names, file structure, or component boundaries.
- Describe LAYOUT and STRUCTURE, not seed data: regions (header,
  sidebar, main, footer), how they stack/align, what KIND of widgets
  appear and roughly how many.
- Do NOT include design/marketing language ("modern", "clean", "user
  friendly"). Describe geometry and visible affordances, period.
- Do NOT prescribe colors, font sizes, or pixel values; describe the
  visible hierarchy (e.g. "primary CTA", "secondary link") instead.

Detail ceiling — what to INCLUDE vs SKIP:
- INCLUDE: page-level region layout (header / sidebar / main / footer),
  the position of each region using the 2-axis vocabulary below, what
  kind of widget appears in each region (e.g. "search input + two
  selects + result count", "vertical list of N cards", "data table
  with M columns"), the SEMANTIC role of buttons/links (e.g. "primary
  submit CTA", "secondary cancel", "row-level delete action"),
  navigation structure (tabs, nav buttons) with their LABELS (since
  navigation labels define the app's screens), empty-state presence.
- SKIP: literal seed-data text (e.g. specific deal titles, store names,
  user names, dates, vote counts, promo codes shown in cards/tables).
  Say "list of N promotion cards with title / store / category /
  author / date / vote-count metadata", NOT "Buy 1 Get 1 — Running
  Shoes (TrailRunner / Sports / bob / 2026-05-12, 27 votes)".
- SKIP: form field placeholder text and exact input types. Describe
  the form region as a list of FIELD LABELS only (e.g. "Form fields
  top-to-bottom: Title, Store + Category in a two-column row,
  Discount + Promo code in a two-column row, Link, Description,
  Username"). Mention the submit button by its semantic role
  ("primary submit button at form-bottom-right"), NOT its literal
  label text.
- SKIP: every table body row. Describe the table by its column
  HEADERS and approximate row count (e.g. "Users table with columns:
  Name, Role, Joined, Status, Actions — 4 rows visible").
- SKIP: footer literal copy. Just note "footer present at viewport-
  bottom-center with site copyright" or similar.

Position vocabulary (be specific — never just "left" / "right"):
- Use 2-axis terms: `top-left`, `top-right`, `top-center`,
  `bottom-left`, `bottom-right`, `bottom-center`, `middle-left`,
  `middle-right`, `center`.
- For elements inside a region, describe the position relative to
  THAT region using the same vocabulary.
- For lists/grids, use the 2-axis term for the group as a whole plus
  the row/column count. Do NOT enumerate individual items.
- For modals/drawers, name the anchoring edge.

Output format (Markdown):

# Layout spec

## Screen: <name>
- Header: <position-of-header> — <kinds of items with positions; for
  navigation, include the visible nav labels since they define screens>
- Main content: <region position> — describe sub-regions with explicit
  positions (top, middle, bottom), how they stack/align, and what
  KIND of widget each sub-region contains (form / list of N cards /
  table with M columns / etc.)
- Forms (if any): list of field LABELS in order, with the form's
  position; semantic role of the submit button (do NOT include
  placeholder text or input types).
- Interactive elements: navigation labels and action-button SEMANTIC
  roles with positions (e.g. "row-level Delete action at row-right").
  Do NOT enumerate every literal button label inside repeating items.
- Empty state / loading region (presence + position only).

Repeat per screen. End with a single empty newline.
"""


LOW_SYSTEM_PROMPT = """You are a layout-intent describer. You are
given a set of screenshots (one per SCREEN) plus a `pages.json` that
records the click path used to reach each screen, plus the original
task instruction. Your job is NOT to redraw the visual layout — it is
to produce a SHORT, INTENT-ONLY spec covering three things:

  (1) the set of top-level PAGES,
  (2) the navigation TREE between screens,
  (3) per-screen PURPOSE + a few hard pre-state rules.

You must NOT describe headers, footers, regions, positions, colors,
button labels, field labels, copy text, badges, table column names,
empty-state copy, or any other visual detail. This is an austerity
mode. Anything beyond intent + tree + pre-state is forbidden.

DEFINITIONS — page vs screen (CRITICAL):
- A `screen` = one captured screenshot frame (one entry in
  pages.json). Many screens can belong to the same page.
- A `page` = one top-level navigable destination. If two screens are
  reached by clicking the same root navigation entry and differ only
  by which tab/sub-section is selected, they belong to ONE page.
  Use the `pages.json` `actions` lists to decide grouping: screens
  sharing the SAME FIRST click belong to the same page. The page name
  should be derived from that shared first action (e.g. `Admin` for
  any screen whose first click is "Admin").
- Tabs / sub-sections inside a page are NOT separate pages.

GROUNDING RULES — Purpose one-liners:
- Each `Purpose` MUST be grounded in the original task instruction.
  Re-state which part of the instruction this screen fulfils.
- Do NOT invent new features, capabilities, sub-pages, or workflows
  that the instruction does not mention. If a screen exists in the
  screenshots but is not directly named in the instruction, attach it
  to the nearest higher-level category the instruction DOES mention
  (e.g. instruction says "management backend" → an admin tab screen's
  Purpose is "part of the management backend, handling <X>" — without
  introducing new admin features beyond what is visible).
- Keep Purpose to ONE short sentence. No layout vocabulary inside it.

PRE-STATE RULES (mandatory; emitted ONCE as global rules BEFORE the
per-screen Purpose sections — NOT inside each screen):
- Produce a single "## Pre-state" section that states two app-wide
  rules verbatim, with no per-screen enumeration:
    - Forms: every form in the app MUST be pre-filled with sensible
      default values on initial render.
    - Lists: every list / table / card grid of user data in the app
      MUST be pre-populated with seed items on initial render.
- Emit both bullets unconditionally. Do NOT list which screens have
  forms or lists. Do NOT enumerate the form fields or list items.
- Reason: the downstream GUI evaluator must not spend actions
  populating inputs or creating data.
- Do NOT repeat Forms/Lists bullets inside the per-screen sections —
  each `## Screen: <name>` section is Purpose-only.

Output format (Markdown) — EXACTLY this structure, nothing else:

# Layout spec

## Pages (<N>)
- <page-name-1>
- <page-name-2>
- ...

## Navigation tree
Entry screen: <name>

- <entry-name> → click "<text>" → <screen-name>
- <entry-name> → click "<text>" → <screen-name>
  - <screen-name> → click "<text>" → <screen-name>
  - <screen-name> → click "<text>" → <screen-name>
- ...

Entry-screen identification:
- Look at the FIRST page in `pages.json`. If its first action has no
  `click_text` (i.e. only `wait_for_text`, optionally with
  `seed_override`), that page IS the entry — use its name on
  `Entry screen:` and as the left side of every top-level edge.
- Otherwise, decide the entry by looking at the screenshots and
  `pages.json` together: pick whichever captured screen is most
  naturally the first thing a user sees (e.g. the one whose header /
  navigation matches what every other screen links FROM, or the one
  the instruction describes as the landing). Use that screen's name
  on `Entry screen:` and as the left side of every top-level edge.

- Mirror `pages.json` click sequences. Edge labels are the literal
  `click_text` values (the only place labels are allowed).
- Screens sharing the same first click are nested as children under
  that parent edge.
- Skip screens whose actions are only `fill_label` / `wait_for_text`
  (no clicks) — mention them in the per-screen sections instead.

## Pre-state
- Forms: every form in the app MUST be pre-filled with sensible
  default values on initial render.
- Lists: every list / table / card grid of user data in the app MUST
  be pre-populated with seed items on initial render.

## Screen: <name>
- Purpose: <one sentence grounded in the original instruction>

Repeat the `## Screen: <name>` block per screen. Each per-screen
section is Purpose-only — do NOT add Forms/Lists bullets here.
End with a single empty newline.
"""


def collect_screenshots(pages_dir: Path) -> list[Path]:
    """Return PNG screenshots sorted by name."""
    pngs = sorted(p for p in pages_dir.glob("*.png") if p.is_file())
    if not pngs:
        sys.exit(f"[describe] no PNGs found in {pages_dir}")
    return pngs


def build_user_prompt(
    pngs: list[Path],
    out_path: Path,
    task_instruction: str | None,
    pages_json_text: str | None = None,
) -> str:
    img_block = "\n".join(
        f"  - screen `{p.stem}` at: {p}" for p in pngs
    )

    instruction_block = ""
    if task_instruction:
        instruction_block = f"""[Original task instruction the app was built from — REQUIRED grounding for Purpose lines in LOW mode, context only in other modes]
{task_instruction}

"""

    pages_json_block = ""
    if pages_json_text:
        pages_json_block = f"""[pages.json — click path to reach each screen. Use this to derive the page grouping (screens sharing the same first click belong to the same page) and to build the Navigation tree section. Edge labels in the tree MUST be the literal `click_text` values shown here.]
```json
{pages_json_text}
```

"""

    return f"""{instruction_block}{pages_json_block}Read each screenshot below using your image-reading tool and write a
Markdown layout spec to: {out_path}

[Screenshots — one per screen]
{img_block}

Follow the system-prompt rules exactly.

Write the file with the Write tool and end the turn.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", required=True, type=Path,
                    help="Directory of <name>.png screenshots")
    ap.add_argument("--out", required=True, type=Path,
                    help="Where to write layout_spec.md")
    ap.add_argument("--task-file", type=Path, default=None,
                    help="Optional: webgen test.jsonl to pull the task "
                         "instruction for context")
    ap.add_argument("--task-id", default=None)
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--medium-detail", action="store_true",
                    help="Use the medium-detail SYSTEM_PROMPT: keep "
                         "region layout + position vocabulary + nav "
                         "labels + form field labels, but skip seed "
                         "data transcripts, placeholder text, input "
                         "types, table body rows, and literal "
                         "button-label dumps. Mutually exclusive with "
                         "--no-form-details.")
    ap.add_argument("--low-detail", action="store_true",
                    help="Use the low-detail SYSTEM_PROMPT: intent-only "
                         "spec — Pages list, Navigation tree from "
                         "pages.json, and per-screen Purpose + Forms/"
                         "Lists pre-state rules. No layout description. "
                         "Requires pages.json (auto-discovered from "
                         "--pages dir parent, or set via --pages-json). "
                         "Mutually exclusive with --medium-detail.")
    ap.add_argument("--pages-json", type=Path, default=None,
                    help="Path to pages.json. LOW mode only — defaults "
                         "to <pages>/../pages.json if not given.")
    args = ap.parse_args()

    if not args.pages.is_dir():
        sys.exit(f"--pages dir not found: {args.pages}")

    pngs = collect_screenshots(args.pages)

    task_instruction = None
    if args.task_file and args.task_id:
        for line in args.task_file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("id")) == args.task_id:
                task_instruction = (row.get("instruction") or "").strip()
                break

    pages_json_text: str | None = None
    if args.low_detail:
        pages_json_path = args.pages_json or (args.pages.parent / "pages.json")
        if not pages_json_path.is_file():
            sys.exit(
                f"--low-detail requires pages.json. Looked at: "
                f"{pages_json_path}. Pass --pages-json to override."
            )
        pages_json_text = pages_json_path.read_text().strip()

    user_text = build_user_prompt(
        pngs, args.out, task_instruction, pages_json_text=pages_json_text,
    )

    log_dir = args.log_dir or (args.out.parent / ".agent_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = log_dir / "describe_layout.txt"
    prompt_file.write_text(user_text)

    # rw: out dir. ro: screenshots dir (agent must Read them).
    rw_dirs = [args.out.parent]
    ro_dirs = [args.pages]

    if sum(bool(x) for x in (args.medium_detail, args.low_detail)) > 1:
        sys.exit("--low-detail, --medium-detail")

    if args.low_detail:
        system_text = LOW_SYSTEM_PROMPT
    elif args.medium_detail:
        system_text = MEDIUM_SYSTEM_PROMPT
    else:
        system_text = SYSTEM_PROMPT

    rc = run_agent(
        prompt_file=prompt_file,
        rw_dirs=rw_dirs,
        ro_dirs=ro_dirs,
        description=f"describe_layout ({len(pngs)} pages"
                    f"{', low' if args.low_detail else ', medium' if args.medium_detail else ''})",
        model=args.model,
        system_text=system_text,
    )
    if rc != 0:
        print(f"[describe] agent failed (rc={rc}). Log: "
              f"{prompt_file.with_suffix('.agent.jsonl')}")
        return rc
    if not args.out.is_file():
        print(f"[describe] agent did not write {args.out}")
        return 1
    print(f"[describe] wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
