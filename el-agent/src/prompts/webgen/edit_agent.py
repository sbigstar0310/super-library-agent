"""Prompts for WebgenEditAgent — suite-level maintenance patching.

Protocol B (suite session) is used by the four library-carrying conditions
(sla-naive / sla-naive-wc / librarian / sla-ours) with the same byte-identical
template — fairness across methods holds within B. baseline runs Protocol C
only (one app per session): its workspace is a single app, so there is nothing
to share across apps and no architecture clause is needed beyond "stay inside
this codebase". The protocol is described in the paper's appendix B.
"""

from __future__ import annotations


EDIT_SYSTEM_PROMPT = """\
You are a senior software maintenance engineer. You are given one or more
existing React+Vite applications and a policy update to apply to them.

Rules:
- Apply the requested behavior changes with minimal, reasonable code changes.
- Do not remove or weaken any existing behavior outside the requested change.
- JSX must live in .jsx files (a .js file with JSX breaks the vite build).
- You may run shell commands to inspect files and verify your work.
- When you believe the work is complete, verify each requested behavior is
  implemented, then finish.

You interact with the workspace through bash commands. Think step by step,
inspect before you edit, and keep your edits minimal and precise.
"""


_PROTOCOL_B_TEMPLATE = """\
[Policy update]
{policy}

[Affected codebases]
This policy applies to the following apps in this suite: {app_list}.
The suite root is {root}. Each app's source lives at {root}/tasks/<app_id>/submission.

[Required behavior after the update — per codebase]
{per_app_behavior}

[How]
Where and how to make the changes is your decision. You may edit the apps,
and you may modify or extend the suite's existing shared code, as you see
fit. Two constraints: do not introduce new top-level packages beside the
existing ones, and do not add cross-app imports that do not already exist.
Do not remove or weaken behavior outside the list above.
"""


_PROTOCOL_C_TEMPLATE = """\
[Policy update]
{policy}

[Required behavior after the update]
This app's source lives at {root}/submission.
{behavior}

[How]
Where and how to make the changes is your decision. Do not change the app's
architecture: do not add imports pointing outside this codebase. Work within
the existing structure. Do not remove or weaken behavior outside the list
above.
"""


def build_edit_user_prompt_suite(
    *,
    policy: str,
    behaviors: dict[str, str],
    root: str,
) -> str:
    """Protocol B prompt: one suite-level session over all target apps."""
    app_list = ", ".join(sorted(behaviors))
    per_app = "\n\n".join(
        f"- {app_id}:\n  {text}" for app_id, text in sorted(behaviors.items())
    )
    return _PROTOCOL_B_TEMPLATE.format(
        policy=policy.strip(),
        app_list=app_list,
        root=root,
        per_app_behavior=per_app,
    )


def build_edit_user_prompt_single(
    *,
    policy: str,
    behavior: str,
    root: str,
) -> str:
    """Protocol C prompt: one app-level session (baseline best-case arm)."""
    return _PROTOCOL_C_TEMPLATE.format(
        policy=policy.strip(),
        behavior=behavior.strip(),
        root=root,
    )
