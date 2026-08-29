"""Prompts for PaperbenchCodingAgent (baseline + sla_naive + sla_ours).

**Upstream-verbatim policy** (see `paperbench-notes.md`): the system and
user prompts are taken 1:1 from paperbench's basicagent reference
solver so behavior stays comparable to the upstream evaluation. We mirror
the **non-iterative** path because that is the upstream default
(`paperbench/solvers/basicagent/solver.py:77` —
``iterative_agent: bool = chz.field(default=False)``). The corresponding
upstream sources are:

  - system  : ``templates.py:SYSTEM_MESSAGE_BASE`` + ``SYSTEM_MESSAGE_END``
              i.e. ``get_system_message(iterative=False, code_only=True)``
  - user    : ``paperbench/instructions/code_only_instructions.txt`` +
              ``additional_notes_template``
              i.e. the file ``get_instructions(iterative_agent=False,
              task.judge.code_only=True)`` returns

The only changes are:

  - Path substitutions: ``/home/paper`` → ``{paper_dir}``,
    ``/home/submission/`` → ``{workspace_dir}``,
    ``/home/agent.env`` (implicit in additional_notes_template) →
    ``{agent_env_path}``.
  - ``additional_notes_template`` trimmed to fields applicable to this
    local env: drop the "root access" bullet, hardcode "no GPU" instead
    of probing, and surface the time limit in hours. Add two safety
    bullets ("Per-bash timeout (10 minutes)" and "Grading is code-only")
    that document mini-swe-agent runtime invariants that are NOT in the
    upstream template.
  - Two optional SLA slots tacked into the RESOURCES section:
    ``{library_block}`` — non-empty in sla_naive / sla_ours rounds where
    a pre-built ``lib/`` is mounted (built from
    :data:`prompts.paperbench.common.LIBRARY_BLOCK`); empty in baseline.
    ``{whitelist_block}`` — `utils.whitelist.render_for_prompt()`.
  - ``{app_rules}`` — :data:`prompts.paperbench.paperbench.APP_RULES`
    injected between RESOURCES and SUBMISSION. SLA-specific stack /
    compute / required-file contract that the upstream prompt does not
    cover (Python 3.11+, no monorepo, no vendoring of large deps). Same
    block is injected by the apply / local_extract agents — keeping it
    here makes the coding submission obey the same rules its downstream
    refactor phases will enforce. Webgen mirror.

Do NOT edit the prompt body. Any deliberate deviation is a documented
change to paperbench reference parity and should be recorded in
``paperbench-notes.md``.
"""

from __future__ import annotations
from typing import Any

from prompts.paperbench.common import LIBRARY_BLOCK
from prompts.paperbench.paperbench import APP_RULES


__all__ = [
    "PAPERBENCH_SYSTEM_PROMPT",
    "PAPERBENCH_USER_PROMPT",
    "build_prompt_from_task",
]


# System prompt: upstream verbatim. get_system_message(iterative=False,
# code_only=True) = SYSTEM_MESSAGE_BASE + SYSTEM_MESSAGE_END
# (templates.py:1-17 + :31-32).

_SYSTEM_MESSAGE_BASE = """You are a helpful agent attempting to solve a task. You have
several functions available to help with solving the task. Each message may
may perform one function call. You will see the result of the function right
after sending the message. If you need to perform multiple actions, you can
always send more messages with subsequent function calls. Do some reasoning
before your actions, describing what function calls you are going to use and
how they fit into your plan.

Use as much of the available time as possible to refine and optimize your solution: Your goal is to achieve the best possible score, not just to submit your first working solution. Keep working until the time limit expires.

YOU MUST use the bash and python tools provided provided to create your replication. ACTUALLY REPLICATE the paper YOURSELF, do not give me instructions how I can do so.

Only execute a single command using each tool at once.

This is a multi-step task. DO THINGS BIT BY BIT by slowly building up your solution with multiple tool calls, don't try to do everything in one go. Just reason about what is the best next step and execute it. Then take the next step after you receive the output of the previous step.

Do not stop until you have replicated all results from the paper. Before calling end_task, YOU MUST have an entire codebase that can reproduce all metrics, figures, tables, and findings from the paper."""

_SYSTEM_MESSAGE_END = """
Remember, you are tasked with replicating a whole paper, not just a single part of it or a minimal example. The file read tool is PAGINATED, so you will need to CALL IT MULTIPLE TIMES to make sure that you have read all the relevant parts of the paper."""

PAPERBENCH_SYSTEM_PROMPT = _SYSTEM_MESSAGE_BASE + _SYSTEM_MESSAGE_END


# User prompt: upstream verbatim (code_only_instructions.txt body +
# trimmed additional_notes_template).

PAPERBENCH_USER_PROMPT = """\
You are tasked with reproducing a research paper.

You have been provided with a research paper located at `{paper_dir}`. The directory contains the paper in both PDF and markdown format.

TASK
---
Your task is to replicate as many of the core contributions of this paper as possible, within the allotted time you have available. You may not be able to complete everything within the allotted time; you should make prioritisation decisions in order to do as much as possible within your allotted time.

Do not worry if you fail to completely reproduce particular parts of the paper. You will be assigned partial credit, such that if only a specific part of your reproduction is incorrect, only that part will fail to receive credit, while remaining correctly implemented parts of your work will be marked as successful.  Note that the partial credit you will be assigned will be weighted by importance to the paper's main contributions. Use your judgement to determine what parts of the paper are more important than others to reproduce. Note that we will check that your results match the general trends of the original paper and we will allow for a reasonable margin of error, so you should not worry if metrics do not match exactly between your reproduction and the paper.

Any experiments that are only introduced in the Appendix are not considered core contributions of the paper, and so are out of scope. This is the case even if the experiment is referenced in the main text. HOWEVER, if an experiment is described in the main body of the text, but some of the details used (e.g. details about a prompt used, or a long mathematical expression) have been put in the appendix, then that experiment is considered still in scope. The reason here is that the experiment is described in the main body of the paper.

RESOURCES
---
You may or may not be provided with a rubric called `rubric.json` in the `{paper_dir}` directory to guide your efforts. The `rubric.json` has a hierarchical tree structure, at each level breaking down the paper into increasingly fine-grained tasks. Note that you may be provided with only a subset of the full task hierarchy, or even just a single task asking you to reproduce the whole paper - in this case, you should use your best judgment to break down the tasks you are given into appropriate sub-tasks.

There is also an addendum file at `{paper_dir}/addendum.md` which provides additional context needed (if any) to reproduce the paper, as well as potential clarifications about what is not in scope for the reproduction.

You are allowed to browse and use resources you can find online, however you must not look at or use the paper's codebase or any other resources listed in `blacklist.txt`, as this is considered cheating.
{library_block}
{whitelist_block}

{app_rules}

SUBMISSION
---
The output of your work must be a git repository containing all the necessary components to reproduce the results of the paper. Your repository should be located at `{workspace_dir}`, and the committed files should not exceed 1GB in size (i.e. you should only commit source code without any heavy artifacts). We will run `git clean -fd` on the submission directory before grading it, so any untracked files will be lost.

Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.

Finally, please also include a README.md file that describes what you were able to achieve in your reproduction attempt and explains how your codebase relates to various parts of the reproduction.

ADDITIONAL NOTES
------
- **Compute**: You don't have access to a GPU.
- **Total Runtime**: You are expected to work until you have spent {time_limit_hours} hours on this task. You may iteratively develop your solution within these constraints, but bear in mind that program runtime also counts towards the time limit. This is a difficult task and we expect a good solution to require using the full time limit.
- **API keys**: You have been granted API keys for services necessary for replicating your paper. The API keys are available to you at `{agent_env_path}`. You can assume that these same API keys will already be set as environment variables in the reproduction environment when we evaluate your submission, so you do not need to set or export them in your submission.
- **Per-bash timeout (10 minutes)**: Each bash command you issue is capped at 10 minutes (600 seconds) of wall-clock time before the runner kills it and returns a TimeoutExpired error. The cap applies to one command at a time, not to the whole task.
- **Grading is code-only**: graders read your files; they do NOT execute your code. Validate by reading the source, running short import/dry-run checks (`python -c "from agent import PPOAgent; PPOAgent(obs_dim=4, action_dim=2)"`), or `--max-epochs 1` smoke runs. Do NOT spend the budget on full training runs to "verify" correctness.
- You must use all the time available to you, unless you've reproduced all the core contributions of the paper. Do not stop until you've reproduced them.
- Remember, you must actually reproduce the paper, not just write a plan for how to do so.
"""


def build_prompt_from_task(
    paper: dict[str, Any],
    *,
    workspace_dir: str,
    paper_dir: str,
    agent_env_path: str,
    library_dir: str | None = None,
    whitelist_block: str = "",
    time_limit_hours: float = 2.0,
) -> str:
    """Render the upstream-verbatim user prompt for PaperbenchCodingAgent.

    Args:
        paper: task entry from ``task_lookup(paper_id)``. Accepted for API
              parity with the webgen helper; the verbatim prompt does not
              consume any of its fields (paper text is read via bash).
        workspace_dir: absolute path of the submission target.
        paper_dir: absolute path of the (whitelisted) paper snapshot.
        agent_env_path: path the prompt advertises for API keys.
        library_dir: when set, render
              :data:`prompts.paperbench.common.LIBRARY_BLOCK` against it
              and splice into the RESOURCES section. ``None`` → empty
              slot (baseline).
        whitelist_block: optional ``[Whitelist]`` block emitted by
              ``utils.whitelist.render_for_prompt()``.
        time_limit_hours: rendered into the ADDITIONAL NOTES
              "Total Runtime" bullet.
    """
    del paper  # unused — kept for API parity with the webgen helper
    library_block = (
        LIBRARY_BLOCK.format(library_dir=library_dir) if library_dir else ""
    )
    return PAPERBENCH_USER_PROMPT.format(
        paper_dir=paper_dir,
        workspace_dir=workspace_dir,
        agent_env_path=agent_env_path,
        library_block=library_block,
        whitelist_block=whitelist_block,
        app_rules=APP_RULES,
        time_limit_hours=round(time_limit_hours, 2),
    )
