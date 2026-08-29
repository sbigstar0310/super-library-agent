"""PaperbenchCodingAgent — paperbench coding peer (baseline + sla_naive + sla_ours).

Subclass of BaseCodingAgent. Generates a paper-replication submission
from a paperbench paper snapshot (paper.md + addendum.md + assets) using
the **upstream-verbatim** code-only iterative prompt (see
`prompts/paperbench/coding_agent.py`).

Workspace layout (host paths, set up by the run-driver):
    {workspace_root}/<paper_id>/
        paper/        — RO snapshot (paper.md, paper.pdf, addendum.md,
                        blacklist.txt, assets/). Caller-managed; agent
                        validates presence.
        submission/   — RW; agent writes the reproduction here
        lib/          — only when library_dir is set (sla_naive / sla_ours r>1)
        agent.env     — API keys + PYTHONPATH for the agent's bash session

Mounts (docker mode, paperbench upstream WORKSPACE_BASE=/home convention):
    {host_dir}/paper/         → /home/paper          (ro)
    {host_dir}/submission/    → /home/submission     (rw)
    {host_dir}/agent.env      → /home/agent.env      (file)
    {library_dir}             → /home/lib            (ro, optional)

Logs written to: {log_dir}/round_{N}/{phase}/task_{paper_id}/step_{N}.json
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Callable

from mswe_agents.base_coding_agent import BaseCodingAgent
from prompts.paperbench.coding_agent import (
    PAPERBENCH_SYSTEM_PROMPT,
    build_prompt_from_task,
)
from utils.whitelist import render_for_prompt


__all__ = ["PaperbenchCodingAgent"]


class PaperbenchCodingAgent(BaseCodingAgent):
    """paperbench coder for baseline + sla_naive + sla_ours coding stage."""

    def __init__(
        self,
        *,
        workspace_root: str,
        task_lookup: Callable[[str], dict],
        time_limit_hours: float = 2.0,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.workspace_root = os.path.abspath(workspace_root)
        # Returns a paper dict with at least {"task_id", "paper_dir"}.
        self.task_lookup = task_lookup
        # Rendered into the ADDITIONAL NOTES "Total Runtime" bullet.
        self.time_limit_hours = time_limit_hours
        os.makedirs(self.workspace_root, exist_ok=True)

    # ---- BaseCodingAgent contract -----------------------------------------

    def system_prompt(self) -> str:
        return PAPERBENCH_SYSTEM_PROMPT

    def setup_workspace(self, task_id: str) -> dict[str, str]:
        host_dir = os.path.join(self.workspace_root, task_id)
        paper_dir = os.path.join(host_dir, "paper")
        submission_dir = os.path.join(host_dir, "submission")
        os.makedirs(host_dir, exist_ok=True)
        os.makedirs(submission_dir, exist_ok=True)

        # Paper snapshot is staged by the run-driver; without it there is
        # nothing to replicate.
        if not os.path.isdir(paper_dir) or not os.listdir(paper_dir):
            raise FileNotFoundError(
                f"PaperbenchCodingAgent expects a paper snapshot at "
                f"{paper_dir} (populated by the run-driver before invocation)."
            )

        lib_host_path = self._stage_library(host_dir)

        if self.docker_image:
            mount_spec: list[tuple[str, str, str]] = [
                (paper_dir, "/home/paper", "ro"),
                (submission_dir, "/home/submission", ""),
                (os.path.join(host_dir, "agent.env"), "/home/agent.env", ""),
            ]
            if lib_host_path:
                mount_spec.append((lib_host_path, "/home/lib", "ro"))
            return {
                "workspace_dir": submission_dir,
                "host_dir": host_dir,
                "agent_cwd": "/home/submission",
                "mount_spec": mount_spec,
                # No network: blocks forbidden pip installs / upstream fetches,
                # pairing with the baked-in package set in
                # docker/paperbench-base/Dockerfile. Note the upstream prompt
                # still mentions browsing online (prompt/runtime mismatch,
                # see paperbench-notes.md).
                "extra_run_args": ["--network=none"],
                "forward_env": [],
            }

        return {
            "workspace_dir": submission_dir,
            "host_dir": host_dir,
            "agent_cwd": submission_dir,
        }

    def build_user_prompt(
        self,
        task_id: str,
        paths: dict[str, str],
        agent_env_path: str,
    ) -> str:
        paper = self.task_lookup(task_id)
        if not isinstance(paper, dict):
            raise ValueError(
                f"task_lookup({task_id!r}) returned non-dict: {type(paper).__name__}"
            )

        if self.docker_image:
            workspace_dir_for_prompt = "/home/submission"
            paper_dir_for_prompt = "/home/paper"
            library_dir_for_prompt = "/home/lib" if self.library_dir else None
        else:
            workspace_dir_for_prompt = os.path.abspath(paths["workspace_dir"])
            paper_dir_for_prompt = os.path.join(
                os.path.abspath(paths["host_dir"]), "paper"
            )
            library_dir_for_prompt = (
                os.path.abspath(self.library_dir) if self.library_dir else None
            )

        return build_prompt_from_task(
            paper,
            workspace_dir=workspace_dir_for_prompt,
            paper_dir=paper_dir_for_prompt,
            agent_env_path=agent_env_path,
            library_dir=library_dir_for_prompt,
            whitelist_block=render_for_prompt(),
            time_limit_hours=self.time_limit_hours,
        )

    def output_path_for(
        self, task_id: str, round_num: int, step_num: int, log_phase: str
    ) -> Path:
        return (
            Path(self.log_dir)
            / f"round_{round_num}"
            / log_phase
            / f"task_{task_id}"
            / f"step_{step_num}.json"
        )

    def agent_env_keys(self) -> dict[str, str]:
        # No keys forwarded: `--network=none` makes them unusable, and
        # surfacing them in agent.env is a leak surface (models that
        # `cat agent.env` send the key into the provider context). agent.env
        # still carries PYTHONPATH=/home for `import lib`.
        return {}
