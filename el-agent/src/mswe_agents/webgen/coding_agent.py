"""WebgenCodingAgent — WebGen-Bench coding peer (baseline + sla_naive + sla_ours).

Generates a React+Vite app from a single `instruction` string in
`data/WebGen-Bench/data/test.jsonl`. Mirrors `ral_coding_agent.py` minus the
api_contract / yaml-load machinery (webgen tasks are JSONL rows).

Workspace layout (host paths):
    {workspace_root}/<task_id>/
        submission/   — agent writes the Vite app here
        lib/          — only when library_dir is set (sla_naive / sla_ours r>1)
        agent.env

Prompts live in `prompts/webgen/coding_agent.py`.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Callable

from mswe_agents.base_coding_agent import BaseCodingAgent
from prompts.webgen.coding_agent import (
    WEBGEN_SYSTEM_PROMPT,
    build_prompt_from_task,
)
from prompts.webgen.common import LIBRARY_BLOCK


__all__ = ["WebgenCodingAgent"]


class WebgenCodingAgent(BaseCodingAgent):
    """WebGen-Bench coder for baseline + sla_naive / sla_ours coding stages.

    Adds `workspace_root` (parent dir under which `<task_id>/{submission,lib?}/`
    is materialized) and `task_lookup` (task_id → loaded JSONL row) on top of
    BaseCodingAgent.
    """

    def __init__(
        self,
        *,
        workspace_root: str,
        task_lookup: Callable[[str], dict],
        layout_specs_dir: str | None = None,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.workspace_root = os.path.abspath(workspace_root)
        self.task_lookup = task_lookup
        # When set, build_user_prompt injects `<dir>/<task_id>.md` as
        # task["layout_spec"]; an absent file falls back to vanilla baseline.
        self.layout_specs_dir = (
            os.path.abspath(layout_specs_dir) if layout_specs_dir else None
        )
        os.makedirs(self.workspace_root, exist_ok=True)

    # ---- BaseCodingAgent contract -----------------------------------------

    def system_prompt(self) -> str:
        return WEBGEN_SYSTEM_PROMPT

    def setup_workspace(self, task_id: str) -> dict[str, str]:
        host_dir = os.path.join(self.workspace_root, task_id)
        submission_dir = os.path.join(host_dir, "submission")
        os.makedirs(host_dir, exist_ok=True)
        os.makedirs(submission_dir, exist_ok=True)

        # Stage lib/ next to submission/ for a carry-forward library;
        # `_stage_library` returns None when self.library_dir is unset.
        lib_host_path = self._stage_library(host_dir)

        if self.docker_image:
            mount_spec: list[tuple[str, str, str]] = [
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
                # Network is REQUIRED for `npm install`; WebFetch / WebSearch /
                # sibling-task cheating are blocked at the prompt level.
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
        task = self.task_lookup(task_id)
        if not isinstance(task, dict):
            raise ValueError(
                f"task_lookup({task_id!r}) returned non-dict: {type(task).__name__}"
            )

        # Optional layout-spec augment: inject a per-task spec file as
        # task["layout_spec"] when present, else no change.
        if self.layout_specs_dir:
            spec_path = Path(self.layout_specs_dir) / f"{task_id}.md"
            if spec_path.is_file():
                task = {**task, "layout_spec": spec_path.read_text()}

        if self.docker_image:
            workspace_dir_for_prompt = "/home/submission"
        else:
            workspace_dir_for_prompt = os.path.abspath(paths["workspace_dir"])

        library_block = ""
        if self.library_dir:
            lib_dir_in_prompt = (
                "/home/lib" if self.docker_image
                else os.path.abspath(self.library_dir)
            )
            library_block = LIBRARY_BLOCK.format(library_dir=lib_dir_in_prompt)

        return build_prompt_from_task(
            task,
            workspace_dir=workspace_dir_for_prompt,
            library_block=library_block,
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
        # No LLM keys run inside the container (npm/vite/bash only); empty
        # agent.env keeps secrets out of the trajectory logs.
        return {}
