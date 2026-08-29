"""WebgenEditAgent — maintenance patch agent over a pre-staged workspace.

Runs ONE patch session per call:
- Protocol B ("suite"): the workspace root holds the whole suite
  (tasks/<id>/submission for every target app, plus lib/ when the source
  condition has one). task_id is the suite name (e.g. "c13").
- Protocol C ("single"): the workspace root holds one app. task_id is the
  app id (e.g. "000027").

The workspace is prepared by run/webgen_maintenance_run.py (copy from backup,
lib-mirror symlinks, git init). This agent only mounts and edits it. The
prompt is method-neutral — see prompts/webgen/edit_agent.py.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Literal

from mswe_agents.base_coding_agent import BaseCodingAgent
from prompts.webgen.edit_agent import (
    EDIT_SYSTEM_PROMPT,
    build_edit_user_prompt_single,
    build_edit_user_prompt_suite,
)

Protocol = Literal["suite", "single"]

__all__ = ["WebgenEditAgent"]


class WebgenEditAgent(BaseCodingAgent):
    """Suite- or app-level maintenance patcher (Protocol B / C)."""

    def __init__(
        self,
        *,
        workspace_root: str,
        protocol: Protocol,
        policy_text: str,
        behaviors: dict[str, str],
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.workspace_root = os.path.abspath(workspace_root)
        self.protocol = protocol
        self.policy_text = policy_text
        # Protocol B: {app_id: behavior} for the whole suite.
        # Protocol C: single-entry dict {app_id: behavior}.
        self.behaviors = behaviors

    # ---- BaseCodingAgent contract -----------------------------------------

    def system_prompt(self) -> str:
        return EDIT_SYSTEM_PROMPT

    def instance_template(self) -> str:
        """Reframe the generic 'Please solve this issue:' opening as a
        maintenance policy application (this is a patch task, not a bug fix)."""
        base = super().instance_template()
        return base.replace(
            "Please solve this issue: {{task}}",
            "Please apply this policy update: {{task}}",
            1,
        )

    def setup_workspace(self, task_id: str) -> dict[str, str]:
        host_dir = self.workspace_root
        if not os.path.isdir(host_dir):
            raise FileNotFoundError(
                f"WebgenEditAgent expects a pre-staged workspace at {host_dir} "
                f"(populated by run/webgen_maintenance_run.py)."
            )

        if self.protocol == "suite":
            tasks_dir = os.path.join(host_dir, "tasks")
            if not os.path.isdir(tasks_dir):
                raise FileNotFoundError(f"missing {tasks_dir} for protocol B")
        else:
            if not os.path.isdir(os.path.join(host_dir, "submission")):
                raise FileNotFoundError(
                    f"missing {host_dir}/submission for protocol C"
                )

        if self.docker_image:
            # Single mount: workspace root shadows /home so both the absolute
            # /home/lib imports (sla-ours apps) and the relative
            # tasks/<id>/lib -> ../../lib symlinks resolve in-container.
            # Exclude .git from the agent's view? git metadata stays visible
            # but the prompt never mentions it; diffs are taken on the host.
            mount_spec: list[tuple[str, str, str]] = [
                (host_dir, "/home", ""),
            ]
            return {
                "workspace_dir": host_dir,
                "host_dir": host_dir,
                "agent_cwd": "/home",
                "mount_spec": mount_spec,
                "forward_env": [],
            }

        return {
            "workspace_dir": host_dir,
            "host_dir": host_dir,
            "agent_cwd": host_dir,
        }

    def build_user_prompt(
        self, task_id: str, paths: dict[str, str], agent_env_path: str
    ) -> str:
        root = "/home" if self.docker_image else paths["workspace_dir"]
        if self.protocol == "suite":
            return build_edit_user_prompt_suite(
                policy=self.policy_text,
                behaviors=self.behaviors,
                root=root,
            )
        (app_id,) = self.behaviors
        return build_edit_user_prompt_single(
            policy=self.policy_text,
            behavior=self.behaviors[app_id],
            root=root,
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
        return {}
