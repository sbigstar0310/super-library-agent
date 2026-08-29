"""WebgenLocalExtractAgent — sla_ours per-app intra-extract + apply.

Reads ONE app's submission, surfaces intra-app duplicate clusters, and
consolidates them into `<submission>/src/local_lib/`, rewriting call sites in
the same run. Mirrors `ral_local_extract_agent.py` minus the yaml-driven
package-name resolver — webgen apps always use `src/local_lib/`.

Workspace layout (host paths):
    {workspace_root}/<task_id>/
        submission/   — RW (incl. new src/local_lib/)
        agent.env
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Callable

from mswe_agents._lib_summary import summarize_lib_dir
from mswe_agents.base_coding_agent import BaseCodingAgent
from prompts.common import LOCAL_EXTRACT_SYSTEM_PROMPT
from prompts.webgen.local_extract_agent import build_local_extract_user_prompt
from utils.candidates import Strategy, get_extract_candidates
from utils.code_index import index_app


__all__ = ["WebgenLocalExtractAgent"]


_LIB_GLOBS = ("*.jsx", "*.js", "*.css")


class WebgenLocalExtractAgent(BaseCodingAgent):
    """Per-app intra-extract peer for sla_ours."""

    def __init__(
        self,
        *,
        workspace_root: str,
        task_lookup: Callable[[str], dict],
        cluster_distance_threshold: float = 1.0,
        cluster_min_mean_sim: float = 0.55,
        cluster_top_k: int = 12,
        cluster_min_line: int = 5,
        candidate_strategy: Strategy = "nl",
        nl_model: str = "gpt-5.4-nano",
        nl_pick_model: str | None = None,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.workspace_root = os.path.abspath(workspace_root)
        self.task_lookup = task_lookup
        self.cluster_distance_threshold = cluster_distance_threshold
        self.cluster_min_mean_sim = cluster_min_mean_sim
        self.cluster_top_k = cluster_top_k
        self.cluster_min_line = cluster_min_line
        self.candidate_strategy = candidate_strategy
        self.nl_model = nl_model
        self.nl_pick_model = nl_pick_model
        os.makedirs(self.workspace_root, exist_ok=True)

    # ---- BaseCodingAgent contract -----------------------------------------

    def system_prompt(self) -> str:
        return LOCAL_EXTRACT_SYSTEM_PROMPT

    def setup_workspace(self, task_id: str) -> dict[str, str]:
        host_dir = os.path.join(self.workspace_root, task_id)
        submission_dir = os.path.join(host_dir, "submission")
        os.makedirs(host_dir, exist_ok=True)

        if not os.path.isdir(submission_dir) or not os.listdir(submission_dir):
            raise FileNotFoundError(
                f"LocalExtract expects a pre-staged submission at "
                f"{submission_dir}."
            )

        # Deliberately do NOT pre-create src/local_lib/: a catch-all dir biases
        # the agent toward dumping into it, the behavior v3 is designed to avoid.
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
                # Network needed for `npm install`. Cheating gated in prompt.
                "forward_env": [],
            }

        return {
            "workspace_dir": submission_dir,
            "host_dir": host_dir,
            "agent_cwd": submission_dir,
        }

    def pre_run(self, task_id: str, paths: dict[str, str]) -> None:
        """Embed the submission so intra-app clustering can run."""
        submission_dir = paths["workspace_dir"]
        index_app(
            submission_dir,
            strategy=self.candidate_strategy,
            app_id=task_id,
            nl_model=self.nl_model,
        )

    def build_user_prompt(
        self,
        task_id: str,
        paths: dict[str, str],
        agent_env_path: str,
    ) -> str:
        submission_host = Path(paths["workspace_dir"])

        if self.docker_image:
            workspace_dir_for_prompt = "/home/submission"
        else:
            workspace_dir_for_prompt = os.path.abspath(submission_host)

        host_apps = {task_id: os.path.abspath(submission_host)}
        result = get_extract_candidates(
            self.candidate_strategy,
            app_dirs=host_apps,
            mode="local",
            top_k=self.cluster_top_k,
            min_line=self.cluster_min_line,
            min_mean_sim=self.cluster_min_mean_sim,
            distance_threshold=self.cluster_distance_threshold,
            library_dir=(
                os.path.abspath(self.library_dir) if self.library_dir else None
            ),
            nl_model=self.nl_model,
            nl_pick_model=self.nl_pick_model,
        )
        candidates_md = result.markdown

        task = self.task_lookup(task_id)
        if not isinstance(task, dict):
            raise ValueError(
                f"task_lookup({task_id!r}) returned non-dict: "
                f"{type(task).__name__}"
            )

        existing_global_summary = summarize_lib_dir(
            Path(self.library_dir) if self.library_dir else None,
            globs=_LIB_GLOBS,
            label="Global ui-lib",
            show_line_counts=False,
            missing_msg="(none — no carry-forward global library this round)",
            empty_msg="(empty — no carry-forward symbols)",
        )

        return build_local_extract_user_prompt(
            task=task,
            task_id=task_id,
            workspace_dir=workspace_dir_for_prompt,
            local_extract_candidates=candidates_md,
            existing_global_lib_summary=existing_global_summary,
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
