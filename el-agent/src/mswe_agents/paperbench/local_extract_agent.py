"""PaperbenchLocalExtractAgent — sla_ours per-paper intra-extract.

Subclass of BaseCodingAgent. Reads ONE paper's submission, surfaces
intra-codebase duplicate clusters, and consolidates them into whatever
spot fits the submission's existing layout (the v3 placement-autonomous
design — agent does NOT receive a forced `local_lib/` directory).

Mirrors `mswe_agents/webgen/local_extract_agent.py`.

Workspace layout (host paths):
    {workspace_root}/<paper_id>/
        paper/       — RO snapshot (read for context)
        submission/  — RW (incl. any new helper modules)
        agent.env

Mounts (docker mode):
    {host_dir}/paper/        → /home/paper          (ro)
    {host_dir}/submission/   → /home/submission     (rw)
    {host_dir}/agent.env     → /home/agent.env      (file)
    {library_dir}            → /home/lib            (ro, optional)
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Callable

from mswe_agents._lib_summary import summarize_lib_dir
from mswe_agents.base_coding_agent import BaseCodingAgent
from prompts.common import LOCAL_EXTRACT_SYSTEM_PROMPT
from prompts.paperbench.local_extract_agent import build_local_extract_user_prompt
from utils.candidates import Strategy, get_extract_candidates
from utils.code_index import index_app


__all__ = ["PaperbenchLocalExtractAgent"]


_LIB_GLOBS = ("*.py",)


class PaperbenchLocalExtractAgent(BaseCodingAgent):
    """Per-paper intra-extract peer for sla_ours."""

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
        paper_dir = os.path.join(host_dir, "paper")
        submission_dir = os.path.join(host_dir, "submission")
        os.makedirs(host_dir, exist_ok=True)

        if not os.path.isdir(submission_dir) or not os.listdir(submission_dir):
            raise FileNotFoundError(
                f"LocalExtract expects a pre-staged submission at "
                f"{submission_dir}."
            )
        if not os.path.isdir(paper_dir) or not os.listdir(paper_dir):
            raise FileNotFoundError(
                f"LocalExtract expects a paper snapshot at {paper_dir}."
            )

        # v3 policy: no pre-created catch-all helper dir — the agent integrates
        # helpers into the existing layout (see prompts.common.local_extract).

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
                "extra_run_args": ["--network=none"],
                "forward_env": [],
            }

        return {
            "workspace_dir": submission_dir,
            "host_dir": host_dir,
            "agent_cwd": submission_dir,
        }

    def pre_run(self, task_id: str, paths: dict[str, str]) -> None:
        """Embed the submission so intra-paper clustering can run."""
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
            paper_dir_for_prompt = "/home/paper"
        else:
            workspace_dir_for_prompt = os.path.abspath(submission_host)
            paper_dir_for_prompt = os.path.join(
                os.path.abspath(paths["host_dir"]), "paper"
            )

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

        paper = self.task_lookup(task_id)
        if not isinstance(paper, dict):
            raise ValueError(
                f"task_lookup({task_id!r}) returned non-dict: "
                f"{type(paper).__name__}"
            )
        # Point paper_dir at the snapshot the agent can read for THIS run.
        paper = {**paper, "paper_dir": paper_dir_for_prompt}

        existing_global_summary = summarize_lib_dir(
            Path(self.library_dir) if self.library_dir else None,
            globs=_LIB_GLOBS,
            label="Global lib",
            show_line_counts=False,
            missing_msg="(none — no carry-forward global library this round)",
            empty_msg="(empty — no carry-forward symbols)",
        )

        return build_local_extract_user_prompt(
            paper,
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
        # See coding_agent.py for rationale — no keys under `--network=none`.
        return {}
