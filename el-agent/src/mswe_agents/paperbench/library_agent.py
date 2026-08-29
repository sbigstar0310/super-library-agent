"""PaperbenchLibraryAgent — sla_naive unified extract + apply (single-shot).

One sub-agent walks every cumulative paper submission, builds the
Python `lib/`, and rewrites the submissions in place so they import
from it. No two-phase split, no candidate plumbing required.

Mirrors `mswe_agents/webgen/library_agent.py`. Differences:
  - mount layout reuses the `/home/tasks/<tid>/{paper,submission}/`
    namespace (same as global_extract). Webgen flattens submissions
    under `/home/apps/<tid>/` because it needs short relative-import
    paths for `../../lib/src/index.js`; paperbench imports go through
    PYTHONPATH so flattening adds no value.
  - per-task paper/ snapshots are co-mounted (ro) so the agent can read
    paper context while choosing which primitives to lift.

Workspace layout (host paths, populated by run-driver):
    {workspace_root}/
        __library__/agent.env
        {library_dir}/                       — RW; agent writes lib here
        (caller-staged) <tid>/{paper(ro), submission(rw)}/

Mounts (docker mode):
    {library_dir}                      → /home/lib                    (rw)
    {host_dir}/agent.env               → /home/agent.env              (file)
    apps[tid]                          → /home/tasks/<tid>/submission (rw)
    paper_snapshot_for(apps[tid])      → /home/tasks/<tid>/paper      (ro)

Single-shot pattern: caller invokes `run("__library__")` exactly once
per round. Optionally takes ward-clustering candidates via the
``candidate_strategy="embed"`` knob; ``"none"`` skips candidates.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, Literal

from mswe_agents._lib_summary import summarize_lib_dir
from mswe_agents.base_coding_agent import BaseCodingAgent
from prompts.paperbench.library_agent import (
    PAPERBENCH_LIBRARY_SYSTEM_PROMPT,
    build_library_user_prompt,
)
from utils.candidates import Strategy, get_extract_candidates
from utils.code_index import index_app


__all__ = ["PaperbenchLibraryAgent", "LIBRARY_TASK_ID"]


LIBRARY_TASK_ID = "__library__"

_LIB_GLOBS = ("*.py",)

LibraryCandidateStrategy = Literal["embed", "nl", "none"]


class PaperbenchLibraryAgent(BaseCodingAgent):
    """sla_naive unified extract + apply peer of PaperbenchCodingAgent.

    See `_paper_snapshot_for` for how this agent locates each paper/
    sibling next to a staged submission/.
    """

    def __init__(
        self,
        *,
        workspace_root: str,
        task_lookup: Callable[[str], dict],
        source_apps: dict[str, str],
        candidate_strategy: LibraryCandidateStrategy = "none",
        cluster_distance_threshold: float = 1.2,
        cluster_min_mean_sim: float = 0.50,
        cluster_top_k: int = 10,
        cluster_min_line: int = 5,
        nl_model: str = "gpt-5.4-nano",
        nl_pick_model: str | None = None,
        **base_kwargs,
    ):
        if not base_kwargs.get("library_dir"):
            raise ValueError(
                "PaperbenchLibraryAgent requires explicit library_dir "
                "(e.g. round_N/apply/lib/)."
            )
        super().__init__(**base_kwargs)
        self.workspace_root = os.path.abspath(workspace_root)
        self.task_lookup = task_lookup
        self.source_apps = dict(source_apps)
        self.candidate_strategy: LibraryCandidateStrategy = candidate_strategy
        self.cluster_distance_threshold = cluster_distance_threshold
        self.cluster_min_mean_sim = cluster_min_mean_sim
        self.cluster_top_k = cluster_top_k
        self.cluster_min_line = cluster_min_line
        self.nl_model = nl_model
        self.nl_pick_model = nl_pick_model
        os.makedirs(self.workspace_root, exist_ok=True)
        os.makedirs(self.library_dir, exist_ok=True)

    # ---- helpers ----------------------------------------------------------

    def _paper_snapshot_for(self, submission_dir: str) -> str:
        return os.path.join(os.path.dirname(submission_dir), "paper")

    # ---- BaseCodingAgent contract -----------------------------------------

    def system_prompt(self) -> str:
        return PAPERBENCH_LIBRARY_SYSTEM_PROMPT

    def setup_workspace(self, task_id: str) -> dict[str, str]:
        host_dir = os.path.join(self.workspace_root, LIBRARY_TASK_ID)
        os.makedirs(host_dir, exist_ok=True)
        lib_dir = self.library_dir

        missing_sub = [
            tid for tid, p in self.source_apps.items()
            if not os.path.isdir(p)
        ]
        missing_paper = [
            tid for tid, p in self.source_apps.items()
            if not os.path.isdir(self._paper_snapshot_for(p))
        ]
        if missing_sub or missing_paper:
            raise FileNotFoundError(
                f"PaperbenchLibraryAgent expects staged submission/ AND paper/ "
                f"for each task. Missing submissions: {missing_sub}, missing "
                f"papers: {missing_paper}."
            )

        if self.docker_image:
            mount_spec: list[tuple[str, str, str]] = [
                (lib_dir, "/home/lib", ""),
                (os.path.join(host_dir, "agent.env"), "/home/agent.env", ""),
            ]
            for tid, sub_path in self.source_apps.items():
                mount_spec.append(
                    (sub_path, f"/home/tasks/{tid}/submission", "")
                )
                mount_spec.append(
                    (self._paper_snapshot_for(sub_path),
                     f"/home/tasks/{tid}/paper", "ro")
                )
            return {
                "workspace_dir": lib_dir,
                "host_dir": host_dir,
                "agent_cwd": "/home/lib",
                "mount_spec": mount_spec,
                "extra_run_args": ["--network=none"],
                "forward_env": [],
            }

        return {
            "workspace_dir": lib_dir,
            "host_dir": host_dir,
            "agent_cwd": lib_dir,
        }

    def pre_run(self, task_id: str, paths: dict[str, str]) -> None:
        """Embed source submissions for optional ward-clustering candidates."""
        if self.candidate_strategy == "none":
            return
        for tid, sub_path in self.source_apps.items():
            index_app(
                sub_path,
                strategy=self.candidate_strategy,
                app_id=tid,
                nl_model=self.nl_model,
            )

    def build_user_prompt(
        self,
        task_id: str,
        paths: dict[str, str],
        agent_env_path: str,
    ) -> str:
        if self.docker_image:
            apps_for_prompt = {
                tid: f"/home/tasks/{tid}/submission" for tid in self.source_apps
            }
            paper_dirs_for_prompt = {
                tid: f"/home/tasks/{tid}/paper" for tid in self.source_apps
            }
            library_dir_for_prompt = "/home/lib"
        else:
            apps_for_prompt = {
                tid: os.path.abspath(sub) for tid, sub in self.source_apps.items()
            }
            paper_dirs_for_prompt = {
                tid: os.path.abspath(self._paper_snapshot_for(sub))
                for tid, sub in self.source_apps.items()
            }
            library_dir_for_prompt = os.path.abspath(self.library_dir)

        candidates_md = ""
        if self.candidate_strategy != "none":
            host_apps = {
                tid: os.path.abspath(sub)
                for tid, sub in self.source_apps.items()
            }
            strategy: Strategy = self.candidate_strategy  # type: ignore[assignment]
            result = get_extract_candidates(
                strategy,
                app_dirs=host_apps,
                mode="global",
                top_k=self.cluster_top_k,
                min_line=self.cluster_min_line,
                min_mean_sim=self.cluster_min_mean_sim,
                distance_threshold=self.cluster_distance_threshold,
                library_dir=os.path.abspath(self.library_dir),
                nl_model=self.nl_model,
                nl_pick_model=self.nl_pick_model,
            )
            candidates_md = result.markdown

        papers: dict[str, dict] = {}
        for tid in self.source_apps:
            paper = self.task_lookup(tid)
            if not isinstance(paper, dict):
                raise ValueError(
                    f"task_lookup({tid!r}) returned non-dict: "
                    f"{type(paper).__name__}"
                )
            papers[tid] = {**paper, "paper_dir": paper_dirs_for_prompt[tid]}

        existing_lib_summary = summarize_lib_dir(
            Path(self.library_dir), globs=_LIB_GLOBS,
        )

        return build_library_user_prompt(
            papers=papers,
            apps=apps_for_prompt,
            library_dir=library_dir_for_prompt,
            existing_lib_summary=existing_lib_summary,
            library_candidates=candidates_md,
        )

    def output_path_for(
        self, task_id: str, round_num: int, step_num: int, log_phase: str
    ) -> Path:
        return (
            Path(self.log_dir)
            / f"round_{round_num}"
            / log_phase
            / f"task_{LIBRARY_TASK_ID}"
            / f"step_{step_num}.json"
        )

    def agent_env_keys(self) -> dict[str, str]:
        # See coding_agent.py for rationale — no keys under `--network=none`.
        return {}
