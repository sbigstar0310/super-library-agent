"""WebgenLibraryAgent — sla_naive unified extract + apply (single-shot).

One sub-agent walks every cumulative app submission, builds `ui-lib`, and
rewrites the apps in place to import from it — no two-phase split, no candidate
plumbing required.

Workspace layout (host paths, populated by run-driver):
    {workspace_root}/
        __library__/agent.env
        {library_dir}/                                 — RW; agent writes lib
        (caller-staged) {apps_root}/<tid>/submission/  — RW mounts (per app)

Single-shot: caller invokes `run("__library__")` once per round. Optionally
takes ward-clustering candidates via ``candidate_strategy="embed"``; ``"none"``
skips them.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, Literal

from mswe_agents._lib_summary import summarize_lib_dir
from mswe_agents.base_coding_agent import BaseCodingAgent
from prompts.webgen.library_agent import (
    WEBGEN_LIBRARY_SYSTEM_PROMPT,
    build_library_user_prompt,
)
from utils.candidates import Strategy, get_extract_candidates
from utils.code_index import index_app


__all__ = ["WebgenLibraryAgent", "LIBRARY_TASK_ID"]


LIBRARY_TASK_ID = "__library__"

_LIB_GLOBS = ("*.jsx", "*.js", "*.css")

LibraryCandidateStrategy = Literal["embed", "nl", "none"]


class WebgenLibraryAgent(BaseCodingAgent):
    """sla_naive unified extract + apply peer of WebgenCodingAgent."""

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
                "WebgenLibraryAgent requires explicit library_dir "
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

    # ---- BaseCodingAgent contract -----------------------------------------

    def system_prompt(self) -> str:
        return WEBGEN_LIBRARY_SYSTEM_PROMPT

    def setup_workspace(self, task_id: str) -> dict[str, str]:
        host_dir = os.path.join(self.workspace_root, LIBRARY_TASK_ID)
        os.makedirs(host_dir, exist_ok=True)
        lib_dir = self.library_dir

        missing = [
            tid for tid, p in self.source_apps.items()
            if not os.path.isdir(p)
        ]
        if missing:
            raise FileNotFoundError(
                f"WebgenLibraryAgent expects staged submissions but the "
                f"following are missing: {missing}."
            )

        if self.docker_image:
            # Layout chosen so the agent computes `../../lib/src/index.js`-shaped
            # relative imports, matching WebgenApplyAgent and the eval_webgen.sh
            # eval_dir staging. Each host `submission/` is mounted with its
            # CONTENTS at `/home/apps/<tid>/` (no extra `submission/` layer), so
            # package.json / vite.config.js / src/ sit directly there; the lib
            # lives alongside at `/home/apps/lib/`.
            mount_spec: list[tuple[str, str, str]] = [
                (lib_dir, "/home/apps/lib", ""),
                (os.path.join(host_dir, "agent.env"), "/home/agent.env", ""),
            ]
            # Apps mounted RW so the single library-agent run can edit in place.
            for tid, sub_path in self.source_apps.items():
                mount_spec.append(
                    (sub_path, f"/home/apps/{tid}", "")
                )
            return {
                "workspace_dir": lib_dir,
                "host_dir": host_dir,
                "agent_cwd": "/home/apps/lib",
                "mount_spec": mount_spec,
                # Network needed for `npm install` during per-app verification.
                # Cheating gated in prompt.
                "forward_env": [],
            }

        return {
            "workspace_dir": lib_dir,
            "host_dir": host_dir,
            "agent_cwd": lib_dir,
        }

    def pre_run(self, task_id: str, paths: dict[str, str]) -> None:
        """Embed source apps for optional ward-clustering candidates."""
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
                tid: f"/home/apps/{tid}"
                for tid in self.source_apps
            }
            library_dir_for_prompt = "/home/apps/lib"
        else:
            apps_for_prompt = {
                tid: os.path.abspath(sub) for tid, sub in self.source_apps.items()
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

        tasks: dict[str, dict] = {}
        for tid in self.source_apps:
            task = self.task_lookup(tid)
            if not isinstance(task, dict):
                raise ValueError(
                    f"task_lookup({tid!r}) returned non-dict: "
                    f"{type(task).__name__}"
                )
            tasks[tid] = task

        existing_lib_summary = summarize_lib_dir(
            Path(self.library_dir), globs=_LIB_GLOBS,
        )

        return build_library_user_prompt(
            tasks=tasks,
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
        return {}
