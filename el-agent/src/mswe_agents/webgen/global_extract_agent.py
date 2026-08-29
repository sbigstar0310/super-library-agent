"""WebgenGlobalExtractAgent — sla_ours cross-app library extractor.

Reads N RO-mounted app submissions at `/home/tasks/<task_id>/submission/`
and writes a JS/JSX library at `/home/lib/`. Mirrors
`ral_global_extract_agent.py` minus api_contract / yaml-load.

Workspace layout (host paths):
    {workspace_root}/
        __extract__/agent.env
        {library_dir}/                           — RW; agent writes here
        (caller-staged) tasks/<tid>/submission/  — RO mounts (per source app)

Single-shot: BaseFullRun calls run("__extract__") exactly once.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, Literal

from mswe_agents._lib_summary import summarize_lib_dir
from mswe_agents.base_coding_agent import BaseCodingAgent
from prompts.common import EXTRACT_SYSTEM_PROMPT
from prompts.webgen.global_extract_agent import build_extract_user_prompt
from utils.candidates import PrepEntry, Strategy, get_extract_candidates
from utils.code_index import index_app


ExtractCandidateStrategy = Literal["embed", "nl", "none"]


__all__ = ["WebgenGlobalExtractAgent", "EXTRACT_TASK_ID"]


EXTRACT_TASK_ID = "__extract__"

# Summarizer extensions, matching the cc-exp ui-lib layout: components/*.jsx
# + .css, hooks/utils/services/*.js.
_LIB_GLOBS = ("*.jsx", "*.js", "*.css")


class WebgenGlobalExtractAgent(BaseCodingAgent):
    """sla_ours extract peer of WebgenCodingAgent / WebgenApplyAgent."""

    def __init__(
        self,
        *,
        workspace_root: str,
        task_lookup: Callable[[str], dict],
        source_apps: dict[str, str],
        top_k: int = 20,
        min_similarity: float = 0.5,
        min_line: int = 5,
        cluster_distance_threshold: float = 1.2,
        cluster_min_mean_sim: float = 0.50,
        cluster_top_k: int = 10,
        candidate_strategy: ExtractCandidateStrategy = "nl",
        nl_model: str = "gpt-5.4-nano",
        nl_pick_model: str | None = None,
        **base_kwargs,
    ):
        if not base_kwargs.get("library_dir"):
            raise ValueError(
                "WebgenGlobalExtractAgent requires explicit library_dir "
                "(phase-direct location, e.g. round_N/extract/lib/)."
            )
        super().__init__(**base_kwargs)
        self.workspace_root = os.path.abspath(workspace_root)
        self.task_lookup = task_lookup
        self.source_apps = dict(source_apps)
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.min_line = min_line
        self.cluster_distance_threshold = cluster_distance_threshold
        self.cluster_min_mean_sim = cluster_min_mean_sim
        self.cluster_top_k = cluster_top_k
        self.candidate_strategy = candidate_strategy
        self.nl_model = nl_model
        self.nl_pick_model = nl_pick_model
        self._last_extract_prep: list[PrepEntry] = []
        os.makedirs(self.workspace_root, exist_ok=True)
        os.makedirs(self.library_dir, exist_ok=True)

    # ---- BaseCodingAgent contract -----------------------------------------

    def system_prompt(self) -> str:
        return EXTRACT_SYSTEM_PROMPT

    def setup_workspace(self, task_id: str) -> dict[str, str]:
        host_dir = os.path.join(self.workspace_root, EXTRACT_TASK_ID)
        os.makedirs(host_dir, exist_ok=True)
        lib_dir = self.library_dir

        missing = [
            tid for tid, p in self.source_apps.items()
            if not os.path.isdir(p)
        ]
        if missing:
            raise FileNotFoundError(
                f"WebgenGlobalExtractAgent expects staged submissions but "
                f"the following are missing: {missing}."
            )

        if self.docker_image:
            mount_spec: list[tuple[str, str, str]] = [
                (lib_dir, "/home/lib", ""),
                (os.path.join(host_dir, "agent.env"), "/home/agent.env", ""),
            ]
            for tid, sub_path in self.source_apps.items():
                mount_spec.append(
                    (sub_path, f"/home/tasks/{tid}/submission", "ro")
                )
            return {
                "workspace_dir": lib_dir,
                "host_dir": host_dir,
                "agent_cwd": "/home/lib",
                "mount_spec": mount_spec,
                # Network needed (lib build verification may invoke npm).
                # Cheating gated in prompt.
                "forward_env": [],
            }

        return {
            "workspace_dir": lib_dir,
            "host_dir": host_dir,
            "agent_cwd": lib_dir,
        }

    def pre_run(self, task_id: str, paths: dict[str, str]) -> None:
        """Embed each staged source app so cross-app clustering can run."""
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
                tid: f"/home/tasks/{tid}/submission"
                for tid in self.source_apps
            }
            library_dir_for_prompt = "/home/lib"
        else:
            apps_for_prompt = {
                tid: os.path.abspath(sub) for tid, sub in self.source_apps.items()
            }
            library_dir_for_prompt = os.path.abspath(self.library_dir)

        if self.candidate_strategy == "none":
            candidates_md = (
                "(no candidates provided — discover shared patterns by reading "
                "the source apps via bash.)"
            )
            self._last_extract_prep = []
        else:
            host_apps = {
                tid: os.path.abspath(sub) for tid, sub in self.source_apps.items()
            }
            strategy: Strategy = self.candidate_strategy  # type: ignore[assignment]
            result = get_extract_candidates(
                strategy,
                app_dirs=host_apps,
                mode="global",
                top_k=self.cluster_top_k,
                min_line=self.min_line,
                min_mean_sim=self.cluster_min_mean_sim,
                distance_threshold=self.cluster_distance_threshold,
                library_dir=os.path.abspath(self.library_dir),
                nl_model=self.nl_model,
                nl_pick_model=self.nl_pick_model,
            )
            candidates_md = result.markdown
            self._last_extract_prep = result.prep

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

        return build_extract_user_prompt(
            tasks=tasks,
            apps=apps_for_prompt,
            library_dir=library_dir_for_prompt,
            extract_candidates=candidates_md,
            existing_lib_summary=existing_lib_summary,
        )

    def output_path_for(
        self, task_id: str, round_num: int, step_num: int, log_phase: str
    ) -> Path:
        return (
            Path(self.log_dir)
            / f"round_{round_num}"
            / log_phase
            / f"task_{EXTRACT_TASK_ID}"
            / f"step_{step_num}.json"
        )

    def agent_env_keys(self) -> dict[str, str]:
        return {}
