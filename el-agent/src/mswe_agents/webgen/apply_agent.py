"""WebgenApplyAgent — sla_ours per-app refactor against the global ui-lib.

Refactors a pre-staged app submission to import from a pre-built `lib/`
package. Mirrors `ral_apply_agent.py` minus api_contract.

Workspace layout (host paths):
    {workspace_root}/<task_id>/
        submission/ — pre-seeded; agent edits in place
        lib/        — copy of library_dir staged by _stage_library
        agent.env
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, Literal

from mswe_agents.base_coding_agent import BaseCodingAgent
from prompts.common import APPLY_SYSTEM_PROMPT
from prompts.webgen.apply_agent import (
    build_apply_user_prompt,
)
from utils.candidates import PrepEntry, Strategy, get_apply_candidates
from utils.code_index import index_app
from utils.extract_map import filter_for_app, read_extract_map


ApplyCandidateStrategy = Literal["embed", "nl", "none"]


__all__ = ["WebgenApplyAgent"]


class WebgenApplyAgent(BaseCodingAgent):
    """sla_ours apply peer of WebgenCodingAgent."""

    def __init__(
        self,
        *,
        workspace_root: str,
        task_lookup: Callable[[str], dict],
        retrieve_top_k: int = 10,
        retrieve_min_line: int = 5,
        retrieve_min_similarity: float = 0.7,
        extract_map: bool = True,
        candidate_strategy: ApplyCandidateStrategy = "nl",
        nl_model: str = "gpt-5.4-nano",
        nl_pick_model: str | None = None,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.workspace_root = os.path.abspath(workspace_root)
        self.task_lookup = task_lookup
        self.retrieve_top_k = retrieve_top_k
        self.retrieve_min_line = retrieve_min_line
        self.retrieve_min_similarity = retrieve_min_similarity
        self.extract_map = extract_map
        self.candidate_strategy = candidate_strategy
        self.nl_model = nl_model
        self.nl_pick_model = nl_pick_model
        # Keyed by task_id so parallel apply runs don't trample each other.
        self._prep_by_task: dict[str, list[PrepEntry]] = {}
        os.makedirs(self.workspace_root, exist_ok=True)

    # ---- BaseCodingAgent contract -----------------------------------------

    def system_prompt(self) -> str:
        return APPLY_SYSTEM_PROMPT

    def setup_workspace(self, task_id: str) -> dict[str, str]:
        if not self.library_dir:
            raise ValueError("WebgenApplyAgent requires library_dir.")

        host_dir = os.path.join(self.workspace_root, task_id)
        submission_dir = os.path.join(host_dir, "submission")
        os.makedirs(host_dir, exist_ok=True)

        if not os.path.isdir(submission_dir) or not os.listdir(submission_dir):
            raise FileNotFoundError(
                f"WebgenApplyAgent expects a pre-seeded submission at "
                f"{submission_dir} (populated by the run-driver from the "
                f"prior coding/extract phase)."
            )

        lib_host_path = self._stage_library(host_dir)

        if self.docker_image:
            mount_spec: list[tuple[str, str, str]] = [
                (submission_dir, "/home/submission", ""),
                (os.path.join(host_dir, "agent.env"), "/home/agent.env", ""),
                (lib_host_path, "/home/lib", "ro"),
            ]
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

    def pre_drive(self) -> None:
        """Embed the source library once, before parallel tasks start.

        Workers share ``self.library_dir``, so embedding it in ``pre_run``
        would race N threads on the same ``embeddings.npz``.
        """
        if not self.library_dir:
            return
        if self.candidate_strategy == "none":
            return
        index_app(
            self.library_dir,
            strategy=self.candidate_strategy,
            nl_model=self.nl_model,
        )

    def pre_run(self, task_id: str, paths: dict[str, str]) -> None:
        """Embed the per-task submission so retrieval can run."""
        if self.candidate_strategy == "none":
            return
        submission_dir = paths["workspace_dir"]
        index_app(
            submission_dir,
            strategy=self.candidate_strategy,
            nl_model=self.nl_model,
        )

    def build_user_prompt(
        self,
        task_id: str,
        paths: dict[str, str],
        agent_env_path: str,
    ) -> str:
        if self.docker_image:
            workspace_dir = "/home/submission"
            library_dir = "/home/lib"
        else:
            workspace_dir = os.path.abspath(paths["workspace_dir"])
            library_dir = os.path.abspath(self.library_dir)

        # Retrieval against host paths regardless of docker mode.
        if self.candidate_strategy == "none":
            apply_candidates = (
                "(no candidates provided — read the library via bash to "
                "discover what to import.)"
            )
            self._prep_by_task[task_id] = []
        else:
            strategy: Strategy = self.candidate_strategy  # type: ignore[assignment]
            result = get_apply_candidates(
                strategy,
                library_dir=os.path.abspath(self.library_dir),
                app_dir=os.path.abspath(paths["workspace_dir"]),
                top_k=self.retrieve_top_k,
                min_line=self.retrieve_min_line,
                min_similarity=self.retrieve_min_similarity,
                nl_model=self.nl_model,
                nl_pick_model=self.nl_pick_model,
            )
            apply_candidates = result.markdown
            self._prep_by_task[task_id] = result.prep

        if self.extract_map:
            try:
                map_text = read_extract_map(self.library_dir)
                extract_map_block = filter_for_app(map_text, task_id)
            except Exception as e:
                print(f"[WebgenApplyAgent:{task_id}] extract_map read failed: {e}")
                extract_map_block = "(extract_map.md read error)"
        else:
            extract_map_block = "(extract_map disabled for this run)"

        task = self.task_lookup(task_id)
        if not isinstance(task, dict):
            raise ValueError(
                f"task_lookup({task_id!r}) returned non-dict: {type(task).__name__}"
            )

        return build_apply_user_prompt(
            task,
            task_id=task_id,
            workspace_dir=workspace_dir,
            library_dir=library_dir,
            apply_candidates=apply_candidates,
            extract_map_block=extract_map_block,
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
