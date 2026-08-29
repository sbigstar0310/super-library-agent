"""BaseCodingAgent — abstract base for mswe-agent based per-task coders.

Holds the shared machinery (model/env build, agent.env writing, DefaultAgent
vs ResumeAgent gate, output-path layout, run loop). Subclasses supply the
benchmark-specific contract via the abstract methods below.

Adopted by the paperbench and webgen agents.
"""

from __future__ import annotations
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from minisweagent.agents.default import DefaultAgent

from mswe_agents._factory import (
    build_environment,
    build_model,
    load_base_config,
)
from mswe_agents.resume_agent import ResumeAgent


CARRY_FORWARD_NOTE = (
    "[Carry-forward] You already refactored THIS SAME app in an earlier round; "
    "that full working session — including the contents of every app file you "
    "read — is in the conversation history above. The app's current files are "
    "exactly as you left them last round. The shared library has since grown: "
    "the [Apply candidates] and [Extract map] below reflect the CURRENT library. "
    "Adopt any newly-applicable library symbols and run the dead-code sweep. "
    "Do NOT re-read app files that are already visible in your history unless you "
    "are about to edit them — read only what actually changed (the new library "
    "symbols and the specific files you will edit)."
)


class BaseCodingAgent(ABC):
    """ABC for benchmark-specific coding agents over mini-swe-agent."""

    def __init__(
        self,
        provider: str,
        model: str,
        log_dir: str,
        library_dir: str | None = None,
        docker_image: str | None = None,
        max_tokens: int = -1,
        temperature: float = 0.0,
        cost_limit: float = 5.0,
        step_limit: int = 80,
        enable_resume: bool = False,
        resume_rebuilds_prompt: bool = False,
        timeout: int = 120,
        reasoning_effort: str | None = None,
    ):
        self.provider = provider
        self.model_name = model
        self.log_dir = log_dir
        self.library_dir = library_dir
        self.docker_image = docker_image
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cost_limit = cost_limit
        self.step_limit = step_limit
        self.enable_resume = enable_resume
        self.resume_rebuilds_prompt = resume_rebuilds_prompt
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

        self._base_config = load_base_config()
        os.makedirs(log_dir, exist_ok=True)

    # ---- abstract: benchmark-specific contract ----------------------------

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system message content sent to the agent's LLM."""

    @abstractmethod
    def build_user_prompt(self, task_id: str, paths: dict[str, str], agent_env_path: str) -> str:
        """Return the initial user task message.

        `paths` is whatever setup_workspace returned. `agent_env_path` is the
        host or container path of agent.env (for prompt substitution).
        """

    @abstractmethod
    def setup_workspace(self, task_id: str) -> dict[str, str]:
        """Create the per-task workspace and return paths.

        Keys: workspace_dir (where the agent writes submission, host abs path),
        host_dir (docker mount root containing paper/, lib/, submission/,
        agent.env), agent_cwd (container path in docker mode, host
        workspace_dir locally).
        """

    @abstractmethod
    def output_path_for(
        self, task_id: str, round_num: int, step_num: int, log_phase: str
    ) -> Path:
        """Where step_N.json trajectories should be saved."""

    def agent_env_keys(self) -> dict[str, str]:
        """{agent.env key: host env var}. Override to inject keys."""
        return {"OPENAI_API_KEY": "OPENAILIKE_API_KEY"}

    def instance_template(self) -> str:
        """The mini-swe-agent instance template wrapping the user task.

        Defaults to minisweagent's bundled template. Override to reframe the
        opening (e.g. a maintenance-patch agent that applies a policy update
        rather than 'solving an issue')."""
        return self._base_config["agent"]["instance_template"]

    # ---- shared machinery -------------------------------------------------

    def _write_agent_env(self, host_dir: str) -> str:
        """Write agent.env to host_dir/agent.env; return its host abs path.

        Docker bind-mounts it to /home/agent.env; local reads it directly.
        PYTHONPATH is set when library_dir is configured so `import lib`
        resolves regardless of cwd (docker: /home; local: dirname(library_dir)).
        """
        agent_env_path = os.path.join(host_dir, "agent.env")
        os.makedirs(host_dir, exist_ok=True)

        body_lines: list[str] = []
        for env_key, host_var in self.agent_env_keys().items():
            val = os.environ.get(host_var, "")
            if val:
                body_lines.append(f"{env_key}={val}")

        if self.library_dir:
            if self.docker_image:
                body_lines.append("PYTHONPATH=/home")
            else:
                lib_parent = os.path.dirname(os.path.abspath(self.library_dir))
                body_lines.append(f"PYTHONPATH={lib_parent}")

        Path(agent_env_path).write_text("\n".join(body_lines) + "\n")
        try:
            os.chmod(agent_env_path, 0o600)
        except OSError:
            pass
        return agent_env_path

    def _container_env_path(self, host_agent_env_path: str) -> str:
        """Path of agent.env as seen by the agent (host vs container)."""
        return "/home/agent.env" if self.docker_image else host_agent_env_path

    def _stage_library(self, host_dir: str) -> str | None:
        """Copy `self.library_dir` into `<host_dir>/lib/` (idempotent).

        Returns the staged path, or None when no library is configured.
        Skips the copy when the target already exists and is non-empty.
        """
        if not self.library_dir:
            return None
        target = os.path.join(host_dir, "lib")
        if os.path.isdir(target) and os.listdir(target):
            return target
        # Clean partial state (empty dir from a prior aborted setup).
        if os.path.exists(target):
            if os.path.islink(target) or os.path.isfile(target):
                os.remove(target)
            else:
                shutil.rmtree(target)
        shutil.copytree(
            os.path.abspath(self.library_dir),
            target,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".git", "embeddings",
            ),
            # An extract phase (running as root in-container) can sporadically
            # leave a dangling symlink in the lib pointing at a container path
            # (e.g. `lib -> /home/lib`, or a stray `node_modules/react`). With
            # the default symlinks=False those links are dereferenced, and a
            # dead target raises shutil.Error mid-copytree → coding produces an
            # empty submission that is silently backed up as a 0-score app.
            # ignore_dangling_symlinks skips only the dead links (valid symlinks
            # are still dereferenced as before), so this is regression-free.
            ignore_dangling_symlinks=True,
        )
        return target

    def pre_drive(self) -> None:
        """Hook called once per phase, before the parallel loop. No-op default.

        Override for shared cross-task setup (e.g. embedding a single library
        consumed by all workers) to avoid a race when every worker thread
        independently materializes the same artifact.
        """
        return None

    def pre_run(self, task_id: str, paths: dict[str, str]) -> None:
        """Hook called between agent.env write and model/env build. No-op
        default. Apply-style subclasses override to embed submission + library
        before retrieval-augmented prompt construction.
        """
        return None

    def run(
        self,
        task_id: str,
        round_num: int = 0,
        step_num: int = 0,
        feedback: str | None = None,
        messages: list[dict] | None = None,
        log_phase: str = "phase_1",
        extra_user_prompt_kwargs: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Run one coding session for `task_id`.

        Returns the agent.messages list (full trajectory).
        """
        print(
            f"[Round {round_num} | {self.__class__.__name__}] task={task_id} "
            f"step={step_num} phase={log_phase}"
        )

        paths = self.setup_workspace(task_id)
        host_dir = paths["host_dir"]
        agent_cwd = paths["agent_cwd"]

        # Write agent.env BEFORE building env — docker bind-mount would
        # auto-create the missing path as a directory and break the writer.
        host_agent_env_path = self._write_agent_env(host_dir)
        env_path_for_prompt = self._container_env_path(host_agent_env_path)

        self.pre_run(task_id, paths)

        output_path = self.output_path_for(task_id, round_num, step_num, log_phase)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = build_model(
            provider=self.provider,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_config=self._base_config,
            reasoning_effort=self.reasoning_effort,
        )
        # setup_workspace may customize Docker mounts/env via extra keys in its
        # return dict. A supplied mount_spec bypasses the paperbench default
        # layout; extra_env/extra_run_args merge on top.
        env = build_environment(
            agent_cwd,
            self._base_config,
            timeout=self.timeout,
            docker_image=self.docker_image,
            host_dir=host_dir,
            has_library=bool(self.library_dir),
            mount_spec=paths.get("mount_spec"),
            extra_env=paths.get("extra_env"),
            extra_run_args=paths.get("extra_run_args"),
            forward_env=paths.get("forward_env"),
        )

        instance_template = self.instance_template()
        agent_cls = ResumeAgent if self.enable_resume else DefaultAgent
        agent = agent_cls(
            model,
            env,
            system_template=self.system_prompt(),
            instance_template=instance_template,
            step_limit=self.step_limit,
            cost_limit=self.cost_limit,
            output_path=output_path,
        )

        if self.enable_resume and messages:
            if self.resume_rebuilds_prompt:
                # Carry-forward: prior session (app already explored) stays in
                # history; the new task is THIS round's freshly-built prompt
                # (current candidates / lib state), prefixed with a note so the
                # agent reuses its prior reads instead of re-cat-ing the app.
                user_prompt = self.build_user_prompt(
                    task_id, paths, env_path_for_prompt,
                    **(extra_user_prompt_kwargs or {}),
                )
                task_text = f"{CARRY_FORWARD_NOTE}\n\n{user_prompt}"
                if feedback:
                    task_text += f"\n\n<feedback>\n{feedback}\n</feedback>"
            else:
                task_text = feedback or "No feedback. Continue refining your submission."
            print(
                f"[{self.__class__.__name__}:{task_id}] resuming from "
                f"{len(messages)} prior messages "
                f"(rebuild_prompt={self.resume_rebuilds_prompt})"
            )
            agent.run(task=task_text, prior_messages=messages)
        else:
            user_prompt = self.build_user_prompt(
                task_id, paths, env_path_for_prompt, **(extra_user_prompt_kwargs or {})
            )
            if feedback:
                user_prompt += f"\n\n<feedback>\n{feedback}\n</feedback>"
            agent.run(task=user_prompt)

        result_meta = (
            agent.messages[-1].get("extra", {}) if agent.messages else {}
        )
        print(
            f"[{self.__class__.__name__}:{task_id}] done: "
            f"{result_meta.get('exit_status', 'unknown')}, "
            f"{agent.n_calls} calls, ${agent.cost:.4f}"
        )
        return agent.messages
