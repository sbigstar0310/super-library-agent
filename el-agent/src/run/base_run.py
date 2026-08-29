"""BaseRun — abstract benchmark-run dispatcher (non-full modes).

Cross-benchmark workflow shared by non-full benchmark runs: resolve
runs_dir/log_dir/backup_dir under the unified layout (see backup_layout.py),
run tasks in parallel via ThreadPoolExecutor, then back up runs_dir → backup_dir
(agent.env stripped).

Subclasses implement:
  - benchmark_name             — sub-dir under runs/ and backups/
  - load_tasks(args)           — return list of task_ids
  - prepare_task_workspace(task_id, args) — materialize per-task input
  - instantiate_coding_agent(args) — return a BaseCodingAgent

For full mode (multi-round, three phases per round), see BaseFullRun.
"""

from __future__ import annotations
import argparse
import os
import shutil
import sys
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from run import backup_layout


def resolve_project_dir() -> str:
    """Project root: the dir three levels above this file (el-agent/src/run/)."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


# Non-full modes share a single round (round_0).
NONFULL_ROUND = 0


class BaseRun(ABC):
    benchmark_name: str = ""  # subclasses override
    extra_modes: tuple[str, ...] = ()
    """Additional --mode choices a subclass exposes beyond the standard
    baseline / coding / apply triple. Subclasses that own the additional
    modes are responsible for handling them in load_tasks /
    prepare_task_workspace / instantiate_coding_agent and for any extra
    validation in validate_args. Example: a subclass may add ``("extract",)``."""

    # ---- abstract: benchmark-specific contract ----------------------------

    @abstractmethod
    def load_tasks(self, args: argparse.Namespace) -> list[str]:
        """Return list of task identifiers (paper_id, app_id, competition_id, ...)."""

    @abstractmethod
    def prepare_task_workspace(self, task_id: str, args: argparse.Namespace) -> None:
        """Materialize per-task inputs under args.runs_dir/<task_id>/.

        For paperbench: snapshot paper assets.
        For apply mode: also copy baseline submission and stage lib/.
        """

    @abstractmethod
    def instantiate_coding_agent(self, args: argparse.Namespace):
        """Return a configured BaseCodingAgent (one instance reused for all tasks)."""

    # ---- shared CLI + driver ---------------------------------------------

    def add_common_args(self, parser: argparse.ArgumentParser) -> None:
        """Common CLI flags. Subclasses can extend by overriding (call super first)."""
        parser.add_argument("--provider", "-p", choices=["openai", "openrouter", "deepseek"],
                            default="openrouter")
        parser.add_argument("--model", "-m", default="deepseek/deepseek-v4-flash")
        parser.add_argument("--tag", required=True,
                            help="Run tag — outputs go to runs/<benchmark>/<tag>/ and "
                                 "backups/<benchmark>/<tag>/final/round_0/<phase>/.")
        parser.add_argument(
            "--mode",
            choices=("baseline", "coding", "apply", *self.extra_modes),
            default="baseline",
            help="Non-full benchmark phase. Use the dedicated *_full_run.py "
                 "for round-based full mode.",
        )
        parser.add_argument("--max-workers", type=int, default=2)
        parser.add_argument("--max-tokens", type=int, default=-1)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--cost-limit", type=float, default=5.0)
        parser.add_argument("--step-limit", type=int, default=80)
        parser.add_argument("--max-iter", type=int, default=1,
                            help="Number of coding iterations per task. 1 = single-shot.")
        parser.add_argument("--source-library-dir", default=None,
                            help="Path to lib/ source tree. Required when --mode != baseline.")
        parser.add_argument("--source-baseline-dir", default=None,
                            help="Path to baseline backup tasks/ root. Required when --mode=apply.")
        parser.add_argument("--docker-image", default=None,
                            help="Docker image for agent runtime. Unset = LocalEnvironment.")

    def resolve_paths(self, args: argparse.Namespace) -> None:
        """Populate args.runs_dir, args.log_dir, args.backup_dir under the
        unified layout. Non-full = round_0. Phase = mode."""
        if not self.benchmark_name:
            raise ValueError("Subclass must set benchmark_name")
        project_dir = resolve_project_dir()
        args.project_dir = project_dir
        phase = args.mode

        runs_phase = backup_layout.runs_phase_dir(
            project_dir, self.benchmark_name, args.tag, NONFULL_ROUND, phase,
        )
        args.runs_dir = os.path.join(runs_phase, "tasks")

        args.log_dir = backup_layout.runs_logs_dir(
            project_dir, self.benchmark_name, args.tag, NONFULL_ROUND, phase,
        )

        args.backup_dir = backup_layout.backup_final_phase_dir(
            project_dir, self.benchmark_name, args.tag, NONFULL_ROUND, phase,
        )
        args.backup_logs_dir = backup_layout.backup_logs_phase_dir(
            project_dir, self.benchmark_name, args.tag, NONFULL_ROUND, phase,
        )

        os.makedirs(args.runs_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        args.phase_resolved = phase
        args.round_num = NONFULL_ROUND

    def validate_args(self, args: argparse.Namespace) -> None:
        if args.mode in ("coding", "apply"):
            if not args.source_library_dir or not os.path.isdir(args.source_library_dir):
                sys.exit(f"--source-library-dir required and must exist for mode={args.mode}")
        if args.mode == "apply":
            if not args.source_baseline_dir or not os.path.isdir(args.source_baseline_dir):
                sys.exit("--source-baseline-dir required for mode=apply")

    def run_one_task(
        self,
        task_id: str,
        agent,
        args: argparse.Namespace,
        step_num: int,
        state: dict,
    ) -> tuple[str, dict]:
        try:
            messages = agent.run(
                task_id=task_id,
                round_num=args.round_num,
                step_num=step_num,
                feedback=state.get("feedback"),
                messages=state.get("messages"),
                log_phase=args.phase_resolved,
            )
            new_state = dict(state)
            new_state["messages"] = messages
            new_state["feedback"] = None
            return task_id, new_state
        except Exception:
            print(f"[{self.benchmark_name}:{args.phase_resolved}] task {task_id} failed:")
            traceback.print_exc()
            new_state = dict(state)
            new_state["error"] = traceback.format_exc()
            new_state["done"] = True
            return task_id, new_state

    def drive(self, task_ids: list[str], args: argparse.Namespace) -> None:
        agent = self.instantiate_coding_agent(args)
        agent.pre_drive()
        states: dict[str, dict] = {
            tid: {"feedback": None, "messages": None, "done": False} for tid in task_ids
        }

        for step_num in range(args.max_iter):
            print()
            print("=" * 60)
            print(f"[{self.benchmark_name}:{args.phase_resolved} | Step {step_num}] start")
            print("=" * 60)

            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                futures = {
                    ex.submit(self.run_one_task, tid, agent, args, step_num, states[tid]): tid
                    for tid in task_ids if not states[tid].get("done")
                }
                for fut in as_completed(futures):
                    tid = futures[fut]
                    try:
                        _, new_state = fut.result()
                        states[tid] = new_state
                    except Exception:
                        print(f"[{self.benchmark_name}] step {step_num} {tid}:")
                        traceback.print_exc()

            if all(s.get("done") for s in states.values()):
                print(f"[{self.benchmark_name}:{args.phase_resolved}] all done at step {step_num}")
                break

    def finalize_backup(self, args: argparse.Namespace) -> None:
        """Copy runs/.../round_0/<phase>/ → backups/.../final/round_0/<phase>/
        and runs logs sibling → backups/.../logs/round_0/<phase>/.
        Strips agent.env."""
        ignore = shutil.ignore_patterns(
            "agent.env", "__pycache__", "*.pyc",
        )

        runs_phase_src = os.path.dirname(args.runs_dir)  # parent of tasks/
        if os.path.isdir(runs_phase_src):
            os.makedirs(os.path.dirname(args.backup_dir), exist_ok=True)
            if os.path.exists(args.backup_dir):
                shutil.rmtree(args.backup_dir)
            shutil.copytree(
                runs_phase_src, args.backup_dir,
                symlinks=True, ignore_dangling_symlinks=True, ignore=ignore,
            )

        # Logs go to a separate top-level sibling under backups/<bench>/<tag>/logs/.
        if os.path.isdir(args.log_dir):
            os.makedirs(os.path.dirname(args.backup_logs_dir), exist_ok=True)
            if os.path.exists(args.backup_logs_dir):
                shutil.rmtree(args.backup_logs_dir)
            shutil.copytree(
                args.log_dir, args.backup_logs_dir,
                symlinks=True, ignore=ignore,
            )

        print(f"\n[{self.benchmark_name}] Backup → {args.backup_dir}")
        print(f"[{self.benchmark_name}] Logs   → {args.backup_logs_dir}")

    def main(self, argv: list[str] | None = None) -> None:
        project_dir = resolve_project_dir()
        load_dotenv(dotenv_path=os.path.join(project_dir, ".env"))

        parser = argparse.ArgumentParser(description=f"{self.benchmark_name} run dispatcher")
        self.add_common_args(parser)
        self.extend_args(parser)
        args = parser.parse_args(argv)

        self.validate_args(args)
        self.resolve_paths(args)

        task_ids = self.load_tasks(args)
        if not task_ids:
            sys.exit("No tasks to run")

        print(f"[Config] benchmark = {self.benchmark_name}")
        print(f"[Config] tag/phase = {args.tag} / {args.phase_resolved}")
        print(f"[Config] runs_dir  = {args.runs_dir}")
        print(f"[Config] log_dir   = {args.log_dir}")
        print(f"[Config] backup    = {args.backup_dir}")
        print(f"[Config] backup_logs = {args.backup_logs_dir}")
        print(f"[Config] tasks     = {task_ids}")
        print(f"[Config] provider  = {args.provider}")
        print(f"[Config] model     = {args.model}")
        print(f"[Config] mode      = {args.mode}")
        print(f"[Config] docker    = {args.docker_image or '(LocalEnvironment)'}")

        for tid in task_ids:
            self.prepare_task_workspace(tid, args)

        self.drive(task_ids, args)
        self.finalize_backup(args)
        print(f"[{self.benchmark_name}] done.")

    def extend_args(self, parser: argparse.ArgumentParser) -> None:
        """Hook for subclasses to add their own CLI args. Default: no-op."""
        return None
