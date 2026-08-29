"""BaseFullRun — abstract orchestrator for round-based full mode.

Full mode = N apps split into N/m rounds; each round runs three phases
sequentially (coding → extract → apply). Subclasses provide:

  - benchmark_name              — sub-dir under runs/ and backups/
  - parse_extra_args(parser)    — benchmark-specific CLI flags
  - load_tasks(args)            — return list of task_ids
  - run_coding_phase(...)       — generate m new submissions in parallel
  - run_extract_phase(...)      — produce / upgrade lib from cumulative apps
  - run_apply_phase(...)        — apply lib to all cumulative apps in parallel

Path layout (cross-benchmark, see run/backup_layout.py):

    runs/<bench>/<tag>/round_<N>/<phase>/{tasks/<id>/, lib/}
    runs/<bench>/<tag>/logs/round_<N>/<phase>/...
    backups/<bench>/<tag>/final/round_<N>/<phase>/...
    backups/<bench>/<tag>/logs/round_<N>/<phase>/...

Subclasses (WebgenFullRun, PaperbenchFullRun) inherit setup_paths / backup_round / main loop.
The agent-instantiation and per-phase logic remain benchmark-specific because
the agent constructors and prompts differ.
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
from run.base_run import resolve_project_dir


_STAGE_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "embeddings", ".git",
    "node_modules",
)
# NOTE: Do NOT add cocoindex artifacts (.cocoindex_code / target_sqlite.db /
# cocoindex.db) here. Excluding them from the staged copy makes cocoindex_app()
# fail to rebuild target_sqlite.db on the staged workspace, so ensure_nl_index
# raises "cocoindex sqlite missing" and extract/apply run WITHOUT NL candidates
# (silent quality regression — observed ~250 failures/run on n=16 sla_ours).
# Disk pressure from the O(N^2) cocoindex copy must be handled by freeing space
# / post-round purge, NOT by ignoring it here.


def _chunked(seq: list[str], m: int) -> list[list[str]]:
    return [seq[i:i + m] for i in range(0, len(seq), m)]


class BaseFullRun(ABC):
    benchmark_name: str = ""

    # ---- abstract: benchmark-specific contract ----------------------------

    @abstractmethod
    def parse_extra_args(self, parser: argparse.ArgumentParser) -> None:
        """Hook for benchmark-specific CLI args (split, cluster, etc.)."""

    @abstractmethod
    def load_tasks(self, args: argparse.Namespace) -> list[str]:
        """Return ordered list of task_ids to chunk into rounds."""

    @abstractmethod
    def run_coding_phase(
        self,
        round_num: int,
        task_ids: list[str],
        library_dir: str | None,
        args: argparse.Namespace,
    ) -> dict[str, str]:
        """Generate len(task_ids) new submissions in parallel.
        Returns {tid: post-coding submission_dir} for successes only."""

    @abstractmethod
    def run_extract_phase(
        self,
        round_num: int,
        source_app_paths: dict[str, str],
        seed_lib_dir: str | None,
        args: argparse.Namespace,
    ) -> tuple[str | None, dict[str, str]]:
        """Extract from cumulative apps (upgrade-mode if seed_lib_dir set).

        Returns ``(lib_dir or None, post-extract submission paths)``.

        Pipelines without intra-app pre-processing (e.g. paperbench/mle) return
        ``(lib_dir, source_app_paths)`` — passthrough. RAL's LocalExtract
        mutates app source before GlobalExtract, so it returns the post-Local
        paths so the caller can route them into the apply phase."""

    @abstractmethod
    def run_apply_phase(
        self,
        round_num: int,
        task_ids: list[str],
        lib_dir: str,
        prev_submissions: dict[str, str],
        args: argparse.Namespace,
    ) -> dict[str, str]:
        """Apply lib_dir to all task_ids in parallel.
        Returns {tid: post-apply submission_dir} (falls back to prev on fail)."""

    # ---- shared CLI -------------------------------------------------------

    def add_common_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--tag", required=True,
                            help=f"Run tag — outputs go to runs/{self.benchmark_name}/<tag>/ "
                                 f"and backups/{self.benchmark_name}/<tag>/.")
        parser.add_argument("--provider", choices=["openai", "openrouter", "deepseek"],
                            default="deepseek")
        parser.add_argument("--model", default="deepseek/deepseek-v4-flash")

        parser.add_argument("--m", "--apps-per-round", type=int, default=2,
                            dest="m",
                            help="Apps generated in parallel per round (= round size).")
        parser.add_argument("--max-workers", type=int, default=4,
                            help="Max parallel workers per phase.")
        parser.add_argument("--max-iter", type=int, default=1,
                            help="Per-agent step iterations.")
        parser.add_argument("--source-library-dir", default=None,
                            help="Optional initial lib seed for round 1.")
        parser.add_argument("--cost-limit", type=float, default=5.0)
        parser.add_argument("--step-limit", type=int, default=150)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--max-tokens", type=int, default=-1)
        parser.add_argument("--docker-image", default=None,
                            help="Docker image for agent runtime. Unset = LocalEnvironment.")

    def parse_args(self, argv: list[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=f"{self.benchmark_name} full-mode (round-based) orchestrator"
        )
        self.add_common_args(parser)
        self.parse_extra_args(parser)
        return parser.parse_args(argv)

    # ---- Path setup -------------------------------------------------------

    def setup_paths(self, args: argparse.Namespace) -> None:
        if not self.benchmark_name:
            raise ValueError("Subclass must set benchmark_name")
        project_dir = resolve_project_dir()
        args.project_dir = project_dir

        args.runs_root = backup_layout.runs_root(project_dir, self.benchmark_name, args.tag)
        args.log_root = os.path.join(args.runs_root, "logs")  # agents' log_dir
        args.backup_final_root = os.path.join(
            backup_layout.backups_root(project_dir, self.benchmark_name, args.tag),
            "final",
        )
        args.backup_logs_root = os.path.join(
            backup_layout.backups_root(project_dir, self.benchmark_name, args.tag),
            "logs",
        )
        os.makedirs(args.runs_root, exist_ok=True)
        os.makedirs(args.log_root, exist_ok=True)
        os.makedirs(args.backup_final_root, exist_ok=True)
        os.makedirs(args.backup_logs_root, exist_ok=True)

    # ---- Per-phase helpers (path builders) --------------------------------

    def phase_workspace(self, args: argparse.Namespace, round_num: int,
                        phase: str) -> str:
        """runs/<bench>/<tag>/round_N/<phase>/"""
        return os.path.join(args.runs_root, f"round_{round_num}", phase)

    def phase_tasks_dir(self, args: argparse.Namespace, round_num: int,
                        phase: str) -> str:
        return os.path.join(self.phase_workspace(args, round_num, phase), "tasks")

    # ---- Per-task runner --------------------------------------------------

    def _run_single(
        self,
        agent,
        task_id: str,
        round_num: int,
        log_phase: str,
    ) -> bool:
        try:
            agent.run(
                task_id=task_id,
                round_num=round_num,
                step_num=0,
                feedback=None,
                messages=None,
                log_phase=log_phase,
            )
            return True
        except Exception:
            print(f"[{type(self).__name__}] {log_phase} round={round_num} "
                  f"task={task_id} failed:")
            traceback.print_exc()
            return False

    def _run_in_parallel(
        self,
        agent,
        task_ids: list[str],
        round_num: int,
        log_phase: str,
        max_workers: int,
    ) -> dict[str, bool]:
        results: dict[str, bool] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(self._run_single, agent, tid, round_num, log_phase): tid
                for tid in task_ids
            }
            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    results[tid] = fut.result()
                except Exception:
                    print(f"[{type(self).__name__}] {log_phase} round={round_num} "
                          f"{tid} unexpected error:")
                    traceback.print_exc()
                    results[tid] = False
        return results

    # ---- Backup -----------------------------------------------------------

    def backup_round(self, round_num: int, args: argparse.Namespace) -> None:
        src_round = os.path.join(args.runs_root, f"round_{round_num}")
        if not os.path.isdir(src_round):
            print(f"[{type(self).__name__}] round {round_num}: nothing to back up")
            return

        final_target = os.path.join(args.backup_final_root, f"round_{round_num}")
        logs_target = os.path.join(args.backup_logs_root, f"round_{round_num}")

        if os.path.exists(final_target):
            shutil.rmtree(final_target)
        shutil.copytree(
            src_round, final_target,
            symlinks=True, ignore_dangling_symlinks=True,
            ignore=shutil.ignore_patterns("agent.env", "__pycache__", "*.pyc"),
        )

        src_logs_round = os.path.join(args.log_root, f"round_{round_num}")
        if os.path.isdir(src_logs_round):
            if os.path.exists(logs_target):
                shutil.rmtree(logs_target)
            shutil.copytree(
                src_logs_round, logs_target,
                symlinks=True,
                ignore=shutil.ignore_patterns("agent.env"),
            )

        print(f"[{type(self).__name__}] round {round_num} backup → {final_target}")

    # ---- Main loop --------------------------------------------------------

    def main(self, argv: list[str] | None = None) -> None:
        project_dir = resolve_project_dir()
        load_dotenv(dotenv_path=os.path.join(project_dir, ".env"))

        args = self.parse_args(argv)
        args.project_dir = project_dir
        self.setup_paths(args)

        task_ids = self.load_tasks(args)
        if not task_ids:
            sys.exit("No tasks to run")

        rounds = _chunked(task_ids, args.m)

        print(f"[Config] benchmark = {self.benchmark_name} (full mode)")
        print(f"[Config] tag       = {args.tag}")
        print(f"[Config] m         = {args.m}  (rounds = {len(rounds)})")
        print(f"[Config] tasks     = {task_ids}")
        print(f"[Config] runs_root = {args.runs_root}")
        print(f"[Config] log_root  = {args.log_root}")
        print(f"[Config] backup    = {args.backup_final_root}")
        print(f"[Config] provider  = {args.provider}")
        print(f"[Config] model     = {args.model}")
        print(f"[Config] docker    = {args.docker_image or '(LocalEnvironment)'}")
        print(f"[Config] seed_lib  = {args.source_library_dir or '(none)'}")

        current_lib_dir: str | None = None
        if args.source_library_dir:
            seed = os.path.abspath(args.source_library_dir)
            if not os.path.isdir(seed):
                sys.exit(f"--source-library-dir not found: {seed}")
            current_lib_dir = seed

        cumulative_subs: dict[str, str] = {}

        for round_num, round_tids in enumerate(rounds, start=1):
            print()
            print("#" * 60)
            print(f"# Round {round_num}/{len(rounds)}  tasks={round_tids}")
            print(f"#   current_lib = {current_lib_dir or '(none)'}")
            print("#" * 60)

            new_subs = self.run_coding_phase(round_num, round_tids, current_lib_dir, args)
            cumulative_subs.update(new_subs)

            if len(cumulative_subs) >= 2:
                new_lib, post_extract_subs = self.run_extract_phase(
                    round_num, dict(cumulative_subs), current_lib_dir, args,
                )
                # Adopt extract's post-mutation paths so apply sees the same
                # code GlobalExtract was built against (RAL's LocalExtract
                # mutates app source; other pipelines return paths verbatim).
                cumulative_subs.update(post_extract_subs)

                if new_lib:
                    cumulative_subs = self.run_apply_phase(
                        round_num, list(cumulative_subs), new_lib,
                        cumulative_subs, args,
                    )
                    current_lib_dir = new_lib
                else:
                    print(f"[Round {round_num}] no usable lib produced; skipping apply")
            else:
                print(f"[Round {round_num}] cumulative={len(cumulative_subs)} < 2 → skip phase 2")

            self.backup_round(round_num, args)

        print()
        print("#" * 60)
        print(f"# DONE  rounds={len(rounds)}  tasks={len(task_ids)}")
        print(f"#   final_lib  = {current_lib_dir or '(none)'}")
        print(f"#   cumulative = {list(cumulative_subs)}")
        print("#" * 60)
