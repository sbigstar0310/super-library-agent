"""PaperbenchFullRun — paperbench round-based orchestrator.

Four modes, dispatched via ``--mode`` (mirrors WebgenFullRun):

  baseline           Single round: every paper runs through PaperbenchCodingAgent
                     in parallel. No library produced.

  sla_naive          Per round: (a) coding (parallel M new papers) →
                     (b) PaperbenchLibraryAgent over cumulative papers
                         (single shot; extract + apply in one).

  sla_naive_split    Per round: (a) coding [parallel M] →
                     (b) global_extract over cumulative papers →
                     (c) apply [parallel, cumulative].
                     Ablation of sla_ours: no local_extract, no extract_map,
                     no extract/apply candidates.

  sla_ours           Per round: (a) coding [parallel M] →
                     (b) local_extract per *new* paper [parallel M] →
                     (c) global_extract over cumulative papers →
                     (d) apply [parallel, cumulative].

Workspace layout (BaseFullRun-managed):
    runs/paperbench/<tag>/round_<i>/{coding,local_extract,extract,apply}/
        tasks/<paper_id>/{paper/, submission/, lib?/}
    runs/paperbench/<tag>/logs/round_<i>/<phase>/...
    backups/paperbench/<tag>/final/round_<i>/<phase>/...
    backups/paperbench/<tag>/logs/round_<i>/<phase>/...

Each task dir carries a paper/ snapshot (whitelisted from the upstream
paperbench data dir) alongside submission/. Webgen has no equivalent —
this is paperbench-specific plumbing.

Task selection: either ``--task-list`` (CSV of paper ids; mirrors webgen's
``--task-list``) or ``--cluster-id`` (int, picked from
``data/augments/paperbench/cluster/cluster.json``).

K-feedback (regression-test inner loop) is plumbed via ``--k`` but
hardcoded off (no paperbench feedback adapter — grading is code-only and
the rubric leaks at evaluation time).
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow `python -m run.paperbench_full_run` from el-agent/src.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mswe_agents.paperbench import (  # noqa: E402
    PaperbenchApplyAgent,
    PaperbenchCodingAgent,
    PaperbenchGlobalExtractAgent,
    PaperbenchLibraryAgent,
    PaperbenchLocalExtractAgent,
)
from mswe_agents.paperbench.global_extract_agent import EXTRACT_TASK_ID  # noqa: E402
from mswe_agents.paperbench.library_agent import LIBRARY_TASK_ID  # noqa: E402
from prompts.common import EXTRACT_MAP_INSTRUCTION  # noqa: E402
from prompts.common.librarian_repair import build_librarian_repair_prompt  # noqa: E402
from run.base_full_run import BaseFullRun, _STAGE_IGNORE, _chunked  # noqa: E402
from run.base_run import resolve_project_dir  # noqa: E402
from run.librarian_select import SamplePaths, select_winner  # noqa: E402
from utils.gates.paperbench_static import (  # noqa: E402
    GateResult,
    run_paperbench_static_gate,
)


# Default upstream paperbench data dir (post-move under data/).
PAPERBENCH_DATA_REL = "data/frontier-evals/project/paperbench/data"
DEFAULT_CLUSTER_JSON_REL = "data/augments/paperbench/cluster/cluster.json"

# Whitelist of paper assets exposed to agents. EXCLUDES rubric.json,
# judge/, judge.addendum.md, config.yaml — grader-only artifacts that
# would either leak grading criteria or confuse the agent.
_PAPER_SNAPSHOT_FILES = ("paper.md", "paper.pdf", "addendum.md", "blacklist.txt")
_PAPER_SNAPSHOT_DIRS = ("assets",)


def _snapshot_paper(source_paper_dir: str, target_paper_dir: str) -> None:
    """Mirror the whitelisted paper assets into a per-task paper/ dir.

    Idempotent: if the target already has any whitelisted file, treats it
    as already-snapshotted and is a no-op. Use rmtree externally if you
    need a forced re-snapshot.
    """
    if os.path.isdir(target_paper_dir) and any(
        os.path.exists(os.path.join(target_paper_dir, f))
        for f in _PAPER_SNAPSHOT_FILES
    ):
        return
    os.makedirs(target_paper_dir, exist_ok=True)
    for fname in _PAPER_SNAPSHOT_FILES:
        src = os.path.join(source_paper_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(target_paper_dir, fname))
    for dname in _PAPER_SNAPSHOT_DIRS:
        src = os.path.join(source_paper_dir, dname)
        if os.path.isdir(src):
            shutil.copytree(
                src, os.path.join(target_paper_dir, dname),
                dirs_exist_ok=True, ignore=_STAGE_IGNORE,
            )


def _carry_paper(src_paper_dir: str, target_paper_dir: str) -> None:
    """Copy an already-snapshotted paper/ from one phase dir to the next."""
    if os.path.isdir(target_paper_dir) and os.listdir(target_paper_dir):
        return
    if not os.path.isdir(src_paper_dir):
        raise FileNotFoundError(f"Missing source paper snapshot: {src_paper_dir}")
    shutil.copytree(
        src_paper_dir, target_paper_dir, dirs_exist_ok=True,
        ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True,
    )


def _gate_lib_root(lib_dir: str | None) -> str | None:
    """PYTHONPATH root the static gate should resolve ``from lib.*`` against.

    The pipeline's ``library_dir`` (both sla_naive and librarian) IS the ``lib``
    package directory itself — its contents (``__init__.py``, subpackages) sit
    directly inside it, and apps import ``from lib.<sub> import X`` with the
    *parent* dir on PYTHONPATH. The static gate, however, resolves ``lib.<sub>``
    as ``<lib_dir>/lib/<sub>`` — i.e. it expects ``lib_dir`` to be the parent of
    the ``lib`` package. So when our dir is itself named ``lib`` (always, here),
    hand the gate its parent; otherwise pass through unchanged. Without this the
    gate false-fails EVERY ``from lib.*`` import (the baseline gate sweep never
    caught it because zero-shot corpora have no lib).
    """
    if not lib_dir:
        return lib_dir
    norm = os.path.normpath(lib_dir)
    if os.path.basename(norm) == "lib":
        return os.path.dirname(norm)
    return lib_dir


def _gate_error_tail(result: GateResult, max_lines: int = 40) -> str:
    """Join a static-gate GateResult's error list into a repair-prompt tail.

    The webgen build gate hands the repair prompt a single ``error_tail``
    string; the paperbench static gate produces a list of per-line
    compile/import errors. Join them (capping to the last ``max_lines`` so a
    pathological failure can't blow up the prompt) into the same shape the
    shared ``build_librarian_repair_prompt`` expects.
    """
    errors = list(result.errors or [])
    if not errors:
        return "(no error output captured)"
    if len(errors) > max_lines:
        head = [f"... ({len(errors) - max_lines} earlier errors omitted) ..."]
        errors = head + errors[-max_lines:]
    return "\n".join(errors)


class PaperbenchFullRun(BaseFullRun):
    benchmark_name = "paperbench"

    # ---- CLI extension ----------------------------------------------------

    def parse_extra_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--mode",
            required=True,
            choices=["baseline", "sla_naive", "sla_naive_split", "sla_ours",
                     "librarian"],
            help="Pipeline variant — see module docstring. 'librarian' = "
                 "single round: ZS-seeded corpus → K PaperbenchLibraryAgent "
                 "samples → static compile/import gate + repair → MDL rerank "
                 "→ promote winner.",
        )
        parser.add_argument(
            "--librarian-k",
            type=int,
            default=8,
            help="librarian mode: number of library+refactor candidates to "
                 "sample (temperature>0). Winner chosen by static gate + MDL "
                 "rerank. Default 8.",
        )
        parser.add_argument(
            "--task-list", default=None,
            help="Comma-separated task_ids (subdirs of "
                 "<paperbench-data-dir>/papers/); overrides --cluster-id. "
                 "Mirrors webgen's --task-list.",
        )
        parser.add_argument(
            "--paperbench-data-dir", default=None,
            help=f"Path to paperbench data dir. Default: "
                 f"<project>/{PAPERBENCH_DATA_REL}",
        )
        parser.add_argument(
            "--cluster-id", type=int, default=None,
            help=f"Cluster id from {DEFAULT_CLUSTER_JSON_REL}.",
        )
        parser.add_argument(
            "--cluster-size", type=int, default=None,
            help="Take the first N tasks from the chosen cluster. "
                 "Default: full cluster.",
        )
        parser.add_argument(
            "--cluster-json", default=None,
            help=f"Path to cluster.json. Default: "
                 f"<project>/{DEFAULT_CLUSTER_JSON_REL}",
        )
        parser.add_argument(
            "--k", type=int, default=0,
            help="Apply-phase feedback iterations. Code-only grading has no "
                 "build/regression-test feedback — keep 0.",
        )
        parser.add_argument(
            "--candidate-strategy",
            choices=("embed", "nl", "none"), default="nl",
            help="Apply/extract candidate retrieval backend. Default nl. "
                 "'none' skips candidate retrieval (used by sla_naive_split).",
        )
        parser.add_argument(
            "--nl-pick-model", default="",
            help="LLM used for NL candidate selection. Empty/auto = derive "
                 "from --model by stripping the litellm provider prefix.",
        )
        # GlobalExtract clustering knobs (match webgen defaults).
        parser.add_argument(
            "--extract-cluster-distance-threshold", type=float, default=1.2,
        )
        parser.add_argument(
            "--extract-cluster-min-mean-sim", type=float, default=0.50,
        )
        parser.add_argument(
            "--extract-cluster-top-k", type=int, default=10,
        )
        parser.add_argument(
            "--extract-map",
            action=argparse.BooleanOptionalAction, default=True,
            help="sla_ours: after extract, resume the agent with a "
                 "map-writing instruction so it produces lib/extract_map.md.",
        )
        # LocalExtract knobs.
        parser.add_argument(
            "--local-extract-distance-threshold", type=float, default=1.0,
        )
        parser.add_argument(
            "--local-extract-min-mean-sim", type=float, default=0.55,
        )
        parser.add_argument(
            "--local-extract-top-k", type=int, default=12,
        )
        # sla_naive optional candidate strategy.
        parser.add_argument(
            "--library-candidate-strategy",
            choices=("none", "embed", "nl"), default="none",
            help="sla_naive: optional candidate list passed to the unified "
                 "PaperbenchLibraryAgent. Default 'none' (agent explores via bash).",
        )
        parser.add_argument(
            "--reasoning-effort",
            choices=("low", "medium", "high"), default=None,
            help="Override the model's reasoning_effort.",
        )
        parser.add_argument(
            "--seed-coding-tag", default=None,
            help="If set, skip round-1 coding generation and seed it from "
                 "backups/paperbench/<seed-coding-tag>/final/round_1/coding/"
                 "tasks/. Used to share a fixed baseline R1 across sla_naive / "
                 "sla_ours trials.",
        )
        parser.add_argument(
            "--time-limit-hours", type=float, default=2.0,
            help="Rendered into the upstream coding prompt's "
                 "'Total Runtime' bullet. Default 2.0.",
        )

    # ---- Task loading -----------------------------------------------------

    def load_tasks(self, args: argparse.Namespace) -> list[str]:
        data_dir = args.paperbench_data_dir or os.path.join(
            args.project_dir, PAPERBENCH_DATA_REL,
        )
        if not os.path.isdir(data_dir):
            sys.exit(f"--paperbench-data-dir not found: {data_dir}")
        args.paperbench_data_dir_resolved = os.path.abspath(data_dir)

        papers_root = os.path.join(args.paperbench_data_dir_resolved, "papers")
        if not os.path.isdir(papers_root):
            sys.exit(f"papers/ not found under {data_dir}")
        args.papers_root = papers_root

        if args.task_list:
            requested = [t.strip() for t in args.task_list.split(",") if t.strip()]
        elif args.cluster_id is not None:
            requested = self._tasks_from_cluster(args)
        else:
            sys.exit(
                "Either --task-list or --cluster-id must be provided "
                "(no default split for paperbench)."
            )

        missing = [t for t in requested if not os.path.isdir(os.path.join(papers_root, t))]
        if missing:
            available = sorted(os.listdir(papers_root))
            sys.exit(
                f"Unknown task_ids: {missing}\n"
                f"Available under {papers_root}: {available}"
            )
        return requested

    def _tasks_from_cluster(self, args: argparse.Namespace) -> list[str]:
        cluster_path = (
            args.cluster_json
            or os.path.join(args.project_dir, DEFAULT_CLUSTER_JSON_REL)
        )
        if not os.path.isfile(cluster_path):
            sys.exit(f"Cluster file not found: {cluster_path}")
        data = json.loads(Path(cluster_path).read_text())
        for c in data.get("clusters", []):
            if c.get("id") == args.cluster_id:
                tasks = list(c.get("tasks", []))
                if args.cluster_size is not None:
                    if args.cluster_size <= 0 or args.cluster_size > len(tasks):
                        sys.exit(
                            f"--cluster-size must be in [1, {len(tasks)}] "
                            f"(cluster {args.cluster_id} has {len(tasks)} tasks)"
                        )
                    tasks = tasks[:args.cluster_size]
                return tasks
        sys.exit(f"Cluster id {args.cluster_id} not found in {cluster_path}")

    # ---- Helpers ----------------------------------------------------------

    def _task_lookup(self, args: argparse.Namespace):
        # task_id == paperbench paper id (subdir under papers/). The agent's
        # build_user_prompt overrides paper_dir from its own setup_workspace.
        return lambda tid: {
            "task_id": tid,
            "paper_source_dir": os.path.join(args.papers_root, tid),
        }

    def _common_kwargs(self, args: argparse.Namespace) -> dict:
        """Args shared by all paperbench agent constructors.

        ``timeout`` = 600s (10 minutes) per-bash command. Paperbench
        upstream uses 7200s (2h) but at that cap a hung command burns
        the whole budget before the agent sees it; the coding/apply
        prompts surface this cap so the agent self-budgets accordingly.
        """
        return dict(
            task_lookup=self._task_lookup(args),
            provider=args.provider,
            model=args.model,
            log_dir=args.log_root,
            docker_image=args.docker_image,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            cost_limit=args.cost_limit,
            step_limit=args.step_limit,
            timeout=600,
            reasoning_effort=(args.reasoning_effort or None),
        )

    def _stage_paper_for_task(
        self,
        task_id: str,
        target_host_dir: str,
        args: argparse.Namespace,
    ) -> None:
        """Snapshot paper/ from the upstream data dir into a per-task host dir.

        Idempotent — no-op if paper.md already present at the target.
        Whitelist filters out rubric.json / judge/ / config.yaml.
        """
        target_paper_dir = os.path.join(target_host_dir, "paper")
        source_paper_dir = os.path.join(args.papers_root, task_id)
        _snapshot_paper(source_paper_dir, target_paper_dir)

    def _carry_paper_to_phase(
        self,
        task_id: str,
        src_host_dir: str,
        target_host_dir: str,
    ) -> None:
        """Bring a snapshotted paper/ from one phase dir to the next."""
        src = os.path.join(src_host_dir, "paper")
        dst = os.path.join(target_host_dir, "paper")
        _carry_paper(src, dst)

    # ---- Phases (sla_ours) -----------------------------------------------

    def run_coding_phase(
        self,
        round_num: int,
        task_ids: list[str],
        library_dir: str | None,
        args: argparse.Namespace,
    ) -> dict[str, str]:
        ws_root = self.phase_tasks_dir(args, round_num, "coding")
        os.makedirs(ws_root, exist_ok=True)

        # Snapshot paper + create submission dir before invoking the agent:
        # PaperbenchCodingAgent.setup_workspace raises if paper/ is absent.
        for tid in task_ids:
            host_dir = os.path.join(ws_root, tid)
            os.makedirs(os.path.join(host_dir, "submission"), exist_ok=True)
            self._stage_paper_for_task(tid, host_dir, args)

        agent = PaperbenchCodingAgent(
            workspace_root=ws_root,
            library_dir=library_dir,
            time_limit_hours=args.time_limit_hours,
            enable_resume=False,
            **self._common_kwargs(args),
        )

        state: dict[str, dict] = {tid: {"alive": False} for tid in task_ids}
        max_workers = min(len(task_ids), args.max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    self._run_phase_inner, agent, tid, round_num, 0,
                    None, None, "coding",
                ): tid
                for tid in task_ids
            }
            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    msgs = fut.result()
                    state[tid]["alive"] = msgs is not None
                except Exception:
                    traceback.print_exc()
                    state[tid]["alive"] = False

        return {tid: os.path.join(ws_root, tid, "submission") for tid in task_ids}

    def run_local_extract_phase(
        self,
        round_num: int,
        new_app_paths: dict[str, str],
        seed_lib_dir: str | None,
        args: argparse.Namespace,
    ) -> dict[str, str]:
        """Per-paper intra-extract over the M *new* tasks only.

        Carry-forward papers are not re-processed. Returns post-LocalExtract
        submission paths for the new tasks.
        """
        ws_root = self.phase_tasks_dir(args, round_num, "local_extract")
        os.makedirs(ws_root, exist_ok=True)

        staged: dict[str, str] = {}
        for tid, src_sub in new_app_paths.items():
            target_host = os.path.join(ws_root, tid)
            target_sub = os.path.join(target_host, "submission")
            os.makedirs(target_host, exist_ok=True)

            if not os.path.isdir(target_sub) or not os.listdir(target_sub):
                if not os.path.isdir(src_sub):
                    print(f"[PaperbenchFullRun] local_extract: missing source "
                          f"submission for {tid}: {src_sub}")
                    continue
                shutil.copytree(src_sub, target_sub, dirs_exist_ok=True,
                                ignore=_STAGE_IGNORE,
                                ignore_dangling_symlinks=True)

            src_host_dir = os.path.dirname(src_sub)
            try:
                self._carry_paper_to_phase(tid, src_host_dir, target_host)
            except FileNotFoundError:
                # Prior phase dir lacked paper/ — re-snapshot from upstream.
                self._stage_paper_for_task(tid, target_host, args)

            staged[tid] = target_sub

        if not staged:
            print(f"[PaperbenchFullRun] round {round_num} local_extract: nothing "
                  f"staged; skipping.")
            return new_app_paths

        global_lib_for_local = (
            seed_lib_dir if seed_lib_dir and os.path.isdir(seed_lib_dir)
            else None
        )

        agent = PaperbenchLocalExtractAgent(
            workspace_root=ws_root,
            library_dir=global_lib_for_local,
            cluster_distance_threshold=args.local_extract_distance_threshold,
            cluster_min_mean_sim=args.local_extract_min_mean_sim,
            cluster_top_k=args.local_extract_top_k,
            enable_resume=False,
            candidate_strategy=args.candidate_strategy,
            nl_pick_model=self._resolve_pick_model(args),
            **self._common_kwargs(args),
        )

        max_workers = min(len(staged), args.max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    self._run_phase_inner, agent, tid, round_num, 0,
                    None, None, "local_extract",
                ): tid
                for tid in staged
            }
            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    fut.result()
                except Exception:
                    print(f"[PaperbenchFullRun] local_extract round={round_num} "
                          f"task={tid} unexpected error:")
                    traceback.print_exc()

        return staged

    def run_extract_phase(
        self,
        round_num: int,
        source_app_paths: dict[str, str],
        seed_lib_dir: str | None,
        args: argparse.Namespace,
    ) -> tuple[str | None, dict[str, str]]:
        """sla_ours: GlobalExtract over cumulative source apps.

        LocalExtract is now driven from _run_sla_ours so it can run even when
        only one cumulative paper exists (R1 with M=1).
        """
        phase_dir = self.phase_workspace(args, round_num, "extract")
        ws_root = self.phase_tasks_dir(args, round_num, "extract")
        lib_dir = os.path.join(phase_dir, "lib")
        os.makedirs(lib_dir, exist_ok=True)
        os.makedirs(ws_root, exist_ok=True)

        if seed_lib_dir and os.path.isdir(seed_lib_dir) and not os.listdir(lib_dir):
            shutil.copytree(seed_lib_dir, lib_dir, dirs_exist_ok=True,
                            ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)

        # Stage submission + paper for each cumulative task (RO inputs).
        staged_apps: dict[str, str] = {}
        for tid, src_sub in source_app_paths.items():
            target_host = os.path.join(ws_root, tid)
            target_sub = os.path.join(target_host, "submission")
            os.makedirs(target_host, exist_ok=True)

            if not os.path.isdir(target_sub) or not os.listdir(target_sub):
                if not os.path.isdir(src_sub):
                    print(f"[PaperbenchFullRun] extract: missing source app for "
                          f"{tid}: {src_sub}")
                    continue
                shutil.copytree(src_sub, target_sub, dirs_exist_ok=True,
                                ignore=_STAGE_IGNORE,
                                ignore_dangling_symlinks=True)

            src_host_dir = os.path.dirname(src_sub)
            try:
                self._carry_paper_to_phase(tid, src_host_dir, target_host)
            except FileNotFoundError:
                self._stage_paper_for_task(tid, target_host, args)

            staged_apps[tid] = target_sub

        if len(staged_apps) < 2:
            print(f"[PaperbenchFullRun] round {round_num} extract: <2 staged "
                  f"apps ({list(staged_apps)}); skipping.")
            return (seed_lib_dir if seed_lib_dir else None, source_app_paths)

        agent = PaperbenchGlobalExtractAgent(
            workspace_root=ws_root,
            source_apps=staged_apps,
            library_dir=lib_dir,
            cluster_distance_threshold=args.extract_cluster_distance_threshold,
            cluster_min_mean_sim=args.extract_cluster_min_mean_sim,
            cluster_top_k=args.extract_cluster_top_k,
            enable_resume=args.extract_map,
            candidate_strategy=args.candidate_strategy,
            nl_pick_model=self._resolve_pick_model(args),
            **self._common_kwargs(args),
        )
        msgs = self._run_phase_inner(
            agent, EXTRACT_TASK_ID, round_num, 0, None, None, "extract",
        )
        if msgs is None:
            print(f"[PaperbenchFullRun] round {round_num} extract failed "
                  f"(continuing with whatever lib exists)")

        if msgs is not None and args.extract_map:
            map_msgs = self._run_extract_map_turn(
                round_num, lib_dir, staged_apps, msgs, args,
            )
            if map_msgs is None:
                print(f"[PaperbenchFullRun] round {round_num} extract_map turn "
                      f"failed (continuing without map).")

        if not os.path.isdir(lib_dir) or not os.listdir(lib_dir):
            print(f"[PaperbenchFullRun] round {round_num} extract produced empty lib")
            return (None, source_app_paths)
        return (lib_dir, source_app_paths)

    def _run_extract_map_turn(
        self,
        round_num: int,
        lib_dir: str,
        staged_apps: dict[str, str],
        prior_messages: list[dict],
        args: argparse.Namespace,
    ) -> list[dict] | None:
        try:
            agent = PaperbenchGlobalExtractAgent(
                workspace_root=self.phase_tasks_dir(args, round_num, "extract"),
                source_apps=staged_apps,
                library_dir=lib_dir,
                cluster_distance_threshold=args.extract_cluster_distance_threshold,
                cluster_min_mean_sim=args.extract_cluster_min_mean_sim,
                cluster_top_k=args.extract_cluster_top_k,
                enable_resume=True,
                candidate_strategy=args.candidate_strategy,
                nl_pick_model=self._resolve_pick_model(args),
                **self._common_kwargs(args),
            )
            library_dir_in_prompt = (
                "/home/lib" if agent.docker_image else os.path.abspath(lib_dir)
            )
            instruction = EXTRACT_MAP_INSTRUCTION.format(
                library_dir=library_dir_in_prompt
            )
            return agent.run(
                task_id=EXTRACT_TASK_ID,
                round_num=round_num,
                step_num=0,
                feedback=instruction,
                messages=prior_messages,
                log_phase="extract",
            )
        except Exception:
            print(f"[PaperbenchFullRun] map turn round={round_num} failed:")
            traceback.print_exc()
            return None

    def run_apply_phase(
        self,
        round_num: int,
        task_ids: list[str],
        lib_dir: str,
        prev_submissions: dict[str, str],
        args: argparse.Namespace,
    ) -> dict[str, str]:
        ws_root = self.phase_tasks_dir(args, round_num, "apply")
        os.makedirs(ws_root, exist_ok=True)

        for tid in task_ids:
            target_host = os.path.join(ws_root, tid)
            target_sub = os.path.join(target_host, "submission")
            os.makedirs(target_host, exist_ok=True)
            if not os.path.isdir(target_sub) or not os.listdir(target_sub):
                src_sub = prev_submissions.get(tid)
                if not src_sub or not os.path.isdir(src_sub):
                    print(f"[PaperbenchFullRun] apply: missing prev submission "
                          f"for {tid}: {src_sub}")
                    continue
                shutil.copytree(src_sub, target_sub, dirs_exist_ok=True,
                                ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)

            src_sub = prev_submissions.get(tid)
            if src_sub:
                try:
                    self._carry_paper_to_phase(tid, os.path.dirname(src_sub), target_host)
                except FileNotFoundError:
                    self._stage_paper_for_task(tid, target_host, args)
            else:
                self._stage_paper_for_task(tid, target_host, args)

        agent = PaperbenchApplyAgent(
            workspace_root=ws_root,
            library_dir=lib_dir,
            enable_resume=False,
            extract_map=args.extract_map,
            candidate_strategy=args.candidate_strategy,
            nl_pick_model=self._resolve_pick_model(args),
            **self._common_kwargs(args),
        )
        agent.pre_drive()

        state: dict[str, dict] = {tid: {"alive": False} for tid in task_ids}
        max_workers = min(len(task_ids), args.max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    self._run_phase_inner, agent, tid, round_num, 0,
                    None, None, "apply",
                ): tid
                for tid in task_ids
            }
            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    msgs = fut.result()
                    state[tid]["alive"] = msgs is not None
                except Exception:
                    traceback.print_exc()
                    state[tid]["alive"] = False

        out: dict[str, str] = {}
        for tid in task_ids:
            if state[tid]["alive"]:
                out[tid] = os.path.join(ws_root, tid, "submission")
            else:
                out[tid] = prev_submissions.get(tid, "")
        return out

    # ---- Round-1 coding seed (variance reduction) -------------------------

    def _seed_round1_coding(
        self,
        round_tids: list[str],
        args: argparse.Namespace,
    ) -> dict[str, str]:
        """Copy round_1 coding artifacts from a sibling backup tag."""
        seed_tag = args.seed_coding_tag
        src_root = os.path.join(
            args.project_dir, "backups", self.benchmark_name, seed_tag,
            "final", "round_1", "coding", "tasks",
        )
        if not os.path.isdir(src_root):
            sys.exit(
                f"--seed-coding-tag={seed_tag}: source not found: {src_root}"
            )

        missing = [tid for tid in round_tids
                   if not os.path.isdir(os.path.join(src_root, tid, "submission"))]
        if missing:
            sys.exit(
                f"--seed-coding-tag={seed_tag}: missing submission/ for "
                f"task_ids {missing} under {src_root}"
            )

        ws_root = self.phase_tasks_dir(args, 1, "coding")
        os.makedirs(ws_root, exist_ok=True)
        out: dict[str, str] = {}
        for tid in round_tids:
            src_host = os.path.join(src_root, tid)
            target_host = os.path.join(ws_root, tid)
            target_sub = os.path.join(target_host, "submission")
            os.makedirs(target_host, exist_ok=True)
            if os.path.isdir(target_sub) and os.listdir(target_sub):
                print(f"[PaperbenchFullRun] seed_round1: {tid} already staged; "
                      f"reusing {target_sub}")
            else:
                shutil.copytree(os.path.join(src_host, "submission"), target_sub,
                                dirs_exist_ok=True,
                                ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)
                # Carry paper from seed tag too, fall back to fresh snapshot
                # if absent.
                try:
                    self._carry_paper_to_phase(tid, src_host, target_host)
                except FileNotFoundError:
                    self._stage_paper_for_task(tid, target_host, args)
                print(f"[PaperbenchFullRun] seed_round1: {tid} ← {seed_tag}")
            out[tid] = target_sub
        return out

    # ---- Generic inner runner --------------------------------------------

    def _run_phase_inner(
        self,
        agent,
        task_id: str,
        round_num: int,
        step_num: int,
        feedback,
        messages,
        log_phase: str,
    ) -> list[dict] | None:
        # Route aux LLM usage (nl summaries + candidate picks, which bypass the
        # step logs) into round_<N>/aux_usage.jsonl so it is backed up alongside
        # the step logs. Phase/task are recoverable from the per-call context path.
        log_root = getattr(agent, "log_dir", None)
        if log_root:
            os.environ["AUX_USAGE_LOG"] = os.path.join(
                log_root, f"round_{round_num}", "aux_usage.jsonl")
            os.environ["AUX_USAGE_ROUND"] = str(round_num)
            os.environ["AUX_USAGE_PHASE"] = log_phase
        try:
            return agent.run(
                task_id=task_id,
                round_num=round_num,
                step_num=step_num,
                feedback=feedback,
                messages=messages,
                log_phase=log_phase,
            )
        except Exception:
            print(f"[PaperbenchFullRun] {log_phase} round={round_num} "
                  f"step={step_num} {task_id} failed:")
            traceback.print_exc()
            return None

    # ---- Top-level dispatch ----------------------------------------------

    def main(self, argv: list[str] | None = None) -> None:
        from dotenv import load_dotenv

        project_dir = resolve_project_dir()
        load_dotenv(dotenv_path=os.path.join(project_dir, ".env"))

        args = self.parse_args(argv)
        args.project_dir = project_dir
        self.setup_paths(args)

        task_ids = self.load_tasks(args)
        if not task_ids:
            sys.exit("No tasks to run")

        print(f"[Config] benchmark = {self.benchmark_name} (mode={args.mode})")
        print(f"[Config] tag       = {args.tag}")
        print(f"[Config] m         = {args.m}")
        print(f"[Config] tasks     = {task_ids}")
        print(f"[Config] runs_root = {args.runs_root}")
        print(f"[Config] log_root  = {args.log_root}")
        print(f"[Config] backup    = {args.backup_final_root}")
        print(f"[Config] provider  = {args.provider}")
        print(f"[Config] model     = {args.model}")
        print(f"[Config] docker    = {args.docker_image or '(LocalEnvironment)'}")
        print(f"[Config] data_dir  = {args.paperbench_data_dir_resolved}")
        print(f"[Config] seed_lib  = {args.source_library_dir or '(none)'}")
        print(f"[Config] seed_R1   = {args.seed_coding_tag or '(none)'}")

        if args.mode == "baseline":
            self._run_baseline(task_ids, args)
        elif args.mode == "librarian":
            # Librarian samples the *unified* PaperbenchLibraryAgent with none
            # of our SLA-specific candidate machinery (design doc §2). Force the
            # overrides here so the shell launcher stays thin.
            args.candidate_strategy = "none"
            args.library_candidate_strategy = "none"
            print("[Config] librarian → candidate_strategy=none, "
                  "library_candidate_strategy=none")
            self._run_librarian(task_ids, args)
        elif args.mode == "sla_naive":
            self._run_sla_naive(task_ids, args)
        elif args.mode == "sla_naive_split":
            # Ablation: sla_ours with local_extract / extract_map / candidates off.
            args.extract_map = False
            args.candidate_strategy = "none"
            os.environ["PAPERBENCH_LOCAL_EXTRACT"] = "0"
            print("[Config] sla_naive_split → extract_map=False, "
                  "candidate_strategy=none, PAPERBENCH_LOCAL_EXTRACT=0")
            self._run_sla_ours(task_ids, args)
        else:  # sla_ours
            self._run_sla_ours(task_ids, args)

    # ---- baseline orchestrator -------------------------------------------

    def _run_baseline(self, task_ids: list[str], args: argparse.Namespace) -> None:
        """Single round, every paper in parallel through PaperbenchCodingAgent."""
        round_num = 1
        print()
        print("#" * 60)
        print(f"# Baseline  round_1  tasks={task_ids}")
        print("#" * 60)

        seed_lib = self._resolve_seed_lib(args)
        if args.seed_coding_tag:
            self._seed_round1_coding(task_ids, args)
        else:
            self.run_coding_phase(round_num, task_ids, seed_lib, args)
        self.backup_round(round_num, args)

        print()
        print("#" * 60)
        print(f"# DONE  mode={args.mode}  tasks={len(task_ids)}")
        print("#" * 60)

    # ---- sla_naive orchestrator ------------------------------------------

    def _run_sla_naive(self, task_ids: list[str], args: argparse.Namespace) -> None:
        """Per round: coding (parallel M) → unified PaperbenchLibraryAgent."""
        rounds = _chunked(task_ids, args.m)
        current_lib_dir = self._resolve_seed_lib(args)
        cumulative_subs: dict[str, str] = {}

        for round_num, round_tids in enumerate(rounds, start=1):
            print()
            print("#" * 60)
            print(f"# {args.mode}  Round {round_num}/{len(rounds)}  tasks={round_tids}")
            print(f"#   current_lib = {current_lib_dir or '(none)'}")
            print("#" * 60)

            if round_num == 1 and args.seed_coding_tag:
                new_subs = self._seed_round1_coding(round_tids, args)
            else:
                new_subs = self.run_coding_phase(
                    round_num, round_tids, current_lib_dir, args,
                )
            cumulative_subs.update(new_subs)

            if len(cumulative_subs) < 2:
                print(f"[Round {round_num}] cumulative<2 → skip LibraryAgent")
                self.backup_round(round_num, args)
                continue

            apply_root = os.path.join(
                args.runs_root, f"round_{round_num}", "apply",
            )
            apply_tasks = os.path.join(apply_root, "tasks")
            apply_lib = os.path.join(apply_root, "lib")
            os.makedirs(apply_tasks, exist_ok=True)
            os.makedirs(apply_lib, exist_ok=True)

            # Seed apply/lib from prior round (upgrade mode).
            if (current_lib_dir and os.path.isdir(current_lib_dir)
                    and not os.listdir(apply_lib)):
                shutil.copytree(current_lib_dir, apply_lib,
                                dirs_exist_ok=True, ignore=_STAGE_IGNORE,
                                ignore_dangling_symlinks=True)

            staged_apps: dict[str, str] = {}
            for tid, src_sub in cumulative_subs.items():
                target_host = os.path.join(apply_tasks, tid)
                target_sub = os.path.join(target_host, "submission")
                os.makedirs(target_host, exist_ok=True)
                if not os.path.isdir(target_sub) or not os.listdir(target_sub):
                    if not os.path.isdir(src_sub):
                        print(f"[PaperbenchFullRun] sla_naive: missing app for "
                              f"{tid}: {src_sub}")
                        continue
                    shutil.copytree(src_sub, target_sub, dirs_exist_ok=True,
                                    ignore=_STAGE_IGNORE,
                                    ignore_dangling_symlinks=True)
                try:
                    self._carry_paper_to_phase(tid, os.path.dirname(src_sub), target_host)
                except FileNotFoundError:
                    self._stage_paper_for_task(tid, target_host, args)
                staged_apps[tid] = target_sub

            agent = PaperbenchLibraryAgent(
                workspace_root=apply_tasks,
                source_apps=staged_apps,
                library_dir=apply_lib,
                candidate_strategy=args.library_candidate_strategy,
                cluster_distance_threshold=args.extract_cluster_distance_threshold,
                cluster_min_mean_sim=args.extract_cluster_min_mean_sim,
                cluster_top_k=args.extract_cluster_top_k,
                enable_resume=False,
                nl_pick_model=self._resolve_pick_model(args),
                **self._common_kwargs(args),
            )
            msgs = self._run_phase_inner(
                agent, LIBRARY_TASK_ID, round_num, 0, None, None, "apply",
            )
            if msgs is None:
                print(f"[PaperbenchFullRun] sla_naive round={round_num} library "
                      f"agent failed")
            else:
                self._mirror_lib_to_tasks(apply_lib, apply_tasks, staged_apps)
                extract_lib = os.path.join(
                    args.runs_root, f"round_{round_num}", "extract", "lib",
                )
                os.makedirs(extract_lib, exist_ok=True)
                if not os.listdir(extract_lib):
                    shutil.copytree(apply_lib, extract_lib,
                                    dirs_exist_ok=True, ignore=_STAGE_IGNORE,
                                    ignore_dangling_symlinks=True)

            cumulative_subs.update(staged_apps)
            current_lib_dir = apply_lib
            self.backup_round(round_num, args)

        print()
        print("#" * 60)
        print(f"# DONE  mode={args.mode}  rounds={len(rounds)}  tasks={len(task_ids)}")
        print(f"#   final_lib  = {current_lib_dir or '(none)'}")
        print("#" * 60)

    # ---- librarian orchestrator ------------------------------------------

    def _run_librarian(self, task_ids: list[str], args: argparse.Namespace) -> None:
        """Librarian post-hoc baseline (single round, no chunking).

        ZS-seeded corpus → gate ORIGINALS once (gate_before) → K
        PaperbenchLibraryAgent samples (temperature>0), each gated + 1 repair
        turn → MDL rerank → promote winner into the standard apply/extract
        layout → backup. Peer of WebgenFullRun._run_librarian; the only
        benchmark-specific differences are (a) the static compile/import gate
        replaces the docker vite build, (b) each sample stages a paper/ snapshot
        beside every submission (PaperbenchLibraryAgent requires it), and (c)
        no node_modules cleanup is needed (the gate runs no containers).
        """
        if not args.seed_coding_tag:
            sys.exit(
                "--mode librarian requires --seed-coding-tag: the corpus must "
                "be seeded from a Zero-Shot baseline backup "
                "(backups/paperbench/<tag>/final/round_1/coding/)."
            )
        if args.temperature <= 0:
            sys.exit(
                "--mode librarian requires --temperature > 0 "
                f"(got {args.temperature}): temperature is the ONLY diversity "
                "source across the K samples — at 0 every candidate is "
                "near-identical and sample-and-rerank degenerates. "
                "Set TEMPERATURE (e.g. 0.8)."
            )
        K = int(args.librarian_k)
        if K < 1:
            sys.exit(f"--librarian-k must be >= 1 (got {K})")

        round_num = 1
        print()
        print("#" * 60)
        print(f"# librarian  round_1  K={K}  temp={args.temperature}  "
              f"tasks={task_ids}")
        print(f"#   seed_coding = {args.seed_coding_tag}")
        print("#" * 60)

        # --- 0) seed corpus (coding cost 0, identical to ZS) --------------
        new_subs = self._seed_round1_coding(task_ids, args)
        round_dir = os.path.join(args.runs_root, f"round_{round_num}")

        # --- 1) gate ORIGINALS once → gate_before (⊆ baseline) ------------
        # No lib yet, so the gate runs compile + internal-import checks only.
        # Tasks with a latent bug in the ZS snapshot (e.g. c0 `fre` imports an
        # undefined `Critic`) fail here and are therefore EXCLUDED from the ⊆
        # requirement — candidates are not penalized for a pre-existing defect.
        coding_tasks_dir = self.phase_tasks_dir(args, round_num, "coding")
        print(f"[librarian] gating ZS originals under {coding_tasks_dir}")
        before_results = run_paperbench_static_gate(coding_tasks_dir, None)
        gate_before = {tid for tid, r in before_results.items() if r.ok}
        excluded = sorted(t for t in before_results if t not in gate_before)
        print(f"[librarian] gate_before = {len(gate_before)}/"
              f"{len(before_results)} compile+import: {sorted(gate_before)}")
        if excluded:
            print(f"[librarian] gate_before EXCLUDED (failed as ZS originals, "
                  f"⊆ rule drops them): {excluded}")
        self._write_gate_json(
            os.path.join(round_dir, "gate_before.json"), before_results,
        )

        # --- 2) sample K candidates in parallel, gate + repair each -------
        # Samples are fully independent (separate workspaces / lib dirs / log
        # phases). The static gate is sub-second and container-free, so the only
        # docker concurrency is the LibraryAgent itself: SAMPLE_WORKERS agents
        # run at once (mirrors webgen's fan-out; gate parallelism is moot).
        samples: dict[int, SamplePaths] = {}
        gate_results: dict[int, dict[str, bool]] = {}
        samples_root = os.path.join(round_dir, "samples")

        budget = max(1, int(args.max_workers))
        sample_workers = min(K, max(1, budget // 2))
        print(f"[librarian] sampling {K} candidates: {sample_workers} "
              f"PaperbenchLibraryAgents in parallel")

        with ThreadPoolExecutor(max_workers=sample_workers) as ex:
            futures = {
                ex.submit(
                    self._run_one_sample, k, K, samples_root, new_subs,
                    gate_before, round_num, args,
                ): k
                for k in range(1, K + 1)
            }
            for fut in as_completed(futures):
                k = futures[fut]
                try:
                    sample, gate = fut.result()
                    samples[k] = sample
                    gate_results[k] = gate
                except Exception:
                    # Should not happen (_run_one_sample catches its own
                    # errors), but keep one bad sample from killing the loop.
                    print(f"[librarian] sample {k} unexpected error:")
                    traceback.print_exc()
                    sample_dir = os.path.join(samples_root, f"sample_{k}")
                    samples[k] = SamplePaths(
                        k=k,
                        tasks_dir=os.path.join(sample_dir, "tasks"),
                        lib_dir=os.path.join(sample_dir, "lib"),
                        order=k,
                    )
                    gate_results[k] = {tid: False for tid in gate_before}

        # --- 3) MDL rerank → winner --------------------------------------
        base_url = os.environ.get("MDL_BASE_URL") or None
        print()
        print("[librarian] MDL rerank over "
              f"{len(samples)} candidates (base_url={base_url or 'default'})")
        report = select_winner(
            samples, gate_results, gate_before,
            task="paperbench", base_url=base_url,
        )
        report_path = os.path.join(round_dir, "rerank_report.json")
        report.save(report_path)
        print(f"[librarian] wrote {report_path}")
        for c in report.candidates:
            mdl = "nan" if c.mdl_total != c.mdl_total else f"{c.mdl_total:.1f}"
            print(f"    sample {c.k} (order {c.order}): rank={c.rank} "
                  f"superset={c.superset} pass={c.pass_count}/{len(gate_before)} "
                  f"mdl={mdl}")
        winner_k = report.winner_k
        print(f"[librarian] winner = sample {winner_k}")

        # --- 4) promote winner into standard apply/extract layout ---------
        if winner_k is not None and winner_k in samples:
            self._promote_librarian_winner(
                round_num, samples[winner_k], task_ids, args,
            )
        else:
            print("[librarian] no winner to promote (all candidates failed)")

        self.backup_round(round_num, args)

        print()
        print("#" * 60)
        print(f"# DONE  mode=librarian  K={K}  tasks={len(task_ids)}  "
              f"winner=sample_{winner_k}")
        print("#" * 60)

    def _run_one_sample(
        self,
        k: int,
        K: int,
        samples_root: str,
        new_subs: dict[str, str],
        gate_before: set[str],
        round_num: int,
        args: argparse.Namespace,
    ) -> tuple[SamplePaths, dict[str, bool]]:
        """Sample one candidate: LibraryAgent → static gate → 1 repair turn.

        Runs inside a ThreadPoolExecutor worker (see ``_run_librarian``). Fully
        self-contained: its own workspace/lib/log-phase. Returns
        ``(SamplePaths, {tid: passed})``. Never raises — an internal failure
        degrades to a gate-failed candidate (mirrors webgen).
        """
        print(f"[librarian] sample {k}/{K}: start")
        sample_dir = os.path.join(samples_root, f"sample_{k}")
        sample_tasks = os.path.join(sample_dir, "tasks")
        sample_lib = os.path.join(sample_dir, "lib")
        os.makedirs(sample_tasks, exist_ok=True)
        os.makedirs(sample_lib, exist_ok=True)

        sample = SamplePaths(
            k=k, tasks_dir=sample_tasks, lib_dir=sample_lib, order=k,
        )
        log_phase = f"sample_{k}"

        try:
            staged_apps = self._stage_apps_into(new_subs, sample_tasks, args)
            agent = PaperbenchLibraryAgent(
                workspace_root=sample_tasks,
                source_apps=staged_apps,
                library_dir=sample_lib,
                candidate_strategy="none",
                cluster_distance_threshold=args.extract_cluster_distance_threshold,
                cluster_min_mean_sim=args.extract_cluster_min_mean_sim,
                cluster_top_k=args.extract_cluster_top_k,
                enable_resume=False,
                nl_pick_model=self._resolve_pick_model(args),
                **self._common_kwargs(args),
            )
            msgs = self._run_phase_inner(
                agent, LIBRARY_TASK_ID, round_num, 0, None, None, log_phase,
            )

            if msgs is None:
                # LibraryAgent failed outright → gate-failed everywhere.
                print(f"[librarian] sample {k}: LibraryAgent returned None; "
                      f"marking gate-failed for all gate_before apps")
                self._write_sample_gate_result(
                    sample_dir, {}, repaired=False, agent_failed=True,
                )
                return sample, {tid: False for tid in gate_before}

            # Scope the gate to the real staged apps: the LibraryAgent creates a
            # ``__library__/`` working dir under sample_tasks (for agent.env),
            # which the static gate would otherwise treat as a submission-less
            # task and always fail. gate_before ⊆ staged_apps, so this loses no
            # coverage.
            only_apps = set(staged_apps)
            gate_lib = _gate_lib_root(sample_lib)
            results = run_paperbench_static_gate(
                sample_tasks, gate_lib, only=only_apps,
            )
            failed = {
                tid: _gate_error_tail(results[tid])
                for tid in gate_before
                if tid in results and not results[tid].ok
            }
            repaired = False
            if failed:
                print(f"[librarian] sample {k}: {len(failed)} gate_before "
                      f"app(s) failed: {sorted(failed)} → repair turn")
                self._run_librarian_repair(
                    round_num, sample_tasks, sample_lib, staged_apps,
                    failed, msgs, log_phase, args,
                )
                repaired = True
                # A lib fix can break previously-passing apps → re-gate ALL.
                results = run_paperbench_static_gate(
                    sample_tasks, gate_lib, only=only_apps,
                )

            n_ok = sum(1 for r in results.values() if r.ok)
            n_before_ok = sum(
                1 for tid in gate_before if results.get(tid) and results[tid].ok
            )
            print(f"[librarian] sample {k}: gate {n_ok}/{len(results)} "
                  f"compile+import; gate_before-preserved "
                  f"{n_before_ok}/{len(gate_before)}; repaired={repaired}")
            self._write_sample_gate_result(
                sample_dir, results, repaired=repaired, agent_failed=False,
            )
            return sample, {tid: r.ok for tid, r in results.items()}
        except Exception:
            print(f"[librarian] sample {k}: worker crashed:")
            traceback.print_exc()
            self._write_sample_gate_result(
                sample_dir, {}, repaired=False, agent_failed=True,
            )
            return sample, {tid: False for tid in gate_before}

    def _stage_apps_into(
        self, src_subs: dict[str, str], dest_tasks: str,
        args: argparse.Namespace,
    ) -> dict[str, str]:
        """Copy each app's submission + paper/ into ``dest_tasks/<tid>/``.

        Mirrors _run_sla_naive's staging (submission via ``_STAGE_IGNORE``,
        paper carried from the source phase dir with a fresh-snapshot
        fallback). PaperbenchLibraryAgent.setup_workspace REQUIRES a paper/
        sibling next to every submission, so unlike webgen this stages both.
        Returns the staged submission paths (the agent edits these in place).
        """
        staged: dict[str, str] = {}
        for tid, src_sub in src_subs.items():
            target_host = os.path.join(dest_tasks, tid)
            target_sub = os.path.join(target_host, "submission")
            os.makedirs(target_host, exist_ok=True)
            if not os.path.isdir(target_sub) or not os.listdir(target_sub):
                if not os.path.isdir(src_sub):
                    print(f"[librarian] staging: missing source app for "
                          f"{tid}: {src_sub}")
                    continue
                shutil.copytree(src_sub, target_sub, dirs_exist_ok=True,
                                ignore=_STAGE_IGNORE,
                                ignore_dangling_symlinks=True)
            try:
                self._carry_paper_to_phase(
                    tid, os.path.dirname(src_sub), target_host,
                )
            except FileNotFoundError:
                self._stage_paper_for_task(tid, target_host, args)
            staged[tid] = target_sub
        return staged

    def _run_librarian_repair(
        self,
        round_num: int,
        sample_tasks: str,
        sample_lib: str,
        staged_apps: dict[str, str],
        failed: dict[str, str],
        prior_messages: list[dict],
        log_phase: str,
        args: argparse.Namespace,
    ) -> list[dict] | None:
        """One repair turn: resume the LibraryAgent with the gate errors.

        NOTE: a resume-enabled agent is required — BaseCodingAgent.run only
        replays ``messages`` when ``enable_resume=True``. Mirrors
        webgen's _run_librarian_repair; only the gate_name differs.
        """
        try:
            feedback = build_librarian_repair_prompt(
                failed, gate_name="static compile/import check",
            )
            agent = PaperbenchLibraryAgent(
                workspace_root=sample_tasks,
                source_apps=staged_apps,
                library_dir=sample_lib,
                candidate_strategy="none",
                cluster_distance_threshold=args.extract_cluster_distance_threshold,
                cluster_min_mean_sim=args.extract_cluster_min_mean_sim,
                cluster_top_k=args.extract_cluster_top_k,
                enable_resume=True,
                nl_pick_model=self._resolve_pick_model(args),
                **self._common_kwargs(args),
            )
            return self._run_phase_inner(
                agent, LIBRARY_TASK_ID, round_num, 1,
                feedback, prior_messages, log_phase,
            )
        except Exception:
            print(f"[librarian] repair turn ({log_phase}) failed:")
            traceback.print_exc()
            return None

    def _promote_librarian_winner(
        self,
        round_num: int,
        winner: SamplePaths,
        task_ids: list[str],
        args: argparse.Namespace,
    ) -> None:
        """Copy the winning candidate into round_1/apply + extract (sla_naive
        layout) so eval/metrics tooling reads it unchanged. Carries each app's
        paper/ snapshot too, matching what run_apply_phase would leave."""
        apply_root = os.path.join(args.runs_root, f"round_{round_num}", "apply")
        apply_tasks = os.path.join(apply_root, "tasks")
        apply_lib = os.path.join(apply_root, "lib")
        os.makedirs(apply_tasks, exist_ok=True)
        os.makedirs(apply_lib, exist_ok=True)

        promoted: list[str] = []
        for tid in task_ids:
            src_host = os.path.join(winner.tasks_dir, tid)
            src = os.path.join(src_host, "submission")
            target_host = os.path.join(apply_tasks, tid)
            dst = os.path.join(target_host, "submission")
            if not os.path.isdir(src):
                print(f"[librarian] promote: missing winner app for {tid}")
                continue
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            os.makedirs(target_host, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)
            # Carry paper/ so the apply layout matches sla_naive exactly.
            try:
                self._carry_paper_to_phase(tid, src_host, target_host)
            except FileNotFoundError:
                self._stage_paper_for_task(tid, target_host, args)
            promoted.append(tid)

        if winner.lib_dir and os.path.isdir(winner.lib_dir) and os.listdir(winner.lib_dir):
            if os.listdir(apply_lib):
                shutil.rmtree(apply_lib)
                os.makedirs(apply_lib, exist_ok=True)
            shutil.copytree(winner.lib_dir, apply_lib, dirs_exist_ok=True,
                            ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)

            # Per-task lib mirror + extract/lib mirror (exactly like sla_naive).
            self._mirror_lib_to_tasks(apply_lib, apply_tasks, promoted)
            extract_lib = os.path.join(
                args.runs_root, f"round_{round_num}", "extract", "lib",
            )
            os.makedirs(extract_lib, exist_ok=True)
            if not os.listdir(extract_lib):
                shutil.copytree(apply_lib, extract_lib, dirs_exist_ok=True,
                                ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)
        print(f"[librarian] promoted winner (sample {winner.k}) → "
              f"{apply_root} ({len(promoted)} apps)")

    def _write_gate_json(self, path: str, results: dict) -> None:
        from dataclasses import asdict
        payload = {tid: asdict(r) for tid, r in results.items()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2))

    def _write_sample_gate_result(
        self, sample_dir: str, results: dict, *, repaired: bool,
        agent_failed: bool,
    ) -> None:
        from dataclasses import asdict
        payload = {
            "agent_failed": agent_failed,
            "repaired": repaired,
            "apps": {tid: asdict(r) for tid, r in results.items()},
        }
        os.makedirs(sample_dir, exist_ok=True)
        Path(os.path.join(sample_dir, "gate_result.json")).write_text(
            json.dumps(payload, indent=2)
        )

    # ---- sla_ours orchestrator -------------------------------------------

    def _run_sla_ours(self, task_ids: list[str], args: argparse.Namespace) -> None:
        """Per round: coding → local_extract (new) → global_extract → apply."""
        rounds = _chunked(task_ids, args.m)
        current_lib_dir = self._resolve_seed_lib(args)
        cumulative_subs: dict[str, str] = {}

        for round_num, round_tids in enumerate(rounds, start=1):
            print()
            print("#" * 60)
            print(f"# {args.mode}  Round {round_num}/{len(rounds)}  tasks={round_tids}")
            print(f"#   current_lib = {current_lib_dir or '(none)'}")
            print("#" * 60)

            args._new_ids_this_round = round_tids

            if round_num == 1 and args.seed_coding_tag:
                new_subs = self._seed_round1_coding(round_tids, args)
            else:
                new_subs = self.run_coding_phase(
                    round_num, round_tids, current_lib_dir, args,
                )

            local_extract_enabled = (
                os.environ.get("PAPERBENCH_LOCAL_EXTRACT", "1") != "0"
            )
            if local_extract_enabled and new_subs:
                try:
                    post_local = self.run_local_extract_phase(
                        round_num, new_subs, current_lib_dir, args,
                    )
                    new_subs = {**new_subs, **post_local}
                except Exception:
                    print(f"[Round {round_num}] local_extract failed "
                          f"(continuing with pre-Local subs):")
                    traceback.print_exc()

            cumulative_subs.update(new_subs)

            if len(cumulative_subs) >= 2:
                new_lib, post_extract_subs = self.run_extract_phase(
                    round_num, dict(cumulative_subs), current_lib_dir, args,
                )
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
                print(f"[Round {round_num}] cumulative<2 → skip global_extract+apply"
                      f" (local_extract still ran)")

            self.backup_round(round_num, args)

        print()
        print("#" * 60)
        print(f"# DONE  mode={args.mode}  rounds={len(rounds)}  tasks={len(task_ids)}")
        print(f"#   final_lib  = {current_lib_dir or '(none)'}")
        print("#" * 60)

    # ---- Misc -------------------------------------------------------------

    def _mirror_lib_to_tasks(
        self,
        lib_dir: str,
        tasks_root: str,
        task_ids,
    ) -> None:
        """Copy a phase-level lib into each task's per-task lib slot.

        Same shape as WebgenFullRun._mirror_lib_to_tasks; sla_naive's library
        agent writes the lib to the phase-level apply/lib, while sla_ours's
        apply produces per-task apply/tasks/<tid>/lib. This bridges so the
        eval chain finds lib/ in either layout.
        """
        if not os.path.isdir(lib_dir) or not os.listdir(lib_dir):
            return
        for tid in task_ids:
            dst = os.path.join(tasks_root, tid, "lib")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isdir(dst) and os.listdir(dst):
                continue
            shutil.copytree(lib_dir, dst, dirs_exist_ok=True,
                            ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)

    def _resolve_pick_model(self, args: argparse.Namespace) -> str:
        """Effective NL pick model: ``--model`` verbatim (picker follows the
        coding backbone) unless ``--nl-pick-model`` overrides. Routing lives in
        ``utils/candidates/nl.py:_llm_pick`` (OpenAI-family → OpenAI, else
        OpenRouter)."""
        explicit = (args.nl_pick_model or "").strip()
        if explicit and explicit.lower() != "auto":
            return explicit
        return (args.model or "").strip()

    def _resolve_seed_lib(self, args: argparse.Namespace) -> str | None:
        if not args.source_library_dir:
            return None
        seed = os.path.abspath(args.source_library_dir)
        if not os.path.isdir(seed):
            sys.exit(f"--source-library-dir not found: {seed}")
        return seed


if __name__ == "__main__":
    PaperbenchFullRun().main()
