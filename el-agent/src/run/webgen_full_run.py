"""WebgenFullRun — WebGen-Bench round-based orchestrator.

Four modes, dispatched via ``--mode``:

  baseline           Single round: every task runs through WebgenCodingAgent
                     in parallel. No library produced.

  sla_naive          Per round: (a) coding (parallel M new tasks) →
                     (b) WebgenLibraryAgent over cumulative apps (single
                         shot; extract + apply in one).
                     Mirrors cc-exp `experiments/cc_exp/runner/modes/sla_naive.py`.

  sla_naive_split    Per round: (a) coding [parallel M] →
                     (b) global_extract over cumulative apps →
                     (c) apply [parallel, cumulative].
                     Ablation of sla_ours: no local_extract, no extract_map,
                     no extract/apply candidates. Isolates the effect of
                     splitting sla_naive's unified library agent into
                     extract + apply phases.

  sla_ours           Per round: (a) coding [parallel M] →
                     (b) local_extract per *new* task [parallel M] →
                     (c) global_extract over cumulative apps →
                     (d) apply [parallel, cumulative].
                     NL candidates + optional extract_map.md. Mirrors cc-exp
                     `experiments/cc_exp/runner/modes/sla_ours.py`.

Workspace layout (matches RAL's full layout; eval_webgen.sh reads it):
    runs/webgen/<tag>/round_<i>/{coding,local_extract,extract,apply}/{tasks/<id>/, lib?/}
    runs/webgen/<tag>/logs/round_<i>/<phase>/...
    backups/webgen/<tag>/final/round_<i>/<phase>/...
    backups/webgen/<tag>/logs/round_<i>/<phase>/...

K-feedback (regression test inner loop) is plumbed but disabled: webgen
has no pytest harness yet. ``--k 0`` is the only validated value.
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

# Allow `python -m run.webgen_full_run` from el-agent/src.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mswe_agents.webgen import (  # noqa: E402
    WebgenApplyAgent,
    WebgenCodingAgent,
    WebgenGlobalExtractAgent,
    WebgenLibraryAgent,
    WebgenLocalExtractAgent,
)
from mswe_agents.webgen.global_extract_agent import EXTRACT_TASK_ID  # noqa: E402
from mswe_agents.webgen.library_agent import LIBRARY_TASK_ID  # noqa: E402
from prompts.common import EXTRACT_MAP_INSTRUCTION  # noqa: E402
from prompts.common.librarian_repair import build_librarian_repair_prompt  # noqa: E402
from run.base_full_run import BaseFullRun, _STAGE_IGNORE, _chunked  # noqa: E402
from run.base_run import resolve_project_dir  # noqa: E402
from run.librarian_select import SamplePaths, select_winner  # noqa: E402
from utils.gates import run_webgen_build_gate  # noqa: E402
from utils.gates.webgen_build import _chown_to_host  # noqa: E402


WEBGEN_BENCH_ROOT_REL = "data/WebGen-Bench"
DEFAULT_TASK_FILE_REL = "data/WebGen-Bench/data/test.jsonl"
DEFAULT_CLUSTER_JSON_REL = "data/augments/webgen/cluster/cluster.json"


class WebgenFullRun(BaseFullRun):
    benchmark_name = "webgen"

    # ---- CLI extension ----------------------------------------------------

    def parse_extra_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--mode",
            required=True,
            choices=["baseline", "sla_naive", "sla_naive_split", "sla_ours",
                     "librarian"],
            help="Pipeline variant — baseline (coding only), sla_naive "
                 "(coding + unified LibraryAgent), sla_naive_split (coding + "
                 "global_extract + apply, no candidates/extract_map/local_extract), "
                 "sla_ours (4-phase with local + global extract + apply), "
                 "librarian (single round: ZS-seeded corpus → K LibraryAgent "
                 "samples → build gate + repair → MDL rerank → promote winner).",
        )
        parser.add_argument(
            "--librarian-k",
            type=int,
            default=8,
            help="librarian mode: number of library+refactor candidates to "
                 "sample (temperature>0). Winner chosen by build gate + MDL "
                 "rerank. Default 8.",
        )
        parser.add_argument(
            "--task-file",
            default=None,
            help=f"Path to WebGen-Bench JSONL. Default: <project>/{DEFAULT_TASK_FILE_REL}",
        )
        parser.add_argument(
            "--task-list",
            default=None,
            help="Comma-separated task ids; overrides --cluster-id.",
        )
        parser.add_argument(
            "--cluster-id",
            type=int,
            default=None,
            help="Cluster id from data/augments/webgen/cluster/cluster.json.",
        )
        parser.add_argument(
            "--cluster-size",
            type=int,
            default=None,
            help="Take the first N tasks from the chosen cluster (centroid-"
                 "distance ascending). Default: full cluster.",
        )
        parser.add_argument(
            "--cluster-json",
            default=None,
            help=f"Path to cluster.json. Default: <project>/{DEFAULT_CLUSTER_JSON_REL}",
        )
        parser.add_argument(
            "--k", type=int, default=0,
            help="Apply-phase feedback iterations. Currently webgen has no "
                 "build/regression-test feedback adapter — keep 0.",
        )
        parser.add_argument(
            "--candidate-strategy",
            choices=("embed", "nl", "none"),
            default="nl",
            help="Apply/extract candidate retrieval backend. Default nl. "
                 "'none' skips candidate retrieval entirely (used by "
                 "sla_naive_split).",
        )
        parser.add_argument(
            "--nl-pick-model",
            default="",
            help="LLM used for NL candidate selection when "
                 "--candidate-strategy=nl. Empty/auto = derive from --model "
                 "by stripping the litellm provider prefix (e.g. "
                 "`deepseek/deepseek-v4-flash` → `deepseek-v4-flash`). NL summary model "
                 "is hardcoded to gpt-5.4-nano (agent constructor default).",
        )
        # GlobalExtract clustering knobs (match RAL defaults).
        parser.add_argument(
            "--extract-cluster-distance-threshold",
            type=float, default=1.2,
        )
        parser.add_argument(
            "--extract-cluster-min-mean-sim",
            type=float, default=0.50,
        )
        parser.add_argument(
            "--extract-cluster-top-k",
            type=int, default=10,
        )
        parser.add_argument(
            "--extract-map",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="sla_ours: after extract, resume the agent with a "
                 "map-writing instruction so it produces lib/extract_map.md "
                 "(injected into apply prompt).",
        )
        # LocalExtract knobs.
        parser.add_argument(
            "--local-extract-distance-threshold",
            type=float, default=1.0,
        )
        parser.add_argument(
            "--local-extract-min-mean-sim",
            type=float, default=0.55,
        )
        parser.add_argument(
            "--local-extract-top-k",
            type=int, default=12,
        )
        # sla_naive optional candidate strategy.
        parser.add_argument(
            "--library-candidate-strategy",
            choices=("none", "embed", "nl"),
            default="none",
            help="sla_naive: optional candidate list passed to the unified "
                 "WebgenLibraryAgent. Default 'none' (agent explores apps via bash).",
        )
        parser.add_argument(
            "--reasoning-effort",
            choices=("low", "medium", "high"),
            default=None,
            help="Override the model's reasoning_effort. None (default) lets "
                 "_factory.build_model pick 'high' for reasoning families. Set "
                 "'low' for grid smokes where wall-clock matters more than depth.",
        )
        parser.add_argument(
            "--seed-coding-tag",
            default=None,
            help="If set, skip round-1 coding generation and seed it from "
                 "backups/webgen/<seed-coding-tag>/final/round_1/coding/tasks/. "
                 "Used to share a fixed baseline R1 across sla_naive / sla_ours "
                 "trials so downstream phases differ only in lib pipeline.",
        )
        parser.add_argument(
            "--layout-specs-dir",
            default=None,
            help="Optional path to a per-task layout-spec directory. When set, "
                 "WebgenCodingAgent reads `<dir>/<task_id>.md` and appends it to "
                 "the task instruction under [Visual & functional layout "
                 "reference]. Missing file → vanilla baseline. "
                 "Convention: data/augments/webgen/layout_specs/",
        )

    # ---- Task loading -----------------------------------------------------

    def load_tasks(self, args: argparse.Namespace) -> list[str]:
        task_file = args.task_file or os.path.join(
            args.project_dir, DEFAULT_TASK_FILE_REL,
        )
        if not os.path.isfile(task_file):
            sys.exit(f"--task-file not found: {task_file}")
        args.task_file_resolved = os.path.abspath(task_file)

        task_map: dict[str, dict] = {}
        with open(task_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tid = row.get("id")
                if not tid:
                    continue
                task_map[str(tid)] = row
        if not task_map:
            sys.exit(f"No tasks loaded from {task_file}")
        args.task_map = task_map

        if args.task_list:
            requested = [t.strip() for t in args.task_list.split(",") if t.strip()]
        elif args.cluster_id is not None:
            requested = self._tasks_from_cluster(args)
        else:
            sys.exit(
                "Either --task-list or --cluster-id must be provided "
                "(no default split for WebGen-Bench)."
            )

        missing = [t for t in requested if t not in task_map]
        if missing:
            sys.exit(
                f"Unknown task ids: {missing}. "
                f"First few known ids: {sorted(task_map)[:10]}"
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
        return lambda tid: args.task_map[tid]

    def _common_kwargs(self, args: argparse.Namespace) -> dict:
        """Args shared by all webgen agent constructors.

        ``timeout`` is bumped to 300s (vs BaseCodingAgent's 120s default)
        because webgen's `npm install && npx vite build` regularly takes
        2–4 minutes on a cold cache. ``reasoning_effort=None`` lets
        ``_factory.build_model`` pick by model family (reasoning → "high").
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
            timeout=300,
            reasoning_effort=(args.reasoning_effort or None),
        )

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
        for tid in task_ids:
            os.makedirs(os.path.join(ws_root, tid, "submission"), exist_ok=True)

        agent = WebgenCodingAgent(
            workspace_root=ws_root,
            library_dir=library_dir,
            layout_specs_dir=getattr(args, "layout_specs_dir", None),
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
        """Per-app intra-extract over the M *new* tasks only.

        Carry-forward apps are not re-processed (matches cc-exp sla_ours).
        Returns post-LocalExtract submission paths for the new tasks.
        """
        ws_root = self.phase_tasks_dir(args, round_num, "local_extract")
        os.makedirs(ws_root, exist_ok=True)

        staged: dict[str, str] = {}
        for tid, src in new_app_paths.items():
            sub_dst = os.path.join(ws_root, tid, "submission")
            if not os.path.isdir(sub_dst) or not os.listdir(sub_dst):
                if not os.path.isdir(src):
                    print(f"[WebgenFullRun] local_extract: missing source app "
                          f"for {tid}: {src}")
                    continue
                os.makedirs(os.path.dirname(sub_dst), exist_ok=True)
                shutil.copytree(src, sub_dst, dirs_exist_ok=True,
                                ignore=_STAGE_IGNORE,
                                ignore_dangling_symlinks=True)
            staged[tid] = sub_dst

        if not staged:
            print(f"[WebgenFullRun] round {round_num} local_extract: nothing "
                  f"staged; skipping.")
            return new_app_paths

        global_lib_for_local = (
            seed_lib_dir if seed_lib_dir and os.path.isdir(seed_lib_dir)
            else None
        )

        agent = WebgenLocalExtractAgent(
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
                    print(f"[WebgenFullRun] local_extract round={round_num} "
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
        """sla_ours: LocalExtract (new tasks) → GlobalExtract (cumulative).

        ``source_app_paths`` is the cumulative dict; ``args._new_ids_this_round``
        names which entries were freshly produced by this round's coding phase.
        Only those are re-processed by LocalExtract (carry-forward apps stay
        with their post-apply state from the prior round).
        """
        local_extract_enabled = (
            os.environ.get("WEBGEN_LOCAL_EXTRACT", "1") != "0"
        )
        new_ids: list[str] = list(getattr(args, "_new_ids_this_round", []) or [])
        new_subs = {tid: source_app_paths[tid] for tid in new_ids
                    if tid in source_app_paths}

        if local_extract_enabled and new_subs:
            try:
                post_local = self.run_local_extract_phase(
                    round_num, new_subs, seed_lib_dir, args,
                )
                source_app_paths = {**source_app_paths, **post_local}
            except Exception:
                print(f"[WebgenFullRun] round {round_num} local_extract failed "
                      f"(continuing with pre-Local source apps):")
                traceback.print_exc()

        phase_dir = self.phase_workspace(args, round_num, "extract")
        ws_root = self.phase_tasks_dir(args, round_num, "extract")
        lib_dir = os.path.join(phase_dir, "lib")
        os.makedirs(lib_dir, exist_ok=True)
        os.makedirs(ws_root, exist_ok=True)

        if seed_lib_dir and os.path.isdir(seed_lib_dir) and not os.listdir(lib_dir):
            shutil.copytree(seed_lib_dir, lib_dir, dirs_exist_ok=True,
                            ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)

        staged_apps: dict[str, str] = {}
        for tid, src in source_app_paths.items():
            sub_dst = os.path.join(ws_root, tid, "submission")
            if not os.path.isdir(sub_dst) or not os.listdir(sub_dst):
                if not os.path.isdir(src):
                    print(f"[WebgenFullRun] extract: missing source app for "
                          f"{tid}: {src}")
                    continue
                os.makedirs(os.path.dirname(sub_dst), exist_ok=True)
                shutil.copytree(src, sub_dst, dirs_exist_ok=True,
                                ignore=_STAGE_IGNORE,
                                ignore_dangling_symlinks=True)
            staged_apps[tid] = sub_dst

        if len(staged_apps) < 2:
            print(f"[WebgenFullRun] round {round_num} extract: <2 staged apps "
                  f"({list(staged_apps)}); skipping.")
            return (seed_lib_dir if seed_lib_dir else None, source_app_paths)

        agent = WebgenGlobalExtractAgent(
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
            print(f"[WebgenFullRun] round {round_num} extract failed "
                  f"(continuing with whatever lib exists)")

        # Optional map-writing turn: resume the agent so it writes
        # lib/extract_map.md (appended to the same step_0.json).
        if msgs is not None and args.extract_map:
            map_msgs = self._run_extract_map_turn(
                round_num, lib_dir, staged_apps, msgs, args,
            )
            if map_msgs is None:
                print(f"[WebgenFullRun] round {round_num} extract_map turn "
                      f"failed (continuing without map).")

        if not os.path.isdir(lib_dir) or not os.listdir(lib_dir):
            print(f"[WebgenFullRun] round {round_num} extract produced empty lib")
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
            agent = WebgenGlobalExtractAgent(
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
            print(f"[WebgenFullRun] map turn round={round_num} failed:")
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
            host_dir = os.path.join(ws_root, tid)
            sub_dir = os.path.join(host_dir, "submission")
            os.makedirs(host_dir, exist_ok=True)
            if not os.path.isdir(sub_dir) or not os.listdir(sub_dir):
                src = prev_submissions.get(tid)
                if not src or not os.path.isdir(src):
                    print(f"[WebgenFullRun] apply: missing prev submission "
                          f"for {tid}: {src}")
                    continue
                shutil.copytree(src, sub_dir, dirs_exist_ok=True,
                                ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)

        # Carry-forward (WEBGEN_APPLY_CARRY_FORWARD=1): resume each app's apply
        # from its prior-round trajectory so it reuses already-read app files
        # instead of re-cat-ing them from scratch every round. resume_rebuilds_
        # prompt=True makes the resumed run still receive THIS round's freshly
        # built apply prompt (current candidates / lib state) as its new task.
        carry_forward = os.environ.get(
            "WEBGEN_APPLY_CARRY_FORWARD", "0"
        ).lower() not in ("0", "false", "no", "off", "")
        if not hasattr(self, "_apply_msgs"):
            self._apply_msgs: dict[str, list] = {}

        agent = WebgenApplyAgent(
            workspace_root=ws_root,
            library_dir=lib_dir,
            enable_resume=carry_forward,
            resume_rebuilds_prompt=carry_forward,
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
                    None, self._apply_msgs.get(tid) if carry_forward else None,
                    "apply",
                ): tid
                for tid in task_ids
            }
            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    msgs = fut.result()
                    state[tid]["alive"] = msgs is not None
                    if carry_forward and msgs:
                        self._apply_msgs[tid] = msgs
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
        """Copy round_1 coding artifacts from a sibling backup tag.

        Used to share a fixed R1/C across sla_naive / sla_ours trials so the
        downstream lib pipeline is the only differing factor. The seeded paths
        live under this run's own ``runs/webgen/<tag>/round_1/coding/`` so
        carry-forward staging in later rounds doesn't reach into another tag.
        """
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
            src = os.path.join(src_root, tid, "submission")
            dst = os.path.join(ws_root, tid, "submission")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isdir(dst) and os.listdir(dst):
                print(f"[WebgenFullRun] seed_round1: {tid} already staged; "
                      f"reusing {dst}")
            else:
                shutil.copytree(src, dst, dirs_exist_ok=True,
                                ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)
                print(f"[WebgenFullRun] seed_round1: {tid} ← {seed_tag}")
            out[tid] = dst
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
            print(f"[WebgenFullRun] {log_phase} round={round_num} "
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
        print(f"[Config] seed_lib  = {args.source_library_dir or '(none)'}")
        print(f"[Config] seed_R1_coding = {args.seed_coding_tag or '(none)'}")

        if args.mode == "baseline":
            self._run_baseline(task_ids, args)
        elif args.mode == "librarian":
            # Librarian samples the *unified* WebgenLibraryAgent with none of
            # our SLA-specific candidate machinery (design doc §2). Force the
            # overrides here so the shell launcher stays thin.
            args.candidate_strategy = "none"
            args.library_candidate_strategy = "none"
            print("[Config] librarian → candidate_strategy=none, "
                  "library_candidate_strategy=none")
            self._run_librarian(task_ids, args)
        elif args.mode == "sla_naive":
            self._run_sla_naive(task_ids, args)
        elif args.mode == "sla_naive_split":
            # Ablation: sla_ours orchestrator with local_extract / extract_map /
            # candidates forced off.
            args.extract_map = False
            args.candidate_strategy = "none"
            os.environ["WEBGEN_LOCAL_EXTRACT"] = "0"
            print("[Config] sla_naive_split → extract_map=False, "
                  "candidate_strategy=none, WEBGEN_LOCAL_EXTRACT=0")
            self._run_sla_ours(task_ids, args)
        else:  # sla_ours
            self._run_sla_ours(task_ids, args)

    # ---- baseline orchestrator -------------------------------------------

    def _run_baseline(self, task_ids: list[str], args: argparse.Namespace) -> None:
        """Single round, every task in parallel through WebgenCodingAgent."""
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
        """Per round: coding (parallel M) → unified WebgenLibraryAgent."""
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
                                dirs_exist_ok=True, ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)

            staged_apps: dict[str, str] = {}
            for tid, src in cumulative_subs.items():
                sub_dst = os.path.join(apply_tasks, tid, "submission")
                if not os.path.isdir(sub_dst) or not os.listdir(sub_dst):
                    if not os.path.isdir(src):
                        print(f"[WebgenFullRun] sla_naive: missing app for "
                              f"{tid}: {src}")
                        continue
                    os.makedirs(os.path.dirname(sub_dst), exist_ok=True)
                    shutil.copytree(src, sub_dst, dirs_exist_ok=True,
                                    ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)
                staged_apps[tid] = sub_dst

            agent = WebgenLibraryAgent(
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
                print(f"[WebgenFullRun] sla_naive round={round_num} library "
                      f"agent failed")
            else:
                # WebgenLibraryAgent writes the lib to the phase-level
                # apply/lib/; mirror it per-task so eval_webgen.sh PHASE=apply
                # finds apply/tasks/<id>/lib/ (sla_ours's layout).
                self._mirror_lib_to_tasks(apply_lib, apply_tasks, staged_apps)
                # Mirror to extract/lib/ too so PHASE=extract eval finds it.
                extract_lib = os.path.join(
                    args.runs_root, f"round_{round_num}", "extract", "lib",
                )
                os.makedirs(extract_lib, exist_ok=True)
                if not os.listdir(extract_lib):
                    shutil.copytree(apply_lib, extract_lib,
                                    dirs_exist_ok=True, ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)

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

        ZS-seeded corpus → gate ORIGINALS once (gate_before) → K LibraryAgent
        samples (temperature>0), each gated + 1 repair turn → MDL rerank →
        promote winner into the standard apply/extract layout → backup.
        """
        if not args.seed_coding_tag:
            sys.exit(
                "--mode librarian requires --seed-coding-tag: the corpus must "
                "be seeded from a Zero-Shot baseline backup "
                "(backups/webgen/<tag>/final/round_1/coding/)."
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
        coding_tasks_dir = self.phase_tasks_dir(args, round_num, "coding")
        print(f"[librarian] gating ZS originals under {coding_tasks_dir}")
        before_results = run_webgen_build_gate(
            coding_tasks_dir, None,
            parallel=args.max_workers, timeout_s=300,
        )
        gate_before = {tid for tid, r in before_results.items() if r.ok}
        print(f"[librarian] gate_before = {len(gate_before)}/"
              f"{len(before_results)} build: {sorted(gate_before)}")
        self._write_gate_json(
            os.path.join(round_dir, "gate_before.json"), before_results,
        )

        # --- 2) sample K candidates in parallel, gate + repair each -------
        # Samples are fully independent (separate workspaces / lib dirs / log
        # phases), so run them through a ThreadPoolExecutor mirroring the
        # coding phase's proven fan-out + per-future error isolation. Docker
        # concurrency is bounded two ways: SAMPLE_WORKERS samples run at once,
        # each gating GATE_PARALLEL apps at a time, so peak concurrent builds
        # ≈ SAMPLE_WORKERS × GATE_PARALLEL ≈ args.max_workers (the same budget
        # the coding phase already sustains).
        samples: dict[int, SamplePaths] = {}
        gate_results: dict[int, dict[str, bool]] = {}
        samples_root = os.path.join(round_dir, "samples")

        budget = max(1, int(args.max_workers))
        sample_workers = min(K, max(1, budget // 2))
        gate_parallel = max(1, budget // sample_workers)
        print(f"[librarian] sampling {K} candidates: {sample_workers} samples "
              f"in parallel × gate parallel={gate_parallel} "
              f"(peak docker builds ≈ {sample_workers * gate_parallel})")

        with ThreadPoolExecutor(max_workers=sample_workers) as ex:
            futures = {
                ex.submit(
                    self._run_one_sample, k, K, samples_root, new_subs,
                    gate_before, round_num, gate_parallel, args,
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
            task="webgen", base_url=base_url,
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
        gate_parallel: int,
        args: argparse.Namespace,
    ) -> tuple[SamplePaths, dict[str, bool]]:
        """Sample one candidate: LibraryAgent → build gate → 1 repair turn.

        Runs inside a ThreadPoolExecutor worker (see ``_run_librarian``). Fully
        self-contained: its own workspace/lib/log-phase, its own gate work dir,
        and it strips node_modules from the sample before returning so the
        round backup stays small. Returns ``(SamplePaths, {tid: passed})``.
        Never raises — an internal failure degrades to a gate-failed candidate.
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
            staged_apps = self._stage_apps_into(new_subs, sample_tasks)
            agent = WebgenLibraryAgent(
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

            results = run_webgen_build_gate(
                sample_tasks, sample_lib,
                parallel=gate_parallel, timeout_s=300,
            )
            failed = {
                tid: results[tid].error_tail
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
                results = run_webgen_build_gate(
                    sample_tasks, sample_lib,
                    parallel=gate_parallel, timeout_s=300,
                )

            n_ok = sum(1 for r in results.values() if r.ok)
            n_before_ok = sum(
                1 for tid in gate_before if results.get(tid) and results[tid].ok
            )
            print(f"[librarian] sample {k}: gate {n_ok}/{len(results)} build; "
                  f"gate_before-preserved {n_before_ok}/{len(gate_before)}; "
                  f"repaired={repaired}")
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
        finally:
            # The LibraryAgent's in-container npm install left root-owned
            # node_modules/ (~47 MB/app) in the sample workspace; strip it
            # before backup regardless of outcome.
            self._clean_sample_workspace(sample_dir, args.docker_image)

    def _clean_sample_workspace(
        self, sample_dir: str, image: str | None,
    ) -> None:
        """Strip node_modules/ from a finished sample before it is backed up.

        The LibraryAgent verifies its refactor with an in-container
        ``npm install && npx vite build``, which writes a root-owned
        ``node_modules/`` (~47 MB/app) into every app under the sample
        workspace. ``backup_round`` copies samples/ wholesale and does NOT
        filter node_modules, so left in place a K=8 round backup balloons to
        ~750 MB (and the runs/ copy fills NFS). dist/ and shots/ are kept
        (small, useful build evidence). docker wrote these dirs as root, so
        reclaim ownership via a throwaway container (mirrors
        webgen_build._chown_to_host) before rmtree, or the host-side delete
        hits EPERM on NFS.
        """
        if not os.path.isdir(sample_dir):
            return
        node_mods: list[str] = []
        for root, dirs, _files in os.walk(sample_dir):
            if "node_modules" in dirs:
                node_mods.append(os.path.join(root, "node_modules"))
                dirs.remove("node_modules")  # don't descend into it
        if not node_mods:
            return
        # Reclaim ownership of the whole sample (node_modules AND any
        # root-owned lib/src the agent wrote) so the delete — and later NFS
        # hygiene — don't hit EPERM. Only needed when the agent ran in docker.
        if image:
            _chown_to_host(sample_dir, image)
        removed = 0
        for nm in node_mods:
            shutil.rmtree(nm, ignore_errors=True)
            if not os.path.isdir(nm):
                removed += 1
        print(f"[librarian] cleaned {removed}/{len(node_mods)} node_modules/ "
              f"from {os.path.basename(sample_dir)}")

    def _stage_apps_into(
        self, src_subs: dict[str, str], dest_tasks: str,
    ) -> dict[str, str]:
        """Copy each app's submission into ``dest_tasks/<tid>/submission``.

        Mirrors _run_sla_naive's staging (``_STAGE_IGNORE``). Returns the staged
        submission paths (the agent edits these in place).
        """
        staged: dict[str, str] = {}
        for tid, src in src_subs.items():
            sub_dst = os.path.join(dest_tasks, tid, "submission")
            if not os.path.isdir(sub_dst) or not os.listdir(sub_dst):
                if not os.path.isdir(src):
                    print(f"[librarian] staging: missing source app for "
                          f"{tid}: {src}")
                    continue
                os.makedirs(os.path.dirname(sub_dst), exist_ok=True)
                shutil.copytree(src, sub_dst, dirs_exist_ok=True,
                                ignore=_STAGE_IGNORE,
                                ignore_dangling_symlinks=True)
            staged[tid] = sub_dst
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
        replays ``messages`` when ``enable_resume=True`` (otherwise it rebuilds
        a fresh user prompt and drops the prior conversation). Mirrors
        ``_run_extract_map_turn``.
        """
        try:
            feedback = build_librarian_repair_prompt(failed)
            agent = WebgenLibraryAgent(
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
        layout) so eval/metrics tooling reads it unchanged."""
        apply_root = os.path.join(args.runs_root, f"round_{round_num}", "apply")
        apply_tasks = os.path.join(apply_root, "tasks")
        apply_lib = os.path.join(apply_root, "lib")
        os.makedirs(apply_tasks, exist_ok=True)
        os.makedirs(apply_lib, exist_ok=True)

        promoted: list[str] = []
        for tid in task_ids:
            src = os.path.join(winner.tasks_dir, tid, "submission")
            dst = os.path.join(apply_tasks, tid, "submission")
            if not os.path.isdir(src):
                print(f"[librarian] promote: missing winner app for {tid}")
                continue
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            ignore=_STAGE_IGNORE, ignore_dangling_symlinks=True)
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

            # run_extract_phase reads this to identify new vs carry-forward apps.
            args._new_ids_this_round = round_tids

            if round_num == 1 and args.seed_coding_tag:
                new_subs = self._seed_round1_coding(round_tids, args)
            else:
                new_subs = self.run_coding_phase(
                    round_num, round_tids, current_lib_dir, args,
                )
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
                print(f"[Round {round_num}] cumulative<2 → skip extract+apply")

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

        sla_naive's WebgenLibraryAgent writes the library to the phase-level
        ``apply/lib/`` (single agent over all apps); sla_ours's apply step
        produces a per-task ``apply/tasks/<tid>/lib/`` via
        ``BaseCodingAgent._stage_library``. This helper bridges the gap so
        both modes leave eval_webgen.sh-compatible per-task ``lib/`` dirs
        regardless of how the library was produced.

        Idempotent: skips tasks whose ``<task>/lib/`` is already non-empty.
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
        """Effective NL *pick* model (candidate selection).

        Empty / ``"auto"`` → the coding backbone (``--model``) verbatim, so the
        picker follows the backbone; explicit ``--nl-pick-model`` overrides. The
        NL *summary* model is hardcoded at the agent constructor (gpt-5.4-nano).

        Routing (``utils/candidates/nl.py:_llm_pick``): OpenAI-family models use
        the OpenAI Responses API; every other model (deepseek, minimax, qwen, …)
        goes via OpenRouter. Pass ``--nl-pick-model`` with its vendor prefix
        (e.g. ``minimax/minimax-m3``) to override.
        """
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
    WebgenFullRun().main()
