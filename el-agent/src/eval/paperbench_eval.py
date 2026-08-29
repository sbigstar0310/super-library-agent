"""Standalone code-only PaperBench grading driver.

Invokes ``paperbench.grade.run_judge`` directly on a pre-built submission
directory. Uses two LLM layers: the main grading model (configurable) and a
structured-output parser hardcoded to OpenAI ``gpt-5-mini`` (non-OpenAI
providers reject the ``json_schema`` response format SimpleJudge requires).
The main endpoint defaults to the Deepseek API; override with ``--base-url`` /
``--api-key-env`` / ``--model``.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from paperbench.grade import run_judge

# Paperbench's preparedness_turn_completer hardcodes a small list of OpenAI
# model → context window mappings and HARD-FAILS on anything else. Register
# the non-OpenAI models we actually use here before any judge gets built.
# Add an entry whenever a new --model is wired in.
from preparedness_turn_completer.utils import CONTEXT_WINDOW_LENGTHS

for _model, _ctx in {
    # Both the bare name (direct DeepSeek API) and the OpenRouter-prefixed
    # slug (lab policy default) must be registered — the judge looks up the
    # exact --model string passed in.
    "deepseek-v4-flash": 1_048_576,
    "deepseek-v4-pro": 1_048_576,
    "deepseek/deepseek-v4-flash": 1_048_576,
    "deepseek/deepseek-v4-pro": 1_048_576,
}.items():
    CONTEXT_WINDOW_LENGTHS.setdefault(_model, _ctx)


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # project root


def _resolve_paths_from_backup_tag(
    tag: str,
    paper_id: str,
    round_n: int,
    phase: str,
) -> tuple[Path, Path, Path | None]:
    """Map (backup-tag, paper-id, round, phase) → (submission, out, lib_dir).

    Unified layout (only mode supported):
      backups/paperbench/<tag>/final/round_<N>/<phase>/tasks/<paper_id>/submission/
      backups/paperbench/<tag>/final/round_<N>/<phase>/tasks/<paper_id>/lib/  (optional)
      backups/paperbench/<tag>/final/round_<N>/<phase>/lib/                   (extract phase, optional)
      backups/paperbench/<tag>/eval_results/round_<N>/<phase>/<paper_id>/

    Per-task lib takes precedence over phase-level lib when both exist.
    """
    bench_root = PROJECT_DIR / "backups" / "paperbench" / tag
    final_dir = bench_root / "final"
    if not final_dir.is_dir():
        raise SystemExit(f"Backup not finalized: {final_dir}")
    phase_dir = final_dir / f"round_{round_n}" / phase
    submission = phase_dir / "tasks" / paper_id / "submission"
    if not submission.is_dir():
        raise SystemExit(f"Submission not found: {submission}")
    per_task_lib = phase_dir / "tasks" / paper_id / "lib"
    phase_level_lib = phase_dir / "lib"
    lib_dir_opt: Path | None = None
    if per_task_lib.is_dir():
        lib_dir_opt = per_task_lib
    elif phase_level_lib.is_dir():
        lib_dir_opt = phase_level_lib
    out = bench_root / "eval_results" / f"round_{round_n}" / phase / paper_id
    return submission, out, lib_dir_opt


def main():
    parser = argparse.ArgumentParser(
        description="Standalone code-only PaperBench grading driver."
    )
    parser.add_argument("--paper-id", required=True)

    # Two modes:
    #   (1) --backup-tag <tag> --round N --phase X  — auto-derive paths
    #       from the unified paperbench layout
    #       (backups/paperbench/<tag>/final/round_<N>/<phase>/tasks/<id>/)
    #   (2) --submission <p> --out <p>              — explicit paths
    parser.add_argument(
        "--backup-tag",
        default=None,
        help="If set, submission and out are auto-derived from the unified "
        "paperbench layout. Requires --round and --phase. "
        "Mutually exclusive with --submission.",
    )
    parser.add_argument(
        "--round",
        default=None,
        type=int,
        help="Round number. Required with --backup-tag.",
    )
    parser.add_argument(
        "--phase",
        default=None,
        choices=[None, "baseline", "coding", "apply", "extract"],
        help="Phase name. Required with --backup-tag.",
    )
    parser.add_argument(
        "--submission",
        default=None,
        type=Path,
        help="Submission directory. Required when --backup-tag is not used.",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=Path,
        help="Output dir for graded_tree.json + leaf_logs/. "
        "Required when --submission is used.",
    )

    # Main grading completer — defaults to direct Deepseek API / v4-flash.
    # Use v4-pro explicitly (--model deepseek-v4-pro) when higher-fidelity
    # grading is needed; default is flash to keep per-paper cost ~$0.28
    # instead of ~$2.40 (≈8.5x).
    parser.add_argument(
        "--model",
        default="deepseek/deepseek-v4-flash",
        help="Main grading model (file ranking + leaf grading + subtree grading). "
        "Default: deepseek/deepseek-v4-flash via OpenRouter (cheap). Pass "
        "deepseek/deepseek-v4-pro for higher fidelity.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=[None, "low", "medium", "high"],
        help="Only meaningful for o-series reasoning models (o1, o3-mini, o4-mini, …)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Skip for o-series reasoning models",
    )

    # Parser completer is HARDCODED to OpenAI gpt-5-mini (see module
    # docstring for the json_schema rationale). No CLI knob.

    # OpenAI-compatible endpoint — defaults to OpenRouter (lab policy: the
    # direct DeepSeek API is no longer used). NOTE: the paperbench judge owns
    # the OpenAI client, so we cannot inject OpenRouter provider routing
    # (extra_body) here; pin the official deepseek provider via OpenRouter
    # account settings if quantized serving must be avoided for grading.
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible endpoint. Default: OpenRouter. "
        "Set to '' to use OPENAI_BASE_URL from env / OpenAI default.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Env var holding the API key (default OPENROUTER_API_KEY)",
    )

    parser.add_argument("--max-depth", type=int, default=999)
    parser.add_argument("--code-only", action="store_true", default=True)
    args = parser.parse_args()

    lib_dir: Path | None = None
    if args.backup_tag and args.submission:
        raise SystemExit("Pass either --backup-tag or --submission, not both.")
    if args.backup_tag:
        if args.round is None or args.phase is None:
            raise SystemExit("--backup-tag requires --round and --phase.")
        args.submission, args.out, lib_dir = _resolve_paths_from_backup_tag(
            args.backup_tag, args.paper_id,
            round_n=args.round, phase=args.phase,
        )
    else:
        if args.submission is None:
            raise SystemExit("Pass --backup-tag <tag> OR --submission <path>.")
        if args.out is None:
            raise SystemExit("--submission mode requires --out.")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Env var {args.api_key_env} is empty (try `export {args.api_key_env}=$OPENAILIKE_API_KEY`)"
        )
    os.environ["OPENAI_API_KEY"] = api_key  # paperbench reads this name for MAIN
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url

    # Parser credentials — independent OpenAI direct client (gpt-5-mini).
    parser_api_key = os.environ.get("OPENAILIKE_API_KEY")
    if not parser_api_key:
        raise SystemExit(
            "OPENAILIKE_API_KEY env var required for the gpt-5-mini parser "
            "(OpenAI direct). Set it in .env or export it."
        )
    import openai as _openai_mod
    # MUST pass base_url explicitly — without it, openai SDK falls back to the
    # `OPENAI_BASE_URL` env var which we just set to the MAIN provider's
    # endpoint (e.g. Deepseek). That would send our OpenAI key to Deepseek →
    # 401 "Your api key …70EA is invalid" from the wrong provider.
    parser_client = _openai_mod.AsyncClient(
        api_key=parser_api_key,
        base_url="https://api.openai.com/v1",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    log_dir = args.out / "leaf_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    from preparedness_turn_completer.oai_completions_turn_completer import (
        OpenAICompletionsTurnCompleter,
    )

    main_kwargs = {"model": args.model}
    if args.reasoning_effort is not None:
        main_kwargs["reasoning_effort"] = args.reasoning_effort
    if args.temperature is not None:
        main_kwargs["temperature"] = args.temperature
    main_cfg = OpenAICompletionsTurnCompleter.Config(**main_kwargs)

    # Force SimpleJudge's structured-output parser to OpenAI gpt-5-mini with a
    # dedicated client; upstream defaults to gpt-4o on the module-level client,
    # which points at the main (Deepseek) endpoint and fails on json_schema.
    from paperbench.judge.simple import SimpleJudge

    def _init_structured_completer_with_openai_parser(
        self, config, response_format,
    ):
        del config  # ignored — parser model + client are hardcoded
        cfg = OpenAICompletionsTurnCompleter.Config(
            model="gpt-5-mini",
            response_format=response_format,
        )
        completer = cfg.build()
        # `_client` is a cached_property; assigning shadows the descriptor.
        completer._client = parser_client
        return cfg, completer

    SimpleJudge._init_structured_completer = _init_structured_completer_with_openai_parser

    # Token usage logging: capture per-call tokens, written to usage.jsonl and
    # aggregated into usage_summary.json. Pricing comes from utils.llm
    # (imported via sys.path injection; not on paperbench's venv by default).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.llm import _get_pricing  # noqa: E402

    usage_log: list[dict] = []
    _orig_async_completion = OpenAICompletionsTurnCompleter.async_completion

    async def async_completion_with_usage_log(self, conversation, **params):
        completion = await _orig_async_completion(self, conversation, **params)
        u = completion.usage
        if u is not None:
            cached = 0
            details = getattr(u, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            usage_log.append({
                "model": self.model,
                "input_tokens": u.prompt_tokens,
                "cached_input_tokens": cached,
                "output_tokens": u.completion_tokens,
            })
        return completion

    OpenAICompletionsTurnCompleter.async_completion = async_completion_with_usage_log

    def _summarize_usage(log: list[dict]) -> dict:
        from collections import defaultdict
        per_model: dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0,
                     "output_tokens": 0, "input_usd": 0.0, "cached_usd": 0.0,
                     "output_usd": 0.0}
        )
        for e in log:
            m = e["model"]
            p = per_model[m]
            p["calls"] += 1
            p["input_tokens"] += e["input_tokens"]
            p["cached_input_tokens"] += e.get("cached_input_tokens", 0)
            p["output_tokens"] += e["output_tokens"]
            try:
                price = _get_pricing(m)
            except Exception:
                price = {"input": 0.0, "cached_input": 0.0, "output": 0.0}
            cached_tok = e.get("cached_input_tokens", 0)
            uncached_in = max(e["input_tokens"] - cached_tok, 0)
            p["input_usd"] += uncached_in * price["input"] / 1_000_000
            p["cached_usd"] += cached_tok * price.get("cached_input", 0.0) / 1_000_000
            p["output_usd"] += e["output_tokens"] * price["output"] / 1_000_000
        totals = {
            "calls": sum(p["calls"] for p in per_model.values()),
            "input_tokens": sum(p["input_tokens"] for p in per_model.values()),
            "cached_input_tokens": sum(p["cached_input_tokens"] for p in per_model.values()),
            "output_tokens": sum(p["output_tokens"] for p in per_model.values()),
            "usd": sum(p["input_usd"] + p["cached_usd"] + p["output_usd"]
                       for p in per_model.values()),
        }
        return {"per_model": dict(per_model), "totals": totals}

    # When --backup-tag mode finds a sibling lib/, temporarily copy it into
    # submission/lib/ so the judge grades the unified codebase. Cleaned up
    # after grading; the backup is untouched.
    submission_lib_path = args.submission / "lib"
    lib_copied = False
    if lib_dir is not None:
        if submission_lib_path.exists():
            raise SystemExit(
                f"Refusing to overwrite existing {submission_lib_path}. "
                "Remove it or use a clean backup."
            )
        shutil.copytree(lib_dir, submission_lib_path, symlinks=True)
        lib_copied = True

    print(f"  submission : {args.submission}")
    print(f"  paper_id   : {args.paper_id}")
    print(f"  main model : {args.model}  (reasoning={args.reasoning_effort})")
    print(f"  parser     : gpt-5-mini (OpenAI direct)")
    print(f"  base_url   : {args.base_url or '(default OpenAI)'}")
    print(f"  code_only  : {args.code_only}")
    print(f"  lib copied : {lib_dir if lib_copied else '(none)'}")
    print()

    import time
    t_start = time.time()
    try:
        graded_tree = asyncio.run(
            run_judge(
                submission_path=args.submission,
                paper_id=args.paper_id,
                judge_type="simple",
                code_only=args.code_only,
                completer_config=main_cfg,
                max_depth=args.max_depth,
                out_dir=log_dir,
            )
        )

        out_path = args.out / "graded_tree.json"
        with open(out_path, "w") as f:
            json.dump(graded_tree.to_dict(), f, indent=2)

        print(f"\nFinal score: {graded_tree.score:.4f}")
        print(f"Graded tree: {out_path}")
        print(f"Per-leaf logs: {log_dir}")
    finally:
        if lib_copied and submission_lib_path.exists():
            shutil.rmtree(submission_lib_path)

        # Always dump usage (even on partial failure) — useful for cost tracking.
        elapsed = time.time() - t_start
        usage_jsonl = args.out / "usage.jsonl"
        with open(usage_jsonl, "w") as f:
            for e in usage_log:
                f.write(json.dumps(e) + "\n")
        summary = _summarize_usage(usage_log)
        summary["elapsed_sec"] = round(elapsed, 1)
        summary_path = args.out / "usage_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Cost] elapsed: {elapsed:.1f}s  calls: {summary['totals']['calls']}")
        print(f"[Cost] tokens : in {summary['totals']['input_tokens']:,} / out {summary['totals']['output_tokens']:,}")
        print(f"[Cost] USD    : ~${summary['totals']['usd']:.4f}")
        for m, p in summary["per_model"].items():
            print(f"[Cost]   {m}: {p['calls']} calls, "
                  f"in {p['input_tokens']:,} / out {p['output_tokens']:,}, "
                  f"~${p['input_usd'] + p['output_usd']:.4f}")
        print(f"[Cost] details: {usage_jsonl}")


if __name__ == "__main__":
    main()
