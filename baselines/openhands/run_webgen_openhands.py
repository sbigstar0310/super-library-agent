"""Run the latest OpenHands agent (SDK v1.34) as a WebGen-Bench generation baseline.

This is the "recent agent-based SE system" comparison point requested by reviewer (4):
OpenHands generates each web app *independently* (no shared library), controlled to the
same backbone LLM (deepseek-v4-flash) as SLA. Outputs are normalized into the SLA backup
layout so the existing 6 metrics + WebGen eval read them unchanged.

Recipe (confirmed against software-agent-sdk examples/02_remote_agent_server):
  DockerDevWorkspace(base_image=<python+node>) -> agent-server in a sandbox container
  get_default_agent(llm, cli_mode=True) -> standard Terminal/FileEditor/TaskTracker agent
  Conversation(agent, workspace).send_message(prompt).run()
  tar /workspace (minus node_modules/.git) -> file_download -> submission/

Usage (smoke, 1 task):
  python run_webgen_openhands.py --limit 1 --tag openhands-baseline-smoke
"""

import argparse
import json
import os
import platform
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from pydantic import SecretStr

from openhands.sdk import LLM, Conversation, get_logger
from openhands.tools.preset.default import get_default_agent
from openhands.workspace.docker.dev_workspace import DockerDevWorkspace

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = REPO_ROOT / "data" / "WebGen-Bench" / "data" / "test.jsonl"
# Fairness: the SLA Zero-Shot CodingAgent injects a per-task layout spec
# (data/augments/webgen/layout_specs/<id>.md) into the task body; mirror it here so
# the only variable vs Zero-Shot is the agent scaffold, not the prompt content.
DEFAULT_LAYOUT_SPECS_DIR = REPO_ROOT / "data" / "augments" / "webgen" / "layout_specs"
# Fairness: OpenHands runs in the SAME sandbox image the Zero-Shot CodingAgent uses
# (sla-base: python3.12 + node20 + npm), so runtime is not a confounder. The
# agent-server layer is built on top of it by DockerDevWorkspace.
DEFAULT_BASE_IMAGE = "sla-base:latest"
# DockerDevWorkspace builds the agent-server image from the SDK source (uv workspace).
# The build resolves the workspace root by climbing up from CWD, so we chdir here first.
DEFAULT_SDK_ROOT = REPO_ROOT / "data" / "software-agent-sdk"

# Fairness: use the EXACT prompt the SLA Zero-Shot WebgenCodingAgent receives, imported
# from the el-agent source (not a hand-written OpenHands prompt), so the only variable
# vs Zero-Shot is the agent scaffold. The container working dir is /workspace.
sys.path.insert(0, str(REPO_ROOT / "el-agent" / "src"))
from prompts.webgen.coding_agent import (  # noqa: E402
    WEBGEN_SYSTEM_PROMPT,
    build_prompt_from_task,
)

CONTAINER_WORKDIR = "/workspace"


def build_message(task):
    """Same content Zero-Shot gets: system framing + task/app-rules/workspace block.

    OpenHands' own scaffold prompt occupies the system role, so we fold the SLA
    system prompt into the message to keep the task specification identical.
    Baseline (no shared library) => empty library_block.
    """
    user = build_prompt_from_task(
        task, workspace_dir=CONTAINER_WORKDIR, library_block=""
    )
    return f"{WEBGEN_SYSTEM_PROMPT}\n\n{user}"


def load_env(keys):
    """Read KEY=VALUE from el-agent/.env then repo .env; do not override real env."""
    out = {}
    for env_path in (REPO_ROOT / "el-agent" / ".env", REPO_ROOT / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in keys and k not in out:
                out[k] = v
    for k in keys:
        if os.getenv(k):
            out[k] = os.getenv(k)
    return out


def detect_platform():
    m = platform.machine().lower()
    return "linux/arm64" if ("arm" in m or "aarch64" in m) else "linux/amd64"


def load_tasks(path, limit, task_ids):
    tasks = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if task_ids:
        want = set(task_ids)
        tasks = [t for t in tasks if str(t["id"]) in want]
    if limit:
        tasks = tasks[:limit]
    return tasks


def inject_layout_specs(tasks, specs_dir):
    """Set task["layout_spec"] from <specs_dir>/<id>.md when present (else unset).

    Mirrors WebgenCodingAgent.build_user_prompt exactly: an absent spec file
    leaves the task vanilla (baseline), a present one appends the
    [Visual & functional layout reference] block via format_task_body.
    """
    if not specs_dir:
        return tasks
    specs_dir = Path(specs_dir)
    for task in tasks:
        spec_path = specs_dir / f"{str(task['id']).zfill(6)}.md"
        if spec_path.is_file():
            task["layout_spec"] = spec_path.read_text()
    return tasks


# OpenHands agent-server persists runtime state into the working dir (/workspace);
# these are NOT part of the generated app and must be excluded so LOC/MDL/duplication
# metrics see only the submission, exactly like the Zero-Shot submission/.
_OH_ARTIFACTS = ["bash_events", "conversations", "project", ".openhands"]
# Build/vcs dirs excluded as everywhere else in the pipeline.
_BUILD_ARTIFACTS = ["node_modules", ".git", "dist"]


def retrieve_workspace(workspace, dest_submission: Path):
    """Tar /workspace inside the container (minus OpenHands/build artifacts) to host."""
    dest_submission.mkdir(parents=True, exist_ok=True)
    remote_tar = "/tmp/webgen_submission.tgz"
    excludes = " ".join(
        "--exclude=./%s" % d for d in (_OH_ARTIFACTS + _BUILD_ARTIFACTS)
    )
    res = workspace.execute_command(
        "cd /workspace && tar czf %s %s ." % (remote_tar, excludes)
    )
    if res.exit_code != 0:
        logger.error("tar failed: %s", res.stdout)
        return False
    with tempfile.TemporaryDirectory() as td:
        local_tar = Path(td) / "submission.tgz"
        workspace.file_download(remote_tar, str(local_tar))
        with tarfile.open(local_tar) as tf:
            tf.extractall(dest_submission, filter="data")
    return True


def run_task(task, args, creds):
    tid = str(task["id"])
    message = build_message(task)
    dest = Path(args.out_root) / "tasks" / tid / "submission"

    llm = LLM(
        usage_id="agent",
        model=args.model,
        base_url=creds.get("OPENROUTER_BASE_URL"),
        api_key=SecretStr(creds["OPENROUTER_API_KEY"]),
        # Fairness: match the SLA Zero-Shot/SLA deepseek config in el-agent
        # _factory.py EXACTLY — temperature=0.0, reasoning_effort="high"
        # (deepseek is a reasoning family), and pin the OpenRouter provider to
        # the native deepseek endpoint so routing/reasoning match the paper runs.
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        litellm_extra_body={"provider": {"only": ["deepseek"]}},
    )
    t0 = time.time()
    cost = None
    status = "ok"
    with DockerDevWorkspace(
        base_image=args.base_image,
        # source-minimal skips the desktop/VNC/VSCode/Docker layers of the default
        # 'source' target — none are needed to write code + run npm, so builds are
        # far lighter. Runtime base stays sla-base, so fairness is unaffected.
        target=args.target,
        platform=detect_platform(),
    ) as workspace:
        agent = get_default_agent(llm=llm, cli_mode=True)
        # Fairness: cap the agent action loop to match Zero-Shot's STEP_LIMIT.
        # stuck_detection stays on as an extra runaway guard.
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            max_iteration_per_run=args.max_iterations,
        )
        try:
            conversation.send_message(message)
            conversation.run()
            status = str(conversation.state.execution_status)
            # Record precise token usage — litellm CANNOT price
            # "openrouter/deepseek/deepseek-v4-flash" (not in its map), so the SDK
            # accumulated_cost is unreliable; we compute cost ourselves from tokens
            # with the deepseek pricing from _factory.py (0.14/0.0028/0.28 per 1M).
            sdk_cost = prompt_tok = completion_tok = cache_tok = None
            try:
                m = conversation.conversation_stats.get_combined_metrics()
                sdk_cost = m.accumulated_cost
                tu = m.accumulated_token_usage
                prompt_tok = tu.prompt_tokens
                completion_tok = tu.completion_tokens
                cache_tok = tu.cache_read_tokens
            except Exception:
                pass
            ok = retrieve_workspace(workspace, dest)
            if not ok:
                status = "retrieve_failed"
        finally:
            conversation.close()
    # cache_read is a subset of prompt_tokens (OpenAI-style) when cache<=prompt;
    # some providers report it separately when cache>prompt.
    cost = None
    if prompt_tok is not None:
        cache = cache_tok or 0
        fresh = prompt_tok - cache if cache <= prompt_tok else prompt_tok
        cost = round(fresh / 1e6 * 0.14 + cache / 1e6 * 0.0028
                     + (completion_tok or 0) / 1e6 * 0.28, 6)
    return {
        "id": tid,
        "status": status,
        "cost": cost,                 # token-based (deepseek pricing), reliable
        "sdk_cost": sdk_cost,         # openhands-sdk value (unreliable for this model)
        "prompt_tokens": prompt_tok,
        "cache_read_tokens": cache_tok,
        "completion_tokens": completion_tok,
        "seconds": round(time.time() - t0, 1),
        "submission": str(dest),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
    ap.add_argument("--tag", default="openhands-baseline")
    ap.add_argument("--out-root", default=None,
                    help="default: backups/webgen/<tag>/final/round_1/coding")
    ap.add_argument("--model", default="openrouter/deepseek/deepseek-v4-flash")
    ap.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    ap.add_argument("--sdk-root", default=str(DEFAULT_SDK_ROOT),
                    help="software-agent-sdk checkout (uv workspace root) for image build")
    ap.add_argument("--target", default="source-minimal",
                    help="agent-server build target (source-minimal skips desktop/VNC)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--task-id", action="append", default=[])
    ap.add_argument("--layout-specs-dir", default=str(DEFAULT_LAYOUT_SPECS_DIR),
                    help="per-task layout spec dir (<id>.md); '' to disable (parity w/ Zero-Shot)")
    # Fairness knobs mirroring the SLA Zero-Shot mini-swe run:
    ap.add_argument("--max-iterations", type=int, default=150,  # = STEP_LIMIT
                    help="cap on the agent action loop per task (Zero-Shot STEP_LIMIT=150)")
    ap.add_argument("--temperature", type=float, default=0.0)   # = TEMPERATURE
    ap.add_argument("--reasoning-effort", default="high")       # = REASONING_EFFORT (deepseek high)
    args = ap.parse_args()

    if args.out_root is None:
        args.out_root = str(
            REPO_ROOT / "backups" / "webgen" / args.tag / "final" / "round_1" / "coding"
        )

    creds = load_env(["OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"])
    if not creds.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY not found in env or .env")

    os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

    # chdir into the SDK source repo so DockerDevWorkspace can resolve the uv
    # workspace root when it builds the agent-server image. All paths above are
    # already absolute, so this does not affect task/output resolution.
    sdk_root = Path(args.sdk_root)
    if not (sdk_root / "pyproject.toml").exists():
        sys.exit(f"--sdk-root does not look like the SDK repo: {sdk_root}")
    os.chdir(sdk_root)

    tasks = load_tasks(args.tasks, args.limit, args.task_id)
    tasks = inject_layout_specs(tasks, args.layout_specs_dir)
    n_spec = sum(1 for t in tasks if t.get("layout_spec"))
    logger.info("Running %d task(s) (%d with layout_spec) -> %s",
                len(tasks), n_spec, args.out_root)

    log_path = Path(args.out_root).parent / "openhands_runlog.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if log_path.exists():
        done = {json.loads(l)["id"] for l in log_path.read_text().splitlines() if l.strip()}

    for task in tasks:
        tid = str(task["id"])
        if tid in done:
            logger.info("skip %s (already logged)", tid)
            continue
        logger.info("=== task %s ===", tid)
        try:
            rec = run_task(task, args, creds)
        except Exception as e:
            logger.exception("task %s failed", tid)
            rec = {"id": tid, "status": "exception", "error": repr(e)}
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        logger.info("result: %s", rec)


if __name__ == "__main__":
    main()
