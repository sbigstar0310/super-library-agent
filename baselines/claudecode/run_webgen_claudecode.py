"""Run the Claude Code CLI as a WebGen-Bench generation baseline, driven by
deepseek-v4-flash (NOT a Claude model) via a local LiteLLM Anthropic proxy.

This is the "recent agent-based SE system" comparison point for reviewer (4):
Claude Code (more famous/recent than Bolt.diy) generating each web app
independently (no shared library), controlled to the SAME backbone as every
other baseline — deepseek-v4-flash with temperature=0, reasoning_effort=high,
provider pinned to native deepseek (all pinned in litellm_config.yaml, matching
el-agent _factory.py). Only the agent SCAFFOLD differs (mini-swe / OpenHands /
Claude Code).

Reference: experiments/cc_exp/runner/docker.py (the original Claude-model runner).
Here we (a) keep only the zero-shot/baseline path, (b) swap Anthropic auth for
the deepseek proxy, (c) use the same SLA coding prompt as the other baselines.

Sandbox: the prebuilt `cc-sandbox` image (node:20-slim + claude-code CLI) — same
node major as sla-base, so runtime is not a confounder.

Usage (smoke, 1 task):
  python run_webgen_claudecode.py --limit 1 --tag cc-deepseek-smoke
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_TASKS = REPO_ROOT / "data" / "WebGen-Bench" / "data" / "test.jsonl"
# Fairness: mirror the SLA Zero-Shot CodingAgent's per-task layout spec injection
# (data/augments/webgen/layout_specs/<id>.md) so only the scaffold differs.
DEFAULT_LAYOUT_SPECS_DIR = REPO_ROOT / "data" / "augments" / "webgen" / "layout_specs"
SANDBOX_IMAGE = "cc-sandbox"
CONTAINER_WORKDIR = "/workspace"

# Same SLA coding prompt as ZS / OpenHands (fairness): only the scaffold differs.
sys.path.insert(0, str(REPO_ROOT / "el-agent" / "src"))
from prompts.webgen.coding_agent import (  # noqa: E402
    WEBGEN_SYSTEM_PROMPT,
    build_prompt_from_task,
)

# Claude-Code artifacts + build/vcs dirs that must NOT pollute the measured
# submission (claude writes directly into the mounted workspace, incl. the
# node_modules/dist it creates when it runs npm install/build).
_CC_ARTIFACTS = [".claude", ".config", ".cache", ".npm",
                 "node_modules", "dist", ".git"]


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

    Mirrors WebgenCodingAgent.build_user_prompt: an absent spec file leaves the
    task vanilla (baseline); a present one appends the layout reference block
    via format_task_body.
    """
    if not specs_dir:
        return tasks
    specs_dir = Path(specs_dir)
    for task in tasks:
        spec_path = specs_dir / f"{str(task['id']).zfill(6)}.md"
        if spec_path.is_file():
            task["layout_spec"] = spec_path.read_text()
    return tasks


def build_prompts(task):
    system = WEBGEN_SYSTEM_PROMPT
    user = build_prompt_from_task(task, workspace_dir=CONTAINER_WORKDIR, library_block="")
    return system, user


def docker_cmd(submission_dir: Path, system_text: str, user_text: str, args):
    """cc-sandbox `claude` invocation, deepseek proxy backbone (ref: cc_exp/docker.py)."""
    return [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--network=host",                      # reach the host LiteLLM proxy on 127.0.0.1
        "-e", f"ANTHROPIC_BASE_URL=http://127.0.0.1:{args.proxy_port}",
        "-e", f"ANTHROPIC_API_KEY={args.anthropic_token}",
        "-e", "IS_SANDBOX=1",
        "-v", f"{submission_dir}:{CONTAINER_WORKDIR}",
        "--workdir", CONTAINER_WORKDIR,
        SANDBOX_IMAGE,
        "claude",
        "--model", args.model,
        "--dangerously-skip-permissions",
        "--disallowed-tools", "WebSearch,WebFetch",
        "--output-format", "stream-json", "--verbose",
        "--append-system-prompt", system_text,
        "-p", user_text,
    ]


def parse_stream_json(log_path: Path):
    """Extract (status, fresh_in, cache_in, out_tok) from the stream-json result event.

    deepseek prompt-caches the (huge, repeated) Claude Code context, so most input
    tokens come back as cache_read_input_tokens — billed at the cache rate, NOT the
    full input rate. Keeping them separate is essential for correct cost.
    """
    status, fresh_in, cache_in, out_tok = "unknown", 0, 0, 0
    try:
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            ev = json.loads(line)
            if ev.get("type") == "result":
                status = "error" if ev.get("is_error") else "ok"
                u = ev.get("usage", {}) or {}
                fresh_in = u.get("input_tokens", 0)
                cache_in = (u.get("cache_read_input_tokens", 0)
                            + u.get("cache_creation_input_tokens", 0))
                out_tok = u.get("output_tokens", 0)
    except Exception:
        pass
    return status, fresh_in, cache_in, out_tok


# deepseek-v4-flash pricing ($/1M) from el-agent _factory.py: input 0.14 / cache 0.0028 / output 0.28
def cost_of(fresh_in, cache_in, out_tok):
    return fresh_in / 1e6 * 0.14 + cache_in / 1e6 * 0.0028 + out_tok / 1e6 * 0.28


def run_task(task, args):
    tid = str(task["id"])
    submission = Path(args.out_root) / "tasks" / tid / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    system, user = build_prompts(task)
    log_path = Path(args.out_root) / "tasks" / tid / "claude.stream.jsonl"

    cmd = docker_cmd(submission, system, user, args)
    t0 = time.time()
    with open(log_path, "w") as lf:
        try:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    status, fresh_in, cache_in, out_tok = parse_stream_json(log_path)
    # scrub Claude-Code artifacts from the measured submission
    for art in _CC_ARTIFACTS:
        p = submission / art
        if p.exists():
            subprocess.run(["rm", "-rf", str(p)])
    return {
        "id": tid, "rc": rc, "status": status,
        "in_tok": fresh_in, "cache_in": cache_in, "out_tok": out_tok,
        "cost": round(cost_of(fresh_in, cache_in, out_tok), 6),
        "seconds": round(time.time() - t0, 1),
        "submission": str(submission),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(DEFAULT_TASKS))
    ap.add_argument("--tag", default="cc-deepseek-baseline")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--model", default="deepseek-cc")   # litellm proxy model alias
    ap.add_argument("--proxy-port", default=None)       # default: read proxy_port.txt
    ap.add_argument("--anthropic-token", default="sk-litellm-dummy")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--task-id", action="append", default=[])
    ap.add_argument("--layout-specs-dir", default=str(DEFAULT_LAYOUT_SPECS_DIR),
                    help="per-task layout spec dir (<id>.md); '' to disable (parity w/ Zero-Shot)")
    args = ap.parse_args()

    if args.proxy_port is None:
        pp = HERE / "proxy_port.txt"
        args.proxy_port = "".join(c for c in pp.read_text() if c.isdigit()) if pp.exists() else "8801"
    if args.out_root is None:
        args.out_root = str(
            REPO_ROOT / "backups" / "webgen" / args.tag / "final" / "round_1" / "coding"
        )

    # sanity: sandbox image + proxy reachable
    if subprocess.run(["docker", "image", "inspect", SANDBOX_IMAGE],
                      capture_output=True).returncode != 0:
        sys.exit(f"missing image '{SANDBOX_IMAGE}': docker build -t {SANDBOX_IMAGE} docker/cc-sandbox/")

    tasks = load_tasks(args.tasks, args.limit, args.task_id)
    tasks = inject_layout_specs(tasks, args.layout_specs_dir)
    log_path = Path(args.out_root).parent / "cc_runlog.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if log_path.exists():
        done = {json.loads(l)["id"] for l in log_path.read_text().splitlines() if l.strip()}

    n_spec = sum(1 for t in tasks if t.get("layout_spec"))
    print(f"[cc-deepseek] {len(tasks)} task(s) ({n_spec} with layout_spec) "
          f"-> {args.out_root} (proxy :{args.proxy_port})")
    for task in tasks:
        tid = str(task["id"])
        if tid in done:
            print(f"  skip {tid} (logged)"); continue
        print(f"  === task {tid} ===")
        try:
            rec = run_task(task, args)
        except Exception as e:
            rec = {"id": tid, "status": "exception", "error": repr(e)}
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  {tid}: {rec.get('status')} cost=${rec.get('cost')} {rec.get('seconds')}s")


if __name__ == "__main__":
    main()
