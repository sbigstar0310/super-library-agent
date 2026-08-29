"""Shared helper: launch a Claude Code agent inside cc-sandbox.

Smoke-pipeline LLM steps (generate_pages_recipe, describe_layout) must
run inside the same Docker isolation as cc_exp baseline so that:

  * CLAUDE.md / MEMORY.md never leak into the prompt
  * Claude Code version is pinned (DOCKER_IMAGE)
  * sibling tasks are invisible (only the requested mounts are visible)

This mirrors `experiments/cc_exp/runner/docker.py:_build_docker_cmd` but
is intentionally standalone so the layout-spec scripts don't depend on the
cc_exp Python package layout.

Output:
  Returns the docker exit code. The agent's stream-json log is written
  to `<prompt_file>.agent.jsonl` alongside the prompt file.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


DOCKER_IMAGE = "cc-sandbox"
DEFAULT_MODEL = "claude-opus-4-7"
CREDS_FILE = Path.home() / ".claude/.credentials.json"


def run_agent(
    prompt_file: Path,
    rw_dirs: list[Path],
    ro_dirs: list[Path],
    description: str,
    model: str = DEFAULT_MODEL,
    effort: str = "medium",
    timeout: int = 1800,
    system_text: str = "",
) -> int:
    """Fire a single Claude Code agent in cc-sandbox.

    Args:
        prompt_file:  File the prompt was written to. The agent receives
                      its contents via ``-p``.
        rw_dirs:      Workspace dirs mounted read-write; first becomes cwd.
        ro_dirs:      Dirs mounted read-only and exposed via ``--add-dir``.
        description:  Human-readable label for stdout.
        model:        Claude model name (default opus-4-7).
        effort:       Reasoning effort (low|medium|high).
        timeout:      Hard kill after this many seconds.
        system_text:  Optional text appended via ``--append-system-prompt``.

    Returns:
        Docker exit code (0 = success).
    """
    if not CREDS_FILE.exists():
        sys.exit(
            f"Claude credentials not found: {CREDS_FILE}\n"
            f"Run 'claude' once on the host to authenticate."
        )

    if not rw_dirs:
        sys.exit("run_agent requires at least one rw_dir (becomes cwd).")

    log_path = prompt_file.with_suffix(".agent.jsonl")

    user_text = prompt_file.read_text()

    cmd: list[str] = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{CREDS_FILE}:/sandbox/.claude/.credentials.json:ro",
        "-v", f"{prompt_file}:{prompt_file}:ro",
        "--network=host",
    ]
    for d in rw_dirs:
        d.mkdir(parents=True, exist_ok=True)
        cmd += ["-v", f"{d}:{d}"]
    for d in ro_dirs:
        if not d.exists():
            sys.exit(f"RO mount does not exist: {d}")
        cmd += ["-v", f"{d}:{d}:ro"]
    cmd += ["--workdir", str(rw_dirs[0])]

    cmd += [
        DOCKER_IMAGE,
        "claude",
        "--model", model,
        "--effort", effort,
        "--dangerously-skip-permissions",
        "--disallowed-tools", "WebSearch,WebFetch",
        "--output-format", "stream-json",
        "--verbose",
    ]
    for d in ro_dirs:
        cmd += ["--add-dir", str(d)]
    if system_text:
        cmd += ["--append-system-prompt", system_text]
    cmd += ["-p", user_text]

    print(f"  ▶ {description}  (model={model})")
    with open(log_path, "w") as lf:
        try:
            rc = subprocess.run(
                cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT ({timeout}s) {description}")
            return 1

    print(f"  {'✓' if rc == 0 else f'✗ exit={rc}'} {description}")
    return rc
