"""WebGen maintenance-patch run driver (rebuttal experiment).

Given a finished source backup tag (baseline / sla-naive / sla-naive-wc /
librarian / sla-ours), stages a patch workspace, runs WebgenEditAgent once
per Protocol-B suite session (or once per app for Protocol C), captures the
git diff, runs symmetric static checks, and saves the patched portfolio as a
new run under backups/webgen-maint/<run-name> (see compute_run_name:
non-baseline keep the source tag; baseline gets a -persuite-/-perapp- marker)
in the standard layout so all
existing metric/eval tooling works unchanged.

Design: paper appendix B (maintenance experiment).

Usage (via scripts/run/run_webgen_maintenance.sh):
    python -m run.webgen_maintenance_run \
        --source-tag sla-ours-c13-t1 --suite c13 --protocol b \
        [--dry-run-workspace]  # stage + build-smoke only, no LLM
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
# Canonical, version-controlled experiment inputs (policy / behavior / change
# tests / eval jsonl). Lives under data/augments/ per the SLA convention for
# tracked text augmentations, so the exact edit request behind reported
# numbers is reproducible from a fresh checkout.
EDIT_REQ_DIR = PROJECT_DIR / "data" / "augments" / "webgen" / "maintenance"
BACKUPS = PROJECT_DIR / "backups" / "webgen"           # source portfolios (read)
MAINT_ROOT = PROJECT_DIR / "backups" / "webgen-maint"  # patched results (write)
RUNS_ROOT = PROJECT_DIR / "runs" / "webgen-maint"

COPY_IGNORE = shutil.ignore_patterns("node_modules", "dist", ".git")

# Per-app limits of the original coding runs; Protocol B scales by app count.
BASE_STEP_LIMIT = 150
BASE_COST_LIMIT = 5.0


# ---------------------------------------------------------------------------
# source-backup resolution
# ---------------------------------------------------------------------------

def source_phase_round(source_tag: str) -> tuple[str, int]:
    """Map a source tag to its final (phase, round)."""
    if source_tag.startswith("baseline-"):
        return "coding", 1
    if source_tag.startswith("librarian-"):
        return "apply", 1
    return "apply", 4


def compute_run_name(source_tag: str, protocol: str) -> str:
    """Maintenance run name = <method w. postfix>-<suite>-<trial>.

    Postfix is added ONLY for the baseline method (no shared library), to
    mark whether the edit was applied per-suite (Protocol B, one session over
    all apps) or per-app (Protocol C, one session per app). Library-bearing
    methods run only per-suite and keep their source tag unchanged.
    """
    if source_tag.startswith("baseline-"):
        postfix = "persuite" if protocol == "b" else "perapp"
        return source_tag.replace("baseline-", f"baseline-{postfix}-", 1)
    return source_tag


def source_paths(source_tag: str) -> dict:
    phase, rnd = source_phase_round(source_tag)
    base = BACKUPS / source_tag / "final" / f"round_{rnd}"
    tasks_dir = base / phase / "tasks"
    lib_dir = base / "extract" / "lib"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"source tasks dir missing: {tasks_dir}")
    return {
        "phase": phase,
        "round": rnd,
        "tasks_dir": tasks_dir,
        "lib_dir": lib_dir if lib_dir.is_dir() else None,
    }


def load_metadata(suite: str) -> tuple[str, dict[str, str]]:
    policy = (EDIT_REQ_DIR / f"policy_{suite}.md").read_text()
    behavior = json.loads((EDIT_REQ_DIR / f"behavior_{suite}.json").read_text())
    behaviors = {
        app_id: entry["required_behavior"]
        for app_id, entry in behavior["targets"].items()
    }
    return policy, behaviors


# ---------------------------------------------------------------------------
# workspace staging
# ---------------------------------------------------------------------------

def _git(workdir: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(workdir), *args],
        capture_output=True, text=True, check=True,
    )
    return res.stdout


def _git_init_commit(root: Path) -> None:
    # Keep in-container npm artifacts + auto-generated lockfiles out of the
    # diff. Lockfile churn from `npm install` is method-incidental noise that
    # would inflate touched-files / LOC and vary by chance, not by method.
    (root / ".gitignore").write_text(
        "node_modules/\ndist/\n"
        "package-lock.json\nyarn.lock\npnpm-lock.yaml\n"
        # agent.env is written by BaseCodingAgent AFTER this snapshot; it is
        # infra config, not part of the patch surface, and would otherwise
        # appear as a new file in every session's diff (+1 for B, +n for C).
        "agent.env\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _git(root, "add", "-A")
    subprocess.run(
        ["git", "-c", "user.name=maint", "-c", "user.email=maint@local",
         "-C", str(root), "commit", "-qm", "pre-patch snapshot"],
        check=True,
    )


def _force_rmtree(dest: Path, docker_image: str = "sla-base") -> None:
    """rmtree that survives root-owned files left by in-container npm."""
    if not dest.exists():
        return
    try:
        shutil.rmtree(dest)
    except PermissionError:
        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{dest.parent}:/w", docker_image,
             "rm", "-rf", f"/w/{dest.name}"],
            check=True,
        )


def stage_suite_workspace(
    source_tag: str, targets: list[str], dest: Path
) -> Path:
    """Protocol B: dest/{lib?, tasks/<id>/submission, tasks/<id>/lib -> ../../lib}."""
    src = source_paths(source_tag)
    _force_rmtree(dest)
    dest.mkdir(parents=True)

    has_lib = src["lib_dir"] is not None
    if has_lib:
        shutil.copytree(src["lib_dir"], dest / "lib", ignore=COPY_IGNORE)

    for app_id in targets:
        app_src = src["tasks_dir"] / app_id / "submission"
        if not app_src.is_dir():
            raise FileNotFoundError(f"missing source app: {app_src}")
        app_dest = dest / "tasks" / app_id / "submission"
        shutil.copytree(app_src, app_dest, ignore=COPY_IGNORE)
        if has_lib:
            # Replace the per-task lib mirror with a relative symlink to the
            # single shared lib so one lib edit propagates to every app.
            link = dest / "tasks" / app_id / "lib"
            link.symlink_to(Path("..") / ".." / "lib")

    _git_init_commit(dest)
    return dest


def stage_single_workspace(source_tag: str, app_id: str, dest: Path) -> Path:
    """Protocol C: dest/submission only (baseline arm — no lib)."""
    src = source_paths(source_tag)
    app_src = src["tasks_dir"] / app_id / "submission"
    if not app_src.is_dir():
        raise FileNotFoundError(f"missing source app: {app_src}")
    _force_rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(app_src, dest / "submission", ignore=COPY_IGNORE)
    _git_init_commit(dest)
    return dest


def build_smoke(workspace: Path, app_rel: str, docker_image: str) -> bool:
    """`npm install && npx vite build` one app inside the run container to
    prove import resolution before patching. Returns True on success."""
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{workspace}:/home",
        "-w", f"/home/{app_rel}",
        docker_image,
        "bash", "-lc",
        # Cleanup must happen in-container: npm runs as root, so the host
        # user cannot rmtree the root-owned node_modules/dist afterwards.
        "npm install --no-audit --no-fund --loglevel=error >/dev/null 2>&1 "
        "&& npx vite build >/tmp/build.log 2>&1 "
        "&& echo BUILD_OK || tail -5 /tmp/build.log; "
        "rm -rf node_modules dist",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    ok = "BUILD_OK" in res.stdout
    print(f"[build-smoke] {app_rel}: {'OK' if ok else 'FAIL'}")
    if not ok:
        print(res.stdout[-2000:], res.stderr[-500:], sep="\n")
    return ok


# ---------------------------------------------------------------------------
# post-run capture
# ---------------------------------------------------------------------------

def dump_trajectory(log_dir: Path, task_key: str) -> None:
    """Write human-readable system.txt / user.txt / assistant.txt beside the
    step_0.json trajectory, so each run's prompt + per-turn actions are
    inspectable without parsing JSON."""
    step_file = (
        log_dir / "round_1" / "maintenance" / f"task_{task_key}" / "step_0.json"
    )
    if not step_file.is_file():
        return
    try:
        msgs = json.loads(step_file.read_text())["messages"]
    except Exception as e:
        print(f"[dump_trajectory] skip {task_key}: {e}")
        return
    out_dir = step_file.parent

    sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
    usr_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
    (out_dir / "system.txt").write_text(sys_msg or "")
    (out_dir / "user.txt").write_text(usr_msg or "")

    lines: list[str] = []
    turn = 0
    for m in msgs:
        role = m["role"]
        if role == "assistant":
            turn += 1
            lines.append(f"{'=' * 72}\n### TURN {turn}  (assistant)\n{'=' * 72}")
            reasoning = (m.get("reasoning_content") or "").strip()
            content = (m.get("content") or "").strip()
            if reasoning:
                lines.append(f"[reasoning]\n{reasoning}")
            if content:
                lines.append(f"[content]\n{content}")
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                try:
                    cmd = json.loads(fn.get("arguments", "{}")).get(
                        "command", fn.get("arguments", "")
                    )
                except Exception:
                    cmd = fn.get("arguments", "")
                lines.append(f"[tool call: {fn.get('name', '')}]\n{cmd}")
        elif role == "tool":
            c = m.get("content") or ""
            if not isinstance(c, str):
                c = str(c)
            if len(c) > 4000:
                c = c[:4000] + f"\n... [truncated, {len(c)} chars total]"
            lines.append(f"[tool result]\n{c}\n")
        elif role == "exit":
            lines.append(f"{'=' * 72}\n### EXIT\n{m.get('content') or ''}")
    (out_dir / "assistant.txt").write_text("\n\n".join(lines))
    print(f"[dump_trajectory] {task_key}: {turn} turns → {out_dir}")


def is_noise_path(path: str) -> bool:
    """Build artifacts / auto-generated lockfiles that must not count toward
    the patch surface. Method-incidental (whether an agent ran `npm install`),
    so they would add variance unrelated to method quality."""
    base = path.rsplit("/", 1)[-1]
    if base in ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "agent.env"):
        return True
    return any(
        seg in path for seg in ("/node_modules/", "/dist/")
    ) or path.startswith(("node_modules/", "dist/"))


def capture_diff(root: Path, out_dir: Path) -> dict:
    """git-diff the workspace vs the pre-patch snapshot; save patch + numstat."""
    _git(root, "add", "-A")
    numstat = _git(root, "diff", "--cached", "--numstat")
    patch = _git(root, "diff", "--cached")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "patch.diff").write_text(patch)
    (out_dir / "numstat.txt").write_text(numstat)

    files, added, deleted = [], 0, 0
    for line in numstat.strip().splitlines():
        a, d, path = line.split("\t", 2)
        if is_noise_path(path):
            continue
        files.append(path)
        added += 0 if a == "-" else int(a)
        deleted += 0 if d == "-" else int(d)
    return {
        "touched_files": files,
        "loc_added": added,
        "loc_deleted": deleted,
        "patch_file": str(out_dir / "patch.diff"),
    }


_IMPORT_RE = re.compile(
    r"(?:from|import)\s+['\"]([^'\"]+)['\"]"      # import 'x' / ... from 'x'
    r"|require\(\s*['\"]([^'\"]+)['\"]"           # require('x')
    r"|import\(\s*['\"]([^'\"]+)['\"]"            # dynamic import('x')
)
_CODE_EXTS = {".js", ".jsx", ".ts", ".tsx"}


def _added_lines_by_file(patch_text: str):
    """Yield (b-path, [added lines]) per file of a unified diff."""
    cur_file: str | None = None
    cur_lines: list[str] = []
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            if cur_file is not None:
                yield cur_file, cur_lines
            m = re.search(r" b/(.+)$", line)
            cur_file = m.group(1) if m else None
            cur_lines = []
        elif line.startswith("+") and not line.startswith("+++"):
            cur_lines.append(line[1:])
    if cur_file is not None:
        yield cur_file, cur_lines


def _import_violation(path: str, spec: str, root: Path, protocol: str) -> str | None:
    """Resolve an import specifier to a workspace-absolute path and judge
    which subtree it lands in. Bare specifiers (npm packages) are skipped.

    Protocol B: a file may import from its own tasks/<id>/ subtree or the
    shared lib/; anything landing in another app's subtree, or escaping the
    workspace, is a violation. Protocol C (single app, no lib): anything
    outside submission/ is a violation.
    """
    if spec.startswith("/home/"):  # container-absolute — /home mounts root
        target = os.path.normpath(str(root / spec[len("/home/"):]))
    elif spec.startswith("/"):
        target = os.path.normpath(spec)
    elif spec.startswith("."):
        target = os.path.normpath(str((root / path).parent / spec))
    else:
        return None  # npm package specifier

    if protocol == "b":
        m = re.match(r"tasks/(\d+)/", path)
        if m:
            allowed = [str(root / "tasks" / m.group(1)), str(root / "lib")]
        elif path.startswith("lib/"):
            allowed = [str(root / "lib")]
        else:
            return None  # root-level file — out of scope
    else:
        if not path.startswith("submission/"):
            return None
        allowed = [str(root / "submission")]

    if any(target == a or target.startswith(a + os.sep) for a in allowed):
        return None
    rootp = str(root)
    kind = ("cross-app import" if target.startswith(rootp + os.sep)
            else "import escapes workspace")
    return f"{kind} in {path}: {spec}"


def static_checks(root: Path, diff_info: dict, protocol: str) -> dict:
    """Symmetric architecture-preservation checks (all methods, all arms).

    Only NEW imports — added lines of the captured patch — are judged, so
    pre-existing quirks in the source portfolio are never attributed to the
    edit session. A violation is recorded for reporting (the run is excluded
    from aggregates); there is no automatic rerun or remediation.
    """
    violations: list[str] = []

    # 1. New top-level entries beyond the staged layout.
    allowed_top = {"lib", "tasks", "submission", "agent.env", ".git", ".gitignore"}
    for entry in os.listdir(root):
        if entry not in allowed_top:
            violations.append(f"new top-level entry: {entry}")

    # 2. New imports crossing app boundaries, judged by resolved path.
    patch_file = diff_info.get("patch_file")
    if patch_file and Path(patch_file).is_file():
        patch_text = Path(patch_file).read_text(errors="ignore")
        for path, added in _added_lines_by_file(patch_text):
            if not path or is_noise_path(path):
                continue
            if os.path.splitext(path)[1] not in _CODE_EXTS:
                continue
            for ln in added:
                for mm in _IMPORT_RE.finditer(ln):
                    spec = mm.group(1) or mm.group(2) or mm.group(3)
                    v = _import_violation(path, spec, root, protocol)
                    if v:
                        violations.append(v)

    return {"violations": violations, "clean": not violations}


def collect_cost(log_dir: Path) -> dict:
    """Sum calls/cost from step_*.json trajectories (info.model_stats)."""
    total = {"llm_calls": 0, "cost_usd": 0.0, "sessions": 0}
    for step_file in log_dir.rglob("step_*.json"):
        try:
            data = json.loads(step_file.read_text())
        except Exception:
            continue
        stats = (data.get("info", {}) or {}).get("model_stats", {}) or {}
        total["llm_calls"] += int(stats.get("api_calls", 0) or 0)
        total["cost_usd"] += float(stats.get("instance_cost", 0.0) or 0.0)
        total["sessions"] += 1
    return total


def save_backup(
    workspace: Path, maint_tag: str, targets: list[str], protocol: str
) -> Path:
    """Persist the patched portfolio in the standard backup layout.

    Per-app idempotent (Protocol C calls this once per app) — only the
    specific tasks/<id> subtrees passed in `targets` are replaced.
    """
    round_dir = MAINT_ROOT / maint_tag / "final" / "round_1"
    out = round_dir / "apply"
    lib = workspace / "lib"
    has_lib = lib.is_dir() and not lib.is_symlink()
    if has_lib:
        lib_out = round_dir / "extract" / "lib"
        if lib_out.exists():
            shutil.rmtree(lib_out)
        shutil.copytree(lib, lib_out, ignore=COPY_IGNORE)
    for app_id in targets:
        if protocol == "b":
            sub = workspace / "tasks" / app_id / "submission"
        else:
            sub = workspace / "submission"
        app_out = out / "tasks" / app_id
        if app_out.exists():
            shutil.rmtree(app_out)
        shutil.copytree(sub, app_out / "submission", ignore=COPY_IGNORE)
        if has_lib:
            # Per-task lib mirror required by get_loc/get_mdl/scb_quality.
            shutil.copytree(lib, app_out / "lib", ignore=COPY_IGNORE)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-tag", required=True)
    p.add_argument("--suite", required=True, choices=["c2", "c5", "c13"])
    p.add_argument("--protocol", required=True, choices=["b", "c"])
    p.add_argument("--provider", default="openrouter")
    p.add_argument("--model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--docker-image", default="sla-base")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--skip-build-smoke", action="store_true")
    p.add_argument("--dry-run-workspace", action="store_true",
                   help="stage workspace + build smoke only; no LLM run")
    args = p.parse_args()

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_DIR / ".env")

    if args.protocol == "c" and not args.source_tag.startswith("baseline-"):
        raise SystemExit("Protocol C is baseline-only by design (see plan doc).")

    policy, behaviors = load_metadata(args.suite)
    targets = sorted(behaviors)
    maint_tag = compute_run_name(args.source_tag, args.protocol)
    log_dir = MAINT_ROOT / maint_tag / "logs"
    n = len(targets)

    # Clear prior logs/diffs for this run name so cost/diff aggregation reflects
    # ONLY this invocation. collect_cost rglobs every step_*.json under log_dir;
    # stale files from an earlier (failed / re-parameterized) run of the same
    # name would otherwise be double-counted. The saved portfolio (final/) is
    # overwritten per-app by save_backup, so it is safe to leave.
    if not args.dry_run_workspace:
        for stale in (log_dir, MAINT_ROOT / maint_tag / "diff"):
            if stale.exists():
                shutil.rmtree(stale)

    from mswe_agents.webgen.edit_agent import WebgenEditAgent

    def make_agent(workspace: Path, protocol: str, beh: dict[str, str]) -> WebgenEditAgent:
        scale = n if protocol == "suite" else 1
        return WebgenEditAgent(
            workspace_root=str(workspace),
            protocol=protocol,  # type: ignore[arg-type]
            policy_text=policy,
            behaviors=beh,
            provider=args.provider,
            model=args.model,
            log_dir=str(log_dir),
            docker_image=args.docker_image,
            temperature=args.temperature,
            step_limit=BASE_STEP_LIMIT * scale,
            cost_limit=BASE_COST_LIMIT * scale,
            timeout=args.timeout,
        )

    results: dict = {"maint_tag": maint_tag, "protocol": args.protocol,
                     "source_tag": args.source_tag, "targets": targets}

    if args.protocol == "b":
        workspace = RUNS_ROOT / maint_tag / "suite"
        stage_suite_workspace(args.source_tag, targets, workspace)
        if not args.skip_build_smoke:
            rel = f"tasks/{targets[0]}/submission"
            if not build_smoke(workspace, rel, args.docker_image):
                raise SystemExit(f"pre-patch build smoke failed for {rel}")
            _git(workspace, "checkout", "--", ".")  # drop lockfile churn
            _git(workspace, "clean", "-qfd",
                 "--exclude=node_modules", "--exclude=dist")
        if args.dry_run_workspace:
            print(json.dumps({"staged": str(workspace), "targets": targets}))
            return
        agent = make_agent(workspace, "suite", behaviors)
        agent.run(args.suite, round_num=1, step_num=0, log_phase="maintenance")
        dump_trajectory(log_dir, args.suite)
        diff_info = capture_diff(workspace, MAINT_ROOT / maint_tag / "diff")
        checks = static_checks(workspace, diff_info, "b")
        save_backup(workspace, maint_tag, targets, "b")
        results.update(diff=diff_info, static_checks=checks,
                       cost=collect_cost(log_dir))
    else:
        per_app: dict[str, dict] = {}
        for app_id in targets:
            workspace = RUNS_ROOT / maint_tag / "apps" / app_id
            stage_single_workspace(args.source_tag, app_id, workspace)
            if args.dry_run_workspace:
                per_app[app_id] = {"staged": str(workspace)}
                continue
            agent = make_agent(workspace, "single", {app_id: behaviors[app_id]})
            agent.run(app_id, round_num=1, step_num=0, log_phase="maintenance")
            dump_trajectory(log_dir, app_id)
            diff_info = capture_diff(
                workspace, MAINT_ROOT / maint_tag / "diff" / app_id
            )
            checks = static_checks(workspace, diff_info, "c")
            save_backup(workspace, maint_tag, [app_id], "c")
            per_app[app_id] = {"diff": diff_info, "static_checks": checks}
        results["per_app"] = per_app
        if not args.dry_run_workspace:
            results["cost"] = collect_cost(log_dir)

    out = MAINT_ROOT / maint_tag / "run_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[maintenance] done → {out}")


if __name__ == "__main__":
    sys.exit(main())
