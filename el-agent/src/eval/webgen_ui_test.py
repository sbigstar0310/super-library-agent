"""WebGen-Bench UI-test evaluation (filtered wrapper).

Reuses helpers from ``data/WebGen-Bench/src/ui_test_webgen`` and drives the
WebVoyager agent (``data/WebGen-Bench/webvoyager/run.py``). Adds:
- ``--app-id-list`` filter (vs original iterating 000001..000101 by index)
- configurable ``--test-file``, ``--num-workers``, ``--batch-size``
- WebVoyager OpenAI base URL via env ``WEBVOYAGER_API_BASE`` / ``--wv-api-base``
- After eval, computes accuracy table (``<in-dir>/table.md``) restricted to
  the filtered app set.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
# `WEBGEN_ROOT` override → `/home/webgen` inside the sla-base eval container
# (see webgen_appearance.py).
WEBGEN_ROOT = Path(os.environ.get("WEBGEN_ROOT") or (PROJECT_DIR / "data" / "WebGen-Bench"))
UI_DIR = WEBGEN_ROOT / "src" / "ui_test_webgen"
sys.path.insert(0, str(UI_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(PROJECT_DIR / ".env"))
except ImportError:
    pass

from start_service import start_services  # noqa: E402


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(rows, path: str, mode: str = "w"):
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


UI_PROMPT_TEMPLATE = """

Task: {task}

Expected Result: {expected_result}

Instructions:
- Attempt the task as a user would, using the UI elements available.
- Make multiple attempts if needed to try and achieve the expected result.
- Observe whether the expected result is fully, partially, or not at all achieved.
- IMPORTANT: You can at most interact with the website {max_iter} times. If the limit is reached, directly output your answer.
- If prompted for a username, password, or email in the process of testing, enter "admin," "admin123456", and "admin@example.com", respectively.

At the end of your testing, answer only with one of the following:
- YES: if the expected result was fully achieved.
- NO: if the expected result could not be achieved at all.
- PARTIAL: if only some aspects of the expected result were achieved.

"""


def make_commands(app_paths, output_root):
    commands = {os.path.basename(p): {"shell_actions": [], "last_start_action": ""} for p in app_paths}
    save_json(commands, os.path.join(output_root, "commands.json"))
    return commands


def build_tasks(test_datas, ports, out_path, max_iter: int):
    rows = []
    for d in test_datas:
        app = d["id"]
        if app not in ports:
            continue
        for ui_idx, ui in enumerate(d["ui_instruct"]):
            rows.append({
                "web_name": app,
                "id": f"{app}_{ui_idx}",
                "ques": UI_PROMPT_TEMPLATE.format(task=ui["task"], expected_result=ui["expected_result"], max_iter=max_iter),
                "web": f"http://localhost:{ports[app]}/",
                "expected_result": ui["expected_result"],
                "task": ui["task"],
            })
    save_jsonl(rows, out_path)


def run_webvoyager(in_dir: Path, num_workers: int, api_key: str, api_model: str,
                   api_base: str, reasoning_effort: str | None, max_iter: int):
    cmd = [
        sys.executable, "-u", "webvoyager/run.py",
        "--test_file", str(in_dir / "tasks_test_with_answer.jsonl"),
        "--api_key", api_key,
        "--api_base", api_base,
        "--api_model", api_model,
        "--headless",
        "--max_iter", str(max_iter),
        "--max_attached_imgs", "3",
        "--temperature", "1",
        "--fix_box_color",
        "--seed", "42",
        "--output_dir", str(in_dir / "results"),
        "--download_dir", str(in_dir / "downloads"),
        "--num_workers", str(num_workers),
    ]
    if reasoning_effort:
        cmd += ["--reasoning_effort", reasoning_effort]
    subprocess.run(cmd, check=True, cwd=str(WEBGEN_ROOT))


PRIMARY_CATEGORIES = ["Content Presentation", "User Interaction", "Data Management"]
INST_PRIMARY_CATEGORIES = ["Functional Testing", "Data Display Testing", "Design Validation Testing"]


def compute_accuracy(in_dir: str, test_datas):
    """Write ``<in_dir>/table.md`` aggregating WebVoyager YES/PARTIAL/NO answers.

    Counts only UI tasks whose app is in the filtered ``test_datas``. Result
    dirs are named ``task{app_id}_{sub_idx}`` (the ``task`` prefix comes from
    webvoyager/run.py).
    """
    result_dir = os.path.join(in_dir, "results")
    cats = {c: {"yes_num": 0, "partial_num": 0, "no_num": 0, "start_failed_num": 0,
                "score": 0.0, "total": 0, "accuracy": 0.0}
            for c in PRIMARY_CATEGORIES + INST_PRIMARY_CATEGORIES}

    id_to_idx = {d["id"]: i for i, d in enumerate(test_datas)}

    total = 0
    for d in test_datas:
        total += len(d["ui_instruct"])
        primary = d["Category"]["primary_category"]
        cats[primary]["total"] += len(d["ui_instruct"])
        for ui in d["ui_instruct"]:
            cats[ui["task_category"]["primary_category"]]["total"] += 1

    if not os.path.isdir(result_dir):
        print(f"[warn] no results dir: {result_dir}")
        return

    yes_num = partial_num = no_num = 0
    score = 0.0
    for entry in os.listdir(result_dir):
        tpath = os.path.join(result_dir, entry)
        if not os.path.isdir(tpath):
            continue
        msg_file = os.path.join(tpath, "interact_messages.json")
        if not os.path.exists(msg_file):
            print(f"interact_messages.json not found in {entry}, skipping")
            continue

        name = entry[len("task"):] if entry.startswith("task") else entry
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        app_id, sub_str = parts
        if app_id not in id_to_idx:
            continue
        try:
            sub_idx = int(sub_str)
        except ValueError:
            continue
        idx = id_to_idx[app_id]
        primary = test_datas[idx]["Category"]["primary_category"]
        ui_list = test_datas[idx]["ui_instruct"]
        if sub_idx >= len(ui_list):
            continue
        task_primary = ui_list[sub_idx]["task_category"]["primary_category"]

        try:
            data = load_json(msg_file)
        except Exception as exc:
            print(f"[warn] failed to read {msg_file}: {exc}")
            continue
        text = ""
        for msg in reversed(data):
            if msg.get("role") == "assistant":
                text = msg.get("content", "") or ""
                break
        if "YES" in text:
            score += 1; yes_num += 1
            cats[primary]["yes_num"] += 1; cats[primary]["score"] += 1
            cats[task_primary]["yes_num"] += 1; cats[task_primary]["score"] += 1
        elif "PARTIAL" in text:
            score += 0.5; partial_num += 1
            cats[primary]["partial_num"] += 1; cats[primary]["score"] += 0.5
            cats[task_primary]["partial_num"] += 1; cats[task_primary]["score"] += 0.5
        else:
            no_num += 1
            cats[primary]["no_num"] += 1
            cats[task_primary]["no_num"] += 1

    for v in cats.values():
        v["start_failed_num"] = v["total"] - v["yes_num"] - v["partial_num"] - v["no_num"]
        v["accuracy"] = v["score"] / v["total"] * 100 if v["total"] > 0 else 0

    start_failed = total - yes_num - partial_num - no_num
    test_name = os.path.basename(in_dir.rstrip("/"))
    pct = lambda n: (n / total * 100) if total else 0
    headers = PRIMARY_CATEGORIES + INST_PRIMARY_CATEGORIES

    table = (
        "| test_name | yes_num | partial_num | no_num | start_failed_num | total | "
        "yes_rate | partial_rate | no_rate | start_failed_rate | accuracy | "
        + " | ".join(headers) + " |\n"
        + "|------|------|------|------|------|------|------|------|------|------|------|"
        + "------|" * len(headers) + "\n"
        + f"| {test_name} | {yes_num} | {partial_num} | {no_num} | {start_failed} | "
          f"{total} | {pct(yes_num):.1f} | {pct(partial_num):.1f} | {pct(no_num):.1f} | "
          f"{pct(start_failed):.1f} | {pct(score):.1f} | "
        + " | ".join(f"{cats[c]['accuracy']:.1f}" for c in headers) + " |\n"
    )
    with open(os.path.join(in_dir, "table.md"), "w", encoding="utf-8") as f:
        f.write(table)
    print(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--test-file", default="data/test.jsonl",
                    help="JSONL test file (relative paths resolved under data/WebGen-Bench)")
    ap.add_argument("--app-id-list", default="")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--wv-api-key",
                    default=os.environ.get("WEBVOYAGER_API_KEY")
                            or os.environ.get("OPENAILIKE_VLM_API_KEY")
                            or os.environ.get("OPENAILIKE_API_KEY")
                            or os.environ.get("OPENAI_API_KEY", "token123"))
    ap.add_argument("--wv-api-model",
                    default=os.environ.get("WEBVOYAGER_API_MODEL", "gpt-5-mini"))
    ap.add_argument("--wv-api-base",
                    default=os.environ.get("WEBVOYAGER_API_BASE")
                            or os.environ.get("OPENAILIKE_VLM_BASE_URL")
                            or os.environ.get("OPENAILIKE_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--wv-reasoning-effort",
                    default=os.environ.get("WEBVOYAGER_REASONING_EFFORT", "high"),
                    choices=["", "minimal", "low", "medium", "high"],
                    help="Reasoning effort for GPT-5 / o-series models (empty string disables)")
    ap.add_argument("--port-base", type=int, default=None,
                    help="Optional. Accepted for interface compatibility with the "
                         "host orchestrator; start_services autodetects the bound port.")
    ap.add_argument("--max-iter", type=int,
                    default=int(os.environ.get("WEBVOYAGER_MAX_ITER", "15")),
                    help="WebVoyager per-task interaction budget (also patched into "
                         "the UI prompt template). Env: WEBVOYAGER_MAX_ITER.")
    args = ap.parse_args()

    test_file = args.test_file
    if not os.path.isabs(test_file):
        test_file = str(WEBGEN_ROOT / test_file)
    test_datas = load_jsonl(test_file)

    if args.app_id_list:
        allowed = {tid.strip() for tid in args.app_id_list.split(",") if tid.strip()}
        test_datas = [d for d in test_datas if d["id"] in allowed]

    if not test_datas:
        print("No test data after filtering.")
        return

    in_dir = args.in_dir
    print(f"[config] in_dir={in_dir} apps={len(test_datas)} batch_size={args.batch_size} "
          f"num_workers={args.num_workers} api_base={args.wv_api_base}")

    app_paths = [os.path.join(in_dir, d["id"]) for d in test_datas]

    # log.jsonl tracks batches already run — preserve upstream semantics.
    log_file = os.path.join(in_dir, "log.jsonl")
    completed = set()
    if os.path.isfile(log_file):
        for row in load_jsonl(log_file):
            completed.add(row.get("app_path"))
    remaining = [p for p in app_paths if p not in completed]
    tasks_file = os.path.join(in_dir, "tasks_test_with_answer.jsonl")

    for i in range(0, len(remaining), args.batch_size):
        batch = remaining[i:i + args.batch_size]
        commands = make_commands(batch, in_dir)
        ports = start_services(in_dir, commands) or {}
        print(ports)
        # Vite banners before dep-scan completes; pause so selenium doesn't
        # navigate to a blank page.
        time.sleep(5)
        batch_ids = {os.path.basename(p) for p in batch}
        batch_datas = [d for d in test_datas if d["id"] in batch_ids]
        build_tasks(batch_datas, ports, tasks_file, args.max_iter)
        try:
            run_webvoyager(Path(in_dir), args.num_workers,
                           args.wv_api_key, args.wv_api_model, args.wv_api_base,
                           args.wv_reasoning_effort or None, args.max_iter)
        finally:
            for tid in batch_ids:
                subprocess.run(["pm2", "delete", tid], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        save_jsonl([{"app_path": p} for p in batch], log_file, mode="a")

    compute_accuracy(in_dir, test_datas)


if __name__ == "__main__":
    main()
