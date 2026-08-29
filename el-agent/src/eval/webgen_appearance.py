"""WebGen-Bench appearance evaluation (filtered wrapper).

Reuses helpers from ``data/WebGen-Bench/src/grade_appearance_webgen`` but adds:
- ``--app-id-list`` filter (vs original iterating the full test.jsonl)
- aggregate grade denominator = number of filtered apps (not 101)
- configurable ``--test-file``, ``--tag``, ``--num-workers``, ``--model``

Outputs (under ``--in-dir``):
- ``<app>/shots/shot_1.png``           — first-fold screenshot
- ``<app>/shots/result{tag}.json``      — VLM model output
- ``grade{tag}.json``                   — aggregate grade across filtered apps
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import as_completed
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
# `WEBGEN_ROOT` override targets the RO-mounted tree at `/home/webgen` inside
# the sla-base eval container (parents[3] resolves outside the mount).
WEBGEN_ROOT = Path(os.environ.get("WEBGEN_ROOT") or (PROJECT_DIR / "data" / "WebGen-Bench"))
APPEARANCE_DIR = WEBGEN_ROOT / "src" / "grade_appearance_webgen"
sys.path.insert(0, str(APPEARANCE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(PROJECT_DIR / ".env"))
except ImportError:
    pass

from start_service import start_services  # noqa: E402
from get_screenshots import capture_scroll_screenshots  # noqa: E402
# Reuse the env-resolved client + payload/prompt; API kwargs stay local so
# reasoning-model handling lives in this wrapper.
from vlm_eval import openai_client, _build_openai_payload, _encode_image  # noqa: E402
from prompt import appearance_prompt  # noqa: E402


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def score_image(image_paths, instruction: str, model: str,
                max_tokens: int = 1024, reasoning_effort: str | None = None) -> str:
    """OpenAI-only appearance scoring.

    For reasoning models (gpt-5/o-series), ``max_tokens`` covers both hidden
    reasoning and visible output, so we 4× it and switch to
    ``max_completion_tokens`` (the only budget param those models accept).
    """
    base64_imgs = [(_encode_image(p), p) for p in image_paths]
    messages = _build_openai_payload(base64_imgs, appearance_prompt.format(instruction=instruction))
    kwargs = {"model": model, "messages": messages}
    if _is_reasoning_model(model):
        kwargs["max_completion_tokens"] = max(max_tokens * 4, 4096)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["max_tokens"] = max_tokens
    resp = openai_client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def make_commands(app_paths, in_dir):
    commands = {os.path.basename(p): {"shell_actions": [], "last_start_action": ""} for p in app_paths}
    save_json(commands, os.path.join(in_dir, "commands.json"))
    return commands


def first_grade_int(text: str) -> int:
    m = re.search(r"Grade.*?(\d)", text, flags=re.IGNORECASE | re.DOTALL)
    return int(m.group(1)) if m else 0


def score_single(app_id: str, datum: dict, in_dir: str, tag: str,
                 model: str, reasoning_effort: str | None) -> str:
    shot_dir = os.path.join(in_dir, app_id, "shots")
    result_path = os.path.join(shot_dir, f"result{tag}.json")
    if os.path.isfile(result_path):
        return f"[{app_id}] result exists — skipped"
    if not os.path.isdir(shot_dir):
        return f"[{app_id}] no shots dir — skipped"
    images = [os.path.join(shot_dir, fn) for fn in os.listdir(shot_dir) if fn.endswith(".png")]
    if not images:
        return f"[{app_id}] no PNGs — skipped"
    output = score_image(images, datum["instruction"], model=model,
                         reasoning_effort=reasoning_effort)
    save_json({"model_output": output}, result_path)
    return f"[{app_id}] scored {len(images)} images"


def _shots_dir_has_png(shots_dir: str) -> bool:
    return os.path.isdir(shots_dir) and any(fn.endswith(".png") for fn in os.listdir(shots_dir))


def capture_phase(in_dir: str, pending, max_attempts: int = 3):
    """Capture first-fold screenshot per pending app (one at a time, like upstream).

    Retries up to ``max_attempts`` per app on missing PNG (Vite cold-start
    race or port-detection timeout).
    """
    if not pending:
        return
    for datum in pending:
        subprocess.run(["pm2", "delete", datum["id"]], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for datum in pending:
        app = datum["id"]
        app_path = os.path.join(in_dir, app)
        if not os.path.isdir(app_path):
            print(f"[skip] {app}: missing app dir {app_path}")
            continue
        shots_dir = os.path.join(in_dir, app, "shots")
        for attempt in range(1, max_attempts + 1):
            commands = make_commands([app_path], in_dir)
            ports = start_services(in_dir, commands) or {}
            print(ports)
            # Vite banners before dep-scan completes; pause so capture doesn't
            # race the first /src/main.jsx transform.
            time.sleep(5)
            for name, port in ports.items():
                capture_scroll_screenshots(
                    url=f"http://localhost:{port}/",
                    out_dir=os.path.join(in_dir, name, "shots"),
                    max_shots=1,
                    pause=0.4,
                    viewport_height=768,
                )
            subprocess.run(["pm2", "delete", app], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if _shots_dir_has_png(shots_dir):
                if attempt > 1:
                    print(f"[retry-ok] {app}: PNG captured on attempt {attempt}/{max_attempts}")
                break
            if attempt < max_attempts:
                backoff = 5 * attempt
                print(f"[retry] {app}: no PNG after attempt {attempt}/{max_attempts}, sleeping {backoff}s")
                time.sleep(backoff)
            else:
                print(f"[give-up] {app}: no PNG after {max_attempts} attempts")


def score_phase(in_dir: str, pending, tag: str, model: str, num_workers: int,
                reasoning_effort: str | None):
    """Score pending apps' screenshots via the VLM.

    Scoring is HTTPS-only, so a thread pool matches process-pool parallelism
    without the multiprocessing fragility seen in the sla-base container
    (workers killed after `capture_phase` spawned PM2/chromium).
    """
    if not pending:
        return
    tasks = [(d["id"], d, in_dir, tag, model, reasoning_effort) for d in pending]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        futures = [pool.submit(score_single, *t) for t in tasks]
        for fut in as_completed(futures):
            try:
                print(fut.result())
            except Exception as exc:
                print(f"Worker error: {exc}")


def aggregate_grade(in_dir: str, test_datas, tag: str):
    total = 0
    count = 0
    for d in test_datas:
        rp = os.path.join(in_dir, d["id"], "shots", f"result{tag}.json")
        if not os.path.isfile(rp):
            continue
        try:
            total += first_grade_int(load_json(rp)["model_output"])
            count += 1
        except Exception as exc:
            print(f"[warn] read {rp}: {exc}")
    denom = max(count, 1)
    grade = round(total / denom, 2)
    save_json({"grade": grade, "total": total, "count": count}, os.path.join(in_dir, f"grade{tag}.json"))
    print(f"Aggregate grade: {grade} (sum={total}, count={count})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory containing per-app subdirs")
    ap.add_argument("--test-file", default="data/test.jsonl",
                    help="JSONL test file (relative paths resolved under data/WebGen-Bench)")
    ap.add_argument("--tag", default="1", help="Suffix for result file names")
    ap.add_argument("--model", default=os.environ.get("WEBGEN_APPEARANCE_MODEL", "gpt-5-mini"),
                    help="VLM model name (env: WEBGEN_APPEARANCE_MODEL)")
    ap.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    ap.add_argument("--app-id-list", default="", help="Comma-separated app IDs to evaluate")
    ap.add_argument("--reasoning-effort",
                    default=os.environ.get("WEBGEN_APPEARANCE_REASONING_EFFORT"),
                    choices=[None, "minimal", "low", "medium", "high"],
                    help="reasoning_effort for GPT-5/o-series (default: SDK default)")
    ap.add_argument("--port-base", type=int, default=None,
                    help="Optional. Accepted for interface compatibility with the "
                         "host orchestrator; current `start_services` flow lets vite "
                         "pick its own port and reads it back from the PM2 log.")
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

    print(f"[config] in_dir={args.in_dir} apps={len(test_datas)} tag={args.tag} model={args.model}")

    pending_shot = [d for d in test_datas
                    if not os.path.isfile(os.path.join(args.in_dir, d["id"], "shots", "shot_1.png"))]
    capture_phase(args.in_dir, pending_shot)

    pending_score = [d for d in test_datas
                     if not os.path.isfile(os.path.join(args.in_dir, d["id"], "shots", f"result{args.tag}.json"))]
    score_phase(args.in_dir, pending_score, args.tag, args.model, args.num_workers,
                args.reasoning_effort)

    aggregate_grade(args.in_dir, test_datas, args.tag)


if __name__ == "__main__":
    main()
