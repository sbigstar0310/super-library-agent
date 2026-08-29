#!/usr/bin/env bash
# webgen_eval_entrypoint.sh
#
# In-container batch dispatcher for WebGen-Bench evaluation. The host
# orchestrator (`scripts/eval/eval_webgen.sh`) stages every task under
# `/home/<task_id>/` (one bind-mount per task, all RW) and a shared library
# under `/home/lib/` (RO). This entrypoint then runs the existing
# `webgen_appearance` / `webgen_ui_test` python modules from the bind-mounted
# `el-agent` source tree, with `--in-dir /home` so task discovery walks the
# `/home/<id>/` siblings directly — matching the training-time cwd layout
# (`/home/submission/` next to `/home/lib/`).
#
# Required env:
#   EVAL_MODE          appearance | ui-test
#
# Optional env (all forwarded as CLI when set):
#   EVAL_TEST_FILE     default /home/webgen/data/test.jsonl
#   NUM_WORKERS        default 4
#   PORT_BASE          default 21000
#   EVAL_BATCH_SIZE    default 10 (ui-test batches per pm2 run)
#   EVAL_APP_LIST      comma-separated task ids; empty = autodiscover under /home
#
# Plus the model/API env vars the python evaluators already read directly:
#   WEBGEN_APPEARANCE_MODEL, WEBGEN_APPEARANCE_REASONING_EFFORT
#   OPENAILIKE_VLM_API_KEY / OPENAILIKE_API_KEY / OPENAI_API_KEY (fallback chain)
#   OPENAILIKE_VLM_BASE_URL / OPENAILIKE_BASE_URL / OPENAI_BASE_URL
#   WEBVOYAGER_API_KEY, WEBVOYAGER_API_MODEL, WEBVOYAGER_API_BASE,
#   WEBVOYAGER_REASONING_EFFORT

set -euo pipefail

: "${EVAL_MODE:?EVAL_MODE required (appearance|ui-test)}"

EVAL_TEST_FILE="${EVAL_TEST_FILE:-/home/webgen/data/test.jsonl}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PORT_BASE="${PORT_BASE:-21000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"

# el-agent source is RO-bind-mounted at /home/el-agent. Make `eval.*` importable.
export PYTHONPATH="/home/el-agent/src:${PYTHONPATH:-}"

# Isolated pm2 daemon per container run (auto-cleaned by --rm on exit).
export PM2_HOME="/tmp/pm2-eval-$$"
mkdir -p "$PM2_HOME"

# Surface chromium binary paths to selenium/webvoyager.
export CHROME_BIN="${CHROME_BIN:-/usr/bin/chromium}"
export CHROMEDRIVER="${CHROMEDRIVER:-/usr/bin/chromedriver}"

# WebGen-Bench source is RO-mounted at /home/webgen.
export WEBGEN_ROOT="${WEBGEN_ROOT:-/home/webgen}"

cd /home

case "$EVAL_MODE" in
  appearance)
    python -m eval.webgen_appearance \
        --in-dir /home \
        --test-file "$EVAL_TEST_FILE" \
        --num-workers "$NUM_WORKERS" \
        --port-base "$PORT_BASE" \
        ${EVAL_APP_LIST:+--app-id-list "$EVAL_APP_LIST"}
    ;;
  ui-test)
    python -m eval.webgen_ui_test \
        --in-dir /home \
        --test-file "$EVAL_TEST_FILE" \
        --num-workers "$NUM_WORKERS" \
        --port-base "$PORT_BASE" \
        --batch-size "$EVAL_BATCH_SIZE" \
        ${EVAL_APP_LIST:+--app-id-list "$EVAL_APP_LIST"}
    ;;
  *)
    echo "[webgen_eval_entrypoint] Unknown EVAL_MODE=$EVAL_MODE" >&2
    exit 64
    ;;
esac
