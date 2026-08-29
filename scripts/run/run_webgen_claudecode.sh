#!/usr/bin/env bash
# Wrapper for baselines/claudecode/run_webgen_claudecode.py — the "recent
# agent-based SE system" comparison arm using the Claude Code CLI as the SCAFFOLD,
# driven by deepseek-v4-flash (NOT a Claude model) via a local LiteLLM proxy.
#
# Fairness (identical to ZS / OpenHands, only the scaffold differs): deepseek-v4-flash
# backbone with temperature=0 + reasoning_effort=high + native-deepseek provider pin
# (all enforced in baselines/claudecode/litellm_config.yaml, matching el-agent
# _factory.py); same SLA coding prompt; node20 sandbox (cc-sandbox ≈ sla-base).
#
# Prereqs: OPENROUTER_API_KEY in .env; `docker build -t cc-sandbox docker/cc-sandbox/`;
# LiteLLM proxy venv at baselines/claudecode/.venv. This script auto-starts the proxy.
#
# Task selection: TASK_LIST (CSV) overrides CLUSTER_ID (cluster.json).
# Usage:  TAG=cc-deepseek-c13-t1 CLUSTER_ID=13 bash scripts/run/run_webgen_claudecode.sh
set -euo pipefail

TAG="${TAG:-cc-deepseek-baseline}"
TASK_LIST="${TASK_LIST:-}"
CLUSTER_ID="${CLUSTER_ID:-13}"
CLUSTER_SIZE="${CLUSTER_SIZE:-8}"
LIMIT="${LIMIT:-0}"
MODEL="${MODEL:-deepseek-cc}"                 # litellm proxy alias (maps to deepseek-v4-flash)
PROXY_PORT="${PROXY_PORT:-}"                   # empty → use/derive baselines/claudecode/proxy_port.txt
TASK_FILE="${TASK_FILE:-}"
CLUSTER_JSON="${CLUSTER_JSON:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CCDIR="$REPO_ROOT/baselines/claudecode"
VENV="$CCDIR/.venv"
TASK_FILE="${TASK_FILE:-$REPO_ROOT/data/WebGen-Bench/data/test.jsonl}"
CLUSTER_JSON="${CLUSTER_JSON:-$REPO_ROOT/data/augments/webgen/cluster/cluster.json}"

[[ -x "$VENV/bin/litellm" ]] || { echo "missing proxy venv $VENV (uv venv + uv pip install 'litellm[proxy]')"; exit 1; }

# ── ensure the deepseek LiteLLM proxy is running ────────────────────────────
PORT="${PROXY_PORT:-$(grep -oE '[0-9]+' "$CCDIR/proxy_port.txt" 2>/dev/null || echo 8801)}"
if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" 2>/dev/null; then
  echo "[proxy] starting LiteLLM deepseek proxy on :$PORT"
  set -a; . "$REPO_ROOT/.env"; set +a
  ( cd "$CCDIR" && nohup "$VENV/bin/litellm" --config litellm_config.yaml --port "$PORT" --host 127.0.0.1 > proxy.log 2>&1 & )
  echo "$PORT" > "$CCDIR/proxy_port.txt"
  for i in $(seq 1 20); do sleep 2; curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" 2>/dev/null && break; done
fi
curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" || { echo "[proxy] failed to come up on :$PORT"; exit 1; }
echo "[proxy] deepseek proxy live on :$PORT"

# ── resolve task ids ────────────────────────────────────────────────────────
TASK_ARGS=()
if [[ -n "$TASK_LIST" ]]; then
  IFS=',' read -ra IDS <<< "$TASK_LIST"
else
  mapfile -t IDS < <(python3 - "$CLUSTER_JSON" "$CLUSTER_ID" "$CLUSTER_SIZE" <<'PY'
import json, sys
cj, cid, size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = json.load(open(cj))
e = next((c for c in d["clusters"] if int(c["id"]) == cid), None)
if e is None: sys.exit(f"cluster {cid} not found")
ids = e.get("tasks", [])
print("\n".join(str(i) for i in (ids[:size] if size > 0 else ids)))
PY
)
fi
for id in "${IDS[@]}"; do [[ -n "$id" ]] && TASK_ARGS+=(--task-id "$id"); done

EXTRA=(); [[ ${#TASK_ARGS[@]} -eq 0 && "$LIMIT" -gt 0 ]] && EXTRA+=(--limit "$LIMIT")
echo "TAG=$TAG  MODEL=$MODEL(deepseek-v4-flash)  IMAGE=cc-sandbox  tasks=${#TASK_ARGS[@]}"

exec python3 "$CCDIR/run_webgen_claudecode.py" \
  --tag "$TAG" --model "$MODEL" --proxy-port "$PORT" --tasks "$TASK_FILE" \
  "${TASK_ARGS[@]}" "${EXTRA[@]}" "$@"
