#!/usr/bin/env bash
# Wrapper for baselines/openhands/run_webgen_openhands.py — the "recent agent-based
# SE system" comparison arm for reviewer (4). Runs the latest OpenHands agent
# (software-agent-sdk v1.34) as a WebGen generation baseline: each app is generated
# INDEPENDENTLY (no shared library), controlled to the same backbone + prompt +
# sandbox image as the SLA Zero-Shot arm.
#
# Fairness controls (identical to Zero-Shot, only the agent scaffold differs):
#   - Prompt : imported verbatim from prompts.webgen.coding_agent (el-agent/src).
#   - Backbone: MODEL (default deepseek/deepseek-v4-flash via openrouter).
#   - Sandbox : DOCKER_IMAGE (default sla-base) — same runtime the CodingAgent uses.
#
# NOTE: OpenHands ships its own agent stack + pinned deps (openhands-sdk), which
# conflict with el-agent's uv env, so it is NOT fused into base_full_run.py's
# CodingAgent contract. This launcher keeps the *interface* (env vars) and *output
# path* (backups/webgen/<TAG>/final/round_1/coding) identical to the other arms.
#
# Task selection: TASK_LIST (CSV of ids) overrides CLUSTER_ID (cluster.json).
#
# Usage:
#   TAG=openhands-c13-t1 CLUSTER_ID=13 bash scripts/run/run_webgen_openhands.sh
#   TAG=openhands-smoke  TASK_LIST=000001,000002 bash scripts/run/run_webgen_openhands.sh

set -euo pipefail

# ============================================================================
# EXP CONFIG  ← edit per run
# ============================================================================
TAG="${TAG:-openhands-baseline}"
TASK_LIST="${TASK_LIST:-}"                          # CSV of task ids; overrides CLUSTER_ID
CLUSTER_ID="${CLUSTER_ID:-13}"                      # int; resolved from cluster.json
CLUSTER_SIZE="${CLUSTER_SIZE:-8}"                    # truncate cluster (0 = all)
LIMIT="${LIMIT:-0}"                                  # cap #tasks (0 = no cap); ignored if TASK_LIST/CLUSTER set
# ----------------------------------------------------------------------------
PROVIDER="${PROVIDER:-openrouter}"                  # litellm provider prefix
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"        # litellm model (sans provider)
DOCKER_IMAGE="${DOCKER_IMAGE:-sla-base:latest}"     # sandbox base image (agent-server built on top)
# Fairness knobs — mirror the SLA Zero-Shot mini-swe run (run_webgen_full.sh):
STEP_LIMIT="${STEP_LIMIT:-150}"                     # → --max-iterations (Zero-Shot STEP_LIMIT=150)
TEMPERATURE="${TEMPERATURE:-0.0}"                   # → --temperature   (Zero-Shot TEMPERATURE=0.0)
REASONING_EFFORT="${REASONING_EFFORT:-high}"        # → --reasoning-effort (deepseek reasoning family = high in _factory.py)
# NOTE: Zero-Shot also sets COST_LIMIT=$5, but deepseek at STEP_LIMIT steps costs
# << $5, so the cost cap is non-binding for both scaffolds (no native cap needed).
TARGET="${TARGET:-source-minimal}"                 # agent-server build target (lighter than 'source')
TASK_FILE="${TASK_FILE:-}"                          # default: data/WebGen-Bench/data/test.jsonl
CLUSTER_JSON="${CLUSTER_JSON:-}"                    # default: data/augments/webgen/cluster/cluster.json
# ============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$REPO_ROOT/baselines/openhands/run_webgen_openhands.py"
VENV="$REPO_ROOT/baselines/openhands/.venv"
TASK_FILE="${TASK_FILE:-$REPO_ROOT/data/WebGen-Bench/data/test.jsonl}"
CLUSTER_JSON="${CLUSTER_JSON:-$REPO_ROOT/data/augments/webgen/cluster/cluster.json}"

[[ -x "$VENV/bin/python" ]] || { echo "missing venv: $VENV (run: uv venv + uv pip install openhands-sdk openhands-tools openhands-workspace)"; exit 1; }

# Resolve task ids -> --task-id args
TASK_ARGS=()
if [[ -n "$TASK_LIST" ]]; then
  IFS=',' read -ra IDS <<< "$TASK_LIST"
  for id in "${IDS[@]}"; do TASK_ARGS+=(--task-id "$id"); done
else
  # pull ids for CLUSTER_ID from cluster.json (optionally truncated to CLUSTER_SIZE)
  mapfile -t IDS < <("$VENV/bin/python" - "$CLUSTER_JSON" "$CLUSTER_ID" "$CLUSTER_SIZE" <<'PY'
import json, sys
cj, cid, size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
data = json.load(open(cj))
# schema: {"clusters": [{"id": N, "size": .., "tasks": ["000019", ...]}, ...]}
entry = next((c for c in data["clusters"] if int(c["id"]) == cid), None)
if entry is None:
    sys.exit(f"cluster id {cid} not found in {cj}")
ids = entry.get("tasks", [])
if size > 0:
    ids = ids[:size]
print("\n".join(str(i) for i in ids))
PY
)
  for id in "${IDS[@]}"; do [[ -n "$id" ]] && TASK_ARGS+=(--task-id "$id"); done
fi

EXTRA=()
[[ ${#TASK_ARGS[@]} -eq 0 && "$LIMIT" -gt 0 ]] && EXTRA+=(--limit "$LIMIT")

echo "TAG=$TAG  MODEL=$PROVIDER/$MODEL  IMAGE=$DOCKER_IMAGE  TARGET=$TARGET  task_args=${#TASK_ARGS[@]}  LIMIT=$LIMIT"

exec "$VENV/bin/python" -u "$RUNNER" \
  --tag "$TAG" \
  --model "$PROVIDER/$MODEL" \
  --base-image "$DOCKER_IMAGE" \
  --target "$TARGET" \
  --max-iterations "$STEP_LIMIT" \
  --temperature "$TEMPERATURE" \
  --reasoning-effort "$REASONING_EFFORT" \
  --tasks "$TASK_FILE" \
  "${TASK_ARGS[@]}" "${EXTRA[@]}" "$@"
