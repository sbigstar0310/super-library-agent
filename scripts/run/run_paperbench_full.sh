#!/usr/bin/env bash
# Wrapper for run/paperbench_full_run.py — paperbench full-mode pipeline.
#
# Four modes (set via MODE):
#   baseline         single round, coding only, no library
#   sla_naive        per round: coding (parallel M) → unified PaperbenchLibraryAgent
#   sla_naive_split  per round: coding → global_extract → apply
#                    (ablation of sla_ours: no local_extract, no extract_map,
#                     no extract/apply candidates — overrides forced in run.py)
#   sla_ours         per round: coding → local_extract → global_extract → apply
#   librarian        single round (Librarian post-hoc baseline): ZS-seeded
#                    corpus → LIBRARIAN_K unified-LibraryAgent samples (temp>0)
#                    → static compile/import gate + 1 repair turn → MDL rerank →
#                    promote winner. REQUIRES SEED_CODING_TAG and TEMPERATURE>0
#                    (candidate strategies forced to none in run.py).
#
# Task selection: either TASK_LIST (CSV) or CLUSTER_ID (int, picked from
# data/augments/paperbench/cluster/cluster.json). CLUSTER_SIZE optionally truncates.
#
# Any extra args you pass on the command line ("$@") are appended after our
# defaults.

set -euo pipefail

# ============================================================================
# EXP CONFIG  ← edit these per run
# ============================================================================
TAG="${TAG:-sla-ours-c1-t1}"
MODE="${MODE:-sla_ours}"                            # baseline | sla_naive | sla_naive_split | sla_ours | librarian
M="${M:-1}"                                         # papers per round (sla_*)
TASK_LIST="${TASK_LIST:-}"                          # CSV; overrides CLUSTER_ID
CLUSTER_ID="${CLUSTER_ID:-1}"                       # int; paperbench cluster 0=rl, 1=vision, 2=llm
CLUSTER_SIZE="${CLUSTER_SIZE:-}"                    # truncate cluster
CLUSTER_JSON="${CLUSTER_JSON:-}"                    # default: data/augments/paperbench/cluster/cluster.json
SOURCE_LIBRARY_DIR="${SOURCE_LIBRARY_DIR:-}"        # OPTIONAL round-1 seed lib (rarely used)
SEED_CODING_TAG="${SEED_CODING_TAG:-}"              # OPTIONAL: copy round_1 coding from sibling tag
MAX_WORKERS="${MAX_WORKERS:-4}"                     # BLAS contention: 4-way → 8x slowdown on 14-core
K="${K:-0}"                                         # keep 0 (no feedback adapter). NOTE: this is --k (apply feedback), NOT the librarian sample count — that's LIBRARIAN_K.
LIBRARIAN_K="${LIBRARIAN_K:-8}"                     # MODE=librarian: number of library candidates to sample (→ --librarian-k). SEED_CODING_TAG REQUIRED; set TEMPERATURE>0 (e.g. 0.8).
CANDIDATE_STRATEGY="${CANDIDATE_STRATEGY:-nl}"      # embed | nl | none (sla_naive_split forces 'none')
NL_PICK_MODEL="${NL_PICK_MODEL:-}"                  # empty = auto-derive from MODEL
EXTRACT_MAP="${EXTRACT_MAP:-true}"                  # sla_ours only
LOCAL_EXTRACT="${LOCAL_EXTRACT:-true}"              # sla_ours only
LIBRARY_CANDIDATE_STRATEGY="${LIBRARY_CANDIDATE_STRATEGY:-none}"  # sla_naive: none|embed|nl
REASONING_EFFORT="${REASONING_EFFORT:-}"            # empty = default; low|medium|high to override
TIME_LIMIT_HOURS="${TIME_LIMIT_HOURS:-2.0}"         # rendered into coding prompt "Total Runtime" bullet
# ============================================================================
# Cocoindex chunk params — read by patched utils.candidates.cocoindex_runner.
# Defaults are cocoindex vanilla (1000/250/150) so unset = backward compat.
# Daemon caches chunker module at first import; for clean param switching
# between runs, also set COCOINDEX_CODE_DIR to a per-run path.
export CCC_CHUNK_SIZE="${CCC_CHUNK_SIZE:-1000}"
export CCC_MIN_CHUNK_SIZE="${CCC_MIN_CHUNK_SIZE:-250}"
export CCC_CHUNK_OVERLAP="${CCC_CHUNK_OVERLAP:-150}"
# ============================================================================
# Static knobs (rarely change)
# ----------------------------------------------------------------------------
# Lab policy: route deepseek via OpenRouter (pinned to the official deepseek
# provider in _build_openrouter_config) instead of the direct DeepSeek API,
# to avoid third-party quantized serving. Model slug is unchanged.
PROVIDER="${PROVIDER:-openrouter}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
DOCKER_IMAGE="${DOCKER_IMAGE:-paperbench-base}"
COST_LIMIT="${COST_LIMIT:-5}"
STEP_LIMIT="${STEP_LIMIT:-150}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_ITER="${MAX_ITER:-1}"
PAPERBENCH_DATA_DIR="${PAPERBENCH_DATA_DIR:-}"      # default: <project>/data/frontier-evals/project/paperbench/data
# ============================================================================

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "$project_dir/.env" ]]; then
    set -a; source "$project_dir/.env"; set +a
fi

abs_path() {
    local p="$1"
    [[ -z "$p" ]] && { echo ""; return; }
    if [[ "$p" = /* ]]; then echo "$p"; else echo "$project_dir/$p"; fi
}
SOURCE_LIBRARY_DIR="$(abs_path "$SOURCE_LIBRARY_DIR")"
PAPERBENCH_DATA_DIR="$(abs_path "$PAPERBENCH_DATA_DIR")"
CLUSTER_JSON="$(abs_path "$CLUSTER_JSON")"

args=(
    --tag "$TAG"
    --mode "$MODE"
    --m "$M"
    --provider "$PROVIDER"
    --model "$MODEL"
    --docker-image "$DOCKER_IMAGE"
    --cost-limit "$COST_LIMIT"
    --step-limit "$STEP_LIMIT"
    --temperature "$TEMPERATURE"
    --max-workers "$MAX_WORKERS"
    --max-iter "$MAX_ITER"
    --k "$K"
    --candidate-strategy "$CANDIDATE_STRATEGY"
    --nl-pick-model "$NL_PICK_MODEL"
    --library-candidate-strategy "$LIBRARY_CANDIDATE_STRATEGY"
    --time-limit-hours "$TIME_LIMIT_HOURS"
)

# librarian mode: sample count (candidate strategies are forced to none in
# run.py). SEED_CODING_TAG + TEMPERATURE>0 are validated python-side.
if [[ "$MODE" == "librarian" ]]; then
    args+=(--librarian-k "$LIBRARIAN_K")
fi

if [[ -n "$TASK_LIST" ]]; then
    args+=(--task-list "$TASK_LIST")
elif [[ -n "${CLUSTER_ID:-}" ]]; then
    args+=(--cluster-id "$CLUSTER_ID")
    if [[ -n "${CLUSTER_SIZE:-}" ]]; then
        args+=(--cluster-size "$CLUSTER_SIZE")
    fi
fi
if [[ -n "$CLUSTER_JSON" ]]; then args+=(--cluster-json "$CLUSTER_JSON"); fi

if [[ -n "$REASONING_EFFORT" ]]; then
    args+=(--reasoning-effort "$REASONING_EFFORT")
fi
if [[ -n "$PAPERBENCH_DATA_DIR" ]]; then
    args+=(--paperbench-data-dir "$PAPERBENCH_DATA_DIR")
fi
if [[ -n "$SOURCE_LIBRARY_DIR" ]]; then
    args+=(--source-library-dir "$SOURCE_LIBRARY_DIR")
fi
if [[ -n "$SEED_CODING_TAG" ]]; then
    args+=(--seed-coding-tag "$SEED_CODING_TAG")
fi

case "${EXTRACT_MAP,,}" in
    1|true|yes|on)   args+=(--extract-map) ;;
    0|false|no|off)  args+=(--no-extract-map) ;;
    *) echo "EXTRACT_MAP='$EXTRACT_MAP' invalid (expected true/false)"; exit 2 ;;
esac

# LocalExtract is gated by the PAPERBENCH_LOCAL_EXTRACT env var (read inside
# PaperbenchFullRun.run_extract_phase), not a CLI flag.
case "${LOCAL_EXTRACT,,}" in
    1|true|yes|on)   export PAPERBENCH_LOCAL_EXTRACT=1 ;;
    0|false|no|off)  export PAPERBENCH_LOCAL_EXTRACT=0 ;;
    *) echo "LOCAL_EXTRACT='$LOCAL_EXTRACT' invalid (expected true/false)"; exit 2 ;;
esac

# Snapshot this wrapper into the backup tag for reproducibility.
backup_tag_dir="$project_dir/backups/paperbench/$TAG"
mkdir -p "$backup_tag_dir"
cp "${BASH_SOURCE[0]}" "$backup_tag_dir/run.sh" 2>/dev/null || true

cd "$project_dir/el-agent/src"
echo "==> paperbench_full_run  tag=$TAG  mode=$MODE  m=$M  workers=$MAX_WORKERS  task_list=${TASK_LIST:-(none)}  cluster_id=${CLUSTER_ID:-(none)}  local_extract=$PAPERBENCH_LOCAL_EXTRACT  extract_map=$EXTRACT_MAP  chunk=${CCC_CHUNK_SIZE}/${CCC_MIN_CHUNK_SIZE}/${CCC_CHUNK_OVERLAP}"
exec uv --project "$project_dir/el-agent" run python -m run.paperbench_full_run "${args[@]}" "$@"
