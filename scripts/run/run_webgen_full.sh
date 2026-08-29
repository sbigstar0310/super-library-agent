#!/usr/bin/env bash
# Wrapper for run/webgen_full_run.py — WebGen-Bench full-mode pipeline.
#
# Five modes (set via MODE):
#   baseline         single round, coding only, no library
#   sla_naive        per round: coding (parallel M) → unified WebgenLibraryAgent
#   sla_naive_split  per round: coding → global_extract → apply
#                    (ablation of sla_ours: no local_extract, no extract_map,
#                     no extract/apply candidates — overrides forced in run.py)
#   sla_ours         per round: coding → local_extract → global_extract → apply
#   librarian        single round (Librarian post-hoc baseline): ZS-seeded
#                    corpus → LIBRARIAN_K unified-LibraryAgent samples (temp>0)
#                    → npm/vite build gate + 1 repair turn → MDL rerank →
#                    promote winner. REQUIRES SEED_CODING_TAG and TEMPERATURE>0
#                    (candidate strategies forced to none in run.py).
#
# Task selection: either TASK_LIST (CSV) or CLUSTER_ID (int, picked from
# data/augments/webgen/cluster/cluster.json). CLUSTER_SIZE optionally truncates.
#
# Any extra args you pass on the command line ("$@") are appended after our
# defaults.

set -euo pipefail

# ============================================================================
# EXP CONFIG  ← edit these per run
# ============================================================================
TAG="${TAG:-baseline-c13-lib-t1}"
MODE="${MODE:-baseline}"                            # baseline | sla_naive | sla_naive_split | sla_ours | librarian
M="${M:-4}"                                         # apps per round (sla_*)
TASK_LIST="${TASK_LIST:-}"                          # CSV; overrides CLUSTER_ID
CLUSTER_ID="${CLUSTER_ID:-13}"                      # int; webgen cluster 13, 2, 5
CLUSTER_SIZE="${CLUSTER_SIZE:-8}"                    # truncate cluster
SOURCE_LIBRARY_DIR="${SOURCE_LIBRARY_DIR:-}"        # OPTIONAL round-1 seed lib
SEED_CODING_TAG="${SEED_CODING_TAG:-}"              # OPTIONAL: copy round_1 coding from backups/webgen/<tag>/final/round_1/coding/ instead of regenerating
MAX_WORKERS="${MAX_WORKERS:-8}"
K="${K:-0}"                                         # keep 0 (no feedback adapter). NOTE: this is --k (apply feedback), NOT the librarian sample count — that's LIBRARIAN_K.
LIBRARIAN_K="${LIBRARIAN_K:-8}"                     # MODE=librarian: number of library candidates to sample (→ --librarian-k). SEED_CODING_TAG REQUIRED; set TEMPERATURE>0 (e.g. 0.8).
CANDIDATE_STRATEGY="${CANDIDATE_STRATEGY:-nl}"      # embed | nl | none (sla_naive_split forces 'none')
NL_PICK_MODEL="${NL_PICK_MODEL:-}"                  # empty = auto-derive from MODEL (strip litellm provider prefix); summary model hardcoded to gpt-5.4-nano
EXTRACT_MAP="${EXTRACT_MAP:-true}"                  # sla_ours only
LOCAL_EXTRACT="${LOCAL_EXTRACT:-true}"              # sla_ours only
INJECT_NEIGHBORS="${INJECT_NEIGHBORS:-true}"        # sla_ours apply: inject 1-hop dep neighbors into apply candidates
LIBRARY_CANDIDATE_STRATEGY="${LIBRARY_CANDIDATE_STRATEGY:-none}"  # sla_naive: none|embed|nl
REASONING_EFFORT="${REASONING_EFFORT:-}"                       # empty = default ("high" for reasoning families), low|medium|high to override
LAYOUT_SPECS_DIR="${LAYOUT_SPECS_DIR-data/augments/webgen/layout_specs}"                        # optional per-task layout-spec dir; empty disables. NOTE: uses `-` (not `:-`) so `LAYOUT_SPECS_DIR=` keeps it empty/disabled.
# ============================================================================
# Static knobs (rarely change)
# ----------------------------------------------------------------------------
PROVIDER="${PROVIDER:-openrouter}"  # deepseek, openai
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"  # litellm: <provider>/<model> required
DOCKER_IMAGE="${DOCKER_IMAGE:-sla-base}"
COST_LIMIT="${COST_LIMIT:-5}"
STEP_LIMIT="${STEP_LIMIT:-150}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_ITER="${MAX_ITER:-1}"
TASK_FILE="${TASK_FILE:-}"                          # default: data/WebGen-Bench/data/test.jsonl
CLUSTER_JSON="${CLUSTER_JSON:-}"                    # default: data/augments/webgen/cluster/cluster.json
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
TASK_FILE="$(abs_path "$TASK_FILE")"
CLUSTER_JSON="$(abs_path "$CLUSTER_JSON")"
LAYOUT_SPECS_DIR="$(abs_path "$LAYOUT_SPECS_DIR")"

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
)

if [[ -n "$REASONING_EFFORT" ]]; then
    args+=(--reasoning-effort "$REASONING_EFFORT")
fi

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
if [[ -n "$TASK_FILE" ]]; then args+=(--task-file "$TASK_FILE"); fi
if [[ -n "$CLUSTER_JSON" ]]; then args+=(--cluster-json "$CLUSTER_JSON"); fi
if [[ -n "$SOURCE_LIBRARY_DIR" ]]; then
    args+=(--source-library-dir "$SOURCE_LIBRARY_DIR")
fi
if [[ -n "$SEED_CODING_TAG" ]]; then
    args+=(--seed-coding-tag "$SEED_CODING_TAG")
fi
if [[ -n "$LAYOUT_SPECS_DIR" ]]; then
    args+=(--layout-specs-dir "$LAYOUT_SPECS_DIR")
fi

case "${EXTRACT_MAP,,}" in
    1|true|yes|on)   args+=(--extract-map) ;;
    0|false|no|off)  args+=(--no-extract-map) ;;
    *) echo "EXTRACT_MAP='$EXTRACT_MAP' invalid (expected true/false)"; exit 2 ;;
esac

# LocalExtract is gated by the WEBGEN_LOCAL_EXTRACT env var (read inside
# WebgenFullRun.run_extract_phase), not a CLI flag.
case "${LOCAL_EXTRACT,,}" in
    1|true|yes|on)   export WEBGEN_LOCAL_EXTRACT=1 ;;
    0|false|no|off)  export WEBGEN_LOCAL_EXTRACT=0 ;;
    *) echo "LOCAL_EXTRACT='$LOCAL_EXTRACT' invalid (expected true/false)"; exit 2 ;;
esac

# Neighbor injection gated by WEBGEN_INJECT_NEIGHBORS (read inside
# utils/candidates/nl.py:get_apply_candidates_nl). No CLI flag.
case "${INJECT_NEIGHBORS,,}" in
    1|true|yes|on)   export WEBGEN_INJECT_NEIGHBORS=1 ;;
    0|false|no|off)  export WEBGEN_INJECT_NEIGHBORS=0 ;;
    *) echo "INJECT_NEIGHBORS='$INJECT_NEIGHBORS' invalid (expected true/false)"; exit 2 ;;
esac

# Snapshot this wrapper into the backup tag for reproducibility.
backup_tag_dir="$project_dir/backups/webgen/$TAG"
mkdir -p "$backup_tag_dir"
cp "${BASH_SOURCE[0]}" "$backup_tag_dir/run.sh" 2>/dev/null || true

# Per-server daemon dir (NFS daemon socket race 방지)
export COCOINDEX_CODE_DIR="${COCOINDEX_CODE_DIR:-/tmp/cocoindex-$USER-$(hostname -s)}"
mkdir -p "$COCOINDEX_CODE_DIR"
[[ -f "$COCOINDEX_CODE_DIR/global_settings.yml" ]] || \
    cp ~/.cocoindex_code/global_settings.yml "$COCOINDEX_CODE_DIR/" 2>/dev/null || true

cd "$project_dir/el-agent/src"
echo "==> webgen_full_run  tag=$TAG  mode=$MODE  m=$M  workers=$MAX_WORKERS  task_list=${TASK_LIST:-(none)}  cluster_id=${CLUSTER_ID:-(none)}  local_extract=$WEBGEN_LOCAL_EXTRACT  inject_neighbors=$WEBGEN_INJECT_NEIGHBORS  extract_map=$EXTRACT_MAP"
exec uv --project "$project_dir/el-agent" run python -m run.webgen_full_run "${args[@]}" "$@"
