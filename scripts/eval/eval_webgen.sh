#!/bin/bash
# Evaluate WebGen-Bench submissions in a finalized backup tag (unified layout).
#
# Iterates tasks under
#     backups/webgen/<tag>/final/round_<N>/<phase>/tasks/<id>/submission/
# stages them into an eval_dir that mirrors the training-time layout:
#     eval_dir/apps/<id>/        ← submission
#     eval_dir/apps/lib/         ← shared library (when phase has one)
# then dispatches to the in-tree WebGen-Bench wrappers under el-agent/src/eval/.
# Result files are written under
#     backups/webgen/<tag>/eval_results/round_<N>/<phase>/
#
# Required env:  TAG
# Optional env:  PHASE         (default: baseline)
#                ROUND         (default: latest round_N under final/ with PHASE/)
#                TASK_LIST     (comma-separated subset; default: all task dirs)
#                MODE          (all | appearance | ui-test; default: all)
#                NUM_WORKERS   (default: 8)
#                BATCH_SIZE    (default: 10)
#                TEST_FILE     (relative to data/WebGen-Bench; default: data/test.jsonl)
#                KEEP_EVAL_DIR (1: keep $eval_dir for debugging; default: cleanup)
#                PORT_BASE     (override random port base; required for parallel
#                              eval runs to guarantee disjoint port ranges)
#                RESUME        (1: stage existing ui_test_results back into the
#                              eval_dir and purge sub-task dirs whose WebVoyager
#                              run hit the max-iter cap. Only those are re-run;
#                              other sub-tasks are skipped via WebVoyager's
#                              idempotent skip-if-interact_messages.json-exists.
#                              Requires MODE includes ui-test.)
#                WEBVOYAGER_MAX_ITER  (override WebVoyager per-task budget;
#                              default 15. Used together with RESUME=1 to widen
#                              the budget on the cap-hit re-runs.)
#
# Notes:
#  - Eval helpers live under data/WebGen-Bench (canonical) and are driven by
#    el-agent/src/eval/webgen_appearance.py and webgen_ui_test.py.
#  - Library lookup (phase != baseline):
#      coding | apply  → per-task tasks/<id>/lib/ — copied into eval_dir/apps/lib
#      extract         → phase-level round_<N>/extract/lib/
#    The lib lands at apps/lib so each app's vite.config.js alias
#    `path.resolve(__dirname, '../lib/src/index.js')` resolves the same way
#    it did at training time. A workspace package.json with workspaces=["apps/*"]
#    additionally makes `"ui-lib": "*"` installable via npm.
#    (cc_baseline currently always has lib_dir = none.)

set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Load .env so OPENAILIKE_* / WEBGEN_* / WEBVOYAGER_* API keys are available
# for forwarding into the sla-base eval container. Host-side dotenv loading
# is no longer implicit since eval now runs inside docker.
if [[ -f "$project_dir/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$project_dir/.env"
    set +a
fi

tag="${TAG:-}"
# BENCH selects the backups/<bench>/ subtree. Default `webgen`; set
# `webgen-maint` to evaluate maintenance-patched portfolios (same layout).
bench="${BENCH:-webgen}"
phase="${PHASE:-baseline}"
task_list="${TASK_LIST:-}"
round_arg="${ROUND:-}"
mode="${MODE:-all}"
num_workers="${NUM_WORKERS:-8}"
batch_size="${BATCH_SIZE:-10}"
test_file="${TEST_FILE:-data/test.jsonl}"
keep_eval_dir="${KEEP_EVAL_DIR:-0}"
resume="${RESUME:-0}"

if [[ -z "$tag" ]]; then
    echo "TAG is required. Example: TAG=cc-baseline-t1 PHASE=baseline bash scripts/eval/eval_webgen.sh"
    exit 1
fi

# PM2 / port pool isolation now happens INSIDE the sla-base eval container —
# each docker run gets its own PM2_HOME under /tmp and its own network
# namespace, so host-side PM2 trapping is no longer needed.
docker_image="${SLA_BASE_IMAGE:-sla-base:latest}"
container_name_prefix="webgen-eval-${tag}-$$"
cleanup_containers() {
    docker ps -aq --filter "name=^${container_name_prefix}" 2>/dev/null | \
        xargs -r docker rm -f >/dev/null 2>&1 || true
}
trap cleanup_containers EXIT

# Resolve round: ROUND env wins; otherwise pick the highest round_N under
# backups/webgen/<tag>/final/ that has the requested phase.
if [[ -n "$round_arg" ]]; then
    round_num="$round_arg"
else
    round_num=$(uv --project "$project_dir/el-agent" run python -c "
import sys
sys.path.insert(0, '$project_dir/el-agent/src')
from run import backup_layout
n = backup_layout.latest_round_in_backup('$project_dir', '$bench', '$tag', '$phase')
print('' if n is None else n)
")
    if [[ -z "$round_num" ]]; then
        echo "No round_N found for phase=$phase under backups/$bench/$tag/final/"
        exit 1
    fi
fi

backup_root="${project_dir}/backups/${bench}/${tag}/final/round_${round_num}/${phase}/tasks"
eval_root="${project_dir}/backups/${bench}/${tag}/eval_results/round_${round_num}/${phase}"
eval_dir="${project_dir}/eval/${bench}/${tag}/round_${round_num}/${phase}_$$"
mkdir -p "$eval_root" "$eval_dir/apps"

# Custom test file: when TEST_FILE is an absolute host path (e.g. a
# maintenance change-specific / regression jsonl outside data/WebGen-Bench),
# stage it into apps/ so it is visible in-container at /home, and reference
# it by that absolute container path. Otherwise keep the relative default
# (resolved against /home/webgen inside the container).
if [[ "$test_file" = /* && -f "$test_file" ]]; then
    cp "$test_file" "$eval_dir/apps/_eval_test.jsonl"
    test_file="/home/_eval_test.jsonl"
    echo "[TestFile] staged custom test file → /home/_eval_test.jsonl"
fi

if [[ ! -d "$backup_root" ]]; then
    echo "Backup not finalized: $backup_root"
    exit 1
fi

# Resolve task list
if [[ -n "$task_list" ]]; then
    IFS=',' read -ra task_ids <<<"$task_list"
else
    task_ids=()
    for d in "$backup_root"/*/; do
        [[ -d "$d/submission" ]] && task_ids+=("$(basename "$d")")
    done
fi

if [[ ${#task_ids[@]} -eq 0 ]]; then
    echo "No tasks to evaluate under $backup_root"
    exit 1
fi

echo "[Config] tag=$tag phase=$phase round=$round_num mode=$mode"
echo "[Config] backup_root=$backup_root"
echo "[Config] eval_root=$eval_root"
echo "[Config] eval_dir=$eval_dir"
echo "[Config] tasks=${task_ids[*]}"

# Stage submissions → eval_dir/apps/<id>/
for tid in "${task_ids[@]}"; do
    sub="$backup_root/$tid/submission"
    if [[ ! -d "$sub" ]]; then
        echo "[skip] $tid: missing submission/"
        continue
    fi
    # Use rsync to drop transient artifacts (mirrors eval_ral.sh).
    rsync -a \
        --exclude='__pycache__' --exclude='node_modules' --exclude='.git' \
        --exclude='shots' --exclude='embeddings' --exclude='dist' \
        "$sub/" "$eval_dir/apps/$tid/"
done

# Stage library at eval_dir/apps/lib/ — matches training-time layout so each
# app's vite alias `path.resolve(__dirname, '../lib/src/index.js')` resolves.
# Prefer phase-level lib (extract), then per-task lib (coding/apply).
phase_lib="${project_dir}/backups/${bench}/${tag}/final/round_${round_num}/${phase}/lib"
if [[ -d "$phase_lib" ]]; then
    rsync -a --exclude='node_modules' --exclude='.git' "$phase_lib/" "$eval_dir/apps/lib/"
    echo "[Lib] staged phase-level lib from $phase_lib → apps/lib"
else
    # Per-task lib (rare for webgen — kept symmetrical with eval_ral.sh).
    for tid in "${task_ids[@]}"; do
        t_lib="$backup_root/$tid/lib"
        if [[ -d "$t_lib" && ! -d "$eval_dir/apps/lib" ]]; then
            rsync -a --exclude='node_modules' --exclude='.git' "$t_lib/" "$eval_dir/apps/lib/"
            echo "[Lib] staged per-task lib from $t_lib → apps/lib (first match)"
            break
        fi
    done
fi
has_lib=0
[[ -d "$eval_dir/apps/lib" ]] && has_lib=1

# Pre-assign a unique port per app to avoid vite's sequential fallback under
# concurrent launch (8 apps simultaneously contending for 5173 → last app
# takes ~220s, exceeding DETECTION_TIMEOUT). Pick a random base in 20000-50000
# range; retry if any of the N consecutive ports is already bound.
pick_base_port() {
    local n="$1"
    for _try in $(seq 1 30); do
        local base=$(( 20000 + RANDOM % 30000 ))
        local conflict=0
        for ((i=0; i<n; i++)); do
            if ss -lntH 2>/dev/null | awk '{print $4}' | grep -qE ":$((base + i))\$"; then
                conflict=1
                break
            fi
        done
        if [[ "$conflict" -eq 0 ]]; then
            echo "$base"
            return 0
        fi
    done
    echo "ERROR: could not find $n free consecutive ports after 30 tries" >&2
    return 1
}

if [[ -n "${PORT_BASE:-}" ]]; then
    base_port="$PORT_BASE"
    echo "[Ports] base=$base_port (from PORT_BASE env) range=$base_port..$(( base_port + ${#task_ids[@]} - 1 ))"
else
    base_port=$(pick_base_port "${#task_ids[@]}") || exit 1
    echo "[Ports] base=$base_port range=$base_port..$(( base_port + ${#task_ids[@]} - 1 ))"
fi

# Inject server={port,strictPort:true} into each app's vite.config so vite
# binds directly to its assigned port (no fallback loop). Replaces the
# legacy "strictPort:true → false" patch.
shopt -s nullglob
i=0
for tid in "${task_ids[@]}"; do
    app_port=$(( base_port + i ))
    i=$(( i + 1 ))
    for vite_cfg in "$eval_dir"/apps/"$tid"/vite.config.js \
                    "$eval_dir"/apps/"$tid"/vite.config.ts \
                    "$eval_dir"/apps/"$tid"/vite.config.mjs; do
        [[ -f "$vite_cfg" ]] || continue
        # Strip any existing `server: { ... }` (single-line) entry first, then
        # inject the fresh server block right after `defineConfig({`.
        sed -i -E 's/[[:space:]]*server:[[:space:]]*\{[^}]*\},?//g' "$vite_cfg"
        sed -i -E "0,/defineConfig\\(\\{/s//defineConfig({\\n  server: { port: $app_port, strictPort: true },/" "$vite_cfg"
        echo "patched $tid → port $app_port: $vite_cfg"
    done
done
shopt -u nullglob

# RESUME=1: re-stage previously-evaluated ui_test_results into the eval_dir
# so that WebVoyager's per-task skip-if-exists logic preserves non-cap results;
# then purge sub-task dirs whose WebVoyager run hit the max-iter cap. Only those
# get re-run. log.jsonl is intentionally NOT restored — batches must re-execute
# so apps come back up; the per-task gate handles dedup at finer granularity.
LIMIT_PHRASE='You have reached the maximum number of allowed interactions'
if [[ "$resume" == "1" ]]; then
    prev_ui="${eval_root}/ui_test_results"
    if [[ ! -d "$prev_ui" ]]; then
        echo "[resume] no prior ui_test_results at $prev_ui — nothing to resume"
    else
        echo "[resume] staging $prev_ui → $eval_dir/apps/results"
        mkdir -p "$eval_dir/apps/results"
        rsync -a "$prev_ui/" "$eval_dir/apps/results/"
        purged=0
        for sub in "$eval_dir/apps/results"/*/; do
            [[ -d "$sub" ]] || continue
            mf="$sub/interact_messages.json"
            if [[ ! -f "$mf" ]]; then
                rm -rf "$sub"; purged=$((purged+1)); continue
            fi
            if grep -Fq "$LIMIT_PHRASE" "$mf"; then
                rm -rf "$sub"; purged=$((purged+1))
            fi
        done
        echo "[resume] purged $purged cap-hit/incomplete sub-task dirs — those will be re-run"
    fi
fi

# NOTE: workspace package.json + host-side npm install REMOVED.
# Each task's submission lives at `/home/<id>/` in the container with
# `/home/lib/` as a sibling (RO). Trained code imports the lib via relative
# filesystem paths (e.g. `../../../home/lib/src/index.js`), which resolve via
# the kernel without any npm/workspace plumbing. If $has_lib==1, the lib
# bind-mount is added below; otherwise it's skipped.

# Ensure the sla-base eval image is available locally.
if ! docker image inspect "$docker_image" >/dev/null 2>&1; then
    echo "[Docker] building $docker_image from docker/sla-base/Dockerfile"
    docker build -t "$docker_image" "$project_dir/docker/sla-base"
fi

# Bind-mount the entire eval_dir/apps/ → /home so the in-container layout
# matches training (`/home/<id>/` next to `/home/lib/`). WebGen source and
# el-agent are then layered as nested mounts under /home; docker handles
# the overlay correctly regardless of order. Bonus: ui-test artefacts
# (table.md, results/, log.jsonl) that webgen_ui_test.py writes directly
# under /home now persist back to the host via this single bind.
mount_args=(
    "-v" "$eval_dir/apps:/home:rw"
    "-v" "$project_dir/data/WebGen-Bench:/home/webgen:ro"
    "-v" "$project_dir/el-agent:/home/el-agent:ro"
)

app_id_list=$(IFS=','; echo "${task_ids[*]}")

# Forward only the API/model env vars the in-container python evaluators read.
# (Container has its own PM2_HOME via the entrypoint script.)
env_args=(
    -e EVAL_TEST_FILE="$test_file"
    -e NUM_WORKERS="$num_workers"
    -e EVAL_BATCH_SIZE="$batch_size"
    -e EVAL_APP_LIST="$app_id_list"
)
[[ -n "${PORT_BASE:-}" ]] && env_args+=(-e PORT_BASE="$PORT_BASE")
for v in \
    WEBGEN_APPEARANCE_MODEL WEBGEN_APPEARANCE_REASONING_EFFORT \
    OPENAILIKE_VLM_API_KEY OPENAILIKE_VLM_BASE_URL \
    OPENAILIKE_API_KEY OPENAILIKE_BASE_URL OPENAI_API_KEY OPENAI_BASE_URL \
    ANTHROPIC_VLM_API_KEY ANTHROPIC_API_KEY ANTHROPIC_BASE_URL \
    WEBVOYAGER_API_KEY WEBVOYAGER_API_MODEL WEBVOYAGER_API_BASE \
    WEBVOYAGER_REASONING_EFFORT WEBVOYAGER_MAX_ITER; do
    val="${!v:-}"
    [[ -n "$val" ]] && env_args+=(-e "$v=$val")
done

run_in_container() {
    local eval_mode="$1"
    local cname="${container_name_prefix}-${eval_mode}"
    echo "[Docker] running $docker_image eval_mode=$eval_mode (container=$cname)"
    docker run --rm \
        --name "$cname" \
        --shm-size=4g \
        -e EVAL_MODE="$eval_mode" \
        "${env_args[@]}" \
        "${mount_args[@]}" \
        "$docker_image" \
        webgen_eval_entrypoint.sh
}

if [[ "$mode" == "all" || "$mode" == "appearance" ]]; then
    run_in_container appearance || echo "[error] appearance eval failed"
fi

if [[ "$mode" == "all" || "$mode" == "ui-test" ]]; then
    run_in_container ui-test || echo "[error] ui-test eval failed"
fi

# Collect artifacts → eval_root. Container has written shots/results back to
# the bind-mounted host paths under $eval_dir/apps/<id>/, so the harvest is
# the same as before.
mkdir -p "$eval_root/tasks"
if [[ "$mode" == "all" || "$mode" == "appearance" ]]; then
    uv --project "$project_dir/el-agent" run python "${project_dir}/scripts/eval/collect_appearance_grades.py" \
        --apps-dir "$eval_dir/apps" \
        --app-id-list "$app_id_list" \
        --tag "1" \
        --save-path "$eval_root/appearance_grade.json" || true
fi
if [[ "$mode" == "all" || "$mode" == "ui-test" ]]; then
    [[ -f "$eval_dir/apps/table.md" ]] && cp "$eval_dir/apps/table.md" "$eval_root/ui_test_results.md"
    [[ -d "$eval_dir/apps/results" ]] && {
        rm -rf "$eval_root/ui_test_results"
        cp -r "$eval_dir/apps/results" "$eval_root/ui_test_results"
        bash "${project_dir}/scripts/eval/ui_test_filter.sh" "$eval_root/ui_test_results" || true
    }
fi

# Per-task screenshot cache → backup_root for re-runs (mirrors legacy).
for shots_dir in "$eval_dir"/apps/*/shots; do
    [[ -d "$shots_dir" ]] || continue
    tid=$(basename "$(dirname "$shots_dir")")
    backup_shots="$backup_root/$tid/submission/shots"
    mkdir -p "$backup_shots"
    cp -r "$shots_dir"/* "$backup_shots"/ 2>/dev/null || true
done

if [[ "$keep_eval_dir" != "1" ]]; then
    # Container ran as root and left root-owned artefacts under $eval_dir.
    # chown back to the host user before rm so cleanup doesn't fail with
    # "Permission denied" on the host side.
    docker run --rm \
        -v "$eval_dir:/target" \
        "$docker_image" \
        chown -R "$(id -u):$(id -g)" /target >/dev/null 2>&1 || true
    rm -rf "$eval_dir"
    echo "[cleanup] removed eval_dir"
else
    echo "[cleanup] KEEP_EVAL_DIR=1 — retained $eval_dir"
fi

echo
echo "================================================="
echo "Done. Results: $eval_root"
echo "================================================="
