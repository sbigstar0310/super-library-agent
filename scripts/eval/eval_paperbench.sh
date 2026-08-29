#!/usr/bin/env bash
# Run paperbench LLM grading for a finalized backup tag (unified layout only).
#
# MDL is NOT run here — invoke `scripts/metrics/get_mdl.py --task paperbench ...`
# separately. This wrapper handles grading only.
#
# Usage:
#   PHASE=<phase> [ROUND=<n>] bash scripts/eval/eval_paperbench.sh <backup-tag>
#
# Layout (only mode supported):
#   backups/paperbench/<tag>/final/round_<N>/<phase>/tasks/<pid>/
#   eval_results: backups/paperbench/<tag>/eval_results/round_<N>/<phase>/<pid>/
#
# Env overrides:
#   PHASE          — REQUIRED. baseline|coding|apply|extract
#   ROUND          — round number. Empty → latest round with PHASE on disk
#                    (resolved via backup_layout.latest_round_in_backup)
#   PAPER_IDS      — comma-separated; empty → auto-derive from
#                    final_dir/round_<N>/<phase>/tasks/*/submission
#   GRADE_MODEL    — default: deepseek/deepseek-v4-flash (paperbench_eval.py default, via OpenRouter)
#   GRADE_BASE_URL — default: https://openrouter.ai/api/v1
#   API_KEY_ENV    — default: OPENROUTER_API_KEY
#
# Lib integration: paperbench_eval.py auto-detects the lib/ sibling
# (per-task in coding/apply, phase-level in extract) and temporarily
# copies it into submission/lib/ before grading (cleaned up afterwards).
#
# One-time setup:
#   cd data/frontier-evals/project/paperbench && uv sync

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
paperbench_dir="$project_dir/data/frontier-evals/project/paperbench"
paperbench_python="$paperbench_dir/.venv/bin/python"

backup_tag="${1:-}"
[[ -z "$backup_tag" ]] && { echo "usage: PHASE=<phase> $0 <backup-tag>" >&2; exit 2; }

paper_ids="${PAPER_IDS:-}"
phase="${PHASE:-}"
round_arg="${ROUND:-}"

[[ -z "$phase" ]] && { echo "PHASE env var is required (baseline|coding|apply|extract)" >&2; exit 2; }

# Resolve round.
if [[ -n "$round_arg" ]]; then
    round_num="$round_arg"
else
    round_num=$(uv --project "$project_dir/el-agent" run python -c "
import sys
sys.path.insert(0, '$project_dir/el-agent/src')
from run import backup_layout
n = backup_layout.latest_round_in_backup('$project_dir', 'paperbench', '$backup_tag', '$phase')
print('' if n is None else n)
")
    if [[ -z "$round_num" ]]; then
        echo "No round_N found for phase=$phase under backups/paperbench/$backup_tag/final/" >&2
        exit 1
    fi
fi

final_dir="$project_dir/backups/paperbench/$backup_tag/final"
tasks_root="$final_dir/round_$round_num/$phase/tasks"
eval_root="$project_dir/backups/paperbench/$backup_tag/eval_results/round_$round_num/$phase"

# Resolve paper list (auto-derive from tasks_root if PAPER_IDS empty).
if [[ -n "$paper_ids" ]]; then
    IFS=',' read -ra pid_arr <<<"$paper_ids"
else
    [[ -d "$tasks_root" ]] || { echo "tasks_root not found: $tasks_root" >&2; exit 1; }
    pid_arr=()
    for d in "$tasks_root"/*/; do
        [[ -d "$d/submission" ]] && pid_arr+=("$(basename "$d")")
    done
fi
[[ ${#pid_arr[@]} -gt 0 ]] || { echo "No tasks to evaluate under $tasks_root" >&2; exit 1; }

echo "[Layout] round=$round_num  phase=$phase"
echo "[Config] final_dir=$final_dir  eval_root=$eval_root"
echo "[Config] papers=${pid_arr[*]}"

if [[ -z "${OPENROUTER_API_KEY:-}" && -f "$project_dir/.env" ]]; then
    set -a; source "$project_dir/.env"; set +a
fi
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "OPENROUTER_API_KEY not set (and missing in .env)" >&2; exit 1; }

model_args=()
if [[ -n "${GRADE_MODEL:-}" ]]; then
    model_args=(--model "$GRADE_MODEL")
fi

echo "==> Grading  tag=$backup_tag  model=${GRADE_MODEL:-(default deepseek/deepseek-v4-flash)}"

for pid in "${pid_arr[@]}"; do
    sub="$tasks_root/$pid/submission"
    [[ -d "$sub" ]] || { echo "[skip] $pid: no submission at $sub" >&2; continue; }
    echo "--- Grading $pid ---"
    "$paperbench_python" "$project_dir/el-agent/src/eval/paperbench_eval.py" \
        --backup-tag "$backup_tag" --paper-id "$pid" \
        --round "$round_num" --phase "$phase" \
        "${model_args[@]}"
done

echo
echo "==> Done. Results: $eval_root"
ls -1 "$eval_root" 2>/dev/null || true
