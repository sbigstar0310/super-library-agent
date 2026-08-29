#!/usr/bin/env bash
# WebGen maintenance-patch launcher (rebuttal experiment).
#
#   SOURCE_TAG=sla-ours-c13-t1 SUITE=c13 PROTOCOL=b bash scripts/run/run_webgen_maintenance.sh
#
# PROTOCOL=b  suite-session (all 5 methods)
# PROTOCOL=c  per-app sessions (baseline source tags only)
#
# Design: paper appendix B (maintenance experiment).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/../.." && pwd)"

SOURCE_TAG="${SOURCE_TAG:?SOURCE_TAG required (e.g. sla-ours-c13-t1)}"
SUITE="${SUITE:?SUITE required (c2|c5|c13)}"
PROTOCOL="${PROTOCOL:?PROTOCOL required (b|c)}"
PROVIDER="${PROVIDER:-openrouter}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
DOCKER_IMAGE="${DOCKER_IMAGE:-sla-base}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TIMEOUT="${TIMEOUT:-180}"

args=(
  --source-tag "$SOURCE_TAG"
  --suite "$SUITE"
  --protocol "$PROTOCOL"
  --provider "$PROVIDER"
  --model "$MODEL"
  --docker-image "$DOCKER_IMAGE"
  --temperature "$TEMPERATURE"
  --timeout "$TIMEOUT"
)
[[ "${SKIP_BUILD_SMOKE:-0}" == "1" ]] && args+=(--skip-build-smoke)
[[ "${DRY_RUN_WORKSPACE:-0}" == "1" ]] && args+=(--dry-run-workspace)

cd "$project_dir/el-agent/src"
exec uv --project "$project_dir/el-agent" run python -m run.webgen_maintenance_run "${args[@]}" "$@"
