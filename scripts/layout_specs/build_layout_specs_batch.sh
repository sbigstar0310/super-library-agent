#!/usr/bin/env bash
# Batch wrapper around build_layout_spec.sh.
#
# Iterates a comma-separated TASK_IDS list and runs the per-task
# orchestrator sequentially (host has only one vite port pool, so the
# postproc pipeline isn't parallelized here). After each task, copies
# the produced layout_spec.md into data/augments/webgen/layout_specs/.
#
# Env:
#   BACKUP_TAG     required
#   TASK_IDS       required (comma-separated, e.g. "000027,000051")
#   PHASE          default: baseline
#   ROUND          default: 0
#   TASK_FILE      default: data/WebGen-Bench/data/test.jsonl
#   DESCRIBE_MODE  high (default) | medium | low — forwarded to per-task
#                  build_layout_spec.sh, which forwards to describe_layout.py.
#   AUGMENTS_DIR   default depends on DESCRIBE_MODE:
#                    high   → data/augments/webgen/layout_specs
#                    medium → data/augments/webgen/layout_specs_medium
#                    low    → data/augments/webgen/layout_specs_low
#   PORT_BASE      first vite port (default: 5273); each task gets PORT_BASE+i
#   EXTRA_FLAGS    forwarded to build_layout_spec.sh (e.g. REGEN=1)
#   RECIPE_ONLY=1  Forwarded to build_layout_spec.sh. In this mode, batch
#                  only verifies pages.json exists (no promote to AUGMENTS_DIR).
#   CRAWL_ONLY=1   Stop after crawl (stage 5). Forwards SKIP_DESCRIBE=1.
#                  Batch reports per-task crawl coverage: # screens in
#                  pages.json vs # *.png in pages/. Names of missing
#                  screens are listed. Used for inspecting crawler
#                  failures before committing to describe.
#
# Usage:
#   BACKUP_TAG=webgen-ref-c13-t1 \
#   TASK_IDS=000027,000051,000052,000077,000090,000091,000092 \
#   bash scripts/layout_specs/build_layout_specs_batch.sh

set -euo pipefail

BACKUP_TAG=${BACKUP_TAG:?missing}
TASK_IDS=${TASK_IDS:?missing}
PHASE=${PHASE:-baseline}
ROUND=${ROUND:-0}
TASK_FILE=${TASK_FILE:-data/WebGen-Bench/data/test.jsonl}
DESCRIBE_MODE=${DESCRIBE_MODE:-high}
case "$DESCRIBE_MODE" in
  high)   _DEFAULT_AUGMENTS=data/augments/webgen/layout_specs ;;
  medium) _DEFAULT_AUGMENTS=data/augments/webgen/layout_specs_medium ;;
  low)    _DEFAULT_AUGMENTS=data/augments/webgen/layout_specs_low ;;
  *) echo "error: DESCRIBE_MODE must be one of: high, medium, low (got: $DESCRIBE_MODE)"; exit 2 ;;
esac
AUGMENTS_DIR=${AUGMENTS_DIR:-$_DEFAULT_AUGMENTS}
PORT_BASE=${PORT_BASE:-5273}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)

# Make TASK_FILE / AUGMENTS_DIR absolute if relative.
to_abs() {
  case "$1" in
    /*) echo "$1" ;;
    *)  echo "${PROJECT_DIR}/$1" ;;
  esac
}
TASK_FILE=$(to_abs "$TASK_FILE")
AUGMENTS_DIR=$(to_abs "$AUGMENTS_DIR")
mkdir -p "$AUGMENTS_DIR"

IFS=',' read -ra IDS <<< "$TASK_IDS"

echo "════════════════════════════════════════════════════════════"
echo "  BATCH layout-spec build"
echo "  tag        : $BACKUP_TAG"
echo "  phase/round: $PHASE / round_$ROUND"
echo "  tasks      : ${IDS[@]}"
echo "  describe   : $DESCRIBE_MODE"
echo "  augments   : $AUGMENTS_DIR"
echo "════════════════════════════════════════════════════════════"

i=0
ok=0
fail=()
for TID in "${IDS[@]}"; do
  PORT=$((PORT_BASE + i))
  i=$((i + 1))
  SUB="${PROJECT_DIR}/backups/webgen/${BACKUP_TAG}/final/round_${ROUND}/${PHASE}/tasks/${TID}/submission"
  if [[ ! -d "$SUB" ]]; then
    echo "[$TID] ✗ submission not found: $SUB"
    fail+=("$TID")
    continue
  fi
  echo
  echo "==================== task $TID (port=$PORT) ===================="
  # CRAWL_ONLY forwards SKIP_DESCRIBE=1 to skip stage 6.
  EFFECTIVE_SKIP_DESCRIBE="${SKIP_DESCRIBE:-0}"
  if [[ "${CRAWL_ONLY:-0}" == "1" ]]; then
    EFFECTIVE_SKIP_DESCRIBE=1
  fi
  if SUBMISSION_SRC="$SUB" \
     TASK_FILE="$TASK_FILE" \
     TASK_ID="$TID" \
     BACKUP_TAG="$BACKUP_TAG" \
     PHASE="$PHASE" \
     ROUND="$ROUND" \
     PORT="$PORT" \
     DESCRIBE_MODE="$DESCRIBE_MODE" \
     RECIPE_ONLY="${RECIPE_ONLY:-0}" \
     SKIP_DESCRIBE="$EFFECTIVE_SKIP_DESCRIBE" \
     bash "$SCRIPT_DIR/build_layout_spec.sh"; then
    POSTPROC_DIR="${PROJECT_DIR}/backups/webgen/${BACKUP_TAG}/postproc/round_${ROUND}/${PHASE}/tasks/${TID}"
    if [[ "${RECIPE_ONLY:-0}" == "1" ]]; then
      if [[ -f "${POSTPROC_DIR}/pages.json" ]]; then
        echo "[$TID] ✓ pages.json ready (RECIPE_ONLY)"
        ok=$((ok + 1))
      else
        echo "[$TID] ✗ pages.json missing at ${POSTPROC_DIR}/pages.json"
        fail+=("$TID")
      fi
    elif [[ "${CRAWL_ONLY:-0}" == "1" ]]; then
      RECIPE_FILE="${POSTPROC_DIR}/pages.json"
      PAGES_DIR="${POSTPROC_DIR}/pages"
      if [[ ! -f "$RECIPE_FILE" ]]; then
        echo "[$TID] ✗ pages.json missing — cannot verify crawl"
        fail+=("$TID")
      else
        REPORT=$(python3 - "$RECIPE_FILE" "$PAGES_DIR" <<'PY'
import sys, json, os
recipe_file, pages_dir = sys.argv[1], sys.argv[2]
data = json.load(open(recipe_file))
pages = data.get("pages") or data.get("recipes") or []
if isinstance(pages, dict):
    expected = list(pages.keys())
else:
    expected = [(x.get("screen") or x.get("name") or "?") for x in pages]
captured = set()
if os.path.isdir(pages_dir):
    for f in os.listdir(pages_dir):
        if f.endswith(".png"):
            captured.add(f[:-4])
missing = [s for s in expected if s not in captured]
print(f"{len(expected)}\t{len(expected) - len(missing)}\t{','.join(missing)}")
PY
        )
        EXPECTED=$(cut -f1 <<< "$REPORT")
        GOT=$(cut -f2 <<< "$REPORT")
        MISSING_LIST=$(cut -f3 <<< "$REPORT")
        if [[ "$EXPECTED" == "$GOT" ]]; then
          echo "[$TID] ✓ crawl ${GOT}/${EXPECTED}"
          ok=$((ok + 1))
        else
          echo "[$TID] ✗ crawl ${GOT}/${EXPECTED} — missing: ${MISSING_LIST}"
          fail+=("$TID")
        fi
      fi
    else
      SPEC="${POSTPROC_DIR}/layout_spec.md"
      if [[ -f "$SPEC" ]]; then
        cp "$SPEC" "$AUGMENTS_DIR/${TID}.md"
        echo "[$TID] ✓ promoted → $AUGMENTS_DIR/${TID}.md"
        ok=$((ok + 1))
      else
        echo "[$TID] ✗ layout_spec.md missing at $SPEC"
        fail+=("$TID")
      fi
    fi
  else
    echo "[$TID] ✗ build_layout_spec.sh failed"
    fail+=("$TID")
  fi
done

echo
echo "════════════════════════════════════════════════════════════"
echo "  DONE  ok=$ok / total=${#IDS[@]}"
[[ ${#fail[@]} -gt 0 ]] && echo "  failed: ${fail[*]}"
echo "════════════════════════════════════════════════════════════"
[[ ${#fail[@]} -eq 0 ]]
