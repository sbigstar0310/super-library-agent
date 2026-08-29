#!/usr/bin/env bash
# Build a per-task `layout_spec.md` from an existing reference submission.
#
# Pipeline (all in one orchestrator):
#   1. Stage submission (+ optional lib) at a fresh /home-shaped layout
#      under $STAGE_ROOT so the original backup is untouched.
#   2. Boot vite dev (background, log to $OUT_ROOT/dev.log).
#   3. generate_pages_recipe.py  → $OUT_ROOT/pages.json     (cc-sandbox)
#   4. verify_pages_recipe.py    → informational            (host)
#   5. crawl_pages.py            → $OUT_ROOT/pages/*.png    (host, best-effort)
#   6. describe_layout.py        → $OUT_ROOT/layout_spec.md (cc-sandbox)
#   7. Tear down vite (port-based kill).
#
# Inputs (env):
#   SUBMISSION_SRC  Path to an EXISTING submission/ dir (read-only).
#   LIB_SRC         Optional. Path to a lib/ dir for sla submissions.
#   TASK_FILE       WebGen-Bench JSONL.
#   TASK_ID         e.g. 000053
#   BACKUP_TAG      Used to derive OUT_ROOT under backups/webgen/<tag>/postproc/...
#   PHASE           default: baseline
#   ROUND           default: 0
#   OUT_ROOT        Override derived path; absolute or relative.
#   STAGE_ROOT      Local-disk stage (default: /tmp/layout-stage-<tag>-<id>)
#   PORT            Dev server port (default: 5273)
#   REGEN=1         Force regeneration even if pages.json exists.
#   REDESCRIBE=1    Force describe even if layout_spec.md exists.
#   RECIPE_ONLY=1   Stop after stage 4 (verify_pages_recipe). Skips
#                   crawl + describe. Useful for inspecting pages.json
#                   across many tasks before committing to full pipeline.
#   DESCRIBE_MODE   high (default) | medium | low
#                   Forwarded to describe_layout.py as --medium-detail /
#                   --low-detail. LOW requires pages.json (auto-discovered
#                   from $OUT_ROOT/pages.json by describe_layout.py).
#
# Usage:
#   SUBMISSION_SRC=.../tasks/000053/submission \
#   TASK_FILE=.../data/WebGen-Bench/data/test.jsonl \
#   TASK_ID=000053 BACKUP_TAG=webgen-ref-c13-t1 \
#   bash scripts/layout_specs/build_layout_spec.sh
#
# Once accepted, promote the resulting spec into the augments tree:
#   cp $OUT_ROOT/layout_spec.md data/augments/webgen/layout_specs/<id>.md

set -euo pipefail

SUBMISSION_SRC=${SUBMISSION_SRC:?missing}
# LIB_SRC is optional — baseline submissions have no shared library.
# When set, the lib is copied alongside submission/ so relative imports
# like `'../../../home/lib/...'` resolve under the /tmp stage layout.
LIB_SRC=${LIB_SRC:-}
TASK_FILE=${TASK_FILE:?missing}
TASK_ID=${TASK_ID:?missing}
PORT=${PORT:-5273}

# Backup-tree layout: artifacts land alongside submission + eval_results
# of the same tag, mirroring the cc_exp / eval_webgen path convention:
#   backups/webgen/<tag>/postproc/round_<N>/<phase>/tasks/<id>/
# BACKUP_TAG is required when OUT_ROOT is not explicitly set.
BACKUP_TAG=${BACKUP_TAG:-}
PHASE=${PHASE:-baseline}
ROUND=${ROUND:-0}

DESCRIBE_MODE=${DESCRIBE_MODE:-high}
case "$DESCRIBE_MODE" in
  high|medium|low) ;;
  *) echo "error: DESCRIBE_MODE must be one of: high, medium, low (got: $DESCRIBE_MODE)"; exit 2 ;;
esac
DESCRIBE_FLAGS=()
case "$DESCRIBE_MODE" in
  medium) DESCRIBE_FLAGS+=(--medium-detail) ;;
  low)    DESCRIBE_FLAGS+=(--low-detail) ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)

if [[ -z "${OUT_ROOT:-}" ]]; then
  if [[ -z "$BACKUP_TAG" ]]; then
    echo "error: set OUT_ROOT explicitly, or set BACKUP_TAG so we can derive"
    echo "       backups/webgen/<BACKUP_TAG>/postproc/round_<ROUND>/<PHASE>/tasks/<TASK_ID>/"
    exit 2
  fi
  OUT_ROOT="${PROJECT_DIR}/backups/webgen/${BACKUP_TAG}/postproc/round_${ROUND}/${PHASE}/tasks/${TASK_ID}"
fi

# Stage (npm-installed submission + lib) lives on local disk, OUTSIDE the
# NFS mount, so vite's mmap'd esbuild binary doesn't leave `.nfs*`
# sillyrename leftovers that block subsequent rm -rf.
STAGE_ROOT=${STAGE_ROOT:-/tmp/layout-stage-${BACKUP_TAG:-norun}-${TASK_ID}}

echo "════════════════════════════════════════════════════════════"
echo "  SMOKE postproc"
echo "  task        : $TASK_ID"
echo "  backup_tag  : ${BACKUP_TAG:-(unset — using explicit OUT_ROOT)}"
echo "  round/phase : round_${ROUND}/${PHASE}"
echo "  submission  : $SUBMISSION_SRC"
echo "  lib         : $LIB_SRC"
echo "  out_root    : $OUT_ROOT"
echo "  stage_root  : $STAGE_ROOT"
echo "  vite port   : $PORT"
echo "  describe    : $DESCRIBE_MODE"
echo "════════════════════════════════════════════════════════════"

mkdir -p "$OUT_ROOT"

# 1. Stage /home-shaped layout on local disk. Preserve original via copy,
#    not symlink — vite's module resolution follows real paths.
STAGE="$STAGE_ROOT/home"
# `.nfs*` sillyrename files may resist rm if a previous process held an
# mmap — ignore those and proceed; the stage dir is on /tmp anyway, so
# residual junk is at worst harmless.
rm -rf "$STAGE" 2>/dev/null || true
find "$STAGE_ROOT" -name '.nfs*' -delete 2>/dev/null || true
rm -rf "$STAGE_ROOT" 2>/dev/null || true
mkdir -p "$STAGE"
echo "[stage] copying submission into $STAGE ..."
cp -a "$SUBMISSION_SRC" "$STAGE/submission"
if [[ -n "$LIB_SRC" ]]; then
  echo "[stage] copying lib into $STAGE ..."
  cp -a "$LIB_SRC" "$STAGE/lib"
else
  echo "[stage] LIB_SRC not set — skipping lib copy (baseline submission)"
fi

# Re-install if node_modules absent or stale (e.g. .bin symlinks broken).
SUBMISSION_DIR="$STAGE/submission"
if [[ ! -x "$SUBMISSION_DIR/node_modules/.bin/vite" ]]; then
  echo "[stage] running npm install in $SUBMISSION_DIR ..."
  ( cd "$SUBMISSION_DIR" && npm install --silent ) || {
    echo "[stage] npm install failed"; exit 1; }
fi

# 2. Boot vite dev in background.
#    Identify by PORT, not by PID — `setsid` + `$!` is unreliable
#    because setsid forks and exits, leaving the actual vite/node
#    process under a PID we never captured. fuser -k on the port +
#    a pkill fallback catches everything that's actually serving.
DEV_LOG="$OUT_ROOT/dev.log"
echo "[dev] launching vite at port $PORT ..."
( cd "$SUBMISSION_DIR" && exec npx vite --port "$PORT" --strictPort ) \
    >"$DEV_LOG" 2>&1 &
DEV_PID=$!
trap '
  echo "[dev] killing vite on port '$PORT' (subshell pid=$DEV_PID)"
  kill $DEV_PID 2>/dev/null || true
  fuser -k '$PORT'/tcp 2>/dev/null || true
  sleep 1
  pkill -KILL -f "vite.*--port '$PORT'" 2>/dev/null || true
' EXIT

URL="http://localhost:$PORT"
# Wait for ready (vite cold start can be slow).
echo -n "[dev] waiting for $URL "
for i in $(seq 1 120); do
  if curl -fs "$URL" >/dev/null 2>&1; then
    echo " READY"; break
  fi
  echo -n "."
  sleep 1
  if [[ $i -eq 120 ]]; then
    echo " TIMEOUT — see $DEV_LOG"
    tail -40 "$DEV_LOG" || true
    exit 1
  fi
done

# 3. generate_pages_recipe (cc-sandbox)
RECIPE="$OUT_ROOT/pages.json"
echo
if [[ -f "$RECIPE" && "${REGEN:-0}" != "1" ]]; then
  echo "[3/4] generate_pages_recipe: SKIP (already at $RECIPE; REGEN=1 to force)"
else
  echo "[3/4] generate_pages_recipe ..."
  python "$SCRIPT_DIR/generate_pages_recipe.py" \
    --submission "$SUBMISSION_DIR" \
    --task-file  "$TASK_FILE" \
    --task-id    "$TASK_ID" \
    --out        "$RECIPE" \
    --log-dir    "$OUT_ROOT/.agent_logs"
fi

# 4. verify_pages_recipe (host)
# Informational only — Opus reads source directly, so static-extraction
# mismatches don't necessarily mean the recipe is wrong. The crawler
# (next step) is the real validator. We DO want verify's diagnostic
# output, just not its exit code blocking the pipeline.
echo
echo "[4/4] verify_pages_recipe (informational) ..."
python "$SCRIPT_DIR/verify_pages_recipe.py" \
  --submission "$SUBMISSION_DIR" \
  --recipe     "$RECIPE" || true

if [[ "${RECIPE_ONLY:-0}" == "1" ]]; then
  echo
  echo "════════════════════════════════════════════════════════════"
  echo "  RECIPE_ONLY=1 — stopping after verify."
  echo "  recipe : $RECIPE"
  echo "════════════════════════════════════════════════════════════"
  exit 0
fi

# 5. crawl_pages (host)
# Best-effort: per-page failures (selector miss, dynamic content
# invisible, etc.) are surfaced in the log but do NOT block describe.
# Layout describer will simply skip missing screenshots.
echo
echo "[5/4] crawl_pages (best-effort) ..."
SHOTS="$OUT_ROOT/pages"
python "$SCRIPT_DIR/crawl_pages.py" \
  --recipe "$RECIPE" \
  --url    "$URL" \
  --out    "$SHOTS" || echo "[crawl] some pages failed — continuing with whatever was captured"

# 6. describe_layout (cc-sandbox)
SPEC="$OUT_ROOT/layout_spec.md"
echo
if [[ "${SKIP_DESCRIBE:-0}" == "1" ]]; then
  echo "[6/4] describe_layout: SKIP (SKIP_DESCRIBE=1 — crawl-only mode)"
elif [[ -f "$SPEC" && "${REDESCRIBE:-0}" != "1" ]]; then
  echo "[6/4] describe_layout: SKIP (already at $SPEC; REDESCRIBE=1 to force)"
else
  echo "[6/4] describe_layout (mode=$DESCRIBE_MODE) ..."
  python "$SCRIPT_DIR/describe_layout.py" \
    --pages     "$SHOTS" \
    --out       "$SPEC" \
    --task-file "$TASK_FILE" \
    --task-id   "$TASK_ID" \
    --log-dir   "$OUT_ROOT/.agent_logs" \
    "${DESCRIBE_FLAGS[@]}"
fi

echo
echo "════════════════════════════════════════════════════════════"
echo "  DONE"
echo "  recipe       : $RECIPE"
echo "  screenshots  : $SHOTS"
echo "  layout spec  : $SPEC"
echo "  agent logs   : $OUT_ROOT/.agent_logs/"
echo "════════════════════════════════════════════════════════════"
