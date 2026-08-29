#!/usr/bin/env bash
# webgen_screenshot_entrypoint.sh
#
# In-container per-task driver for the webgen-screenshot skill.
#
# The host orchestrator that drives this (an internal helper, not part of
# this distribution) bind-mounts:
#   $eval_dir/apps/<id>           → /work   (rw — staged submission + optional lib)
#   $eval_root/tasks/<id>         → /out    (rw — must already contain
#                                              screenshot_script.py; receives
#                                              screenshots/*.png + dev.log)
#
# Per task this script:
#   1. cd /work, npm install if node_modules/.bin/vite is missing/broken
#   2. launch vite dev on $PORT, log to /out/dev.log
#   3. poll http://127.0.0.1:$PORT until ready (max 180s)
#   4. exec /out/screenshot_script.py against the URL
#   5. teardown vite (trap on EXIT — fires even when --rm tears the container)
#
# Exit code:
#   0  screenshot script completed (some pages may have failed — the script
#      decides; this entrypoint just propagates that)
#   1  vite never came up / npm install failed
#   2  /out/screenshot_script.py missing

set -uo pipefail   # NOT -e: we want to surface specific exit codes per stage

: "${PORT:?missing PORT}"
: "${TASK_ID:?missing TASK_ID}"

SCRIPT_PATH="/out/screenshot_script.py"
SHOTS_DIR="/out/screenshots"
DEV_LOG="/out/dev.log"

mkdir -p "$SHOTS_DIR"

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "[screenshot] FATAL: $SCRIPT_PATH missing — skill must write it before docker run" >&2
    exit 2
fi

# Submission lives at /work/submission/ (lib, if present, at /work/lib/) —
# layout mirrors the backup so apps' `../../lib/src/index.js` imports
# resolve as-is.
# Some apps hard-code absolute imports `/home/lib/...` (matches the eval
# container's lib bind-mount). Mirror that path here so those imports resolve.
if [[ -d /work/lib && ! -e /home/lib ]]; then
    ln -s /work/lib /home/lib
fi

cd /work/submission

# .bin/.vite is often a broken symlink in backups (npm audit fix --force
# replaces .bin/ symlinks with regular file copies; copytree preserves the
# state but the binary stays broken). Re-run npm install to regenerate.
if [[ ! -x node_modules/.bin/vite ]]; then
    echo "[screenshot] node_modules/.bin/vite missing — running npm install ..."
    if ! npm install --silent 2>&1 | tee -a "$DEV_LOG"; then
        echo "[screenshot] npm install FAILED — see $DEV_LOG" >&2
        exit 1
    fi
fi

# Background vite. setsid so the trap KILL targets the whole process group
# (npx forks node and exits — `kill $PID` would orphan the actual vite).
echo "[screenshot] launching vite at port $PORT (task=$TASK_ID) ..."
setsid bash -c "exec npx vite --port $PORT --strictPort --host 127.0.0.1" \
    >>"$DEV_LOG" 2>&1 &
VITE_PGID=$!

cleanup() {
    echo "[screenshot] tearing down vite (pgid=$VITE_PGID) ..."
    # Kill the whole process group so node + esbuild children are reaped.
    kill -- "-$VITE_PGID" 2>/dev/null || true
    sleep 0.5
    # Belt-and-braces for any straggler still on the port.
    fuser -k "$PORT/tcp" 2>/dev/null || true
}
trap cleanup EXIT

# Poll for ready. Cold start (npm install + vite warm-up) can take ~3min
# on first run; subsequent runs reuse node_modules and are ~10s.
URL="http://127.0.0.1:$PORT"
echo -n "[screenshot] waiting for $URL "
for i in $(seq 1 180); do
    if curl -fs "$URL" >/dev/null 2>&1; then
        echo "READY"
        break
    fi
    echo -n "."
    sleep 1
    if [[ $i -eq 180 ]]; then
        echo " TIMEOUT — last 40 lines of dev.log:"
        tail -40 "$DEV_LOG" || true
        exit 1
    fi
done

echo "[screenshot] running $SCRIPT_PATH --url $URL --out $SHOTS_DIR"
python "$SCRIPT_PATH" --url "$URL" --out "$SHOTS_DIR"
script_rc=$?

echo "[screenshot] script exit=$script_rc shots=$(ls "$SHOTS_DIR"/*.png 2>/dev/null | wc -l)"
exit "$script_rc"
