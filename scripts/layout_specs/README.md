# Layout-spec generation

Builds the per-task specs in `data/augments/webgen/layout_specs/`, which fix the visible design of a
WebGen task across arms. `data/augments/webgen/layout_specs/README.md` says why they exist. Every
task in the reported suites already has one, so you need this pipeline only to add a task or redo a
spec.

A spec is derived from a reference implementation: stage it, boot vite, screenshot every page, then
have a vision model write the page-by-page description. `build_layout_spec.sh` runs that chain and
documents each environment variable it reads in its header.

```bash
docker build -t cc-sandbox docker/cc-sandbox

SUBMISSION_SRC=backups/webgen/<ref-tag>/final/round_1/coding/tasks/000053/submission \
TASK_FILE=data/WebGen-Bench/data/test.jsonl \
TASK_ID=000053 BACKUP_TAG=<ref-tag> PHASE=coding ROUND=1 \
bash scripts/layout_specs/build_layout_spec.sh
```

`pages.json`, `pages/*.png`, `layout_spec.md` and `dev.log` land under
`backups/webgen/<ref-tag>/postproc/round_1/coding/tasks/000053/`. Read the spec, then install it:

```bash
cp backups/webgen/<ref-tag>/postproc/round_1/coding/tasks/000053/layout_spec.md \
   data/augments/webgen/layout_specs/000053.md
```

`build_layout_specs_batch.sh` takes `TASK_IDS=000027,000051` and does that copy itself. It is
sequential because each task boots a real dev server on one port.

This is the one part of the repo that runs on the host: `npm install` and `npx vite` need Node, and
`crawl_pages.py` needs a Python with `playwright` installed. Only the two LLM stages are
containerized. They call the Claude Code CLI inside `cc-sandbox` with the host's
`~/.claude/.credentials.json` bind-mounted read-only, so `claude` has to have been authenticated
once on the host. Running them there is what keeps repo-level agent instructions (`CLAUDE.md`) out
of the prompt and pins the CLI version. The default model is `claude-opus-4-7`, in `_sandbox.py`.

The reference implementation is not produced here. It came from a separate generation run with a
stronger model, screened on Acc and Appearance first so the spec describes an app that works. That
driver is not part of this release.

The orchestrator kills vite by port on exit. To clear one left by a crash:

```bash
fuser -k 5273/tcp; pkill -f "vite.*--port 5273"
```
