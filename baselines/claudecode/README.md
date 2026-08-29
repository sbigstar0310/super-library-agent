# Claude Code scaffold, WebGen

A recent agent-based SE system as a comparison arm. The Claude Code CLI is the scaffold; the model
behind it is deepseek-v4-flash, the backbone every other arm uses, so no Claude model produces any
of the numbers. Zero-shot only: one app per task, no shared library.

`claude` runs headless inside `cc-sandbox` and its Anthropic API calls go to a local LiteLLM proxy
that rewrites them onto deepseek-v4-flash at OpenRouter. `litellm_config.yaml` pins the fairness
parameters there to match `el-agent/src/mswe_agents/_factory.py`, and the run reuses the WebGen
coding prompt of Zero-Shot and OpenHands. Only the scaffold differs.

Setup needs `OPENROUTER_API_KEY` in the repo `.env` plus:

```bash
docker build -t cc-sandbox docker/cc-sandbox
cd baselines/claudecode && uv venv .venv && uv pip install 'litellm[proxy]'
```

The launcher starts the proxy itself when it is not already up.

```bash
TAG=cc-deepseek-c13-t1 CLUSTER_ID=13 bash scripts/run/run_webgen_claudecode.sh
```

The reported campaign is nine such runs, clusters 2, 5 and 13 by trials 1 to 3, looped from a driver
under `scripts/local/` that is not released. Submissions land in
`backups/webgen/<TAG>/final/round_1/coding/tasks/<id>/submission/`, the layout every other arm
writes, so evaluation and the metric scripts read them with no special casing.
