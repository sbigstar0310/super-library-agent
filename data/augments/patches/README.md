# Patches for the cloned benchmark harnesses

Neither harness runs as shipped. Apply these after cloning, at the pinned commits: the diffs
will not apply to a later tree, and the pinned commits are what the paper's numbers come from.
The repository README has the full clone sequence.

```bash
git -C data/WebGen-Bench   checkout 1e69d30 && git -C data/WebGen-Bench   apply ../augments/patches/webgen-bench.diff
git -C data/frontier-evals checkout 51052ce && git -C data/frontier-evals apply ../augments/patches/frontier-evals.diff
```

We ship diffs rather than copies because WebGen-Bench has no license file, so we have no right to
redistribute its source.

## webgen-bench.diff

**Credentials are literal placeholders upstream.** `vlm_eval.py` builds its clients with
`api_key="API_KEY"`, `webvoyager/run.py` with `base_url="http://PI_ADDRESS:PORT/v1"`. Now read
from the environment.

**The UI driver predates reasoning models.** It sends `max_tokens`, which gpt-5 and the o-series
reject. Now switches to `max_completion_tokens` for those, with 4× the budget since hidden
reasoning tokens share the allowance.

**`start_service.py` (both copies) was not concurrency-safe.** `pm2 delete all` killed other
evaluations on the same host and wiped all of `~/.pm2/logs`, crashing when it did not exist. Now
scoped to this run's apps and logs, honouring `PM2_HOME`. `DETECTION_TIMEOUT` goes from 60s to
240s/180s: a cold vite start takes 180 to 240s here, and an undetected port scores the app dead,
so at 60s the harness returns low numbers rather than failing.

## frontier-evals.diff

**`.lfsconfig` excluded the paper data from LFS fetch.** A clone succeeded and left pointer stubs
where the papers should be, which the pipeline reads as papers.

**Adds the dependency whitelist.** PaperBench agents get numerical and deep-learning primitives
but no domain framework. One that already implements a paper's method end-to-end would
remove the repeated implementation the shared library exists to capture. Stated in appendix A.2.1
of the paper; keep `agent_requirements.txt` in sync with `el-agent/src/utils/whitelist.py`.

Deliberately excluded: our local switch of miniconda to aarch64, which suits neither an x86 host
nor the machine the experiments ran on.
