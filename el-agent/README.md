# el-agent

The Python package behind SLA. `uv sync` here is what the root README tells you
to run first. Anything under `scripts/metrics/` imports from `src/`, so launch
those through uv rather than a bare interpreter:

```bash
uv --project el-agent run python scripts/metrics/get_loc.py --help
```

Where things live that the filenames do not already tell you:

```
src/
  run/          base_full_run.py holds the round loop, coding -> extract ->
                apply; webgen_full_run.py and paperbench_full_run.py fill in
                the benchmark-specific parts; backup_layout.py is the single
                source of truth for every runs/ and backups/ path
  mswe_agents/  webgen/ and paperbench/ each hold the coding, local extract,
                global extract, apply and single-shot library agents, all
                subclassing base_coding_agent.py; webgen/edit_agent.py is the
                patcher the maintenance experiment drives
  prompts/      common/ is cross-benchmark, webgen/ and paperbench/ hold one
                module per agent plus the domain rules
  utils/        candidates/ decides what to extract and where to apply it;
                mdl/ has its own README for the formula
```

The name is historical and does not stand for anything.
