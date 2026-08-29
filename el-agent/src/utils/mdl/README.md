# utils.mdl

Description length of generated code under a shared library: how many nats a
fixed language model needs to write the corpus, given the library it imports.
Lower means the portfolio repeats itself less. `scripts/metrics/get_mdl.py` is
the CLI, this package is the implementation.

## Formula

```
MDL  =  NLL(lib_stripped)  +  Σ_f  NLL(file_stripped | direct_deps_of_f)
```

`NLL(x | y)` is the token-level negative log-likelihood of `x` with `y` as
prefix, read off a vLLM completions endpoint with `echo=True, logprobs=1,
max_tokens=1` and the BOS and generated tokens trimmed (`llm.py`). The library
term is the whole library directory in one string, one markdown fence per file.
Comments and docstrings are stripped from the library and from every scored
file, so the score measures code and not prose.

Three parts of that line are choices, not defaults.

- `direct_deps_of_f` is what `f` imports, one level, never transitive.
  Conditioning every file on the entire library instead is `--method concat`,
  kept for comparison against pre-2026 numbers: it credits an app for library
  code it never imports. Transitive closure errs the other way and reprints one
  shared dependency into the context of every importer.
- Dep context keeps its comments. Only the file being scored is stripped. The
  context stands in for what the model would really be reading.
- The library is scored once per library directory, not once per app. Every app
  record carries the same `library_nll`, and the summary averages it back down
  to a single copy.

## Running

Start the scoring server first (`bash scripts/infra/deploy_vllm.sh`, serves
`Qwen/Qwen2.5-7B` on port 8000), then:

```bash
uv --project el-agent run python scripts/metrics/get_mdl.py \
    --task webgen --backup-tag demo --phase apply --round 4
```

`--task` selects `configs/webgen.yaml` or `configs/paperbench.yaml`, which set
the file extensions, the ignore rules and the parser language. `--round`
defaults to the highest round on disk carrying the requested `--phase`.
`--task-ids`, or `--paper-ids` on PaperBench, restricts the run to a subset.
`--openai_base_url`, or `MDL_BASE_URL`, moves the endpoint off
`http://127.0.0.1:8000/v1`; a second server with half the tags pointed at it
roughly halves the wall clock. `--help` covers the rest.

Output lands under `backups/<bench>/<tag>/eval_results/round_<N>/<phase>/`, as
`tasks/<id>/mdl_results.json` per app plus `mdl_summary.json` for the phase.
`--method concat` and `--method shared_concat` write `*_old.json` and
`*_shared.json` beside those instead of overwriting them.

## Reading the output

Both files are `{"apps": [...], "summary": {...}}`. The paper's MDL column is
`summary.mdl_nll`, which is `sum_app_nll + avg_library_nll`. That average runs
over records that all hold the same library, so the library is counted once. The
Tok column is not in `summary`: it is `Σ apps[*].app_tokens +
apps[0].library_tokens`, assembled in `scripts/paper/aggregate.py`. Per file,
`file_details[i].dep_paths` says what that file was conditioned on, which is the
quickest check that the dep graph resolved the imports you expected.

## What goes wrong quietly

- Library NLL is flat-concatenated while app files are dep-aware, so lib
  NLL/token carries a prior penalty that app NLL/token does not. Compare lib
  against lib and app against app; lib against app means nothing.
- Under `--phase coding` and `--phase apply` the library is auto-discovered at
  `tasks/<id>/lib/`. A backup assembled by hand without that per-task copy is
  scored with no library at all, `library_nll` comes back 0, and nothing warns.
- Comment stripping runs per file, before the markdown fence. Strip after the
  fence and JS keeps its comments: the string-literal regex in `parsers/js.py`
  reads a triple-backtick fence as a template literal. Passing `strip_fn` to
  `read_dir_to_text` gets the order right for you.

## Calling it directly

```python
from utils.mdl import MDLMetric, load_task_config

metric = MDLMetric(model="Qwen/Qwen2.5-7B", base_url="http://127.0.0.1:8000/v1")
result = await metric.ascore(app_dir, lib_dir, task=load_task_config("webgen"))
print(result.app_nll, result.library_nll)
```

`get_mdl_score(app_dir, lib_dir)` is the synchronous one-shot form and returns
`nan` instead of raising. `get_maintainability_metrics` gives LOC and token
counts with no LLM. `__init__.py` lists the rest of the surface.
