# `results/` — the metric files behind the paper's tables

The paper's main and per-suite tables are recomputed from this directory alone. Two cells are the
exception: Librarian K=1 Acc. and Appearance on WebGen have no stored grader output here.

The files are verbatim copies of the per-run metric JSONs written into
`backups/<bench>/<tag>/eval_results/`. Three things are not straight copies:
`mdl_summary_calibrated.json` and `graded_tree_score.json` are derived, and `librarian-c2-t3-s1`
is one tag carried under a second name. All three are explained below. Being verbatim, the files
still carry the absolute paths of the machine they were measured on, under a directory named for
this project's former name (`el-poc`).

## Two campaigns

`webgen/` and `paperbench/` hold the initial-construction runs behind Table 1 and the per-suite
tables. `webgen-maint/` holds a separate campaign: each finished portfolio absorbs one
cross-application policy update, and the resulting patch is measured. That is Table 2 and
Appendix B.2. Same three WebGen suites and three trials, but round 1 of its own runs rather than
round 4 of the construction runs, which is why it is a third top-level directory rather than more
rounds under `webgen/`.

Table 2 reads four files per run. `maintenance_metrics.json` gives the patch size, split by
location (`App` is `apps + other`). `ui_test_results_regression.json` and
`ui_test_results_change.json` give the post-patch pass rate on behavior that already existed and
on the behavior the update asked for. `appearance_grade.json` gives the visual score. Each printed
cell is the mean over the nine runs. `loc_summary.json`, `scb_quality_summary.json` and
`mdl_summary.json` are here for the Appendix B.2 delta columns, which subtract the pre-patch
portfolio's values from these.

The 3.7x headline is `Total` for Zero-Shot over `SLA-Full`: 935.9 / 256.2 = 3.65, or 936 / 256 =
3.66 from the rounded cells the paper prints. It rounds to 3.7 at one decimal, which is a boundary
case, so the two endpoints are always shown alongside it.

## Checking the tables

```bash
python3 scripts/paper/build_main_table.py --diff        # Table 1 against the .tex, cell by cell
python3 scripts/paper/check_per_suite_tables.py         # the same for both per-suite tables
python3 scripts/paper/build_maintenance_table.py --diff # Table 2
```

All three compare against the paper's LaTeX sources, which live under `paper/` and are not in this repository.
From a clone, run the two builders without `--diff` to emit the table rows from `results/` alone and compare
them against the published PDF by eye.

Current state:

| Table | Exact | Within rounding | Mismatch | No data |
| --- | --- | --- | --- | --- |
| `main-combined.tex` | 72 | 8 | **1** | 0 |
| both per-suite tables | 417 | 0 | **3** | 0 |
| `maintenance-posthoc.tex` | 30 | 0 | 0 | 0 |

*Within rounding* means the recomputed value is inside ±1 unit in the last place the paper prints.
All eight are in the WebGen half of `main-combined.tex`, which was transcribed by hand from the
per-suite table's already-rounded cells and re-averaged, so it double-rounds. Two of the eight
(Naive-Implicit and Naive-Ward MDL) also come from an earlier MDL run than the rest of the column.
The MDL ordering is the same either way.

The four mismatches are one number: **PaperBench Librarian K=8 MDL, 30375 against a published
30400** (−0.082%), and the S2 and S3 cells behind it (29595/29656 and 36074/36140). The Librarian
MDL numbers live in a separate calibration file that was lost and regenerated; the regenerated
file reproduces PaperBench Tok exactly and both WebGen Librarian MDL cells exactly, so the residual
is scorer-level rather than a different corpus. The authors reviewed this and the published values
stand.

`scripts/paper/build_main_table.py --diff` prints both numbers for every non-exact cell, so
nothing here is hidden. The full derivation of each case is in this file's git history.

## Librarian K=1

The K=1 rows are in this directory but are left out of the table check — the WebGen row's stored
grader outputs did not survive, so two of its cells cannot be recomputed here and checking the
rest in isolation is not worth the exception. `build_main_table.py` still emits both rows; only
`--diff` skips them, and it says so in its summary.

K=1 is the first of the K=8 samples with best-of-K selection skipped. PaperBench snapshotted it as
real `-s1` tags. WebGen did not, so its code metrics were re-measured from the surviving
portfolios at `backups/webgen-rb/librarian-c{c}-t{t}/final/round_1/samples/sample_1/` with
`scripts/paper/stage_librarian_k1.py`; all seven reproduce the published values.

## Row → tag mapping

Three trials (`-t1/-t2/-t3`) per suite everywhere. `source` is the `backups/` directory the files
were copied from. `scripts/paper/rows.py` is the machine-readable form of these two tables and is
what `build_main_table.py` reads.

### WebGen-Bench — suites S1/S2/S3 = cluster ids 2/5/13

| Paper row | Tags | Source | Round | Phase | N |
| --- | --- | --- | --- | --- | --- |
| Zero-Shot | `baseline-c{2,5,13}-t{1,2,3}` | `backups/webgen` | 1 | `coding` | 9 |
| Librarian (K=1) | `librarian-c{2,5,13}-t{1,2,3}-s1` | `backups/webgen-k1` | 1 | `apply` | 9 |
| Librarian (K=8) | `librarian-c{2,5,13}-t{1,2,3}` | `backups/webgen-rb` | 1 | `apply` | 9 |
| Naive-Implicit | `sla-naive-c{2,5,13}-t{1,2,3}` | `backups/webgen` | 4 | `apply` | 9 |
| Naive-Ward | `sla-naive-c{2,5,13}-wc-t{1,2,3}` | `backups/webgen` | 4 | `apply` | 9 |
| SLA-Full | `sla-ours-c{2,5,13}-t{1,2,3}` | `backups/webgen` | 4 | `apply` | 9 |

### PaperBench — suites S1..S5 = cluster ids 0/1/2/4/5

| Paper row | Tags | Source | Round | Phase | N |
| --- | --- | --- | --- | --- | --- |
| Zero-Shot | `baseline-c{0,1,2,4,5}-t{1,2,3}` | `backups/paperbench` | 1 | `coding` | 15 |
| Librarian (K=1) | `librarian-c{0,1,2,4,5}-t{1,2,3}-s1` | `backups/paperbench` | 1 | `apply` | 15 |
| Librarian (K=8) | `librarian-c{0,1,2,4,5}-t{1,2,3}` | `backups/paperbench` | 1 | `apply` | 15 |
| Naive-Implicit | `sla-naive-c{0,1,2,4,5}-t{1,2,3}` | `backups/paperbench` | 4 | `apply` | 15 |
| Naive-Ward | `sla-naive-wc-c{0,1,2,4,5}-t{1,2,3}` | `backups/paperbench` | 4 | `apply` | 15 |
| SLA-Full | `sla-ours-c{0,1,2,4,5}-t{1,2,3}` | `backups/paperbench` | 4 | `apply` | 15 |

Three details in those tables are easy to get wrong:

- **PaperBench cluster 3 is excluded.** It holds twelve papers — the union of clusters 0, 1 and 2 —
  not a sixth suite of four.
- **The Naive-Ward tag convention differs between benchmarks**: `sla-naive-c{c}-wc-t{t}` on WebGen,
  `sla-naive-wc-c{c}-t{t}` on PaperBench. Both spellings are in `scripts/paper/rows.py`.
- **`librarian-c2-t3-s1` and `librarian-c2-t3` are the same run.** For that one trial the MDL
  rerank picked sample 1 as the K=8 winner (`winner_k: 1` in its `rerank_report.json`; the other
  fourteen picked 2 through 8), so K=1 and K=8 select the same portfolio. The campaign therefore
  never wrote a separate `-s1` snapshot; this directory carries one anyway, copied from
  `librarian-c2-t3`, so that all fifteen K=1 cells are addressed the same way.

## The files

Per `<bench>/<tag>/eval_results/round_<N>/<phase>/`:

| File | Feeds | Written by |
| --- | --- | --- |
| `loc_summary.json` | LOC, lib LOC | `scripts/metrics/get_loc.py` |
| `mdl_summary.json` | MDL, Tok, lib MDL, lib Tok | `scripts/metrics/get_mdl.py` |
| `mdl_summary_shared.json` | shared-concat MDL variant (not in the main table) | `get_mdl.py --method shared_concat` |
| `scb_quality_summary.json` | Eros., Verb. | `scripts/metrics/scb_quality.py` |
| `lib_usage_results.json` | library-usage analysis (not in the main table) | `scripts/metrics/get_lib_usage.py` |
| `ui_test_results.json` | WebGen Acc. | `scripts/eval/eval_webgen.sh` (WebVoyager) |
| `appearance_grade.json` | WebGen Appr. | `scripts/eval/eval_webgen.sh` (VLM grader) |
| `<paper_id>/graded_tree_score.json` | PaperBench Acc. | derived, see below |
| `<paper_id>/usage_summary.json` | grader token/cost accounting | `el-agent/src/eval/paperbench_eval.py` |
| `mdl_summary_calibrated.json` | MDL for Librarian rows only | derived, see below |

Two of those are derived rather than copied:

- **`graded_tree_score.json`** is a root-only digest of PaperBench's `graded_tree.json` (`id`,
  `score`, `valid_score`, `weight`, node/leaf counts). The full rubric trees carry a judge
  explanation on every leaf and run 284 MB in total; only the root score reaches the table. `aggregate.py` reads `graded_tree.json` when present and falls back to the digest, so
  dropping the full trees in beside these files just works. They are not lost — they stay in
  `backups/` and ship with the artifact portfolio.
- **`mdl_summary_calibrated.json`** exists for the Librarian tags only (25 directories: the 24
  K=8 tags plus the `librarian-c2-t3-s1` copy). When present it
  overrides `mdl_summary.json`, because the paper's Librarian MDL columns are the calibrated
  measurements; the raw pipeline values were taken on a differently-configured vLLM server.
  `--no-calibrated` shows the raw numbers.

Deliberately not copied: `logs/**`, `final/**` (the submissions themselves), `leaf_logs/`,
`usage.jsonl`, and the `*_backup` variants of the summaries.

## Caveats from the campaign

These are properties of the underlying runs, not of this directory.

- WebGen Acc. is WebVoyager end-to-end accuracy and varies between repeated trials, so small
  differences between arms should not be read as real. The paper's claim rests on the code-volume
  and duplication metrics, not on Acc.
- Librarian is a 1-round post-hoc arm with an 8× sampling budget, compared against 4-round
  cumulative arms. The asymmetry is deliberate and reported.
- PaperBench rubric grading routed through two different providers across suites.
