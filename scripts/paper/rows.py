"""Tag -> paper-row mapping for the main results table.

Single source of truth for which backup tags produce which row of
`paper/Super-library-agent-paper/tables/main-combined.tex` (and of the two
per-suite tables). Consumed by `build_main_table.py` and by the script that
populated `results/`.

Evidence for every mapping decision is recorded in `results/README.md`.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

# WebGen suites S1/S2/S3 == cluster ids 2/5/13 (data/augments/cluster/cluster.json).
WEBGEN_CLUSTERS = [2, 5, 13]
# PaperBench suites S1..S5 == cluster ids 0/1/2/4/5. Cluster 3 is a 12-task
# union of c0+c1+c2 and is NOT one of the paper's five suites.
PAPERBENCH_CLUSTERS = [0, 1, 2, 4, 5]
TRIALS = [1, 2, 3]

SUITES = {"webgen": WEBGEN_CLUSTERS, "paperbench": PAPERBENCH_CLUSTERS}


@dataclass
class Row:
    label: str            # row label as it appears in main-combined.tex
    bench: str            # webgen | paperbench  (== results/<bench>/)
    template: str         # tag template with {c} and {t}
    source: str           # backups/<source>/ the tags were copied from
    round: int
    phase: str
    clusters: List[int]
    # (cluster, trial) -> the backup tag to copy from, when the campaign did
    # not snapshot one under this row's own naming scheme.
    overrides: dict = field(default_factory=dict)

    def tags(self) -> List[Tuple[str, str]]:
        """[(tag, source_backup_dir)] in suite-major, trial-minor order.

        `tag` is the name this row uses under `results/`; it is uniform across
        the row. Use `backup_tags()` when you need where the files came from.
        """
        return [(tag, self.source) for tag, _src, _c, _t in self.backup_tags()]

    def backup_tags(self) -> List[Tuple[str, str, int, int]]:
        """[(results_tag, backup_tag, cluster, trial)]."""
        out = []
        for c in self.clusters:
            for t in TRIALS:
                tag = self.template.format(c=c, t=t)
                src = self.overrides.get((c, t), tag)
                out.append((tag, src, c, t))
        return out


ROWS = [
    # ---------------- WebGen-Bench ----------------
    Row("Zero-Shot", "webgen", "baseline-c{c}-t{t}", "webgen", 1, "coding",
        WEBGEN_CLUSTERS),
    # K=1 = the first of the K=8 samples, with best-of-K selection skipped.
    # No `-s1` tag was ever snapshotted on WebGen, so these were re-derived from
    # `backups/webgen-rb/librarian-c{c}-t{t}/final/round_1/samples/sample_1/`
    # and staged as tag-shaped trees; see results/README.md.
    Row("Librarian K=1", "webgen", "librarian-c{c}-t{t}-s1", "webgen-k1",
        1, "apply", WEBGEN_CLUSTERS),
    Row("Librarian K=8", "webgen", "librarian-c{c}-t{t}", "webgen-rb", 1, "apply",
        WEBGEN_CLUSTERS),
    Row("Naive-Implicit", "webgen", "sla-naive-c{c}-t{t}", "webgen", 4, "apply",
        WEBGEN_CLUSTERS),
    Row("Naive-Ward", "webgen", "sla-naive-c{c}-wc-t{t}", "webgen", 4, "apply",
        WEBGEN_CLUSTERS),
    Row("SLA-Full", "webgen", "sla-ours-c{c}-t{t}", "webgen", 4, "apply",
        WEBGEN_CLUSTERS),
    # ---------------- PaperBench ----------------
    Row("Zero-Shot", "paperbench", "baseline-c{c}-t{t}", "paperbench", 1, "coding",
        PAPERBENCH_CLUSTERS),
    # K'=1 re-selection over the same samples. `librarian-c2-t3-s1` was never
    # written because there the K'=1 winner is the K=8 winner, so that cell
    # reuses the K=8 tag (verified numerically — see results/README.md).
    Row("Librarian K=1", "paperbench", "librarian-c{c}-t{t}-s1", "paperbench",
        1, "apply", PAPERBENCH_CLUSTERS,
        overrides={(2, 3): "librarian-c2-t3"}),
    Row("Librarian K=8", "paperbench", "librarian-c{c}-t{t}", "paperbench",
        1, "apply", PAPERBENCH_CLUSTERS),
    Row("Naive-Implicit", "paperbench", "sla-naive-c{c}-t{t}", "paperbench",
        4, "apply", PAPERBENCH_CLUSTERS),
    Row("Naive-Ward", "paperbench", "sla-naive-wc-c{c}-t{t}", "paperbench",
        4, "apply", PAPERBENCH_CLUSTERS),
    Row("SLA-Full", "paperbench", "sla-ours-c{c}-t{t}", "paperbench",
        4, "apply", PAPERBENCH_CLUSTERS),
]

# Display order within each benchmark half of main-combined.tex.
ROW_ORDER = ["Zero-Shot", "Librarian K=1", "Librarian K=8",
             "Naive-Implicit", "Naive-Ward", "SLA-Full"]

# Rows the table checks skip. Their files are still in `results/` and
# `build_main_table.py` (without --diff) still emits them; only the comparison
# against the .tex leaves them out.
UNCHECKED_ROWS = {"Librarian K=1"}


# ---------------------------------------------------------------------------
# Post-construction maintenance (paper Table 2, `tables/maintenance-posthoc.tex`)
#
# A separate campaign: each finished portfolio absorbs one cross-application
# policy update, and we measure the resulting patch. Same three WebGen suites
# and three trials, but round 1 of `backups/webgen-maint/` rather than round 4
# of the construction run, so it needs its own row list.
MAINT_ROWS = [
    Row("Zero-Shot", "webgen-maint", "baseline-perapp-c{c}-t{t}", "webgen-maint",
        1, "apply", WEBGEN_CLUSTERS),
    Row("Librarian", "webgen-maint", "librarian-c{c}-t{t}", "webgen-maint",
        1, "apply", WEBGEN_CLUSTERS),
    Row("Naive-Implicit", "webgen-maint", "sla-naive-c{c}-t{t}", "webgen-maint",
        1, "apply", WEBGEN_CLUSTERS),
    # `wc` = Ward clustering, the candidate-selection strategy of Naive-Ward.
    Row("Naive-Ward", "webgen-maint", "sla-naive-c{c}-wc-t{t}", "webgen-maint",
        1, "apply", WEBGEN_CLUSTERS),
    Row("SLA-Full", "webgen-maint", "sla-ours-c{c}-t{t}", "webgen-maint",
        1, "apply", WEBGEN_CLUSTERS),
]

MAINT_ROW_ORDER = ["Zero-Shot", "Librarian", "Naive-Implicit", "Naive-Ward",
                   "SLA-Full"]
