# WebGen suites

`cluster.json` is a Ward clustering of the 101 WebGen-Bench tasks over `text-embedding-3-small`
embeddings, k=14, ids 0 to 13. `build_cluster.py` regenerates it; its docstring records which task
fields were embedded and the ablation that settled on them.

Inside a cluster, `tasks` is ordered by ascending distance to the cluster centroid. `CLUSTER_SIZE=8`
therefore keeps the 8 tightest members and drops the stragglers, and rounds take that order in
blocks of `M`. The suites in the paper are the tightest 8 of clusters 2, 5 and 13.

## diverse16.json

A constructed 16-task suite (id 20), not a clustering output: the centroid-closest task of each Ward
cluster, minus everything already used in the reported suites. The file's `_note` field has the
exact derivation.

It answers the objection that 8-task suites are too small, and it does so in the regime that is
hardest for the method. The 16 tasks come from 13 of the 14 clusters and cover all three
`primary_category` families (7 Content Presentation, 6 User Interaction, 3 Data Management), so
there is about as little for a shared library to capture as this benchmark allows.

```bash
CLUSTER_JSON=data/augments/webgen/cluster/diverse16.json \
CLUSTER_ID=20 CLUSTER_SIZE=16 M=2 \
TAG=sla-ours-div16-t1 MODE=sla_ours \
  bash scripts/run/run_webgen_full.sh
```

Its task ids are in ascending order, which is also the round order: 8 rounds of 2.
