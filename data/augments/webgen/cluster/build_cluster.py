"""Cluster WebGen-Bench tasks by topical similarity.

Uses text-embedding-3-small + AgglomerativeClustering (ward + euclidean),
with per-cluster centroid-distance ordering.

Embedding signal ("full" — selected after ablation):
    instruction
    primary_category
    subcategories
    application_type
    ui_interactions       (ui_instruct[*].task)
    ui_categories         (ui_instruct[*].task_category, multiset)

Ablation summary (see git history; n_clusters=8 unless noted):
    baseline (no ui_*)            best mean_d=0.125 (n=4 Analytics)
    with_ui   (+ ui task text)    best mean_d=0.146  ← worse (noise)
    with_cat  (+ ui categories)   best mean_d=0.118 (n=8 Travel)
    full      (both)              best mean_d=0.101 (n=6 Travel)  ← winner
    weighted_cat (cat ×3)         best mean_d=0.079, but cluster meaning warps
    cat_centric (drop instruction) mean_d=0.012  ← spurious precision
With n_clusters=14, full-signal recovers an additional N=8 cross-domain
profile/match cluster (top8_d=0.101) that subsumes the Travel single-domain
group as a higher-tightness alternative.

Output (`data/augments/webgen/cluster/cluster.json`):

```
{"model": "...", "linkage": "ward", "metric": "euclidean", "n_clusters": 14,
 "clusters": [{"id": int, "size": int, "tasks": [...], "centroid_distances": [...],
               "members": [{"task_id": "...", "application_type": "...", ...}]}, ...]}
```

Usage:
    python data/augments/webgen/cluster/build_cluster.py [--n-clusters 14]
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering


SCRIPT_DIR = Path(__file__).resolve().parent              # …/data/webgen-bench
PROJECT_DIR = SCRIPT_DIR.parent.parent                    # project root
EMB_DIR = SCRIPT_DIR / "embeddings"
CACHE_FILE = EMB_DIR / "text_embeddings.npz"


def load_tasks(jsonl_path: Path) -> list[dict]:
    return [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_text(task: dict) -> str:
    """Full signal: description + categories + ui interaction text + ui category multiset."""
    cat = task.get("Category", {}) or {}
    primary = (cat.get("primary_category") or "").strip()
    subs = [str(s).strip() for s in (cat.get("subcategories") or []) if str(s).strip()]
    instruction = (task.get("instruction") or "").strip()
    app_type = (task.get("application_type") or "").strip()

    parts = [
        f"instruction: {instruction}",
        f"primary_category: {primary}",
        f"subcategories: {', '.join(subs)}",
        f"application_type: {app_type}",
    ]

    ui = task.get("ui_instruct") or []

    ui_tasks = [(u.get("task") or "").strip() for u in ui]
    ui_tasks = [t for t in ui_tasks if t]
    if ui_tasks:
        parts.append("ui_interactions:\n" + "\n".join(f"  - {t}" for t in ui_tasks))

    cat_tokens = []
    for u in ui:
        uc = u.get("task_category", {}) or {}
        up = (uc.get("primary_category") or "").strip()
        usubs = [str(s).strip() for s in (uc.get("subcategories") or []) if str(s).strip()]
        if up:
            cat_tokens.append(up)
        cat_tokens.extend(usubs)
    if cat_tokens:
        parts.append("ui_categories: " + ", ".join(cat_tokens))

    return "\n".join(parts)


def embed_batch(client: OpenAI, texts: list[str], model: str, batch_size: int = 32) -> np.ndarray:
    out = []
    nbatches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  embed batch {i // batch_size + 1}/{nbatches}")
        resp = client.embeddings.create(input=batch, model=model)
        out.extend([d.embedding for d in sorted(resp.data, key=lambda x: x.index)])
    return np.asarray(out, dtype=np.float32)


def load_or_compute_embeddings(
    tasks: list[dict],
    texts: list[str],
    model: str,
    force_reembed: bool,
) -> np.ndarray:
    if not force_reembed and CACHE_FILE.exists():
        npz = np.load(CACHE_FILE)
        if "E" in npz and npz["E"].shape[0] == len(tasks):
            print(f"  reusing cached embeddings: {CACHE_FILE} (shape={npz['E'].shape})")
            return np.asarray(npz["E"], dtype=np.float32)
        print(f"  cache shape mismatch; recomputing")

    load_dotenv(PROJECT_DIR / ".env")
    api_key = (
        os.environ.get("OPENAILIKE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        sys.exit("Need OPENAILIKE_API_KEY / OPENAI_API_KEY in .env")
    base_url = os.environ.get("OPENAILIKE_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)
    E = embed_batch(client, texts, model)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_FILE, E=E)
    print(f"  wrote {CACHE_FILE}")
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clusters", type=int, default=14,
                    help="Target cluster count for AgglomerativeClustering. "
                         "WebGen-Bench has 101 test tasks; 14 chosen via ablation "
                         "(recovers cross-domain N=8 profile/match cluster).")
    ap.add_argument("--linkage", default="ward", choices=["ward", "average", "complete"])
    ap.add_argument("--metric", default="euclidean", choices=["cosine", "euclidean"])
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--input", default=str(SCRIPT_DIR / "test.jsonl"))
    ap.add_argument("--force-reembed", action="store_true",
                    help="Ignore cached embeddings and recompute.")
    ap.add_argument("--out", default=str(SCRIPT_DIR / "cluster.json"))
    args = ap.parse_args()

    if args.linkage == "ward" and args.metric != "euclidean":
        print("[warn] linkage=ward forces metric=euclidean; switching.")
        args.metric = "euclidean"

    tasks = load_tasks(Path(args.input))
    print(f"Loaded {len(tasks)} tasks from {args.input}")
    if len(tasks) < args.n_clusters:
        sys.exit(f"--n-clusters={args.n_clusters} but only {len(tasks)} tasks present")

    texts = [build_text(t) for t in tasks]
    char_lens = [len(t) for t in texts]
    print(f"  text chars: min={min(char_lens)} max={max(char_lens)} mean={int(sum(char_lens)/len(char_lens))}")

    E = load_or_compute_embeddings(tasks, texts, args.model, args.force_reembed)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    cl = AgglomerativeClustering(
        n_clusters=args.n_clusters, linkage=args.linkage, metric=args.metric
    )
    labels = cl.fit_predict(E)

    clusters: dict[int, list[dict]] = {}
    for idx, (task, label) in enumerate(zip(tasks, labels)):
        cat = task.get("Category", {}) or {}
        clusters.setdefault(int(label), []).append({
            "task_id": task.get("id", ""),
            "application_type": task.get("application_type", ""),
            "primary_category": cat.get("primary_category", ""),
            "subcategories": cat.get("subcategories", []),
            "_idx": idx,
        })

    cluster_blocks = []
    for cid, mem in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        idxs = [m["_idx"] for m in mem]
        sub_E = E[idxs]
        centroid = sub_E.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        dists = 1.0 - sub_E @ centroid
        order = np.argsort(dists)
        sorted_mem = [mem[i] for i in order]
        sorted_dists = [float(dists[i]) for i in order]
        for m in sorted_mem:
            m.pop("_idx", None)
        cluster_blocks.append({
            "id": cid,
            "size": len(sorted_mem),
            "tasks": [m["task_id"] for m in sorted_mem],
            "centroid_distances": sorted_dists,
            "members": sorted_mem,
        })

    out = {
        "model": args.model,
        "linkage": args.linkage,
        "metric": args.metric,
        "n_clusters": args.n_clusters,
        "clusters": cluster_blocks,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    print("\nCluster sizes (sorted desc):")
    for c in out["clusters"]:
        sample = ", ".join(c["tasks"][:5]) + ("..." if c["size"] > 5 else "")
        print(f"  cluster {c['id']:2d} (n={c['size']:2d}): {sample}")


if __name__ == "__main__":
    main()
