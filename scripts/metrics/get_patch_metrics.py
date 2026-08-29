#!/usr/bin/env python3
"""Aggregate maintenance-patch metrics for a maint-* backup tag.

Reads the diff artifacts + run_results.json produced by
run.webgen_maintenance_run and computes the rebuttal-table row:

    touched files / changed LOC / patch location / token-cost proxy (llm
    calls, usd)

DEPRECATED — duplicate edit sites. This metric is OFF by default and must be
requested with --dup-sites. It is confounded by session structure and MUST
NOT be reported: Protocol B applies one session across all apps, producing
stylistically uniform edits that the text-similarity detector matches;
Protocol C (baseline) uses a fresh session per app, producing diverse edits
that do NOT match — even though every app independently re-implemented the
same policy. The detector therefore UNDERCOUNTS baseline's conceptual
redundancy, biasing the comparison against the hypothesis. Capturing "how
many apps independently re-implemented the same requirement" needs a
concept-level definition, not diff-text similarity. The implementation is
kept below for exploration only.

Duplicate edit sites (v2, hybrid) — DEPRECATED, --dup-sites only:

1. Deterministic prefilter (recall): added lines are normalized with
   identifiers masked but STRING LITERALS PRESERVED, and import/boilerplate
   lines dropped. A cross-app hunk pair becomes a candidate when its shingle
   Jaccard >= --prefilter-threshold (low, default 0.2) OR the two hunks share
   a non-trivial string literal (policy edits carry policy-mandated strings:
   "Pending review", "Received", "Last updated", ID prefixes, ...).
2. LLM judge (precision + relevance): each candidate pair is classified
   against the suite's policy text as policy_duplicate / shared_call_site /
   incidental_duplicate / not_duplicate (temperature 0; judgments cached in
   eval_results/.../dup_judgments.json so re-runs are free).
   shared_call_site separates thin per-app wiring that merely invokes a
   shared lib implementation — unavoidable for library methods and NOT
   duplicated logic — from true per-app re-implementation.
3. Clusters: connected components over policy_duplicate pairs.
   duplicate_edit_sites = hunks belonging to any cluster;
   duplicate_clusters   = one entry per repeated edit ("the same policy logic
   re-implemented in these N places");
   redundant_edits      = sum(cluster_size - 1) — edits a shared library
   would have collapsed into one.

A purely deterministic strict-threshold count (`deterministic_v2`) is
reported alongside for robustness, and is the primary result under --no-llm.

Usage:
    uv --project el-agent run python scripts/metrics/get_patch_metrics.py \
        --maint-tag sla-ours-c13-t1        # non-baseline: source tag as-is
        --maint-tag baseline-persuite-c13-t1   # baseline Protocol B
        --maint-tag baseline-perapp-c13-t1     # baseline Protocol C
Saves: backups/webgen-maint/<maint-tag>/eval_results/round_1/apply/maintenance_metrics.json
"""

from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import threading
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
BACKUPS = PROJECT_DIR / "backups" / "webgen-maint"
MAINT_DATA = PROJECT_DIR / "data" / "augments" / "webgen" / "maintenance"


_NOISE_BASENAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "agent.env"}


def is_noise_path(path: str | None) -> bool:
    """Build artifacts / auto-generated lockfiles — excluded from patch surface
    (mirrors run.webgen_maintenance_run.is_noise_path)."""
    if not path:
        return False
    if path.rsplit("/", 1)[-1] in _NOISE_BASENAMES:
        return True
    return ("/node_modules/" in path or "/dist/" in path
            or path.startswith(("node_modules/", "dist/")))


def parse_hunks(patch_text: str) -> list[dict]:
    """Split a unified diff into per-file hunks with added-line content."""
    hunks: list[dict] = []
    current_file = None
    current: dict | None = None
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            m = re.search(r" b/(.+)$", line)
            current_file = m.group(1) if m else None
            current = None
        elif line.startswith("@@"):
            current = {"file": current_file, "added": []}
            hunks.append(current)
        elif current is not None and line.startswith("+") and not line.startswith("+++"):
            current["added"].append(line[1:])
    return [h for h in hunks if h["added"] and not is_noise_path(h["file"])]


# --- normalization (v2: literal-preserving, boilerplate-dropping) -----------

_WS = re.compile(r"\s+")
_IDENT = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
_STR = re.compile(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"|`(?:[^`\\\n]|\\.)*`")
# import/re-export/require-assignment/comment lines — near-zero information,
# collapse to identical shingles across unrelated edits.
_BOILER = re.compile(
    r"^(?:import\b|export\s+(?:\{|\*)|\}?\s*from\s+['\"]"
    r"|(?:const|let|var)\s+\S+\s*=\s*require\(|//|/\*|\*)"
)
_JUNK = {"{", "}", "};", ")", ");", "(", "[", "]", "],", "/>", ">", "<>", "</>",
         "})", "});", "return (", "return(", "} else {", "try {", "} catch {"}


def strip_comments(line: str) -> str:
    """Drop // line comments and /* */ block comments, while never cutting a
    delimiter that lives inside a string literal. Only code survives — so the
    duplicate detector compares behavior, not commentary. Continuation lines
    of a multi-line block comment (` * ...`) are caught by _BOILER instead."""
    out: list[str] = []
    i, n, quote = 0, len(line), None
    while i < n:
        c = line[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(line[i + 1]); i += 2; continue
            if c == quote:
                quote = None
            i += 1; continue
        if c in "'\"`":
            quote = c; out.append(c); i += 1; continue
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            break                                  # line comment → drop rest
        if c == "/" and i + 1 < n and line[i + 1] == "*":
            j = line.find("*/", i + 2)
            if j == -1:
                break                              # unterminated → drop rest
            i = j + 2; continue                    # skip the block comment
        out.append(c); i += 1
    return "".join(out)


def normalize(lines: list[str]) -> set[str]:
    """Line shingles: comments stripped, whitespace collapsed, identifiers
    masked, string literals kept verbatim (they carry the policy semantics)."""
    out = set()
    for ln in lines:
        ln = _WS.sub(" ", strip_comments(ln).strip())
        if not ln or ln in _JUNK or _BOILER.match(ln):
            continue
        literals: list[str] = []

        def _stash(m: re.Match) -> str:
            literals.append(m.group(0))
            return f"\x00{len(literals) - 1}\x00"

        masked = _IDENT.sub("ID", _STR.sub(_stash, ln))
        for i, lit in enumerate(literals):
            masked = masked.replace(f"\x00{i}\x00", lit)
        out.add(masked)
    return out


def literals_of(lines: list[str]) -> set[str]:
    """Non-trivial string literals of a hunk (shared-literal prefilter).
    Path-like / URL literals are skipped — they mark imports, not policy."""
    lits: set[str] = set()
    for ln in lines:
        ln = strip_comments(ln)
        if _BOILER.match(ln.strip()):
            continue
        for m in _STR.finditer(ln):
            s = m.group(0)[1:-1].strip()
            if len(s) >= 5 and not s.startswith((".", "/", "@", "#", "http")):
                lits.add(s.lower())
    return lits


def app_of(path: str) -> str | None:
    m = re.match(r"tasks/(\d+)/", path or "")
    return m.group(1) if m else None


def location_of(path: str) -> str:
    if path.startswith("lib/"):
        return "lib"
    if app_of(path):
        return "apps"
    return "other"


# --- LLM judge ----------------------------------------------------------------

_JUDGE_PROMPT = """\
You are auditing a maintenance patch applied to a portfolio of independent \
web apps. The following policy update was requested:

--- POLICY UPDATE ---
{policy}
--- END POLICY ---

Below are two diff hunks (added lines only) from DIFFERENT apps in the \
portfolio.

Hunk A — {file_a}:
```
{added_a}
```

Hunk B — {file_b}:
```
{added_b}
```

Question: are these two hunks the SAME edit implemented in two places?

- "policy_duplicate": both hunks IMPLEMENT the same logic required by the \
policy update (same validation logic, same badge markup/styling, same \
normalization code, ...), duplicated because each app carries its own copy \
of the implementation.
- "shared_call_site": both hunks merely INVOKE a shared library \
function/component/hook for the policy behavior (import + a thin call or \
JSX tag); the implementation itself lives in one shared place. Minimal \
per-app wiring, not duplicated logic.
- "incidental_duplicate": the same/near-identical edit repeated, but NOT \
implementing the policy update (formatting, unrelated refactor, imports).
- "not_duplicate": only superficially similar — they implement different \
behavior or different parts of the policy.

Answer with JSON only:
{{"verdict": "policy_duplicate" | "shared_call_site" | "incidental_duplicate" | "not_duplicate", "reason": "<one sentence>"}}
"""

_PROMPT_VERSION = hashlib.sha1(_JUDGE_PROMPT.encode()).hexdigest()[:12]
_JUDGE_TEMPERATURE = 0.0
_JUDGE_MAX_TOKENS = 8000

_VERDICTS = {"policy_duplicate", "shared_call_site", "incidental_duplicate",
             "not_duplicate"}


class PairJudge:
    """LLM verdict per candidate hunk pair, cached on disk by content hash."""

    def __init__(self, model: str, provider: str, policy_text: str,
                 cache_path: Path):
        agent_src = str(PROJECT_DIR / "el-agent" / "src")
        if agent_src not in sys.path:
            sys.path.insert(0, agent_src)
        from dotenv import load_dotenv
        load_dotenv(PROJECT_DIR / ".env")
        from utils.llm import llm_generation
        self._generate = llm_generation
        self.model = model
        self.provider = provider
        self.policy_text = policy_text
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text())
        self.calls = 0
        self.cost_usd = 0.0
        self._lock = threading.Lock()  # guards cache + counters

    def _key(self, ha: dict, hb: dict) -> str:
        blob = "\x1e".join([
            _PROMPT_VERSION, self.model, self.provider,
            str(_JUDGE_TEMPERATURE), str(_JUDGE_MAX_TOKENS), self.policy_text,
            ha["file"], hb["file"],
            "\n".join(ha["added"]), "\n".join(hb["added"]),
        ])
        return hashlib.sha1(blob.encode()).hexdigest()

    def judge(self, ha: dict, hb: dict) -> dict:
        key = self._key(ha, hb)
        with self._lock:
            if key in self.cache:
                return self.cache[key]
        prompt = _JUDGE_PROMPT.format(
            policy=self.policy_text.strip(),
            file_a=ha["file"], added_a="\n".join(ha["added"][:80]),
            file_b=hb["file"], added_b="\n".join(hb["added"][:80]),
        )
        # generous budget: reasoning models spend tokens before the JSON.
        # max_tokens is a cap, not a target — only generated tokens billed.
        try:
            resp = self._generate(
                [{"role": "user", "content": prompt}],
                model=self.model, provider=self.provider,
                temperature=_JUDGE_TEMPERATURE, max_tokens=_JUDGE_MAX_TOKENS,
            )
        except Exception as exc:  # transient API failure must not abort the run
            with self._lock:
                self.calls += 1
            return {"verdict": "not_duplicate",
                    "reason": f"unparseable judge output (api error: {exc.__class__.__name__})"}
        verdict, reason = "not_duplicate", "unparseable judge output"
        parsed = False
        m = re.search(r"\{.*\}", resp.get("content") or "", re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                if data.get("verdict") in _VERDICTS:
                    verdict = data["verdict"]
                    reason = str(data.get("reason", ""))[:300]
                    parsed = True
            except json.JSONDecodeError:
                pass
        result = {"verdict": verdict, "reason": reason}
        with self._lock:
            self.calls += 1
            self.cost_usd += float(resp.get("cost") or 0.0)
            if parsed:  # never cache fallback verdicts — re-judged on next run
                self.cache[key] = result
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self.cache, indent=1))
        return result


# --- duplicate edit sites (prefilter -> judge -> clusters) ---------------------


def _jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Connected components (size >= 2) over n nodes."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        parent[find(i)] = find(j)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)
    return [m for m in groups.values() if len(m) >= 2]


def duplicate_edit_sites(hunks: list[dict], judge: PairJudge | None,
                         prefilter_t: float, strict_t: float,
                         max_judge_pairs: int) -> dict:
    """Hybrid cross-app duplicate detection; see module docstring."""
    sigs = []
    for h in hunks:
        app = app_of(h["file"])
        if app is None:
            continue
        sig = normalize(h["added"])
        lits = literals_of(h["added"])
        # tiny hunks still count when they carry a policy literal
        # (one-line "Pending review" badge edits are exactly the signal)
        if len(sig) >= 3 or (sig and lits):
            sigs.append({"app": app, "file": h["file"], "sig": sig,
                         "lits": lits, "added": h["added"]})

    # 1. prefilter: low-Jaccard OR shared-literal cross-app pairs
    candidates: list[tuple[int, int, float, list[str]]] = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            if sigs[i]["app"] == sigs[j]["app"]:
                continue
            jac = _jaccard(sigs[i]["sig"], sigs[j]["sig"])
            shared = sorted(sigs[i]["lits"] & sigs[j]["lits"])
            if jac >= prefilter_t or shared:
                candidates.append((i, j, jac, shared))

    # deterministic strict-threshold flags (robustness / --no-llm fallback)
    det_edges = [(i, j) for i, j, jac, _ in candidates if jac >= strict_t]

    # 2. judge candidates. If we must truncate, PROTECT shared-literal pairs
    #    first: a genuine policy duplicate can have near-zero Jaccard while
    #    carrying the policy literal, and dropping those preferentially would
    #    under-count exactly the methods that generate the most pairs.
    truncated = 0
    if judge is not None and len(candidates) > max_judge_pairs:
        candidates.sort(key=lambda c: (not c[3], -c[2]))
        truncated = len(candidates) - max_judge_pairs
        candidates = candidates[:max_judge_pairs]
        print(f"WARNING: judge truncated {truncated} candidate pairs — "
              f"raise --max-judge-pairs for a reported run", file=sys.stderr)

    policy_edges, call_edges, incidental_edges, judged_pairs = [], [], [], []
    if judge is not None:
        for i, j, jac, shared in candidates:
            verdict = judge.judge(sigs[i], sigs[j])
            judged_pairs.append({
                "files": [sigs[i]["file"], sigs[j]["file"]],
                "jaccard": round(jac, 2), "shared_literals": shared[:5],
                **verdict,
            })
            edges = {"policy_duplicate": policy_edges,
                     "shared_call_site": call_edges,
                     "incidental_duplicate": incidental_edges}.get(verdict["verdict"])
            if edges is not None:
                edges.append((i, j))
    else:
        policy_edges = det_edges

    # 3. connected components over policy-duplicate pairs
    def cluster_of(members: list[int]) -> dict:
        return {
            "size": len(members),
            "apps": sorted({sigs[m]["app"] for m in members}),
            "files": sorted({sigs[m]["file"] for m in members}),
            "sample": sigs[members[0]]["added"][:6],
        }

    comps = _components(len(sigs), policy_edges)
    clusters = sorted((cluster_of(m) for m in comps), key=lambda c: -c["size"])
    flagged = {m for comp in comps for m in comp}
    per_app: dict[str, int] = defaultdict(int)
    for m in flagged:
        per_app[sigs[m]["app"]] += 1

    call_comps = _components(len(sigs), call_edges)
    det_comps = _components(len(sigs), det_edges)

    out = {
        "duplicate_edit_sites": len(flagged),
        "duplicate_clusters": clusters,
        "redundant_edits": sum(c["size"] - 1 for c in clusters),
        "shared_call_sites": sum(len(m) for m in call_comps),
        "shared_call_site_clusters": [cluster_of(m) for m in call_comps],
        "compared_hunks": len(sigs),
        "candidate_pairs": len(candidates) + truncated,
        "per_app": dict(per_app),
        "deterministic_v2": {
            "duplicate_edit_sites": sum(len(m) for m in det_comps),
            "clusters": len(det_comps),
            "strict_threshold": strict_t,
        },
    }
    if judge is not None:
        out["judge"] = {
            "model": judge.model,
            "incidental_duplicate_pairs": len(incidental_edges),
            "judged_pairs": judged_pairs,
            "new_calls": judge.calls,
            "new_cost_usd": round(judge.cost_usd, 4),
            "truncated_pairs": truncated,
            "unparseable_pairs": sum(
                1 for p in judged_pairs
                if p["reason"].startswith("unparseable judge output")),
        }
    else:
        out["judge"] = {"model": None, "note": "--no-llm: deterministic strict-threshold edges used as policy duplicates"}
    return out


def analyze_patch(patch_path: Path, judge: PairJudge | None,
                  prefilter_t: float, strict_t: float,
                  max_judge_pairs: int, compute_dups: bool = False) -> dict:
    text = patch_path.read_text(errors="ignore")
    hunks = parse_hunks(text)
    files = sorted({h["file"] for h in hunks if h["file"]})
    loc = {"lib": 0, "apps": 0, "other": 0}
    for h in hunks:
        loc[location_of(h["file"])] += len(h["added"])
    locations = {location_of(f) for f in files}
    if locations <= {"lib"}:
        patch_location = "lib only"
    elif "lib" in locations:
        patch_location = "lib + apps"
    else:
        patch_location = "apps only"
    out = {
        "touched_files": len(files),
        "files": files,
        "added_loc_by_location": loc,
        "patch_location": patch_location,
    }
    if compute_dups:  # DEPRECATED metric — see module docstring / --dup-sites
        out.update(duplicate_edit_sites(hunks, judge, prefilter_t, strict_t,
                                        max_judge_pairs))
    return out


def load_policy(maint_tag: str, policy_file: str | None) -> str:
    if policy_file:
        return Path(policy_file).read_text()
    # tolerate method postfixes between suite and trial, e.g. the sla-naive-wc
    # variant tag `sla-naive-c2-wc-t1`.
    m = re.search(r"-(c\d+)(?:-[a-z0-9]+)*-t\d+$", maint_tag)
    if not m:
        raise SystemExit(
            f"cannot derive suite from tag '{maint_tag}'; pass --policy-file")
    path = MAINT_DATA / f"policy_{m.group(1)}.md"
    if not path.exists():
        raise SystemExit(f"policy file not found: {path}")
    return path.read_text()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--maint-tag", required=True)
    p.add_argument("--prefilter-threshold", type=float, default=0.2,
                   help="Jaccard floor for LLM-judge candidates (recall stage)")
    p.add_argument("--strict-threshold", type=float, default=0.5,
                   help="Jaccard for the deterministic_v2 fallback count")
    p.add_argument("--dup-sites", action="store_true",
                   help="[DEPRECATED] compute cross-app duplicate-edit-sites. "
                        "OFF by default: the text-similarity signal is "
                        "confounded by session structure (Protocol B's single "
                        "session yields uniform edits that match, Protocol C's "
                        "per-app sessions yield diverse edits that do not) — so "
                        "it undercounts baseline's conceptual redundancy. Kept "
                        "for exploration only; do not report.")
    p.add_argument("--no-llm", action="store_true",
                   help="with --dup-sites: skip the judge; deterministic_v2 becomes the result")
    p.add_argument("--judge-model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--judge-provider", default="openrouter")
    p.add_argument("--max-judge-pairs", type=int, default=300)
    p.add_argument("--policy-file", default=None,
                   help="override policy text path (default: derived from tag)")
    args = p.parse_args()

    root = BACKUPS / args.maint_tag
    results_file = root / "run_results.json"
    run_results = json.loads(results_file.read_text()) if results_file.exists() else {}

    diff_root = root / "diff"
    patches = sorted(diff_root.rglob("patch.diff"))
    if not patches:
        raise SystemExit(f"no patch.diff found under {diff_root}")

    out_dir = root / "eval_results" / "round_1" / "apply"
    if args.dup_sites:
        print("WARNING: --dup-sites is DEPRECATED — the duplicate-edit-sites "
              "signal is confounded by session structure and undercounts "
              "baseline redundancy; exploratory only, do not report.",
              file=sys.stderr)
    judge = None
    if args.dup_sites and not args.no_llm:
        judge = PairJudge(args.judge_model, args.judge_provider,
                          load_policy(args.maint_tag, args.policy_file),
                          out_dir / "dup_judgments.json")

    dup_kw = dict(judge=judge, prefilter_t=args.prefilter_threshold,
                  strict_t=args.strict_threshold,
                  max_judge_pairs=args.max_judge_pairs,
                  compute_dups=args.dup_sites)

    # Protocol from run_results (authoritative); patch count only as fallback.
    # A single-target Protocol C run also yields one patch.diff — inferring
    # from count alone would silently misparse its submission/... paths as B.
    protocol = (run_results.get("protocol")
                or ("b" if len(patches) == 1 else "c")).lower()

    if protocol == "b":  # one suite-level patch
        if len(patches) != 1:
            raise SystemExit(f"protocol b expects 1 patch, found {len(patches)}")
        analysis = analyze_patch(patches[0], **dup_kw)
    else:  # Protocol C — one patch per app; merge
        per_app = {p_.parent.name: analyze_patch(
                       p_, judge=None, prefilter_t=args.prefilter_threshold,
                       strict_t=args.strict_threshold,
                       max_judge_pairs=args.max_judge_pairs, compute_dups=False)
                   for p_ in patches}
        analysis = {
            "touched_files": sum(a["touched_files"] for a in per_app.values()),
            "added_loc_by_location": {
                "lib": 0,
                "apps": sum(a["added_loc_by_location"]["apps"]
                            + a["added_loc_by_location"]["other"]
                            for a in per_app.values()),
                "other": 0,
            },
            "patch_location": "apps only",
            "per_app_analysis": per_app,
        }
        if args.dup_sites:  # DEPRECATED — cross-app dedup over merged patches
            merged_hunks = []
            for p_ in patches:
                app_id = p_.parent.name
                for h in parse_hunks(p_.read_text(errors="ignore")):
                    h["file"] = f"tasks/{app_id}/{h['file']}"
                    merged_hunks.append(h)
            merged_kw = {k: v for k, v in dup_kw.items() if k != "compute_dups"}
            analysis.update(duplicate_edit_sites(merged_hunks, **merged_kw))

    out = {
        "maint_tag": args.maint_tag,
        "protocol": run_results.get("protocol"),
        "source_tag": run_results.get("source_tag"),
        "cost": run_results.get("cost"),
        "static_checks": run_results.get("static_checks")
                         or {k: v.get("static_checks")
                             for k, v in (run_results.get("per_app") or {}).items()},
        **analysis,
    }
    out_path = out_dir / "maintenance_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")
    print(json.dumps({k: out[k] for k in
                      ("touched_files", "patch_location", "added_loc_by_location",
                       "duplicate_edit_sites", "redundant_edits",
                       "shared_call_sites", "candidate_pairs")
                      if k in out}, indent=2))
    if judge is not None:
        print(f"judge: {judge.calls} new calls, ${judge.cost_usd:.4f}")


if __name__ == "__main__":
    main()
