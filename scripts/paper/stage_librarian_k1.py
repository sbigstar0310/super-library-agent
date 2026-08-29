#!/usr/bin/env python3
"""Stage Librarian K=1 (= sample_1, no best-of-K selection) as tag-shaped trees
so the normal --backup-tag/--phase/--round metric interface can read them.

Writes into backups/webgen-k1/ (a new sibling bench dir, nothing existing is
touched):

    librarian-c{c}-t{t}-s1     <- final/round_1/samples/sample_1   (the K=1 arm)
    librarian-c{c}-t{t}-k8ctl  <- final/round_1/apply              (CONTROL)

The control is staged by the identical procedure from the already-published
K=8 portfolio; if the control reproduces the stored K=8 metrics, the staging
procedure is sound and the K=1 numbers can be trusted.
"""
import os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = f"{ROOT}/backups/webgen-rb"
DST = f"{ROOT}/backups/webgen-k1"
CL, TR = (2, 5, 13), (1, 2, 3)


def stage(src_phase_dir, dst_tag):
    """Copy a {lib/, tasks/<id>/submission/} portfolio into tag shape and
    create the per-task lib mirror that the metric scripts require
    (CLAUDE.md gotcha 4: no mirror -> library metrics read 0)."""
    dst = f"{DST}/{dst_tag}/final/round_1/apply"
    if os.path.exists(f"{DST}/{dst_tag}"):
        shutil.rmtree(f"{DST}/{dst_tag}")
    os.makedirs(dst)
    lib_src = f"{src_phase_dir}/lib"
    shutil.copytree(lib_src, f"{dst}/lib", symlinks=True,
                    ignore_dangling_symlinks=True)
    os.makedirs(f"{dst}/tasks")
    n = 0
    for tid in sorted(os.listdir(f"{src_phase_dir}/tasks")):
        # `__library__` is the extract agent's scratch dir, not a task
        if tid == "__library__":
            continue
        sub = f"{src_phase_dir}/tasks/{tid}/submission"
        if not os.path.isdir(sub):
            print(f"  [skip] {tid}: no submission/")
            continue
        os.makedirs(f"{dst}/tasks/{tid}")
        shutil.copytree(sub, f"{dst}/tasks/{tid}/submission", symlinks=True,
                        ignore_dangling_symlinks=True)
        shutil.copytree(lib_src, f"{dst}/tasks/{tid}/lib", symlinks=True,
                        ignore_dangling_symlinks=True)
        n += 1
    return n


total = 0
for c in CL:
    for t in TR:
        base = f"{SRC}/librarian-c{c}-t{t}/final/round_1"
        n1 = stage(f"{base}/samples/sample_1", f"librarian-c{c}-t{t}-s1")
        n2 = stage(f"{base}/apply", f"librarian-c{c}-t{t}-k8ctl")
        print(f"c{c}-t{t}: K1 {n1} apps, K8-control {n2} apps")
        total += n1 + n2
print(f"staged {total} portfolios' worth of apps under backups/webgen-k1/")
