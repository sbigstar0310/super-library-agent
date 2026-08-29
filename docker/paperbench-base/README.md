# paperbench-base

Runtime image for every PaperBench agent. Python 3.12 plus torch (CPU) and the numerical, plotting
and IO packages listed in the `pip install` block of the `Dockerfile`; the same file explains the
mount layout and the conda-path stubs.

```bash
docker build -t paperbench-base docker/paperbench-base
```

Domain frameworks (transformers, stable-baselines3, timm, diffusers, gymnasium, and the rest) are
absent on purpose. A framework that already implements a paper's method removes the repeated
implementation the shared library exists to capture.

The package set is the whole gate. `mswe_agents/paperbench/*.py` passes `--network=none` to
`docker run`, so an agent cannot `pip install` its way around a missing package, and no
forbidden-list validator has to be maintained.

That set is pinned in two places: this `Dockerfile` and `_ALLOWED_RAW` in
`el-agent/src/utils/whitelist.py`. The agent prompt reads `whitelist.py`, so edit it first, mirror
the change here, and rebuild.
