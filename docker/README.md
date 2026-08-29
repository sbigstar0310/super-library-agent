# Agent images

Every SLA agent runs in a container, in training and in evaluation alike.
`el-agent/src/mswe_agents/_factory.py:build_environment()` returns a `DockerEnvironment` whenever
`--docker-image` is set, and the run scripts always set it. Build the image for your benchmark
before the first run.

```bash
docker build -t sla-base        docker/sla-base         # WebGen
docker build -t paperbench-base docker/paperbench-base  # PaperBench
```

`sla-base` carries the Node and vite toolchain, so WebGen apps install, build and serve entirely
inside the container and the host needs no Node. `scripts/eval/eval_webgen.sh` builds it if it is
missing; nothing else self-builds.

`cc-sandbox` (node:20-slim plus a pinned Claude Code CLI) is used only by the layout-spec generator
in `scripts/layout_specs/` and by the Claude Code baseline in `baselines/claudecode/`. Build it only
if you run one of those:

```bash
docker build -t cc-sandbox docker/cc-sandbox
```

`bash docker/build.sh <image-tag>` does the same build for any subdirectory here that holds a
`Dockerfile`.
