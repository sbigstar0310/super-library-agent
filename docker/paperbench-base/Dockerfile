# paperbench-base: Super-Library-Agent runtime for paperbench (all clusters).
#
# Standalone image (no upstream paperbench-base layer). Cluster-agnostic:
# the only packages pre-installed are tier-1 numerical / DL / IO primitives
# that recur across RL, Vision, and LLM paper reproductions. Domain
# frameworks (transformers, stable-baselines3, timm, open_clip, diffusers,
# tianshou, ...) are deliberately ABSENT — when these are available the
# agent imports them instead of using `lib/`, and the learning signal we
# want to measure disappears.
#
# Enforcement at runtime: the run-driver passes `--network=none` to docker
# run (see mswe_agents/paperbench/*.py mount_spec.extra_run_args), so the
# agent cannot `pip install <forbidden>` to bypass the image's package set.
#
# KEEP the pip-install list below IN SYNC WITH:
#   el-agent/src/utils/whitelist.py (_ALLOWED_RAW)
# (Single source of truth lives in whitelist.py — that's what the L1
# prompt + L3 validator read. This Dockerfile is the second pinned spot.)
#
# Build:
#   docker build --platform=linux/amd64 \
#       -f docker/paperbench-base/Dockerfile \
#       -t paperbench-base \
#       docker/paperbench-base/

FROM python:3.12-slim

# Upstream paperbench's mount layout: WORKSPACE_BASE=/home, agent edits
# /home/submission and reads /home/paper. Preserved so our mount_spec
# stays compatible.
ENV WORKSPACE_BASE=/home \
    SUBMISSION_DIR=/home/submission \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        ffmpeg \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Stub the conda-env layout the mswe-agent factory's default PATH expects
# (`/opt/conda/envs/agent/bin:/opt/conda/bin:...`). No actual conda; just
# symlinks so the agent's `python`/`pip` resolve without per-image overrides.
RUN mkdir -p /opt/conda/envs/agent/bin /opt/conda/bin \
    && ln -sf /usr/local/bin/python3 /opt/conda/envs/agent/bin/python \
    && ln -sf /usr/local/bin/python3 /opt/conda/envs/agent/bin/python3 \
    && ln -sf /usr/local/bin/pip     /opt/conda/envs/agent/bin/pip     \
    && ln -sf /usr/local/bin/python3 /opt/conda/bin/python             \
    && ln -sf /usr/local/bin/python3 /opt/conda/bin/python3

# Tier-1 cluster-agnostic baseline. Torch CPU build (no GPU in paperbench
# host) keeps image size reasonable. KEEP IN SYNC WITH whitelist.py.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch torchvision torchaudio \
        numpy scipy scikit-learn pandas numba \
        matplotlib seaborn tqdm tensorboard \
        Pillow opencv-python h5py joblib \
        PyYAML omegaconf hydra-core \
        einops absl-py cloudpickle dill rich typer click requests

RUN mkdir -p /home/submission /home/paper

WORKDIR /home
CMD ["bash"]
