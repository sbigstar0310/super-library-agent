"""Global package whitelist for paperbench submissions (all clusters).

Single source of truth used by:
  - L1 prompt: prompts/paperbench/coding_agent.py (rendered into the
    upstream-verbatim user prompt via `render_for_prompt()`)
  - Image build: docker/paperbench-base/Dockerfile (the inline pip install
    list there is manually kept in sync with _ALLOWED_RAW below; rebuild
    the `paperbench-base` image when this file changes)

Cluster-agnostic tier-1 list. Covers RL / Vision / LLM paper
reproductions with only generic numerical / DL / IO primitives. Domain
frameworks (transformers / sb3 / timm / open_clip / diffusers / ...) are
NOT installed — if they were, agents would import them instead of using
`lib/` and the SuperLibraryAgent learning signal would collapse.

Runtime enforcement: the run-driver passes `--network=none` to docker
run, so an agent cannot `pip install <anything>` to add packages
mid-run. Image absence + network=none = sufficient gate; we do not
maintain an explicit forbidden-list or post-hoc validator.
"""

from __future__ import annotations


def _norm(name: str) -> str:
    """Normalize a package name for comparison: lowercase, hyphen→underscore."""
    return name.lower().replace("-", "_")


_ALLOWED_RAW = (
    # Numerical / DL core
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "scipy",
    "scikit-learn",
    "sklearn",
    "pandas",
    "numba",
    # Plotting / progress / logging
    "matplotlib",
    "seaborn",
    "tqdm",
    "tensorboard",
    # Image / IO / config
    "Pillow",
    "PIL",
    "opencv-python",
    "cv2",
    "h5py",
    "joblib",
    "PyYAML",
    "yaml",
    "omegaconf",
    "hydra-core",
    "hydra",
    # Misc generic utilities (no domain-framework overlap)
    "einops",
    "absl-py",
    "absl",
    "cloudpickle",
    "dill",
    "rich",
    "typer",
    "click",
    "requests",
)


ALLOWED_NAMES: frozenset[str] = frozenset(_norm(n) for n in _ALLOWED_RAW)


def is_allowed(top_level: str) -> bool:
    return _norm(top_level) in ALLOWED_NAMES


def render_for_prompt() -> str:
    """Return a markdown-formatted DEPENDENCIES section for the L1 prompt."""
    allowed_display = sorted({n for n in _ALLOWED_RAW}, key=str.lower)
    return (
        "DEPENDENCIES\n"
        "---\n"
        "This task runs in a sandboxed environment with a fixed package "
        "set and no network access. Only the following packages are "
        "available; you cannot `pip install` new dependencies and cannot "
        "clone external repositories.\n\n"
        "**Available packages**: "
        + ", ".join(f"`{n}`" for n in allowed_display)
        + ".\n\n"
        "If you need domain machinery (RL training loops, transformer "
        "model loaders, vision foundation models, diffusion schedulers, "
        "...), prefer the provided `lib/` (when present) or implement it "
        "from scratch on top of torch + numpy. The standard library plus "
        "the available packages cover everything else."
    )
