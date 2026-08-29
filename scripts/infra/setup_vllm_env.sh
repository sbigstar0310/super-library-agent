#!/bin/bash
set -euo pipefail

# Create the conda env `deploy_vllm.sh` expects. Run once.
#
#   bash scripts/infra/setup_vllm_env.sh
#
# Only the MDL metric needs this. Everything else in the repo runs from
# el-agent/.venv via uv, and you can skip this entirely if you are not
# computing MDL.
#
# The GPU side is yours to arrange: deploy_vllm.sh defaults to four devices
# with tensor-parallel-size 4, which is what our 2080 Ti box needed. Edit
# CUDA_VISIBLE_DEVICES and --tensor-parallel-size there to match your hardware
# — they have to agree with each other, and vLLM will fail at startup if the
# device count and the parallel size disagree. Leave --dtype at float32; the
# paper's MDL numbers were measured there and fp16 does not reproduce them.

ENV_NAME="${VLLM_CONDA_ENV:-vllm}"
PY_VERSION="${VLLM_PY_VERSION:-3.12}"
VLLM_VERSION="${VLLM_VERSION:-0.13.0}"

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "conda not found at ${CONDA_BASE}. Install it, or set CONDA_BASE." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "env '${ENV_NAME}' already exists; installing/updating vllm==${VLLM_VERSION} into it."
else
  echo "creating env '${ENV_NAME}' (python ${PY_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi

conda activate "${ENV_NAME}"
pip install --upgrade pip
pip install "vllm==${VLLM_VERSION}"

echo
python - <<'PY'
import torch, vllm
print(f"vllm  {vllm.__version__}")
print(f"torch {torch.__version__} (cuda {torch.version.cuda})")
if not torch.cuda.is_available():
    raise SystemExit("torch reports no CUDA device — vLLM will not serve.")
n = torch.cuda.device_count()
print(f"visible GPUs: {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  sm_{p.major}{p.minor}  {p.total_memory/2**30:.0f} GiB")
if n and torch.cuda.get_device_properties(0).major < 8:
    print(
        "\nPre-Ampere (sm_80) device: no bf16 and no FlashAttention2.\n"
        "deploy_vllm.sh sets VLLM_ATTENTION_BACKEND=TRITON_ATTN for this case;\n"
        "on Ampere or newer you can drop it and gain throughput. The float32\n"
        "dtype is not part of that workaround — keep it on every GPU, or the\n"
        "MDL numbers will not match the paper."
    )
PY

echo
echo "done. next: bash scripts/infra/deploy_vllm.sh   (check its GPU settings first)"
