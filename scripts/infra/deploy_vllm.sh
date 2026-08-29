#!/bin/bash
set -e

# Reference model server for the MDL metric. Create the env once with
# scripts/infra/setup_vllm_env.sh, then check the GPU block below.
#
# dtype is float32 and should stay that way. The paper's MDL numbers were
# measured in fp32; fp16 shifts them by +5.8-9.8%, which is larger than the
# gaps between arms, so an fp16 server does not reproduce the table. bf16 is
# not an option on Turing. Everything else here is tuning you should adjust.

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${VLLM_CONDA_ENV:-vllm}"

# ---------------------------------------------------------------------------
# GPUs. These three have to agree: the device list, the tensor-parallel size,
# and what your cards can hold. Ours is 4x RTX 2080 Ti (Turing, 11 GB).
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"

# Turing has no FlashAttention2 (needs sm_80+), and the default FlashInfer
# backend JIT-compiles at first use, which fails without nvcc. On Ampere or
# newer you can drop this line.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"   # native context, no YaRN

# fp32 weights are ~29 GB, so TP=4 puts ~7.3 GB on each 11 GB card before the
# KV cache. 0.82 is what fits. --max-num-batched-tokens matters more than it
# looks: get_mdl.py asks for prompt_logprobs, and in fp32 each chunk
# materializes a chunk x 152k-vocab logit tensor, so the default chunk size
# OOMs the engine on cards this size. Raise both if you have more memory.
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.82}"
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-1024}"

vllm serve "${VLLM_MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size "${VLLM_TP_SIZE}" \
  --dtype float32 \
  --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${VLLM_MAX_BATCHED_TOKENS}" \
  --max-num-seqs 4 \
  --enforce-eager
