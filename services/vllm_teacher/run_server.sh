#!/usr/bin/env bash
set -euo pipefail

: "${VDT_TEACHER_API_TOKEN_FILE:?set the owner-only bearer token file}"
: "${VDT_TEACHER_PROFILE_TEACHER_IDS_SHA256:?set the pinned overlap-ID hash}"

export VLLM_PLUGINS=vdt_teacher
export VDT_TEACHER_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
export VDT_TEACHER_MODEL_REVISION="a09a35458c702b33eeacc393d103063234e8bc28"
export VDT_TEACHER_PROFILE_ID="${VDT_TEACHER_PROFILE_ID:-gemma2-qwen25-paper-v1}"
export VDT_TEACHER_TOKENIZER_VOCAB_SIZE=151665
export VDT_TEACHER_MAX_MODEL_LEN=8192
export VDT_TEACHER_PRIVATE_ONLY="${VDT_TEACHER_PRIVATE_ONLY:-1}"

exec vllm serve Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --tokenizer-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --served-model-name qwen2.5-7b-instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --generation-config vllm \
  --max-logprobs -1 \
  --download-dir "${VDT_TEACHER_MODEL_CACHE:-/workspace/models}" \
  --host 127.0.0.1 \
  --port "${VDT_TEACHER_PORT:-18000}"
