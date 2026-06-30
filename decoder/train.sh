#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f /home/liuyue/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/liuyue/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-starvla}"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  elif [[ -x /usr/local/cuda-12.9/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.9
  elif command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(cd "$(dirname "$(dirname "$(command -v nvcc)")")" && pwd)"
  fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: set CHECKPOINT=/path/to/final_latent_vlm/checkpoints/steps_xxx_pytorch_model.pt" >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/playground/Checkpoints/Text2Latent/Text2Latent_decoder}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${SCRIPT_DIR}/accelerate_deepspeed_zero2.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_BATCHES="${EVAL_BATCHES:-2}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"
EVAL_DECODE_TOKENS="${EVAL_DECODE_TOKENS:-64}"

DECODER_LR="${DECODER_LR:-1e-4}"
VLM_LR="${VLM_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
GRADIENT_CLIPPING="${GRADIENT_CLIPPING:-1.0}"
DECODER_LOSS_WEIGHT="${DECODER_LOSS_WEIGHT:-1.0}"
ACTION_TOKEN_LOSS_WEIGHT="${ACTION_TOKEN_LOSS_WEIGHT:-1.0}"
DECODER_TYPE="${DECODER_TYPE:-one_block}"
DTYPE="${DTYPE:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
BASE_VLM_PATH="${BASE_VLM_PATH:-}"
DECODER_BASE_VLM_PATH="${DECODER_BASE_VLM_PATH:-}"
FAST_TOKENIZER_PATH="${FAST_TOKENIZER_PATH:-}"

mkdir -p "${OUTPUT_DIR}"

LAUNCH_ARGS=(
  --config_file "${ACCELERATE_CONFIG}"
  --num_processes "${NUM_PROCESSES}"
  --main_process_port "${MAIN_PROCESS_PORT}"
)

TRAIN_ARGS=(
  decoder/train_latent_decoder.py
  --checkpoint "${CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --save_interval "${SAVE_INTERVAL}"
  --logging_frequency "${LOGGING_FREQUENCY}"
  --eval_interval "${EVAL_INTERVAL}"
  --eval_batches "${EVAL_BATCHES}"
  --eval_samples "${EVAL_SAMPLES}"
  --eval_decode_tokens "${EVAL_DECODE_TOKENS}"
  --decoder_learning_rate "${DECODER_LR}"
  --vlm_learning_rate "${VLM_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --gradient_clipping "${GRADIENT_CLIPPING}"
  --decoder_loss_weight "${DECODER_LOSS_WEIGHT}"
  --action_token_loss_weight "${ACTION_TOKEN_LOSS_WEIGHT}"
  --decoder_type "${DECODER_TYPE}"
  --dtype "${DTYPE}"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
)

if [[ -n "${BASE_VLM_PATH}" ]]; then
  TRAIN_ARGS+=(--base_vlm "${BASE_VLM_PATH}")
fi
if [[ -n "${DECODER_BASE_VLM_PATH}" ]]; then
  TRAIN_ARGS+=(--decoder_base_vlm "${DECODER_BASE_VLM_PATH}")
fi
if [[ -n "${FAST_TOKENIZER_PATH}" ]]; then
  TRAIN_ARGS+=(--fast_tokenizer_path "${FAST_TOKENIZER_PATH}")
fi

if [[ -n "${DATA_ROOT_DIR:-}" ]]; then
  TRAIN_ARGS+=(--data_root_dir "${DATA_ROOT_DIR}")
fi
if [[ -n "${DATA_MIX:-}" ]]; then
  TRAIN_ARGS+=(--data_mix "${DATA_MIX}")
fi
if [[ -n "${STEPS_CACHE_PATH:-}" ]]; then
  TRAIN_ARGS+=(--steps_cache_path "${STEPS_CACHE_PATH}")
fi
if [[ -n "${WRITE_STEPS_CACHE:-}" ]]; then
  TRAIN_ARGS+=(--write_steps_cache "${WRITE_STEPS_CACHE}")
fi
if [[ "${FREEZE_VLM:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--freeze_vlm)
fi

printf -v ACCELERATE_LAUNCH_ARGS '%q ' "${LAUNCH_ARGS[@]}"
export ACCELERATE_LAUNCH_ARGS="${ACCELERATE_LAUNCH_ARGS% }"

echo "Launching decoder training:"
printf '  accelerate launch'
printf ' %q' "${LAUNCH_ARGS[@]}" "${TRAIN_ARGS[@]}"
printf '\n'

exec accelerate launch "${LAUNCH_ARGS[@]}" "${TRAIN_ARGS[@]}"
