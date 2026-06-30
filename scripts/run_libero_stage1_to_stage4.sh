#!/usr/bin/env bash
# Libero-all :: Stage 1 explicit-CoT VLM -> Stage 4 full latent VLM.
# Both stages use train.py, so latent thinking tokens are reserved from Stage 1.
# img_next is disabled by default to keep checkpoint shapes stable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f /home/liuyue/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/liuyue/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-starvla}"
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  elif [[ -x /usr/local/cuda-12.9/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.9
  elif command -v nvcc >/dev/null 2>&1; then
    CUDA_BIN="$(command -v nvcc)"
    export CUDA_HOME="$(cd "$(dirname "$(dirname "${CUDA_BIN}")")" && pwd)"
  fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ===================== Edit/override as needed =====================
CONFIG_YAML="${CONFIG_YAML:-laravla/config/training/libero.yaml}"
RUN_ROOT="${RUN_ROOT:-/home/liuyue/starVLA/playground/Checkpoints/LiberoVLM_Stage1To4}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-libero_vlm_s1_to_s4}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29514}"
WANDB_PROJECT="${WANDB_PROJECT:-libero_vlm_s1_to_s4}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# Non-empty => only reload selected modules; empty => load the full model.
RELOAD_MODULES="${RELOAD_MODULES:-}"
STEPS_CACHE_PATH="${STEPS_CACHE_PATH:-${RUN_ROOT}/steps_cache/libero_vlm_s1_to_s4}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
FAST_TOKENIZER_PATH="${FAST_TOKENIZER_PATH:-/home/liuyue/starVLA/playground/Pretrained_models/fast}"

declare -A BRIDGE_STAGE=(
  [1]=1
  [4]=4
)

declare -A VLM_LOSS_WEIGHT=(
  [1]="${VLM_LOSS_WEIGHT_STAGE1:-1.0}"
  [4]="${VLM_LOSS_WEIGHT_STAGE4:-1.0}"
)

declare -A PER_DEVICE_BATCH=(
  [1]="${PER_DEVICE_BATCH_STAGE1:-12}"
  [4]="${PER_DEVICE_BATCH_STAGE4:-16}"
)

declare -A MAX_STEPS=(
  [1]="${MAX_STEPS_STAGE1:-5000}"
  [4]="${MAX_STEPS_STAGE4:-2000}"
)

# Must divide MAX_STEPS for the default checkpoint path to exist.
declare -A SAVE_INTERVAL=(
  [1]="${SAVE_INTERVAL_STAGE1:-5000}"
  [4]="${SAVE_INTERVAL_STAGE4:-2000}"
)

declare -A CKPT_STEP=(
  [1]="${CKPT_STEP_STAGE1:-${MAX_STEPS[1]}}"
  [4]="${CKPT_STEP_STAGE4:-${MAX_STEPS[4]}}"
)

START_STAGE="${START_STAGE:-1}"
EXTRA_ARGS=("$@")
# ================================================================

mkdir -p "${STEPS_CACHE_PATH}"

run_one_stage() {
  local stage="$1"
  local load_ckpt="$2"

  local run_id="${RUN_ID_PREFIX}_stage_${stage}"
  local out="${RUN_ROOT}/${run_id}"

  mkdir -p "${out}"
  cp "$0" "${out}/run_script_snapshot.sh"

  local args=(
    --config_yaml "${CONFIG_YAML}"
    --run_root_dir "${RUN_ROOT}"
    --run_id "${run_id}"
    --wandb_project "${WANDB_PROJECT}"
    --framework.training_stage reasoning_only
    --datasets.vla_data.bridge_reasoning.stage "${BRIDGE_STAGE[$stage]}"
    --datasets.vla_data.bridge_reasoning.include_img_next false
    --trainer.max_train_steps "${MAX_STEPS[$stage]}"
    --trainer.save_interval "${SAVE_INTERVAL[$stage]}"
    --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH[$stage]}"
    --framework.latent_reasoning.vlm_loss_weight "${VLM_LOSS_WEIGHT[$stage]}"
    --framework.img_next.enable false
    --framework.img_next.loss_weight 0
    --framework.img_next.use_teacher false
    --datasets.vla_data.bridge_annotations.steps_cache_path "${STEPS_CACHE_PATH}"
    --datasets.vla_data.bridge_annotations.write_steps_cache true
    --datasets.vla_data.bridge_annotations.fast_tokenizer_name "${FAST_TOKENIZER_PATH}"
  )
  [[ -n "${WANDB_ENTITY}" ]] && args+=( --wandb_entity "${WANDB_ENTITY}" )

  if [[ -n "${load_ckpt}" ]]; then
    args+=( --trainer.pretrained_checkpoint "${load_ckpt}" )
    [[ -n "${RELOAD_MODULES}" ]] && args+=( --trainer.reload_modules "${RELOAD_MODULES}" )
  fi

  local launch=(
    torchrun
    --nproc_per_node="${NUM_GPUS}"
    --master_port="${MASTER_PORT}"
    laravla/training/train.py
  )

  local cmd_tmp
  cmd_tmp="$(mktemp "/tmp/${run_id}.run_command.XXXXXX")"
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "${REPO_ROOT}"
    printf '%q ' "${launch[@]}" "${args[@]}" "${EXTRA_ARGS[@]}"
    printf '\n'
  } > "${cmd_tmp}"
  cp "${cmd_tmp}" "${out}/run_command.sh" 2>/dev/null || true
  rm -f "${cmd_tmp}"
  chmod +x "${out}/run_command.sh" 2>/dev/null || true

  "${launch[@]}" "${args[@]}" "${EXTRA_ARGS[@]}"
}

default_stage1_ckpt="${RUN_ROOT}/${RUN_ID_PREFIX}_stage_1/checkpoints/steps_${CKPT_STEP[1]}_pytorch_model.pt"
stage1_ckpt="${STAGE1_CKPT:-${default_stage1_ckpt}}"

case "${START_STAGE}" in
  1)
    run_one_stage 1 ""
    if [[ ! -f "${stage1_ckpt}" ]]; then
      echo "Missing Stage 1 checkpoint: ${stage1_ckpt}" >&2
      exit 1
    fi
    run_one_stage 4 "${stage1_ckpt}"
    ;;
  4)
    if [[ ! -f "${stage1_ckpt}" ]]; then
      echo "Missing Stage 1 checkpoint: ${stage1_ckpt}" >&2
      exit 1
    fi
    run_one_stage 4 "${stage1_ckpt}"
    ;;
  *)
    echo "START_STAGE must be 1 or 4, got: ${START_STAGE}" >&2
    exit 2
    ;;
esac
