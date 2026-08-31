#!/usr/bin/env bash
# Bridge-LeRobot :: four-stage curriculum training
# - Stage 1: no pretrained checkpoint by default
# - Stage 2-4: load the previous stage checkpoint
# - train.py: when training_stage != full, bridge_reasoning.stage is not overridden
# Repository root: bash scripts/run_bridge_multistage.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/qwen_cache}"

# ===================== edit as needed =====================
CONFIG_YAML="${CONFIG_YAML:-laravla/config/training/bridge.yaml}"
RUN_ROOT="${RUN_ROOT:-results/BridgeLeRobot_VLM}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-bridge_multistage}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29512}"
WANDB_PROJECT="${WANDB_PROJECT:-bridge_multistage_vlm}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# Non-empty => only reload selected modules; empty => load the full model.
RELOAD_MODULES="${RELOAD_MODULES:-}"
INITIAL_PRETRAINED_CKPT="${INITIAL_PRETRAINED_CKPT:-}"
CKPT_PRE="${CKPT_PRE:-}"
STEPS_CACHE_PATH="${STEPS_CACHE_PATH:-}"

# Optional action-model knobs kept for Bridge experiments.
USE_REASONING_SUMMARY="${USE_REASONING_SUMMARY:-false}"
REASONING_SUMMARY_TOKENS="${REASONING_SUMMARY_TOKENS:-2}"
REASONING_SUMMARY_HEADS="${REASONING_SUMMARY_HEADS:-4}"
REASONING_SUMMARY_DROPOUT="${REASONING_SUMMARY_DROPOUT:-0.1}"
USE_REASONING_FILM="${USE_REASONING_FILM:-false}"
REASONING_FILM_FIRST_K="${REASONING_FILM_FIRST_K:-0}"
REASONING_FILM_DROPOUT="${REASONING_FILM_DROPOUT:-0.1}"
REASONING_FILM_HIDDEN="${REASONING_FILM_HIDDEN:-1024}"
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}"

declare -A BRIDGE_STAGE=(
  [1]=1
  [2]=2
  [3]=3
  [4]=4
)

declare -A SCHEDULED_STAGE=(
  [1]=1
  [2]=2
  [3]=3
  [4]=4
)

declare -A TRAINING_STAGE=(
  [1]="reasoning_only"
  [2]="reasoning_only"
  [3]="reasoning_only"
  [4]="reasoning_only"
)

declare -A VLM_LOSS_WEIGHT=(
  [1]=1.0
  [2]=1.0
  [3]=1.0
  [4]=1.0
)

declare -A IMG_NEXT_LOSS_WEIGHT=(
  [1]="${IMG_NEXT_LOSS_WEIGHT_STAGE1:-0.1}"
  [2]="${IMG_NEXT_LOSS_WEIGHT_STAGE2:-0.1}"
  [3]="${IMG_NEXT_LOSS_WEIGHT_STAGE3:-0.2}"
  [4]="${IMG_NEXT_LOSS_WEIGHT_STAGE4:-0.2}"
)

declare -A PER_DEVICE_BATCH=(
  [1]=12
  [2]=16
  [3]=16
  [4]=16
)

declare -A MAX_STEPS=(
  [1]=10000
  [2]=5000
  [3]=5000
  [4]=10000
)

declare -A SAVE_INTERVAL=(
  [1]=10000
  [2]=5000
  [3]=5000
  [4]=10000
)

declare -A CKPT_STEP=(
  [1]=10000
  [2]=5000
  [3]=5000
  [4]=5000
)

START_STAGE="${START_STAGE:-1}"
# =========================================================

[[ -n "${STEPS_CACHE_PATH}" ]] && mkdir -p "${STEPS_CACHE_PATH}"

run_one_stage() {
  local stage="$1"
  local load_ckpt="$2"

  local run_id="${RUN_ID_PREFIX}_stage_${stage}"
  local out="${RUN_ROOT}/${run_id}"
  local component_order="SUBTASK,BBOX,REASON"

  mkdir -p "${out}"
  cp "$0" "${out}/run_command.sh"

  local args=(
    --config_yaml "${CONFIG_YAML}"
    --run_root_dir "${RUN_ROOT}"
    --run_id "${run_id}"
    --wandb_project "${WANDB_PROJECT}"
    --framework.training_stage "${TRAINING_STAGE[$stage]}"
    --datasets.vla_data.bridge_reasoning.stage "${BRIDGE_STAGE[$stage]}"
    --datasets.vla_data.ecot.scheduled_stage "${SCHEDULED_STAGE[$stage]}"
    --trainer.max_train_steps "${MAX_STEPS[$stage]}"
    --trainer.save_interval "${SAVE_INTERVAL[$stage]}"
    --trainer.eval_interval 50000000
    --trainer.logging_frequency 20
    --trainer.warmup_ratio 0.1
    --trainer.learning_rate.base 3.0e-5
    --trainer.learning_rate.action_model 1.0e-4
    --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH[$stage]}"
    --datasets.vla_data.bridge_reasoning.include_action_tokens "true"
    --datasets.vla_data.bridge_reasoning.component_order "${component_order}"
    --framework.latent_reasoning.vlm_loss_weight "${VLM_LOSS_WEIGHT[$stage]}"
    --framework.img_next.loss_weight "${IMG_NEXT_LOSS_WEIGHT[$stage]}"
    --framework.action_model.use_reasoning_film "${USE_REASONING_FILM}"
    --framework.action_model.reasoning_film_first_k "${REASONING_FILM_FIRST_K}"
    --framework.action_model.reasoning_film_dropout "${REASONING_FILM_DROPOUT}"
    --framework.action_model.reasoning_film_hidden "${REASONING_FILM_HIDDEN}"
    --framework.action_model.use_reasoning_summary "${USE_REASONING_SUMMARY}"
    --framework.action_model.reasoning_summary_tokens "${REASONING_SUMMARY_TOKENS}"
    --framework.action_model.reasoning_summary_heads "${REASONING_SUMMARY_HEADS}"
    --framework.action_model.reasoning_summary_dropout "${REASONING_SUMMARY_DROPOUT}"
    --framework.qwenvl.attn_implementation "${ATTENTION_IMPLEMENTATION}"
  )

  [[ -n "${STEPS_CACHE_PATH}" ]] && args+=( --datasets.vla_data.bridge_annotations.steps_cache_path "${STEPS_CACHE_PATH}" )
  [[ -n "${WANDB_ENTITY}" ]] && args+=( --wandb_entity "${WANDB_ENTITY}" )

  if [[ -n "${load_ckpt}" ]]; then
    args+=( --trainer.pretrained_checkpoint "${load_ckpt}" )
    [[ -n "${RELOAD_MODULES}" ]] && args+=( --trainer.reload_modules "${RELOAD_MODULES}" )
  fi

  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    laravla/training/train.py \
    "${args[@]}" \
    "$@"
}

for stage in 1 2 3 4; do
  (( stage < START_STAGE )) && continue

  load_ckpt=""
  if (( stage == START_STAGE )); then
    load_ckpt="${CKPT_PRE:-${INITIAL_PRETRAINED_CKPT:-}}"
  else
    prev=$((stage - 1))
    load_ckpt="${RUN_ROOT}/${RUN_ID_PREFIX}_stage_${prev}/checkpoints/steps_${CKPT_STEP[$prev]}_pytorch_model.pt"
    if [[ ! -f "${load_ckpt}" ]]; then
      echo "Missing previous-stage checkpoint: ${load_ckpt}" >&2
      exit 1
    fi
  fi

  run_one_stage "${stage}" "${load_ckpt}"
done
