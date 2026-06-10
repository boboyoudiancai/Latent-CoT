#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/liuyue/LaRA-VLA}"
cd "${REPO_DIR}"

source /home/liuyue/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-starvla}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

NPROC="${NPROC:-8}"
VLM_MASTER_PORT="${VLM_MASTER_PORT:-29518}"
ACTION_MASTER_PORT="${ACTION_MASTER_PORT:-29519}"

RUN_ROOT="${RUN_ROOT:-/home/liuyue/starVLA/playground/Checkpoints/ExplicitCOT}"
VLM_RUN_ID="${VLM_RUN_ID:-libero_all_explicit_cot_vlm_8k}"
ACTION_RUN_ID="${ACTION_RUN_ID:-explicit_cot_action_gen}"
DATA_ROOT="${DATA_ROOT:-/home/liuyue/LaRA-VLA/data/libero_lerobot}"
BASE_VLM="${BASE_VLM:-/home/liuyue/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
FAST_TOKENIZER="${FAST_TOKENIZER:-/home/liuyue/starVLA/playground/Pretrained_models/fast}"

VLM_STEPS="${VLM_STEPS:-8000}"
ACTION_STEPS="${ACTION_STEPS:-40000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-14}"
EXPLICIT_COT_MAX_NEW_TOKENS="${EXPLICIT_COT_MAX_NEW_TOKENS:-128}"
ACTION_CONTEXT_MAX_TOKENS="${ACTION_CONTEXT_MAX_TOKENS:-320}"

COMMON_ARGS=(
  --config_yaml laravla/config/training/libero.yaml
  --datasets.vla_data.data_root_dir "${DATA_ROOT}"
  --datasets.vla_data.data_mix libero_all
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --datasets.vla_data.bridge_annotations.fast_tokenizer_name "${FAST_TOKENIZER}"
  --datasets.vla_data.bridge_annotations.filters.require_cot_episode true
  --datasets.vla_data.bridge_reasoning.stage 1
  --datasets.vla_data.bridge_reasoning.as_solution true
  --datasets.vla_data.bridge_reasoning.include_bbox true
  --datasets.vla_data.bridge_reasoning.include_action_tokens false
  --datasets.vla_data.bridge_reasoning.include_img_next false
  --datasets.vla_data.bridge_reasoning.component_order SUBTASK,BBOX,REASON
  --framework.qwenvl.base_vlm "${BASE_VLM}"
  --framework.img_next.enable false
  --framework.img_next.loss_weight 0
  --framework.img_next.use_teacher false
  --framework.latent_reasoning.compute_language_loss true
  --framework.action_model.diffusion_model_cfg.dropout 0.1
)

torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${VLM_MASTER_PORT}" \
  laravla/training/train_explicit_cot.py \
  "${COMMON_ARGS[@]}" \
  --run_root_dir "${RUN_ROOT}" \
  --run_id "${VLM_RUN_ID}" \
  --wandb_project explicit_cot_libero \
  --framework.training_stage reasoning_only \
  --framework.latent_reasoning.vlm_loss_weight 1.0 \
  --trainer.pretrained_checkpoint null \
  --trainer.reload_modules null \
  --trainer.max_train_steps "${VLM_STEPS}" \
  --trainer.save_interval 4000 \
  --trainer.eval_interval 20000000 \
  --trainer.logging_frequency 10 \
  --trainer.min_save_step 0 \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.learning_rate.base 1.0e-5 \
  --trainer.learning_rate.qwen_vl_interface 1.0e-5 \
  --trainer.learning_rate.action_model 1.0e-4

VLM_FINAL="${RUN_ROOT}/${VLM_RUN_ID}/final_model/pytorch_model.pt"
if [[ ! -f "${VLM_FINAL}" ]]; then
  echo "Missing VLM final model: ${VLM_FINAL}" >&2
  exit 1
fi

torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${ACTION_MASTER_PORT}" \
  laravla/training/train_explicit_cot.py \
  "${COMMON_ARGS[@]}" \
  --run_root_dir "${RUN_ROOT}" \
  --run_id "${ACTION_RUN_ID}" \
  --wandb_project explicit_cot_libero \
  --framework.training_stage full \
  --framework.explicit_cot.action_input generated \
  --framework.explicit_cot.max_new_tokens "${EXPLICIT_COT_MAX_NEW_TOKENS}" \
  --framework.explicit_cot.action_context_max_tokens "${ACTION_CONTEXT_MAX_TOKENS}" \
  --framework.latent_reasoning.vlm_loss_weight 0 \
  --trainer.pretrained_checkpoint "${VLM_FINAL}" \
  --trainer.reload_modules qwen_vl_interface \
  --trainer.max_train_steps "${ACTION_STEPS}" \
  --trainer.save_interval 4000 \
  --trainer.eval_interval 20000000 \
  --trainer.logging_frequency 10 \
  --trainer.min_save_step 16000 \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.learning_rate.base 3.0e-5 \
  --trainer.learning_rate.qwen_vl_interface 1.0e-5 \
  --trainer.learning_rate.action_model 1.0e-4
