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
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29531}"

RUN_ROOT="${RUN_ROOT:-/home/liuyue/LaRA-VLA/playground/Checkpoints/OFT}"
RUN_ID="${RUN_ID:-libero_all_laravla_oft_action}"
DATA_ROOT="${DATA_ROOT:-/mnt/oss/liuyue/wangbo/data/libero_cot}"
BASE_VLM="${BASE_VLM:-/home/liuyue/LaRA-VLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
FAST_TOKENIZER="${FAST_TOKENIZER:-/home/liuyue/LaRA-VLA/playground/Pretrained_models/fast}"
VLM_CKPT="${VLM_CKPT:-}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-40000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-14}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20000000}"
ACTION_LR="${ACTION_LR:-1.0e-4}"
QWEN_LR="${QWEN_LR:-1.0e-5}"
BASE_LR="${BASE_LR:-3.0e-5}"

OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}" trainlog
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-trainlog/${RUN_ID}_${TIMESTAMP}.log}"

args=(
  --config_yaml laravla/config/training/libero.yaml
  --run_root_dir "${RUN_ROOT}"
  --run_id "${RUN_ID}"
  --wandb_project laravla_oft_libero
  --datasets.vla_data.data_root_dir "${DATA_ROOT}"
  --datasets.vla_data.data_mix libero_all
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --datasets.vla_data.bridge_annotations.fast_tokenizer_name "${FAST_TOKENIZER}"
  --datasets.vla_data.bridge_annotations.write_steps_cache false
  --datasets.vla_data.bridge_reasoning.stage 4
  --datasets.vla_data.bridge_reasoning.include_action_tokens false
  --datasets.vla_data.bridge_reasoning.include_img_next false
  --datasets.vla_data.bridge_reasoning.component_order SUBTASK,BBOX,REASON
  --framework.name QwenOFT
  --framework.training_stage action_only
  --framework.qwenvl.base_vlm "${BASE_VLM}"
  --framework.img_next.enable false
  --framework.img_next.loss_weight 0
  --framework.img_next.use_teacher false
  --framework.latent_reasoning.vlm_loss_weight 0
  --framework.action_model.action_model_type MLP
  --framework.action_model.action_horizon 8
  --framework.action_model.future_action_window_size 7
  --framework.action_model.past_action_window_size 0
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.eval_interval "${EVAL_INTERVAL}"
  --trainer.logging_frequency "${LOGGING_FREQUENCY}"
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
  --trainer.learning_rate.base "${BASE_LR}"
  --trainer.learning_rate.qwen_vl_interface "${QWEN_LR}"
  --trainer.learning_rate.action_model "${ACTION_LR}"
  --trainer.freeze_modules qwen_vl_interface
)

if [[ -n "${VLM_CKPT}" ]]; then
  args+=(
    --trainer.pretrained_checkpoint "${VLM_CKPT}"
    --trainer.reload_modules qwen_vl_interface
  )
fi

cmd=(
  torchrun
  --nproc_per_node="${NPROC}"
  --master_port="${MASTER_PORT}"
  laravla/training/train.py
  "${args[@]}"
)

{
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf 'cd %q\n' "${REPO_DIR}"
  printf 'source /home/liuyue/miniconda3/etc/profile.d/conda.sh\n'
  printf 'conda activate %q\n' "${CONDA_ENV:-starvla}"
  printf 'export CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'export CUDA_HOME=%q\n' "${CUDA_HOME}"
  printf 'export PATH=%q\n' "${PATH}"
  printf 'export LD_LIBRARY_PATH=%q\n' "${LD_LIBRARY_PATH}"
  printf 'export WANDB_MODE=%q\n' "${WANDB_MODE}"
  printf 'export TOKENIZERS_PARALLELISM=false\n'
  printf '%q ' "${cmd[@]}"
  printf '"$@"\n'
} > "${OUTPUT_DIR}/run_command.sh"
chmod +x "${OUTPUT_DIR}/run_command.sh"

echo "Logging to ${LOG_PATH}"
"${cmd[@]}" "$@" 2>&1 | tee -a "${LOG_PATH}"
