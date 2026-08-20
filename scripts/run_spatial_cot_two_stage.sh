#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/liuyue/LaRA-VLA}"
cd "${REPO_ROOT}"

source /home/liuyue/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-starvla}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda-12.9 /usr/local/cuda-12.1 /usr/local/cuda-12 /usr/local/cuda; do
    if [[ -d "${candidate}" ]]; then
      export CUDA_HOME="${candidate}"
      break
    fi
  done
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

NPROC="${NPROC:-6}"
START_STAGE="${START_STAGE:-1}"
STAGE1_MASTER_PORT="${STAGE1_MASTER_PORT:-29671}"
STAGE2_MASTER_PORT="${STAGE2_MASTER_PORT:-29655}"

DATA_ROOT="${DATA_ROOT:-/mnt/oss/liuyue/wangbo/data/libero_cot}"
BASE_VLM="${BASE_VLM:-${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
FAST_TOKENIZER="${FAST_TOKENIZER:-${REPO_ROOT}/playground/Pretrained_models/fast}"
STEPS_CACHE="${STEPS_CACHE:-${REPO_ROOT}/playground/Checkpoints/Text2Latent/steps_cache/libero_vlm_spatial_cot}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/trainlog}"

STAGE1_ROOT="${STAGE1_ROOT:-${REPO_ROOT}/playground/Checkpoints/ExplicitCOT}"
STAGE1_RUN_ID="${STAGE1_RUN_ID:-libero_all_explicit_spatial_cot_vlm_8k}"
STAGE1_DIR="${STAGE1_DIR:-${STAGE1_ROOT}/${STAGE1_RUN_ID}}"

ADAPTER_DIR="${ADAPTER_DIR:-${REPO_ROOT}/playground/Checkpoints/Text2Latent/ExplicitSpatialCOT_libero_all_8k_as_latent_stage4}"
STAGE2_DIR="${STAGE2_DIR:-${REPO_ROOT}/playground/Checkpoints/Text2Latent/Text2Latent_full_qwen4b_decoder_explicit_spatialcot8k_stage1_to_stage4_no_actiontok_no_imgnext_decoderloss5k}"

STAGE1_STEPS="${STAGE1_STEPS:-8000}"
STAGE1_PER_DEVICE_BATCH="${STAGE1_PER_DEVICE_BATCH:-18}"
STAGE2_STEPS="${STAGE2_STEPS:-5000}"
STAGE2_PER_DEVICE_BATCH="${STAGE2_PER_DEVICE_BATCH:-9}"
STAGE2_GRAD_ACCUM="${STAGE2_GRAD_ACCUM:-2}"

mkdir -p "${LOG_DIR}" "${STEPS_CACHE}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STAGE1_LOG="${STAGE1_LOG:-${LOG_DIR}/${STAGE1_RUN_ID}_${TIMESTAMP}.log}"
STAGE2_LOG="${STAGE2_LOG:-${LOG_DIR}/$(basename "${STAGE2_DIR}")_${TIMESTAMP}.log}"

save_pipeline_command() {
  local output_dir="$1"
  mkdir -p "${output_dir}"
  cp "$0" "${output_dir}/pipeline_run_command.sh"
  chmod +x "${output_dir}/pipeline_run_command.sh" 2>/dev/null || true
}

run_stage1() {
  save_pipeline_command "${STAGE1_DIR}"
  cp "$0" "${STAGE1_DIR}/run_command.sh"
  chmod +x "${STAGE1_DIR}/run_command.sh" 2>/dev/null || true

  torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${STAGE1_MASTER_PORT}" \
    laravla/training/train_explicit_cot.py \
    --config_yaml laravla/config/training/libero.yaml \
    --run_root_dir "${STAGE1_ROOT}" \
    --run_id "${STAGE1_RUN_ID}" \
    --wandb_project explicit_spatial_cot_libero \
    --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
    --datasets.vla_data.data_mix libero_all \
    --datasets.vla_data.per_device_batch_size "${STAGE1_PER_DEVICE_BATCH}" \
    --datasets.vla_data.bridge_annotations.cot_path annotations/episode_dense_captions_full.jsonl \
    --datasets.vla_data.bridge_annotations.bbox_path annotations/episode_sam3_bboxes_from_dino_final.jsonl \
    --datasets.vla_data.bridge_annotations.spatial_cot_path annotations/episode_spatial_cot.jsonl \
    --datasets.vla_data.bridge_annotations.fast_tokenizer_name "${FAST_TOKENIZER}" \
    --datasets.vla_data.bridge_annotations.steps_cache_path "${STEPS_CACHE}" \
    --datasets.vla_data.bridge_annotations.write_steps_cache true \
    --datasets.vla_data.bridge_annotations.filters.require_cot_episode true \
    --datasets.vla_data.bridge_annotations.filters.require_spatial_cot_episode true \
    --datasets.vla_data.bridge_annotations.filters.require_bbox_episode true \
    --datasets.vla_data.bridge_annotations.filters.require_bbox_step true \
    --datasets.vla_data.bridge_annotations.filters.require_spatial_cot_step true \
    --datasets.vla_data.bridge_reasoning.stage 1 \
    --datasets.vla_data.bridge_reasoning.as_solution true \
    --datasets.vla_data.bridge_reasoning.include_bbox true \
    --datasets.vla_data.bridge_reasoning.include_action_tokens false \
    --datasets.vla_data.bridge_reasoning.include_img_next false \
    --datasets.vla_data.bridge_reasoning.component_order SUBTASK,BBOX,SPATIAL,REASON \
    --datasets.vla_data.bridge_reasoning.tag2think_count.SUBTASK 1 \
    --datasets.vla_data.bridge_reasoning.tag2think_count.BBOX 1 \
    --datasets.vla_data.bridge_reasoning.tag2think_count.SPATIAL 1 \
    --datasets.vla_data.bridge_reasoning.tag2think_count.REASON 1 \
    --framework.qwenvl.base_vlm "${BASE_VLM}" \
    --framework.qwenvl.attn_implementation flash_attention_2 \
    --framework.training_stage reasoning_only \
    --framework.img_next.enable false \
    --framework.img_next.loss_weight 0 \
    --framework.img_next.use_teacher false \
    --framework.latent_reasoning.compute_language_loss true \
    --framework.latent_reasoning.vlm_loss_weight 1.0 \
    --trainer.pretrained_checkpoint null \
    --trainer.reload_modules null \
    --trainer.max_train_steps "${STAGE1_STEPS}" \
    --trainer.save_interval 4000 \
    --trainer.eval_interval 20000000 \
    --trainer.logging_frequency 10 \
    --trainer.min_save_step 0 \
    --trainer.gradient_accumulation_steps 1 \
    --trainer.enable_gradient_checkpointing true \
    --trainer.warmup_ratio 0.1 \
    --trainer.learning_rate.base 1.0e-5 \
    --trainer.learning_rate.qwen_vl_interface 1.0e-5 \
    --trainer.learning_rate.action_model 1.0e-4 \
    2>&1 | tee -a "${STAGE1_LOG}"
}

prepare_stage4_adapter() {
  local source_checkpoint="${STAGE1_DIR}/final_model/pytorch_model.pt"
  local source_config="${STAGE1_DIR}/config.yaml"
  local adapter_checkpoint="${ADAPTER_DIR}/final_model/pytorch_model.pt"

  [[ -f "${source_checkpoint}" ]] || { echo "Missing Stage 1 checkpoint: ${source_checkpoint}" >&2; exit 1; }
  [[ -f "${source_config}" ]] || { echo "Missing Stage 1 config: ${source_config}" >&2; exit 1; }

  mkdir -p "${ADAPTER_DIR}/final_model"
  cp --reflink=auto "${source_checkpoint}" "${adapter_checkpoint}"

  python - "${source_config}" "${ADAPTER_DIR}/config.yaml" "${ADAPTER_DIR}" \
    "${STAGE2_PER_DEVICE_BATCH}" "${STAGE2_GRAD_ACCUM}" <<'PY'
import sys
from omegaconf import OmegaConf

source, destination, output_dir, per_device_batch, grad_accum = sys.argv[1:]
cfg = OmegaConf.load(source)
cfg.run_id = "ExplicitSpatialCOT_libero_all_8k_as_latent_stage4"
cfg.run_root_dir = str(output_dir.rsplit("/", 1)[0])
cfg.output_dir = output_dir
cfg.datasets.vla_data.per_device_batch_size = int(per_device_batch)
cfg.datasets.vla_data.bridge_reasoning.stage = 4
cfg.datasets.vla_data.bridge_reasoning.as_solution = False
cfg.datasets.vla_data.bridge_reasoning.include_bbox = True
cfg.datasets.vla_data.bridge_reasoning.include_action_tokens = False
cfg.datasets.vla_data.bridge_reasoning.include_img_next = False
cfg.datasets.vla_data.bridge_reasoning.component_order = ["SUBTASK", "BBOX", "SPATIAL", "REASON"]
cfg.datasets.vla_data.bridge_reasoning.tag2think_count = {
    "SUBTASK": 1, "BBOX": 1, "SPATIAL": 1, "REASON": 1,
}
cfg.framework.training_stage = "reasoning_only"
cfg.framework.cot_mode = "implicit"
cfg.framework.enable_latent_reasoning = True
cfg.framework.emit_thinking_tokens = False
cfg.framework.cot_mode_flags = {
    "enable_latent_reasoning": True,
    "emit_thinking_tokens": False,
    "use_iterative_forward": True,
    "generate_thinking": True,
    "reasoning_stage": 4,
}
cfg.framework.latent_reasoning.compute_language_loss = True
cfg.framework.latent_reasoning.vlm_loss_weight = 1.0
cfg.framework.latent_reasoning.tag2think_count = {
    "SUBTASK": 1, "BBOX": 1, "SPATIAL": 1, "REASON": 1,
}
cfg.framework.img_next.enable = False
cfg.framework.img_next.loss_weight = 0
cfg.framework.img_next.use_teacher = False
cfg.trainer.pretrained_checkpoint = None
cfg.trainer.reload_modules = None
cfg.trainer.max_train_steps = 5000
cfg.trainer.gradient_accumulation_steps = int(grad_accum)
cfg.trainer.save_interval = 1000
cfg.trainer.eval_interval = 100
cfg.trainer.logging_frequency = 10
cfg.trainer.warmup_ratio = 0.03
OmegaConf.save(cfg, destination)
PY

  save_pipeline_command "${ADAPTER_DIR}"
}

run_stage2() {
  local adapter_checkpoint="${ADAPTER_DIR}/final_model/pytorch_model.pt"
  [[ -f "${adapter_checkpoint}" ]] || { echo "Missing Stage 4 adapter checkpoint: ${adapter_checkpoint}" >&2; exit 1; }

  save_pipeline_command "${STAGE2_DIR}"
  export ACCELERATE_LAUNCH_ARGS="--config_file ${REPO_ROOT}/decoder/accelerate_deepspeed_zero2.yaml --num_processes ${NPROC} --main_process_port ${STAGE2_MASTER_PORT}"

  accelerate launch \
    --config_file "${REPO_ROOT}/decoder/accelerate_deepspeed_zero2.yaml" \
    --num_processes "${NPROC}" \
    --main_process_port "${STAGE2_MASTER_PORT}" \
    decoder/train_latent_decoder.py \
    --checkpoint "${adapter_checkpoint}" \
    --output_dir "${STAGE2_DIR}" \
    --max_train_steps "${STAGE2_STEPS}" \
    --per_device_batch_size "${STAGE2_PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${STAGE2_GRAD_ACCUM}" \
    --save_interval 1000 \
    --logging_frequency 10 \
    --eval_interval 100 \
    --eval_batches 2 \
    --eval_samples 4 \
    --eval_decode_tokens 64 \
    --decoder_learning_rate 1e-4 \
    --vlm_learning_rate 1e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine_with_min_lr \
    --min_lr 5e-7 \
    --gradient_clipping 1.0 \
    --decoder_loss_weight 1.0 \
    --action_token_loss_weight 0.0 \
    --decoder_loss_weight_schedule constant \
    --decoder_type full_vlm \
    --dtype bf16 \
    --attn_implementation flash_attention_2 \
    --base_vlm "${BASE_VLM}" \
    --decoder_base_vlm "${BASE_VLM}" \
    --fast_tokenizer_path "${FAST_TOKENIZER}" \
    --data_root_dir "${DATA_ROOT}" \
    --data_mix libero_all \
    --steps_cache_path "${STEPS_CACHE}" \
    --write_steps_cache false \
    2>&1 | tee -a "${STAGE2_LOG}"
}

case "${START_STAGE}" in
  1)
    run_stage1
    prepare_stage4_adapter
    run_stage2
    ;;
  2)
    prepare_stage4_adapter
    run_stage2
    ;;
  *)
    echo "START_STAGE must be 1 or 2, got: ${START_STAGE}" >&2
    exit 2
    ;;
esac
