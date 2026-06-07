#!/usr/bin/env bash
# Libero-all :: train.py — repository root: bash scripts/run_laravla_libero.sh
set -euo pipefail
export TOKENIZERS_PARALLELISM=false


PRETRAINED_CKPT=
RELOAD_MODULES=qwen_vl_interface

# 若改 run_root_dir / run_id，请同步改下面 mkdir 路径
mkdir -p results/Libero_VLA/libero_all_vla

args=(
  --config_yaml laravla/config/training/libero.yaml
  --run_root_dir results/Libero_VLA
  --run_id libero_all_vla
  --wandb_project libero_vla
  --framework.training_stage full
  --datasets.vla_data.bridge_reasoning.stage 4
  --trainer.max_train_steps 60000
  --datasets.vla_data.per_device_batch_size 8
  --framework.img_next.use_teacher false
)

if [[ -n "${PRETRAINED_CKPT}" ]]; then
  args+=( --trainer.pretrained_checkpoint "${PRETRAINED_CKPT}" )
  [[ -n "${RELOAD_MODULES}" ]] && args+=( --trainer.reload_modules "${RELOAD_MODULES}" )
fi

exec torchrun \
  --nproc_per_node=1 \
  --master_port=29513 \
  laravla/training/train.py \
  "${args[@]}" \
  "$@"
