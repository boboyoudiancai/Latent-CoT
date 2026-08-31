#!/usr/bin/env bash
# Bridge-LeRobot :: train.py — repository root: bash scripts/run_laravla_bridge.sh
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

PRETRAINED_CKPT=
RELOAD_MODULES=qwen_vl_interface

# 若改 run_root_dir / run_id，请同步改下面 mkdir 路径（与 bridge.yaml 默认一致）
mkdir -p results/Bridge/Bridge_VLA

args=(
  --config_yaml laravla/config/training/bridge.yaml
  --run_root_dir results/Bridge
  --run_id bridge_vla
  --wandb_project bridge_vla
  --framework.training_stage full
  --datasets.vla_data.bridge_reasoning.stage 4
  --trainer.max_train_steps 20000
  --datasets.vla_data.per_device_batch_size 8
  --framework.img_next.use_teacher false
)

if [[ -n "${PRETRAINED_CKPT}" ]]; then
  args+=( --trainer.pretrained_checkpoint "${PRETRAINED_CKPT}" )
  [[ -n "${RELOAD_MODULES}" ]] && args+=( --trainer.reload_modules "${RELOAD_MODULES}" )
fi

exec torchrun \
  --nproc_per_node=8 \
  --master_port=29512 \
  laravla/training/train.py \
  "${args[@]}" \
  "$@"
