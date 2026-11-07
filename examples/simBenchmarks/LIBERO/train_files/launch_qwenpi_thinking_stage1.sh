#!/usr/bin/env bash
set -euo pipefail
repo=/share/home/u25505/apps/wangbo/starVLA
env_bin=/share/home/u25505/apps/wangbo/LatentVLA/env/starvla/bin
cd "$repo"
export PYTHONPATH="$repo:/share/home/u25505/third_party_vggt${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=2
exec "$env_bin/accelerate" launch \
  --config_file examples/simBenchmarks/LIBERO/train_files/qwenpi_thinking_zero2.yaml \
  --num_processes 8 --main_process_port "${QWENPI_MAIN_PORT:-29537}" \
  --gradient_accumulation_steps 2 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/LIBERO/train_files/starvla_qwenpi_thinking_stage1.yaml "$@"
