#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  printf 'Usage: %s STAGE1_CHECKPOINT [training overrides...]\n' "$0" >&2
  exit 2
fi
stage1_checkpoint=$1
shift
repo=/share/home/u25505/apps/wangbo/starVLA
env_bin=/share/home/u25505/apps/wangbo/LatentVLA/env/starvla/bin
cd "$repo"
export PYTHONPATH="$repo:/share/home/u25505/third_party_vggt${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1
exec "$env_bin/accelerate" launch \
  --config_file examples/simBenchmarks/LIBERO/train_files/qwenpi_thinking_zero2.yaml \
  --num_processes 8 --main_process_port "${QWENPI_MAIN_PORT:-29538}" \
  --gradient_accumulation_steps 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/LIBERO/train_files/starvla_qwenpi_thinking_stage2.yaml \
  --trainer.pretrained_checkpoint "$stage1_checkpoint" "$@"
