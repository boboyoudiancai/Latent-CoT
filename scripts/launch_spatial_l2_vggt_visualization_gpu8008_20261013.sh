#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/share/home/u25505/apps/wangbo/LatentVLA/LaRA-VLA
PYTHON_ENV=/share/home/u25505/apps/wangbo/LatentVLA/env/starvla
OUTPUT=/share/home/u25505/apps/wangbo/LatentVLA/results/spatial-latent/spatial_l2_stage1_vggt_visualization_20261013
SCRIPT=$REPO/scripts/eval_spatial_l2_vggt_diagnostic_20261013.py
VGGT=/share/home/u25505/apps/wangbo/LatentVLA/playground/Pretrained_models/VGGT-1B/model.pt
COSINE=$REPO/playground/Checkpoints/Text2Latent/Text2Latent_spatial_baseonly16_stage1_10k_20261008/checkpoints/steps_10000_pytorch_model.pt
L2=$REPO/playground/Checkpoints/Text2Latent/Text2Latent_spatial_baseonly16_stage1_10k_l2_gpu8019_20261012/checkpoints/steps_10000_pytorch_model.pt

test "$(hostname)" = gpu8008
test -s "$SCRIPT"
test -s "$VGGT"
test -s "$COSINE"
test -s "$L2"
test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
cd "$REPO"

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "physical_gpu=6"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
} > "$OUTPUT/run_contract.txt"
cp -p "$SCRIPT" "$OUTPUT/eval_spatial_l2_vggt_diagnostic.py"

set +e
"$PYTHON_ENV/bin/python" "$SCRIPT" \
  --experiment cosine_stage1_16token="$COSINE" \
  --experiment l2_stage1_16token="$L2" \
  --output-dir "$OUTPUT" \
  --sample-indices 22788,136732,250676 \
  --device cuda:0 \
  --vggt-weights "$VGGT" \
  --max-ply-points 100000 \
  --max-plot-points 18000 \
  --save-latents > "$OUTPUT/eval.log" 2>&1
code=$?
set -e

echo "$code" > "$OUTPUT/exitcode"
echo "finished_at=$(date --iso-8601=seconds)" >> "$OUTPUT/run_contract.txt"
exit "$code"
