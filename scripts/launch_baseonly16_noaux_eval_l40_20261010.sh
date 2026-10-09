#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/share/home/u25505/apps/wangbo/LatentVLA/LaRA-VLA
TEXT_ROOT=$REPO/playground/Checkpoints/Text2Latent
EVAL_PY=/share/home/u25505/apps/wangbo/LatentVLA/env/liberoplus/bin/python
SERVER_PY=/share/home/u25505/apps/wangbo/LatentVLA/env/starvla/bin/python
RESULT_ROOT=/share/home/u25505/apps/wangbo/LatentVLA/results/libero-plus/spatial_baseonly16_noaux_25k30k_l40_20261010
RUN_ROOT=$TEXT_ROOT/Text2Latent_action_from_spatial_baseonly16_noaux_stage2_40k_gpu8008_20261008

export LIBERO_HOME=$REPO/LIBERO-plus
export LIBERO_CONFIG_PATH=/share/home/u25505/apps/wangbo/LatentVLA/env/liberoplus/.libero-config
export PYTHONPATH=$REPO:$REPO/LIBERO-plus:${PYTHONPATH:-}
export LARAVLA_PYTHON=$SERVER_PY
export SERVER_PY
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl
export MAGICK_HOME=/share/home/u25505/apps/wangbo/LatentVLA/env/liberoplus
export LD_LIBRARY_PATH=/share/home/u25505/apps/wangbo/LatentVLA/env/liberoplus/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

mkdir -p $RESULT_ROOT
exec >> $RESULT_ROOT/controller.log 2>&1
echo "$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID:-none} start"
git -C $REPO rev-parse HEAD > $RESULT_ROOT/git_commit.txt
git -C $REPO status --short > $RESULT_ROOT/git_status.txt
git -C $REPO diff -- examples/LIBERO/model2libero_interface.py > $RESULT_ROOT/inference_code.diff
if [ -f /tmp/baseonly16_contract.json ]; then
  cp /tmp/baseonly16_contract.json $RESULT_ROOT/baseonly16_contract.json
else
  cat > $RESULT_ROOT/baseonly16_contract.json <<'EOF'
{
  "spatial_view_mode": "base_only",
  "spatial_query_mode": "visual_prefix_joint",
  "prompt_exact_match": true,
  "input_ids_exact_match": true,
  "spatial_base_count": 16,
  "spatial_wrist_count": 0,
  "base_block_full_attention": true,
  "spatial_to_future_blocked": true,
  "contract": "passed"
}
EOF
fi
nvidia-smi > $RESULT_ROOT/nvidia_smi.before.txt

run_suite() {
  local label=$1 ckpt=$2 suite=$3 server_gpus=$4 render_gpu=$5 port=$6
  local out=$RESULT_ROOT/$label/$suite
  local ports="$port,$((port+1)),$((port+2)),$((port+3)),$((port+4)),$((port+5)),$((port+6)),$((port+7))"
  local hosts="127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1"
  mkdir -p $out/server_logs
  echo "$(date --iso-8601=seconds) start label=$label suite=$suite ckpt=$ckpt gpus=$server_gpus ports=$ports" >> $out/launcher.log
  set +e
  $EVAL_PY $REPO/examples/LIBERO-plus/eval_files/eval_lara_plus.py +    --pretrained-path $ckpt +    --host 127.0.0.1 --hosts $hosts --port $port --ports $ports +    --task-suite-name $suite --num-trials-per-task 1 +    --log-path $out --video-out-path $out --seed 7 +    --cot-mode none --thinking-token-count 0 --img-next-count 0 +    --launch-servers --server-gpus $server_gpus --num-processes 8 +    --render-gpu-ids $render_gpu --server-python $SERVER_PY +    --server-log-path $out/server_logs --server-timeout 1800 +    > $out/eval.log 2>&1
  local rc=$?
  set -e
  echo $rc > $out/eval.exitcode
  echo "$(date --iso-8601=seconds) exit=$rc" >> $out/launcher.log
  return $rc
}

run_checkpoint() {
  local label=$1 step=$2 gpus=$3 render=$4 port=$5
  local ckpt=$RUN_ROOT/checkpoints/steps_${step}_pytorch_model.pt
  local root=$RESULT_ROOT/$label
  test -s $ckpt
  mkdir -p $root
  cat > $root/experiment_contract.txt <<EOF
label=$label
checkpoint=$ckpt
hardware=L40
gpus=$gpus
workers=8 total, 2 per GPU
seed=7
trials=1
cot_mode=none
latent_reasoning=false
thinking_tokens=0
img_next_tokens=0
state=false
spatial_tokens=base16+wrist0
spatial_view_mode=base_only
spatial_query_mode=visual_prefix_joint
prompt_contract=base16 plus instruction period
EOF
  for suite in libero_spatial libero_object libero_goal libero_10; do
    run_suite $label $ckpt $suite $gpus $render $port
  done
  date --iso-8601=seconds > $root/complete.txt
}

run_checkpoint baseonly16_noaux_step25000 25000 0,0,1,1,2,2,3,3 0 16100 &
pid25=$!
run_checkpoint baseonly16_noaux_step30000 30000 4,4,5,5,6,6,7,7 4 16120 &
pid30=$!
set +e
wait $pid25; rc25=$?
wait $pid30; rc30=$?
set -e
echo "step25000_rc=$rc25" > $RESULT_ROOT/status.txt
echo "step30000_rc=$rc30" >> $RESULT_ROOT/status.txt
nvidia-smi > $RESULT_ROOT/nvidia_smi.after.txt
echo "$(date --iso-8601=seconds) done rc25=$rc25 rc30=$rc30"
test $rc25 -eq 0 -a $rc30 -eq 0
