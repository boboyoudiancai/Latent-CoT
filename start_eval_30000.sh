#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/u25505/apps/wangbo/LatentVLA
CKPT=${ROOT}/LaRA-VLA/playground/Checkpoints/Text2Latent/Text2Latent_action_from_full_qwen4b_decoder_explicit_spatialcot8k_stage1to4_decoderloss5k_no_actiontok_no_imgnext_vlm_unfrozen/checkpoints/steps_30000_pytorch_model.pt
TAG=Text2Latent_action_from_full_qwen4b_decoder_explicit_spatialcot8k_stage1to4_decoderloss5k_no_actiontok_no_imgnext_vlm_unfrozen_checkpoints_steps_30000_pytorch_model.pt
PLUS=${ROOT}/LaRA-VLA/results/libero-plus/${TAG}_20260817_1800s
LIBERO=${ROOT}/LaRA-VLA/results/libero/${TAG}

make_plus() {
  local suite="$1" port="$2" dir="$PLUS/$1"
  mkdir -p "$dir"
  cat > "$dir/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
REPO=${ROOT}/LaRA-VLA
CKPT=$CKPT
DIR=$dir
export LIBERO_HOME=${ROOT}/LIBERO-plus
export LIBERO_CONFIG_PATH=${ROOT}/env/liberoplus/.libero-config
export PYTHONPATH=\$REPO:${ROOT}/LIBERO-plus
export TOKENIZERS_PARALLELISM=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl
export MAGICK_HOME=${ROOT}/env/liberoplus
export LD_LIBRARY_PATH=\$MAGICK_HOME/lib:/usr/local/cuda/lib64
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
echo \$\$ > \$DIR/launcher.pid
mkdir -p \$DIR/server_logs
exec ${ROOT}/env/liberoplus/bin/python \$REPO/examples/LIBERO-plus/eval_files/eval_lara_plus.py \\
  --pretrained-path \$CKPT --host 127.0.0.1 --port $port --ports $(seq -s, $port $((port+7))) \\
  --hosts 127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1 \
  --task-suite-name $suite --num-trials-per-task 1 --log-path \$DIR --video-out-path \$DIR --seed 7 \\
  --enable-latent-reasoning --cot-mode implicit --thinking-token-count 4 --img-next-count -1 \\
  --launch-servers --server-gpus 0,1,2,3,4,5,6,7 --num-processes 8 \\
  --render-gpu-ids 0 --server-python ${ROOT}/env/starvla/bin/python --server-log-path \$DIR/server_logs --server-timeout 1800 > \$DIR/eval.log 2>&1
EOF
  chmod +x "$dir/run_command.sh"
}

make_libero() {
  local suite="$1" port="$2" dir="$LIBERO/$1"
  mkdir -p "$dir"
  cat > "$dir/run_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
REPO=${ROOT}/LaRA-VLA
CKPT=$CKPT
DIR=$dir
export LIBERO_HOME=${ROOT}/LIBERO LIBERO_CONFIG_PATH=/share/home/u25505/.libero
export PYTHONPATH=\$REPO:${ROOT}/LIBERO
export TOKENIZERS_PARALLELISM=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl
export MAGICK_HOME=${ROOT}/env/liberoplus
export LD_LIBRARY_PATH=\$MAGICK_HOME/lib:/usr/local/cuda/lib64
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
echo \$\$ > \$DIR/launcher.pid
mkdir -p \$DIR/server_logs
exec ${ROOT}/env/liberoplus/bin/python \$REPO/examples/LIBERO/eval_libero.py \\
  --pretrained-path \$CKPT --host 127.0.0.1 --port $port --ports $(seq -s, $port $((port+7))) \\
  --hosts 127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1 \
  --task-suite-name $suite --num-trials-per-task 50 --log-path \$DIR --video-out-path \$DIR --seed 7 \\
  --enable-latent-reasoning --cot-mode implicit --thinking-token-count 4 --img-next-count -1 --action-decode-mode diffusion \\
  --launch-servers --server-gpus 0,1,2,3,4,5,6,7 --num-processes 8 --render-gpu-ids 0 \\
  --server-python ${ROOT}/env/starvla/bin/python --server-log-path \$DIR/server_logs --server-timeout 1800 > \$DIR/eval.log 2>&1
EOF
  chmod +x "$dir/run_command.sh"
}

for spec in libero_spatial:13000 libero_object:13020 libero_goal:13040 libero_10:13060; do IFS=: read -r s p <<< "$spec"; make_plus "$s" "$p"; done
for spec in libero_spatial:15000 libero_object:15020 libero_goal:15040 libero_10:15060; do IFS=: read -r s p <<< "$spec"; make_libero "$s" "$p"; done
(for s in libero_spatial libero_object libero_goal libero_10; do
  (cd "$PLUS/$s" && ./run_command.sh > nohup.log 2>&1)
done) > "$PLUS/benchmark.log" 2>&1 &
echo $! > "$PLUS/benchmark.pid"

(for s in libero_spatial libero_object libero_goal libero_10; do
  (cd "$LIBERO/$s" && ./run_command.sh > nohup.log 2>&1)
done) > "$LIBERO/benchmark.log" 2>&1 &
echo $! > "$LIBERO/benchmark.pid"
echo "plus=$PLUS"
echo "libero=$LIBERO"
