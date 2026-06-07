#!/usr/bin/env bash
# ============================================================================
# SimplerEnv parallel evaluation script (Stage-4 action-only / latent reasoning)
# Usage:
#   bash examples/SimplerEnv/bridge_eval.sh /path/to/steps_xxxxx_pytorch_model.pt
# Environment variables:
#   TSET_NUM              Repeat count per task, default 1
#   NUM_EPISODES          Number of episodes per task, default 24
#   BASE_PORT             Starting port, default 10120
#   GPU_ID                GPU used when CUDA_VISIBLE_DEVICES is unset (default 0)
#   CUDA_VISIBLE_DEVICES  Optional comma-separated GPU list for parallel task allocation
#   LOG_DIR               Log directory (default ckpt_dir/eval_stage4_parallel)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"


########you need to secect your python path######
laravla_python="${laravla_python:-}"
sim_python="${sim_python:-}"
SimplerEnv_PATH="${SimplerEnv_PATH:-}"
#######you need to secect your python path######


DEFAULT_CKPT_PATH="${DEFAULT_CKPT_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/qwen_cache}"

require_python() {
  local label="$1"
  local value="$2"
  if [[ -x "${value}" ]]; then
    return 0
  fi
  if command -v "${value}" >/dev/null 2>&1; then
    return 0
  fi
  echo "❌ ${label} is not available: ${value}" >&2
  exit 1
}

require_python "laravla_python" "${laravla_python}"
require_python "sim_python" "${sim_python}"

if [[ -z "${SimplerEnv_PATH}" ]]; then
  echo "❌ Please set SimplerEnv_PATH, for example: SimplerEnv_PATH=/abs/path/to/SimplerEnv" >&2
  exit 1
fi
if [[ ! -d "${SimplerEnv_PATH}" ]]; then
  echo "❌ SimplerEnv_PATH does not exist: ${SimplerEnv_PATH}" >&2
  exit 1
fi
export SimplerEnv_PATH


CKPT_PATH="${1:-${YOUR_CKPT:-${DEFAULT_CKPT_PATH:-}}}"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "❌ Please provide a checkpoint path, for example: bash $0 /abs/path/to/steps_10000_pytorch_model.pt"
  exit 1
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "❌ Checkpoint file not found: ${CKPT_PATH}"
  exit 1
fi

TSET_NUM="${TSET_NUM:-1}"
NUM_EPISODES="${NUM_EPISODES:-24}"
BASE_PORT="${BASE_PORT:-10320}"
GPU_ID="${GPU_ID:-0}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
IFS=',' read -r -a CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#CUDA_DEVICES[@]}"

CKPT_DIR="$(cd "$(dirname "${CKPT_PATH}")" && pwd)"
CKPT_BASENAME="$(basename "${CKPT_PATH%.*}")"
LOG_DIR="${LOG_DIR:-${CKPT_DIR}/eval_stage4_parallel60000}"
mkdir -p "${LOG_DIR}"

# Inference mode: none / vlm_seen_no_out / explicit / implicit
COT_MODE="${COT_MODE:-implicit}"
IMG_NEXT_COUNT="${IMG_NEXT_COUNT:-16}"

echo "======================================================"
echo "📊 Stage-4 SimplerEnv Parallel Evaluation"
echo "------------------------------------------------------"
echo "Checkpoint : ${CKPT_PATH}"
echo "Logs       : ${LOG_DIR}"
echo "Repeat     : ${TSET_NUM}"
echo "Episodes   : ${NUM_EPISODES}"
echo "Ports      : from ${BASE_PORT}"
echo "GPU Pool   : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs)"
echo "CoT Mode   : ${COT_MODE}"
echo "======================================================"

policyserver_pids=()
server_ports=()
eval_pids=()

cleanup_port() {
  local port="$1"
  local stale
  stale=$(ps aux | grep "server_policy.py" | grep "--port ${port}" | grep -v grep | awk '{print $2}' || true)
  if [[ -n "${stale}" ]]; then
    echo "   Cleaning up stale server processes on port ${port}: ${stale}"
    kill -9 ${stale} 2>/dev/null || true
    sleep 1
  fi
}

start_server() {
  local gpu_id="$1"
  local port="$2"
  local server_logs="${LOG_DIR}/server_logs"
  mkdir -p "${server_logs}"
  local log_file="${server_logs}/${CKPT_BASENAME}_server_${port}.log"

  cleanup_port "${port}"
  echo "▶️  Starting policy server (GPU ${gpu_id}, port ${port})"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${laravla_python}" deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port "${port}" \
    --use_bf16 \
    > "${log_file}" 2>&1 &

  local pid=$!
  policyserver_pids+=("${pid}")
  server_ports+=("${port}")
  sleep 8

  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "⚠️  Server (port ${port}) may have failed to start. Check ${log_file}"
  fi
}

stop_all_servers() {
  echo ""
  echo "⏳ Waiting for all evaluation tasks to finish..."
  for pid in "${eval_pids[@]}"; do
    if ps -p "${pid}" > /dev/null 2>&1; then
      wait "${pid}" || true
    fi
  done

  echo "⏹  Stopping policy servers..."
  for pid in "${policyserver_pids[@]}"; do
    if ps -p "${pid}" > /dev/null 2>&1; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2

  for pid in "${policyserver_pids[@]}"; do
    if ps -p "${pid}" > /dev/null 2>&1; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done

  for port in "${server_ports[@]}"; do
    cleanup_port "${port}"
  done

  eval_pids=()
  policyserver_pids=()
  server_ports=()
}

run_task() {
  local gpu_id="$1"
  local env_name="$2"
  local scene="$3"
  local robot="$4"
  local rgb_overlay="$5"
  local robot_x="$6"
  local robot_y="$7"
  local run_idx="$8"
  local port="$9"

  start_server "${gpu_id}" "${port}"

  local tag="run${run_idx}"
  local log_file="${LOG_DIR}/${CKPT_BASENAME}_stage4_${env_name}.log.${tag}"
  echo "🧪 Task ${env_name} | run ${run_idx}/${TSET_NUM} | GPU ${gpu_id} | port ${port}"
  echo "   Log: ${log_file}"

  # Enable implicit reasoning only when COT_MODE=implicit.
  local reasoning_flag=()
  if [[ "${COT_MODE}" == "implicit" ]]; then
    reasoning_flag+=(--enable-latent-reasoning --thinking-token-count 3)
  fi

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${sim_python}" examples/SimplerEnv/start_simpler_env.py \
    --port "${port}" \
    --ckpt-path "${CKPT_PATH}" \
    --policy-setup widowx_bridge \
    --cot-mode "${COT_MODE}" \
    --img-next-count "${IMG_NEXT_COUNT}" \
    --robot "${robot}" \
    --control-freq 5 \
    --sim-freq 500 \
    --max-episode-steps 120 \
    --env-name "${env_name}" \
    --scene-name "${scene}" \
    --rgb-overlay-path "${rgb_overlay}" \
    --robot-init-x "${robot_x}" "${robot_x}" 1 \
    --robot-init-y "${robot_y}" "${robot_y}" 1 \
    --obj-variation-mode episode \
    --obj-episode-range 0 "${NUM_EPISODES}" \
    --robot-init-rot-quat-center 0 0 0 1 \
    --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --logging-dir "${LOG_DIR}" \
    "${reasoning_flag[@]}" \
    > "${log_file}" 2>&1 &

  eval_pids+=("$!")
}

trap stop_all_servers EXIT

declare -a TASKS=(
  "StackGreenCubeOnYellowCubeBakedTexInScene-v0|bridge_table_1_v1|widowx|${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png|0.147|0.028"
  "PutCarrotOnPlateInScene-v0|bridge_table_1_v1|widowx|${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png|0.147|0.028"
  "PutSpoonOnTableClothInScene-v0|bridge_table_1_v1|widowx|${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png|0.147|0.028"
  "PutEggplantInBasketScene-v0|bridge_table_1_v2|widowx_sink_camera_setup|${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png|0.127|0.06"
)

task_index=0
for task_spec in "${TASKS[@]}"; do
  IFS='|' read -r ENV_NAME SCENE_NAME ROBOT_NAME RGB_PATH RX RY <<< "${task_spec}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    gpu_id="${CUDA_DEVICES[$((task_index % NUM_GPUS))]}"
    port=$((BASE_PORT + task_index))
    run_task "${gpu_id}" "${ENV_NAME}" "${SCENE_NAME}" "${ROBOT_NAME}" "${RGB_PATH}" "${RX}" "${RY}" "${run_idx}" "${port}"
    task_index=$((task_index + 1))
  done
done

echo ""
echo "🚀 Started ${task_index} tasks. Waiting for completion..."
stop_all_servers

echo ""
echo "✅ Stage-4 parallel evaluation complete. Logs written to: ${LOG_DIR}"
