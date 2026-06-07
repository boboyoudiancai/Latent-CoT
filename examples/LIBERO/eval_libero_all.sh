#!/usr/bin/env bash
# 
set -euo pipefail

LARAVLA_PYTHON="${LARAVLA_PYTHON:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-}"
export LIBERO_HOME="${LIBERO_HOME:-}"

export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/qwen_cache}"

# Default python interpreters (override via env vars).


# Default checkpoint (override by arg $1 or YOUR_CKPT).
DEFAULT_CKPT_PATH="${DEFAULT_CKPT_PATH:-}"

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
  return 1
}

require_python "LARAVLA_PYTHON" "${LARAVLA_PYTHON}"
require_python "LIBERO_PYTHON" "${LIBERO_PYTHON}"

CKPT_PATH="${1:-${YOUR_CKPT:-${DEFAULT_CKPT_PATH:-}}}"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "❌ Please provide a checkpoint path, preferably absolute. Example:"
  echo "   YOUR_CKPT=/abs/path/to/steps_15000_pytorch_model.pt bash $0"
  echo "Or set DEFAULT_CKPT_PATH in the script or environment."
  exit 1
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "❌ Checkpoint file not found: ${CKPT_PATH}"
  exit 1
fi

TASK_SUITES="${TASK_SUITES:-libero_goal,libero_spatial,libero_object,libero_10}"
IFS=',' read -r -a SUITES <<< "${TASK_SUITES}"
if [[ "${#SUITES[@]}" -eq 0 ]]; then
  echo "❌ TASK_SUITES is empty"
  exit 1
fi

BASE_PORT="${BASE_PORT:-10093}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SAVE_VIDEOS="${SAVE_VIDEOS:-false}"

GPU_ID="${GPU_ID:-0}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
IFS=',' read -r -a CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#CUDA_DEVICES[@]}"


if [[ -z "${LIBERO_HOME}" ]]; then
  echo "❌ Please set LIBERO_HOME, for example: LIBERO_HOME=/abs/path/to/LIBERO" >&2
  exit 1
fi
if [[ ! -d "${LIBERO_HOME}" ]]; then
  echo "❌ LIBERO_HOME does not exist: ${LIBERO_HOME}" >&2
  exit 1
fi


EVAL_PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  EVAL_PYTHONPATH="${EVAL_PYTHONPATH}:${PYTHONPATH}"
fi

CKPT_DIR="$(cd "$(dirname "${CKPT_PATH}")" && pwd)"
CKPT_BASENAME="$(basename "${CKPT_PATH%.pt}")"
EVAL_DIR="${EVAL_DIR:-${CKPT_DIR}/eval_libero_implicit_parallel/${CKPT_BASENAME}}"
SERVER_LOG_DIR="${EVAL_DIR}/server_logs"
LOG_DIR="${EVAL_DIR}/logs"
VIDEO_DIR="${EVAL_DIR}/videos"
mkdir -p "${SERVER_LOG_DIR}" "${LOG_DIR}" "${VIDEO_DIR}"

video_args=()
if [[ "${SAVE_VIDEOS}" == "true" ]]; then
  video_args+=(--args.save-videos)
fi

echo "======================================================"
echo "📊 LIBERO Parallel Evaluation (implicit latent reasoning)"
echo "------------------------------------------------------"
echo "Checkpoint : ${CKPT_PATH}"
echo "Suites     : ${SUITES[*]}"
echo "Ports      : from ${BASE_PORT} (N=${#SUITES[@]})"
echo "GPU Pool   : ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} GPUs)"
echo "Trials     : ${NUM_TRIALS_PER_TASK}"
echo "Videos     : ${SAVE_VIDEOS}"
echo "Output Dir : ${EVAL_DIR}"
echo "======================================================"

server_pids=()
server_ports=()
eval_pids=()

cleanup_port() {
  local port="$1"
  local stale_pids=""
  if command -v pgrep >/dev/null 2>&1; then
    stale_pids="$(pgrep -f "deployment/model_server/server_policy.py.*--port ${port}" || true)"
  else
    stale_pids="$(ps aux | grep "deployment/model_server/server_policy.py" | grep "--port ${port}" | grep -v grep | awk '{print $2}' || true)"
  fi
  if [[ -n "${stale_pids}" ]]; then
    echo "🧹 Cleaning up stale server processes on port ${port}: ${stale_pids}"
    kill ${stale_pids} 2>/dev/null || true
    sleep 2
    kill -9 ${stale_pids} 2>/dev/null || true
  fi
}

cleanup() {
  for pid in "${eval_pids[@]}"; do
    if ps -p "${pid}" > /dev/null 2>&1; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${server_pids[@]}"; do
    if ps -p "${pid}" > /dev/null 2>&1; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${server_pids[@]}"; do
    if ps -p "${pid}" > /dev/null 2>&1; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
  for port in "${server_ports[@]}"; do
    cleanup_port "${port}"
  done
}
trap cleanup EXIT

wait_port_ready() {
  local port="$1"
  echo "⏳ Waiting for server to become ready: 127.0.0.1:${port}"
  "${LARAVLA_PYTHON}" - <<PY
import os
import sys
import time

try:
    import websockets.sync.client
except Exception as e:
    print(
        "missing python dependency for port check: websockets (sync client). "
        "Install it in the LARAVLA_PYTHON environment, e.g. `pip install websockets>=11`.",
        file=sys.stderr,
    )
    raise

host="127.0.0.1"
port=int("${port}")
timeout_s=180
start=time.time()

for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

while True:
    if time.time() - start > timeout_s:
        print(f"server not ready after {timeout_s}s: {host}:{port}", file=sys.stderr)
        sys.exit(1)
    try:
        uri = f"ws://{host}:{port}"
        conn = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            open_timeout=5,
            ping_interval=None,
        )
        # If handshake succeeded, the server is ready. Close immediately.
        conn.close()
        print(f"server ready: {host}:{port}")
        sys.exit(0)
    except Exception:
        time.sleep(2)
PY
}

start_server() {
  local gpu_id="$1"
  local port="$2"
  local log_file="${SERVER_LOG_DIR}/server_${port}.log"
  cleanup_port "${port}"
  echo "▶️  Starting server: GPU=${gpu_id} port=${port} log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${LARAVLA_PYTHON}" deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port "${port}" \
    --use_bf16 \
    > "${log_file}" 2>&1 &
  server_pids+=("$!")
  server_ports+=("${port}")
}

run_suite_eval() {
  local suite="$1"
  local port="$2"
  local suite_video_dir="${VIDEO_DIR}/${suite}"
  local stdout_log="${LOG_DIR}/${suite}.stdout.log"

  mkdir -p "${suite_video_dir}"
  echo "🧪 Starting evaluation: suite=${suite} port=${port} stdout_log=${stdout_log}"
  LIBERO_HOME="${LIBERO_HOME}" \
  LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
  PYTHONPATH="${EVAL_PYTHONPATH}" \
  "${LIBERO_PYTHON}" ./examples/LIBERO/eval_libero.py \
    --args.pretrained-path "${CKPT_PATH}" \
    --args.host "127.0.0.1" \
    --args.port "${port}" \
    --args.task-suite-name "${suite}" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.video-out-path "${suite_video_dir}" \
    "${video_args[@]}" \
    --args.enable-latent-reasoning \
    --args.cot-mode implicit \
    --args.log_path "${LOG_DIR}" \
    > "${stdout_log}" 2>&1 &
  eval_pids+=("$!")
}

for idx in "${!SUITES[@]}"; do
  suite="${SUITES[$idx]}"
  port=$((BASE_PORT + idx))
  gpu_id="${CUDA_DEVICES[$((idx % NUM_GPUS))]}"

  start_server "${gpu_id}" "${port}"
done

if (( NUM_GPUS < ${#SUITES[@]} )); then
  echo "⚠️  The GPU pool has only ${NUM_GPUS} devices, but there are ${#SUITES[@]} suites. Multiple servers will share GPUs, which may cause OOM or slowdowns."
fi

echo ""
echo "⏳ Waiting for all servers to become ready..."
for idx in "${!SUITES[@]}"; do
  port=$((BASE_PORT + idx))
  wait_port_ready "${port}"
done

for idx in "${!SUITES[@]}"; do
  suite="${SUITES[$idx]}"
  port=$((BASE_PORT + idx))
  run_suite_eval "${suite}" "${port}"
done

echo ""
echo "🚀 Started ${#server_pids[@]} servers and ${#eval_pids[@]} eval jobs. Waiting for completion..."
for pid in "${eval_pids[@]}"; do
  wait "${pid}"
done

echo ""
echo "✅ LIBERO parallel evaluation complete: ${EVAL_DIR}"
