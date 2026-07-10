#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

export HF_HOME="${HF_HOME:-${REPO_ROOT}/qwen_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

LARAVLA_PYTHON="${LARAVLA_PYTHON:-/home/liuyue/miniconda3/envs/starvla/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/home/liuyue/miniconda3/envs/libero/bin/python}"
LIBERO_HOME="${LIBERO_HOME:-/home/liuyue/LIBERO}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/home/liuyue/.libero}"

DEFAULT_CKPT_PATH="${DEFAULT_CKPT_PATH:-}"
CKPT_PATH="${1:-${YOUR_CKPT:-${DEFAULT_CKPT_PATH:-}}}"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "Please provide a checkpoint path, preferably absolute." >&2
  echo "Example: YOUR_CKPT=/abs/path/to/steps_25000_pytorch_model.pt bash $0" >&2
  exit 1
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint file not found: ${CKPT_PATH}" >&2
  exit 1
fi

require_python() {
  local label="$1"
  local value="$2"
  if [[ -x "${value}" ]]; then
    return 0
  fi
  if command -v "${value}" >/dev/null 2>&1; then
    return 0
  fi
  echo "${label} is not available: ${value}" >&2
  return 1
}

require_python "LARAVLA_PYTHON" "${LARAVLA_PYTHON}"
require_python "LIBERO_PYTHON" "${LIBERO_PYTHON}"

if [[ ! -d "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME does not exist: ${LIBERO_HOME}" >&2
  exit 1
fi
if [[ ! -e "${LIBERO_CONFIG_PATH}" ]]; then
  echo "LIBERO_CONFIG_PATH does not exist: ${LIBERO_CONFIG_PATH}" >&2
  exit 1
fi

TASK_SUITES="${TASK_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
IFS=',' read -r -a SUITES <<< "${TASK_SUITES}"
if [[ "${#SUITES[@]}" -eq 0 ]]; then
  echo "TASK_SUITES is empty" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a CUDA_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#CUDA_DEVICES[@]}" -eq 0 ]]; then
  echo "CUDA_VISIBLE_DEVICES is empty" >&2
  exit 1
fi

BASE_PORT="${BASE_PORT:-10093}"
PORT_STRIDE="${PORT_STRIDE:-20}"
REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-2}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:-}"
SAVE_VIDEOS="${SAVE_VIDEOS:-false}"
SEED="${SEED:-7}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
RENDER_GPUS="${RENDER_GPUS:-0}"
SERVER_USE_BF16="${SERVER_USE_BF16:-true}"
START_EVAL="${START_EVAL:-true}"

ENABLE_LATENT_REASONING="${ENABLE_LATENT_REASONING:-true}"
COT_MODE="${COT_MODE:-implicit}"
THINKING_TOKEN_COUNT="${THINKING_TOKEN_COUNT:--1}"
IMG_NEXT_COUNT="${IMG_NEXT_COUNT:--1}"
ACTION_DECODE_MODE="${ACTION_DECODE_MODE:-diffusion}"
FAST_MAX_NEW_TOKENS="${FAST_MAX_NEW_TOKENS:-256}"
UNNORM_KEY="${UNNORM_KEY:-}"

if ! [[ "${BASE_PORT}" =~ ^[0-9]+$ ]]; then
  echo "BASE_PORT must be a non-negative integer. Got: ${BASE_PORT}" >&2
  exit 1
fi
if ! [[ "${PORT_STRIDE}" =~ ^[0-9]+$ ]] || (( PORT_STRIDE <= 0 )); then
  echo "PORT_STRIDE must be a positive integer. Got: ${PORT_STRIDE}" >&2
  exit 1
fi
if ! [[ "${REPLICAS_PER_GPU}" =~ ^[0-9]+$ ]] || (( REPLICAS_PER_GPU <= 0 )); then
  echo "REPLICAS_PER_GPU must be a positive integer. Got: ${REPLICAS_PER_GPU}" >&2
  exit 1
fi
if ! [[ "${NUM_TRIALS_PER_TASK}" =~ ^[0-9]+$ ]] || (( NUM_TRIALS_PER_TASK <= 0 )); then
  echo "NUM_TRIALS_PER_TASK must be a positive integer. Got: ${NUM_TRIALS_PER_TASK}" >&2
  exit 1
fi

join_csv() {
  local IFS=,
  echo "$*"
}

build_default_server_gpus() {
  local out=()
  local gpu
  local replica
  for gpu in "${CUDA_DEVICES[@]}"; do
    for ((replica = 0; replica < REPLICAS_PER_GPU; replica++)); do
      out+=("${gpu}")
    done
  done
  join_csv "${out[@]}"
}

SERVER_GPUS="${SERVER_GPUS:-$(build_default_server_gpus)}"
IFS=',' read -r -a SERVER_GPU_LIST <<< "${SERVER_GPUS}"
NUM_ENDPOINTS="${NUM_ENDPOINTS:-${#SERVER_GPU_LIST[@]}}"
if ! [[ "${NUM_ENDPOINTS}" =~ ^[0-9]+$ ]] || (( NUM_ENDPOINTS <= 0 )); then
  echo "NUM_ENDPOINTS must be a positive integer. Got: ${NUM_ENDPOINTS}" >&2
  exit 1
fi
if (( ${#SERVER_GPU_LIST[@]} < NUM_ENDPOINTS )); then
  echo "SERVER_GPUS has fewer entries than NUM_ENDPOINTS: ${SERVER_GPUS} vs ${NUM_ENDPOINTS}" >&2
  exit 1
fi
if (( PORT_STRIDE < NUM_ENDPOINTS )); then
  echo "PORT_STRIDE (${PORT_STRIDE}) must be >= NUM_ENDPOINTS (${NUM_ENDPOINTS}) to avoid port overlap" >&2
  exit 1
fi

build_ports() {
  local base_port="$1"
  local count="$2"
  local ports=()
  local idx
  for ((idx = 0; idx < count; idx++)); do
    ports+=("$((base_port + idx))")
  done
  join_csv "${ports[@]}"
}

build_hosts() {
  local count="$1"
  local hosts=()
  local idx
  for ((idx = 0; idx < count; idx++)); do
    hosts+=("127.0.0.1")
  done
  join_csv "${hosts[@]}"
}

CKPT_DIR="$(cd "$(dirname "${CKPT_PATH}")" && pwd)"
CKPT_BASENAME="$(basename "${CKPT_PATH%.pt}")"
EVAL_ROOT="${EVAL_DIR:-${CKPT_DIR}/eval_libero_implicit_parallel/${CKPT_BASENAME}}"
mkdir -p "${EVAL_ROOT}"

export LIBERO_HOME
export LIBERO_CONFIG_PATH
export LARAVLA_PYTHON
export SERVER_PY="${LARAVLA_PYTHON}"
export CUDA_VISIBLE_DEVICES

EVAL_PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  EVAL_PYTHONPATH="${EVAL_PYTHONPATH}:${PYTHONPATH}"
fi

write_suite_script() {
  local suite="$1"
  local base_port="$2"
  local suite_dir="${EVAL_ROOT}/${suite}"
  local ports
  local hosts

  ports="$(build_ports "${base_port}" "${NUM_ENDPOINTS}")"
  hosts="$(build_hosts "${NUM_ENDPOINTS}")"
  mkdir -p "${suite_dir}"

  cat > "${suite_dir}/run_command.sh" <<SUITEEOF
#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO_ROOT}"
CKPT="${CKPT_PATH}"
SUITE="${suite}"
BASE_PORT="${base_port}"
PORTS="${ports}"
HOSTS="${hosts}"
SERVER_GPUS="${SERVER_GPUS}"
RENDER_GPUS="${RENDER_GPUS}"
SUITE_DIR="${suite_dir}"
EVAL_PY="${LIBERO_PYTHON}"
SERVER_PY="${LARAVLA_PYTHON}"
MAX_TASKS="${MAX_TASKS}"
SAVE_VIDEOS="${SAVE_VIDEOS}"
ENABLE_LATENT_REASONING="${ENABLE_LATENT_REASONING}"
UNNORM_KEY="${UNNORM_KEY}"
SERVER_USE_BF16="${SERVER_USE_BF16}"

echo \$\$ > "\$SUITE_DIR/launcher.pid"
cd "\$REPO"

export LIBERO_HOME="${LIBERO_HOME}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}"
export PYTHONPATH="${EVAL_PYTHONPATH}"
export LARAVLA_PYTHON="\$SERVER_PY"
export SERVER_PY="\$SERVER_PY"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}"
export MUJOCO_GL="${MUJOCO_GL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cat > "\$SUITE_DIR/meta_lara_infer.txt" <<META
ckpt=\$CKPT
suite=\$SUITE
base_port=\$BASE_PORT
ports=\$PORTS
server_gpus=\$SERVER_GPUS
render_gpu_ids=\$RENDER_GPUS
num_processes=${NUM_ENDPOINTS}
repo=\$REPO
eval_python=\$EVAL_PY
server_python=\$SERVER_PY
libero_home=\$LIBERO_HOME
libero_config_path=\$LIBERO_CONFIG_PATH
started_at=\$(date -Is)
META
cat > "\$SUITE_DIR/log_paths.txt" <<META
launcher=\$SUITE_DIR/launcher.log
eval_stdout=\$SUITE_DIR/eval.log
suite_log=\$SUITE_DIR/\$SUITE.log
suite_json=\$SUITE_DIR/\$SUITE.json
parallel_metrics=\$SUITE_DIR/parallel_metrics.json
server_logs=\$SUITE_DIR/server_logs
META

printf "%s launcher start ckpt=%s suite=%s base_port=%s server_gpus=%s render_gpu_ids=%s num_processes=${NUM_ENDPOINTS} timeout=${SERVER_TIMEOUT}\\n" "\$(date -Is)" "\$CKPT" "\$SUITE" "\$BASE_PORT" "\$SERVER_GPUS" "\$RENDER_GPUS" >> "\$SUITE_DIR/launcher.log"
eval_args=(
  --pretrained-path "\$CKPT"
  --host 127.0.0.1
  --hosts "\$HOSTS"
  --port "\$BASE_PORT"
  --ports "\$PORTS"
  --task-suite-name "\$SUITE"
  --num-trials-per-task "${NUM_TRIALS_PER_TASK}"
  --log-path "\$SUITE_DIR"
  --video-out-path "\$SUITE_DIR"
  --seed "${SEED}"
  --cot-mode "${COT_MODE}"
  --thinking-token-count "${THINKING_TOKEN_COUNT}"
  --img-next-count "${IMG_NEXT_COUNT}"
  --action-decode-mode "${ACTION_DECODE_MODE}"
  --fast-max-new-tokens "${FAST_MAX_NEW_TOKENS}"
  --launch-servers
  --server-gpus "\$SERVER_GPUS"
  --num-processes "${NUM_ENDPOINTS}"
  --render-gpu-ids "\$RENDER_GPUS"
  --server-python "\$SERVER_PY"
  --server-log-path "\$SUITE_DIR/server_logs"
  --server-timeout "${SERVER_TIMEOUT}"
)
if [[ -n "\$MAX_TASKS" ]]; then
  eval_args+=(--max-tasks "\$MAX_TASKS")
fi
if [[ "\$SAVE_VIDEOS" == "true" ]]; then
  eval_args+=(--save-videos)
fi
if [[ -n "\$UNNORM_KEY" ]]; then
  eval_args+=(--unnorm-key "\$UNNORM_KEY")
fi
if [[ "\$ENABLE_LATENT_REASONING" == "true" ]]; then
  eval_args+=(--enable-latent-reasoning)
fi
if [[ "\$SERVER_USE_BF16" == "true" ]]; then
  eval_args+=(--server-use-bf16)
else
  eval_args+=(--no-server-use-bf16)
fi

set +e
"\$EVAL_PY" "\$REPO/examples/LIBERO/eval_libero.py" "\${eval_args[@]}" > "\$SUITE_DIR/eval.log" 2>&1
CODE=\$?
echo "\$CODE" > "\$SUITE_DIR/eval.exitcode"
printf "%s eval exit code=%s\\n" "\$(date -Is)" "\$CODE" >> "\$SUITE_DIR/launcher.log"
exit "\$CODE"
SUITEEOF
  chmod +x "${suite_dir}/run_command.sh"
}

write_run_all_script() {
  local suite_array=""
  local suite
  for suite in "${SUITES[@]}"; do
    suite_array+=" ${suite}"
  done

  cat > "${EVAL_ROOT}/run_all_libero.sh" <<RUNEOF
#!/usr/bin/env bash
set -euo pipefail
ROOT="${EVAL_ROOT}"
SUITES=(${suite_array# })
status=0
echo \$\$ > "\$ROOT/launcher.pid"
printf "%s run_all start root=%s suites=%s timeout=${SERVER_TIMEOUT}\\n" "\$(date -Is)" "\$ROOT" "\${SUITES[*]}" >> "\$ROOT/launcher.log"
for suite in "\${SUITES[@]}"; do
  suite_dir="\$ROOT/\$suite"
  printf "%s suite start %s\\n" "\$(date -Is)" "\$suite" >> "\$ROOT/launcher.log"
  set +e
  bash "\$suite_dir/run_command.sh" > "\$suite_dir/nohup.log" 2>&1
  code=\$?
  set -e
  echo "\$code" > "\$suite_dir/launcher.exitcode"
  printf "%s suite done %s code=%s\\n" "\$(date -Is)" "\$suite" "\$code" >> "\$ROOT/launcher.log"
  if [[ "\$code" -ne 0 ]]; then
    status=1
  fi
done
printf "%s run_all done status=%s\\n" "\$(date -Is)" "\$status" >> "\$ROOT/launcher.log"
exit "\$status"
RUNEOF
  chmod +x "${EVAL_ROOT}/run_all_libero.sh"
}

idx=0
for suite in "${SUITES[@]}"; do
  suite_base_port=$((BASE_PORT + idx * PORT_STRIDE))
  write_suite_script "${suite}" "${suite_base_port}"
  idx=$((idx + 1))
done
write_run_all_script

cat > "${EVAL_ROOT}/log_paths.txt" <<META
launcher=${EVAL_ROOT}/launcher.log
nohup=${EVAL_ROOT}/nohup.log
run_all=${EVAL_ROOT}/run_all_libero.sh
META

echo "======================================================"
echo "LIBERO accelerated evaluation"
echo "Checkpoint       : ${CKPT_PATH}"
echo "Suites           : ${SUITES[*]}"
echo "Base port        : ${BASE_PORT} (stride ${PORT_STRIDE})"
echo "Server GPUs      : ${SERVER_GPUS}"
echo "Endpoints/suite  : ${NUM_ENDPOINTS}"
echo "Render GPUs      : ${RENDER_GPUS}"
echo "Trials/task      : ${NUM_TRIALS_PER_TASK}"
echo "Output root      : ${EVAL_ROOT}"
echo "======================================================"
echo "Run commands have been written under each suite directory."
if [[ "${START_EVAL}" != "true" ]]; then
  echo "START_EVAL=${START_EVAL}; not launching evaluation."
  exit 0
fi
echo "Starting run_all_libero.sh..."

set +e
bash "${EVAL_ROOT}/run_all_libero.sh" > "${EVAL_ROOT}/nohup.log" 2>&1
code=$?
set -e
echo "${code}" > "${EVAL_ROOT}/launcher.exitcode"
if [[ "${code}" -ne 0 ]]; then
  echo "LIBERO evaluation failed with exit code ${code}: ${EVAL_ROOT}" >&2
  exit "${code}"
fi
echo "LIBERO accelerated evaluation complete: ${EVAL_ROOT}"
