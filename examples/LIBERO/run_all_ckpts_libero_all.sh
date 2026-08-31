#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/qwen_cache}"
# Match eval_libero_all.sh: use OSMesa to avoid EGL issues when needed.
# export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
DEFAULT_CKPT_DIR="${DEFAULT_CKPT_DIR:-}"
CKPT_DIR="${1:-${YOUR_CKPT_DIR:-${DEFAULT_CKPT_DIR:-}}}"
MIN_STEP_ARG="${2:-}"

if [[ -z "${CKPT_DIR}" ]]; then
  echo "❌ Please provide a checkpoints directory, for example: bash $0 /abs/path/to/checkpoints [MIN_STEP]" >&2
  exit 1
fi
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "❌ Checkpoint directory does not exist: ${CKPT_DIR}" >&2
  exit 1
fi

MIN_STEP="${MIN_STEP:-0}"
MAX_STEP="${MAX_STEP:-}"
if [[ -n "${MIN_STEP_ARG}" ]]; then
  MIN_STEP="${MIN_STEP_ARG}"
fi
if [[ -n "${MIN_STEP}" ]] && ! [[ "${MIN_STEP}" =~ ^[0-9]+$ ]]; then
  echo "❌ MIN_STEP must be a non-negative integer. Got: ${MIN_STEP}" >&2
  exit 1
fi
if [[ -n "${MAX_STEP}" ]] && ! [[ "${MAX_STEP}" =~ ^[0-9]+$ ]]; then
  echo "❌ MAX_STEP must be a non-negative integer or empty. Got: ${MAX_STEP}" >&2
  exit 1
fi

BASE_PORT="${BASE_PORT:-10093}"
PORT_STRIDE="${PORT_STRIDE:-50}"
SAVE_VIDEOS="${SAVE_VIDEOS:-false}"
STOP_ON_FAIL="${STOP_ON_FAIL:-false}"

if ! [[ "${BASE_PORT}" =~ ^[0-9]+$ ]]; then
  echo "❌ BASE_PORT must be a non-negative integer. Got: ${BASE_PORT}" >&2
  exit 1
fi
if ! [[ "${PORT_STRIDE}" =~ ^[0-9]+$ ]]; then
  echo "❌ PORT_STRIDE must be a non-negative integer. Got: ${PORT_STRIDE}" >&2
  exit 1
fi

echo "======================================================"
echo "📊 Batch LIBERO Eval (serial per-ckpt)"
echo "Checkpoint dir : ${CKPT_DIR}"
echo "Base port      : ${BASE_PORT} (stride ${PORT_STRIDE})"
echo "Min step       : ${MIN_STEP}"
echo "Max step       : ${MAX_STEP:-<unset>}"
echo "Save videos    : ${SAVE_VIDEOS}"
echo "Stop on fail   : ${STOP_ON_FAIL}"
echo "======================================================"

shopt -s nullglob
mapfile -t CKPTS < <(printf '%s\n' "${CKPT_DIR}"/steps_*_pytorch_model.pt | sort -V)
if (( ${#CKPTS[@]} == 0 )); then
  echo "❌ No steps_*_pytorch_model.pt files found in: ${CKPT_DIR}" >&2
  exit 1
fi

FILTERED_CKPTS=()
for ckpt in "${CKPTS[@]}"; do
  base="$(basename "${ckpt}")"
  if [[ "${base}" =~ ^steps_([0-9]+)_pytorch_model\.pt$ ]]; then
    step="${BASH_REMATCH[1]}"
    if (( step < MIN_STEP )); then
      continue
    fi
    if [[ -n "${MAX_STEP}" ]] && (( step > MAX_STEP )); then
      continue
    fi
    FILTERED_CKPTS+=("${ckpt}")
  else
    echo "⚠️ Skipping checkpoint with non-standard name: ${ckpt}" >&2
  fi
done

if (( ${#FILTERED_CKPTS[@]} == 0 )); then
  echo "❌ No checkpoints found within the requested step range (MIN_STEP=${MIN_STEP}, MAX_STEP=${MAX_STEP:-<unset>})" >&2
  exit 1
fi

idx=0
for ckpt in "${FILTERED_CKPTS[@]}"; do
  ckpt_name="$(basename "${ckpt%.*}")"
  ckpt_base_port=$((BASE_PORT + idx * PORT_STRIDE))
  echo "------------------------------------------------------"
  echo "▶️  Eval ${ckpt_name} | base_port=${ckpt_base_port}"

  if ! (
    YOUR_CKPT="${ckpt}" \
    BASE_PORT="${ckpt_base_port}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" \
      bash examples/LIBERO/eval_libero_all.sh "${ckpt}"
  ); then
    echo "⚠️  Eval failed: ${ckpt}" >&2
    if [[ "${STOP_ON_FAIL}" == "true" ]]; then
      exit 1
    fi
  fi

  idx=$((idx + 1))
done

echo "✅ All checkpoint evaluations completed: ${CKPT_DIR}"
