#!/usr/bin/env bash
export LATENTVLA_ROOT="${LATENTVLA_ROOT:-/share/home/u25505/apps/wangbo/LatentVLA}"
export LIBERO_HOME="${LIBERO_HOME:-${LATENTVLA_ROOT}/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/share/home/u25505/.libero}"
export LARAVLA_BASE_VLM="${LARAVLA_BASE_VLM:-${LATENTVLA_ROOT}/LaRA-VLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
export LARAVLA_PYTHON="${LARAVLA_PYTHON:-${LATENTVLA_ROOT}/env/starvla/bin/python}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-${LATENTVLA_ROOT}/env/libero/bin/python}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
