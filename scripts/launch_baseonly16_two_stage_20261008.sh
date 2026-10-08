#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/share/home/u25505/apps/wangbo/LatentVLA/LaRA-VLA
TEXT_ROOT=$REPO/playground/Checkpoints/Text2Latent
PIPE_ID=spatial_baseonly16_two_stage_8008_8017_20261008
PIPE=$TEXT_ROOT/pipelines/$PIPE_ID
STAGE1_ID=Text2Latent_spatial_baseonly16_stage1_10k_20261008
NOAUX_ID=Text2Latent_action_from_spatial_baseonly16_noaux_stage2_40k_gpu8008_20261008
AUX_ID=Text2Latent_action_from_spatial_baseonly16_aux05_stage2_40k_gpu8017_20261008
STAGE1=$TEXT_ROOT/$STAGE1_ID
NOAUX=$TEXT_ROOT/$NOAUX_ID
AUX=$TEXT_ROOT/$AUX_ID
CACHE=$TEXT_ROOT/cache/spatial_vggt_base_only_518_resize_16_20261007
PYTHON_ENV=/share/home/u25505/apps/wangbo/LatentVLA/env/starvla
BASE_STAGE1=$TEXT_ROOT/Text2Latent_spatial_only_no_thinking_no_imgnext_stage1_10k_gpu8008_20261003/config.yaml
BASE_NOAUX=$TEXT_ROOT/Text2Latent_action_from_spatial_only_no_thinking_no_imgnext_stage2_40k_gpu8008_20261003/config.yaml
BASE_AUX=$TEXT_ROOT/Text2Latent_action_from_spatial_only_no_thinking_no_imgnext_spatialaux05_stage2_40k_gpu8017_20261004/config.yaml

mode=${1:?expected stage1, stage2_noaux, or stage2_aux}
mkdir -p "$PIPE" "$STAGE1" "$NOAUX" "$AUX"
exec >> "$PIPE/${mode}.console.log" 2>&1
echo "$(date --iso-8601=seconds) start mode=$mode host=$(hostname) pid=$$"

export CUDA_HOME=/share/apps/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$REPO"

pick_port() {
  "$PYTHON_ENV/bin/python" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

wait_for_gpus() {
  local log_path=$1
  while true; do
    local free=1
    local snapshot
    snapshot=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    while IFS=',' read -r index used util; do
      used=$(echo "$used" | tr -d ' ')
      if [ "$used" -gt 1024 ]; then
        free=0
      fi
    done <<< "$snapshot"
    echo "$(date --iso-8601=seconds) gpu_gate free=$free $snapshot" >> "$log_path"
    [ "$free" -eq 1 ] && return 0
    sleep 60
  done
}

if [ "$mode" = "stage1" ]; then
  mkdir -p "$PIPE/code_snapshot"
  git rev-parse HEAD > "$PIPE/git_commit.txt"
  git status --short -uno > "$PIPE/git_status.txt"
  git diff --binary > "$PIPE/code.diff"
  nvidia-smi > "$PIPE/nvidia_smi.before.txt"

  "$PYTHON_ENV/bin/python" - "$BASE_STAGE1" "$STAGE1/config.yaml" "$STAGE1_ID" "$STAGE1" "$CACHE" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
src, dst, run_id, output_dir, cache = sys.argv[1:]
cfg = OmegaConf.load(src)
def put(key, value):
    OmegaConf.update(cfg, key, value, merge=False, force_add=True)
put("run_id", run_id)
put("run_root_dir", str(Path(output_dir).parent))
put("output_dir", output_dir)
put("framework.spatial.view_mode", "base_only")
put("framework.spatial.target_cache_dir", cache)
put("framework.spatial.target_cache_only", True)
put("framework.spatial.token_count", 16)
put("datasets.vla_data.bridge_reasoning.spatial_view_mode", "base_only")
put("datasets.vla_data.bridge_reasoning.spatial_count", 16)
put("datasets.vla_data.bridge_reasoning.spatial_query_mode", "visual_prefix_joint")
put("framework.training_stage", "reasoning_only")
put("framework.spatial.use_teacher", True)
put("framework.spatial.loss_weight", 0.1)
put("trainer.pretrained_checkpoint", None)
put("trainer.start_step", 0)
put("trainer.max_train_steps", 10000)
put("trainer.gradient_accumulation_steps", 1)
put("trainer.save_interval", 1000)
put("trainer.eval_interval", 0)
OmegaConf.save(cfg, dst)
PY

  "$PYTHON_ENV/bin/python" - "$STAGE1/config.yaml" "$PIPE/contract_validation.txt" "$CACHE" <<'PY'
import json, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from laravla.model.modules.vlm.QWen3 import _QWen3_VL_Interface
from laravla.dataloader.gr00t_lerobot.bridge_reasoning_formatter import BridgeReasoningFormatter
cfg = OmegaConf.load(sys.argv[1])
out = Path(sys.argv[2])
cache = Path(sys.argv[3])
formatter = BridgeReasoningFormatter(
    include_bbox=False, include_action_tokens=False, include_img_next=False,
    include_spatial=True, spatial_count=16,
    spatial_query_mode="visual_prefix_joint", spatial_view_mode="base_only",
)
text = formatter.format("pick up the object", {})
assert text.count("<spatial_base>") == 16
assert text.count("<spatial_wrist>") == 0
base_id, wrist_id = 101, 102
ids = torch.tensor([[11] + [base_id] * 16 + [12, 13]])
dummy = _QWen3_VL_Interface.__new__(_QWen3_VL_Interface)
dummy.spatial_view_mode = "base_only"
dummy.spatial_query_mode = "visual_prefix_joint"
dummy.spatial_base_token_id = base_id
dummy.spatial_wrist_token_id = wrist_id
dummy.img_next_token_id = None
mask = dummy._build_img_next_full_attention_mask(torch.ones_like(ids), ids)
allow = mask[0, 0] == 0
assert bool(allow[1:17, 1:17].all())
assert not bool(allow[1:17, 17:].any())
assert bool(allow[17:, 1:17].all())
assert cache.is_dir()
contract = [
    "view_mode=base_only",
    "spatial_tokens=16",
    "base_tokens=16",
    "wrist_tokens=0",
    "joint_attention=base16_full_attention",
    "causal_hierarchy=2d_image_to_base_spatial_to_language_action",
    "spatial_to_future_blocked=passed",
    "base_only_cache_present=passed",
]
out.write_text("\n".join(contract) + "\n", encoding="utf-8")
print("\n".join(contract))
PY

  "$PYTHON_ENV/bin/python" "$REPO/scripts/precompute_vggt_base_cache.py" --validate_only --cache_dir "$CACHE" --expected_count 273465 --expected_shards 8 > "$PIPE/base_only_cache_validation.txt"
  wait_for_gpus "$PIPE/stage1_gpu_wait.log"
  echo "stage=stage1_baseonly16 status=started" > "$PIPE/pipeline_state.txt"
  port=$(pick_port)
  set +e
  "$PYTHON_ENV/bin/accelerate" launch --config_file "$REPO/decoder/accelerate_deepspeed_zero2.yaml" --num_processes 8 --main_process_port "$port" "$REPO/laravla/training/train.py" --config_yaml "$STAGE1/config.yaml" > "$STAGE1/train.log" 2>&1
  code=$?
  set -e
  echo "$code" > "$STAGE1/train.exitcode"
  if [ "$code" -ne 0 ]; then
    echo "stage=stage1_baseonly16 status=failed code=$code" >> "$PIPE/pipeline_state.txt"
    exit "$code"
  fi
  test -s "$STAGE1/checkpoints/steps_10000_pytorch_model.pt"
  echo "stage=stage1_baseonly16 status=completed checkpoint=$STAGE1/checkpoints/steps_10000_pytorch_model.pt" > "$PIPE/pipeline_state.txt"
  case "$(hostname)" in
    gpu8008*)
      systemctl --user start lara-spatial-baseonly16-noaux-gpu8008-20261008.service
      ssh gpu8017 "systemctl --user daemon-reload && systemctl --user start lara-spatial-baseonly16-aux-gpu8017-20261008.service"
      ;;
    gpu8017*)
      ssh gpu8008 "systemctl --user daemon-reload && systemctl --user start lara-spatial-baseonly16-noaux-gpu8008-20261008.service"
      systemctl --user start lara-spatial-baseonly16-aux-gpu8017-20261008.service
      ;;
    *)
      echo "unsupported Stage 1 host: $(hostname)" >&2
      exit 1
      ;;
  esac
  exit 0
fi

if [ "$mode" = "stage2_noaux" ]; then
  test -s "$STAGE1/checkpoints/steps_10000_pytorch_model.pt"
  "$PYTHON_ENV/bin/python" - "$BASE_NOAUX" "$NOAUX/config.yaml" "$NOAUX_ID" "$NOAUX" "$CACHE" "$STAGE1/checkpoints/steps_10000_pytorch_model.pt" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
src, dst, run_id, output_dir, cache, ckpt = sys.argv[1:]
cfg = OmegaConf.load(src)
def put(key, value):
    OmegaConf.update(cfg, key, value, merge=False, force_add=True)
put("run_id", run_id)
put("run_root_dir", str(Path(output_dir).parent))
put("output_dir", output_dir)
put("framework.spatial.view_mode", "base_only")
put("framework.spatial.target_cache_dir", cache)
put("framework.spatial.target_cache_only", True)
put("framework.spatial.token_count", 16)
put("datasets.vla_data.bridge_reasoning.spatial_view_mode", "base_only")
put("datasets.vla_data.bridge_reasoning.spatial_count", 16)
put("datasets.vla_data.bridge_reasoning.spatial_query_mode", "visual_prefix_joint")
put("framework.training_stage", "full")
put("framework.spatial.use_teacher", False)
put("framework.spatial.loss_weight", 0.0)
put("trainer.pretrained_checkpoint", ckpt)
put("trainer.start_step", 0)
put("trainer.max_train_steps", 40000)
put("trainer.gradient_accumulation_steps", 1)
put("trainer.save_interval", 5000)
put("trainer.eval_interval", 2000)
OmegaConf.save(cfg, dst)
PY
  wait_for_gpus "$PIPE/noaux_gpu_wait.log"
  echo "stage=stage2_noaux status=started" >> "$PIPE/pipeline_state.txt"
  port=$(pick_port)
  set +e
  "$PYTHON_ENV/bin/accelerate" launch --config_file "$REPO/decoder/accelerate_deepspeed_zero2.yaml" --num_processes 8 --main_process_port "$port" "$REPO/laravla/training/train.py" --config_yaml "$NOAUX/config.yaml" > "$NOAUX/train.log" 2>&1
  code=$?
  set -e
  echo "$code" > "$NOAUX/train.exitcode"
  if [ "$code" -ne 0 ]; then
    echo "stage=stage2_noaux status=failed code=$code" >> "$PIPE/pipeline_state.txt"
    exit "$code"
  fi
  test -s "$NOAUX/checkpoints/steps_40000_pytorch_model.pt"
  test -s "$NOAUX/final_model/pytorch_model.pt"
  echo "stage=stage2_noaux status=completed" >> "$PIPE/pipeline_state.txt"
  exit 0
fi

if [ "$mode" = "stage2_aux" ]; then
  test -s "$STAGE1/checkpoints/steps_10000_pytorch_model.pt"
  "$PYTHON_ENV/bin/python" - "$BASE_AUX" "$AUX/config.yaml" "$AUX_ID" "$AUX" "$CACHE" "$STAGE1/checkpoints/steps_10000_pytorch_model.pt" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
src, dst, run_id, output_dir, cache, ckpt = sys.argv[1:]
cfg = OmegaConf.load(src)
def put(key, value):
    OmegaConf.update(cfg, key, value, merge=False, force_add=True)
put("run_id", run_id)
put("run_root_dir", str(Path(output_dir).parent))
put("output_dir", output_dir)
put("framework.spatial.view_mode", "base_only")
put("framework.spatial.target_cache_dir", cache)
put("framework.spatial.target_cache_only", True)
put("framework.spatial.token_count", 16)
put("datasets.vla_data.bridge_reasoning.spatial_view_mode", "base_only")
put("datasets.vla_data.bridge_reasoning.spatial_count", 16)
put("datasets.vla_data.bridge_reasoning.spatial_query_mode", "visual_prefix_joint")
put("framework.training_stage", "full")
put("framework.spatial.use_teacher", True)
put("framework.spatial.loss_weight", 0.05)
put("trainer.pretrained_checkpoint", ckpt)
put("trainer.start_step", 0)
put("trainer.max_train_steps", 40000)
put("trainer.gradient_accumulation_steps", 1)
put("trainer.save_interval", 5000)
put("trainer.eval_interval", 2000)
OmegaConf.save(cfg, dst)
PY
  wait_for_gpus "$PIPE/aux_gpu_wait.log"
  echo "stage=stage2_aux status=started" >> "$PIPE/pipeline_state.txt"
  port=$(pick_port)
  set +e
  "$PYTHON_ENV/bin/accelerate" launch --config_file "$REPO/decoder/accelerate_deepspeed_zero2.yaml" --num_processes 8 --main_process_port "$port" "$REPO/laravla/training/train.py" --config_yaml "$AUX/config.yaml" > "$AUX/train.log" 2>&1
  code=$?
  set -e
  echo "$code" > "$AUX/train.exitcode"
  if [ "$code" -ne 0 ]; then
    echo "stage=stage2_aux status=failed code=$code" >> "$PIPE/pipeline_state.txt"
    exit "$code"
  fi
  test -s "$AUX/checkpoints/steps_40000_pytorch_model.pt"
  test -s "$AUX/final_model/pytorch_model.pt"
  echo "stage=stage2_aux status=completed" >> "$PIPE/pipeline_state.txt"
  exit 0
fi
