#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/share/home/u25505/apps/wangbo/LatentVLA/LaRA-VLA
TEXT_ROOT=$REPO/playground/Checkpoints/Text2Latent
PIPE_ID=spatial_baseonly16_stage1_l2_gpu8019_20261012
RUN_ID=Text2Latent_spatial_baseonly16_stage1_10k_l2_gpu8019_20261012
PIPE=$TEXT_ROOT/pipelines/$PIPE_ID
RUN=$TEXT_ROOT/$RUN_ID
BASE_CONFIG=$TEXT_ROOT/Text2Latent_spatial_baseonly16_stage1_10k_20261008/config.yaml
CACHE=$TEXT_ROOT/cache/spatial_vggt_base_only_518_resize_16_20261007
PYTHON_ENV=/share/home/u25505/apps/wangbo/LatentVLA/env/starvla

test -s "$BASE_CONFIG"
test -d "$CACHE"
test ! -e "$RUN"
mkdir -p "$PIPE" "$RUN"
exec >> "$PIPE/launch.console.log" 2>&1
echo "$(date --iso-8601=seconds) start host=$(hostname) requested=gpu8019 pid=$$"

export CUDA_HOME=/share/apps/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$REPO"

git rev-parse HEAD > "$PIPE/git_commit.txt"
git status --short -uno > "$PIPE/git_status.txt"
git diff -- laravla/model/framework/laravla.py > "$PIPE/spatial_loss_l2.code.diff"
nvidia-smi > "$PIPE/nvidia_smi.before.txt"
cp -p "$BASE_CONFIG" "$PIPE/base_config.yaml"

"$PYTHON_ENV/bin/python" - "$BASE_CONFIG" "$RUN/config.yaml" "$RUN_ID" "$RUN" <<'PY'
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf

src, dst, run_id, output_dir = sys.argv[1:]
base = OmegaConf.load(src)
cfg = OmegaConf.load(src)

def put(key, value):
    OmegaConf.update(cfg, key, value, merge=False, force_add=True)

put("run_id", run_id)
put("output_dir", output_dir)
put("framework.spatial.loss_type", "l2")
OmegaConf.save(cfg, dst)

def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        rows = []
        for key in sorted(set(a) | set(b)):
            child = f"{path}.{key}" if path else key
            rows.extend(diff(a.get(key), b.get(key), child))
        return rows
    return [] if a == b else [{"key": path, "baseline": a, "experiment": b}]

base_data = OmegaConf.to_container(base, resolve=True)
cfg_data = OmegaConf.to_container(cfg, resolve=True)
Path(output_dir, "config_diff.json").write_text(
    json.dumps(diff(base_data, cfg_data), indent=2, ensure_ascii=False), encoding="utf-8"
)
PY

"$PYTHON_ENV/bin/python" - "$RUN/config.yaml" "$PIPE/contract_validation.txt" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
out = Path(sys.argv[2])
spatial = cfg.framework.spatial
data = cfg.datasets.vla_data
checks = {
    "loss_type": spatial.loss_type == "l2",
    "loss_weight_unchanged": float(spatial.loss_weight) == 0.1,
    "training_stage": cfg.framework.training_stage == "reasoning_only",
    "base_tokens_16": data.bridge_reasoning.spatial_count == 16 and data.bridge_reasoning.spatial_view_mode == "base_only",
    "wrist_tokens_0": "wrist" not in data.bridge_reasoning.spatial_view_mode,
    "visual_prefix_joint": data.bridge_reasoning.spatial_query_mode == "visual_prefix_joint",
    "batch_14": data.per_device_batch_size == 14,
    "gradient_accumulation_1": cfg.trainer.gradient_accumulation_steps == 1,
    "steps_10000": cfg.trainer.max_train_steps == 10000,
    "img_next_disabled": not cfg.framework.img_next.enable and float(cfg.framework.img_next.loss_weight) == 0.0,
    "latent_decoder_disabled": not cfg.framework.latent_decoder.enable,
    "state_disabled": not data.include_state,
    "teacher_enabled": spatial.use_teacher and spatial.target_cache_only,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("contract failed: " + ", ".join(failed))
lines = [f"{name}=passed" for name in checks]
lines.append("controlled_change=framework.spatial.loss_type cosine->l2")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

"$PYTHON_ENV/bin/python" - "$REPO/laravla/model/framework/laravla.py" <<'PY'
import importlib.util
import sys
import torch
from types import SimpleNamespace

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("lara_spatial_loss_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class QwenStub:
    spatial_target_tokens = 1
    def get_external_spatial_target(self, images, device, cache_keys=None):
        return torch.tensor([[[[2.0, 4.0]]]], device=device)
    def project_spatial_prediction(self, pred):
        return pred

for loss_type, expected, tolerance in (("cosine", 0.0, 0.02), ("l2", 2.5, 1e-6)):
    model = module.Qwen_GR00T.__new__(module.Qwen_GR00T)
    model.config = SimpleNamespace(framework=SimpleNamespace(spatial={"loss_type": loss_type}))
    model.qwen_vl_interface = QwenStub()
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    base_mask = torch.tensor([[True, False]])
    loss = model._compute_spatial_loss(hidden, [None], base_mask, None, view_mode="base_only")
    actual = float(loss.detach())
    if abs(actual - expected) > tolerance:
        raise SystemExit(f"loss test failed: type={loss_type} expected={expected} actual={actual}")
print("spatial_loss_if_else_test=passed cosine=0.0 l2=2.5")
PY

port=$("$PYTHON_ENV/bin/python" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)
echo "stage=spatial_baseonly16_stage1_l2 status=started port=$port" > "$PIPE/pipeline_state.txt"

set +e
"$PYTHON_ENV/bin/accelerate" launch \
  --config_file "$REPO/decoder/accelerate_deepspeed_zero2.yaml" \
  --num_processes 8 \
  --main_process_port "$port" \
  "$REPO/laravla/training/train.py" \
  --config_yaml "$RUN/config.yaml" > "$RUN/train.log" 2>&1
code=$?
set -e
echo "$code" > "$RUN/train.exitcode"

if [ "$code" -ne 0 ]; then
  echo "stage=spatial_baseonly16_stage1_l2 status=failed code=$code" >> "$PIPE/pipeline_state.txt"
  exit "$code"
fi
test -s "$RUN/checkpoints/steps_10000_pytorch_model.pt"
echo "stage=spatial_baseonly16_stage1_l2 status=completed checkpoint=$RUN/checkpoints/steps_10000_pytorch_model.pt" >> "$PIPE/pipeline_state.txt"
