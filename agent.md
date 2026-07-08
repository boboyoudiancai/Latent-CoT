# LaRA-VLA Agent Notes

Repository-level notes for future sessions on liuyue.

## Entry

```bash
ssh liuyue-8-130-97-174
cd /home/liuyue/LaRA-VLA
```

The active repository is `/home/liuyue/LaRA-VLA`.

## Environment

Use the existing `starvla` environment:

```bash
source /home/liuyue/miniconda3/etc/profile.d/conda.sh
conda activate starvla
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
```

Before launching GPU work:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
ps -ef | grep -E "torchrun|train.py|train_latent_decoder.py" | grep -v grep
```

If overwriting files under OSS-backed checkpoint paths fails with `scp`, upload to `/tmp` first, then `cp` into the target path.

## Key Paths

```text
repo:        /home/liuyue/LaRA-VLA
logs:        /home/liuyue/LaRA-VLA/trainlog
checkpoints: /home/liuyue/LaRA-VLA/playground/Checkpoints
Text2Latent: /home/liuyue/LaRA-VLA/playground/Checkpoints/Text2Latent
pretrained:  /home/liuyue/LaRA-VLA/playground/Pretrained_models
base VLM:    /home/liuyue/LaRA-VLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
fast tok:    /home/liuyue/LaRA-VLA/playground/Pretrained_models/fast
LIBERO data: /mnt/oss/liuyue/wangbo/data/libero_cot
steps cache: /home/liuyue/LaRA-VLA/playground/Checkpoints/Text2Latent/steps_cache/libero_vlm
```

Always pass the LIBERO data root explicitly. Do not rely on `data/libero_lerobot`, which may be a stale symlink.

## Run Rules

Every training run should have:

```text
output dir:  <checkpoint_root>/<run_id>
command:     <checkpoint_root>/<run_id>/run_command.sh
log:         trainlog/<run_id>_<timestamp>.log
```

`run_command.sh` is the source of truth. Keep it runnable from a fresh shell and include environment setup, `cd /home/liuyue/LaRA-VLA`, and the full train command.

Use existing project naming:

```text
<task>_<source>_<important_settings>
```

Include batch size and max steps in log names when relevant, for example:

```text
trainlog/<run_id>_bs112_40000_<timestamp>.log
```

Background launch pattern:

```bash
mkdir -p "<output_dir>"
log="trainlog/<run_id>_$(date +%Y%m%d_%H%M%S).log"
nohup bash "<output_dir>/run_command.sh" > "$log" 2>&1 &
```

## Training Defaults

For LIBERO/Text2Latent action-head runs, match the existing project convention unless the user asks otherwise:

```text
launcher: torchrun
gpus: 8
per-device batch: 14
global batch: 112
config: laravla/config/training/libero.yaml
data root: /mnt/oss/liuyue/wangbo/data/libero_cot
fast tokenizer: /home/liuyue/LaRA-VLA/playground/Pretrained_models/fast
base VLM: /home/liuyue/LaRA-VLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
img_next: disabled unless explicitly requested
steps cache write: false unless rebuilding cache
```

For action-head training from a VLM checkpoint, the usual important overrides are:

```text
--framework.training_stage full
--datasets.vla_data.bridge_reasoning.stage 4
--datasets.vla_data.bridge_reasoning.include_img_next false
--datasets.vla_data.data_root_dir /mnt/oss/liuyue/wangbo/data/libero_cot
--datasets.vla_data.bridge_annotations.fast_tokenizer_name /home/liuyue/LaRA-VLA/playground/Pretrained_models/fast
--framework.qwenvl.base_vlm /home/liuyue/LaRA-VLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
--framework.img_next.enable false
--framework.img_next.loss_weight 0
--framework.img_next.use_teacher false
--framework.latent_reasoning.vlm_loss_weight 0
--trainer.reload_modules qwen_vl_interface
```

Set `--trainer.pretrained_checkpoint`, `--trainer.max_train_steps`, `--trainer.save_interval`, and the run id according to the user request.

## Verification

After launch, check:

```bash
tail -n 120 trainlog/<log_file>
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
ps -ef | grep -E "<run_id>|torchrun|train.py" | grep -v grep
```

A run is not confirmed until the log passes dataset loading and shows training progress or loss lines.

## Git

Check status before editing:

```bash
git status --short --branch
```

The branch may be ahead/behind origin. Large `D` entries under `playground/Checkpoints` or `playground/Pretrained_models` are usually artifact/symlink noise. Do not stage them unless explicitly requested.

