## SimplerEnv

This directory contains the recommended SimplerEnv evaluation entrypoints.

Main files:

- `bridge_eval.sh`: recommended parallel evaluation for one checkpoint
- `run_all_ckpts_bridge.sh`: batch evaluation for many checkpoints
- `start_simpler_env.py`: simulator-side evaluation entrypoint
- `model2simpler_interface.py`: SimplerEnv-side policy adapter
- `test_your_simplerEnv.py`: quick environment check

## Prerequisites

You usually need:

- one trained checkpoint
- one LaRA-VLA Python environment (the code package namespace is `laravla`)
- one SimplerEnv Python environment
- `SimplerEnv_PATH`

To set up the environment, please first follow the official [SimplerEnv repository](https://github.com/simpler-env/SimplerEnv) to install the base simpler_env environment.

Afterwards, inside the simpler_env environment, install the following dependencies:

conda activate simpler_env
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4

Useful checks:

```bash
python examples/SimplerEnv/test_your_simplerEnv.py
```

If this script succeeds, your SimplerEnv setup is likely usable.

## Recommended: Parallel Evaluation

```bash
laravla_python=/path/to/laravla/python \
sim_python=/path/to/simpler_env/python \
SimplerEnv_PATH=/path/to/SimplerEnv \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/SimplerEnv/bridge_eval.sh /abs/path/to/checkpoint.pt
```

The script launches policy servers and SimplerEnv tasks in parallel, then writes
logs under the checkpoint directory unless `LOG_DIR` is overridden.

## Batch Evaluation for Many Checkpoints

```bash
laravla_python=/path/to/laravla/python \
sim_python=/path/to/simpler_env/python \
SimplerEnv_PATH=/path/to/SimplerEnv \
bash examples/SimplerEnv/run_all_ckpts_bridge.sh /abs/path/to/checkpoints
```
