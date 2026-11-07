# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

deepspeed_plugin = None if os.environ.get("STARVLA_DISABLE_DEEPSPEED") == "1" else DeepSpeedPlugin()
gradient_accumulation_plugin = GradientAccumulationPlugin(
    num_steps=int(os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "1")),
    sync_each_batch=deepspeed_plugin is not None,
)
accelerator = Accelerator(
    deepspeed_plugin=deepspeed_plugin,
    gradient_accumulation_plugin=gradient_accumulation_plugin,
)
accelerator.print(accelerator.state)


def _unwrap_model(accelerator, model):
    """Avoid importing a broken optional DeepSpeed install in single-GPU mode."""
    if os.environ.get("STARVLA_DISABLE_DEEPSPEED") == "1":
        return model.module if hasattr(model, "module") else model
    return accelerator.unwrap_model(model)


def _get_cpu_state_dict(accelerator, model):
    """Copy parameters to CPU without gathering a second full GPU state dict.

    ZeRO-2 keeps model parameters resident on every rank.  Calling
    ``accelerator.get_state_dict`` while the training graph is also resident
    can transiently allocate another full GPU state dict and OOM at checkpoint
    time.  The main process instead copies each detached parameter directly to
    CPU, where the resulting checkpoint is serialized.
    """
    module = _unwrap_model(accelerator, model)
    return {
        name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for name, value in module.state_dict().items()
    }


def _atomic_save_state_dict(state_dict, checkpoint_path, save_format):
    """Write a checkpoint atomically so interrupted saves are never resumable."""
    temporary_path = checkpoint_path + ".tmp"
    try:
        if save_format == "safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, temporary_path)
        elif save_format == "pt":
            torch.save(state_dict, temporary_path)
        else:
            raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
        os.replace(temporary_path, checkpoint_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def _validate_thinking_stage1_checkpoint(checkpoint_path):
    """Reject an old pre-thinking model as a Stage 2 initialization."""
    if str(checkpoint_path).endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
            keys = list(checkpoint.keys())
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        keys = list(checkpoint.keys())
        del checkpoint
    if not any(key.startswith("latent_decoder.model.") for key in keys):
        raise ValueError("Stage 2 initialization must be a full Stage 1 thinking checkpoint with decoder weights")


def _thinking_stage1_module_names(model):
    """Transfer the backbone plus the spatial projector only when it exists."""
    modules = ["qwen_vl_interface"]
    if getattr(model, "spatial_forcing_projector", None) is not None:
        modules.append("spatial_forcing_projector")
    return tuple(modules)


def _load_thinking_stage1_modules(model, checkpoint_path):
    """Strictly transfer Stage 1 features without the decoder or action head."""
    _validate_thinking_stage1_checkpoint(checkpoint_path)
    if str(checkpoint_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        checkpoint = load_file(str(checkpoint_path), device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)

    modules = _thinking_stage1_module_names(model)
    if "spatial_forcing_projector" not in modules and any(
        key.startswith("spatial_forcing_projector.") for key in checkpoint
    ):
        raise ValueError("Spatial-disabled Stage 2 requires a spatial-disabled Stage 1 checkpoint")
    states = {}
    for name in modules:
        module = getattr(model, name, None)
        if module is None:
            raise ValueError(f"Stage 2 model is missing required module {name}")
        prefix = name + "."
        state = {key[len(prefix):]: value for key, value in checkpoint.items() if key.startswith(prefix)}
        expected = module.state_dict()
        missing, unexpected = set(expected) - set(state), set(state) - set(expected)
        wrong_shapes = [key for key in set(expected) & set(state) if expected[key].shape != state[key].shape]
        if not state or missing or unexpected or wrong_shapes:
            raise ValueError(
                f"Incompatible Stage 1 module {name}: missing={sorted(missing)[:5]}, "
                f"unexpected={sorted(unexpected)[:5]}, shape_mismatch={sorted(wrong_shapes)[:5]}"
            )
        states[name] = state
    # Validate every module before changing any, and leave the action head untouched.
    for name in modules:
        getattr(model, name).load_state_dict(states[name], strict=True)
    return model


def _validate_thinking_accumulation(cfg, accelerator):
    latent_cfg = cfg.framework.get("latent_reasoning", {})
    if bool(latent_cfg.get("enable", False)):
        configured = int(cfg.trainer.gradient_accumulation_steps)
        actual = int(accelerator.gradient_accumulation_steps)
        if configured != actual:
            raise ValueError(
                f"Thinking training requires actual gradient accumulation to match config: "
                f"configured={configured}, accelerator={actual}. Use the provided launcher."
            )


def _prune_completed_training_states(checkpoint_dir, keep_last, completed_steps):
    """Keep recent optimizer states within this run; never prune weight files or links."""
    if keep_last is None:
        return []
    if isinstance(keep_last, bool) or int(keep_last) != keep_last or keep_last < 2:
        raise ValueError("trainer.keep_last_training_states must be an integer >= 2")
    root = Path(checkpoint_dir).resolve(strict=True)
    candidates = []
    for entry in root.iterdir():
        match = re.fullmatch(r"steps_(\d+)_training_state", entry.name)
        if not match or entry.is_symlink() or not entry.is_dir():
            continue
        step = int(match.group(1))
        metadata = entry / "trainer_state.json"
        if step > completed_steps or not metadata.is_file() or metadata.is_symlink():
            continue
        with metadata.open(encoding="utf-8") as handle:
            if json.load(handle).get("completed_steps") != step:
                raise ValueError(f"Checkpoint retention metadata mismatch: {entry}")
        candidates.append((step, entry))
    candidates.sort(key=lambda row: row[0])
    if not any(step == completed_steps for step, _ in candidates):
        raise ValueError("Checkpoint retention requires the newly completed state")
    removed = []
    for _, entry in candidates[:-int(keep_last)]:
        target = entry.resolve(strict=True)
        if target.parent != root or target != entry or entry.is_symlink():
            raise ValueError(f"Unsafe checkpoint retention target: {entry}")
        shutil.rmtree(target)
        removed.append(str(target))
    return removed

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=bool(cfg.trainer.optimizer.get("fused", True)),
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    scheduler_scale = accelerator.num_processes if cfg.trainer.get("scale_scheduler_by_world_size", False) else 1
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps * scheduler_scale,
        num_training_steps=cfg.trainer.max_train_steps * scheduler_scale,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self._future_ema_updates = 0
        self.data_batches_consumed = 0
        self.vla_epoch_count = 0
        self._batches_in_current_epoch = 0
        self._resume_state_dir = None
        self._resume_metadata = None
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._future_ema_updates = self.completed_steps

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        # Hugging Face from_pretrained() returns nested models in eval mode.
        # Explicitly activate training before DeepSpeed wraps the model.
        self.model.train()
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader, self.lr_scheduler = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.lr_scheduler,
        )

        latent_cfg = self.config.framework.get("latent_reasoning", {})
        ds_accumulation = getattr(self.model, "gradient_accumulation_steps", None)
        if bool(latent_cfg.get("enable", False)) and callable(ds_accumulation):
            actual_ds_steps = int(ds_accumulation())
            if actual_ds_steps != int(self.accelerator.gradient_accumulation_steps):
                raise ValueError(
                    f"DeepSpeed gradient accumulation {actual_ds_steps} does not match "
                    f"Accelerate {self.accelerator.gradient_accumulation_steps}"
                )

        # Preserve the old explicit behaviour for an opt-in legacy
        # weights-only continuation.  Strict resume restores the scheduler's
        # serialized state later, after positioning the dataloader.
        if self.resume_from_checkpoint and not self._resume_state_dir and self.completed_steps > 0:
            self._adjust_lr_scheduler_for_weights_only_resume()

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases (best-effort; must not block training)."""
        self._wandb_enabled = False
        if os.environ.get("WANDB_MODE") == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            self.accelerator.wait_for_everyone()
            return
        if self.accelerator.is_main_process:
            try:
                wandb.init(
                    name=self.config.run_id,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                )
                self._wandb_enabled = True
            except Exception as exc:
                logger.warning(f"W&B init failed; continuing without W&B: {exc}")
                self._wandb_enabled = False
        # Rendezvous after rank-0 W&B init. Otherwise a slow or failing init on
        # rank 0 lets the other ranks reach the first collective alone and
        # eventually hit an NCCL watchdog timeout.
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Resolve the checkpoint to restore before distributed wrapping.

        A strict resume uses an Accelerate state directory, not a standalone
        model file.  The actual ``load_state`` call must happen only after
        model, optimizer, dataloader, and scheduler have been prepared.
        """
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        latent_cfg = self.config.framework.get("latent_reasoning", {})
        if (bool(latent_cfg.get("require_base_initialization", False))
                and pretrained_checkpoint and not is_resume):
            raise ValueError("This thinking experiment must start from base pretrained VLM weights, not a training checkpoint")
        if (bool(latent_cfg.get("require_stage1_checkpoint", False))
                and not pretrained_checkpoint and not is_resume):
            raise ValueError("Stage 2 requires trainer.pretrained_checkpoint to point to a completed Stage 1 checkpoint")
        if bool(latent_cfg.get("require_stage1_checkpoint", False)) and pretrained_checkpoint and not is_resume:
            _validate_thinking_stage1_checkpoint(pretrained_checkpoint)
        resume_mode = str(getattr(self.config.trainer, "resume_mode", "strict"))
        if resume_mode not in {"strict", "weights_only"}:
            raise ValueError("trainer.resume_mode must be 'strict' or 'weights_only'")
        self.resume_from_checkpoint = None

        if is_resume:
            if resume_mode == "strict":
                resume_from_checkpoint, self.completed_steps = self._get_latest_training_state(self.checkpoint_dir)
                if not resume_from_checkpoint:
                    raise RuntimeError(
                        "Strict resume requested, but no complete training-state checkpoint was found. "
                        "Legacy *_pytorch_model.pt files contain weights only and cannot restore AdamW, "
                        "scheduler, RNG, or dataloader position. Use trainer.resume_mode=weights_only "
                        "only when non-equivalent continuation is acceptable."
                    )
                self._resume_state_dir = resume_from_checkpoint
                self._resume_metadata = self._read_resume_metadata(resume_from_checkpoint)
                logger.info(
                    "Strict resume selected: state=%s, steps=%d",
                    self._resume_state_dir,
                    self.completed_steps,
                )
                return

            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.warning(
                    "Weights-only resume selected: optimizer/RNG/dataloader state is intentionally not restored. "
                    "checkpoint=%s steps=%d",
                    self.resume_from_checkpoint,
                    self.completed_steps,
                )
                return
            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0
            if bool(latent_cfg.get("require_stage1_checkpoint", False)) and not pretrained_checkpoint:
                raise ValueError("No Stage 2 resume checkpoint found and no Stage 1 checkpoint was provided")

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            if bool(latent_cfg.get("require_stage1_checkpoint", False)):
                selected = {name.strip() for name in str(reload_modules or "").split(",") if name.strip()}
                expected_modules = set(_thinking_stage1_module_names(self.model))
                if selected != expected_modules:
                    raise ValueError(f"Stage 2 must reload only {sorted(expected_modules)}")
                self.model = _load_thinking_stage1_modules(self.model, pretrained_checkpoint)
            else:
                self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            latent_cfg = self.config.framework.get("latent_reasoning", {})
            if bool(latent_cfg.get("initialize_tokens_from_base_checkpoint", False)):
                if int(latent_cfg.get("stage", 0)) != 1:
                    raise ValueError("initialize_tokens_from_base_checkpoint is only valid for Stage 1")
                self.model.qwen_vl_interface._initialize_thinking_token_rows()
                logger.info("Initialized thinking-token rows after loading the pre-thinking checkpoint")
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _get_latest_training_state(self, checkpoint_dir):
        """Return the latest atomically-published full training state."""
        candidates = []
        for entry in Path(checkpoint_dir).iterdir():
            match = re.fullmatch(r"steps_(\d+)_training_state", entry.name)
            if not match or not entry.is_dir():
                continue
            if not (entry / "trainer_state.json").is_file():
                continue
            candidates.append((int(match.group(1)), entry))
        if not candidates:
            return None, 0
        completed_steps, state_dir = max(candidates, key=lambda item: item[0])
        return str(state_dir), completed_steps

    @staticmethod
    def _read_resume_metadata(state_dir):
        metadata_path = Path(state_dir) / "trainer_state.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        required = {"completed_steps", "data_batches_consumed", "vla_epoch_count", "batches_in_current_epoch"}
        missing = sorted(required.difference(metadata))
        if missing:
            raise RuntimeError(f"Incomplete strict-resume metadata in {metadata_path}: missing {missing}")
        return metadata

    def _restore_training_state(self):
        """Restore prepared model/optimizer/scheduler/RNG after data positioning."""
        if not self._resume_state_dir:
            return
        saved_world_size = int(self._resume_metadata.get("world_size", -1))
        saved_accumulation = int(self._resume_metadata.get("gradient_accumulation_steps", -1))
        if saved_world_size != self.accelerator.num_processes:
            raise RuntimeError(
                f"Strict resume requires the same world size: checkpoint={saved_world_size}, "
                f"current={self.accelerator.num_processes}"
            )
        if saved_accumulation != self.accelerator.gradient_accumulation_steps:
            raise RuntimeError(
                "Strict resume requires the same gradient accumulation steps: "
                f"checkpoint={saved_accumulation}, current={self.accelerator.gradient_accumulation_steps}"
            )
        self.accelerator.load_state(self._resume_state_dir)
        self.accelerator.print(f"Strictly resumed from training state: {self._resume_state_dir}")

    def _adjust_lr_scheduler_for_weights_only_resume(self):
        """Retain historical LR behaviour for explicitly non-strict continuation."""
        logger.warning("Advancing scheduler for weights-only resume from step %d", self.completed_steps)
        for _ in range(self.completed_steps):
            self.lr_scheduler.step()

    def _save_checkpoint(self):
        """Save an inference model plus an atomically-published full training state."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")

            state_dict = _get_cpu_state_dict(self.accelerator, self.model)
            if save_format == "safetensors":
                checkpoint_file = checkpoint_path + "_model.safetensors"
            elif save_format == "pt":
                checkpoint_file = checkpoint_path + "_pytorch_model.pt"
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            _atomic_save_state_dict(state_dict, checkpoint_file, save_format)
        self.accelerator.wait_for_everyone()

        state_dir = Path(f"{checkpoint_path}_training_state")
        temporary_state_dir = Path(f"{state_dir}.incomplete")
        if self.accelerator.is_main_process:
            if state_dir.exists():
                raise FileExistsError(f"Refusing to overwrite existing full training state: {state_dir}")
            if temporary_state_dir.exists():
                shutil.rmtree(temporary_state_dir)
        self.accelerator.wait_for_everyone()

        self.accelerator.save_state(str(temporary_state_dir))
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            metadata = {
                "completed_steps": self.completed_steps,
                "data_batches_consumed": self.data_batches_consumed,
                "vla_epoch_count": self.vla_epoch_count,
                "batches_in_current_epoch": self._batches_in_current_epoch,
                "world_size": self.accelerator.num_processes,
                "gradient_accumulation_steps": self.accelerator.gradient_accumulation_steps,
            }
            with (temporary_state_dir / "trainer_state.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, sort_keys=True)
            os.replace(temporary_state_dir, state_dir)

            summary_data = {"steps": self.completed_steps, "full_training_state": str(state_dir)}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}; strict state={state_dir}")
            removed = _prune_completed_training_states(
                self.checkpoint_dir,
                self.config.trainer.get("keep_last_training_states", None),
                self.completed_steps,
            )
            if removed:
                logger.info("Removed older optimizer states (model weights retained): %s", removed)

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and rank == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            if getattr(self, "_wandb_enabled", False):
                try:
                    wandb.log(metrics, step=self.completed_steps)
                except Exception as exc:
                    self._wandb_enabled = False
                    logger.warning(f"W&B log failed; disabling W&B: {exc}")
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators, restoring the next unread batch on strict resume."""
        if self._resume_metadata is not None:
            self.data_batches_consumed = int(self._resume_metadata["data_batches_consumed"])
            self.vla_epoch_count = int(self._resume_metadata["vla_epoch_count"])
            self._batches_in_current_epoch = int(self._resume_metadata["batches_in_current_epoch"])
            self._set_dataloader_epoch(self.vla_epoch_count)
        self.vla_iter = iter(self.vla_train_dataloader)

        # The dataset used here deterministically derives a sample from
        # (epoch, index).  Advance only the iterator position before loading
        # RNG state so preprocessing cannot perturb the restored training RNG.
        for _ in range(self._batches_in_current_epoch):
            try:
                next(self.vla_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "Strict-resume data position exceeds the restored dataloader epoch; "
                    "the dataset or world-size changed since checkpointing."
                ) from exc

        self._restore_training_state()

    def _set_dataloader_epoch(self, epoch: int) -> None:
        sampler = getattr(self.vla_train_dataloader, "sampler", None)
        if sampler is not None and callable(getattr(sampler, "set_epoch", None)):
            sampler.set_epoch(epoch)
        dataset = getattr(self.vla_train_dataloader, "dataset", None)
        if dataset is not None and callable(getattr(dataset, "set_epoch", None)):
            dataset.set_epoch(epoch)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_epoch_count += 1
            self._batches_in_current_epoch = 0
            self._set_dataloader_epoch(self.vla_epoch_count)
            self.vla_iter = iter(self.vla_train_dataloader)
            batch_vla = next(self.vla_iter)
        self.data_batches_consumed += 1
        self._batches_in_current_epoch += 1
        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if not self.accelerator.sync_gradients:
                continue

            progress_bar.update(1)
            self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        was_training = self.model.training
        self.model.eval()
        try:
            output_dict = _unwrap_model(self.accelerator, self.model).predict_action(
                examples=examples, use_ddim=True, num_ddim_steps=20
            )
        finally:
            self.model.train(was_training)

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = output_dict.get("total_loss", action_loss)

            self.accelerator.backward(total_loss)

            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
                self._future_ema_updates += 1
                if self._future_ema_updates % 2 == 0:
                    unwrapped_model = _unwrap_model(self.accelerator, self.model)
                    qwen_interface = getattr(unwrapped_model, "qwen_vl_interface", None)
                    if qwen_interface is not None and hasattr(qwen_interface, "update_future_ema"):
                        qwen_interface.update_future_ema()
            self.optimizer.zero_grad()

        metrics = {"total_loss": total_loss.item()}
        if (output_dict.get("decoder_loss") is None
                or int(self.config.framework.get("latent_reasoning", {}).get("stage", 0)) != 1):
            metrics["action_dit_loss"] = action_loss.item()
        if output_dict.get("future_loss") is not None:
            metrics["future_loss"] = output_dict["future_loss"].item()
        if output_dict.get("spatial_forcing_loss") is not None:
            metrics["spatial_forcing_loss"] = output_dict["spatial_forcing_loss"].item()
        if output_dict.get("decoder_loss") is not None:
            metrics["decoder_loss"] = output_dict["decoder_loss"].item()
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = _get_cpu_state_dict(self.accelerator, self.model)
            if save_format == "safetensors":
                checkpoint_file = os.path.join(final_checkpoint, "model.safetensors")
            elif save_format == "pt":
                checkpoint_file = os.path.join(final_checkpoint, "pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            _atomic_save_state_dict(state_dict, checkpoint_file, save_format)
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    _validate_thinking_accumulation(cfg, accelerator)

    from starVLA.model.modules.vlm.thinking_config import validate_stage2_decoder_config

    validate_stage2_decoder_config(cfg)

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_tighten.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
