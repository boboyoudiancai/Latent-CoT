# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
Latent Reasoning Training Script for LaRA-VLA

Training with implicit latent reasoning (thinking tokens + KV-Cache iterative forward).
Supports VLM loss computation and img_next alignment loss.
"""

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
import time

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

# Local Modules
from laravla.training.trainer_utils.trainer_tools import normalize_dotlist_args
from laravla.model.framework import build_framework
from laravla.training.trainer_utils.trainer_tools import TrainerUtils
from laravla.training.trainer_utils.trainer_tools import build_param_lr_groups
from laravla.training.trainer_utils.cot_mode_utils import get_implicit_flags


accelerator = None


def _gradient_accumulation_steps(cfg) -> int:
    trainer_cfg = getattr(cfg, "trainer", None)
    steps = getattr(trainer_cfg, "gradient_accumulation_steps", 1) if trainer_cfg is not None else 1
    return max(1, int(steps or 1))


def build_accelerator(cfg) -> Accelerator:
    global accelerator
    grad_acc_steps = _gradient_accumulation_steps(cfg)
    enable_mixed_precision = bool(
        getattr(getattr(cfg, "trainer", None), "enable_mixed_precision_training", False)
    )
    deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=grad_acc_steps)
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=grad_acc_steps,
        sync_each_batch=True,
    )
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        mixed_precision="bf16" if enable_mixed_precision else "no",
    )
    accelerator.print(accelerator.state)
    return accelerator

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # save config
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


def attach_auxiliary_latent_decoder(model, cfg, device: torch.device) -> None:
    """Load Stage-1's FullQwen latent decoder for a frozen auxiliary loss."""
    decoder_cfg = cfg.framework.get("latent_decoder", {})
    if not bool(decoder_cfg.get("enable", False)):
        return

    checkpoint_path = str(decoder_cfg.get("checkpoint", "") or "").strip()
    if not checkpoint_path:
        raise ValueError("framework.latent_decoder.checkpoint is required when enabled")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"latent decoder checkpoint not found: {checkpoint_path}")

    from decoder.latent_text_decoder import FullQwenLatentTextDecoder
    from laravla.model.modules.vlm import get_vlm_model

    decoder_cfg_full = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    decoder_base_vlm = str(
        decoder_cfg.get("base_vlm", decoder_cfg_full.framework.qwenvl.base_vlm)
    )
    decoder_cfg_full.framework.qwenvl.base_vlm = decoder_base_vlm
    decoder_cfg_full.framework.img_next.use_teacher = False
    decoder_cfg_full.framework.spatial.use_teacher = False

    decoder_iface = get_vlm_model(config=decoder_cfg_full)
    decoder_qwen = decoder_iface.model.to(device=device, dtype=torch.bfloat16)
    decoder_qwen.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(decoder_qwen, "enable_input_require_grads"):
        decoder_qwen.enable_input_require_grads()
    decoder_qwen.config.use_cache = False
    decoder = FullQwenLatentTextDecoder(
        qwen_model=decoder_qwen,
        hidden_size=int(decoder_qwen.config.hidden_size),
        vocab_size=int(decoder_qwen.lm_head.out_features),
    ).to(device=device, dtype=torch.bfloat16)

    payload = torch.load(checkpoint_path, map_location="cpu")
    decoder_state = payload.get("decoder_state_dict", payload)
    decoder.load_state_dict(decoder_state, strict=True)
    decoder.eval()

    bridge_cfg = cfg.datasets.vla_data.bridge_reasoning
    component_order = str(getattr(bridge_cfg, "component_order", "SUBTASK,BBOX,REASON"))
    component_order = [item.strip().upper() for item in component_order.split(",") if item.strip()]
    model.attach_latent_decoder(
        decoder=decoder,
        tokenizer=decoder_iface.processor.tokenizer,
        component_order=component_order,
        include_bbox=bool(getattr(bridge_cfg, "include_bbox", True)),
        max_target_tokens=int(decoder_cfg.get("max_target_tokens", 128)),
        loss_weight=float(decoder_cfg.get("loss_weight", 0.02)),
        source_checkpoint=checkpoint_path,
    )
    if accelerator is not None and accelerator.is_main_process:
        logger.info("[latent_decoder] loaded decoder checkpoint=%s", checkpoint_path)


# here changes need to 📦 encapsulate Dataloader
from laravla.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    # Safely extract data_mix for logging (could be in ecot.* or vla_data.*)
    cot_mode = getattr(getattr(cfg, "framework", {}), "cot_mode", "implicit")
    mode_flags = getattr(getattr(cfg, "framework", {}), "cot_mode_flags", {}) or {}
    generate_thinking = mode_flags.get("generate_thinking", cot_mode in ("explicit", "implicit"))
    reasoning_stage = getattr(cfg.datasets.vla_data.bridge_reasoning, "stage", "unknown")

    dataset_py = cfg.datasets.vla_data.dataset_py
    data_mix = getattr(cfg.datasets.vla_data, "data_mix", "unknown")
    logger.info(
        "Creating VLA Dataset with dataset_py=`%s`, data_mix=`%s`, cot_mode=`%s`, stage=`%s`, generate_thinking=%s",
        dataset_py,
        data_mix,
        cot_mode,
        reasoning_stage,
        generate_thinking,
    )
    if generate_thinking and isinstance(reasoning_stage, int) and reasoning_stage < 1:
        logger.warning(
            "cot_mode=%s 需要 reasoning 数据，但当前 stage=%s 可能缺少 reasoning 标注，请检查配置/数据混合。",
            cot_mode,
            reasoning_stage,
        )
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Determine warmup steps: use warmup_ratio if provided, otherwise use num_warmup_steps
    if hasattr(cfg.trainer, 'warmup_ratio') and cfg.trainer.warmup_ratio > 0:
        num_warmup_steps = int(cfg.trainer.max_train_steps * cfg.trainer.warmup_ratio)
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.info(f"Using warmup_ratio={cfg.trainer.warmup_ratio}: num_warmup_steps={num_warmup_steps} (from {cfg.trainer.max_train_steps} total steps)")
    else:
        num_warmup_steps = cfg.trainer.num_warmup_steps
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.info(f"Using num_warmup_steps={num_warmup_steps}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


def validate_ecot_config(cfg):
    """Validate latent reasoning configuration consistency."""
    latent_cfg = cfg.framework.get("latent_reasoning", {})
    if not latent_cfg:
        logger.warning("latent_reasoning config is missing, using defaults")
        return

    vlm_loss_weight = latent_cfg.get("vlm_loss_weight", 0.1)
    compute_language_loss = latent_cfg.get("compute_language_loss", False)
    logger.info(f"Latent reasoning: compute_language_loss={compute_language_loss}, "
                f"vlm_loss_weight={vlm_loss_weight}")

    reasoning_stage = getattr(cfg.datasets.vla_data.bridge_reasoning, "stage", 4)
    logger.info(f"Bridge reasoning stage: {reasoning_stage}")

    img_next_cfg = cfg.framework.get("img_next", {})
    if img_next_cfg and img_next_cfg.get("enable", False):
        logger.info(f"img_next: res={img_next_cfg.get('res')}, "
                    f"loss_weight={img_next_cfg.get('loss_weight')}, "
                    f"use_teacher={img_next_cfg.get('use_teacher', True)}")

    logger.info("Config validation passed")


def sync_bridge_reasoning_to_framework(cfg):
    """
    Sync thinking token definitions from bridge_reasoning (dataloader side)
    to framework.latent_reasoning (model side), ensuring both use the same tokens.
    """
    try:
        bridge_cfg = cfg.datasets.vla_data.bridge_reasoning
    except AttributeError:
        return

    if not getattr(bridge_cfg, "enable", False):
        return

    cfg.framework.enable_latent_reasoning = True

    latent_cfg = getattr(cfg.framework, "latent_reasoning", None)
    if latent_cfg is None:
        cfg.framework.latent_reasoning = {}
        latent_cfg = cfg.framework.latent_reasoning

    def _set(key, value):
        if isinstance(latent_cfg, dict):
            latent_cfg[key] = value
        else:
            setattr(latent_cfg, key, value)

    # Sync token definitions: bridge_reasoning is the single source of truth
    _set("thinking_token", getattr(bridge_cfg, "thinking_token", "<|thinking|>"))
    _set("start_of_thinking_token", getattr(bridge_cfg, "start_token", "<|start_of_thinking|>"))
    _set("end_of_thinking_token", getattr(bridge_cfg, "end_token", "<|end_of_thinking|>"))

    tag2think = getattr(bridge_cfg, "tag2think_count", None)
    if tag2think is not None:
        _set("tag2think_count", tag2think)


class LaRA_VLA_Trainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = self._get_start_step()
        self.total_batch_size = self._calculate_total_batch_size()
        self.min_save_step = getattr(self.config.trainer, "min_save_step", 0)
        self._img_next_ema_updates = 0

    def _get_start_step(self):
        """Optional absolute step offset for weight-only resume."""
        trainer_cfg = self.config.trainer
        if hasattr(trainer_cfg, "get"):
            start_step = trainer_cfg.get("start_step", 0)
        else:
            start_step = getattr(trainer_cfg, "start_step", 0)
        return int(start_step or 0)

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        if bool(getattr(self.config.trainer, "enable_gradient_checkpointing", False)):
            qwen_interface = getattr(self.model, "qwen_vl_interface", None)
            qwen_model = getattr(qwen_interface, "model", None)
            if qwen_model is None or not hasattr(qwen_model, "gradient_checkpointing_enable"):
                raise RuntimeError(
                    "trainer.enable_gradient_checkpointing=true, but the Qwen backbone "
                    "does not support gradient checkpointing"
                )
            qwen_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(qwen_model, "enable_input_require_grads"):
                qwen_model.enable_input_require_grads()
            qwen_model.config.use_cache = False
            logger.info("Enabled Qwen gradient checkpointing (use_reentrant=False, use_cache=False)")

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            # self.vlm_train_dataloader
        )

        self._fast_forward_scheduler()
        self._init_wandb()
        self._init_checkpointing()

    def _fast_forward_scheduler(self):
        """Advance LR scheduler when resuming from model weights only."""
        if self.completed_steps <= 0:
            return
        self.lr_scheduler.last_epoch = self.completed_steps
        resumed_lrs = None
        if hasattr(self.lr_scheduler, "lr_lambdas") and hasattr(self.lr_scheduler, "base_lrs"):
            resumed_lrs = [
                lr_lambda(self.completed_steps) * base_lr
                for lr_lambda, base_lr in zip(self.lr_scheduler.lr_lambdas, self.lr_scheduler.base_lrs)
            ]
        elif hasattr(self.lr_scheduler, "_get_closed_form_lr"):
            resumed_lrs = self.lr_scheduler._get_closed_form_lr()

        if resumed_lrs is not None:
            self.lr_scheduler._last_lr = resumed_lrs
            optimizers = [self.optimizer, getattr(self.lr_scheduler, "optimizer", None)]
            seen_optimizers = set()
            for optimizer in optimizers:
                if optimizer is None or id(optimizer) in seen_optimizers:
                    continue
                param_groups = getattr(optimizer, "param_groups", None)
                if param_groups is None:
                    continue
                seen_optimizers.add(id(optimizer))
                for param_group, lr in zip(param_groups, resumed_lrs):
                    param_group["lr"] = lr
        self.accelerator.print(f"Weight-only resume: start_step={self.completed_steps}")

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="ecot-qwengrt00t-train",
            )

    def _init_checkpointing(self):
        """initialize checkpoint directory"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        # resume training state
        print(f"pretrained_checkpoint: {pretrained_checkpoint}")
        print(f"is_resume: {is_resume}")
        if pretrained_checkpoint and is_resume:
            print(f"Resuming from checkpoint: {self.config.resume_from_checkpoint}")
            self._load_checkpoint(self.config.resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """save current training state"""

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # add learning rate
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]

                # add epoch info
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # update progress
            did_optimizer_step = bool(self.accelerator.sync_gradients)
            if did_optimizer_step:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

            eval_interval = int(getattr(self.config.trainer, "eval_interval", 0) or 0)
            if (
                did_optimizer_step
                and eval_interval > 0
                and self.completed_steps > 0
                and self.completed_steps % eval_interval == 0
            ):
                self.accelerator.wait_for_everyone()
                step_metrics = self.eval_action_model(
                    step_metrics,
                    eval_examples=batch_vla,
                )
                self.accelerator.wait_for_everyone()

            save_interval = int(getattr(self.config.trainer, "save_interval", 0) or 0)
            save_due = (
                did_optimizer_step
                and save_interval > 0
                and self.completed_steps >= self.min_save_step
                and self.completed_steps > 0
                and self.completed_steps % save_interval == 0
            )
            if save_due:
                decoder_metrics = self.eval_latent_decoder(batch_vla)
                step_metrics.update(decoder_metrics)

            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            if did_optimizer_step:
                self._log_metrics(step_metrics)

            # save checkpoint
            if save_due:
                self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    @torch.no_grad()
    def eval_action_model(self, step_metrics: dict = None, eval_examples=None) -> dict:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        if eval_examples is None:
            raise ValueError("eval_examples must be provided at an optimizer-step boundary")

        batch_images = [example["image"] for example in eval_examples]
        instructions = [example["lang"] for example in eval_examples]
        actions = [example["action"] for example in eval_examples]
        states = (
            [example["state"] for example in eval_examples]
            if "state" in eval_examples[0]
            else None
        )

        # Keep the distributed training module in train mode. Switching the
        # full Qwen/DeepSpeed tree to eval and back mutates multimodal runtime
        # state used by the next iterative latent forward. no_grad already
        # prevents this diagnostic action rollout from building a graph.
        output_dict = self.model.predict_action(
            batch_images=batch_images,
            instructions=instructions,
            state=states,
            use_ddim=True,
            num_ddim_steps=20,
        )

        normalized_actions = output_dict["normalized_actions"]
        actions = np.array(actions)
        local_score = float(TrainerUtils.euclidean_distance(normalized_actions, actions))
        local_count = float(np.prod(actions.shape))
        totals = torch.tensor(
            [local_score, local_count],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        step_metrics["mse_score"] = (totals[0] / totals[1].clamp_min(1.0)).item()
        return step_metrics

    @torch.no_grad()
    def eval_latent_decoder(self, examples) -> dict:
        """Run a synchronized decoder eval before each checkpoint save."""
        decoder_cfg = self.config.framework.get("latent_decoder", {})
        if not bool(decoder_cfg.get("enable", False)):
            return {}
        if not bool(decoder_cfg.get("eval_on_save", True)):
            return {}

        model = self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "evaluate_latent_decoder"):
            raise RuntimeError("latent decoder is enabled but evaluation is unavailable")

        result = model.evaluate_latent_decoder(
            examples,
            max_new_tokens=int(decoder_cfg.get("eval_decode_tokens", 64) or 64),
        )
        local_count = int(result["num_samples"]) if result is not None else 0
        local_loss_sum = (
            float(result["loss"].item()) * local_count
            if result is not None and local_count > 0
            else 0.0
        )
        totals = torch.tensor(
            [local_loss_sum, float(local_count)],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if totals[1].item() <= 0:
            self.accelerator.wait_for_everyone()
            return {}
        eval_loss = (totals[0] / totals[1].clamp_min(1.0)).item()

        if self.accelerator.is_main_process:
            sample_limit = max(0, int(decoder_cfg.get("eval_samples", 4) or 0))
            samples = []
            if result is not None:
                for tag, decoded, target in zip(
                    result["tags"],
                    result["decoded_texts"],
                    result["target_texts"],
                ):
                    if len(samples) >= sample_limit:
                        break
                    samples.append(
                        {
                            "tag": tag,
                            "decoded": " ".join((decoded or "").replace("\n", " ").split()),
                            "target": target,
                        }
                    )
            record = {
                "step": self.completed_steps,
                "eval_decoder_loss": eval_loss,
                "num_eval_samples_global": int(totals[1].item()),
                "samples": samples,
            }
            output_path = os.path.join(
                self.config.output_dir,
                "decoder_eval_samples.jsonl",
            )
            with open(output_path, "a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(
                "Step %s, decoder eval loss: %.6f",
                self.completed_steps,
                eval_loss,
            )

        self.accelerator.wait_for_everyone()
        return {"eval_decoder_loss": eval_loss}

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

            latent_cfg = self.config.framework.get("latent_reasoning", {})
            reasoning_stage = getattr(self.config.datasets.vla_data.bridge_reasoning, "stage", 4)
            logger.info("***** Latent Reasoning Configuration *****")
            logger.info(f"  Reasoning Stage: {reasoning_stage}")
            logger.info(f"  Compute Language Loss: {latent_cfg.get('compute_language_loss', False)}")
            logger.info(f"  VLM Loss Weight: {latent_cfg.get('vlm_loss_weight', 0.1)}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla, optimizer_step=self.completed_steps)

                # Use model-computed total_loss if available (includes vlm_loss if enabled)
                # Otherwise fallback to action_loss only (backward compatibility)
                if "total_loss" in output_dict:
                    total_loss = output_dict["total_loss"]
                else:
                    total_loss = output_dict["action_loss"]

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            # gradient clipping
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # optimizer step
            self.optimizer.step()

            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

                # EMA update for img_next target vision encoder (if enabled)
                self._img_next_ema_updates += 1
                if self._img_next_ema_updates % 2 == 0:
                    try:
                        qwen_iface = getattr(self.model, "qwen_vl_interface", None)
                        if (
                            qwen_iface is not None
                            and getattr(qwen_iface, "use_img_next_teacher", True)
                            and hasattr(qwen_iface, "update_img_next_ema")
                        ):
                            qwen_iface.update_img_next_ema()
                    except Exception as e:
                        logger.warning(f"[img_next_ema] update skipped due to error: {e}")

            self.optimizer.zero_grad()

        # Build metrics dictionary
        metrics = {}
        
        # Add action_loss if available (not present in reasoning_only mode)
        if "action_loss" in output_dict and output_dict["action_loss"] is not None:
            metrics["action_loss"] = output_dict["action_loss"].item()
        
        # Add vlm_loss if available and computed
        if "vlm_loss" in output_dict and output_dict["vlm_loss"] is not None:
            metrics["vlm_loss"] = output_dict["vlm_loss"].item()

        if "img_next_loss" in output_dict and output_dict["img_next_loss"] is not None:
            metrics["img_next_loss"] = output_dict["img_next_loss"].item()

        if "spatial_loss" in output_dict and output_dict["spatial_loss"] is not None:
            metrics["spatial_loss"] = output_dict["spatial_loss"].item()
        if "decoder_loss" in output_dict and output_dict["decoder_loss"] is not None:
            metrics["decoder_loss"] = output_dict["decoder_loss"].item()
        
        # Add total_loss for monitoring (ensures consistency with backward pass)
        metrics["total_loss"] = total_loss.item()
        
        return metrics

    def _finalize_training(self):
        """training end processing"""
        self.accelerator.wait_for_everyone()
        # save final model
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("ECoT VLA Training :: Warming Up")
    accelerator = build_accelerator(cfg)

    mode_flags = get_implicit_flags()

    # Inject derived flags into cfg
    cfg.framework.enable_latent_reasoning = True
    cfg.framework.emit_thinking_tokens = False
    cfg.framework.cot_mode = "implicit"
    cfg.framework.cot_mode_flags = mode_flags

    training_stage = getattr(cfg.framework, "training_stage", "full")
    if training_stage == "full":
        cfg.datasets.vla_data.bridge_reasoning.stage = mode_flags["reasoning_stage"]

    logger.info(f"[Implicit Reasoning] training_stage={training_stage}, flags={mode_flags}")

    sync_bridge_reasoning_to_framework(cfg)

    raw_include_state = cfg.datasets.vla_data.get("include_state", False)
    include_state = (
        raw_include_state.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(raw_include_state, str)
        else bool(raw_include_state)
    )
    if include_state:
        # Keep the saved config and constructed action head aligned with native LIBERO state.
        cfg.framework.action_model.state_dim = 8
        if cfg.framework.action_model.get("state_dropout", None) is None:
            cfg.framework.action_model.state_dropout = 0.5

    # Validate ECoT configuration
    validate_ecot_config(cfg)

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    accelerator.wait_for_everyone()
    # build model
    vla = build_framework(cfg)
    attach_auxiliary_latent_decoder(vla, cfg, accelerator.device)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run ECoT VLA Training
    trainer = LaRA_VLA_Trainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECoT Implicit Reasoning Training Script")
    parser.add_argument("--config_yaml", type=str, default="laravla/config/training/bridge.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
