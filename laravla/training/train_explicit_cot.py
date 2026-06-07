# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0.

"""Explicit-CoT LIBERO training entrypoint.

This reuses the LaRA-VLA trainer/model/dataloader stack, but changes the mode
injection that `train.py` hard-codes for implicit latent reasoning.
"""

import argparse

import torch.distributed as dist
from omegaconf import OmegaConf

from laravla.model.framework import build_framework
from laravla.training.train import (
    LaRA_VLA_Trainer,
    accelerator,
    logger,
    prepare_data,
    setup_directories,
    setup_optimizer_and_scheduler,
    validate_ecot_config,
)
from laravla.training.trainer_utils.trainer_tools import normalize_dotlist_args


EXPLICIT_COT_FLAGS = {
    "enable_latent_reasoning": False,
    "emit_thinking_tokens": True,
    "use_iterative_forward": False,
    "generate_thinking": True,
    "reasoning_stage": 1,
}


def _ensure_explicit_cot_config(cfg):
    cfg.framework.enable_latent_reasoning = False
    cfg.framework.emit_thinking_tokens = True
    cfg.framework.cot_mode = "explicit"
    cfg.framework.cot_mode_flags = dict(EXPLICIT_COT_FLAGS)

    cfg.datasets.vla_data.bridge_reasoning.enable = True
    cfg.datasets.vla_data.bridge_reasoning.stage = 1
    cfg.datasets.vla_data.bridge_reasoning.as_solution = True
    cfg.datasets.vla_data.bridge_reasoning.include_img_next = False

    cfg.framework.img_next.enable = False
    cfg.framework.img_next.loss_weight = 0
    cfg.framework.img_next.use_teacher = False

    cfg.framework.latent_reasoning.compute_language_loss = True
    if not hasattr(cfg.framework.latent_reasoning, "vlm_loss_weight"):
        cfg.framework.latent_reasoning.vlm_loss_weight = 1.0

    return cfg


def main(cfg) -> None:
    cfg = _ensure_explicit_cot_config(cfg)
    training_stage = getattr(cfg.framework, "training_stage", "full")
    logger.info("[Explicit CoT] training_stage=%s, flags=%s", training_stage, EXPLICIT_COT_FLAGS)

    validate_ecot_config(cfg)

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = LaRA_VLA_Trainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()
    trainer.train()

    logger.info("Explicit CoT training finished")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explicit CoT Training Script")
    parser.add_argument("--config_yaml", type=str, default="laravla/config/training/libero.yaml")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
