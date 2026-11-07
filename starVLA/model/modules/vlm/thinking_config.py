"""Configuration guards for optional QwenPI latent-language experiments."""

import os


def latent_forward_mode(latent_config):
    """Resolve the opt-in query ablation without changing legacy checkpoints."""
    mode = str(latent_config.get("forward_mode", "recursive"))
    if mode not in ("recursive", "independent_queries"):
        raise ValueError("latent_reasoning.forward_mode must be recursive or independent_queries")
    if mode == "independent_queries" and not bool(latent_config.get("enable", False)):
        raise ValueError("independent_queries requires enabled latent reasoning")
    return mode


def validate_stage2_decoder_config(config):
    """Keep frozen-decoder single-stage training separate from Stage1 transfer.

    Existing Stage1 and action-only Stage2 configurations are unchanged. A later
    strict resume of this experiment is supported; its initial run must use only
    the original pretrained VLM and an independently loaded, frozen decoder.
    """
    latent = config.framework.get("latent_reasoning", {})
    latent_forward_mode(latent)
    if not bool(latent.get("stage2_decoder_supervision", False)):
        return
    if not bool(latent.get("enable", False)) or int(latent.get("stage", 0)) != 2:
        raise ValueError("stage2_decoder_supervision requires enabled latent reasoning with stage=2")
    if not bool(latent.get("freeze_decoder", False)):
        raise ValueError("Single-stage Stage2 decoder supervision requires freeze_decoder=true")
    if str(latent.get("decoder_input_projector", "auto")) not in ("auto", "residual_mlp"):
        raise ValueError("decoder_input_projector must be auto or residual_mlp")
    if bool(latent.get("require_stage1_checkpoint", False)):
        raise ValueError("Single-stage frozen-decoder training must not require a Stage1 checkpoint")
    if not bool(latent.get("require_base_initialization", False)):
        raise ValueError("Single-stage frozen-decoder training requires require_base_initialization=true")
    student_source = str(config.framework.qwenvl.base_vlm)
    decoder_source = str(latent.get("decoder_base_vlm", student_source))
    if os.path.realpath(student_source) != os.path.realpath(decoder_source):
        raise ValueError("Frozen decoder must use the same original base VLM as the QwenPI student")
    trainer = config.get("trainer", {})
    if trainer.get("freeze_modules"):
        raise ValueError("Single-stage training must leave the student VLM, action head and projections trainable")
    if trainer.get("reload_modules"):
        raise ValueError("Single-stage training must not specify Stage1 reload_modules")
    if bool(trainer.get("is_resume", False)):
        if str(trainer.get("resume_mode", "strict")) != "strict":
            raise ValueError("Frozen-decoder experiment requires strict resume to preserve training state")
        if trainer.get("pretrained_checkpoint"):
            raise ValueError("Strict resume restores this run's training state, not a separate pretrained_checkpoint")
    elif trainer.get("pretrained_checkpoint") or int(trainer.get("start_step", 0) or 0):
        raise ValueError("Single-stage frozen-decoder training must start from base weights at step 0")
