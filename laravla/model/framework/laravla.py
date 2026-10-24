# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image



from laravla.training.trainer_utils import initialize_overwatch
from laravla.model.modules.vlm.spatial_cross_attention import dense_spatial_cosine_loss, validate_spatial_only_config
from laravla.model.modules.vlm.spatial_forcing_aux import (
    SpatialForcingAlignProjector,
    select_base_image_hidden,
    spatial_forcing_cosine_loss,
)
from laravla.model.modules.vlm.vggt_teacher import VGGTTeacher
from laravla.model.modules.vlm.img_next_residual import masked_img_next_loss

logger = initialize_overwatch(__name__)


def _is_vjepa21_teacher(teacher_type: str) -> bool:
    return str(teacher_type).strip().lower().startswith("vjepa2_1_vit_")


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from laravla.model.framework.base_framework import baseframework
from laravla.model.framework.latent_analysis_mixin import LatentAnalysisMixin
from laravla.model.modules.vlm import get_vlm_model
from laravla.model.modules.vlm.QWen3 import format_reasoning_prompt
from laravla.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from laravla.model.modules.action_model.fast_ActionHeader import Fast_Action_Tokenizer
from laravla.training.trainer_utils.trainer_tools import resize_images
from laravla.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(LatentAnalysisMixin, baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        validate_spatial_only_config(config)
        raw_include_state = config.datasets.vla_data.get("include_state", False)
        include_state = (
            raw_include_state.strip().lower() in {"1", "true", "yes", "on"}
            if isinstance(raw_include_state, str)
            else bool(raw_include_state)
        )
        if include_state:
            # Native LIBERO state is xyz + axis-angle + two gripper qpos values.
            self.config.framework.action_model.state_dim = 8
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用
        self.training_stage = config.framework.get("training_stage", "full")
        self._init_action_vggt_condition(config)
        self._init_spatial_forcing_aux(config)
        self._fast_action_processor = None
        self._fast_action_token_range = None

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        # Training stage control: "reasoning_only", "action_only", or "full"
        # Apply parameter freezing based on training stage
        if self.training_stage == "spatial_only":
            self.requires_grad_(True)
            self.qwen_vl_interface.spatial_external_teacher.encoder.requires_grad_(False)
            self.qwen_vl_interface.spatial_external_teacher.encoder.eval()
            print("[Training Stage] spatial_only - Only VGGT frozen; no CE or action loss")
        elif self.training_stage == "reasoning_only":
            print(f"[Training Stage] reasoning_only mode - Freezing action_model parameters")
            for param in self.action_model.parameters():
                param.requires_grad = False
        elif self.training_stage == "action_only":
            print(f"[Training Stage] action_only mode - Freezing VLM parameters")
            for param in self.qwen_vl_interface.parameters():
                param.requires_grad = False
        else:
            print(f"[Training Stage] full mode - All parameters trainable")

    def _init_action_vggt_condition(self, config) -> None:
        action_vggt_cfg = getattr(config.framework, "action_vggt_condition", {})
        self.action_vggt_enabled = bool(action_vggt_cfg.get("enable", False))
        self.action_vggt_teacher = None
        self.action_vggt_projector = None
        self.action_vggt_token_count = 0
        self._action_vggt_shape_logged = False
        if not self.action_vggt_enabled:
            return
        if self.training_stage not in {"action_only", "full"}:
            raise ValueError("action_vggt_condition requires training_stage='action_only' or 'full'")
        spatial_cfg = getattr(config.framework, "spatial", {})
        bridge_cfg = getattr(config.datasets.vla_data, "bridge_reasoning", {})
        if bool(spatial_cfg.get("enable", False)) or bool(bridge_cfg.get("include_spatial", False)):
            raise ValueError("action_vggt_condition is an upper-bound baseline without spatial latent tokens")
        input_size = int(action_vggt_cfg.get("input_size", 518))
        if input_size <= 0 or input_size % 14:
            raise ValueError("action_vggt_condition.input_size must be a positive multiple of 14")
        expected_patch_tokens = (input_size // 14) ** 2
        token_count = int(action_vggt_cfg.get("token_count", expected_patch_tokens))
        token_side = int(token_count ** 0.5)
        if token_count <= 0 or token_count > expected_patch_tokens or token_side * token_side != token_count:
            raise ValueError(
                "action_vggt_condition.token_count must be a positive square no larger than the "
                f"{expected_patch_tokens} VGGT patch tokens, got {token_count}"
            )
        if bool(action_vggt_cfg.get("use_vggt_pe", False)):
            raise ValueError("action_vggt_condition must use the original VGGT patch tokens without extra PE")
        with torch.random.fork_rng(devices=[]):
            teacher = VGGTTeacher(
                teacher_repo=str(action_vggt_cfg.get("teacher_repo", "")),
                input_size=input_size,
                layer=int(action_vggt_cfg.get("teacher_layer", -1)),
                patch_start_idx=int(action_vggt_cfg.get("patch_start_idx", 5)),
                use_vggt_pe=bool(action_vggt_cfg.get("use_vggt_pe", False)),
                preprocess_mode=str(action_vggt_cfg.get("preprocess_mode", "resize")),
                cache_size=int(action_vggt_cfg.get("target_cache_size", 0) or 0),
                cache_dir=str(action_vggt_cfg.get("target_cache_dir", "") or ""),
                cache_only=bool(action_vggt_cfg.get("target_cache_only", False)),
                view_mode="base_only",
                expected_patch_tokens=expected_patch_tokens,
                allow_token_pooling=True,
            )
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        projector = torch.nn.Linear(int(teacher.embed_dim), hidden_size, bias=False)
        torch.nn.init.xavier_uniform_(projector.weight)
        self.action_vggt_teacher = teacher
        self.action_vggt_projector = projector
        self.action_vggt_token_count = token_count
        if token_count == expected_patch_tokens:
            logger.info(
                "[action_vggt_condition] appending all %d VGGT patch tokens to action conditioning",
                token_count,
            )
        else:
            logger.info(
                "[action_vggt_condition] bilinearly downsampling %d VGGT patch tokens to %d tokens and "
                "appending them to action conditioning",
                expected_patch_tokens,
                token_count,
            )

    def _append_action_vggt_condition(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        images: List,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not getattr(self, "action_vggt_enabled", False):
            return hidden, attention_mask
        teacher = getattr(self, "action_vggt_teacher", None)
        projector = getattr(self, "action_vggt_projector", None)
        if teacher is None or projector is None:
            raise RuntimeError("action VGGT condition is enabled but its teacher/projector is unavailable")
        features = teacher.encode(
            images=images,
            device=hidden.device,
            target_tokens=self.action_vggt_token_count,
        )
        expected = (
            hidden.shape[0],
            1,
            self.action_vggt_token_count,
            int(teacher.embed_dim),
        )
        if tuple(features.shape) != expected:
            raise ValueError(f"action VGGT feature shape mismatch: {tuple(features.shape)} vs {expected}")
        features = features[:, 0].detach()
        with torch.autocast(
            device_type=hidden.device.type,
            dtype=torch.bfloat16,
            enabled=hidden.device.type == "cuda",
        ):
            projected = projector(features)
        projected = projected.to(dtype=hidden.dtype)
        if projected.shape[-1] != hidden.shape[-1]:
            raise ValueError("projected VGGT tokens must match the Qwen hidden dimension")
        if attention_mask is None:
            attention_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        else:
            attention_mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
        vggt_mask = torch.ones(projected.shape[:2], dtype=torch.bool, device=hidden.device)
        combined = torch.cat([hidden, projected], dim=1)
        combined_mask = torch.cat([attention_mask, vggt_mask], dim=1)
        if not getattr(self, "_action_vggt_shape_logged", False):
            logger.info(
                "[action_vggt_condition] qwen_tokens=%d vggt_tokens=%d combined_tokens=%d",
                hidden.shape[1],
                projected.shape[1],
                combined.shape[1],
            )
            self._action_vggt_shape_logged = True
        return combined, combined_mask

    def _init_spatial_forcing_aux(self, config) -> None:
        forcing_cfg = getattr(config.framework, "spatial_forcing_aux", {})
        self.spatial_forcing_enabled = bool(forcing_cfg.get("enable", False))
        self.spatial_forcing_teacher = None
        self.spatial_forcing_projector = None
        self.spatial_forcing_student_layer = 0
        self.spatial_forcing_target_tokens = 0
        self.spatial_forcing_loss_weight = 0.0
        if not self.spatial_forcing_enabled:
            return
        if self.action_vggt_enabled:
            raise ValueError("spatial_forcing_aux and action_vggt_condition are mutually exclusive")
        if self.training_stage != "full":
            raise ValueError("spatial_forcing_aux requires training_stage='full'")
        input_size = int(forcing_cfg.get("input_size", 518))
        with torch.random.fork_rng(devices=[]):
            teacher = VGGTTeacher(
                teacher_repo=str(forcing_cfg.get("teacher_repo", "")),
                input_size=input_size,
                layer=int(forcing_cfg.get("teacher_layer", -1)),
                patch_start_idx=int(forcing_cfg.get("patch_start_idx", 5)),
                use_vggt_pe=bool(forcing_cfg.get("use_vggt_pe", True)),
                preprocess_mode=str(forcing_cfg.get("preprocess_mode", "resize")),
                cache_size=int(forcing_cfg.get("target_cache_size", 0) or 0),
                cache_dir=str(forcing_cfg.get("target_cache_dir", "") or ""),
                cache_only=bool(forcing_cfg.get("target_cache_only", False)),
                view_mode="base_only",
            )
        hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.spatial_forcing_teacher = teacher
        self.spatial_forcing_projector = SpatialForcingAlignProjector(
            input_dim=hidden_size,
            target_dim=int(teacher.embed_dim),
            use_vlm_norm=bool(forcing_cfg.get("use_vlm_norm", True)),
        )
        self.spatial_forcing_student_layer = int(forcing_cfg.get("student_layer", 24))
        self.spatial_forcing_target_tokens = int(forcing_cfg.get("target_tokens", 49))
        self.spatial_forcing_loss_weight = float(forcing_cfg.get("loss_weight", 0.5))
        logger.info(
            "[spatial_forcing_aux] qwen_layer=%d base_tokens=%d target_dim=%d weight=%.3f",
            self.spatial_forcing_student_layer,
            self.spatial_forcing_target_tokens,
            int(teacher.embed_dim),
            self.spatial_forcing_loss_weight,
        )

    def _compute_spatial_forcing_aux_loss(
        self,
        student_hidden: torch.Tensor,
        qwen_inputs: dict,
        images: List,
    ) -> torch.Tensor:
        vision_hidden = select_base_image_hidden(
            hidden=student_hidden,
            input_ids=qwen_inputs["input_ids"],
            attention_mask=qwen_inputs["attention_mask"],
            image_grid_thw=qwen_inputs["image_grid_thw"],
            image_token_id=int(self.qwen_vl_interface.model.config.image_token_id),
            spatial_merge_size=int(self.qwen_vl_interface.model.config.vision_config.spatial_merge_size),
            image_counts=[len(sample) if isinstance(sample, (list, tuple)) else 1 for sample in images],
        )
        if vision_hidden.shape[1] != self.spatial_forcing_target_tokens:
            raise ValueError(
                f"Spatial Forcing expected {self.spatial_forcing_target_tokens} base-image tokens, "
                f"got {vision_hidden.shape[1]}"
            )
        target = self.spatial_forcing_teacher.encode(
            images=images,
            device=student_hidden.device,
            target_tokens=vision_hidden.shape[1],
        )
        expected = (
            student_hidden.shape[0],
            1,
            vision_hidden.shape[1],
            int(self.spatial_forcing_teacher.embed_dim),
        )
        if tuple(target.shape) != expected:
            raise ValueError(f"Spatial Forcing VGGT target shape mismatch: {tuple(target.shape)} vs {expected}")
        with torch.autocast(
            device_type=student_hidden.device.type,
            dtype=torch.bfloat16,
            enabled=student_hidden.device.type == "cuda",
        ):
            return spatial_forcing_cosine_loss(
                self.spatial_forcing_projector,
                vision_hidden,
                target[:, 0],
            )

    def attach_latent_decoder(
        self,
        decoder,
        tokenizer,
        component_order,
        include_bbox: bool = True,
        max_target_tokens: int = 128,
        loss_weight: float = 0.0,
        source_checkpoint: str = "",
    ) -> None:
        """Attach Stage-1's auxiliary decoder without saving it in the VLA state dict."""
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        decoder.train()
        if hasattr(decoder, "model"):
            decoder.model.eval()
        object.__setattr__(self, "_latent_decoder", decoder)
        object.__setattr__(self, "_latent_decoder_tokenizer", tokenizer)
        object.__setattr__(self, "_latent_decoder_component_order", tuple(component_order))
        object.__setattr__(self, "_latent_decoder_include_bbox", bool(include_bbox))
        object.__setattr__(self, "_latent_decoder_max_target_tokens", int(max_target_tokens))
        object.__setattr__(self, "_latent_decoder_loss_weight", float(loss_weight))
        object.__setattr__(self, "_latent_decoder_source_checkpoint", str(source_checkpoint))
        logger.info(
            "[latent_decoder] attached frozen decoder loss_weight=%s source=%s",
            loss_weight,
            source_checkpoint,
        )

    def _compute_latent_decoder_loss(self, examples, vlm_outputs):
        decoder = getattr(self, "_latent_decoder", None)
        if decoder is None:
            return None
        latent_embeds = vlm_outputs.get("latent_embeds")
        latent_attention_mask = vlm_outputs.get("latent_attention_mask")
        if latent_embeds is None or latent_attention_mask is None:
            raise RuntimeError(
                "latent decoder is enabled, but forward_latent did not return latent embeddings"
            )

        from decoder.latent_text_decoder import build_latent_decoder_batch

        decoder_batch = build_latent_decoder_batch(
            examples=examples,
            latent_embeds=latent_embeds,
            latent_attention_mask=latent_attention_mask,
            tokenizer=getattr(self, "_latent_decoder_tokenizer"),
            component_order=getattr(self, "_latent_decoder_component_order"),
            include_bbox=getattr(self, "_latent_decoder_include_bbox", True),
            max_target_tokens=getattr(self, "_latent_decoder_max_target_tokens", 128),
        )
        if decoder_batch is None:
            return None
        decoder_outputs = decoder(
            latent_prefix=decoder_batch.latent_prefix,
            latent_prefix_attention_mask=decoder_batch.latent_prefix_attention_mask,
            target_input_ids=decoder_batch.target_input_ids,
            target_labels=decoder_batch.target_labels,
            target_attention_mask=decoder_batch.target_attention_mask,
            return_per_sample_loss=False,
        )
        return decoder_outputs["loss"]

    @torch.no_grad()
    def evaluate_latent_decoder(self, examples, max_new_tokens: int = 64):
        """Evaluate the attached decoder against the current VLM latent prefix."""
        decoder = getattr(self, "_latent_decoder", None)
        if decoder is None:
            return None

        from decoder.latent_text_decoder import build_latent_decoder_batch

        tokenizer = getattr(self, "_latent_decoder_tokenizer")
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        action_tokens = [example.get("action_tokens", "") for example in examples]

        decoder_was_training = decoder.training
        decoder.eval()
        try:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                solutions=None,
                action_tokens=action_tokens,
            )
            vlm_outputs = self.qwen_vl_interface.forward_latent(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs.get("pixel_values"),
                image_grid_thw=qwen_inputs.get("image_grid_thw"),
                labels=None,
                position_ids=qwen_inputs.get("position_ids"),
                return_latent_embeds=True,
            )
            decoder_batch = build_latent_decoder_batch(
                examples=examples,
                latent_embeds=vlm_outputs["latent_embeds"],
                latent_attention_mask=vlm_outputs["latent_attention_mask"],
                tokenizer=tokenizer,
                component_order=getattr(self, "_latent_decoder_component_order"),
                include_bbox=getattr(self, "_latent_decoder_include_bbox", True),
                max_target_tokens=getattr(self, "_latent_decoder_max_target_tokens", 128),
            )
            if decoder_batch is None:
                return None

            decoder_outputs = decoder(
                latent_prefix=decoder_batch.latent_prefix,
                latent_prefix_attention_mask=decoder_batch.latent_prefix_attention_mask,
                target_input_ids=decoder_batch.target_input_ids,
                target_labels=decoder_batch.target_labels,
                target_attention_mask=decoder_batch.target_attention_mask,
                return_per_sample_loss=False,
            )

            bos_id = tokenizer.bos_token_id
            if bos_id is None:
                bos_id = tokenizer.eos_token_id
            if bos_id is None:
                bos_id = tokenizer.pad_token_id
            if bos_id is None:
                raise ValueError("latent decoder tokenizer has no BOS, EOS, or PAD token")

            decoded_ids = decoder.greedy_decode(
                latent_prefix=decoder_batch.latent_prefix,
                latent_prefix_attention_mask=decoder_batch.latent_prefix_attention_mask,
                bos_token_id=int(bos_id),
                eos_token_id=(
                    int(tokenizer.eos_token_id)
                    if tokenizer.eos_token_id is not None
                    else None
                ),
                pad_token_id=(
                    int(tokenizer.pad_token_id)
                    if tokenizer.pad_token_id is not None
                    else None
                ),
                max_new_tokens=max(1, int(max_new_tokens)),
            )
            return {
                "loss": decoder_outputs["loss"],
                "decoded_texts": tokenizer.batch_decode(
                    decoded_ids,
                    skip_special_tokens=True,
                ),
                "target_texts": list(decoder_batch.target_texts),
                "tags": list(decoder_batch.tags),
                "num_samples": len(decoder_batch.target_texts),
            }
        finally:
            decoder.train(decoder_was_training)
            if decoder_was_training and hasattr(decoder, "model"):
                decoder.model.eval()

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load checkpoints while allowing Stage 2 to omit the spatial projector."""
        spatial_cfg = getattr(self.config.framework, "spatial", {}) if hasattr(self.config, "framework") else {}
        spatial_enabled = bool(spatial_cfg.get("enable", False))
        spatial_teacher = bool(spatial_cfg.get("use_teacher", True)) if spatial_enabled else False
        if spatial_enabled and not spatial_teacher:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if "spatial_align_projector." not in key
            }
        return super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        if self.training_stage == "spatial_only":
            return self._forward_spatial_only(examples)
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        action_tokens = [example.get("action_tokens", "") for example in examples]
        cot_mode = str(self.config.framework.get("cot_mode", "none")).lower()
        cot_solutions = None
        use_generated_cot = False
        if cot_mode == "explicit":
            if self.training_stage == "reasoning_only":
                instructions = [format_reasoning_prompt(instruction) for instruction in instructions]
            explicit_cot_cfg = self.config.framework.get("explicit_cot", {})
            action_input = str(explicit_cot_cfg.get("action_input", "gt")).lower()
            use_generated_cot = action_input in {"generated", "model", "sampled"}
            if use_generated_cot:
                with torch.no_grad():
                    cot_solutions, _ = self._generate_explicit_cot(
                        batch_images=batch_images,
                        instructions=instructions,
                    )
            else:
                cot_solutions = [example.get("cot_solution", "") for example in examples]
        # img_next: List of PIL list (primary view), fallback flags
        image_next = [example.get("image_next", None) for example in examples]
        image_next_fallback = torch.tensor(
            [bool(example.get("image_next_fallback", False)) for example in examples],
            device=self.qwen_vl_interface.model.device,
        )
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        object.__setattr__(self, "_spatial_cache_keys", [example.get("spatial_cache_key") for example in examples])
        

        # Step 1: QWenVL input format (tokenization and thinking token alignment if enabled)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions,
            solutions=cot_solutions,
            action_tokens=action_tokens,
        )
        if self.action_vggt_enabled or self.spatial_forcing_enabled:
            qwen_inputs.pop("labels", None)
        if use_generated_cot:
            qwen_inputs.pop("labels", None)
        # Check if iterative implicit reasoning is enabled
        enable_latent_reasoning = self.config.framework.get("enable_latent_reasoning", False)
        use_iterative_forward = (
            enable_latent_reasoning
            and hasattr(self.qwen_vl_interface, "forward_latent")
        )

        if use_iterative_forward:
            if self.spatial_forcing_enabled:
                raise ValueError("spatial_forcing_aux requires the normal Qwen forward with hidden-state layers")
            # Step 2: Iterative forward with KV-Cache for implicit reasoning
            vlm_outputs = self.qwen_vl_interface.forward_latent(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs.get("pixel_values"),
                image_grid_thw=qwen_inputs.get("image_grid_thw"),
                labels=qwen_inputs.get("labels"),  # May contain masked labels
                position_ids=qwen_inputs.get("position_ids"),
                return_latent_embeds=hasattr(self, "_latent_decoder"),
            )
            
            last_hidden = vlm_outputs['hidden_states']  # [B, L, H]
            vlm_loss = vlm_outputs.get('loss')  # May be None if no labels
        else:
            # Step 2: Normal forward pass (no iterative reasoning)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
                spatial_forcing_hidden = (
                    qwenvl_outputs.hidden_states[self.spatial_forcing_student_layer]
                    if self.spatial_forcing_enabled
                    else None
                )
                vlm_loss = qwenvl_outputs.loss if hasattr(qwenvl_outputs, 'loss') else None

        latent_decoder_loss = None
        latent_decoder_loss_scale = 1.0
        if use_iterative_forward and hasattr(self, "_latent_decoder"):
            decoder_cfg = self.config.framework.get("latent_decoder", {})
            decoder_interval = max(1, int(decoder_cfg.get("compute_interval", 1) or 1))
            optimizer_step = int(kwargs.get("optimizer_step", 0) or 0)
            should_compute_decoder = (
                not self.training
                or decoder_interval == 1
                or optimizer_step % decoder_interval == 0
            )
            if should_compute_decoder:
                latent_decoder_loss = self._compute_latent_decoder_loss(examples, vlm_outputs)
                if self.training and decoder_interval > 1 and latent_decoder_loss is not None:
                    latent_decoder_loss_scale = float(decoder_interval)

        backbone_attention_mask = qwen_inputs.get("attention_mask")
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        explicit_cot_cfg = self.config.framework.get("explicit_cot", {})
        action_context_max_tokens = int(explicit_cot_cfg.get("action_context_max_tokens", 0) or 0)
        if action_context_max_tokens > 0:
            last_hidden, backbone_attention_mask = self._limit_action_context(
                last_hidden=last_hidden,
                attention_mask=backbone_attention_mask,
                max_tokens=action_context_max_tokens,
            )

        action_condition_hidden, action_condition_mask = self._append_action_vggt_condition(
            hidden=last_hidden,
            attention_mask=backbone_attention_mask,
            images=batch_images,
        )

        # Step 3: Compute losses based on training stage
        result = {}

        spatial_forcing_loss = None
        if self.spatial_forcing_enabled:
            spatial_forcing_loss = self._compute_spatial_forcing_aux_loss(
                spatial_forcing_hidden,
                qwen_inputs,
                batch_images,
            )

        img_next_loss = None
        img_next_cfg = getattr(self.config.framework, "img_next", {}) if hasattr(self.config, "framework") else {}
        enable_img_next = img_next_cfg.get("enable", False)
        img_next_loss_weight = img_next_cfg.get("loss_weight", 0.5)
        img_next_res = img_next_cfg.get("res", 112)
        img_next_teacher_type = str(img_next_cfg.get("teacher_type", "qwen")).strip().lower()
        img_next_target_frame = str(img_next_cfg.get("target_frame", "next")).strip().lower()
        img_next_target_mode = str(img_next_cfg.get("target_mode", "absolute")).strip().lower()
        img_next_loss_type = str(img_next_cfg.get("loss_type", "l1")).strip().lower()
        img_next_token_id = getattr(self.qwen_vl_interface, "img_next_token_id", None)

        use_img_next_teacher = img_next_cfg.get("use_teacher", True)
        img_next_mask_for_action = (
            (qwen_inputs["input_ids"] == img_next_token_id) if img_next_token_id is not None else None
        )

        if (
            enable_img_next
            and use_img_next_teacher
            and img_next_token_id is not None
            and img_next_loss_weight is not None
            and img_next_loss_weight > 0
        ):
            img_next_mask = (qwen_inputs["input_ids"] == img_next_token_id)
            target_images = batch_images if img_next_target_frame == "current" else image_next
            target_fallback = (
                torch.zeros_like(image_next_fallback)
                if img_next_target_frame == "current"
                else image_next_fallback
            )
            try:
                img_next_loss = self._compute_img_next_loss(
                    last_hidden,
                    target_images,
                    img_next_mask,
                    target_fallback,
                    target_res=img_next_res,
                    teacher_type=img_next_teacher_type,
                    current_images=batch_images,
                    target_mode=img_next_target_mode,
                    loss_type=img_next_loss_type,
                )
            except Exception as e:
                if _is_vjepa21_teacher(img_next_teacher_type) or img_next_target_mode == "residual_ema_rms":
                    raise RuntimeError("ImageNext target generation failed") from e
                logger.warning(f"[img_next_loss] skipped due to error: {e}")
                img_next_loss = None
        
        spatial_cfg = getattr(self.config.framework, "spatial", {}) if hasattr(self.config, "framework") else {}
        enable_spatial = bool(spatial_cfg.get("enable", False))
        spatial_loss_weight = float(spatial_cfg.get("loss_weight", 0.0) or 0.0)
        spatial_base_id = getattr(self.qwen_vl_interface, "spatial_base_token_id", None)
        spatial_wrist_id = getattr(self.qwen_vl_interface, "spatial_wrist_token_id", None)
        spatial_view_mode = getattr(self.qwen_vl_interface, "spatial_view_mode", "dual")
        spatial_loss = None
        base_mask = (
            qwen_inputs["input_ids"] == int(spatial_base_id)
            if spatial_base_id is not None
            else None
        )
        wrist_mask = (
            qwen_inputs["input_ids"] == int(spatial_wrist_id)
            if spatial_wrist_id is not None
            else None
        )
        if (
            enable_spatial
            and bool(spatial_cfg.get("use_teacher", True))
            and spatial_loss_weight > 0
            and (
                spatial_base_id is not None
                if spatial_view_mode == "base_only"
                else spatial_base_id is not None and spatial_wrist_id is not None
            )
        ):
            if spatial_cfg.get("alignment_mode") == "cross_attention":
                spatial_loss = self._compute_dense_spatial_loss(last_hidden, qwen_inputs, batch_images)
            else:
                spatial_loss = self._compute_spatial_loss(
                    last_hidden=last_hidden,
                    target_images=batch_images,
                    base_mask=base_mask,
                    wrist_mask=wrist_mask,
                    view_mode=spatial_view_mode,
                )

        if self.training_stage == "reasoning_only":
            # Stage 1: Only train VLM reasoning, skip action head
            if (
                vlm_loss is None
                and img_next_loss is None
                and spatial_loss is None
                and latent_decoder_loss is None
            ):
                raise ValueError(
                    "training_stage='reasoning_only' requires at least one active "
                    "VLM, ImageNext, spatial, or decoder loss."
                )
            result["total_loss"] = last_hidden.new_zeros((), dtype=torch.float32)
            if vlm_loss is not None:
                result["vlm_loss"] = vlm_loss
                result["total_loss"] = result["total_loss"] + vlm_loss
            if img_next_loss is not None:
                result["img_next_loss"] = img_next_loss
                result["total_loss"] = result["total_loss"] + img_next_loss_weight * img_next_loss
            if spatial_loss is not None:
                result["spatial_loss"] = spatial_loss
                result["total_loss"] = result["total_loss"] + spatial_loss_weight * spatial_loss
            if latent_decoder_loss is not None:
                result["decoder_loss"] = latent_decoder_loss
                result["total_loss"] = (
                    result["total_loss"]
                    + getattr(self, "_latent_decoder_loss_weight", 0.0) * latent_decoder_loss_scale * latent_decoder_loss
                )
            return result

        elif self.training_stage == "action_only":
            # action_only mode: Only train action head, VLM is frozen
            with torch.autocast("cuda", dtype=torch.float32):
                # 标签对齐：取最后 chunk_len 段
                actions = torch.tensor(
                    np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
                )  # [B, T_full, action_dim]
                actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

                repeated_diffusion_steps = (
                    self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
                )
                actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                last_hidden_repeated = action_condition_hidden.repeat(repeated_diffusion_steps, 1, 1)
                action_attention_mask = (
                    action_condition_mask.repeat(repeated_diffusion_steps, 1)
                    if action_condition_mask is not None
                    else None
                )
                
                state_repeated = None
                if state is not None:
                    state = torch.tensor(
                        np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                    )  # [B, state_dim] or [B, 1, state_dim]
                    
                    # Ensure state is 3D: [B, 1, state_dim]
                    if state.ndim == 2:
                        state = state.unsqueeze(1)  # [B, state_dim] -> [B, 1, state_dim]
                    
                    state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)  # [B*repeated_diffusion_steps, 1, state_dim]

                action_loss = self.action_model(
                    last_hidden_repeated,
                    actions_target_repeated,
                    state_repeated,
                    encoder_attention_mask=action_attention_mask,
                )

                result["action_loss"] = action_loss
                result["total_loss"] = action_loss  # Only action loss
                if vlm_loss is not None:
                    result["vlm_loss"] = vlm_loss
                return result
        else:
            # full mode: Train both VLM and action head
            with torch.autocast("cuda", dtype=torch.float32):
                actions = torch.tensor(
                    np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
                )  # [B, T_full, action_dim]
                actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

                repeated_diffusion_steps = (
                    self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
                )
                actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                last_hidden_repeated = action_condition_hidden.repeat(repeated_diffusion_steps, 1, 1)
                action_attention_mask = (
                    action_condition_mask.repeat(repeated_diffusion_steps, 1)
                    if action_condition_mask is not None
                    else None
                )
                
                state_repeated = None
                if state is not None:
                    state = torch.tensor(
                        np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                    )  # [B, state_dim] or [B, 1, state_dim]
                    
                    if state.ndim == 2:
                        state = state.unsqueeze(1)
                    
                    state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

                action_loss = self.action_model(
                    last_hidden_repeated,
                    actions_target_repeated,
                    state_repeated,
                    encoder_attention_mask=action_attention_mask,
                )

            result["action_loss"] = action_loss
            
            # Combine with VLM loss if available
        if vlm_loss is not None:
            vlm_loss_weight = self.config.framework.get("latent_reasoning", {}).get("vlm_loss_weight", 0.5)
            result["vlm_loss"] = vlm_loss
            result["total_loss"] = action_loss + vlm_loss_weight * vlm_loss
        else:
            result["total_loss"] = action_loss

        if (
            img_next_loss is not None
            and enable_img_next
            and use_img_next_teacher
            and img_next_loss_weight > 0
        ):
            result["img_next_loss"] = img_next_loss
            result["total_loss"] = result["total_loss"] + img_next_loss_weight * img_next_loss
        if spatial_loss is not None:
            result["spatial_loss"] = spatial_loss
            result["total_loss"] = result["total_loss"] + spatial_loss_weight * spatial_loss
        if spatial_forcing_loss is not None:
            result["spatial_forcing_loss"] = spatial_forcing_loss
            result["total_loss"] = (
                result["total_loss"] + self.spatial_forcing_loss_weight * spatial_forcing_loss
            )
        if latent_decoder_loss is not None:
            result["decoder_loss"] = latent_decoder_loss
            result["total_loss"] = (
                result["total_loss"]
                + getattr(self, "_latent_decoder_loss_weight", 0.0) * latent_decoder_loss_scale * latent_decoder_loss
            )

        return result

    def _forward_spatial_only(self, examples) -> dict:
        if not examples:
            raise ValueError("spatial-only forward requires a nonempty image/language batch")
        images = [example["image"] for example in examples]
        if any(not isinstance(sample, (list, tuple)) or len(sample) != 2 for sample in images):
            raise ValueError("spatial-only Stage1 requires exactly base+wrist images per sample")
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images,
            instructions=[example["lang"] for example in examples],
            solutions=None,
        )
        inputs.pop("labels", None)
        last_hidden = self.qwen_vl_interface.forward_spatial_features(**inputs)
        spatial_loss = self._compute_dense_spatial_loss(last_hidden, inputs, images)
        loss_weight = float(self.config.framework.spatial.get("loss_weight", 0.1))
        return {"spatial_loss": spatial_loss, "total_loss": loss_weight * spatial_loss}

    def _compute_dense_spatial_loss(self, last_hidden, inputs, images) -> torch.Tensor:
        prediction = self.qwen_vl_interface.reconstruct_spatial_features(
            last_hidden, inputs, image_counts=[len(sample) for sample in images]
        )
        target = self.qwen_vl_interface.get_dense_spatial_target(images, last_hidden.device)
        return dense_spatial_cosine_loss(prediction, target)

    def _compute_spatial_loss(
        self,
        last_hidden: torch.Tensor,
        target_images: List,
        base_mask: torch.Tensor,
        wrist_mask: Optional[torch.Tensor],
        view_mode: str = "dual",
    ) -> Optional[torch.Tensor]:
        """Align spatial special-token hidden states to pooled VGGT features."""
        if last_hidden is None:
            raise ValueError("spatial supervision requires VLM hidden states")
        batch_size = last_hidden.shape[0]
        expected_count = int(getattr(self.qwen_vl_interface, "spatial_target_tokens", 16))
        base_counts = base_mask.sum(dim=1)
        if view_mode == "base_only":
            if not torch.all(base_counts == expected_count):
                raise ValueError(
                    "base-only spatial token count mismatch: "
                    f"base={base_counts.detach().cpu().tolist()}, "
                    f"expected={expected_count}"
                )
            pred = last_hidden[base_mask].view(batch_size, 1, expected_count, -1)
        else:
            if wrist_mask is None:
                raise ValueError("dual-view spatial supervision requires a wrist mask")
            wrist_counts = wrist_mask.sum(dim=1)
            if not torch.all(base_counts == expected_count) or not torch.all(
                wrist_counts == expected_count
            ):
                raise ValueError(
                    "spatial token count mismatch: "
                    f"base={base_counts.detach().cpu().tolist()}, "
                    f"wrist={wrist_counts.detach().cpu().tolist()}, "
                    f"expected_per_sample={expected_count}"
                )
            pred_base = last_hidden[base_mask].view(batch_size, expected_count, -1)
            pred_wrist = last_hidden[wrist_mask].view(batch_size, expected_count, -1)
            pred = torch.stack([pred_base, pred_wrist], dim=1)
        target = self.qwen_vl_interface.get_external_spatial_target(
            images=target_images,
            device=last_hidden.device,
            cache_keys=getattr(self, "_spatial_cache_keys", None),
        )
        pred = self.qwen_vl_interface.project_spatial_prediction(pred)
        if pred.shape != target.shape:
            raise ValueError(
                f"VGGT spatial shape mismatch: prediction={tuple(pred.shape)} target={tuple(target.shape)}"
            )
        spatial_cfg = getattr(self.config.framework, "spatial", {})
        loss_type = str(spatial_cfg.get("loss_type", "cosine")).strip().lower()
        if loss_type == "l2":
            return F.mse_loss(pred.float(), target.float())
        elif loss_type == "cosine":
            cosine = F.cosine_similarity(pred.float(), target.float(), dim=-1)
            return (1.0 - cosine).mean()
        raise ValueError(f"unsupported spatial loss_type={loss_type!r}")

    def _compute_img_next_loss(
        self,
        last_hidden: torch.Tensor,
        target_images: List,
        img_next_mask: torch.Tensor,
        fallback_mask: torch.Tensor,
        target_res: int = 112,
        teacher_type: str = "qwen",
        current_images: Optional[List] = None,
        target_mode: str = "absolute",
        loss_type: str = "l1",
    ) -> Optional[torch.Tensor]:
        """
        Align ImageNext hidden states with absolute or normalized residual visual targets.
        """
        if last_hidden is None or target_images is None or len(target_images) == 0:
            return None

        # shape check for mask
        if img_next_mask is None or not torch.any(img_next_mask):
            return None

        device = last_hidden.device
        dtype = last_hidden.dtype

        # Extract predicted embeddings at img_next positions
        try:
            # mask shape [B, L]; expect count per sample = img_next_count (16)
            B = last_hidden.shape[0]
            token_counts = img_next_mask.sum(dim=1)
            if not torch.all(token_counts == token_counts[0]):
                raise ValueError(f"inconsistent ImageNext token counts: {token_counts.tolist()}")
            img_next_count = int(token_counts[0].item())
            pred = last_hidden[img_next_mask].view(B, img_next_count, -1)
        except Exception as e:
            logger.warning(f"[img_next_loss] mask reshape failed: {e}")
            return None

        try:
            target_feats = self._encode_img_next_target_features(
                images=target_images,
                target_res=target_res,
                teacher_type=teacher_type,
                target_tokens=img_next_count,
                device=device,
                dtype=dtype,
            )
            if target_feats is None:
                return None
            normalized_target_mode = str(target_mode).strip().lower()
            valid_samples = ~fallback_mask.to(device=device, dtype=torch.bool)
            if normalized_target_mode == "residual_ema_rms":
                if current_images is None:
                    raise ValueError("residual_ema_rms ImageNext requires current_images")
                current_feats = self._encode_img_next_target_features(
                    images=current_images,
                    target_res=target_res,
                    teacher_type=teacher_type,
                    target_tokens=img_next_count,
                    device=device,
                    dtype=dtype,
                )
                if current_feats is None:
                    return None
                target_feats = self.qwen_vl_interface.normalize_img_next_residual(
                    (target_feats - current_feats).detach(),
                    valid_samples,
                ).detach()
            elif normalized_target_mode != "absolute":
                raise ValueError(f"unsupported ImageNext target_mode={target_mode!r}")
            pred = self.qwen_vl_interface.project_img_next_prediction(pred)
        except Exception as e:
            if _is_vjepa21_teacher(teacher_type) or str(target_mode).strip().lower() == "residual_ema_rms":
                raise
            logger.warning(f"[img_next_loss] visual encoding failed: {e}")
            return None

        return masked_img_next_loss(pred, target_feats, ~fallback_mask, loss_type)

    def _encode_img_next_target_features(
        self,
        images: List,
        target_res: int,
        teacher_type: str,
        target_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        flat_images = []
        for sample_images in images:
            if isinstance(sample_images, list):
                flat_images.append(sample_images[0] if sample_images else None)
            else:
                flat_images.append(sample_images)
        if not flat_images or any(image is None for image in flat_images):
            logger.warning("[img_next_loss] missing image target")
            return None
        if _is_vjepa21_teacher(teacher_type):
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                return self.qwen_vl_interface.get_external_img_next_target(
                    images=flat_images,
                    device=device,
                    dtype=dtype,
                    target_tokens=target_tokens,
                )
        if target_res is not None and int(target_res) > 0:
            resized = []
            for image in flat_images:
                try:
                    resized.append(image.resize((int(target_res), int(target_res))))
                except Exception:
                    resized.append(image)
            flat_images = resized
        processor = getattr(self.qwen_vl_interface, "processor", None)
        if processor is None and hasattr(self.qwen_vl_interface, "model"):
            processor = getattr(self.qwen_vl_interface.model, "processor", None)
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            logger.warning("[img_next_loss] image processor is unavailable")
            return None
        with torch.no_grad():
            processor_output = dict(image_processor(images=flat_images, return_tensors="pt"))
            pixel_values = processor_output.get("pixel_values")
            image_grid_thw = processor_output.get("image_grid_thw")
            if pixel_values is None:
                logger.warning("[img_next_loss] processor returned no pixel_values")
                return None
            pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device=device, non_blocking=True)
            main_model = getattr(self.qwen_vl_interface, "model", None)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                if hasattr(self.qwen_vl_interface, "get_image_features_target"):
                    image_embeddings, _ = self.qwen_vl_interface.get_image_features_target(
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                    )
                elif main_model is not None and hasattr(main_model, "get_image_features"):
                    image_embeddings, _ = main_model.get_image_features(
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                    )
                else:
                    logger.warning("[img_next_loss] image feature encoder is unavailable")
                    return None
        if isinstance(image_embeddings, (list, tuple)):
            features = torch.stack([embedding for embedding in image_embeddings], dim=0)
        else:
            features = image_embeddings
        if features is None or features.numel() == 0:
            logger.warning("[img_next_loss] extracted features are empty")
            return None
        features = features.to(device=device, dtype=dtype)
        if features.dim() == 2:
            features = features.unsqueeze(0)
        grid_side = int(features.shape[1] ** 0.5)
        target_side = int(target_tokens ** 0.5)
        if grid_side * grid_side != features.shape[1] or target_side * target_side != target_tokens:
            logger.warning(
                f"[img_next_loss] unexpected token grid: tokens={features.shape[1]}, "
                f"target={target_tokens}"
            )
            return None
        features_2d = features.transpose(1, 2).reshape(
            features.shape[0],
            features.shape[2],
            grid_side,
            grid_side,
        )
        features_2d = F.adaptive_avg_pool2d(features_2d, output_size=(target_side, target_side))
        return features_2d.flatten(2).transpose(1, 2)

    def _limit_action_context(
        self,
        last_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_tokens: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if max_tokens <= 0 or last_hidden.shape[1] <= max_tokens:
            return last_hidden, attention_mask

        if attention_mask is None or attention_mask.shape[:2] != last_hidden.shape[:2]:
            hidden = last_hidden[:, :max_tokens, :]
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            return hidden, mask

        mask = attention_mask.to(device=last_hidden.device, dtype=torch.bool)
        selected = []
        selected_masks = []
        head_tokens = max_tokens // 2
        tail_tokens = max_tokens - head_tokens
        for sample_hidden, sample_mask in zip(last_hidden, mask):
            valid_hidden = sample_hidden[sample_mask]
            if valid_hidden.shape[0] > max_tokens:
                valid_hidden = torch.cat(
                    [valid_hidden[:head_tokens], valid_hidden[-tail_tokens:]],
                    dim=0,
                )
            valid_count = valid_hidden.shape[0]
            if valid_count < max_tokens:
                valid_hidden = F.pad(valid_hidden, (0, 0, 0, max_tokens - valid_count))
            selected.append(valid_hidden)
            selected_masks.append(
                torch.arange(max_tokens, device=last_hidden.device) < valid_count
            )
        return torch.stack(selected, dim=0), torch.stack(selected_masks, dim=0)

    def _get_fast_action_processor(self):
        if self._fast_action_processor is not None:
            return self._fast_action_processor

        bridge_cfg = getattr(self.config.datasets.vla_data, "bridge_annotations", {}) or {}
        fast_tokenizer_name = bridge_cfg.get("fast_tokenizer_name", "playground/Pretrained_models/fast")
        fast_tokenizer = Fast_Action_Tokenizer(fast_tokenizer_name=fast_tokenizer_name)
        processor = fast_tokenizer.fast_tokenizer
        setattr(processor, "time_horizon", 1)
        setattr(processor, "action_dim", int(self.config.framework.action_model.action_dim))
        self._fast_action_processor = processor
        logger.info("Initialized FAST action processor from %s", fast_tokenizer_name)
        return processor

    def _resolve_fast_action_token_range(self) -> Tuple[int, int]:
        if self._fast_action_token_range is not None:
            return self._fast_action_token_range

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        action_min = tokenizer.convert_tokens_to_ids("<robot_action_0>")
        action_max = tokenizer.convert_tokens_to_ids("<robot_action_2047>")
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if (
            action_min is None
            or action_max is None
            or action_min == unk_id
            or action_max == unk_id
            or action_min < 0
            or action_max < 0
        ):
            raise ValueError(
                "FAST action tokens are not present in the loaded VLM tokenizer. "
                "Use the action-token-augmented Qwen checkpoint."
            )
        if action_min > action_max:
            action_min, action_max = action_max, action_min

        self._fast_action_token_range = (int(action_min), int(action_max))
        return self._fast_action_token_range

    def _extract_fast_action_token_ids(self, generated_ids: torch.LongTensor) -> List[List[int]]:
        action_min, action_max = self._resolve_fast_action_token_range()
        mask = (generated_ids >= action_min) & (generated_ids <= action_max)
        batch_tokens = []
        for batch_idx in range(generated_ids.size(0)):
            idx = mask[batch_idx].nonzero(as_tuple=False).flatten()
            tokens = generated_ids[batch_idx, idx].detach().cpu().tolist() if idx.numel() else []
            batch_tokens.append([int(token) for token in tokens])
        return batch_tokens

    def _decode_fast_actions(self, batch_vlm_tokens: List[List[int]]) -> np.ndarray:
        processor = self._get_fast_action_processor()
        action_min, _ = self._resolve_fast_action_token_range()
        action_dim = int(self.config.framework.action_model.action_dim)
        zero_action = np.zeros((1, action_dim), dtype=np.float32)
        decoded_actions = []

        setattr(processor, "time_horizon", 1)
        setattr(processor, "action_dim", action_dim)
        for vlm_tokens in batch_vlm_tokens:
            fast_ids = [int(token) - action_min for token in vlm_tokens]
            if not fast_ids:
                logger.warning("FAST generation produced no action tokens; using zero action")
                decoded_actions.append(zero_action.copy())
                continue
            try:
                decoded = processor.decode([fast_ids])
                action = np.asarray(decoded[0] if isinstance(decoded, list) else decoded, dtype=np.float32)
                action = np.squeeze(action)
                if action.ndim == 1 and action.size == action_dim:
                    action = action.reshape(1, action_dim)
                elif action.ndim == 1 and action.size > action_dim:
                    action = action.reshape(-1, action_dim)[:1]
                elif action.ndim == 3 and action.shape[0] == 1:
                    action = action[0]
                if action.ndim != 2 or action.shape[-1] != action_dim:
                    raise ValueError(f"decoded FAST action has invalid shape {action.shape}")
                decoded_actions.append(action[:1].astype(np.float32, copy=False))
            except Exception as exc:
                logger.exception("FAST decode failed for %d tokens: %s", len(fast_ids), exc)
                decoded_actions.append(zero_action.copy())

        logger.info("FAST action token counts: %s", [len(tokens) for tokens in batch_vlm_tokens])
        return np.stack(decoded_actions, axis=0)

    def _prepare_fast_generation_inputs(self, qwen_inputs):
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs["attention_mask"]
        pad_id = int(tokenizer.pad_token_id)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        newline_ids = tokenizer("\n", add_special_tokens=False).input_ids
        action_prefix_ids = tokenizer(" Action: ", add_special_tokens=False).input_ids

        if not action_prefix_ids:
            raise ValueError("Failed to tokenize FAST action prefix")

        prepared = []
        max_len = 0
        for ids, mask in zip(input_ids, attention_mask):
            valid_len = int(mask.sum().item())
            seq = ids[:valid_len]
            while seq.numel() > 0 and int(seq[-1].item()) in newline_ids:
                seq = seq[:-1]
            if im_end_id is not None and im_end_id >= 0 and seq.numel() > 0 and int(seq[-1].item()) == int(im_end_id):
                seq = seq[:-1]
            prefix = torch.tensor(action_prefix_ids, device=seq.device, dtype=seq.dtype)
            seq = torch.cat([seq, prefix], dim=0)
            prepared.append(seq)
            max_len = max(max_len, int(seq.numel()))

        padded_ids = input_ids.new_full((len(prepared), max_len), pad_id)
        padded_mask = attention_mask.new_zeros((len(prepared), max_len))
        for row_idx, seq in enumerate(prepared):
            seq_len = int(seq.numel())
            padded_ids[row_idx, :seq_len] = seq
            padded_mask[row_idx, :seq_len] = 1

        qwen_inputs["input_ids"] = padded_ids
        qwen_inputs["attention_mask"] = padded_mask
        return qwen_inputs

    def _greedy_generate_fast_with_latent(self, qwen_inputs, max_new_tokens: int) -> torch.LongTensor:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        action_min, action_max = self._resolve_fast_action_token_range()
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        generated = []
        seen_action = torch.zeros((qwen_inputs["input_ids"].shape[0],), device=qwen_inputs["input_ids"].device, dtype=torch.bool)

        for _ in range(max_new_tokens):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.qwen_vl_interface.forward_latent(
                    input_ids=qwen_inputs["input_ids"],
                    attention_mask=qwen_inputs["attention_mask"],
                    pixel_values=qwen_inputs.get("pixel_values"),
                    image_grid_thw=qwen_inputs.get("image_grid_thw"),
                )
            logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token)

            qwen_inputs["input_ids"] = torch.cat([qwen_inputs["input_ids"], next_token], dim=1)
            next_mask = torch.ones_like(next_token, dtype=qwen_inputs["attention_mask"].dtype)
            qwen_inputs["attention_mask"] = torch.cat([qwen_inputs["attention_mask"], next_mask], dim=1)

            is_action = (next_token[:, 0] >= action_min) & (next_token[:, 0] <= action_max)
            seen_action |= is_action
            is_stop = torch.zeros_like(seen_action)
            if eos_id is not None:
                is_stop |= next_token[:, 0] == int(eos_id)
            if pad_id is not None:
                is_stop |= next_token[:, 0] == int(pad_id)
            is_stop |= seen_action & (~is_action)
            if bool(is_stop.all().item()):
                break

        if not generated:
            return qwen_inputs["input_ids"].new_empty((qwen_inputs["input_ids"].shape[0], 0))
        return torch.cat(generated, dim=1)

    @torch.inference_mode()
    def _predict_action_fast(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        **kwargs,
    ) -> dict:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        qwen_inputs.pop("labels", None)
        qwen_inputs = self._prepare_fast_generation_inputs(qwen_inputs)

        max_new_tokens = int(kwargs.get("fast_max_new_tokens", 256))
        do_sample = bool(kwargs.get("do_sample", False))
        use_iterative_forward = bool(kwargs.get("use_iterative_forward", False)) and hasattr(
            self.qwen_vl_interface, "forward_latent"
        )
        if use_iterative_forward:
            if do_sample:
                logger.warning("FAST latent generation currently uses greedy decoding; ignoring do_sample=True")
            generated_ids = self._greedy_generate_fast_with_latent(qwen_inputs, max_new_tokens=max_new_tokens)
        else:
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.qwen_vl_interface.processor.tokenizer.pad_token_id,
                "eos_token_id": self.qwen_vl_interface.processor.tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = float(kwargs.get("temperature", 0.7))

            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated_ids = self.qwen_vl_interface.model.generate(**qwen_inputs, **gen_kwargs)

            prompt_len = qwen_inputs["input_ids"].shape[1]
            generated_ids = generated_ids[:, prompt_len:]
        batch_vlm_tokens = self._extract_fast_action_token_ids(generated_ids)
        normalized_actions = self._decode_fast_actions(batch_vlm_tokens)
        return {
            "normalized_actions": normalized_actions,
            "thinking_gen_time": 0.0,
            "fast_action_token_counts": [len(tokens) for tokens in batch_vlm_tokens],
        }

    def _generate_explicit_cot(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        **kwargs,
    ) -> Tuple[List[str], float]:
        processor = self.qwen_vl_interface.processor
        messages = []
        for imgs, instruction in zip(batch_images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": format_reasoning_prompt(instruction)})
            messages.append([{"role": "user", "content": content}])

        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.qwen_vl_interface.model.device)
        finally:
            processor.tokenizer.padding_side = old_padding_side

        explicit_cot_cfg = self.config.framework.get("explicit_cot", {})
        max_new_tokens = int(kwargs.get("cot_max_new_tokens", explicit_cot_cfg.get("max_new_tokens", 192)))
        do_sample = bool(kwargs.get("cot_do_sample", explicit_cot_cfg.get("do_sample", False)))
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": processor.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(kwargs.get("cot_temperature", 0.2))

        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated = self.qwen_vl_interface.model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - t0

        prompt_len = inputs["input_ids"].shape[1]
        generated = generated[:, prompt_len:]
        cot_texts = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [text.strip() for text in cot_texts], elapsed

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Inference: predict future actions via latent reasoning + diffusion sampling.

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL
             - forward_latent for implicit reasoning (iterative KV-Cache)
             - Fallback to normal forward if forward_latent unavailable
          3. Action model prediction from hidden states

        Returns:
            dict with normalized_actions (np.ndarray [B, T, action_dim]).
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        if bool(kwargs.get("use_fast_action_tokens", False)):
            if self.action_vggt_enabled:
                raise ValueError("fast action-token inference bypasses action_vggt_condition")
            return self._predict_action_fast(batch_images=batch_images, instructions=instructions, **kwargs)
    
        use_iterative_forward = bool(kwargs.get("use_iterative_forward", False)) and hasattr(
            self.qwen_vl_interface, "forward_latent"
        )
        cot_mode = str(kwargs.get("cot_mode", getattr(self.config.framework, "cot_mode", "none"))).lower()

        # Step 1: QWenVL input format
        thinking_gen_time = 0.0
        explicit_cot = None
        if cot_mode == "explicit":
            explicit_cot, thinking_gen_time = self._generate_explicit_cot(
                batch_images=batch_images,
                instructions=instructions,
                **kwargs,
            )
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                solutions=explicit_cot,
            )
            qwen_inputs.pop("labels", None)
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask")
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        # Step 2: Forward pass
        if use_iterative_forward:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vlm_outputs = self.qwen_vl_interface.forward_latent(
                    input_ids=qwen_inputs["input_ids"],
                    attention_mask=qwen_inputs["attention_mask"],
                    pixel_values=qwen_inputs.get("pixel_values"),
                    image_grid_thw=qwen_inputs.get("image_grid_thw"),
                )
                # forward_latent returns a dict with 'hidden_states', 'num_reasoning_passes', etc.
                last_hidden = vlm_outputs['hidden_states']  # [B, L, H]
                
                # Optional: Log reasoning passes for debugging
                num_passes = vlm_outputs.get('num_reasoning_passes', 0)
                if num_passes > 0:
                    logger.info(f" Completed {num_passes} reasoning passes in predict_action")
        else:
            # Baseline mode: Normal forward pass (no iterative reasoning)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        action_condition_hidden, action_condition_mask = self._append_action_vggt_condition(
            hidden=last_hidden,
            attention_mask=backbone_attention_mask,
            images=batch_images,
        )

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                action_condition_hidden,
                state,
                encoder_attention_mask=action_condition_mask,
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        output = {"normalized_actions": normalized_actions, "thinking_gen_time": thinking_gen_time}
        if explicit_cot is not None:
            output["explicit_cot"] = explicit_cot
        return output



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./laravla/config/training/bridge.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)

    # Smoke test with fake data
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }

    batch = [sample, sample]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    forward_output = model(batch)
    print(f"Action Loss: {forward_output['action_loss'].item()}")

    predict_output = model.predict_action(
        batch_images=[batch[0]["image"]],
        instructions=[batch[0]["lang"]],
        state=[batch[0]["state"]],
    )
    print(f"Predicted Action: {predict_output['normalized_actions']}")
