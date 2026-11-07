# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
QwenPI_v3 Framework
A Qwen2.5-VL / Qwen3-VL + layer-wise cross-DiT flow-matching action head.

Released checkpoint
─────────────────────────────
- Qwen3-VL-4B + Bridge V2 + RT-1 (OXE) co-training, 69.8% avg success on
  SimplerEnv WidowX:
  https://huggingface.co/StarVLA/Qwen3VL-PI_v3-Bridge-RT_1

Key improvements over QwenPI
─────────────────────────────
1. **Compressed Action DiT via per-layer projectors**
   Each of the N VLM hidden-state layers is passed through a dedicated
   LayerNorm + Linear projector (`project_layers`) that maps the VLM hidden
   dimension (e.g. 2560) down to a smaller Action DiT latent dimension
   (e.g. 1024, controlled by `action_dit_hidden_dim`).  This reduces the
   action head parameter count by ~(vl_hidden / dit_hidden)² while keeping
   the full layer-wise cross-attention structure.

2. **Discretised-state language injection** (`add_discretized_state_to_instruction`)
   Proprioceptive state is quantised into 256 bins and appended to the
   language instruction as plain tokens (``[STATE] <bins> [ACTION]``),
   following the π₀.5 design.  This lets the VLM attend to state without
   any extra encoder module.

Together these two features bring QwenPI_v3 close to all the core
capabilities of π₀.5 within a single open-weight VLM framework.

Parameter breakdown (Qwen3-VL-4B + action_dit_hidden_dim=1024)
═══════════════════════════════════════════════════════════════
  Module                               Params        %
  ───────────────────────────────────────────────────
  qwen_vl_interface         4,437,815,808   87.5%
  action_model                538,678,305   10.6%
  project_layers               94,593,024    1.9%
  ───────────────────────────────────────────────────
  TOTAL                     5,071,087,137  100.0%
═══════════════════════════════════════════════════════════════
"""
from dataclasses import dataclass, field
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.vlm.future_residual import masked_future_loss
from starVLA.model.modules.vlm.spatial_forcing_aux import (
    SpatialForcingAlignProjector,
    select_base_image_hidden,
    spatial_forcing_cosine_loss,
)
from starVLA.model.modules.vlm.vggt_teacher import VGGTTeacher
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenPI_v3
#  - Same shape as QwenPIDefaultConfig (see QwenPI.py) but introduces the
#    optional `action_dit_hidden_dim` knob inside diffusion_model_cfg.
#  - Setting action_dit_hidden_dim to a value smaller than the VLM hidden
#    size lets the Action DiT run at a "compressed" latent dim while
#    `Qwen_PI_v3.project_layers` does the LayerNorm+Linear compression of
#    each VL hidden state to that dim.
#  - Leaving it None (or omitting it from YAML) reproduces the QwenPI
#    behaviour: DiT hidden = VLM hidden, projection becomes nn.Identity().
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenPI_v3DefaultConfig:
    """QwenPI_v3 framework default parameters.

    See ``starVLA/model/framework/VLM4A/diffusion_model_cfg.md`` for the
    relationship between vl_hidden_dim, action_dit_hidden_dim and
    cross_attention_dim.
    """

    name: str = "QwenPI_v3"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,  # auto-overridden at runtime from the loaded VLM
            "num_vl_layers": 36,  # auto-overridden at runtime from the loaded VLM
        }
    )

    future: dict = field(
        default_factory=lambda: {
            "enable": False,
            "token": "<future>",
            "token_count": 16,
            "loss_weight": 0.1,
            "use_teacher": True,
            "teacher_update_mode": "ema",
            "ema_momentum": 0.999,
            "teacher_input_size": 112,
            "target_mode": "residual_ema_rms",
            "loss_type": "l2",
            "prediction_mlp_ratio": 2.0,
            "residual_ema_decay": 0.99,
            "residual_ema_eps": 1.0e-6,
        }
    )

    latent_reasoning: dict = field(default_factory=lambda: {"enable": False, "stage": 2, "token_count": 3})

    spatial_forcing_aux: dict = field(
        default_factory=lambda: {
            "enable": False,
            "input_size": 518,
            "preprocess_mode": "resize",
            "target_tokens": 49,
            "student_layer": 24,
            "teacher_layer": -1,
            "patch_start_idx": 5,
            "pooling_func": "bilinear",
            "use_vggt_pe": True,
            "use_vlm_norm": True,
            "loss_weight": 0.5,
            "teacher_repo": "",
            "target_cache_size": 0,
            "target_cache_dir": "",
            "target_cache_only": False,
        }
    )

    # === Action head (Layer-wise Flow-matching / cross-DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 16,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                # When set (e.g. 1024), DiT internal hidden = action_dit_hidden_dim
                # and Qwen_PI_v3.project_layers compress VL hidden to this dim.
                # When None, DiT internal hidden = vl_hidden_dim (== QwenPI behaviour).
                "action_dit_hidden_dim": 1024,
                "dropout": 0.2,
                "final_dropout": True,
                # A single mode switch: false is legacy all-cross, true is
                # canonical interleaved self/cross attention.
                # A checkpoint config containing this field overrides the
                # default; for old HF releases, use the matching code snapshot.
                "interleave_self_attention": False,
                # QwenPI_v3 intentionally keeps the historical layer-wise
                # all-cross-attention semantics by default, without changing
                # defaults for other frameworks.
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenPI_v3")
class Qwen_PI_v3(baseframework):
    """
    Multimodal vision-language-action model (QwenPI_v3 variant).

    Architecture
    ────────────
    - Qwen2.5-VL / Qwen3-VL backbone for fused language / vision token embeddings.
    - Per-layer projectors (``project_layers``): one LayerNorm + Linear per VLM
      layer that compresses VLM hidden states from ``vl_hidden_dim`` down to
      ``action_dit_hidden_dim`` before feeding the Action DiT.
    - Layer-wise cross-DiT flow-matching action head that attends to every
      selected VLM layer in parallel.

    QwenPI_v3 defaults to the historical legacy all-cross-attention DiT
    forward. Set ``interleave_self_attention=true`` only for a checkpoint
    explicitly trained with the canonical interleaved forward. The historical
    ``use_canonical_forward`` boolean remains accepted as a compatibility alias
    for existing launchers and is mapped to the same single mode switch.

    Focus: predict a future action chunk conditioned on multi-view images
    and a natural-language instruction (with optional discretised state prefix).
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
        # Merge framework defaults with YAML config (YAML wins on conflicts).
        self.config = merge_framework_config(QwenPI_v3DefaultConfig, config)
        from starVLA.model.modules.vlm.thinking_config import validate_stage2_decoder_config

        validate_stage2_decoder_config(self.config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Read the actual hidden size and layer count from the loaded VLM.
        # `output_hidden_states=True` returns (num_hidden_layers + 1) tensors
        # (embedding output + every layer's output), and we keep the last
        # `num_hidden_layers` of them for layer-wise cross-attn — so the DiT
        # depth and project_layers count must match `num_hidden_layers` exactly.
        # Qwen3-VL stores num_hidden_layers under text_config; Qwen2.5-VL puts it
        # on the top-level config.  getattr(..., vlm_hf_cfg) handles both cases.
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # Resolve the Action DiT hidden dim BEFORE building the action head,
        # so that LayerwiseFlowmatchingActionHead constructs DiT at the right size.
        # If the user did not specify it, fall back to the LLM hidden size
        # (i.e. behave like QwenPI: project_layers becomes nn.Identity()).
        #
        # NOTE: `action_dit_hidden_dim` is a framework-side hint only — it is
        # NOT a DiT constructor kwarg, so we keep it out of diffusion_model_cfg
        # and instead pass it through `populate_layerwise_dit_cfg`, which writes
        # the canonical DiT-shape fields (input_embedding_dim, cross_attention_dim,
        # num_attention_heads).
        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", None)
        if action_dit_hidden_dim is None:
            action_dit_hidden_dim = llm_hidden_size
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)

        # Push the resolved DiT shape into diffusion_model_cfg.  The action head
        # is intentionally agnostic of qwenvl.* — it only consumes this dict.
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)

        # Layer-wise projector: map each selected VL hidden to Action DiT hidden space.
        # This explicitly decouples VL representation size from action DiT latent size.
        self.project_layers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if llm_hidden_size == self.action_dit_hidden_dim
                    else nn.Sequential(
                        nn.LayerNorm(llm_hidden_size),
                        nn.Linear(llm_hidden_size, self.action_dit_hidden_dim),
                    )
                )
                for _ in range(self.num_action_dit_layers)
            ]
        )

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self._init_spatial_forcing_aux(
            llm_hidden_size=llm_hidden_size,
            num_vl_layers=num_vl_layers,
        )
        self._init_latent_decoder(llm_hidden_size)

    def _init_latent_decoder(self, llm_hidden_size: int) -> None:
        """Build a trainable Stage1 decoder or opt-in frozen Stage2 decoder."""
        from starVLA.model.modules.vlm.thinking_config import validate_stage2_decoder_config

        validate_stage2_decoder_config(self.config)
        latent_cfg = self.config.framework.get("latent_reasoning", {})
        self.latent_stage = int(latent_cfg.get("stage", 2)) if latent_cfg.get("enable", False) else 0
        if self.latent_stage not in (0, 1, 2):
            raise ValueError("latent_reasoning.stage must be 1 or 2")
        self.latent_decoder = None
        self._latent_decoder_frozen = bool(latent_cfg.get("freeze_decoder", False))
        if self.latent_stage == 1 or bool(latent_cfg.get("stage2_decoder_supervision", False)):
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            from starVLA.model.modules.vlm.latent_text_decoder import FullQwenLatentTextDecoder

            decoder_path = str(latent_cfg.get("decoder_base_vlm", self.config.framework.qwenvl.base_vlm))
            decoder_model = Qwen3VLForConditionalGeneration.from_pretrained(
                decoder_path, attn_implementation="sdpa", dtype=torch.bfloat16,
                local_files_only=bool(latent_cfg.get("decoder_local_files_only", False)),
            )
            decoder_hidden_size = int(decoder_model.config.text_config.hidden_size)
            if self._latent_decoder_frozen and decoder_hidden_size != llm_hidden_size:
                raise ValueError("Frozen 4B decoder requires matching latent/decoder hidden sizes; refusing a frozen random projection")
            decoder_model.model.visual.requires_grad_(False)
            decoder_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            decoder_model.config.use_cache = False
            self.latent_decoder = FullQwenLatentTextDecoder(
                qwen_model=decoder_model,
                hidden_size=decoder_hidden_size,
                vocab_size=int(decoder_model.config.text_config.vocab_size),
                latent_hidden_size=llm_hidden_size,
                micro_batch_size=int(latent_cfg.get("decoder_micro_batch_size", 7)),
                logit_row_chunk_size=int(latent_cfg.get("decoder_logit_row_chunk_size", 64)),
                use_forward_checkpoint=bool(latent_cfg.get("decoder_use_forward_checkpoint", True)),
                input_projection_type=str(latent_cfg.get("decoder_input_projector", "auto")),
            )
            self._latent_decoder_tokenizer = AutoProcessor.from_pretrained(
                decoder_path, local_files_only=bool(latent_cfg.get("decoder_local_files_only", False)),
            ).tokenizer
            # Target trimming and EOS insertion require right padding. This is
            # independent of the student's left-padded multimodal tokenizer.
            self._latent_decoder_tokenizer.padding_side = "right"
            if self._latent_decoder_frozen:
                self.latent_decoder.requires_grad_(False)
                if latent_cfg.get("decoder_input_projector", "auto") == "residual_mlp":
                    self.latent_decoder.latent_input_projection.requires_grad_(True)
                self.latent_decoder.model.eval()
                logging.info(
                    "Frozen decoder LM; input_projector=%s trainable_projector_params=%d",
                    latent_cfg.get("decoder_input_projector", "auto"),
                    sum(p.numel() for p in self.latent_decoder.latent_input_projection.parameters() if p.requires_grad),
                )
        if self.latent_stage == 1:
            self.action_model.requires_grad_(False)
            self.project_layers.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_latent_decoder_frozen", False) and self.latent_decoder is not None:
            # Keep the frozen LM deterministic while the outer decoder wrapper
            # stays in train mode to enable differentiable forward checkpointing.
            self.latent_decoder.model.eval()
        return self

    def _init_spatial_forcing_aux(self, llm_hidden_size: int, num_vl_layers: int) -> None:
        forcing_cfg = self.config.framework.get("spatial_forcing_aux", {})
        self.spatial_forcing_enabled = bool(forcing_cfg.get("enable", False))
        self.spatial_forcing_teacher = None
        self.spatial_forcing_projector = None
        self.spatial_forcing_student_layer = 0
        self.spatial_forcing_target_tokens = 0
        self.spatial_forcing_loss_weight = 0.0
        if not self.spatial_forcing_enabled:
            return

        self.spatial_forcing_student_layer = int(forcing_cfg.get("student_layer", 24))
        if not 0 <= self.spatial_forcing_student_layer <= num_vl_layers:
            raise ValueError(
                "spatial_forcing_aux.student_layer must index Qwen hidden_states, "
                f"got {self.spatial_forcing_student_layer} for {num_vl_layers} layers"
            )
        self.spatial_forcing_target_tokens = int(forcing_cfg.get("target_tokens", 49))
        target_side = int(self.spatial_forcing_target_tokens**0.5)
        if target_side * target_side != self.spatial_forcing_target_tokens:
            raise ValueError(
                "spatial_forcing_aux.target_tokens must form a square grid, "
                f"got {self.spatial_forcing_target_tokens}"
            )

        with torch.random.fork_rng(devices=[]):
            teacher = VGGTTeacher(
                teacher_repo=str(forcing_cfg.get("teacher_repo", "")),
                input_size=int(forcing_cfg.get("input_size", 518)),
                layer=int(forcing_cfg.get("teacher_layer", -1)),
                patch_start_idx=int(forcing_cfg.get("patch_start_idx", 5)),
                use_vggt_pe=bool(forcing_cfg.get("use_vggt_pe", True)),
                preprocess_mode=str(forcing_cfg.get("preprocess_mode", "resize")),
                cache_size=int(forcing_cfg.get("target_cache_size", 0) or 0),
                cache_dir=str(forcing_cfg.get("target_cache_dir", "") or ""),
                cache_only=bool(forcing_cfg.get("target_cache_only", False)),
                view_mode="base_only",
            )
        self.spatial_forcing_teacher = teacher
        self.spatial_forcing_projector = SpatialForcingAlignProjector(
            input_dim=llm_hidden_size,
            target_dim=int(teacher.embed_dim),
            use_vlm_norm=bool(forcing_cfg.get("use_vlm_norm", True)),
        )
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

    def _project_vl_hidden_for_action(self, vl_embs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project layer-wise VL hidden states to the hidden space expected by Action DiT."""
        if len(vl_embs_list) != len(self.project_layers):
            raise ValueError(
                f"Layer number mismatch: got {len(vl_embs_list)} VL layers, "
                f"but project_layers has {len(self.project_layers)} layers."
            )
        return [proj(vl_h) for proj, vl_h in zip(self.project_layers, vl_embs_list)]

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str]
    ) -> tuple:
        """Run QwenVL and retain action, Future, and Spatial Forcing features."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.latent_stage:
                latent_outputs = self.qwen_vl_interface.forward_latent(**qwen_inputs)
                all_hidden_states = latent_outputs["hidden_states"]
            else:
                latent_outputs = None
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                all_hidden_states = qwenvl_outputs.hidden_states
            raw_vl_embs_list = list(all_hidden_states[-self.num_action_dit_layers:])
            raw_last_hidden = raw_vl_embs_list[-1]
            spatial_forcing_hidden = (
                all_hidden_states[self.spatial_forcing_student_layer]
                if self.spatial_forcing_enabled
                else None
            )
            vl_embs_list = self._project_vl_hidden_for_action(raw_vl_embs_list) if self.latent_stage != 1 else None
        return (
            vl_embs_list,
            raw_last_hidden,
            attention_mask,
            qwen_inputs["input_ids"],
            spatial_forcing_hidden,
            qwen_inputs,
            latent_outputs,
        )

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  # List[List[PIL.Image]], length B
        instructions = [example["lang"] for example in examples]  # List[str], length B
        actions = [example["action"] for example in examples] if self.latent_stage != 1 else None
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        # Prepend discretised proprioceptive state to each instruction string.
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )
        state = None  # state is now encoded in the instruction tokens

        # Step 1: encode through QwenVL
        (
            vl_embs_list,
            raw_last_hidden,
            backbone_attention_mask,
            input_ids,
            spatial_forcing_hidden,
            qwen_inputs,
            latent_outputs,
        ) = self._encode_vl_hidden_states(batch_images, instructions)
        decoder_loss = None
        latent_cfg = self.config.framework.get("latent_reasoning", {})
        if self.latent_stage == 1 or bool(latent_cfg.get("stage2_decoder_supervision", False)):
            from starVLA.model.modules.vlm.latent_text_decoder import build_latent_decoder_batch

            decoder_batch = build_latent_decoder_batch(
                examples=examples,
                latent_embeds=latent_outputs["latent_embeds"],
                latent_attention_mask=latent_outputs["latent_attention_mask"],
                tokenizer=self._latent_decoder_tokenizer,
                component_order=("SUBTASK", "BBOX", "REASON"),
                include_bbox=True,
                max_target_tokens=int(self.config.framework.latent_reasoning.get("max_target_tokens", 128)),
            )
            if decoder_batch is None or len(decoder_batch.tags) != len(examples) * 3:
                raise ValueError("Decoder supervision requires complete subtask, bbox and reasoning targets")
            decoder_loss = self.latent_decoder(
                latent_prefix=decoder_batch.latent_prefix,
                latent_prefix_attention_mask=decoder_batch.latent_prefix_attention_mask,
                target_input_ids=decoder_batch.target_input_ids,
                target_labels=decoder_batch.target_labels,
                target_attention_mask=decoder_batch.target_attention_mask,
            )["loss"]
        if self.latent_stage == 1:
            weight = float(latent_cfg.get("decoder_loss_weight", 1.0))
            result = {"action_loss": decoder_loss.detach().new_zeros(()), "decoder_loss": decoder_loss,
                      "total_loss": weight * decoder_loss}
        else:
            result = None
            base_hidden = vl_embs_list[-1]

        # Step 2: compute flow-matching loss over the action chunk
        if self.latent_stage != 1:
            with torch.autocast("cuda", dtype=torch.float32):
                # Align labels: keep only the last action_horizon timesteps.
                actions = torch.tensor(
                    np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
                )
                actions_target = actions[:, -self.action_horizon :, :]
                repeated_diffusion_steps = (
                    self.config.trainer.get("repeated_diffusion_steps", 16) if self.config and self.config.trainer else 4
                )
                actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
                if backbone_attention_mask is not None:
                    backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                        dtype=torch.bool
                    )
                state_repeated = None
                if state is not None:
                    state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                    state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)
                action_loss = self.action_model(
                    vl_embs_list_repeated,
                    actions_target_repeated,
                    state_repeated,
                    encoder_attention_mask=backbone_attention_mask,
                )
            result = {"action_loss": action_loss, "total_loss": action_loss}
            if decoder_loss is not None:
                result["decoder_loss"] = decoder_loss
                result["total_loss"] = action_loss + float(latent_cfg.get("decoder_loss_weight", 1.0)) * decoder_loss
        future_cfg = self.config.framework.get("future", {})
        if self.training and bool(future_cfg.get("enable", False)):
            future_loss = self._compute_future_loss(
                raw_last_hidden=raw_last_hidden,
                input_ids=input_ids,
                current_images=batch_images,
                future_images=[example.get("image_next") for example in examples],
                fallback_mask=torch.tensor(
                    [bool(example.get("image_next_fallback", True)) for example in examples],
                    device=raw_last_hidden.device,
                    dtype=torch.bool,
                ),
            )
            future_loss_weight = float(future_cfg.get("loss_weight", 0.1))
            result["future_loss"] = future_loss
            result["total_loss"] = result["total_loss"] + future_loss_weight * future_loss
        if self.training and self.spatial_forcing_enabled:
            spatial_forcing_loss = self._compute_spatial_forcing_aux_loss(
                student_hidden=spatial_forcing_hidden,
                qwen_inputs=qwen_inputs,
                images=batch_images,
            )
            result["spatial_forcing_loss"] = spatial_forcing_loss
            result["total_loss"] = (
                result["total_loss"] + self.spatial_forcing_loss_weight * spatial_forcing_loss
            )
        return result

    def _compute_future_loss(
        self,
        raw_last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        current_images: List,
        future_images: List,
        fallback_mask: torch.Tensor,
    ) -> torch.Tensor:
        future_token_id = getattr(self.qwen_vl_interface, "future_token_id", None)
        if future_token_id is None:
            raise RuntimeError("Future token was not initialized")
        future_mask = input_ids == int(future_token_id)
        token_counts = future_mask.sum(dim=1)
        expected_count = int(self.config.framework.future.get("token_count", 16))
        if not torch.all(token_counts == expected_count):
            raise ValueError(f"Expected {expected_count} Future tokens per sample, got {token_counts.tolist()}")
        batch_size = raw_last_hidden.shape[0]
        prediction = raw_last_hidden[future_mask].view(batch_size, expected_count, -1)
        current_features = self._encode_future_target_features(
            images=current_images,
            target_tokens=expected_count,
            device=raw_last_hidden.device,
            dtype=raw_last_hidden.dtype,
        )
        future_features = self._encode_future_target_features(
            images=future_images,
            target_tokens=expected_count,
            device=raw_last_hidden.device,
            dtype=raw_last_hidden.dtype,
        )
        valid_mask = ~fallback_mask.to(device=raw_last_hidden.device, dtype=torch.bool)
        target = self.qwen_vl_interface.normalize_future_residual(
            (future_features - current_features).detach(),
            valid_mask,
        ).detach()
        prediction = self.qwen_vl_interface.project_future_prediction(prediction)
        return masked_future_loss(prediction, target, valid_mask)

    def _encode_future_target_features(
        self,
        images: List,
        target_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        first_view_images = []
        for sample_images in images:
            if not isinstance(sample_images, (list, tuple)) or not sample_images:
                raise ValueError("Future supervision requires the first camera view for every sample")
            image = sample_images[0]
            target_size = int(self.config.framework.future.get("teacher_input_size", 112))
            first_view_images.append(image.resize((target_size, target_size)))

        image_processor = getattr(self.qwen_vl_interface.processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError("Qwen image processor is unavailable for Future supervision")
        with torch.no_grad():
            processor_output = dict(image_processor(images=first_view_images, return_tensors="pt"))
            pixel_values = processor_output.get("pixel_values")
            image_grid_thw = processor_output.get("image_grid_thw")
            if pixel_values is None or image_grid_thw is None:
                raise RuntimeError("Qwen image processor did not return pixel_values and image_grid_thw")
            pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
            image_grid_thw = image_grid_thw.to(device=device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                image_embeddings, _ = self.qwen_vl_interface.get_future_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
        features = torch.stack(list(image_embeddings), dim=0).to(device=device, dtype=dtype)
        grid_side = int(features.shape[1] ** 0.5)
        target_side = int(target_tokens ** 0.5)
        if grid_side * grid_side != features.shape[1] or target_side * target_side != target_tokens:
            raise ValueError(
                f"Future target grid mismatch: source_tokens={features.shape[1]}, target_tokens={target_tokens}"
            )
        features_2d = features.transpose(1, 2).reshape(
            features.shape[0], features.shape[2], grid_side, grid_side
        )
        return F.adaptive_avg_pool2d(features_2d, output_size=(target_side, target_side)).flatten(2).transpose(1, 2)

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Run inference and return a denoised action trajectory.

        Steps:
          1. Optionally resize images to the training observation resolution.
          2. Encode images + instruction (with discretised state prefix) through QwenVL.
          3. Project layer-wise VLM hidden states to the Action DiT latent space.
          4. Run the flow-matching sampler to produce the action chunk.

        Args:
            examples: List[dict], each entry requires:
                - image: List[PIL.Image] (multi-view)
                - lang:  str instruction
                - state: np.ndarray shaped (1, state_dim), optional

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape (B, action_horizon, action_dim),
                    denoised actions in the normalised action space.
        """

        batch_images = [to_pil_preserve(example["image"]) for example in examples]  # List[List[PIL.Image]]
        instructions = [example["lang"] for example in examples]  # List[str]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # List[ndarray] or None

        # Encode proprioceptive state into the instruction string, then discard raw state.
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )
        state = None

        # Optionally resize images to the resolution used during training.
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: encode through QwenVL
        if self.latent_stage == 1:
            raise RuntimeError("Stage 1 has no action training; use a Stage 2 checkpoint for inference")
        vl_embs_list, _, backbone_attention_mask, _, _, _, _ = self._encode_vl_hidden_states(
            batch_images, instructions
        )
        base_hidden = vl_embs_list[-1]
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        # Step 2: run the flow-matching sampler to produce the denoised action chunk.
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list, state, encoder_attention_mask=backbone_attention_mask
            )  # (B, action_horizon, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def state2str_transform(self, state: np.ndarray) -> str:
        """Quantise a state vector into 256 uniform bins and return it as a space-separated token string.

        Follows the π₀.5 convention: bins span [-1, 1] uniformly.
        Example: [-0.5, 0.1, 0.8] -> "95 133 203"
        """
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return " ".join(map(str, discretized_state))

    def add_discretized_state_to_instruction(self, instructions: List[str], states: List[np.ndarray]) -> List[str]:
        """Append discretised proprioceptive state tokens to each instruction.

        Format: ``<original instruction> [STATE] <bin indices> [ACTION]``
        This lets the VLM attend to the robot state purely through its
        existing text-token pathway — no extra encoder required.
        """
        updated_instructions = []
        for instr, state in zip(instructions, states):
            state_str = self.state2str_transform(state[0])
            updated_instructions.append(f"{instr} [STATE] {state_str} [ACTION]")
        return updated_instructions


if __name__ == "__main__":
    import argparse

    import debugpy
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()
    import os

    if os.getenv("DEBUG_MODE", "0") == "1":
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()
    args.config_yaml = "examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"

    model = Qwen_PI_v3(cfg)
    # ckpt="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1011_qwenpi/checkpoints/need_steps_10000_pytorch_model.pt"
    # model = Qwen_PI.from_pretrained(ckpt)
    print(model)

    def print_model_size(m: nn.Module, depth: int = 1):
        """Print parameter counts for each top-level submodule (depth=1)."""
        total = sum(p.numel() for p in m.parameters())
        print(f"\n{'='*55}")
        print(f"{'Module':<35} {'Params':>12}  {'%':>6}")
        print(f"{'-'*55}")
        for name, child in m.named_children():
            n = sum(p.numel() for p in child.parameters())
            print(f"  {name:<33} {n:>12,}  {100*n/total:>5.1f}%")
        print(f"{'-'*55}")
        print(f"  {'TOTAL':<33} {total:>12,}  100.0%")
        print(f"{'='*55}\n")

    print_model_size(model, depth=1)

    # fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),  # action_chunk, action_dim
        "image": [image, image],  # two views
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }

    batch = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    # # # Advance: try forward model with dataloader
    # # # can be fake sample, but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # vla_dataset_cfg.include_state = True

    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # #
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch)
