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

logger = initialize_overwatch(__name__)

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
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用
        self._fast_action_processor = None
        self._fast_action_token_range = None

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        # Training stage control: "reasoning_only", "action_only", or "full"
        self.training_stage = config.framework.get("training_stage", "full")

        # Apply parameter freezing based on training stage
        if self.training_stage == "reasoning_only":
            print(f"[Training Stage] reasoning_only mode - Freezing action_model parameters")
            for param in self.action_model.parameters():
                param.requires_grad = False
        elif self.training_stage == "action_only":
            print(f"[Training Stage] action_only mode - Freezing VLM parameters")
            for param in self.qwen_vl_interface.parameters():
                param.requires_grad = False
        else:
            print(f"[Training Stage] full mode - All parameters trainable")
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
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
        

        # Step 1: QWenVL input format (tokenization and thinking token alignment if enabled)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions,
            solutions=cot_solutions,
            action_tokens=action_tokens,
        )
        if use_generated_cot:
            qwen_inputs.pop("labels", None)
        # Check if iterative implicit reasoning is enabled
        enable_latent_reasoning = self.config.framework.get("enable_latent_reasoning", False)
        use_iterative_forward = (
            enable_latent_reasoning
            and hasattr(self.qwen_vl_interface, "forward_latent")
        )

        if use_iterative_forward:
            # Step 2: Iterative forward with KV-Cache for implicit reasoning
            vlm_outputs = self.qwen_vl_interface.forward_latent(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs.get("pixel_values"),
                image_grid_thw=qwen_inputs.get("image_grid_thw"),
                labels=qwen_inputs.get("labels"),  # May contain masked labels
                position_ids=qwen_inputs.get("position_ids"),
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
                vlm_loss = qwenvl_outputs.loss if hasattr(qwenvl_outputs, 'loss') else None

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

        # Step 3: Compute losses based on training stage
        result = {}

        img_next_loss = None
        img_next_cfg = getattr(self.config.framework, "img_next", {}) if hasattr(self.config, "framework") else {}
        enable_img_next = img_next_cfg.get("enable", False)
        img_next_loss_weight = img_next_cfg.get("loss_weight", 0.5)
        img_next_res = img_next_cfg.get("res", 112)
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
            try:
                img_next_loss = self._compute_img_next_loss(
                    last_hidden,
                    image_next,
                    img_next_mask,
                    image_next_fallback,
                    target_res=img_next_res,
                )
            except Exception as e:
                logger.warning(f"[img_next_loss] skipped due to error: {e}")
                img_next_loss = None
        
        if self.training_stage == "reasoning_only":
            # Stage 1: Only train VLM reasoning, skip action head
            if vlm_loss is None:
                raise ValueError(
                    "training_stage='reasoning_only' requires VLM loss, but vlm_loss is None. "
                    "Please ensure enable_latent_reasoning=True and labels are provided."
                )
            result["vlm_loss"] = vlm_loss
            if img_next_loss is not None:
                result["img_next_loss"] = img_next_loss
                result["total_loss"] = vlm_loss + img_next_loss_weight * img_next_loss
            else:
                result["total_loss"] = vlm_loss
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
                last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
                action_attention_mask = (
                    backbone_attention_mask.repeat(repeated_diffusion_steps, 1)
                    if backbone_attention_mask is not None
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
                last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
                action_attention_mask = (
                    backbone_attention_mask.repeat(repeated_diffusion_steps, 1)
                    if backbone_attention_mask is not None
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

        return result

    def _compute_img_next_loss(
        self,
        last_hidden: torch.Tensor,
        image_next: List,
        img_next_mask: torch.Tensor,
        fallback_mask: torch.Tensor,
        target_res: int = 112,
    ) -> Optional[torch.Tensor]:
        """
        Compute L1 loss between img_next token hidden states and visual encoder features of next frame.
        """
        if last_hidden is None or image_next is None or len(image_next) == 0:
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
            img_next_count = img_next_mask.sum(dim=1).max().item()
            pred = last_hidden[img_next_mask].view(B, img_next_count, -1)
        except Exception as e:
            logger.warning(f"[img_next_loss] mask reshape failed: {e}")
            return None

        try:
            # 获取 processor
            proc = getattr(self.qwen_vl_interface, "processor", None)
            if proc is None and hasattr(self.qwen_vl_interface, "model"):
                proc = getattr(self.qwen_vl_interface.model, "processor", None)
            
            if proc is None:
                logger.warning("[img_next_loss] processor is None, skip img_next_loss")
                return None
            
            # Use only the primary (first) view for img_next loss to match the single-view Bridge setup,
            # while remaining compatible with single-view datasets (non-list entries).
            flat_images = []
            for sample_imgs in image_next:
                if isinstance(sample_imgs, list):
                    flat_images.append(sample_imgs[0] if len(sample_imgs) > 0 else None)
                else:
                    flat_images.append(sample_imgs)
            
            if len(flat_images) == 0:
                logger.warning("[img_next_loss] no images to process")
                return None

            # Resize next-frame images before processor to ensure `res` takes effect.
            if target_res is not None and int(target_res) > 0:
                resized = []
                for img in flat_images:
                    try:
                        resized.append(img.resize((int(target_res), int(target_res))))
                    except Exception:
                        resized.append(img)
                flat_images = resized
            
           
            img_processor = getattr(proc, "image_processor", None)
            if img_processor is None:
                logger.warning("[img_next_loss] processor.image_processor is None, skip img_next_loss")
                return None
            with torch.no_grad():
                proc_out = img_processor(images=flat_images, return_tensors="pt")
                proc_out = dict(proc_out)
                pixel_values = proc_out.get("pixel_values", None)
                image_grid_thw = proc_out.get("image_grid_thw", None)
                if pixel_values is None:
                    logger.warning("[img_next_loss] processor returned None pixel_values")
                    return None
                pixel_values = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(device=device, non_blocking=True)

                main_model = getattr(self.qwen_vl_interface, "model", None)
                if main_model is None:
                    logger.warning("[img_next_loss] main_model.get_image_features not available")
                    return None
                
                # Prefer EMA teacher vision encoder when available.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if hasattr(self.qwen_vl_interface, "get_image_features_target"):
                        img_embeds, _ = self.qwen_vl_interface.get_image_features_target(
                            pixel_values=pixel_values, image_grid_thw=image_grid_thw
                        )
                    else:
                        if not hasattr(main_model, "get_image_features"):
                            logger.warning("[img_next_loss] main_model.get_image_features not available")
                            return None
                        img_embeds, _ = main_model.get_image_features(
                            pixel_values=pixel_values, image_grid_thw=image_grid_thw
                        )
                
              
                if isinstance(img_embeds, (list, tuple)):
                    feats = torch.stack([emb for emb in img_embeds], dim=0).to(device, dtype)
                else:
                    feats = img_embeds.to(device, dtype)
                
                if feats is None or feats.numel() == 0:
                    logger.warning("[img_next_loss] extracted features are empty")
                    return None
                
                if feats.dim() == 2:
                    feats = feats.unsqueeze(0)

                grid_side = int(feats.shape[1] ** 0.5)
                target_side = int(img_next_count ** 0.5)
                if grid_side * grid_side != feats.shape[1] or target_side * target_side != img_next_count:
                    logger.warning(f"[img_next_loss] unexpected token grid: tokens={feats.shape[1]}, target={img_next_count}")
                    return None

                feats_2d = feats.transpose(1, 2).reshape(feats.shape[0], feats.shape[2], grid_side, grid_side)
                feats_2d = F.adaptive_avg_pool2d(feats_2d, output_size=(target_side, target_side))
                target_feats = feats_2d.flatten(2).transpose(1, 2)  # [B, target_tokens, C]
        except Exception as e:
            logger.warning(f"[img_next_loss] visual encoding failed: {e}")
            return None

        valid_mask = (~fallback_mask).float().view(-1, 1, 1)
        if valid_mask.sum() <= 0:
            return None

        l1 = torch.nn.functional.l1_loss(pred, target_feats, reduction="none")  # [B, tokens, C]
        mask_full = valid_mask.expand_as(l1)
        l1 = (l1 * mask_full).sum() / mask_full.sum()
        return l1

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

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                state,
                encoder_attention_mask=backbone_attention_mask,
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
