# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import torch
import copy
from typing import Optional, List, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from torch.nn.utils.rnn import pad_sequence


from laravla.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)
BASE_PROMPT = (
    "Robot task reasoning: first output the Subtask to perform next, then output the BBox "
    "of the active object, then describe the current Spatial State of the gripper, active "
    "object, and task reference, and finally generate the Motion Reasoning. Instruction:"
)
IGNORE_INDEX = -100


def format_reasoning_prompt(instruction: str) -> str:
    """Use one user prompt for explicit-CoT and latent-reasoning stages."""
    instruction = (instruction or "").strip()
    if instruction.startswith(BASE_PROMPT):
        return instruction
    return f"{BASE_PROMPT} {instruction}".strip()

IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"

# [151936, 153984]
_ACTION_TOKEN_MIN = 151936 # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
_ACTION_TOKEN_MAX = 153984 # here only for fast_tokenizer, see laravla/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        cache_dir = qwenvl_config.get("cache_dir", None)
        attn_impl = qwenvl_config.get("attn_implementation", "flash_attention_2")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
            device_map="cuda",
            cache_dir=cache_dir,
        )
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        
        
        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        
        enable_latent_reasoning = config.framework.get("enable_latent_reasoning", False)
        if enable_latent_reasoning:
            token_ids = self._add_thinking_tokens(self.processor.tokenizer, config)
            self.thinking_token_id = token_ids["thinking_token_id"]
            self.start_thinking_id = token_ids["start_thinking_id"]
            self.end_thinking_id = token_ids["end_thinking_id"]
            logger.info(f"Added thinking tokens: thinking={token_ids['thinking_token_id']}, "
                        f"start={token_ids['start_thinking_id']}, end={token_ids['end_thinking_id']}")
        else:
            self.thinking_token_id = None
            self.start_thinking_id = None
            self.end_thinking_id = None

        img_next_cfg = getattr(config.framework, "img_next", {}) if hasattr(config, "framework") else {}
        enable_img_next = False
        try:
            enable_img_next = img_next_cfg.get("enable", False)
        except Exception:
            enable_img_next = False

        if enable_img_next:
            self.img_next_token_id = self._add_img_next_token(self.processor.tokenizer, config)
            logger.info(f"Added img_next token id={self.img_next_token_id}")
        else:
            self.img_next_token_id = None

        # img_next EMA target vision encoder (teacher)
        self.enable_img_next = bool(enable_img_next)
        self.use_img_next_teacher = bool(img_next_cfg.get("use_teacher", True)) if enable_img_next else False
        self.img_next_ema_momentum = 0.999
        self.visual_ema: Optional[nn.Module] = None
        if self.enable_img_next and self.use_img_next_teacher:
            self._init_img_next_visual_ema()

    def _get_student_visual(self) -> Optional[nn.Module]:
        """
        Return the student vision encoder used by Qwen3-VL `get_image_features`.

        Upstream (transformers) implementation uses:
          Qwen3VLForConditionalGeneration.get_image_features -> self.model.get_image_features
          Qwen3VLModel.get_image_features -> self.visual(...)
        """
        base = getattr(self.model, "model", None)
        if base is None:
            return None
        return getattr(base, "visual", None)

    def _init_img_next_visual_ema(self) -> None:
        student_visual = self._get_student_visual()
        if student_visual is None:
            logger.warning("[img_next_ema] student visual encoder not found; disable img_next EMA teacher")
            self.visual_ema = None
            return
        try:
            self.visual_ema = copy.deepcopy(student_visual)
            self.visual_ema.requires_grad_(False)
            self.visual_ema.eval()
            logger.info("[img_next_ema] Initialized EMA teacher vision encoder")
        except Exception as exc:
            logger.warning(f"[img_next_ema] Failed to init EMA teacher vision encoder: {exc}")
            self.visual_ema = None

    @torch.no_grad()
    def update_img_next_ema(self, momentum: Optional[float] = None) -> None:
        """
        EMA update for teacher vision encoder parameters:
          ema = m * ema + (1-m) * student
        """
        if (not self.enable_img_next) or (not getattr(self, "use_img_next_teacher", True)):
            return
        student_visual = self._get_student_visual()
        teacher_visual = getattr(self, "visual_ema", None)
        if student_visual is None or teacher_visual is None:
            return

        m = float(self.img_next_ema_momentum if momentum is None else momentum)
        if not (0.0 <= m <= 1.0):
            raise ValueError(f"EMA momentum must be in [0,1], got {m}")

        for p_ema, p in zip(teacher_visual.parameters(), student_visual.parameters()):
            p_ema.data.mul_(m).add_(p.data, alpha=1.0 - m)
        for b_ema, b in zip(teacher_visual.buffers(), student_visual.buffers()):
            b_ema.copy_(b)

    @torch.no_grad()
    def get_image_features_target(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        """
        Teacher-only image feature extraction for img_next alignment.
        Matches transformers `Qwen3VLModel.get_image_features` behavior but uses EMA vision encoder.
        """
        base = getattr(self.model, "model", None)
        if base is None or not hasattr(base, "visual"):
            raise RuntimeError("Qwen3 base model visual encoder not found")

        teacher_visual = self.visual_ema if self.visual_ema is not None else base.visual
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required for Qwen3-VL image feature splitting")

        pv = pixel_values.type(teacher_visual.dtype)
        image_embeds, deepstack_image_embeds = teacher_visual(pv, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // teacher_visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def _add_thinking_tokens(self, tokenizer, cfg):
        """
        Add thinking tokens to tokenizer and initialize embeddings.
        
        Args:
            tokenizer: Qwen3-VL processor's tokenizer
            cfg: Configuration object containing latent_reasoning settings
            
        Returns:
            dict: Contains thinking_token_id, start_thinking_id, end_thinking_id
        """
        latent_cfg = cfg.framework.get("latent_reasoning", {})
        thinking_token = latent_cfg.get("thinking_token", "<|thinking|>")
        start_token = latent_cfg.get("start_of_thinking_token", "<|start_of_thinking|>")
        end_token = latent_cfg.get("end_of_thinking_token", "<|end_of_thinking|>")
        
        existing_tokens = set(tokenizer.get_vocab().keys())
        tokens_to_add = []
        
        if thinking_token not in existing_tokens:
            tokens_to_add.append(thinking_token)
        if start_token not in existing_tokens:
            tokens_to_add.append(start_token)
        if end_token not in existing_tokens:
            tokens_to_add.append(end_token)
        
        if tokens_to_add:
            logger.info(f"Adding thinking tokens to tokenizer: {tokens_to_add}")
            tokenizer.add_tokens(tokens_to_add, special_tokens=True)
        
        old_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        new_vocab_size = len(tokenizer)
        if new_vocab_size > old_vocab_size:
            logger.info(f"Resizing model embeddings from {old_vocab_size} to {new_vocab_size}")
            self.model.resize_token_embeddings(new_vocab_size)
        
        thinking_token_id = tokenizer.convert_tokens_to_ids(thinking_token)
        start_thinking_id = tokenizer.convert_tokens_to_ids(start_token)
        end_thinking_id = tokenizer.convert_tokens_to_ids(end_token)
        
        if thinking_token_id == tokenizer.unk_token_id:
            raise ValueError(f"Failed to add thinking token: {thinking_token}")
        if start_thinking_id == tokenizer.unk_token_id:
            raise ValueError(f"Failed to add start thinking token: {start_token}")
        if end_thinking_id == tokenizer.unk_token_id:
            raise ValueError(f"Failed to add end thinking token: {end_token}")
        
        embeddings = self.model.get_input_embeddings()
        
        target_token = "<<"
        if target_token not in tokenizer.get_vocab():
            target_token = tokenizer.pad_token if tokenizer.pad_token else tokenizer.bos_token
        target_id = tokenizer.convert_tokens_to_ids(target_token)
        
        if target_id == tokenizer.unk_token_id:
            target_id = 0
            while target_id < len(tokenizer) and (
                tokenizer.convert_ids_to_tokens(target_id).startswith("<") or
                tokenizer.convert_ids_to_tokens(target_id).endswith(">")
            ):
                target_id += 1
        
        target_embedding = embeddings.weight.data[target_id].clone()
        
        for token_id in [thinking_token_id, start_thinking_id, end_thinking_id]:
            if token_id < embeddings.weight.shape[0]:
                embeddings.weight.data[token_id] = target_embedding
        
        if hasattr(self.model, 'lm_head') and self.model.lm_head is not None:
            if not (hasattr(self.model, 'tie_word_embeddings') and self.model.tie_word_embeddings):
                lm_head = self.model.lm_head
                if hasattr(lm_head, 'weight'):
                    target_lm_weight = lm_head.weight.data[target_id].clone()
                    for token_id in [thinking_token_id, start_thinking_id, end_thinking_id]:
                        if token_id < lm_head.weight.shape[0]:
                            lm_head.weight.data[token_id] = target_lm_weight
        
        logger.info("Initialized new thinking token embeddings")
        
        return {
            "thinking_token_id": thinking_token_id,
            "start_thinking_id": start_thinking_id,
            "end_thinking_id": end_thinking_id,
        }

    def _add_img_next_token(self, tokenizer, cfg) -> int:
        """
        Add img_next special token and initialize embeddings.

        Returns:
            int: img_next token id
        """
        img_cfg = cfg.framework.get("img_next", {}) if hasattr(cfg, "framework") else {}
        img_next_token = img_cfg.get("token", "<img_next>")

        existing_tokens = set(tokenizer.get_vocab().keys())
        tokens_to_add = []
        if img_next_token not in existing_tokens:
            tokens_to_add.append(img_next_token)

        if tokens_to_add:
            logger.info(f"Adding img_next token to tokenizer: {tokens_to_add}")
            tokenizer.add_tokens(tokens_to_add, special_tokens=True)

        old_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        new_vocab_size = len(tokenizer)
        if new_vocab_size > old_vocab_size:
            logger.info(f"Resizing embeddings for img_next from {old_vocab_size} to {new_vocab_size}")
            self.model.resize_token_embeddings(new_vocab_size)

        img_next_token_id = tokenizer.convert_tokens_to_ids(img_next_token)
        if img_next_token_id == tokenizer.unk_token_id:
            raise ValueError(f"Failed to add img_next token: {img_next_token}")

        embeddings = self.model.get_input_embeddings()
        target_token = "<<"
        if target_token not in tokenizer.get_vocab():
            target_token = tokenizer.pad_token if tokenizer.pad_token else tokenizer.bos_token
        target_id = tokenizer.convert_tokens_to_ids(target_token)

        if target_id == tokenizer.unk_token_id:
            target_id = 0
            while target_id < len(tokenizer):
                tok = tokenizer.convert_ids_to_tokens(target_id)
                if not (tok.startswith("<") and tok.endswith(">")):
                    break
                target_id += 1

        target_embedding = embeddings.weight.data[target_id].clone()
        embeddings.weight.data[img_next_token_id] = target_embedding

        # If LM head is not tied, also init
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            if not (hasattr(self.model, "tie_word_embeddings") and self.model.tie_word_embeddings):
                lm_head = self.model.lm_head
                if hasattr(lm_head, "weight"):
                    target_lm_weight = lm_head.weight.data[target_id].clone()
                    if img_next_token_id < lm_head.weight.shape[0]:
                        lm_head.weight.data[img_next_token_id] = target_lm_weight

        return img_next_token_id

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def _should_enable_img_next_full_attention(self) -> bool:
        """
        Single switch: only enable custom 4D attention mask when the model uses SDPA.
        """
        cfg = getattr(self.model, "config", None)
        attn_impl = getattr(cfg, "_attn_implementation", None)
        if attn_impl is None and cfg is not None:
            attn_impl = getattr(getattr(cfg, "text_config", None), "_attn_implementation", None)
        return attn_impl == "sdpa"

    def _build_img_next_full_attention_mask(
        self,
        attention_mask: torch.Tensor,  # [B, T] pad mask (1=valid)
        input_ids: torch.Tensor,       # [B, T]
    ) -> torch.Tensor:
        """
        Build a full 4D additive mask [B, 1, T, T] that is:
          - causal + padding for the whole sequence
          - plus bidirectional attention inside the *last contiguous 16* <img_next> tokens

        NOTE: This assumes <img_next> has no tokens after it (end of sequence segment).
        """
        if attention_mask.ndim != 2 or input_ids.ndim != 2:
            raise ValueError(
                f"Expected attention_mask/input_ids to be 2D [B,T], got {attention_mask.ndim}D/{input_ids.ndim}D"
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(f"attention_mask shape {tuple(attention_mask.shape)} != input_ids shape {tuple(input_ids.shape)}")

        B, T = attention_mask.shape
        device = attention_mask.device
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        min_val = torch.finfo(dtype).min

        valid = attention_mask.to(torch.bool)

        # Base causal + padding mask (bool allow matrix)
        base = torch.tril(torch.ones((T, T), device=device, dtype=torch.bool))
        allow = base.unsqueeze(0) & valid.unsqueeze(2) & valid.unsqueeze(1)  # [B, T, T]

        img_next_id = getattr(self, "img_next_token_id", None)
        if img_next_id is not None:
            img = (input_ids == img_next_id) & valid  # [B, T]
            has_img = img.any(dim=1)  # [B]

            idx = torch.arange(T, device=device)
            end = (img.to(torch.long) * idx).max(dim=1).values  # [B]
            block_len = 16
            start = end - (block_len - 1)  # [B]

            block = (idx.unsqueeze(0) >= start.unsqueeze(1)) & (idx.unsqueeze(0) <= end.unsqueeze(1))  # [B, T]
            block_ok = (
                has_img
                & (start >= 0)
                & (block.sum(dim=1) == block_len)
                & ((img & block).sum(dim=1) == block_len)
            )
            if block_ok.any():
                block = block & block_ok.unsqueeze(1)
                allow = allow | (block.unsqueeze(2) & block.unsqueeze(1))  # bidirectional inside block

        full = torch.full((B, 1, T, T), min_val, device=device, dtype=dtype)
        full.masked_fill_(allow.unsqueeze(1), 0)
        return full

    def forward_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        KV-Cache based iterative forward pass for implicit reasoning with thinking tokens.
        
        This method implements the ECoT implicit reasoning strategy:
        1. Find all <|thinking|> token positions in the batch
        2. Perform multiple short forward passes, stopping before each thinking token
        3. Use last-hidden state before each thinking token to update its embedding
        4. Leverage KV-Cache to avoid redundant computation
        5. Final full forward pass to get language model loss (if labels provided)
        
        Args:
            input_ids: [B, T] token IDs (aligned, with thinking tokens)
            attention_mask: [B, T] attention mask
            pixel_values: Vision input (from processor)
            image_grid_thw: Image grid dimensions (for Qwen3-VL)
            labels: [B, T] labels for language model loss (optional)
            **kwargs: Additional arguments for the model
        
        Returns:
            dict with keys:
                - 'loss': language model loss (if labels provided)
                - 'logits': final logits [B, T, vocab_size]
                - 'hidden_states': final hidden states [B, T, hidden_size]
                - 'latent_embeds': embeddings written into <|thinking|> tokens
                - 'latent_attention_mask': valid latent embedding mask
                - 'num_reasoning_passes': number of iterative passes performed
        """
        return_latent_embeds = bool(kwargs.pop("return_latent_embeds", False))
        B, T = input_ids.shape
        device = input_ids.device
        
        # Get thinking token ID
        thinking_token_id = getattr(self, "thinking_token_id", None)
        if thinking_token_id is None:
            logger.warning("thinking_token_id not found, falling back to normal forward")
            return self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                **kwargs
            )
        
        # Find all thinking token positions: (num_latent_tokens_in_batch, 2)
        latent_indices = (input_ids == thinking_token_id).nonzero(as_tuple=False)
        
        # Group by batch: latent_lists[i] = [pos1, pos2, ...] for sample i
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(B)
        ]
        
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0

        # Get text embedding layer
        embeddings = self.model.get_input_embeddings()
        inputs_embeds = embeddings(input_ids)  # [B, T, H]

        full_attention = None
        if (
            self._should_enable_img_next_full_attention()
            and attention_mask is not None
            and isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim == 2
        ):
            # Build once, slice per forward call.
            full_attention = self._build_img_next_full_attention_mask(attention_mask=attention_mask, input_ids=input_ids)

        def _log_visual_mismatch(stage: str, s_idx: int, e_idx: int, error: ValueError):
            """Log diagnostic info when image features / tokens mismatch."""
            def _shape(t):
                if t is None:
                    return None
                return tuple(t.shape) if hasattr(t, "shape") else str(type(t))
            try:
                sl = e_idx if e_idx is not None else input_ids.shape[1]
                img_cnt = (input_ids[:, :sl] == IMAGE_TOKEN_INDEX).sum(dim=1).tolist()
                think_cnt = (input_ids[:, :sl] == thinking_token_id).sum(dim=1).tolist() if thinking_token_id else None
                logger.error(
                    "[forward_latent:%s] %s | slice=(%s:%s) img_tokens=%s think_tokens=%s "
                    "input_ids=%s pixel_values=%s image_grid_thw=%s",
                    stage, error, s_idx, e_idx, img_cnt, think_cnt,
                    _shape(input_ids), _shape(pixel_values), _shape(image_grid_thw),
                )
            except Exception as diag_exc:
                logger.error("[forward_latent:%s] %s (diagnostics failed: %s)", stage, error, diag_exc)
            return error
        
        # If no thinking tokens, skip iterative reasoning
        if max_n_latents == 0:
            logger.info(
                "[forward_latent] No thinking tokens, normal forward | input_ids=%s attention_mask=%s",
                tuple(input_ids.shape),
                tuple(attention_mask.shape),
            )
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=full_attention if full_attention is not None else attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        labels=labels,
                        output_hidden_states=True,
                        **kwargs
                    )
            except ValueError as ve:
                if "Image features and image tokens do not match" in str(ve):
                    raise _log_visual_mismatch("single_pass", 0, input_ids.shape[1], ve)
                raise
            return {
                'loss': outputs.loss if labels is not None else None,
                'logits': outputs.logits,
                'hidden_states': outputs.hidden_states[-1],
                'latent_embeds': outputs.hidden_states[-1].new_zeros((B, 0, outputs.hidden_states[-1].shape[-1])),
                'latent_attention_mask': torch.zeros((B, 0), dtype=torch.bool, device=device),
                'num_reasoning_passes': 0,
            }
        
        # Initialize compute range: start from 0, end before earliest thinking token
        earliest_latent_pos = latent_indices[:, 1].min().item()
        next_compute_range = (0, earliest_latent_pos)
        kv_cache = None
        
        # Pre-allocate full hidden states tensor for efficient filling
        H = inputs_embeds.shape[-1]  # hidden_size
        full_hidden_states = torch.zeros(
            (B, T, H), 
            dtype=inputs_embeds.dtype,
            device=device
        )
        latent_embeds = torch.zeros(
            (B, max_n_latents, H),
            dtype=inputs_embeds.dtype,
            device=device,
        )
        latent_attention_mask = torch.zeros(
            (B, max_n_latents),
            dtype=torch.bool,
            device=device,
        )
        
        # Iterative reasoning passes
        for pass_idx in range(max_n_latents):
            s, e = next_compute_range
            
            if kv_cache is None:
                # First forward pass (no cache)
                # Note: s=0 in first pass, so inputs_embeds[:, s:e, :] is same as inputs_embeds[:, :e, :]
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outputs = self.model(
                            inputs_embeds=inputs_embeds[:, s:e, :],
                            attention_mask=(
                                full_attention[:, :, :e, :e] if full_attention is not None else attention_mask[:, s:e]
                            ),
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                            output_hidden_states=True,
                            use_cache=True,
                            **kwargs
                        )
                except ValueError as ve:
                    if "Image features and image tokens do not match" in str(ve):
                        raise _log_visual_mismatch("first_pass", s, e, ve)
                    raise
                hidden_states_offset = 0
            else:
                # Subsequent passes with KV-Cache
                # IMPORTANT: We updated the thinking token embedding at position s, so cache from 
                # position s onwards is invalid. However, we can use cache_position to tell the model
                # which positions to process, and the attention mask will automatically handle cache usage.
                # No need to manually slice or crop the cache!
                cache_position = torch.arange(s, e, device=inputs_embeds.device)
                
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outputs = self.model(
                            inputs_embeds=inputs_embeds[:, s:e, :],
                            attention_mask=(
                                full_attention[:, :, s:e, :e] if full_attention is not None else attention_mask[:, :e]
                            ),  # Full attention mask up to current position
                            pixel_values=None,  # Vision already processed in first pass
                            image_grid_thw=None,
                            past_key_values=kv_cache,  # Full cache (attention mask handles usage)
                            cache_position=cache_position,
                            output_hidden_states=True,
                            use_cache=True,
                            **kwargs
                        )
                except ValueError as ve:
                    if "Image features and image tokens do not match" in str(ve):
                        raise _log_visual_mismatch("iter_pass", s, e, ve)
                    raise
                hidden_states_offset = s
            
            # Extract last hidden states
            hidden_states = outputs.hidden_states[-1]  # [B, L_sub, H]
            kv_cache = outputs.past_key_values
            
            # Fill hidden states into full tensor at corresponding positions
            full_hidden_states[:, s:e, :] = hidden_states
            
            # Feedback: update thinking token embeddings with preceding hidden states
            # Since thinking tokens are consecutive, we update the thinking token at position (e) in this pass
            # The preceding position is (e - 1), which should be in the current hidden_states
            filling_indices = [
                (instance_idx, latent_lists[instance_idx][pass_idx])
                for instance_idx in range(B)
                if len(latent_lists[instance_idx]) > pass_idx
            ]
            
            # Avoid in-place operation: decompose -> replace -> reassemble
            # Convert inputs_embeds to list of lists of 1D tensors
            tensor_list = [
                [inputs_embeds[b, pos, :] for pos in range(T)]
                for b in range(B)
            ]
            
            # Replace thinking token embeddings with preceding hidden states
            # For consecutive thinking tokens: token at position (e) should use hidden state at position (e - 1)
            for batch_idx, token_idx in filling_indices:
                # Get hidden state at position (token_idx - 1) in the current output
                # Note: need to account for hidden_states_offset
                local_pos = token_idx - 1 - hidden_states_offset
                
                if 0 <= local_pos < hidden_states.shape[1]:
                    latent_embed = hidden_states[batch_idx, local_pos, :]
                    tensor_list[batch_idx][token_idx] = latent_embed
                    latent_embeds[batch_idx, pass_idx, :] = latent_embed
                    latent_attention_mask[batch_idx, pass_idx] = True
                else:
                    # This should not happen if thinking tokens are consecutive
                    logger.warning(
                        f"[forward_latent:pass{pass_idx}] Cannot update thinking token at position {token_idx} "
                        f"in sample {batch_idx}: local_pos={local_pos} out of range [0, {hidden_states.shape[1]}) "
                        f"(s={s}, e={e}, hidden_states_offset={hidden_states_offset}, token_idx={token_idx})"
                    )
            
            # Reassemble inputs_embeds
            inputs_embeds = torch.stack([
                torch.stack(tensor_list[b])
                for b in range(B)
            ])
            
            # Update compute range for next pass
            # Since thinking tokens are consecutive, each pass processes one more token
            if pass_idx + 1 >= max_n_latents:
                # Last pass: compute till end
                next_compute_range = (e, T)
            else:
                # Next pass: compute one more token (next thinking token)
                next_compute_range = (e, e + 1)
        
        # Final forward pass with updated embeddings (post-thinking tokens)
        s, e = next_compute_range
        cache_position = torch.arange(s, e, device=inputs_embeds.device) if kv_cache is not None else None
        
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    inputs_embeds=inputs_embeds[:, s:e, :],
                    attention_mask=(
                        (full_attention[:, :, s:e, :e] if full_attention is not None else attention_mask[:, :e])
                        if kv_cache
                        else (full_attention[:, :, s:e, s:e] if full_attention is not None else attention_mask)
                    ),
                    pixel_values=None if kv_cache else pixel_values,
                    image_grid_thw=None if kv_cache else image_grid_thw,
                    past_key_values=kv_cache,
                    cache_position=cache_position,
                    labels=labels[:, s:e] if labels is not None and kv_cache else labels,
                    output_hidden_states=True,
                    use_cache=False,
                    **kwargs
                )
        except ValueError as ve:
            if "Image features and image tokens do not match" in str(ve):
                raise _log_visual_mismatch("final_pass", s, e, ve)
            raise
        
        # Fill the final segment of hidden states
        full_hidden_states[:, s:e, :] = outputs.hidden_states[-1]
        
        # Return full hidden states [B, T, H] for action model's cross attention
        # Note: Gradient flows correctly through indexed assignment
        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits,
            'hidden_states': full_hidden_states,  # Complete sequence with all reasoning passes
            'latent_embeds': latent_embeds,
            'latent_attention_mask': latent_attention_mask,
            'num_reasoning_passes': max_n_latents + 1,
        }

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def load_state_dict(self, state_dict, strict: bool = True):

        if not getattr(self, "use_img_next_teacher", True):
            pruned = {k: v for k, v in state_dict.items() if not k.startswith("visual_ema")}
            dropped = len(state_dict) - len(pruned)
            if dropped > 0:
                logger.info(f"[load_state_dict] Dropped {dropped} visual_ema keys (use_img_next_teacher=False)")
            state_dict = pruned
        return super().load_state_dict(state_dict, strict=strict)

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        
        If implicit reasoning is enabled, this method will also align thinking tokens across
        batch samples for efficient latent reasoning training.
        """

        # Check if implicit reasoning is enabled
        enable_latent_reasoning = self.config.framework.get("enable_latent_reasoning", False)
        thinking_token_id = getattr(self, "thinking_token_id", None)
        action_tokens = kwargs.get("action_tokens", None)
        
        # If implicit reasoning is enabled, we need to align thinking tokens
        if enable_latent_reasoning and thinking_token_id is not None:
            # Note: solutions parameter is not used in alignment path
            # print('build_qwenvl_inputs_with_alignment')
            return self._build_qwenvl_inputs_with_alignment(
                images,
                instructions,
                solutions,
                thinking_token_id,
                action_tokens=action_tokens,
            )
        
        # Normal path: batch tokenization (no alignment needed)
        # Create messages: one message per sample
        messages = []
        prompt_messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"


        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": instruction})
            msg = [{"role": "user", "content": content}]
            prompt_messages.append([{"role": "user", "content": content}])

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference
        self.processor.tokenizer.padding_side = "right"
        batch_inputs = self.processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None: #  here only for fast_tokenizer now. 
            action_token_min = _ACTION_TOKEN_MIN # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX # here only for fast_tokenizer, see laravla/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs['input_ids'].clone()
            prompt_lengths = []
            for prompt_msg in prompt_messages:
                prompt_inputs = self.processor.apply_chat_template(
                    prompt_msg,
                    tokenize=True,
                    padding=False,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                prompt_lengths.append(int(prompt_inputs["attention_mask"].sum().item()))
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    prompt_len = min(prompt_lengths[i], seq.numel())
                    seq[:prompt_len] = IGNORE_INDEX
            
            labels[labels == self.processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
            # Mask img_next tokens out of VLM loss
            if getattr(self, "img_next_token_id", None) is not None:
                labels[labels == self.img_next_token_id] = IGNORE_INDEX
            batch_inputs['labels'] = labels

        return batch_inputs.to(self.model.device)

    def _build_qwenvl_inputs_with_alignment(
        self, 
        images, 
        instructions, 
        solutions, 
        thinking_token_id: int,
        action_tokens=None,
    ):
        """
        Build model inputs with thinking token alignment for implicit reasoning training.
        
        This method follows the same structure as normal path, but with thinking token alignment:
        1. Build messages (same as normal path)
        2. Batch process to get complete batch_inputs (same as normal path)
        3. Extract and align input_ids individually (for thinking token alignment)
        4. Replace batch_inputs with aligned input_ids
        5. Move to device (same as normal path)
        """
        pad_token_id = self.processor.tokenizer.pad_token_id
        model_max_length = (
            getattr(self.config.framework.qwenvl, "model_max_length", None) or 
            getattr(self.config, "model_max_length", 8192)
        )
        
        # Create messages: one message per sample (same as normal path)
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for sample_idx, (imgs, instruction) in enumerate(zip(images, instructions)):
            content = [{"type": "image", "image": img} for img in imgs]

            prompt = format_reasoning_prompt(instruction)

            if action_tokens is not None and isinstance(action_tokens, (list, tuple)) and sample_idx < len(action_tokens):
                act_text = action_tokens[sample_idx] or ""
                if act_text:
                    prompt = f"{prompt} Action: {act_text}"

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            messages.append(msg)

        # Batch process (same as normal path)
        self.processor.tokenizer.padding_side = "right"

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt"
        )

        # Convert to CPU and extract per-sample sequences
        batched_ids = batch_inputs["input_ids"]          # [B, T_pad]
        batched_mask = batch_inputs["attention_mask"]    # [B, T_pad]
        B, T_pad = batched_ids.shape
        # print(f"batched_ids: {batched_ids[0]}")
        ids_cpu = batched_ids.cpu()
        mask_cpu = batched_mask.cpu()
        
        input_ids_list = [ids_cpu[b] for b in range(B)]
        attention_mask_list = [mask_cpu[b] for b in range(B)]

        # Align thinking tokens (position_ids will be computed by Qwen3-VL internally)
        input_ids_list, attention_mask_list = self._align_thinking_tokens(
            input_ids_list, attention_mask_list, thinking_token_id, model_max_length, pad_token_id
        )
        
        # Re-batch the aligned inputs
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        
        # Truncate if necessary
        input_ids = input_ids[:, :model_max_length]
        attention_mask = attention_mask[:, :model_max_length]

        # Action tokens are appended into `prompt` before tokenization; no tensor-level suffixing here.

        # Replace with aligned input_ids and attention_mask (position_ids will be auto-computed by Qwen3-VL)
        batch_inputs["input_ids"] = input_ids
        batch_inputs["attention_mask"] = attention_mask

        img_next_id = getattr(self, "img_next_token_id", None)
        if img_next_id is not None:
            batch_inputs["img_next_mask"] = (input_ids == img_next_id).to(input_ids.dtype)
        # Remove position_ids - let Qwen3-VL compute it automatically based on attention_mask

        # Optional: attach masked labels (instruction+latent masked; post-latent unmasked)
        latent_cfg = self.config.framework.get("latent_reasoning", {}) if hasattr(self.config.framework, "get") else {}
        compute_language_loss = latent_cfg.get("compute_language_loss", False)
        if compute_language_loss and solutions is None:
            start_id = getattr(self, "start_thinking_id", None)
            end_id = getattr(self, "end_thinking_id", None)
            labels = self._build_ecot_labels_batch(
                input_ids=input_ids,
                pad_id=pad_token_id,
                start_id=start_id,
                think_id=thinking_token_id,
                end_id=end_id,
            )
            batch_inputs["labels"] = labels


        return batch_inputs.to(self.model.device)

    def _align_thinking_tokens(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        thinking_token_id: int,
        model_max_length: int,
        pad_token_id: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Align thinking token positions across batch samples for efficient latent reasoning training.
        
        This method implements the key insight: by padding sequences to align thinking token
        starting positions, we can maintain batch processing efficiency while handling different
        thinking token positions across samples.
        
        Args:
            input_ids_list: List of input_ids tensors, one per sample
            attention_mask_list: List of attention_mask tensors, one per sample
            thinking_token_id: ID of the thinking token to align
            model_max_length: Maximum sequence length
            pad_token_id: Padding token ID
            
        Returns:
            Tuple of (aligned_input_ids_list, aligned_attention_mask_list)
        """
        # Find the earliest thinking token position in each sample
        earliest_thinking_positions = []
        for ids in input_ids_list:
            thinking_mask = (ids == thinking_token_id)
            if thinking_mask.any():
                earliest_thinking_positions.append(thinking_mask.nonzero()[0].item())
            else:
                earliest_thinking_positions.append(-1)
        
        # Get valid positions (samples that have thinking tokens)
        valid_positions = [pos for pos in earliest_thinking_positions if pos >= 0]
        if not valid_positions:
            # No thinking tokens found: align @ delimiter for batch-level masking consistency.
            return self._align_at_delimiter(
                input_ids_list=input_ids_list,
                attention_mask_list=attention_mask_list,
                model_max_length=model_max_length,
                pad_token_id=pad_token_id,
            )
        
        # Align to the latest position (rightmost thinking token)
        latest_thinking_pos = max(valid_positions)
        
        # Check if alignment would exceed model_max_length
        max_original_length = max(len(ids) for ids in input_ids_list)
        if latest_thinking_pos + max_original_length > model_max_length:
            # Skip alignment if it would exceed max length
            logger.warning(
                f"Thinking token alignment would exceed model_max_length ({model_max_length}). "
                f"Using regular padding instead."
            )
            return input_ids_list, attention_mask_list
        
        # Apply pre-padding to align thinking tokens
        aligned_input_ids = []
        aligned_attention_mask = []
        
        for i, (input_ids, attention_mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            if earliest_thinking_positions[i] != -1:
                # Sample has thinking token - align it
                pad_count = latest_thinking_pos - earliest_thinking_positions[i]
            else:
                # Sample has no thinking token - add padding to align with others
                pad_count = latest_thinking_pos
                logger.warning(f"No thinking token found in sample {i}, padding to align with others")
            
            if pad_count > 0:
                # Add padding at the beginning (left padding)
                pad_tensor = torch.full(
                    (pad_count,), pad_token_id, dtype=input_ids.dtype, device=input_ids.device
                )
                mask_pad_tensor = torch.zeros(
                    (pad_count,), dtype=attention_mask.dtype, device=attention_mask.device
                )
                
                aligned_input_ids.append(torch.cat([pad_tensor, input_ids]))
                aligned_attention_mask.append(torch.cat([mask_pad_tensor, attention_mask]))
            else:
                aligned_input_ids.append(input_ids)
                aligned_attention_mask.append(attention_mask)
        
        return aligned_input_ids, aligned_attention_mask

    def _align_at_delimiter(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        model_max_length: int,
        pad_token_id: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Align the first occurrence of the "@ delimiter" token sequence across batch samples
        using left pre-padding (same strategy as `_align_thinking_tokens`).

        Assumption (per user): every sample contains the delimiter.
        """
        try:
            at_token_ids = self.processor.tokenizer.encode(" @", add_special_tokens=False)
        except Exception as e:
            logger.warning(f"[align_at] Failed to encode @ delimiter: {e}")
            return input_ids_list, attention_mask_list

        if not at_token_ids:
            logger.warning("[align_at] Encoded @ delimiter is empty; skip alignment")
            return input_ids_list, attention_mask_list

        # Build a tensor for subsequence matching once.
        at_len = len(at_token_ids)
        at_tensor = None
        at_positions: List[int] = []
        if at_len == 1:
            at_id = int(at_token_ids[0])
            for ids in input_ids_list:
                mask = (ids == at_id)
                if mask.any():
                    at_positions.append(int(mask.nonzero(as_tuple=False)[0].item()))
                else:
                    at_positions.append(-1)
        else:
            # Fallback: multi-token delimiter; scan for the first matching subsequence.
            for ids in input_ids_list:
                if at_tensor is None:
                    at_tensor = torch.tensor(at_token_ids, device=ids.device, dtype=ids.dtype)
                elif at_tensor.device != ids.device or at_tensor.dtype != ids.dtype:
                    at_tensor = torch.tensor(at_token_ids, device=ids.device, dtype=ids.dtype)
                pos = -1
                for i in range(int(ids.shape[0]) - at_len + 1):
                    if torch.equal(ids[i : i + at_len], at_tensor):
                        pos = i
                        break
                at_positions.append(pos)

        if any(p < 0 for p in at_positions):
            logger.warning("[align_at] Some samples do not contain @ delimiter; skip alignment")
            return input_ids_list, attention_mask_list

        latest_at_pos = max(at_positions)
        # Ensure alignment won't exceed model_max_length (conservative per-sample new length check).
        max_new_len = 0
        for ids, pos in zip(input_ids_list, at_positions):
            pad_count = latest_at_pos - pos
            max_new_len = max(max_new_len, int(ids.shape[0]) + int(pad_count))
        if max_new_len > int(model_max_length):
            logger.warning(
                "[align_at] @ alignment would exceed model_max_length (%s) (max_new_len=%s); skip alignment",
                model_max_length,
                max_new_len,
            )
            return input_ids_list, attention_mask_list

        aligned_input_ids: List[torch.Tensor] = []
        aligned_attention_mask: List[torch.Tensor] = []
        for ids, mask, pos in zip(input_ids_list, attention_mask_list, at_positions):
            pad_count = latest_at_pos - pos
            if pad_count > 0:
                pad_tensor = torch.full((pad_count,), pad_token_id, dtype=ids.dtype, device=ids.device)
                mask_pad_tensor = torch.zeros((pad_count,), dtype=mask.dtype, device=mask.device)
                aligned_input_ids.append(torch.cat([pad_tensor, ids]))
                aligned_attention_mask.append(torch.cat([mask_pad_tensor, mask]))
            else:
                aligned_input_ids.append(ids)
                aligned_attention_mask.append(mask)

        return aligned_input_ids, aligned_attention_mask

    # ==============================
    # ECoT span detection & labels (batch-optimized)
    # ==============================
    def _find_ecot_spans_aligned_batch(
        self,
        input_ids: torch.Tensor,           # [B, T]
        attention_mask: torch.Tensor,      # [B, T]
        start_id: Optional[int],
        end_id: Optional[int],
    ) -> Tuple[int, int, int]:
        """
        Find (instruction_end_idx, latent_start_idx, latent_end_idx) for the ALIGNED batch.
        
        Key insight: After thinking token alignment, these positions are IDENTICAL across all samples.
        We only need to find them once from any valid (non-padded) sample.
        
        Boundary detection priority:
        1. <|start_of_thinking|> token (for stage 2+ with thinking tokens)
        2. " @ " delimiter (fallback for stage 0 without thinking tokens)
        3. No masking (if neither found)
        
        Returns:
            Tuple of (instruction_end, latent_start, latent_end) - single set of indices for entire batch
        """
        B, T = input_ids.shape
        
        # Find a valid sample (with non-zero attention mask)
        valid_lengths = attention_mask.sum(dim=1)  # [B]
        valid_sample_idx = torch.argmax(valid_lengths).item()  # Sample with longest valid sequence
        
        ids = input_ids[valid_sample_idx]  # [T]
        
        # Helper to find first occurrence of token_id
        def pos_of(token_id: Optional[int]) -> Optional[int]:
            if token_id is None:
                return None
            mask = (ids == token_id)
            if mask.any():
                return int(mask.nonzero(as_tuple=False)[0].item())
            return None
        
        # Priority 1: Find start_thinking and end_thinking positions
        start_pos = pos_of(start_id)
        end_pos = pos_of(end_id)
        
        # If end_pos not found, try to find it in the full sequence (including padding)
        if start_pos is not None and end_pos is None:
            # Check if end_id is valid
            if end_id is not None:
                # Search in full sequence (not just valid part)
                full_mask = (ids == end_id)
                if full_mask.any():
                    end_pos = int(full_mask.nonzero(as_tuple=False)[0].item())
        
        # Determine instruction_end and latent span
        if start_pos is not None:
            # Stage 2+: Has thinking tokens
            instruction_end = start_pos
            latent_start = start_pos + 1  # Start masking AFTER <|start_of_thinking|> token
            if end_pos is not None and end_pos >= start_pos:
                latent_end = end_pos  # End masking BEFORE <|end_of_thinking|> token (exclude end token)
            else:
                # If no end_thinking found, mask till end of sequence
                # This is expected if all reasoning content is converted to thinking tokens
                latent_end = T
                logger.warning(
                    f"[_find_ecot_spans] end_thinking token not found, masking thinking tokens from position {latent_start} till end. "
                    f"<|start_of_thinking|> at position {start_pos} remains trainable."
                )
        else:
            # Priority 2: Stage 0 fallback - try @ delimiter
            # Encode " @ " (with spaces) to get token sequence
            try:
                at_token_ids = self.processor.tokenizer.encode(" @", add_special_tokens=False)
            except Exception as e:
                logger.warning(f"Failed to encode @ delimiter: {e}")
                at_token_ids = []
            
            at_pos = None
            if at_token_ids:
                # Find first occurrence of " @ " token sequence
                at_len = len(at_token_ids)
                at_tensor = torch.tensor(at_token_ids, device=ids.device, dtype=ids.dtype)
                
                for i in range(len(ids) - at_len + 1):
                    if torch.equal(ids[i:i+at_len], at_tensor):
                        at_pos = i
                        break
            
            if at_pos is not None:
                # Found @ delimiter - instruction ends after the @
                instruction_end = at_pos + len(at_token_ids)
                latent_start = -1  # No latent span in stage 0
                latent_end = -1
                logger.debug(f"Using @ delimiter at position {at_pos} for instruction boundary (likely stage 0)")
            else:
                # No delimiter found - don't mask instruction
                instruction_end = 0
                latent_start = -1
                latent_end = -1
                logger.warning("No thinking tokens or @ delimiter found, no instruction masking will be applied")
        
        return instruction_end, latent_start, latent_end

    def _build_ecot_labels_batch(
        self,
        input_ids: torch.Tensor,           # [B, T]
        pad_id: int,
        start_id: Optional[int],
        think_id: Optional[int],
        end_id: Optional[int],
    ) -> torch.Tensor:
        """
        Build masked labels with BATCH-LEVEL operations (leveraging alignment).
        
        Key optimization: Since thinking tokens are aligned, instruction and latent spans
        are at the SAME positions across all samples. We can use batch slicing instead of loops.
        
        Masking strategy (no @ delimiter needed):
        - pads -> IGNORE_INDEX
        - [0:instruction_end) -> IGNORE_INDEX (instruction part, up to <|start_of_thinking|>)
        - [latent_start:latent_end) -> IGNORE_INDEX (thinking tokens span)
        - [latent_end:] -> trainable (post-thinking generation)
        """
        labels = input_ids.clone()
        B, T = labels.shape
        
        # Step 1: Mask all padding tokens
        labels[labels == pad_id] = IGNORE_INDEX
        
        # Step 2: Find aligned span positions (same for all samples after alignment)
        attention_mask = (input_ids != pad_id).long()
        instr_end, lat_start, lat_end = self._find_ecot_spans_aligned_batch(
            input_ids, attention_mask, start_id, end_id
        )
        
        # Step 3: Batch-level masking (single slice operation for entire batch)
        if instr_end > 0:
            # Mask instruction span for ALL samples at once
            labels[:, :instr_end] = IGNORE_INDEX
        
        if lat_start >= 0 and lat_end > lat_start:
            # Mask latent thinking span for ALL samples at once
            labels[:, lat_start:lat_end] = IGNORE_INDEX
        
        # Post-latent tokens (lat_end:) remain trainable (already copied from input_ids)
        # Note: If all reasoning is converted to thinking tokens, post-thinking part may be short or empty
        # This is expected behavior for implicit reasoning training
        
        # Mask img_next tokens out of VLM loss
        img_next_id = getattr(self, "img_next_token_id", None)
        if img_next_id is not None:
            labels[labels == img_next_id] = IGNORE_INDEX

        return labels
