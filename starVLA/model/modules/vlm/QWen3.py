# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import copy
from typing import Optional

import torch
import torch.nn as nn
from starVLA.model.tools import has_flash_attn  # unified flash-attn detection (GPU / NPU)
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from .future_residual import EMARMSNormalizer, FuturePredictionMLP
from .thinking_config import latent_forward_mode

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
_ACTION_TOKEN_MIN = 151669  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    153716  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
)

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
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        self.latent_forward_mode = latent_forward_mode(config.framework.get("latent_reasoning", {}))

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        attn_implementation = "sdpa"
        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            if not has_flash_attn():
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, # resize image no longer needed? @TODO check bug
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        future_cfg = config.framework.get("future", {})
        self.enable_future = bool(future_cfg.get("enable", False))
        self.future_token = str(future_cfg.get("token", "<future>"))
        self.future_token_count = int(future_cfg.get("token_count", 16))
        self.future_token_id = None
        self.use_future_teacher = bool(future_cfg.get("use_teacher", True)) if self.enable_future else False
        self.future_teacher_update_mode = str(future_cfg.get("teacher_update_mode", "ema")).strip().lower()
        self.future_ema_momentum = float(future_cfg.get("ema_momentum", 0.999))
        self.future_teacher_input_size = int(future_cfg.get("teacher_input_size", 112))
        self.future_prediction_mlp = None
        self.future_residual_normalizer = None
        self.visual_ema = None

        if self.enable_future:
            if self.future_token_count != 16:
                raise ValueError(f"Future requires exactly 16 tokens, got {self.future_token_count}")
            if self.future_teacher_update_mode not in {"ema", "frozen"}:
                raise ValueError(
                    "framework.future.teacher_update_mode must be 'ema' or 'frozen', "
                    f"got {self.future_teacher_update_mode!r}"
                )
            if str(future_cfg.get("target_mode", "residual_ema_rms")).strip().lower() != "residual_ema_rms":
                raise ValueError("Future requires target_mode='residual_ema_rms'")
            if str(future_cfg.get("loss_type", "l2")).strip().lower() not in {"l2", "mse"}:
                raise ValueError("Future requires loss_type='l2' or 'mse'")
            self.future_token_id = self._add_future_token(self.processor.tokenizer)
            hidden_size = int(self.model.config.text_config.hidden_size)
            self.future_prediction_mlp = FuturePredictionMLP(
                hidden_size=hidden_size,
                expansion_ratio=float(future_cfg.get("prediction_mlp_ratio", 2.0)),
            )
            self.future_residual_normalizer = EMARMSNormalizer(
                feature_dim=hidden_size,
                decay=float(future_cfg.get("residual_ema_decay", 0.99)),
                eps=float(future_cfg.get("residual_ema_eps", 1.0e-6)),
            )
            if self.use_future_teacher:
                self._init_future_visual_ema()

        thinking_cfg = config.framework.get("latent_reasoning", {})
        self.enable_latent_reasoning = bool(thinking_cfg.get("enable", False))
        self.thinking_token = str(thinking_cfg.get("thinking_token", "<|thinking|>"))
        self.start_thinking_token = str(thinking_cfg.get("start_token", "<|start_of_thinking|>"))
        self.end_thinking_token = str(thinking_cfg.get("end_token", "<|end_of_thinking|>"))
        self.thinking_token_count = int(thinking_cfg.get("token_count", 3))
        self.thinking_token_id = None
        if self.enable_latent_reasoning:
            if self.thinking_token_count != 3:
                raise ValueError("Latent reasoning requires exactly three thinking tokens")
            self.thinking_token_id = self._add_thinking_tokens(self.processor.tokenizer)
            if self.latent_forward_mode == "independent_queries":
                self._init_independent_thinking_embeddings()
            logger.info("Latent forward mode: %s", self.latent_forward_mode)

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        if (
            self.enable_future
            and self._uses_sdpa()
            and isinstance(input_ids, torch.Tensor)
            and isinstance(attention_mask, torch.Tensor)
            and input_ids.ndim == 2
            and attention_mask.ndim == 2
        ):
            kwargs["attention_mask"] = self._build_future_attention_mask(attention_mask, input_ids)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

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

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            if self.enable_future:
                prompt = self._append_future_tokens(prompt)
            if self.enable_latent_reasoning:
                prompt += " " + self.start_thinking_token + self.thinking_token * self.thinking_token_count + self.end_thinking_token

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        if self.enable_future or self.enable_latent_reasoning:
            # All prompts have the same suffix; left padding aligns every thinking
            # position while keeping the Future block together for its full attention.
            self.processor.tokenizer.padding_side = "left" if self.enable_latent_reasoning else "right"
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=not (self.enable_future or self.enable_latent_reasoning),
            return_dict=True,
            return_tensors="pt",
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
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
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        "action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)

    def _add_thinking_tokens(self, tokenizer) -> int:
        tokens = [self.thinking_token, self.start_thinking_token, self.end_thinking_token]
        missing = [token for token in tokens if token not in tokenizer.get_vocab()]
        if missing:
            tokenizer.add_tokens(missing, special_tokens=True)
        if len(tokenizer) > self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(tokenizer))
        ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
        if any(token_id == tokenizer.unk_token_id for token_id in ids):
            raise ValueError("Failed to register thinking special tokens")
        self._initialize_thinking_token_rows(tokenizer)
        return ids[0]

    def _initialize_thinking_token_rows(self, tokenizer=None):
        """Initialize new LaRA special-token rows after an old checkpoint load."""
        tokenizer = tokenizer or self.processor.tokenizer
        ids = [tokenizer.convert_tokens_to_ids(token) for token in
               (self.thinking_token, self.start_thinking_token, self.end_thinking_token)]
        target_id = tokenizer.convert_tokens_to_ids("<<")
        if target_id == tokenizer.unk_token_id:
            target_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.bos_token_id
        with torch.no_grad():
            for token_id in ids:
                self.model.get_input_embeddings().weight[token_id].copy_(
                    self.model.get_input_embeddings().weight[target_id]
                )
                if getattr(self.model, "lm_head", None) is not None:
                    self.model.lm_head.weight[token_id].copy_(self.model.lm_head.weight[target_id])

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # A pre-thinking QwenPI checkpoint may be up to three vocabulary rows
        # smaller. Preserve its existing rows and initialize only the added
        # tokens; no other checkpoint shape mismatch is silently tolerated.
        if self.enable_latent_reasoning:
            destinations = {
                "model.model.language_model.embed_tokens.weight": self.model.get_input_embeddings().weight,
                "model.lm_head.weight": self.model.lm_head.weight,
            }
            for suffix, destination in destinations.items():
                key = prefix + suffix
                source = state_dict.get(key)
                if source is not None and source.shape != destination.shape:
                    extra = destination.shape[0] - source.shape[0]
                    if source.ndim == 2 and source.shape[1] == destination.shape[1] and 0 < extra <= 3:
                        expanded = destination.detach().to(device=source.device, dtype=source.dtype).clone()
                        expanded[:source.shape[0]].copy_(source)
                        state_dict[key] = expanded
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _init_independent_thinking_embeddings(self):
        """Three independent query rows, initialized like the legacy placeholder.

        Keep tokenizer IDs, prompt text, positions and the base-model RNG stream
        unchanged. Only the opt-in branch registers this checkpoint parameter.
        """
        row = self.model.get_input_embeddings().weight[self.thinking_token_id]
        self.thinking_embeddings = nn.Parameter(
            row.detach().unsqueeze(0).repeat(self.thinking_token_count, 1).clone()
        )

    def forward_latent(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None, **kwargs):
        """Latent reasoning with full per-layer states for QwenPI's action head.

        By default each input is the previous position's final hidden state; the
        prefix (including the bidirectional Future block when enabled) is evaluated
        only once, then a differentiable KV cache is reused for the suffix.
        Independent queries instead use one full pass and decode the three query
        positions' OUTPUT states, never the sample-independent input parameters.
        """
        if not self.enable_latent_reasoning:
            raise RuntimeError("latent reasoning is disabled")
        if not self._uses_sdpa():
            raise RuntimeError("Thinking requires SDPA attention")
        if kwargs.get("pixel_values_videos") is not None:
            raise ValueError("Video inputs are not supported in the latent path")
        positions = (input_ids == self.thinking_token_id).nonzero(as_tuple=False)
        counts = (input_ids == self.thinking_token_id).sum(dim=1)
        if not torch.all(counts == self.thinking_token_count):
            raise ValueError(f"Expected {self.thinking_token_count} thinking tokens per sample: {counts.tolist()}")
        positions = positions[:, 1].reshape(input_ids.shape[0], self.thinking_token_count)
        if not torch.equal(positions, positions[:1].expand_as(positions)):
            raise ValueError("Thinking positions must be aligned across the batch")
        first = int(positions[0, 0].item())
        if first == 0 or not torch.equal(positions[0], torch.arange(first, first + self.thinking_token_count, device=input_ids.device)):
            raise ValueError("Thinking tokens must be consecutive and have a prefix")
        if self.enable_future:
            future_positions = (input_ids == self.future_token_id).nonzero(as_tuple=False)[:, 1]
            if future_positions.numel() == 0 or int(future_positions.max().item()) >= first:
                raise ValueError("The entire Future block must precede thinking")
        full_attention = self._build_future_attention_mask(attention_mask, input_ids)
        # Passing only inputs_embeds makes Qwen infer text-only positions. Compute
        # image-aware MRoPE from the original IDs once and slice it per pass.
        position_ids, _ = self.model.model.get_rope_index(
            input_ids, image_grid_thw, None, attention_mask=attention_mask
        )
        embeddings = self.model.get_input_embeddings()(input_ids)
        if self.latent_forward_mode == "independent_queries":
            queries = self.thinking_embeddings.to(dtype=embeddings.dtype)
            embeddings = torch.cat([
                embeddings[:, :first],
                queries.unsqueeze(0).expand(input_ids.shape[0], -1, -1),
                embeddings[:, first + self.thinking_token_count:],
            ], dim=1)
            with torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16,
                                enabled=input_ids.device.type == "cuda"):
                outputs = self.model.model(
                    inputs_embeds=embeddings,
                    attention_mask=full_attention,
                    position_ids=position_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            last_hidden = outputs.hidden_states[-1]
            return {
                "hidden_states": outputs.hidden_states,
                "latent_embeds": last_hidden.gather(
                    1, positions.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1])
                ),
                "latent_attention_mask": torch.ones_like(positions, dtype=torch.bool),
            }
        layer_segments = None
        latents = []
        cache = None
        previous = None
        segments = [(0, first)] + [(first + i, first + i + 1) for i in range(self.thinking_token_count - 1)] + [(first + self.thinking_token_count - 1, input_ids.shape[1])]
        for segment_idx, (start, end) in enumerate(segments):
            segment_embeds = embeddings[:, start:end]
            if previous is not None:
                latents.append(previous[:, -1, :])
                segment_embeds = torch.cat([previous[:, -1:, :], segment_embeds[:, 1:, :]], dim=1)
            with torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16, enabled=input_ids.device.type == "cuda"):
                outputs = self.model.model(
                    inputs_embeds=segment_embeds,
                    attention_mask=full_attention[:, :, start:end, :end],
                    position_ids=position_ids[:, :, start:end],
                    pixel_values=pixel_values if segment_idx == 0 else None,
                    image_grid_thw=image_grid_thw if segment_idx == 0 else None,
                    past_key_values=cache,
                    cache_position=torch.arange(start, end, device=input_ids.device),
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True,
                )
            if outputs.past_key_values is None:
                raise RuntimeError("Latent forward requires a differentiable KV cache")
            cache = outputs.past_key_values
            previous = outputs.hidden_states[-1]
            if layer_segments is None:
                layer_segments = [[] for _ in outputs.hidden_states]
            for parts, layer in zip(layer_segments, outputs.hidden_states):
                parts.append(layer)
        return {
            "hidden_states": tuple(torch.cat(parts, dim=1) for parts in layer_segments),
            "latent_embeds": torch.stack(latents, dim=1),
            "latent_attention_mask": torch.ones_like(positions, dtype=torch.bool),
        }

    def train(self, mode: bool = True):
        result = super().train(mode)
        teacher_visual = getattr(self, "visual_ema", None)
        if teacher_visual is not None:
            teacher_visual.eval()
        return result

    def _append_future_tokens(self, instruction: str) -> str:
        instruction = str(instruction or "").strip().rstrip(".")
        future_suffix = " ".join([self.future_token] * self.future_token_count)
        return f"{instruction}. {future_suffix}"

    def _add_future_token(self, tokenizer) -> int:
        if self.future_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([self.future_token], special_tokens=True)
        old_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > old_vocab_size:
            self.model.resize_token_embeddings(len(tokenizer))
        future_token_id = tokenizer.convert_tokens_to_ids(self.future_token)
        if future_token_id == tokenizer.unk_token_id:
            raise ValueError(f"Failed to add Future token: {self.future_token}")

        embeddings = self.model.get_input_embeddings()
        target_token = "<<"
        if target_token not in tokenizer.get_vocab():
            target_token = tokenizer.pad_token if tokenizer.pad_token else tokenizer.bos_token
        target_id = tokenizer.convert_tokens_to_ids(target_token)
        if target_id == tokenizer.unk_token_id:
            target_id = next(
                token_id
                for token_id in range(len(tokenizer))
                if not (
                    tokenizer.convert_ids_to_tokens(token_id).startswith("<")
                    and tokenizer.convert_ids_to_tokens(token_id).endswith(">")
                )
            )
        with torch.no_grad():
            embeddings.weight[future_token_id].copy_(embeddings.weight[target_id])
            if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
                if not getattr(self.model, "tie_word_embeddings", False):
                    self.model.lm_head.weight[future_token_id].copy_(self.model.lm_head.weight[target_id])
        logger.info(f"Added Future token {self.future_token!r} with id={future_token_id}")
        return future_token_id

    def _get_student_visual(self) -> Optional[nn.Module]:
        base_model = getattr(self.model, "model", None)
        return getattr(base_model, "visual", None) if base_model is not None else None

    def _init_future_visual_ema(self) -> None:
        student_visual = self._get_student_visual()
        if student_visual is None:
            raise RuntimeError("Qwen3 visual encoder was not found for Future EMA teacher")
        self.visual_ema = copy.deepcopy(student_visual)
        self.visual_ema.requires_grad_(False)
        self.visual_ema.eval()

    @torch.no_grad()
    def update_future_ema(self, momentum: Optional[float] = None) -> None:
        if not self.enable_future or not self.use_future_teacher:
            return
        if self.future_teacher_update_mode == "frozen":
            return
        student_visual = self._get_student_visual()
        teacher_visual = self.visual_ema
        if student_visual is None or teacher_visual is None:
            return
        ema_momentum = self.future_ema_momentum if momentum is None else float(momentum)
        if not 0.0 <= ema_momentum <= 1.0:
            raise ValueError(f"EMA momentum must be in [0, 1], got {ema_momentum}")
        for teacher_param, student_param in zip(teacher_visual.parameters(), student_visual.parameters()):
            teacher_param.data.mul_(ema_momentum).add_(student_param.data, alpha=1.0 - ema_momentum)
        for teacher_buffer, student_buffer in zip(teacher_visual.buffers(), student_visual.buffers()):
            teacher_buffer.copy_(student_buffer)

    @torch.no_grad()
    def get_future_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ):
        student_visual = self._get_student_visual()
        teacher_visual = self.visual_ema if self.visual_ema is not None else student_visual
        if teacher_visual is None:
            raise RuntimeError("Qwen3 visual encoder was not found")
        pixel_values = pixel_values.type(teacher_visual.dtype)
        image_embeds, deepstack_image_embeds = teacher_visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // teacher_visual.spatial_merge_size**2).tolist()
        return torch.split(image_embeds, split_sizes), deepstack_image_embeds

    def project_future_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.future_prediction_mlp is None:
            return prediction
        return self.future_prediction_mlp(prediction)

    def normalize_future_residual(
        self,
        residual: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.future_residual_normalizer is None:
            return residual
        return self.future_residual_normalizer(residual, valid_mask, update=self.training)

    def _uses_sdpa(self) -> bool:
        model_config = getattr(self.model, "config", None)
        attention_implementation = getattr(model_config, "_attn_implementation", None)
        if attention_implementation is None and model_config is not None:
            attention_implementation = getattr(
                getattr(model_config, "text_config", None), "_attn_implementation", None
            )
        return attention_implementation == "sdpa"

    def _build_future_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask and input_ids shape mismatch")
        batch_size, sequence_length = attention_mask.shape
        device = attention_mask.device
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        valid = attention_mask.to(torch.bool)
        allow = torch.tril(torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool))
        allow = allow.unsqueeze(0) & valid.unsqueeze(2) & valid.unsqueeze(1)
        if self.enable_future:
            token_mask = (input_ids == int(self.future_token_id)) & valid
            counts = token_mask.sum(dim=1)
            if not torch.all(counts == self.future_token_count):
                raise ValueError(f"Expected {self.future_token_count} Future tokens per sample, got {counts.tolist()}")
            allow = allow | (token_mask.unsqueeze(2) & token_mask.unsqueeze(1))
        full_mask = torch.full(
            (batch_size, 1, sequence_length, sequence_length),
            torch.finfo(dtype).min,
            device=device,
            dtype=dtype,
        )
        full_mask.masked_fill_(allow.unsqueeze(1), 0)
        return full_mask


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
