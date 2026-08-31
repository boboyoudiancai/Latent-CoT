# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-OFT Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on multi-view images plus a language instruction (shares parameters with the VLM).
Inspired by OpenVLA-OFT
Key Points:
  - Qwen2.5 vision-language backbone
  - Injects an action special token into the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states


Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
  or /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md （adpat a little code)

"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from laravla.model.framework.base_framework import baseframework
from laravla.model.framework.latent_analysis_mixin import LatentAnalysisMixin
from laravla.model.modules.action_model.MLP_ActionHeader import get_action_model
from laravla.model.modules.vlm import get_vlm_model
from laravla.model.tools import FRAMEWORK_REGISTRY
from laravla.training.trainer_utils import initialize_overwatch
from laravla.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenOFT
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenOFTDefaultConfig:
    """QwenOFT framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier (must match @FRAMEWORK_REGISTRY.register) ---
    name: str = "QwenOFT"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
        }
    )

    # === Action head (MLP regression over action special tokens) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Action head architecture type
            "action_model_type": "MLP",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # Hidden dim for the action MLP (auto-set from VLM hidden_size at runtime)
            "action_hidden_dim": 2560,
            # How many future steps to predict
            "future_action_window_size": 8,
            # How many past steps included in action chunk (usually 0)
            "past_action_window_size": 0,
        }
    )


def _cfg_get(node, key, default=None):
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)


def _cfg_setdefault(node, key, value):
    if node is None:
        return
    current = _cfg_get(node, key, None)
    if current is None:
        setattr(node, key, value)


def state2str_transform(state: np.ndarray, num_bins: int = 256) -> str:
    """Quantize a state vector into uniform bins over [-1, 1]."""
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, num_bins + 1)[:-1]) - 1
    return " ".join(map(str, discretized_state))


def add_discretized_state_to_instruction(
    instructions: List[str],
    states: List[np.ndarray],
    num_bins: int = 256,
) -> List[str]:
    """Append discretized proprioceptive state tokens to each instruction."""
    updated_instructions = []
    for instr, state in zip(instructions, states):
        state_arr = np.asarray(state)
        state_vec = state_arr[0] if state_arr.ndim > 1 else state_arr
        state_str = state2str_transform(state_vec, num_bins=num_bins)
        updated_instructions.append(f"{instr} [STATE] {state_str} [ACTION]")
    return updated_instructions


@FRAMEWORK_REGISTRY.register("QwenOFT")
class Qwenvl_OFT(LatentAnalysisMixin, baseframework):
    """
    Multimodal vision-language-action model (OFT variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Action special token injected into the VLM sequence
      - MLP regression head over action token hidden states (L1 loss)

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
        self.config = self._ensure_oft_config(config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align action_hidden_dim to VLM hidden_size at runtime
        self.config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon

        self.action_token = "🔍"  # TODO also can add spacail token to Qwen, but too complex
        action_token_ids = self.qwen_vl_interface.processor.tokenizer(
            self.action_token,
            add_special_tokens=False,
        )["input_ids"]
        if len(action_token_ids) != 1:
            raise ValueError(
                f"OFT action token {self.action_token!r} must tokenize to one id, got {action_token_ids}"
            )
        self.action_token_id = int(action_token_ids[0])

        # L1 loss
        self.l1_loss = nn.L1Loss()
        self.training_stage = self.config.framework.get("training_stage", "full")

        if self.training_stage == "reasoning_only":
            print("[Training Stage] reasoning_only mode - Freezing action_model parameters")
            for param in self.action_model.parameters():
                param.requires_grad = False
        elif self.training_stage == "action_only":
            print("[Training Stage] action_only mode - Freezing VLM parameters")
            for param in self.qwen_vl_interface.parameters():
                param.requires_grad = False
        else:
            print("[Training Stage] full mode - All parameters trainable")

    def _ensure_oft_config(self, config):
        if config is None:
            raise ValueError("QwenOFT requires a LaRA config")

        framework_cfg = config.framework
        action_cfg = framework_cfg.action_model
        framework_cfg.name = "QwenOFT"
        action_cfg.action_model_type = "MLP"
        _cfg_setdefault(action_cfg, "action_dim", 7)
        _cfg_setdefault(action_cfg, "past_action_window_size", 0)
        if _cfg_get(action_cfg, "action_horizon", None) is None:
            future = int(_cfg_get(action_cfg, "future_action_window_size", 7))
            past = int(_cfg_get(action_cfg, "past_action_window_size", 0))
            action_cfg.action_horizon = past + 1 + future
        if _cfg_get(action_cfg, "future_action_window_size", None) is None:
            action_cfg.future_action_window_size = int(action_cfg.action_horizon) - 1
        _cfg_setdefault(action_cfg, "action_hidden_dim", _cfg_get(action_cfg, "hidden_size", 1024))
        return config

    def _vlm_loss_weight(self) -> float:
        latent_cfg = self.config.framework.get("latent_reasoning", {})
        return float(latent_cfg.get("vlm_loss_weight", 0.0))

    def _action_prompt_suffix(self) -> str:
        action_tokens = self.action_token * self.chunk_len
        return f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."

    def _append_action_prompt(self, instructions: List[str]) -> List[str]:
        suffix = self._action_prompt_suffix()
        img_cfg = self.config.framework.get("img_next", {})
        img_next_token = img_cfg.get("token", "<img_next>")
        out = []
        for instruction in instructions:
            text = (instruction or "").strip()
            tail = ""
            if img_cfg.get("enable", False) and img_next_token:
                stripped = text.rstrip()
                while stripped.endswith(img_next_token):
                    tail = img_next_token + tail
                    stripped = stripped[: -len(img_next_token)].rstrip()
                if tail:
                    out.append(f"{stripped}{suffix} {tail}".strip())
                    continue
            out.append(f"{text}{suffix}".strip())
        return out

    @staticmethod
    def _normalize_generated_cot(generated_cot: List[str]) -> List[str]:
        normalized = []
        for cot in generated_cot:
            cot = (cot or "").strip()
            if "@" in cot:
                cot = "@ " + cot.split("@", 1)[1].strip()
            normalized.append(cot)
        return normalized

    @classmethod
    def _condition_on_generated_cot(cls, instructions: List[str], generated_cot: List[str]) -> List[str]:
        if len(instructions) != len(generated_cot):
            raise ValueError("Generated CoT batch size must match the instruction batch size")

        conditioned = []
        for instruction, cot in zip(instructions, cls._normalize_generated_cot(generated_cot)):
            instruction = (instruction or "").strip()
            if instruction and cot and instruction[-1] not in ".!?":
                instruction += "."
            conditioned.append(f"{instruction} {cot}".strip())
        return conditioned

    def _build_explicit_action_inputs(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        generated_cot: List[str],
        state=None,
    ):
        if not (len(batch_images) == len(instructions) == len(generated_cot)):
            raise ValueError("Images, instructions, and generated CoT must have the same batch size")

        cot_texts = self._normalize_generated_cot(generated_cot)
        action_prompts = [""] * len(instructions)
        if state is not None:
            action_prompts = self.add_discretized_state_to_instruction(action_prompts, state)
        action_prompts = self._append_action_prompt(action_prompts)

        messages = []
        for imgs, instruction, cot, action_prompt in zip(
            batch_images, instructions, cot_texts, action_prompts
        ):
            user_content = [{"type": "image", "image": img} for img in imgs]
            user_content.append({"type": "text", "text": (instruction or "").strip()})
            messages.append(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": cot}]},
                    {"role": "user", "content": [{"type": "text", "text": action_prompt}]},
                ]
            )

        processor = self.qwen_vl_interface.processor
        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "right"
        try:
            qwen_inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            processor.tokenizer.padding_side = old_padding_side
        return qwen_inputs.to(self.qwen_vl_interface.model.device)

    def _mask_action_prompt_labels(self, qwen_inputs) -> None:
        labels = qwen_inputs.get("labels", None)
        input_ids = qwen_inputs.get("input_ids", None)
        if labels is None or input_ids is None:
            return

        suffix_ids = self.qwen_vl_interface.processor.tokenizer(
            self._action_prompt_suffix(),
            add_special_tokens=False,
        )["input_ids"]
        suffix_ids_tensor = torch.tensor(suffix_ids, dtype=input_ids.dtype, device=input_ids.device)
        suffix_len = int(suffix_ids_tensor.numel())
        for row_idx in range(input_ids.shape[0]):
            row = input_ids[row_idx]
            start = None
            if suffix_len > 0 and row.numel() >= suffix_len:
                for pos in range(0, int(row.numel()) - suffix_len + 1):
                    if torch.equal(row[pos : pos + suffix_len], suffix_ids_tensor):
                        start = pos
                        break
            if start is None:
                matches = (row == self.action_token_id).nonzero(as_tuple=False)
                if matches.numel() > 0:
                    start = int(matches[0].item())
            if start is not None:
                labels[row_idx, start:] = IGNORE_INDEX

    def _run_qwen(
        self,
        batch_images,
        instructions,
        compute_vlm_loss: bool,
        use_iterative_forward: bool = True,
        qwen_inputs=None,
    ):
        if qwen_inputs is None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
            )
        if compute_vlm_loss:
            self._mask_action_prompt_labels(qwen_inputs)
        else:
            qwen_inputs.pop("labels", None)

        enable_latent_reasoning = self.config.framework.get("enable_latent_reasoning", False)
        run_latent = bool(use_iterative_forward) and enable_latent_reasoning and hasattr(
            self.qwen_vl_interface,
            "forward_latent",
        )

        if run_latent:
            vlm_outputs = self.qwen_vl_interface.forward_latent(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs.get("pixel_values"),
                image_grid_thw=qwen_inputs.get("image_grid_thw"),
                labels=qwen_inputs.get("labels"),
                position_ids=qwen_inputs.get("position_ids"),
            )
            return vlm_outputs["hidden_states"], vlm_outputs.get("loss"), qwen_inputs

        model_inputs = dict(qwen_inputs)
        model_inputs.pop("img_next_mask", None)
        model_inputs["use_cache"] = False
        if not compute_vlm_loss:
            backbone = getattr(self.qwen_vl_interface.model, "model", None)
            if backbone is None:
                raise RuntimeError("QwenOFT requires a Qwen backbone at qwen_vl_interface.model.model")
            model_inputs.pop("labels", None)
            model_inputs.pop("logits_to_keep", None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                backbone_outputs = backbone(
                    **model_inputs,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            return backbone_outputs.last_hidden_state, None, qwen_inputs

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **model_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        vlm_loss = qwenvl_outputs.loss if hasattr(qwenvl_outputs, "loss") else None
        return qwenvl_outputs.hidden_states[-1], vlm_loss, qwen_inputs

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: directly regress future actions (no diffusion).

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        cot_mode = str(self.config.framework.get("cot_mode", "none")).lower()
        explicit_action_inputs = None
        if self.training_stage != "reasoning_only" and cot_mode == "explicit":
            from laravla.model.framework.laravla import Qwen_GR00T
            generated_cot, _ = Qwen_GR00T._generate_explicit_cot(self, batch_images, instructions, **kwargs)
            explicit_action_inputs = self._build_explicit_action_inputs(
                batch_images=batch_images,
                instructions=instructions,
                generated_cot=generated_cot,
                state=state,
            )
        else:
            # Optionally append discretised proprioceptive state tokens to each instruction (π₀.5 style).
            instructions = (
                self.add_discretized_state_to_instruction(instructions, state)
                if state is not None
                else instructions
            )
            if self.training_stage != "reasoning_only":
                instructions = self._append_action_prompt(instructions)

        # Step 1: QWenVL input format
        compute_vlm_loss = self.training_stage == "reasoning_only" or (
            self.training_stage == "full" and self._vlm_loss_weight() > 0
        )
        last_hidden, vlm_loss, qwen_inputs = self._run_qwen(
            batch_images=batch_images,
            instructions=instructions,
            compute_vlm_loss=compute_vlm_loss,
            use_iterative_forward=True,
            qwen_inputs=explicit_action_inputs,
        )

        result = {}
        if self.training_stage == "reasoning_only":
            if vlm_loss is None:
                raise ValueError("training_stage='reasoning_only' requires VLM loss, but vlm_loss is None.")
            result["vlm_loss"] = vlm_loss
            result["total_loss"] = vlm_loss
            return result

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

            # Label alignment: take the last chunk_len segment
            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            # Compute L1 loss
            action_loss = self.l1_loss(pred_actions, actions_target)

        result["action_loss"] = action_loss
        if self.training_stage == "action_only":
            result["total_loss"] = action_loss
            return result

        result["total_loss"] = action_loss
        if vlm_loss is not None:
            vlm_loss_weight = self._vlm_loss_weight()
            result["vlm_loss"] = vlm_loss
            result["total_loss"] = result["total_loss"] + vlm_loss_weight * vlm_loss
        return result

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: Optional[List[List[Image.Image]]] = None,
        instructions: Optional[List[str]] = None,
        state: Optional[np.ndarray] = None,
        examples: Optional[List[dict]] = None,
        **kwargs,
    ) -> np.ndarray:
        """

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if examples is not None and type(examples) is not list:
            examples = [examples]
        if examples is not None:
            batch_images = [to_pil_preserve(example["image"]) for example in examples]
            instructions = [example["lang"] for example in examples]
            state = [example["state"] for example in examples] if "state" in examples[0] else state
        if batch_images is None or instructions is None:
            raise ValueError("predict_action requires either examples or batch_images+instructions")

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size is None:
            train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        cot_mode = str(kwargs.get("cot_mode", self.config.framework.get("cot_mode", "none"))).lower()
        explicit_action_inputs = None
        if cot_mode == "explicit":
            from laravla.model.framework.laravla import Qwen_GR00T
            generated_cot, _ = Qwen_GR00T._generate_explicit_cot(self, batch_images, instructions, **kwargs)
            explicit_action_inputs = self._build_explicit_action_inputs(
                batch_images=batch_images,
                instructions=instructions,
                generated_cot=generated_cot,
                state=state,
            )
        else:
            instructions = (
                self.add_discretized_state_to_instruction(instructions, state)
                if state is not None
                else instructions
            )
            instructions = self._append_action_prompt(instructions)

        # Step 1: QWenVL input format
        use_iterative_forward = bool(kwargs.get("use_iterative_forward", False))
        last_hidden, _vlm_loss, qwen_inputs = self._run_qwen(
            batch_images=batch_images,
            instructions=instructions,
            compute_vlm_loss=False,
            use_iterative_forward=use_iterative_forward,
            qwen_inputs=explicit_action_inputs,
        )

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,  # [B, L, H]
        input_ids: torch.Tensor,  # [B, L]
        action_token_id=None,  # Can be int or List[int]
    ) -> torch.Tensor:
        """
        Vectorized batch extraction of action token embeddings:
          - No per-sample for loop
          - Select the last chunk_len action placeholder tokens from each sample
        Args:
            last_hidden: [B, L, H]
            input_ids:   [B, L]
            action_token_id: int or List[int]
        Returns:
            action_queries: [B, chunk_len, H]
        """
        if action_token_id is None:
            raise ValueError("action_token_id must not be None")

        device = input_ids.device
        B, L, H = last_hidden.shape

        # Support multiple ids (e.g., multiple variants)
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            # torch.isin requires PyTorch >=1.10
            mask = torch.isin(input_ids, id_list)
        else:
            mask = input_ids == action_token_id  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"The following samples have insufficient action tokens (< {self.chunk_len}): {insufficient} |"
                f" counts={counts.tolist()}"
            )

        # Position indices
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))  # Set non-action positions to -1

        # Take the last chunk_len positions (higher indices = later in sequence)
        # Note: count sufficiency already verified, so -1 won't be incorrectly selected
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values  # [B, chunk_len] unsorted
        # Sort in temporal order
        selected_pos = topk_pos.sort(dim=-1).values  # [B, chunk_len]

        # Gather
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)  # [B, chunk_len, H]
        action_queries = last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk_len, H]
        return action_queries

    # Discretised state → instruction prefix (π₀.5 style); shared with QwenPI_v3.
    add_discretized_state_to_instruction = staticmethod(add_discretized_state_to_instruction)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwenvl_OFT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"[train] Action Loss (with state): {action_loss.item()}")

    predict_output = model.predict_action(examples=[batch[0]])
    normalized_actions = predict_output["normalized_actions"]
    print(f"[infer] Predicted Action shape: {normalized_actions.shape}")

    # Backward-compat: examples without `state` should still work.
    sample_no_state = {k: v for k, v in sample.items() if k != "state"}
    forward_no_state = model([sample_no_state, sample_no_state])
    print(f"[train] Action Loss (no state): {forward_no_state['action_loss'].item()}")
    predict_no_state = model.predict_action(examples=[sample_no_state])
    print(f"[infer] Predicted Action shape (no state): {predict_no_state['normalized_actions'].shape}")

    print("Finished")
