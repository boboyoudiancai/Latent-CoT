import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def validate_spatial_only_config(config) -> None:
    framework = config.get("framework", {})
    spatial = framework.get("spatial", {})
    stage = framework.get("training_stage", "full")
    mode = spatial.get("alignment_mode", "teacher_downsample")
    if stage != "spatial_only" and mode != "cross_attention":
        return
    bridge = config.get("datasets", {}).get("vla_data", {}).get("bridge_reasoning", {})
    required = {
        "training_stage": stage in {"spatial_only", "full"},
        "alignment_mode": mode == "cross_attention",
        "enable_spatial": bool(spatial.get("enable", False)),
        "use_teacher": bool(spatial.get("use_teacher", False)),
        "token_count": int(spatial.get("token_count", 0)) == 16,
        "view_mode": spatial.get("view_mode") == "base_only",
        "base_token": spatial.get("base_token", "<spatial_base>") == "<spatial_base>",
        "wrist_token": spatial.get("wrist_token", "<spatial_wrist>") == "<spatial_wrist>",
        "loss_type": spatial.get("loss_type", "cosine") == "cosine",
        "preprocess_mode": spatial.get("preprocess_mode", "resize") == "resize",
        "cot_mode": framework.get("cot_mode", "none") == "none",
        "enable_latent_reasoning": not framework.get("enable_latent_reasoning", False),
        "emit_thinking_tokens": not framework.get("emit_thinking_tokens", False),
        "compute_language_loss": not framework.get("latent_reasoning", {}).get("compute_language_loss", False),
        "vlm_loss_weight": float(framework.get("latent_reasoning", {}).get("vlm_loss_weight", 0.0)) == 0.0,
        "img_next": not framework.get("img_next", {}).get("enable", False),
        "latent_decoder": not framework.get("latent_decoder", {}).get("enable", False),
        "bridge_enable": bool(bridge.get("enable", False)),
        "bridge_spatial": bool(bridge.get("include_spatial", False)),
        "bridge_count": int(bridge.get("spatial_count", 0)) == 16,
        "bridge_view": bridge.get("spatial_view_mode") == "base_only",
        "query_mode": bridge.get("spatial_query_mode") == "visual_prefix_joint",
        "bridge_stage": int(bridge.get("stage", 0)) == 0,
        "action_tokens": not bridge.get("include_action_tokens", False),
        "bridge_img_next": not bridge.get("include_img_next", False),
        "freeze_modules": not config.get("trainer", {}).get("freeze_modules", ""),
        "target_cache_dir": not spatial.get("target_cache_dir"),
        "target_cache_only": not spatial.get("target_cache_only", False),
        "target_cache_size": int(spatial.get("target_cache_size", 0) or 0) == 0,
        "sdpa": framework.get("qwenvl", {}).get("attn_implementation", "sdpa") == "sdpa",
    }
    invalid = [name for name, valid in required.items() if not valid]
    if invalid:
        raise ValueError("cross-attention configuration mismatch: " + ", ".join(invalid))
    input_size = int(spatial.get("input_size", 518))
    if input_size <= 0 or input_size % 14:
        raise ValueError("VGGT input_size must be positive and divisible by 14")
    weight = float(spatial.get("loss_weight", 0.1))
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("spatial loss_weight must be finite and positive")


def select_base_visual_queries(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    spatial_merge_size: int,
    target_side: int,
    image_counts: Sequence[int],
) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
        raise ValueError("hidden must be [B,L,H] and match input_ids")
    if attention_mask.shape != input_ids.shape or attention_mask.ndim != 2:
        raise ValueError("visual selection requires the original [B,L] padding mask")
    if len(image_counts) != hidden.shape[0] or any(count != 2 for count in image_counts):
        raise ValueError("cross-attention Stage1 requires base+wrist images per sample")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape != (sum(image_counts), 3):
        raise ValueError("image_grid_thw must contain one grid per image, in batch/view order")
    if spatial_merge_size <= 0 or target_side <= 0:
        raise ValueError("spatial merge size and target side must be positive")
    grids = image_grid_thw.detach().cpu().tolist()
    visual_mask = input_ids.eq(image_token_id)
    valid_mask = attention_mask.bool()
    if torch.any(visual_mask & ~valid_mask):
        raise ValueError("image placeholders cannot be padding tokens")
    queries = []
    grid_offset = 0
    for batch_index, image_count in enumerate(image_counts):
        sample_grids = grids[grid_offset : grid_offset + image_count]
        merged_sizes = []
        for temporal, height, width in sample_grids:
            if temporal != 1 or height <= 0 or width <= 0:
                raise ValueError("only single-frame positive image grids are supported")
            if height % spatial_merge_size or width % spatial_merge_size:
                raise ValueError("image grid must be divisible by the spatial merge size")
            merged_sizes.append((height // spatial_merge_size, width // spatial_merge_size))
        image_hidden = hidden[batch_index, visual_mask[batch_index]]
        expected_count = sum(height * width for height, width in merged_sizes)
        if image_hidden.shape[0] != expected_count:
            raise ValueError(f"image token/grid count mismatch in sample {batch_index}")
        base_height, base_width = merged_sizes[0]
        base_hidden = image_hidden[: base_height * base_width]
        base_grid = base_hidden.reshape(base_height, base_width, hidden.shape[-1])
        base_grid = base_grid.permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(
            base_grid.float(), size=(target_side, target_side), mode="bilinear", align_corners=False
        )
        queries.append(resized.squeeze(0).permute(1, 2, 0).reshape(target_side**2, hidden.shape[-1]))
        grid_offset += image_count
    return torch.stack(queries).to(dtype=hidden.dtype)


def select_spatial_hidden(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    base_token_id: int,
    wrist_token_id: int,
    image_token_id: int,
    token_count: int = 16,
) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
        raise ValueError("spatial hidden must match [B,L] input_ids")
    if attention_mask.shape != input_ids.shape or attention_mask.ndim != 2:
        raise ValueError("spatial selection requires the original padding mask")
    valid = attention_mask.bool()
    base_mask = input_ids.eq(base_token_id)
    if torch.any(base_mask & ~valid) or not torch.all(base_mask.sum(dim=1) == token_count):
        raise ValueError(f"expected exactly {token_count} valid <spatial_base> tokens per sample")
    if torch.any(input_ids.eq(wrist_token_id) & valid):
        raise ValueError("base-only Stage1 must not contain <spatial_wrist> tokens")
    for batch_index in range(input_ids.shape[0]):
        positions = torch.nonzero(base_mask[batch_index], as_tuple=False).flatten()
        if int(positions[-1] - positions[0] + 1) != token_count:
            raise ValueError("spatial tokens must form one contiguous visual-prefix block")
        image_positions = torch.nonzero(input_ids[batch_index].eq(image_token_id) & valid[batch_index], as_tuple=False).flatten()
        if image_positions.numel() == 0 or positions[0] <= image_positions[-1]:
            raise ValueError("spatial prefix must follow the base+wrist image tokens")
    return hidden[base_mask].reshape(hidden.shape[0], token_count, hidden.shape[-1])


class SpatialCrossAttentionDecoder(nn.Module):
    def __init__(self, hidden_size: int, target_dim: int, attention_dim: int = 512, num_heads: int = 8):
        super().__init__()
        if min(hidden_size, target_dim, attention_dim, num_heads) <= 0 or attention_dim % num_heads:
            raise ValueError("positive dimensions and attention_dim divisible by num_heads are required")
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.query_norm = nn.LayerNorm(hidden_size)
        self.spatial_norm = nn.LayerNorm(hidden_size)
        self.query_projection = nn.Linear(hidden_size, attention_dim, bias=False)
        self.key_projection = nn.Linear(hidden_size, attention_dim, bias=False)
        self.value_projection = nn.Linear(hidden_size, attention_dim, bias=False)
        self.output_projection = nn.Linear(attention_dim, target_dim, bias=False)

    def _split_heads(self, features: torch.Tensor) -> torch.Tensor:
        return features.reshape(features.shape[0], features.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, visual_queries: torch.Tensor, spatial_hidden: torch.Tensor) -> torch.Tensor:
        if visual_queries.ndim != 3 or spatial_hidden.ndim != 3:
            raise ValueError("visual queries and spatial hidden must be [B,N,H]")
        if visual_queries.shape[0] != spatial_hidden.shape[0] or visual_queries.shape[-1] != spatial_hidden.shape[-1]:
            raise ValueError("visual queries and spatial hidden must share batch and hidden dimensions")
        if spatial_hidden.shape[1] != 16:
            raise ValueError("cross-attention requires the existing 16 spatial tokens")
        queries = self._split_heads(self.query_projection(self.query_norm(visual_queries)))
        normalized_spatial = self.spatial_norm(spatial_hidden)
        keys = self._split_heads(self.key_projection(normalized_spatial))
        values = self._split_heads(self.value_projection(normalized_spatial))
        decoded = F.scaled_dot_product_attention(queries, keys, values, dropout_p=0.0, is_causal=False)
        decoded = decoded.transpose(1, 2).reshape(visual_queries.shape[0], visual_queries.shape[1], -1)
        return self.output_projection(decoded)


def dense_spatial_cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError(f"dense spatial shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    return (1.0 - F.cosine_similarity(prediction.float(), target.detach().float(), dim=-1)).mean()
