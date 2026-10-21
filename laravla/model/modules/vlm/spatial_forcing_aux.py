from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class SpatialForcingAlignProjector(nn.Module):
    def __init__(self, input_dim: int, target_dim: int, use_vlm_norm: bool = True):
        super().__init__()
        self.vlm_norm = nn.LayerNorm(input_dim) if use_vlm_norm else None
        self.fc1 = nn.Linear(input_dim, target_dim, bias=True)
        self.fc2 = nn.Linear(target_dim, target_dim, bias=True)
        self.act_fn = nn.GELU()
        for module in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.vlm_norm is not None:
            hidden = self.vlm_norm(hidden)
        return self.fc2(self.act_fn(self.fc1(hidden)))


def select_base_image_hidden(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    spatial_merge_size: int,
    image_counts: Sequence[int],
) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
        raise ValueError("hidden must be [B,L,H] and match input_ids")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    if len(image_counts) != hidden.shape[0] or any(count < 1 for count in image_counts):
        raise ValueError("each sample must contain at least one image")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape != (sum(image_counts), 3):
        raise ValueError("image_grid_thw must contain one grid per image")
    visual_mask = input_ids.eq(int(image_token_id)) & attention_mask.bool()
    grids = image_grid_thw.detach().cpu().tolist()
    selected = []
    grid_offset = 0
    expected_base_tokens = None
    for batch_index, image_count in enumerate(image_counts):
        sample_grids = grids[grid_offset : grid_offset + image_count]
        merged_sizes = []
        for temporal, height, width in sample_grids:
            if temporal != 1 or height % spatial_merge_size or width % spatial_merge_size:
                raise ValueError("invalid Qwen image grid for Spatial Forcing alignment")
            merged_sizes.append((height // spatial_merge_size, width // spatial_merge_size))
        image_hidden = hidden[batch_index, visual_mask[batch_index]]
        expected_total = sum(height * width for height, width in merged_sizes)
        if image_hidden.shape[0] != expected_total:
            raise ValueError("Qwen image token count does not match image_grid_thw")
        base_tokens = merged_sizes[0][0] * merged_sizes[0][1]
        if expected_base_tokens is None:
            expected_base_tokens = base_tokens
        elif base_tokens != expected_base_tokens:
            raise ValueError("base-image token count must be constant across the batch")
        selected.append(image_hidden[:base_tokens])
        grid_offset += image_count
    return torch.stack(selected)


def spatial_forcing_cosine_loss(
    projector: SpatialForcingAlignProjector,
    vision_hidden: torch.Tensor,
    vggt_hidden: torch.Tensor,
) -> torch.Tensor:
    projected = projector(vision_hidden)
    if projected.shape != vggt_hidden.shape:
        raise ValueError(
            f"Spatial Forcing shape mismatch: prediction={tuple(projected.shape)} "
            f"target={tuple(vggt_hidden.shape)}"
        )
    cosine = F.cosine_similarity(projected.float(), vggt_hidden.detach().float(), dim=-1)
    return (1.0 - cosine).mean()
