from __future__ import annotations

from typing import Optional

import torch


def compose_total_loss(
    training_stage: str,
    action_loss: Optional[torch.Tensor],
    *,
    vlm_loss: Optional[torch.Tensor] = None,
    vlm_loss_weight: float = 0.0,
    img_next_loss: Optional[torch.Tensor] = None,
    img_next_loss_weight: float = 0.0,
    spatial_loss: Optional[torch.Tensor] = None,
    spatial_loss_weight: float = 0.0,
    decoder_loss: Optional[torch.Tensor] = None,
    decoder_loss_weight: float = 0.0,
) -> torch.Tensor:
    """Compose action, language, and auxiliary losses by explicit stage."""
    stage = str(training_stage).strip().lower()
    if stage == "action_only":
        if action_loss is None:
            raise ValueError("action_only requires action_loss")
        return action_loss
    if stage == "reasoning_only":
        if vlm_loss is None:
            raise ValueError("reasoning_only requires vlm_loss")
        total = vlm_loss
    else:
        if action_loss is None:
            raise ValueError(f"{stage} requires action_loss")
        total = action_loss
        if vlm_loss is not None and float(vlm_loss_weight) != 0.0:
            total = total + float(vlm_loss_weight) * vlm_loss
    if img_next_loss is not None and float(img_next_loss_weight) != 0.0:
        total = total + float(img_next_loss_weight) * img_next_loss
    if spatial_loss is not None and float(spatial_loss_weight) != 0.0:
        total = total + float(spatial_loss_weight) * spatial_loss
    if decoder_loss is not None and float(decoder_loss_weight) != 0.0:
        total = total + float(decoder_loss_weight) * decoder_loss
    return total
