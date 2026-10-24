from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class ImageNextPredictionMLP(nn.Module):
    def __init__(self, hidden_size: int, expansion_ratio: float = 2.0) -> None:
        super().__init__()
        expanded_size = max(hidden_size, int(round(hidden_size * expansion_ratio)))
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, expanded_size),
            nn.GELU(),
            nn.Linear(expanded_size, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class EMARMSNormalizer(nn.Module):
    def __init__(self, feature_dim: int, decay: float = 0.99, eps: float = 1.0e-6) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.decay = float(decay)
        self.eps = float(eps)
        self.register_buffer("ema_square", torch.zeros(feature_dim, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def _update(self, residual: torch.Tensor, valid_mask: torch.Tensor) -> None:
        valid = valid_mask.to(device=residual.device, dtype=torch.bool).view(-1)
        square_sum = residual.detach().float()[valid].square().sum(dim=(0, 1))
        count = torch.tensor(
            float(valid.sum().item() * residual.shape[1]),
            device=residual.device,
            dtype=torch.float32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(square_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        if count.item() <= 0.0:
            return
        batch_square = square_sum / count
        if not bool(self.initialized.item()):
            self.ema_square.copy_(batch_square)
            self.initialized.fill_(True)
        else:
            self.ema_square.mul_(self.decay).add_(batch_square, alpha=1.0 - self.decay)

    def forward(
        self,
        residual: torch.Tensor,
        valid_mask: torch.Tensor,
        update: Optional[bool] = None,
    ) -> torch.Tensor:
        if residual.ndim != 3:
            raise ValueError(f"residual must be [B, N, C], got {tuple(residual.shape)}")
        if residual.shape[-1] != self.ema_square.numel():
            raise ValueError(
                f"feature dimension mismatch: residual={residual.shape[-1]} "
                f"normalizer={self.ema_square.numel()}"
            )
        should_update = self.training if update is None else bool(update)
        if should_update:
            self._update(residual, valid_mask)
        if not bool(self.initialized.item()):
            return residual
        rms = torch.sqrt(self.ema_square.to(device=residual.device) + self.eps)
        return residual / rms.to(dtype=residual.dtype).view(1, 1, -1)


def masked_img_next_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"ImageNext shape mismatch: prediction={tuple(prediction.shape)} "
            f"target={tuple(target.shape)}"
        )
    valid = valid_mask.to(device=prediction.device, dtype=torch.bool).view(-1, 1, 1)
    if not torch.any(valid):
        return prediction.sum() * 0.0
    normalized_loss_type = str(loss_type).strip().lower()
    if normalized_loss_type in {"l2", "mse"}:
        elementwise = F.mse_loss(prediction.float(), target.float(), reduction="none")
    elif normalized_loss_type in {"l1", "mae"}:
        elementwise = F.l1_loss(prediction.float(), target.float(), reduction="none")
    else:
        raise ValueError(f"unsupported ImageNext loss_type={loss_type!r}")
    full_mask = valid.expand_as(elementwise)
    return (elementwise * full_mask).sum() / full_mask.sum()
