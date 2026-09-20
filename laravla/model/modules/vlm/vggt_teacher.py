"""VGGT feature adapter used by the Spatial-Forcing style auxiliary loss."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def _resize_vggt_tensor(image: torch.Tensor, target_size: int, mode: str = "crop") -> torch.Tensor:
    """Match Spatial Forcing's crop/pad preprocessing on a [3,H,W] image tensor."""
    if image.ndim != 3:
        raise ValueError(f"expected image tensor [3,H,W], got {tuple(image.shape)}")
    _, height, width = image.shape
    target_size = int(target_size)
    if mode not in {"resize", "crop", "pad"}:
        raise ValueError(f"unsupported VGGT preprocess mode: {mode}")
    if mode == "resize":
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
        )
        return resized.squeeze(0).contiguous()
    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = max(14, round(height * (new_width / width) / 14) * 14)
        else:
            new_height = target_size
            new_width = max(14, round(width * (new_height / height) / 14) * 14)
    else:
        new_width = target_size
        new_height = max(14, round(height * (new_width / width) / 14) * 14)
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
    )
    if mode == "crop":
        if resized.shape[-2] < target_size:
            pad_h = target_size - resized.shape[-2]
            resized = F.pad(
                resized,
                (0, 0, pad_h // 2, pad_h - pad_h // 2),
                mode="constant",
                value=1.0,
            )
        start_y = max(0, (resized.shape[-2] - target_size) // 2)
        resized = resized[:, :, start_y : start_y + target_size, :]
    else:
        pad_h = max(0, target_size - resized.shape[-2])
        pad_w = max(0, target_size - resized.shape[-1])
        resized = F.pad(
            resized,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            mode="constant",
            value=1.0,
        )
    return resized[:, :, :target_size, :target_size].squeeze(0).contiguous()


def build_vggt_transform(input_size: int, mode: str = "crop"):
    def transform(image: Image.Image) -> torch.Tensor:
        return _resize_vggt_tensor(
            transforms.ToTensor()(image.convert("RGB")),
            target_size=input_size,
            mode=mode,
        )
    return transform


def _sinusoidal_2d(height: int, width: int, dim: int, device, dtype) -> torch.Tensor:
    if dim < 4:
        return torch.zeros((height * width, dim), device=device, dtype=dtype)
    quarter = dim // 4
    coords_y = torch.arange(height, device=device, dtype=torch.float32)
    coords_x = torch.arange(width, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (torch.arange(quarter, device=device, dtype=torch.float32) / max(1, quarter)))
    y = coords_y[:, None] * omega[None, :]
    x = coords_x[:, None] * omega[None, :]
    yy = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
    xx = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    pe = torch.cat(
        [
            yy[:, None, :].expand(height, width, -1),
            xx[None, :, :].expand(height, width, -1),
        ],
        dim=-1,
    ).reshape(height * width, -1)
    if pe.shape[-1] < dim:
        pe = F.pad(pe, (0, dim - pe.shape[-1]))
    return pe[:, :dim].to(dtype=dtype)


def _add_vggt_positional_encoding(grid: torch.Tensor) -> torch.Tensor:
    """Apply SF's small (0.1-scaled) 2-D positional signal before pooling."""
    _, _, height, width = grid.shape
    channels = grid.shape[1]
    try:
        from vggt.heads.utils import create_uv_grid, position_grid_to_embed
        pos = create_uv_grid(width, height, aspect_ratio=width / height, dtype=grid.dtype, device=grid.device)
        pos = position_grid_to_embed(pos, channels).permute(2, 0, 1).unsqueeze(0)
    except Exception:
        pos = _sinusoidal_2d(height, width, channels, grid.device, grid.dtype).T.reshape(
            1, channels, height, width
        )
    return grid + 0.1 * pos.to(dtype=grid.dtype)


def pool_vggt_tokens(tokens: torch.Tensor, target_tokens: int, add_positional_encoding: bool = False) -> torch.Tensor:
    """SF-style bilinear pooling of [B, views, patches, C] to [B, views, N, C]."""
    if tokens.ndim != 4:
        raise ValueError(f"expected VGGT features [B,V,N,C], got {tuple(tokens.shape)}")
    source_tokens = int(tokens.shape[2])
    source_side = int(math.sqrt(source_tokens))
    target_side = int(math.sqrt(target_tokens))
    if source_side * source_side != source_tokens:
        raise ValueError(f"VGGT patch count is not square after special-token removal: {source_tokens}")
    if target_side * target_side != int(target_tokens):
        raise ValueError(f"spatial token count must be square, got {target_tokens}")
    bsz, views, _, channels = tokens.shape
    grid = tokens.reshape(bsz * views, source_side, source_side, channels).permute(0, 3, 1, 2)
    if add_positional_encoding:
        grid = _add_vggt_positional_encoding(grid)
    pooled = F.interpolate(
        grid.float(),
        size=(target_side, target_side),
        mode="bilinear",
        align_corners=True,
    )
    return pooled.to(dtype=tokens.dtype).permute(0, 2, 3, 1).reshape(
        bsz, views, target_tokens, channels
    ).contiguous()


class AlignProjector(nn.Module):
    """Spatial Forcing AlignProjector: LLM dim -> 2*target dim -> target dim."""

    def __init__(self, input_dim: int, output_dim: int, use_vlm_norm: bool = True):
        super().__init__()
        self.vlm_norm = nn.LayerNorm(input_dim) if use_vlm_norm else None
        self.fc1 = nn.Linear(input_dim, 2 * output_dim, bias=True)
        self.fc2 = nn.Linear(2 * output_dim, output_dim, bias=True)
        self.act_fn = nn.GELU()
        self.initialize_weights()

    def initialize_weights(self):
        for module in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.vlm_norm is not None:
            hidden = self.vlm_norm(hidden)
        return self.fc2(self.act_fn(self.fc1(hidden)))


class VGGTTeacher:
    """Frozen VGGT feature extractor with upstream/fork compatibility."""

    def __init__(
        self,
        teacher_repo: str,
        input_size: int = 518,
        layer: int = -1,
        patch_start_idx: int = 5,
        use_vggt_pe: bool = True,
        preprocess_mode: str = "resize",
    ) -> None:
        if not teacher_repo:
            raise ValueError("framework.spatial.teacher_repo must point to a local VGGT repository")
        self.teacher_repo = str(teacher_repo)
        self.input_size = int(input_size)
        self.layer = int(layer)
        self.patch_start_idx = int(patch_start_idx)
        self.use_vggt_pe = bool(use_vggt_pe)
        self.preprocess_mode = str(preprocess_mode)
        try:
            from vggt.models.vggt import VGGT
        except ImportError as exc:
            raise ImportError(
                "VGGT spatial alignment requires the official vggt package."
            ) from exc

        repo = Path(self.teacher_repo).expanduser()
        local_weights = repo / "model.pt" if repo.is_dir() else None
        if local_weights is not None and local_weights.is_file():
            # Feature-only construction follows Spatial Forcing and avoids allocating
            # VGGT prediction heads that are irrelevant to alignment.
            self.encoder = VGGT(
                enable_camera=False,
                enable_point=False,
                enable_depth=False,
                enable_track=False,
            )
            self.encoder.load_state_dict(
                torch.load(local_weights, map_location="cpu"),
                strict=False,
            )
        elif hasattr(VGGT, "from_pretrained"):
            self.encoder = VGGT.from_pretrained(self.teacher_repo)
        else:
            raise FileNotFoundError(
                f"VGGT local model.pt not found and from_pretrained is unavailable: {self.teacher_repo}"
            )
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.transform = build_vggt_transform(self.input_size, self.preprocess_mode)
        self.embed_dim = self._infer_embed_dim()

    def _infer_embed_dim(self) -> int:
        aggregator = getattr(self.encoder, "aggregator", None)
        base_dim = getattr(aggregator, "embed_dim", None)
        if base_dim is None:
            base_dim = getattr(self.encoder, "embed_dim", None)
        return 2 * int(base_dim if base_dim is not None else 1024)

    @staticmethod
    def _as_rgb(image: Any) -> Image.Image:
        if isinstance(image, (list, tuple)):
            if not image:
                raise ValueError("empty image view list")
            image = image[0]
        if not isinstance(image, Image.Image):
            raise TypeError(f"VGGT expects PIL images, got {type(image)!r}")
        return image.convert("RGB")

    def _prepare_views(self, images: Sequence, device: torch.device) -> torch.Tensor:
        rows = []
        for sample in images:
            views = list(sample) if isinstance(sample, (list, tuple)) else [sample]
            if len(views) < 2:
                raise ValueError(
                    "VGGT spatial alignment requires two views per sample: base and wrist"
                )
            rows.append(torch.stack([self.transform(self._as_rgb(view)) for view in views[:2]], dim=0))
        return torch.stack(rows, dim=0).to(device=device, dtype=torch.float32)

    def _unwrap_features(self, output: Any, images: torch.Tensor):
        patch_start_idx = self.patch_start_idx
        features = output.get("features") if isinstance(output, dict) else None
        if isinstance(output, dict):
            patch_start_idx = int(output.get("patch_start_idx", patch_start_idx))
        if features is None:
            aggregator = getattr(self.encoder, "aggregator", None)
            if aggregator is None:
                raise RuntimeError("VGGT output has no features and encoder has no aggregator")
            features, patch_start_idx = aggregator(images)
        if isinstance(features, (list, tuple)):
            features = features[self.layer]
        return features, patch_start_idx

    @torch.no_grad()
    def encode(self, images: Sequence, device: torch.device, target_tokens: int) -> torch.Tensor:
        batch = self._prepare_views(images, device)
        self.encoder.to(device=device)
        self.encoder.eval()
        # The official VGGT model returns prediction heads from forward(); for
        # alignment we intentionally run only its frozen aggregator once.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            features, patch_start_idx = self._unwrap_features({}, batch)
        if features.ndim == 3:
            bsz, views = batch.shape[:2]
            if features.shape[0] == bsz * views:
                features = features.reshape(bsz, views, features.shape[1], features.shape[2])
            elif features.shape[0] == bsz:
                features = features.unsqueeze(1)
            else:
                raise ValueError(f"unsupported VGGT feature shape {tuple(features.shape)}")
        if features.ndim != 4:
            raise ValueError(f"unsupported VGGT feature shape {tuple(features.shape)}")
        features = features[..., int(patch_start_idx):, :]
        return pool_vggt_tokens(features, int(target_tokens), self.use_vggt_pe).detach()
