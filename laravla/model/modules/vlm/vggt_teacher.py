"""VGGT feature adapter used by the Spatial-Forcing style auxiliary loss."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


_PACKED_CACHE_FORMAT = "vggt-packed-bfloat16-v1"


class _PackedVGGTShard:
    """Read-only memory-mapped VGGT targets committed by a precompute worker."""

    def __init__(self, manifest_path: Path) -> None:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format") != _PACKED_CACHE_FORMAT:
            raise ValueError(f"unsupported VGGT packed-cache manifest: {manifest_path}")
        self.manifest_path = manifest_path
        self.data_path = manifest_path.parent / manifest["data_file"]
        self.keys_path = manifest_path.parent / manifest["keys_file"]
        self.count = int(manifest["count"])
        self.sample_shape = tuple(int(value) for value in manifest["sample_shape"])
        self.sample_numel = math.prod(self.sample_shape)
        expected_bytes = self.count * self.sample_numel * 2
        actual_bytes = self.data_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"invalid VGGT packed-cache data size for {self.data_path}: "
                f"expected {expected_bytes}, got {actual_bytes}"
            )
        self._tensor = None

    def iter_keys(self):
        with self.keys_path.open("r", encoding="ascii") as handle:
            for row, line in enumerate(handle):
                key_hex = line.strip()
                if key_hex:
                    yield row, bytes.fromhex(key_hex)

    def get(self, row: int) -> torch.Tensor:
        if self._tensor is None:
            self._tensor = torch.from_file(
                str(self.data_path),
                shared=False,
                size=self.count * self.sample_numel,
                dtype=torch.bfloat16,
            ).reshape(self.count, *self.sample_shape)
        return self._tensor[int(row)]


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
        cache_size: int = 0,
        cache_dir: str = "",
        cache_only: bool = False,
        view_mode: str = "dual",
        expected_patch_tokens: int = None,
    ) -> None:
        if not teacher_repo:
            raise ValueError("framework.spatial.teacher_repo must point to a local VGGT repository")
        self.teacher_repo = str(teacher_repo)
        self.input_size = int(input_size)
        self.layer = int(layer)
        self.expected_patch_tokens = expected_patch_tokens
        self.patch_start_idx = int(patch_start_idx)
        self.use_vggt_pe = bool(use_vggt_pe)
        self.preprocess_mode = str(preprocess_mode)
        self.view_mode = str(view_mode or "dual").strip().lower()
        if self.view_mode not in {"dual", "base_only"}:
            raise ValueError(f"unsupported VGGT view_mode={self.view_mode!r}")
        self.cache_size = max(0, int(cache_size))
        self._feature_cache = OrderedDict()
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.cache_only = bool(cache_only)
        self._packed_shards = []
        self._packed_index = {}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_packed_cache_index()
        self.transform = build_vggt_transform(self.input_size, self.preprocess_mode)
        if self.cache_only and self._packed_shards:
            feature_dims = {shard.sample_shape[-1] for shard in self._packed_shards}
            if len(feature_dims) != 1:
                raise ValueError(f"inconsistent VGGT feature dims across packed shards: {feature_dims}")
            self.encoder = None
            self.embed_dim = feature_dims.pop()
            return
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
        aggregator = getattr(self.encoder, "aggregator", None)
        if aggregator is not None and self.layer == -1 and hasattr(aggregator, "cached_layer_indices"):
            aggregator.cached_layer_indices = {int(aggregator.depth) - 1}
        self.embed_dim = self._infer_embed_dim()

    def _load_packed_cache_index(self) -> None:
        manifest_paths = sorted(self.cache_dir.glob("packed_*.manifest.json"))
        for manifest_path in manifest_paths:
            shard = _PackedVGGTShard(manifest_path)
            shard_index = len(self._packed_shards)
            key_count = 0
            for row, key in shard.iter_keys():
                if row >= shard.count:
                    raise ValueError(f"too many keys in VGGT cache shard: {manifest_path}")
                if key in self._packed_index:
                    raise ValueError(f"duplicate key across VGGT cache shards: {key.hex()}")
                self._packed_index[key] = (shard_index, row)
                key_count += 1
            if key_count != shard.count:
                raise ValueError(
                    f"VGGT cache shard key count mismatch for {manifest_path}: "
                    f"expected {shard.count}, got {key_count}"
                )
            self._packed_shards.append(shard)

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
            required_views = 1 if self.view_mode == "base_only" else 2
            if len(views) < required_views:
                raise ValueError(
                    "VGGT spatial alignment requires "
                    f"{required_views} view(s) per sample for view_mode={self.view_mode}"
                )
            rows.append(
                torch.stack(
                    [self.transform(self._as_rgb(view)) for view in views[:required_views]],
                    dim=0,
                )
            )
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

    def _image_cache_key(self, sample: Any, target_tokens: int, cache_key=None) -> bytes:
        digest = hashlib.sha256()
        digest.update(str((self.teacher_repo, self.input_size, self.layer, self.patch_start_idx, self.use_vggt_pe, self.preprocess_mode, int(target_tokens))).encode("utf-8"))
        if cache_key is not None:
            digest.update(str(cache_key).encode("utf-8"))
            return digest.digest()
        views = list(sample) if isinstance(sample, (list, tuple)) else [sample]
        required_views = 1 if self.view_mode == "base_only" else 2
        for view in views[:required_views]:
            image = self._as_rgb(view)
            digest.update(str(image.size).encode("ascii"))
            digest.update(image.tobytes())
        return digest.digest()

    def _disk_cache_path(self, key: bytes):
        if self.cache_dir is None:
            return None
        return self.cache_dir / (key.hex() + ".pt")

    def _load_cached_target(self, key: bytes):
        value = self._feature_cache.get(key)
        if value is not None:
            self._feature_cache.move_to_end(key)
            return value
        packed_location = self._packed_index.get(key)
        if packed_location is not None:
            shard_index, row = packed_location
            value = self._packed_shards[shard_index].get(row)
            if self.cache_size > 0:
                self._feature_cache[key] = value
                self._feature_cache.move_to_end(key)
                while len(self._feature_cache) > self.cache_size:
                    self._feature_cache.popitem(last=False)
            return value
        path = self._disk_cache_path(key)
        if path is None or not path.is_file():
            return None
        try:
            value = torch.load(path, map_location="cpu")
            if not isinstance(value, torch.Tensor):
                raise TypeError("cached VGGT target is not a tensor")
        except Exception:
            path.unlink(missing_ok=True)
            return None
        self._feature_cache[key] = value
        self._feature_cache.move_to_end(key)
        while len(self._feature_cache) > self.cache_size:
            self._feature_cache.popitem(last=False)
        return value

    def cache_digest(self, sample: Any, target_tokens: int, cache_key=None) -> bytes:
        """Return the exact cache key used by both packed and legacy caches."""
        return self._image_cache_key(sample, target_tokens, cache_key=cache_key)

    def _store_cached_target(self, key: bytes, value: torch.Tensor):
        value = value.detach().cpu()
        self._feature_cache[key] = value
        self._feature_cache.move_to_end(key)
        while len(self._feature_cache) > self.cache_size:
            self._feature_cache.popitem(last=False)
        path = self._disk_cache_path(key)
        if path is None:
            return
        tmp = path.with_name(path.name + ".tmp." + str(os.getpid()))
        try:
            torch.save(value, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    @torch.no_grad()
    def _encode_uncached(self, images: Sequence, device: torch.device, target_tokens: int) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("VGGT encoder is unavailable in packed cache-only mode")
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
        if self.expected_patch_tokens is not None:
            if features.shape[-2] != self.expected_patch_tokens or int(target_tokens) != self.expected_patch_tokens:
                raise ValueError(
                    f"full-resolution VGGT alignment requires {self.expected_patch_tokens} patches, "
                    f"got features={features.shape[-2]}, target_tokens={target_tokens}"
                )
            if not self.use_vggt_pe:
                return features.detach()
        return pool_vggt_tokens(features, int(target_tokens), self.use_vggt_pe).detach()

    @torch.no_grad()
    def encode(self, images: Sequence, device: torch.device, target_tokens: int, cache_keys=None) -> torch.Tensor:
        if cache_keys is not None and len(cache_keys) != len(images):
            raise ValueError("VGGT cache_keys length must match images length")
        if self.cache_size <= 0 and self.cache_dir is None and not self.cache_only:
            return self._encode_uncached(images, device, target_tokens)

        keys = [
            self._image_cache_key(sample, target_tokens, cache_key=cache_keys[index] if cache_keys is not None else None)
            for index, sample in enumerate(images)
        ]
        cached = [self._load_cached_target(key) for key in keys]
        missing_indices = [index for index, value in enumerate(cached) if value is None]
        if missing_indices and self.cache_only:
            missing_keys = [str(cache_keys[index]) if cache_keys is not None else keys[index].hex() for index in missing_indices]
            raise FileNotFoundError(
                "VGGT spatial cache miss in cache-only mode: " + ", ".join(missing_keys[:8])
            )
        if missing_indices:
            missing_images = [images[index] for index in missing_indices]
            computed = self._encode_uncached(missing_images, device, target_tokens)
            for index, value in zip(missing_indices, computed):
                key = keys[index]
                self._store_cached_target(key, value)
                cached[index] = value.detach().cpu()

        return torch.stack(
            [value.to(device=device, non_blocking=True) for value in cached],
            dim=0,
        )
