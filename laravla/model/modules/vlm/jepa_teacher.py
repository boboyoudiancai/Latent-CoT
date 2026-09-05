"""Adapter for a frozen V-JEPA 2.1 ViT-L target encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def pool_token_grid(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Pool a square spatial token grid to another square token grid."""
    if tokens.ndim != 3:
        raise ValueError(f"expected [B, N, C] tokens, got {tuple(tokens.shape)}")
    source_side = int(tokens.shape[1] ** 0.5)
    target_side = int(target_tokens**0.5)
    if source_side * source_side != tokens.shape[1]:
        raise ValueError(f"source token count is not square: {tokens.shape[1]}")
    if target_side * target_side != target_tokens:
        raise ValueError(f"target token count is not square: {target_tokens}")
    grid = tokens.transpose(1, 2).reshape(
        tokens.shape[0], tokens.shape[2], source_side, source_side
    )
    pooled = F.adaptive_avg_pool2d(grid, (target_side, target_side))
    return pooled.flatten(2).transpose(1, 2).contiguous()


def build_image_transform(input_size: int):
    """Match LaRA's full-frame geometry while using V-JEPA's native resolution."""
    return transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


class VJEPA21ViTLargeTeacher:
    """Frozen V-JEPA 2.1 ViT-L/16 teacher loaded from the official repository.

    This wrapper is intentionally not an ``nn.Module``. The external 300M
    teacher is therefore absent from LaRA checkpoints and optimizer groups.
    Only the small student projection used to match its 1024-d targets is
    registered on the LaRA model.
    """

    model_name = "vjepa2_1_vit_large_384"
    embed_dim = 1024

    def __init__(
        self,
        teacher_repo: str,
        input_size: int = 384,
    ) -> None:
        repo = Path(teacher_repo).expanduser().resolve()
        if not (repo / "config.json").is_file():
            raise FileNotFoundError(
                "V-JEPA 2.1 teacher_repo must be a local Transformers model directory: "
                f"{repo}"
            )
        from transformers import AutoModel

        hf_model = AutoModel.from_pretrained(
            str(repo),
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.float32,
        )
        self.encoder = getattr(hf_model, "encoder", hf_model)
        encoder_config = getattr(self.encoder, "config", None)
        self.embed_dim = int(getattr(encoder_config, "hidden_size", self.embed_dim))

        self.input_size = int(input_size)
        self.transform = build_image_transform(self.input_size)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @staticmethod
    def _primary_image(image):
        if isinstance(image, (list, tuple)):
            if not image:
                raise ValueError("empty image view list")
            image = image[0]
        if not isinstance(image, Image.Image):
            raise TypeError(f"V-JEPA teacher expects PIL images, got {type(image)!r}")
        return image.convert("RGB")

    def _to_device(self, device: torch.device, dtype: torch.dtype) -> None:
        parameter = next(self.encoder.parameters())
        if parameter.device != device or parameter.dtype != dtype:
            self.encoder.to(device=device, dtype=dtype)
        self.encoder.eval()

    @torch.no_grad()
    def encode(
        self,
        images: Sequence,
        device: torch.device,
        dtype: torch.dtype,
        target_tokens: int,
    ) -> torch.Tensor:
        self._to_device(device, dtype)
        batch = torch.stack(
            [self.transform(self._primary_image(image)) for image in images], dim=0
        ).to(device=device, dtype=dtype)
        # T=1 selects V-JEPA 2.1's image tokenizer instead of its video tubelet tokenizer.
        features = self.encoder(batch.unsqueeze(2))
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state
        return pool_token_grid(features, target_tokens).detach()
