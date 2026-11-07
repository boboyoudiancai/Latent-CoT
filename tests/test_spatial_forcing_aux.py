from pathlib import Path

import torch
import yaml

from starVLA.model.modules.vlm.spatial_forcing_aux import (
    SpatialForcingAlignProjector,
    select_base_image_hidden,
    spatial_forcing_cosine_loss,
)
from starVLA.model.modules.vlm.vggt_teacher import pool_vggt_tokens


def test_select_base_image_hidden_keeps_first_view_tokens():
    image_token_id = 99
    input_ids = torch.zeros(2, 100, dtype=torch.long)
    input_ids[:, 1:99] = image_token_id
    attention_mask = torch.ones_like(input_ids)
    hidden = torch.arange(2 * 100 * 3, dtype=torch.float32).reshape(2, 100, 3)
    image_grid_thw = torch.tensor([[1, 14, 14]] * 4)

    selected = select_base_image_hidden(
        hidden=hidden,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        image_token_id=image_token_id,
        spatial_merge_size=2,
        image_counts=[2, 2],
    )

    assert selected.shape == (2, 49, 3)
    assert torch.equal(selected[0], hidden[0, 1:50])
    assert torch.equal(selected[1], hidden[1, 1:50])


def test_spatial_forcing_projector_and_cosine_loss_contract():
    projector = SpatialForcingAlignProjector(input_dim=8, target_dim=6, use_vlm_norm=True)
    hidden = torch.randn(2, 49, 8)
    target = projector(hidden).detach()

    loss = spatial_forcing_cosine_loss(projector, hidden, target)

    assert projector.fc1.in_features == 8
    assert projector.fc1.out_features == 12
    assert projector.fc2.in_features == 12
    assert projector.fc2.out_features == 6
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1.0e-6)


def test_vggt_tokens_downsample_to_qwen_8x8_grid():
    vggt_hidden = torch.randn(2, 1, 37 * 37, 8)
    pooled = pool_vggt_tokens(vggt_hidden, target_tokens=64, add_positional_encoding=True)
    assert pooled.shape == (2, 1, 64, 8)


def test_future_spatial_training_config_matches_laravla_gr00t():
    config_path = (
        Path(__file__).parents[1]
        / "examples"
        / "simBenchmarks"
        / "LIBERO"
        / "train_files"
        / "starvla_qwenpiv3_future_libero.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    future = config["framework"]["future"]
    spatial = config["framework"]["spatial_forcing_aux"]

    assert future["enable"] is True
    assert future["loss_weight"] == 0.1
    assert spatial == {
        "enable": True,
        "input_size": 518,
        "preprocess_mode": "resize",
        "target_tokens": 64,
        "student_layer": 24,
        "teacher_layer": -1,
        "patch_start_idx": 5,
        "pooling_func": "bilinear",
        "use_vggt_pe": True,
        "use_vlm_norm": True,
        "loss_weight": 0.5,
        "teacher_repo": "/share/home/u25505/apps/wangbo/LatentVLA/LaRA-VLA/playground/Pretrained_models/VGGT-1B",
        "target_cache_size": 0,
        "target_cache_dir": None,
        "target_cache_only": False,
    }
