import torch

from starVLA.model.modules.vlm.future_residual import (
    EMARMSNormalizer,
    FuturePredictionMLP,
    masked_future_loss,
)


def test_future_prediction_mlp_preserves_shape():
    module = FuturePredictionMLP(hidden_size=8, expansion_ratio=2.0)
    output = module(torch.randn(2, 16, 8))
    assert output.shape == (2, 16, 8)
    assert module.net[1].out_features == 16


def test_ema_rms_uses_only_valid_samples():
    normalizer = EMARMSNormalizer(feature_dim=2, decay=0.99, eps=1.0e-6)
    residual = torch.tensor([[[1.0, 2.0]], [[100.0, 200.0]]])
    normalized = normalizer(residual, torch.tensor([True, False]), update=True)
    assert torch.allclose(normalizer.ema_square, torch.tensor([1.0, 4.0]))
    assert torch.allclose(normalized[0], torch.ones_like(normalized[0]), atol=1.0e-5)


def test_masked_future_loss_ignores_fallback_samples():
    prediction = torch.zeros(2, 16, 4)
    target = torch.cat([torch.ones(1, 16, 4), torch.full((1, 16, 4), 100.0)], dim=0)
    loss = masked_future_loss(prediction, target, torch.tensor([True, False]))
    assert torch.allclose(loss, torch.tensor(1.0))
