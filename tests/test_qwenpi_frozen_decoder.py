"""CPU regressions for single-stage QwenPI + a frozen original 4B decoder.

The full-size model loaders are mocked with tiny real Qwen3-VL models; these
checks neither load pretrained weights nor launch training or use a GPU.
"""

import os
import runpy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["STARVLA_DISABLE_DEEPSPEED"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.modules.vlm.thinking_config import validate_stage2_decoder_config
from starVLA.model.modules.vlm.latent_text_decoder import FullQwenLatentTextDecoder, ResidualLatentProjector
from starVLA.training.train_starvla import VLATrainer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/simBenchmarks/LIBERO/train_files/starvla_qwenpi_frozen4b_decoder_single_stage.yaml"
helpers = runpy.run_path(str(Path(__file__).with_name("test_qwenpi_language_only.py")))


class Tokenizer:
    eos_token_id, pad_token_id, padding_side = 2, 0, "left"

    def __call__(self, texts, **kwargs):
        ids = torch.tensor([[3 + i % 3, 7] for i in range(len(texts))])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class Action(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, layers, actions, state, encoder_attention_mask):
        return self.scale.square() * sum(layer[:, -4:].square().mean() for layer in layers)

    def predict_action(self, layers, state, encoder_attention_mask):
        return layers[-1].new_zeros((layers[-1].shape[0], 8, 7))


def example():
    return {"image": [Image.new("RGB", (32, 32))], "lang": "reach cup",
            "action": np.zeros((8, 7), dtype=np.float32), "cot_subtask": "reach",
            "cot_reasoning": "move toward cup", "bbox": [0.1, 0.2, 0.3, 0.4], "bbox_valid": True}


def tiny_model():
    torch.manual_seed(29)
    model = Qwen_PI_v3.__new__(Qwen_PI_v3)
    torch.nn.Module.__init__(model)
    model.config = OmegaConf.load(CONFIG)
    model.config.trainer.repeated_diffusion_steps = 1
    model.qwen_vl_interface = helpers["language_interface"]()
    ids = torch.tensor([[5, 6, 44, 43, 43, 43, 45]])
    model.qwen_vl_interface.build_qwenvl_inputs = lambda images, instructions: {
        "input_ids": ids.repeat(len(images), 1),
        "attention_mask": torch.ones(len(images), 7, dtype=torch.long)}
    model.num_action_dit_layers = 2
    model.project_layers = torch.nn.ModuleList([torch.nn.Linear(32, 32), torch.nn.Linear(32, 32)])
    model.action_model = Action()
    model.spatial_forcing_enabled = False
    model.action_horizon = 8
    decoder = helpers["tiny_interface"]().model
    tokenizer = Tokenizer()
    with patch("transformers.Qwen3VLForConditionalGeneration.from_pretrained", return_value=decoder) as loader, \
         patch("transformers.AutoProcessor.from_pretrained", return_value=SimpleNamespace(tokenizer=tokenizer)), \
         patch.object(decoder, "resize_token_embeddings", wraps=decoder.resize_token_embeddings) as resize:
        model._init_latent_decoder(32)
        resize.assert_not_called()
    assert loader.call_args.args[0] == model.config.framework.qwenvl.base_vlm
    assert loader.call_args.kwargs["local_files_only"]
    assert tokenizer.padding_side == "right"
    return model


class FrozenDecoderTest(unittest.TestCase):
    def test_configuration_and_real_optimizer_step_warmup(self):
        cfg = OmegaConf.load(CONFIG)
        validate_stage2_decoder_config(cfg)
        assert cfg.framework.name == "QwenPI_v3"
        assert cfg.trainer.max_train_steps == 40000
        assert cfg.datasets.vla_data.per_device_batch_size * 8 * cfg.trainer.gradient_accumulation_steps == 112
        assert cfg.trainer.num_warmup_steps == 5000 and cfg.trainer.scale_scheduler_by_world_size
        assert cfg.datasets.vla_data.include_thinking_annotations
        assert cfg.datasets.vla_data.sampling_mode == "lara_uniform_steps"
        assert cfg.framework.latent_reasoning.token_count == 3
        assert cfg.framework.latent_reasoning.decoder_loss_weight == 0.1
        assert cfg.framework.latent_reasoning.decoder_input_projector == "residual_mlp"
        assert not cfg.framework.future.enable and not cfg.framework.spatial_forcing_aux.enable
        assert not cfg.datasets.vla_data.include_state and not cfg.datasets.vla_data.include_future

    def test_guard_rejects_non_single_stage_settings(self):
        changes = {"framework.latent_reasoning.enable": False,
                   "framework.latent_reasoning.stage": 1,
                   "framework.latent_reasoning.freeze_decoder": False,
                   "framework.latent_reasoning.require_stage1_checkpoint": True,
                   "framework.latent_reasoning.require_base_initialization": False,
                   "framework.latent_reasoning.decoder_base_vlm": "/models/Qwen3-0.6B",
                   "framework.latent_reasoning.decoder_input_projector": "unknown",
                   "trainer.pretrained_checkpoint": "/stage1/10000.pt",
                   "trainer.reload_modules": "qwen_vl_interface",
                   "trainer.start_step": 10000,
                   "trainer.freeze_modules": "qwen_vl_interface"}
        for key, value in changes.items():
            cfg = OmegaConf.load(CONFIG)
            OmegaConf.update(cfg, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_stage2_decoder_config(cfg)

    def test_existing_two_stage_configs_are_unchanged(self):
        for stage in (1, 2):
            cfg = OmegaConf.load(CONFIG.with_name(f"starvla_qwenpi_language_only_stage{stage}.yaml"))
            validate_stage2_decoder_config(cfg)
        # Existing action-only Stage2 must not instantiate any decoder.
        model = Qwen_PI_v3.__new__(Qwen_PI_v3)
        torch.nn.Module.__init__(model)
        model.config = cfg
        with patch("transformers.Qwen3VLForConditionalGeneration.from_pretrained") as loader:
            model._init_latent_decoder(32)
        loader.assert_not_called()
        assert model.latent_decoder is None

    def test_frozen_decoder_mode_and_optimizer_exclusion(self):
        model = tiny_model().train()
        assert model.latent_stage == 2
        assert model.latent_decoder.training and not model.latent_decoder.model.training
        projector = model.latent_decoder.latent_input_projection
        assert isinstance(projector, ResidualLatentProjector)
        assert all(param.requires_grad for param in projector.parameters())
        assert all(not param.requires_grad for param in model.latent_decoder.model.parameters())
        for module in (model.qwen_vl_interface, model.action_model, model.project_layers):
            assert any(param.requires_grad for param in module.parameters())
        from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
        groups = build_param_lr_groups(model, model.config)
        optimizer_ids = {id(param) for group in groups for param in group["params"]}
        assert not optimizer_ids.intersection(id(param) for param in model.latent_decoder.model.parameters())
        assert {id(param) for param in projector.parameters()} <= optimizer_ids
        group = next(g for g in groups if g["name"] == "latent_decoder.latent_input_projection")
        assert group["lr"] == 1e-4
        grouped_ids = [id(param) for g in groups for param in g["params"]]
        assert len(grouped_ids) == len(set(grouped_ids))
        model.eval().train()
        assert not model.latent_decoder.model.training

    def test_joint_loss_gradients_and_frozen_weights(self):
        model = tiny_model().train()
        before = {key: value.detach().clone() for key, value in model.latent_decoder.model.state_dict().items()}
        projector = model.latent_decoder.latent_input_projection
        projector_before = projector.fc2.weight.detach().clone()
        result = model([example()])
        assert set(result) == {"action_loss", "decoder_loss", "total_loss"}
        torch.testing.assert_close(result["total_loss"], result["action_loss"] + 0.1 * result["decoder_loss"])
        optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=0.01)
        result["total_loss"].backward()
        for module in (model.qwen_vl_interface, model.action_model, model.project_layers):
            assert any(param.grad is not None and torch.isfinite(param.grad).all()
                       and param.grad.abs().sum() > 0 for param in module.parameters())
        assert all(param.grad is None for param in model.latent_decoder.model.parameters())
        assert projector.fc2.weight.grad is not None and projector.fc2.weight.grad.abs().sum() > 0
        optimizer.step()
        assert not torch.equal(projector.fc2.weight, projector_before)
        for key, value in model.latent_decoder.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_decoder_loss_alone_backpropagates_to_recursive_student(self):
        model = tiny_model().train()
        result = model([example()])
        result["decoder_loss"].backward()
        query = model.qwen_vl_interface.model.model.language_model.layers[0].self_attn.q_proj.weight
        assert query.grad is not None and torch.isfinite(query.grad).all() and query.grad.abs().sum() > 0
        assert all(param.grad is None for param in model.action_model.parameters())
        assert all(param.grad is None for param in model.latent_decoder.model.parameters())
        assert model.latent_decoder.latent_input_projection.fc2.weight.grad.abs().sum() > 0

    def test_teacher_forcing_alignment_and_no_target_leakage(self):
        decoder = tiny_model().train().latent_decoder
        # Non-identity weights ensure both teacher forcing and text generation
        # actually apply the learned projector, rather than merely testing its
        # identity initialization.
        with torch.no_grad():
            decoder.latent_input_projection.fc2.weight.normal_(std=0.02)
        latent = torch.randn(3, 1, 32, requires_grad=True)
        ids = torch.tensor([[3, 4, 5, 2], [6, 7, 2, 0], [8, 2, 0, 0]])
        mask = ids.ne(0)
        labels = ids.masked_fill(~mask, -100)
        actual = decoder(latent, ids, labels, mask)["loss"]
        projected = decoder.latent_input_projection(latent)
        inputs = torch.cat([projected, decoder.input_embeddings(ids)], dim=1)
        attention = torch.cat([torch.ones(3, 1, dtype=torch.bool), mask], dim=1)
        logits = decoder.model(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits
        reference = F.cross_entropy(logits[:, :-1].reshape(-1, 64).float(), labels.reshape(-1), ignore_index=-100)
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        first = decoder._next_token_logits(latent, ids[:, :0])
        torch.testing.assert_close(first, logits[:, 0], rtol=1e-5, atol=1e-6)
        changed = ids.clone()
        changed[:, 1:] = 10
        alternative = decoder.model(inputs_embeds=torch.cat([projected, decoder.input_embeddings(changed)], dim=1),
                                    attention_mask=attention, use_cache=False).logits
        torch.testing.assert_close(logits[:, 1], alternative[:, 1], rtol=1e-5, atol=1e-6)
        actual.backward()
        assert (latent.grad.abs().sum(dim=(1, 2)) > 0).all()

    def test_strict_state_roundtrip_and_action_inference_without_annotations(self):
        model = tiny_model().eval()
        with torch.no_grad():
            model.latent_decoder.latent_input_projection.fc2.weight.normal_(std=0.02)
        restored = tiny_model().eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        assert all(not param.requires_grad for param in restored.latent_decoder.model.parameters())
        assert all(param.requires_grad for param in restored.latent_decoder.latent_input_projection.parameters())
        for key, value in restored.latent_decoder.latent_input_projection.state_dict().items():
            torch.testing.assert_close(value, model.latent_decoder.latent_input_projection.state_dict()[key])
        action_example = {key: value for key, value in example().items()
                          if key in {"image", "lang", "action"}}
        assert restored.predict_action([action_example])["normalized_actions"].shape == (1, 8, 7)

    def test_projector_identity_initialization_and_all_layers_eventually_train(self):
        projector = ResidualLatentProjector(32)
        latent = torch.randn(3, 1, 32, requires_grad=True)
        torch.testing.assert_close(projector(latent), latent, rtol=0, atol=0)
        optimizer = torch.optim.SGD(projector.parameters(), lr=0.01)
        projector(latent).square().mean().backward()
        assert projector.fc2.weight.grad.abs().sum() > 0
        assert latent.grad.abs().sum() > 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        projector(latent).square().mean().backward()
        for name, param in projector.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0, name

    def test_default_projection_keeps_legacy_decoder_compatible(self):
        decoder = FullQwenLatentTextDecoder(helpers["tiny_interface"]().model, 32, 64)
        assert isinstance(decoder.latent_input_projection, torch.nn.Identity)
        assert not any(key.startswith("latent_input_projection.") for key in decoder.state_dict())
        with self.assertRaises(ValueError):
            FullQwenLatentTextDecoder(helpers["tiny_interface"]().model, 32, 64,
                                     latent_hidden_size=16, input_projection_type="residual_mlp")

    def test_trainer_reports_both_losses(self):
        model = tiny_model().train()
        trainer = VLATrainer.__new__(VLATrainer)
        trainer.config, trainer.model = model.config, model
        trainer.accelerator = SimpleNamespace(accumulate=lambda _: nullcontext(),
                                              backward=lambda loss: loss.backward(), sync_gradients=False)
        trainer.optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.001)
        metrics = trainer._train_step([example()])
        assert set(metrics) == {"action_dit_loss", "decoder_loss", "total_loss"}
        self.assertAlmostEqual(metrics["total_loss"], metrics["action_dit_loss"] + 0.1 * metrics["decoder_loss"], places=5)

    def test_allows_only_this_runs_strict_resume(self):
        cfg = OmegaConf.load(CONFIG)
        cfg.trainer.is_resume = True
        validate_stage2_decoder_config(cfg)
        cfg.trainer.resume_mode = "weights_only"
        with self.assertRaises(ValueError):
            validate_stage2_decoder_config(cfg)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
