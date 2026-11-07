"""CPU regressions for language-only thinking, without Future or Spatial."""
import os
import runpy
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["STARVLA_DISABLE_DEEPSPEED"] = "1"

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.modules.vlm.latent_text_decoder import FullQwenLatentTextDecoder
from starVLA.training.train_starvla import _load_thinking_stage1_modules, _prune_completed_training_states

tiny_interface = runpy.run_path(str(Path(__file__).with_name("test_qwenpi_thinking.py")))["tiny_interface"]


def language_interface():
    interface = tiny_interface()
    interface.enable_future = False
    interface.future_token_id = None
    return interface


def test_language_recursion_matches_causal_full_forward_and_visual_gradients():
    torch.manual_seed(19)
    interface = language_interface().eval()
    ids = torch.tensor([[0, 5, 62, 60, 63, 6, 44, 43, 43, 43, 45],
                        [7, 5, 62, 60, 63, 6, 44, 43, 43, 43, 45]])
    mask = ids.ne(0).long()
    grids = torch.tensor([[1, 2, 2], [1, 2, 2]])
    pixels = torch.randn(8, 24)
    actual = interface.forward_latent(input_ids=ids, attention_mask=mask,
                                      pixel_values=pixels, image_grid_thw=grids)
    full_mask = interface._build_future_attention_mask(mask, ids)
    assert full_mask[0, 0, 2, 3] < 0 and full_mask[0, 0, 3, 2] == 0
    assert torch.all(full_mask[0, 0, :, 0] < 0)
    positions, _ = interface.model.model.get_rope_index(ids, grids, None, attention_mask=mask)
    embeds = interface.model.get_input_embeddings()(ids)
    for i in range(3):
        output = interface.model.model(inputs_embeds=embeds, attention_mask=full_mask,
            position_ids=positions, pixel_values=pixels, image_grid_thw=grids,
            output_hidden_states=True, use_cache=False, return_dict=True)
        latent = output.hidden_states[-1][:, 6 + i]
        torch.testing.assert_close(actual["latent_embeds"][:, i], latent, rtol=1e-4, atol=1e-5)
        embeds = torch.cat([embeds[:, :7+i], latent[:, None], embeds[:, 8+i:]], dim=1)
    expected = interface.model.model(inputs_embeds=embeds, attention_mask=full_mask,
        position_ids=positions, pixel_values=pixels, image_grid_thw=grids,
        output_hidden_states=True, use_cache=False, return_dict=True)
    for got, reference in zip(actual["hidden_states"], expected.hidden_states):
        torch.testing.assert_close(got[mask.bool()], reference[mask.bool()], rtol=1e-4, atol=1e-5)
    actual["latent_embeds"].square().mean().backward()
    grad = interface.model.model.visual.patch_embed.proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_language_stage1_decoder_only_stage2_action_only():
    class Tokenizer:
        eos_token_id, pad_token_id = 2, 0
        def __call__(self, texts, **kwargs):
            ids = torch.tensor([[3, 4]] * len(texts))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class Action(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
        def forward(self, layers, actions, state, encoder_attention_mask):
            return self.scale * sum(x[:, -4:].square().mean() for x in layers)
        def predict_action(self, layers, state, encoder_attention_mask):
            return layers[-1].new_zeros((layers[-1].shape[0], 8, 7))

    model = Qwen_PI_v3.__new__(Qwen_PI_v3)
    torch.nn.Module.__init__(model)
    model.config = OmegaConf.create({"framework": {"future": {"enable": False},
        "latent_reasoning": {"decoder_loss_weight": 0.1, "max_target_tokens": 8}},
        "datasets": {"vla_data": {}}, "trainer": {"repeated_diffusion_steps": 1}})
    model.qwen_vl_interface = language_interface()
    ids = torch.tensor([[5, 6, 44, 43, 43, 43, 45]])
    model.qwen_vl_interface.build_qwenvl_inputs = lambda images, instructions: {
        "input_ids": ids.repeat(len(images), 1), "attention_mask": torch.ones(len(images), 7, dtype=torch.long)}
    model.num_action_dit_layers = 2
    model.project_layers = torch.nn.ModuleList([torch.nn.Linear(32, 32), torch.nn.Linear(32, 32)])
    model.spatial_forcing_enabled = False
    model.action_horizon = 8
    model.action_model = Action()
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled auxiliary objective was executed")
    model._compute_future_loss = forbidden
    model._compute_spatial_forcing_aux_loss = forbidden
    model._latent_decoder_tokenizer = Tokenizer()
    model.latent_decoder = FullQwenLatentTextDecoder(tiny_interface().model, 32, 64,
                                                    use_forward_checkpoint=False)
    example = {"image": [Image.new("RGB", (32, 32))], "lang": "task",
        "action": np.zeros((8, 7), dtype=np.float32), "cot_subtask": "reach",
        "cot_reasoning": "go", "bbox": [0.1, 0.2, 0.3, 0.4], "bbox_valid": True}
    model.train()
    model.latent_stage = 1
    result = model([example])
    assert set(result) == {"action_loss", "decoder_loss", "total_loss"}
    torch.testing.assert_close(result["total_loss"], result["decoder_loss"] * 0.1)
    result["total_loss"].backward()
    assert model.action_model.scale.grad is None
    assert all(p.grad is None for p in model.project_layers.parameters())
    assert model.latent_decoder.model.lm_head.weight.grad.abs().sum() > 0
    grad = model.qwen_vl_interface.model.model.language_model.layers[0].self_attn.q_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.latent_stage = 2
    model.latent_decoder = None
    result = model([example])
    assert set(result) == {"action_loss", "total_loss"}
    torch.testing.assert_close(result["total_loss"], result["action_loss"])
    result["total_loss"].backward()
    assert model.action_model.scale.grad.abs().sum() > 0
    assert model.project_layers[0].weight.grad.abs().sum() > 0
    assert model.qwen_vl_interface.model.model.language_model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    model.eval()
    assert model.predict_action([example])["normalized_actions"].shape == (1, 8, 7)


def test_language_stage2_transfers_only_backbone_and_rejects_spatial_source():
    def model():
        obj = torch.nn.Module()
        obj.qwen_vl_interface = torch.nn.Linear(2, 2)
        obj.spatial_forcing_projector = None
        obj.action_model = torch.nn.Linear(2, 2)
        return obj
    with tempfile.TemporaryDirectory() as tmp:
        source, target = model(), model()
        checkpoint = dict(source.state_dict())
        checkpoint["latent_decoder.model.weight"] = torch.ones(1)
        path = Path(tmp) / "stage1.pt"
        torch.save(checkpoint, path)
        before_action = target.action_model.weight.detach().clone()
        _load_thinking_stage1_modules(target, path)
        torch.testing.assert_close(target.qwen_vl_interface.weight, source.qwen_vl_interface.weight)
        torch.testing.assert_close(target.action_model.weight, before_action)
        checkpoint["spatial_forcing_projector.weight"] = torch.ones(2, 2)
        torch.save(checkpoint, path)
        try:
            _load_thinking_stage1_modules(target, path)
        except ValueError as exc:
            assert "spatial-disabled" in str(exc)
        else:
            raise AssertionError("Language-only Stage 2 accepted the original spatial Stage 1")


def test_language_only_configs():
    root = Path("examples/simBenchmarks/LIBERO/train_files")
    for stage in (1, 2):
        cfg = OmegaConf.load(root / f"starvla_qwenpi_language_only_stage{stage}.yaml")
        assert not cfg.framework.future.enable and not cfg.framework.future.use_teacher
        assert not cfg.framework.spatial_forcing_aux.enable
        assert not cfg.datasets.vla_data.include_future and not cfg.datasets.vla_data.include_state
        assert cfg.framework.latent_reasoning.token_count == 3
        assert cfg.datasets.vla_data.sampling_mode == "lara_uniform_steps"
        assert cfg.datasets.vla_data.per_device_batch_size * cfg.trainer.gradient_accumulation_steps * 8 == 112
        assert cfg.trainer.max_train_steps == (10000 if stage == 1 else 100000)
        assert cfg.trainer.keep_last_training_states == 2
        if stage == 1:
            assert cfg.framework.latent_reasoning.decoder_loss_weight == 0.1
            assert cfg.trainer.pretrained_checkpoint is None
        else:
            assert cfg.trainer.reload_modules == "qwen_vl_interface"


def test_retention_keeps_weights_links_and_latest_two_states():
    import json
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for step in (1000, 2000, 3000):
            state = root / f"steps_{step}_training_state"
            state.mkdir()
            (state / "trainer_state.json").write_text(json.dumps({"completed_steps": step}))
            (root / f"steps_{step}_pytorch_model.pt").touch()
        incomplete = root / "steps_4000_training_state.incomplete"
        incomplete.mkdir()
        external = root / "external"
        external.mkdir()
        link = root / "steps_500_training_state"
        link.symlink_to(external, target_is_directory=True)
        removed = _prune_completed_training_states(root, 2, 3000)
        assert removed == [str(root / "steps_1000_training_state")]
        assert all((root / f"steps_{step}_pytorch_model.pt").exists() for step in (1000, 2000, 3000))
        assert link.is_symlink() and external.exists() and incomplete.exists()
        assert (root / "steps_2000_training_state").exists()
        assert (root / "steps_3000_training_state").exists()
        try:
            _prune_completed_training_states(root, 1, 3000)
        except ValueError:
            pass
        else:
            raise AssertionError("Retention accepted less than two states")


def test_real_language_only_processor_init_and_batch_consistency():
    """Use the local processor but a tiny backbone; no GPU or model download."""
    path = os.environ.get("QWENPI_THINKING_PROCESSOR")
    if not path:
        raise RuntimeError("Set QWENPI_THINKING_PROCESSOR to the local pretrained model")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(path)
    model_cfg = tiny_interface().model.config
    model_cfg.image_token_id, model_cfg.video_token_id = 151655, 151656
    model_cfg.vision_start_token_id, model_cfg.vision_end_token_id = 151652, 151653
    model_cfg.vision_config.patch_size = 16
    model_cfg.text_config.vocab_size = 153984
    model = Qwen3VLForConditionalGeneration(model_cfg)
    cfg = OmegaConf.create({"framework": {"qwenvl": {"base_vlm": "tiny-Action"},
        "future": {"enable": False}, "latent_reasoning": {"enable": True}},
        "datasets": {"vla_data": {}}})
    with patch("starVLA.model.modules.vlm.QWen3.Qwen3VLForConditionalGeneration.from_pretrained", return_value=model), \
         patch("starVLA.model.modules.vlm.QWen3.AutoProcessor.from_pretrained", return_value=processor):
        interface = _QWen3_VL_Interface(cfg).eval()
    assert interface.visual_ema is None and interface.future_prediction_mlp is None
    assert interface.future_residual_normalizer is None and interface.future_token_id is None
    views = [[Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]]
    captured = {}
    original = processor.apply_chat_template
    def capture(messages, **kwargs):
        captured.update(messages=messages, kwargs=kwargs)
        return original(messages, **kwargs)
    with patch.object(processor, "apply_chat_template", side_effect=capture):
        batch = interface.build_qwenvl_inputs(views * 2, ["short", "a substantially longer instruction"])
    text = captured["messages"][0][0]["content"][-1]["text"]
    assert text == "short <|start_of_thinking|>" + "<|thinking|>" * 3 + "<|end_of_thinking|>"
    assert not captured["kwargs"]["add_generation_prompt"] and processor.tokenizer.padding_side == "left"
    with torch.no_grad():
        batched = interface.forward_latent(**batch)
        single = interface.forward_latent(**interface.build_qwenvl_inputs(views, ["short"]))
    torch.testing.assert_close(batched["latent_embeds"][0], single["latent_embeds"][0], rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    checks = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for check in checks:
        check()
        print("PASS", check.__name__, flush=True)
    print(f"PASS {len(checks)} language-only regressions", flush=True)
