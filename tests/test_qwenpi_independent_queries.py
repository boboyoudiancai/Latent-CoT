"""CPU tests for the opt-in single-pass, three-query latent ablation."""

import os
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["STARVLA_DISABLE_DEEPSPEED"] = "1"

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
from starVLA.model.modules.vlm.thinking_config import latent_forward_mode, validate_stage2_decoder_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "examples/simBenchmarks/LIBERO/train_files"
BASE_CONFIG = CONFIG_ROOT / "starvla_qwenpi_frozen4b_decoder_single_stage.yaml"
CONFIG = CONFIG_ROOT / "starvla_qwenpi_frozen4b_independent_queries_single_stage.yaml"
helpers = runpy.run_path(str(Path(__file__).with_name("test_qwenpi_frozen_decoder.py")))


def query_interface():
    interface = helpers["helpers"]["language_interface"]()
    interface.latent_forward_mode = "independent_queries"
    interface._init_independent_thinking_embeddings()
    return interface


def query_model():
    model = helpers["tiny_model"]()
    model.config.framework.latent_reasoning.forward_mode = "independent_queries"
    model.qwen_vl_interface.latent_forward_mode = "independent_queries"
    model.qwen_vl_interface._init_independent_thinking_embeddings()
    return model


class IndependentQueriesTest(unittest.TestCase):
    def test_only_mode_and_run_id_differ_from_frozen_projector_baseline(self):
        original, ablation = OmegaConf.load(BASE_CONFIG), OmegaConf.load(CONFIG)
        validate_stage2_decoder_config(ablation)
        assert ablation.run_id != original.run_id
        assert ablation.framework.latent_reasoning.forward_mode == "independent_queries"
        ablation.run_id = original.run_id
        del ablation.framework.latent_reasoning.forward_mode
        assert OmegaConf.to_container(ablation, resolve=True) == OmegaConf.to_container(original, resolve=True)
        assert latent_forward_mode(original.framework.latent_reasoning) == "recursive"
        for cfg in ({"enable": True, "forward_mode": "typo"},
                    {"enable": False, "forward_mode": "independent_queries"}):
            with self.assertRaises(ValueError):
                latent_forward_mode(cfg)

    def test_constructor_registers_only_opt_in_parameter_without_advancing_rng(self):
        for mode in ("recursive", "independent_queries"):
            cfg = OmegaConf.load(CONFIG)
            cfg.framework.latent_reasoning.forward_mode = mode
            model = helpers["helpers"]["tiny_interface"]().model
            processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))
            rng = torch.random.get_rng_state()
            with patch("transformers.Qwen3VLForConditionalGeneration.from_pretrained", return_value=model), \
                 patch("transformers.AutoProcessor.from_pretrained", return_value=processor), \
                 patch.object(_QWen3_VL_Interface, "_add_thinking_tokens", return_value=43):
                interface = _QWen3_VL_Interface(cfg)
            assert interface.latent_forward_mode == mode
            assert torch.equal(rng, torch.random.get_rng_state())
            if mode == "recursive":
                assert "thinking_embeddings" not in interface.state_dict()
            else:
                queries = interface.thinking_embeddings
                assert queries.shape == (3, 32) and queries.requires_grad
                torch.testing.assert_close(queries, model.get_input_embeddings().weight[43].expand(3, -1))

    def test_single_forward_reads_same_position_outputs_not_inputs_or_previous(self):
        torch.manual_seed(31)
        interface = query_interface().eval()
        ids = torch.tensor([[0, 5, 6, 44, 43, 43, 43, 45], [7, 5, 6, 44, 43, 43, 43, 45]])
        mask = ids.ne(0).long()
        with patch.object(interface.model.model, "forward", wraps=interface.model.model.forward) as forward:
            actual = interface.forward_latent(input_ids=ids, attention_mask=mask)
        assert forward.call_count == 1
        passed = forward.call_args.kwargs
        assert passed["use_cache"] is False and "past_key_values" not in passed
        torch.testing.assert_close(passed["inputs_embeds"][:, 4:7], interface.thinking_embeddings.expand(2, -1, -1))
        assert len(actual["hidden_states"]) == 3
        assert all(layer.shape == (2, 8, 32) for layer in actual["hidden_states"])
        torch.testing.assert_close(actual["latent_embeds"], actual["hidden_states"][-1][:, 4:7])
        assert not torch.allclose(actual["latent_embeds"], passed["inputs_embeds"][:, 4:7])
        assert not torch.allclose(actual["latent_embeds"], actual["hidden_states"][-1][:, 3:6])
        assert actual["latent_attention_mask"].all()
        with torch.no_grad():
            single = interface.forward_latent(input_ids=ids[:1, 1:], attention_mask=mask[:1, 1:])
        torch.testing.assert_close(actual["latent_embeds"][:1], single["latent_embeds"], rtol=1e-4, atol=1e-5)

    def test_causal_queries_and_context_dependence(self):
        torch.manual_seed(32)
        interface = query_interface().eval()
        ids = torch.tensor([[5, 6, 44, 43, 43, 43, 45]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            original = interface.forward_latent(input_ids=ids, attention_mask=mask)["latent_embeds"]
            interface.thinking_embeddings[2].add_(torch.linspace(-1, 1, 32))
            changed = interface.forward_latent(input_ids=ids, attention_mask=mask)["latent_embeds"]
            torch.testing.assert_close(original[:, :2], changed[:, :2], rtol=0, atol=0)
            assert not torch.allclose(original[:, 2], changed[:, 2])
            other_ids = ids.clone()
            other_ids[0, 0] = 9
            other = interface.forward_latent(input_ids=other_ids, attention_mask=mask)["latent_embeds"]
            assert not torch.allclose(changed, other)

    def test_query_rows_update_independently_and_placeholder_embedding_is_unused(self):
        interface = query_interface().eval()
        ids = torch.tensor([[5, 6, 44, 43, 43, 43, 45]])
        mask = torch.ones_like(ids)
        before = interface.thinking_embeddings.detach().clone()
        optimizer = torch.optim.SGD([interface.thinking_embeddings], lr=0.01)
        interface.thinking_embeddings[1].sum().backward()
        optimizer.step()
        torch.testing.assert_close(interface.thinking_embeddings[[0, 2]], before[[0, 2]], rtol=0, atol=0)
        assert not torch.equal(interface.thinking_embeddings[1], before[1])
        with torch.no_grad():
            expected = interface.forward_latent(input_ids=ids, attention_mask=mask)["latent_embeds"]
            interface.model.get_input_embeddings().weight[43].add_(10)
            actual = interface.forward_latent(input_ids=ids, attention_mask=mask)["latent_embeds"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get("QWENPI_THINKING_PROCESSOR"), "optional local processor integration")
    def test_real_processor_preserves_prompt_and_multimodal_batch_consistency(self):
        from PIL import Image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(os.environ["QWENPI_THINKING_PROCESSOR"], local_files_only=True)
        model_cfg = helpers["helpers"]["tiny_interface"]().model.config
        model_cfg.image_token_id, model_cfg.video_token_id = 151655, 151656
        model_cfg.vision_start_token_id, model_cfg.vision_end_token_id = 151652, 151653
        model_cfg.vision_config.patch_size = 16
        model_cfg.text_config.vocab_size = 153984
        backbone = Qwen3VLForConditionalGeneration(model_cfg)
        cfg = OmegaConf.load(CONFIG)
        with patch("transformers.Qwen3VLForConditionalGeneration.from_pretrained", return_value=backbone), \
             patch("transformers.AutoProcessor.from_pretrained", return_value=processor):
            interface = _QWen3_VL_Interface(cfg).eval()
        views = [[Image.new("RGB", (32, 32), (40, 50, 60)), Image.new("RGB", (32, 32), (1, 2, 3))]]
        with patch.object(processor, "apply_chat_template", wraps=processor.apply_chat_template) as template:
            batch = interface.build_qwenvl_inputs(views * 2, ["short", "a substantially longer instruction"])
        messages = template.call_args.args[0]
        text = messages[0][0]["content"][-1]["text"]
        assert text == "short <|start_of_thinking|>" + "<|thinking|>" * 3 + "<|end_of_thinking|>"
        assert processor.tokenizer.padding_side == "left" and not template.call_args.kwargs["add_generation_prompt"]
        with torch.no_grad():
            batched = interface.forward_latent(**batch)
            single = interface.forward_latent(**interface.build_qwenvl_inputs(views, ["short"]))
        torch.testing.assert_close(batched["latent_embeds"][:1], single["latent_embeds"], rtol=2e-4, atol=2e-5)

    def test_multimodal_positions_visual_gradients_and_future_mask_unchanged(self):
        torch.manual_seed(33)
        for future in (False, True):
            interface = query_interface().eval()
            interface.enable_future = future
            interface.future_token_id = 42 if future else None
            prefix = [0, 5, 62, 60, 63, 6] + ([42] * 16 if future else [])
            ids = torch.tensor([prefix + [44, 43, 43, 43, 45]])
            mask = ids.ne(0).long()
            grids, pixels = torch.tensor([[1, 2, 2]]), torch.randn(4, 24)
            with patch.object(interface.model.model, "forward", wraps=interface.model.model.forward) as forward:
                out = interface.forward_latent(input_ids=ids, attention_mask=mask,
                                                pixel_values=pixels, image_grid_thw=grids)
            passed = forward.call_args.kwargs
            expected_positions, _ = interface.model.model.get_rope_index(ids, grids, None, attention_mask=mask)
            torch.testing.assert_close(passed["position_ids"], expected_positions)
            torch.testing.assert_close(passed["attention_mask"], interface._build_future_attention_mask(mask, ids))
            out["latent_embeds"].float().square().mean().backward()
            grad = interface.model.model.visual.patch_embed.proj.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
            query_grad = interface.thinking_embeddings.grad
            assert torch.isfinite(query_grad).all() and (query_grad.abs().sum(dim=1) > 0).all()

    def test_joint_loss_and_separate_decoder_action_gradients(self):
        for loss_name in ("decoder_loss", "action_loss", "total_loss"):
            model = query_model().train()
            result = model([helpers["example"]()])
            torch.testing.assert_close(result["total_loss"], result["action_loss"] + 0.1 * result["decoder_loss"])
            result[loss_name].backward()
            grad = model.qwen_vl_interface.thinking_embeddings.grad
            assert grad is not None and torch.isfinite(grad).all() and (grad.abs().sum(dim=1) > 0).all()
            assert all(param.grad is None for param in model.latent_decoder.model.parameters())
            query_weight = model.qwen_vl_interface.model.model.language_model.layers[0].self_attn.q_proj.weight
            assert query_weight.grad is not None and query_weight.grad.abs().sum() > 0
            if loss_name != "action_loss":
                assert model.latent_decoder.latent_input_projection.fc2.weight.grad.abs().sum() > 0
            if loss_name == "decoder_loss":
                assert all(param.grad is None for param in model.action_model.parameters())

    def test_optimizer_checkpoint_and_same_train_inference_encoding(self):
        from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
        model = query_model().eval()
        groups = build_param_lr_groups(model, model.config)
        query = model.qwen_vl_interface.thinking_embeddings
        matches = [group for group in groups if any(param is query for param in group["params"])]
        assert len(matches) == 1 and matches[0]["lr"] == 1e-5
        example = helpers["example"]()
        restored = query_model().eval()
        with torch.no_grad():
            query[1].add_(torch.linspace(-0.1, 0.1, 32))
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            expected = model._encode_vl_hidden_states([example["image"]], [example["lang"]])[-1]["latent_embeds"]
            with patch.object(restored.qwen_vl_interface, "forward_latent",
                              wraps=restored.qwen_vl_interface.forward_latent) as encode:
                predicted = restored.predict_action([{key: example[key] for key in ("image", "lang")}])
            assert encode.call_count == 1 and predicted["normalized_actions"].shape == (1, 8, 7)
            actual = restored._encode_vl_hidden_states([example["image"]], [example["lang"]])[-1]["latent_embeds"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(helpers["tiny_model"]().state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            helpers["tiny_model"]().load_state_dict(restored.state_dict(), strict=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
