import os
import torch
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
from starVLA.model.modules.vlm.latent_text_decoder import FullQwenLatentTextDecoder, build_latent_decoder_batch


def tiny_interface():
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=4, head_dim=8,
                         rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
                           num_heads=4, patch_size=2, temporal_patch_size=2,
                           spatial_merge_size=2, out_hidden_size=32,
                           num_position_embeddings=16, deepstack_visual_indexes=[]),
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3VLForConditionalGeneration(config)
    interface = _QWen3_VL_Interface.__new__(_QWen3_VL_Interface)
    torch.nn.Module.__init__(interface)
    interface.model = model
    interface.enable_future = True
    interface.future_token_id = 42
    interface.future_token_count = 16
    interface.enable_latent_reasoning = True
    interface.latent_forward_mode = "recursive"
    interface.thinking_token_id = 43
    interface.thinking_token_count = 3
    return interface


def test_recursive_thinking_matches_full_recomputation_and_backpropagates():
    torch.manual_seed(1)
    interface = tiny_interface().eval()
    ids = torch.tensor([[0, 0, 5, 6] + [42] * 16 + [44] + [43] * 3 + [45],
                        [8, 9, 5, 6] + [42] * 16 + [44] + [43] * 3 + [45]])
    mask = ids.ne(0).long()
    actual = interface.forward_latent(input_ids=ids, attention_mask=mask)
    assert len(actual["hidden_states"]) == 3
    assert actual["latent_embeds"].shape == (2, 3, 32)
    assert all(layer.shape == (2, ids.shape[1], 32) for layer in actual["hidden_states"])

    positions, _ = interface.model.model.get_rope_index(ids, None, None, attention_mask=mask)
    full_mask = interface._build_future_attention_mask(mask, ids)
    embeds = interface.model.get_input_embeddings()(ids)
    reference_latents = []
    first = int((ids[0] == 43).nonzero()[0])
    for index in range(3):
        output = interface.model.model(inputs_embeds=embeds, attention_mask=full_mask,
                                       position_ids=positions, output_hidden_states=True,
                                       use_cache=False, return_dict=True)
        latent = output.hidden_states[-1][:, first + index - 1]
        reference_latents.append(latent)
        embeds = torch.cat([embeds[:, :first + index], latent[:, None], embeds[:, first + index + 1:]], dim=1)
    reference = interface.model.model(inputs_embeds=embeds, attention_mask=full_mask,
                                      position_ids=positions, output_hidden_states=True,
                                      use_cache=False, return_dict=True)
    torch.testing.assert_close(actual["latent_embeds"], torch.stack(reference_latents, dim=1), rtol=1e-4, atol=1e-5)
    for got, expected in zip(actual["hidden_states"], reference.hidden_states):
        # Masked padding queries may differ; action attention ignores them.
        torch.testing.assert_close(got[mask.bool()], expected[mask.bool()], rtol=1e-4, atol=1e-5)
    loss = sum(layer[:, -3:].square().mean() for layer in actual["hidden_states"][1:])
    loss.backward()
    embedding_grad = interface.model.get_input_embeddings().weight.grad
    assert torch.isfinite(embedding_grad).all()
    assert embedding_grad[42].abs().sum() > 0
    assert interface.model.model.language_model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


def test_future_attention_and_alignment_validation():
    interface = tiny_interface()
    ids = torch.tensor([[5] + [42] * 16 + [44] + [43] * 3 + [45]])
    mask = torch.ones_like(ids)
    full = interface._build_future_attention_mask(mask, ids)[0, 0]
    assert full[1, 16] == 0
    assert full[1, 17] < 0
    bad = ids.repeat(2, 1)
    bad[1, 17], bad[1, 18] = bad[1, 18].clone(), bad[1, 17].clone()
    try:
        interface.forward_latent(input_ids=bad, attention_mask=torch.ones_like(bad))
    except ValueError as exc:
        assert "aligned" in str(exc)
    else:
        raise AssertionError("unaligned thinking tokens accepted")


def test_image_aware_positions_and_visual_gradient():
    torch.manual_seed(3)
    interface = tiny_interface().eval()
    ids = torch.tensor([[0, 5, 62, 60, 63, 6] + [42] * 16 + [44] + [43] * 3 + [45],
                        [7, 5, 62, 60, 63, 6] + [42] * 16 + [44] + [43] * 3 + [45]])
    mask = ids.ne(0).long()
    grids = torch.tensor([[1, 2, 2], [1, 2, 2]])
    pixels = torch.randn(8, 24)
    result = interface.forward_latent(input_ids=ids, attention_mask=mask,
                                      pixel_values=pixels, image_grid_thw=grids)
    positions, _ = interface.model.model.get_rope_index(ids, grids, None, attention_mask=mask)
    full_mask = interface._build_future_attention_mask(mask, ids)
    embeds = interface.model.get_input_embeddings()(ids)
    first = int((ids[0] == 43).nonzero()[0])
    for index in range(3):
        out = interface.model.model(inputs_embeds=embeds, attention_mask=full_mask,
                                    position_ids=positions, pixel_values=pixels, image_grid_thw=grids,
                                    output_hidden_states=True, use_cache=False, return_dict=True)
        latent = out.hidden_states[-1][:, first + index - 1]
        torch.testing.assert_close(result["latent_embeds"][:, index], latent, rtol=1e-4, atol=1e-5)
        embeds = torch.cat([embeds[:, :first + index], latent[:, None], embeds[:, first + index + 1:]], dim=1)
    result["latent_embeds"].square().mean().backward()
    grad = interface.model.model.visual.patch_embed.proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_decoder_targets_match_thinking_order():
    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.ones(len(texts), 2, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), 2, dtype=torch.long)}

    examples = [{"cot_subtask": "reach", "cot_reasoning": "pick it", "bbox_valid": True,
                 "bbox": [0.1, 0.2, 0.3, 0.4]}]
    latent = torch.randn(1, 3, 32, requires_grad=True)
    result = build_latent_decoder_batch(examples, latent, torch.ones(1, 3, dtype=torch.bool),
                                        Tokenizer(), ("SUBTASK", "BBOX", "REASON"))
    assert result.tags == ["SUBTASK", "BBOX", "REASON"]
    torch.testing.assert_close(result.latent_prefix[:, 0], latent[0])
    result.latent_prefix.sum().backward()
    assert torch.equal(latent.grad, torch.ones_like(latent))


def test_thinking_prompt_keeps_original_qwenpi_instruction():
    from types import SimpleNamespace
    from omegaconf import OmegaConf

    class Batch(dict):
        def to(self, device):
            return self

    class Processor:
        def __init__(self):
            self.tokenizer = SimpleNamespace(padding_side="right")
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return Batch()

    interface = tiny_interface()
    interface.processor = Processor()
    interface.config = OmegaConf.create({"datasets": {"vla_data": {}}})
    interface.future_token = "<future>"
    interface.thinking_token = "<|thinking|>"
    interface.start_thinking_token = "<|start_of_thinking|>"
    interface.end_thinking_token = "<|end_of_thinking|>"
    images = [[object(), object()]]
    interface.build_qwenvl_inputs(images, ["pick up the object."])
    text = interface.processor.messages[0][0]["content"][-1]["text"]
    expected_base = "pick up the object. " + " ".join(["<future>"] * 16)
    expected_thinking = " <|start_of_thinking|>" + "<|thinking|>" * 3 + "<|end_of_thinking|>"
    assert text == expected_base + expected_thinking
    assert interface.processor.tokenizer.padding_side == "left"

    interface.enable_latent_reasoning = False
    interface.build_qwenvl_inputs(images, ["pick up the object."])
    assert interface.processor.messages[0][0]["content"][-1]["text"] == expected_base
    assert interface.processor.tokenizer.padding_side == "right"


def test_old_checkpoint_vocab_expansion_preserves_existing_rows():
    interface = tiny_interface()
    checkpoint = {key: value.clone() for key, value in interface.state_dict().items()}
    keys = ["model.model.language_model.embed_tokens.weight", "model.lm_head.weight"]
    new_rows = {key: checkpoint[key][-3:].clone() for key in keys}
    for key in keys:
        checkpoint[key] = checkpoint[key][:-3].clone()
    interface.load_state_dict(checkpoint, strict=False)
    loaded = interface.state_dict()
    for key in keys:
        torch.testing.assert_close(loaded[key][:-3], checkpoint[key])
        torch.testing.assert_close(loaded[key][-3:], new_rows[key])


def test_stage1_dataset_packs_three_annotation_targets():
    from types import SimpleNamespace
    import numpy as np
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

    class Annotations:
        def get_step_cot(self, episode, frame):
            return SimpleNamespace(subtask="reach", reasoning="toward it")
        def get_step_bbox(self, episode, frame):
            return SimpleNamespace(valid=True, bbox=np.array([0.1, 0.2, 0.3, 0.4]))
        def get_step_bbox2(self, episode, frame):
            return None

    dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
    dataset.data_cfg = {"image_size": 32}
    dataset._modality_keys = {"video": ["video.base"], "language": ["annotation.task"], "action": ["action.arm"]}
    dataset.include_future = False
    dataset.thinking_annotations = Annotations()
    dataset.tag = "test"
    data = {"video.base": np.zeros((1, 32, 32, 3), dtype=np.uint8),
            "annotation.task": ["move"], "action.arm": np.zeros((8, 7))}
    sample = dataset._pack_sample(data, 2, 3)
    assert sample["lang"] == "move" and sample["cot_subtask"] == "reach"
    assert sample["cot_reasoning"] == "toward it" and sample["bbox_valid"]
    assert sample["episode_index"] == 2 and sample["frame_index"] == 3


def test_decoder_language_loss_reaches_recursive_thinking():
    torch.manual_seed(7)
    interface = tiny_interface().train()
    ids = torch.tensor([[5, 6] + [42] * 16 + [44] + [43] * 3 + [45]])
    result = interface.forward_latent(input_ids=ids, attention_mask=torch.ones_like(ids))
    decoder_model = tiny_interface().model
    decoder_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    decoder = FullQwenLatentTextDecoder(decoder_model, hidden_size=32, vocab_size=64,
                                        micro_batch_size=2, use_forward_checkpoint=True)
    targets = torch.tensor([[3, 4, 2], [6, 7, 2], [9, 10, 2]])
    decoder_output = decoder(latent_prefix=result["latent_embeds"].reshape(3, 1, 32),
                             target_input_ids=targets, target_labels=targets,
                             target_attention_mask=torch.ones_like(targets))
    decoder_output["loss"].backward()
    assert interface.model.get_input_embeddings().weight.grad[42].abs().sum() > 0
    assert decoder.model.lm_head.weight.grad.abs().sum() > 0


def test_real_processor_train_inference_prompt_consistency():
    model_path = os.environ.get("QWENPI_THINKING_PROCESSOR")
    if not model_path:
        return
    from omegaconf import OmegaConf
    from PIL import Image

    interface = tiny_interface().eval()
    cfg = interface.model.config
    cfg.image_token_id, cfg.video_token_id = 151655, 151656
    cfg.vision_start_token_id, cfg.vision_end_token_id = 151652, 151653
    cfg.text_config.vocab_size = 153984
    cfg.vision_config.patch_size = 16
    interface.model = Qwen3VLForConditionalGeneration(cfg).eval()
    interface.processor = AutoProcessor.from_pretrained(model_path)
    interface.config = OmegaConf.create({"datasets": {"vla_data": {}}})
    interface.future_token = "<future>"
    interface.future_token_id = interface._add_future_token(interface.processor.tokenizer)
    interface.thinking_token = "<|thinking|>"
    interface.start_thinking_token = "<|start_of_thinking|>"
    interface.end_thinking_token = "<|end_of_thinking|>"
    interface.thinking_token_id = interface._add_thinking_tokens(interface.processor.tokenizer)
    views = [[Image.new("RGB", (32, 32), (40, 50, 60)), Image.new("RGB", (32, 32), (1, 2, 3))]]
    batch = interface.build_qwenvl_inputs(views * 2, ["short", "a substantially longer instruction"])
    assert torch.equal((batch["input_ids"] == interface.thinking_token_id).nonzero()[:3, 1],
                       (batch["input_ids"] == interface.thinking_token_id).nonzero()[3:, 1])
    with torch.no_grad():
        batched = interface.forward_latent(**batch)
        one = interface.build_qwenvl_inputs(views, ["short"])
        single = interface.forward_latent(**one)
    torch.testing.assert_close(batched["latent_embeds"][0], single["latent_embeds"][0], rtol=2e-4, atol=2e-5)


def test_two_stage_framework_routes_losses_and_action():
    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image
    from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3

    class Tokenizer:
        eos_token_id, pad_token_id = 2, 0
        def __call__(self, texts, **kwargs):
            ids = torch.tensor([[3, 4]] * len(texts))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class ActionHead(torch.nn.Module):
        def forward(self, states, actions, state, encoder_attention_mask):
            assert len(states) == 2 and encoder_attention_mask.shape == states[0].shape[:2]
            return sum(x[:, -3:].square().mean() for x in states) + actions.square().mean()
        def predict_action(self, states, state, encoder_attention_mask):
            return states[-1].new_zeros((states[-1].shape[0], 8, 7))

    model = Qwen_PI_v3.__new__(Qwen_PI_v3)
    torch.nn.Module.__init__(model)
    model.config = OmegaConf.create({"framework": {"future": {"enable": True, "loss_weight": 0.1},
        "latent_reasoning": {"decoder_loss_weight": 1.0, "max_target_tokens": 8}},
        "datasets": {"vla_data": {}}, "trainer": {"repeated_diffusion_steps": 1}})
    model.qwen_vl_interface = tiny_interface()
    ids = torch.tensor([[5, 6] + [42] * 16 + [44] + [43] * 3 + [45]])
    model.qwen_vl_interface.build_qwenvl_inputs = lambda images, instructions: {
        "input_ids": ids.repeat(len(images), 1), "attention_mask": torch.ones(len(images), ids.shape[1], dtype=torch.long)}
    model.num_action_dit_layers = 2
    model.project_layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    model.spatial_forcing_enabled = False
    model.action_horizon = 8
    model.action_model = ActionHead()
    model._compute_future_loss = lambda raw_last_hidden, **kwargs: raw_last_hidden.square().mean()
    model._latent_decoder_tokenizer = Tokenizer()
    model.latent_decoder = FullQwenLatentTextDecoder(tiny_interface().model, 32, 64,
                                                     use_forward_checkpoint=False)
    example = {"image": [Image.new("RGB", (32, 32))], "lang": "task",
               "action": np.zeros((8, 7), dtype=np.float32), "cot_subtask": "reach",
               "cot_reasoning": "go", "bbox": [0.1, 0.2, 0.3, 0.4], "bbox_valid": True}
    model.train()
    model.latent_stage = 1
    stage1 = model([example])
    assert stage1["action_loss"].item() == 0 and stage1["decoder_loss"].item() > 0
    stage1["total_loss"].backward()
    assert model.qwen_vl_interface.model.get_input_embeddings().weight.grad[42].abs().sum() > 0
    model.zero_grad()
    model.latent_stage = 2
    model.latent_decoder = None
    stage2 = model([example])
    assert "decoder_loss" not in stage2 and stage2["action_loss"].item() > 0
    stage2["total_loss"].backward()
    assert model.qwen_vl_interface.model.get_input_embeddings().weight.grad[42].abs().sum() > 0
    model.eval()
    assert model.predict_action([example])["normalized_actions"].shape == (1, 8, 7)
