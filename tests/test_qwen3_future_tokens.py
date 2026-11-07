import torch
import torch.nn as nn

from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface


def build_interface():
    interface = _QWen3_VL_Interface.__new__(_QWen3_VL_Interface)
    nn.Module.__init__(interface)
    interface.enable_future = True
    interface.future_token = "<future>"
    interface.future_token_count = 16
    interface.future_token_id = 99
    return interface


def test_future_prompt_contract():
    interface = build_interface()
    prompt = interface._append_future_tokens("pick up the mug.")
    assert prompt == "pick up the mug. " + " ".join(["<future>"] * 16)


def test_future_attention_is_bidirectional_only_inside_future_block():
    interface = build_interface()
    input_ids = torch.tensor([[1, 2] + [99] * 16])
    attention_mask = torch.ones_like(input_ids)
    mask = interface._build_future_attention_mask(attention_mask, input_ids)
    assert mask.shape == (1, 1, 18, 18)
    assert mask[0, 0, 2, 17].item() == 0.0
    assert mask[0, 0, 0, 17].item() < -1.0e20
