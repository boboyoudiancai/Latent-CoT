"""Read-only check that the pre-thinking checkpoint can expand by three rows."""
import argparse
import torch
from transformers import AutoConfig, AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    tokenizer = AutoProcessor.from_pretrained(args.model).tokenizer
    config = AutoConfig.from_pretrained(args.model)
    tokenizer.add_tokens(["<future>"], special_tokens=True)
    tokenizer.add_tokens(["<|thinking|>", "<|start_of_thinking|>", "<|end_of_thinking|>"], special_tokens=True)
    new_rows = max(len(tokenizer), config.text_config.vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    for key in ("qwen_vl_interface.model.model.language_model.embed_tokens.weight",
                "qwen_vl_interface.model.lm_head.weight"):
        tensor = checkpoint[key]
        extra = new_rows - tensor.shape[0]
        print(f"{key}: {tuple(tensor.shape)} -> ({new_rows}, {tensor.shape[1]}), extra={extra}")
        if extra not in range(4):
            raise ValueError(f"Unexpected vocabulary difference: {extra}")


if __name__ == "__main__":
    main()
