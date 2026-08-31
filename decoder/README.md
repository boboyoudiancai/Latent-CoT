# Latent Text Decoder

SIM-CoT-style auxiliary decoder experiment for LaRA-VLA latent reasoning.

It does not modify the main LaRA-VLA training code. The script loads a trained
latent VLM checkpoint and jointly trains the VLM plus a one-block causal
Transformer decoder. The decoder reconstructs explicit reasoning component text
from hidden states at `<|thinking|>` positions, while the VLM can continue to
receive action-token language loss after the latent span.

## Train

Example after the final latent VLM checkpoint exists:

```bash
cd /home/liuyue/LaRA-VLA

CHECKPOINT=/path/to/final_latent_vlm/checkpoints/steps_2000_pytorch_model.pt \
OUTPUT_DIR=/home/liuyue/LaRA-VLA/playground/Checkpoints/Text2Latent/Text2Latent_decoder \
bash decoder/train.sh
```

`decoder/train.sh` launches `accelerate launch` with
`decoder/accelerate_deepspeed_zero2.yaml` by default. Override knobs with
environment variables, for example:

```bash
NUM_PROCESSES=4 MAX_TRAIN_STEPS=2000 PER_DEVICE_BATCH_SIZE=2 bash decoder/train.sh
```

Use the SIM-CoT-style full-model auxiliary decoder with:

```bash
DECODER_TYPE=full_vlm bash decoder/train.sh
```

The command used for a run is saved to `output_dir/run_command.sh`. Evaluation
samples and losses are appended to `output_dir/eval_samples.jsonl`.

## What Is Trained

Trainable by default:

- VLM parameters loaded from `--checkpoint`
- `one_block` decoder parameters, or the full copied Qwen decoder when
  `DECODER_TYPE=full_vlm`

Reused from Qwen:

- tokenizer
- input embedding and `lm_head` for `one_block`

`--freeze_vlm` exists only as an ablation switch.

## Target Mapping

The decoder pairs the current VLM latent sequence
with explicit component targets one by one:

```text
latent[0] -> target[0]
latent[1] -> target[1]
latent[2] -> target[2]
```

The target order follows `bridge_reasoning.component_order`, normally `BBOX`,
`SUBTASK`, `REASON`. If a sample has fewer valid targets or fewer valid latents,
only the aligned prefix is used.


## Loss

The main VLM-side supervision here is action-token language loss, not a generic VLM reasoning loss. Current total loss is:

```text
total_loss = decoder_loss_weight * decoder_loss + action_token_loss_weight * action_token_loss
```

`action_token_loss` comes from the existing ECoT label mask: instruction and latent thinking span are masked, and post-latent tokens remain supervised. With the final latent VLM and image-next disabled, those post-latent tokens are the action-token text.

## Validation

Every `--eval_interval` optimizer steps, the script reports
`eval_total_loss`, `eval_decoder_loss`, `eval_action_token_loss`, and decoded
samples as JSONL. Each sample contains:

- `tag`
- `decoded`
- `target`

The decoded text is produced by greedy decoding through the auxiliary decoder using the same Qwen tokenizer, input embedding, and `lm_head` as the base VLM.
