"""Jointly train LaRA-VLA latent reasoning with a one-block text decoder.

This script is intentionally isolated under decoder/. It loads an existing
latent VLM checkpoint and trains it with an auxiliary SIM-CoT-style decoder.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from accelerate import Accelerator, DeepSpeedPlugin
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decoder.latent_text_decoder import (  # noqa: E402
    FullQwenLatentTextDecoder,
    OneBlockLatentTextDecoder,
    build_latent_decoder_batch,
    trainable_parameter_count,
)
from laravla.dataloader import build_dataloader  # noqa: E402
from laravla.model.framework import build_framework  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone latent text decoder")
    parser.add_argument("--checkpoint", required=True, help="Path to latent VLM checkpoint .pt")
    parser.add_argument("--output_dir", required=True, help="Directory for decoder checkpoints/logs")
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--decoder_learning_rate", type=float, default=1e-4)
    parser.add_argument("--vlm_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--logging_frequency", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=2)
    parser.add_argument("--eval_samples", type=int, default=4)
    parser.add_argument("--eval_decode_tokens", type=int, default=64)
    parser.add_argument("--decoder_loss_weight", type=float, default=1.0)
    parser.add_argument("--action_token_loss_weight", type=float, default=None)
    parser.add_argument("--freeze_vlm", action="store_true", help="Ablation only: train decoder without updating VLM")
    parser.add_argument(
        "--decoder_type",
        default="one_block",
        choices=["one_block", "full_vlm"],
        help="Auxiliary decoder architecture. full_vlm copies the full Qwen LM like SIM-CoT.",
    )
    parser.add_argument("--decoder_num_heads", type=int, default=8)
    parser.add_argument("--decoder_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--decoder_dropout", type=float, default=0.0)
    parser.add_argument("--max_target_tokens", type=int, default=128)
    parser.add_argument("--max_position_embeddings", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default=None, help="Override Qwen attention backend, e.g. sdpa")
    parser.add_argument("--data_root_dir", default=None)
    parser.add_argument("--data_mix", default=None)
    parser.add_argument("--steps_cache_path", default=None)
    parser.add_argument("--write_steps_cache", default=None, choices=["true", "false"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin())
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        save_run_command(output_dir)
    accelerator.wait_for_everyone()

    device = accelerator.device if torch.cuda.is_available() else torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    cfg = load_checkpoint_config(Path(args.checkpoint))
    cfg = apply_runtime_overrides(cfg, args)
    cfg.output_dir = str(output_dir)
    if args.action_token_loss_weight is None:
        args.action_token_loss_weight = 1.0

    if accelerator.is_main_process:
        OmegaConf.save(cfg, output_dir / "config.yaml")
        OmegaConf.save(cfg, output_dir / "vlm_config_used.yaml")
        with open(output_dir / "train_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    accelerator.wait_for_everyone()

    vlm_model = build_vlm(cfg, Path(args.checkpoint), device=device, dtype=dtype, freeze_vlm=args.freeze_vlm)
    qwen_iface = vlm_model.qwen_vl_interface
    tokenizer = qwen_iface.processor.tokenizer
    qwen_model = qwen_iface.model
    hidden_size = int(qwen_model.config.hidden_size)
    vocab_size = int(qwen_model.lm_head.out_features if hasattr(qwen_model.lm_head, "out_features") else len(tokenizer))

    decoder = build_decoder(
        args=args,
        qwen_model=qwen_model,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        device=device,
        dtype=dtype,
    )

    training_module = SimCoTTrainingModule(
        vlm_model=vlm_model,
        decoder=decoder,
        tokenizer=tokenizer,
        cfg=cfg,
        args=args,
        dtype=dtype,
    )

    optimizer = torch.optim.AdamW(
        build_optimizer_groups(training_module, args),
        weight_decay=args.weight_decay,
    )
    warmup_steps = int(args.max_train_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    training_module, optimizer, scheduler, dataloader = accelerator.prepare(
        training_module,
        optimizer,
        scheduler,
        dataloader,
    )
    data_iter = iter(dataloader)

    if accelerator.is_main_process:
        accelerator.print(accelerator.state)
        accelerator.print(f"Loaded VLM checkpoint: {args.checkpoint}")
        base_module = accelerator.unwrap_model(training_module)
        accelerator.print(f"VLM trainable parameters: {trainable_parameter_count(base_module.vlm_model):,}")
        accelerator.print(f"Decoder trainable parameters: {trainable_parameter_count(base_module.decoder):,}")
        accelerator.print(f"Decoder type: {args.decoder_type}")
        accelerator.print(f"Shared tokenizer vocab size: {len(tokenizer)}")
        accelerator.print(f"Decoder logits vocab size: {vocab_size}")
        accelerator.print("Decoder and base VLM use the same tokenizer/embedding/lm_head references.")
        accelerator.print(f"Output dir: {output_dir}")

    completed_steps = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_train_steps, disable=not accelerator.is_main_process)

    while completed_steps < args.max_train_steps:
        accum_loss_dict: Dict[str, torch.Tensor] | None = None
        usable_micro_batches = 0
        for micro_step in range(args.gradient_accumulation_steps):
            try:
                examples = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                examples = next(data_iter)

            loss_dict = training_module(examples)
            if loss_dict is None:
                continue
            loss = loss_dict["total_loss"]
            accelerator.backward(loss / args.gradient_accumulation_steps)
            usable_micro_batches += 1
            accum_loss_dict = merge_loss_dicts(accum_loss_dict, loss_dict)

        if usable_micro_batches == 0 or accum_loss_dict is None:
            continue

        if args.gradient_clipping and args.gradient_clipping > 0:
            accelerator.clip_grad_norm_(
                [p for p in training_module.parameters() if p.requires_grad],
                args.gradient_clipping,
            )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        completed_steps += 1
        progress.update(1)

        metrics = None
        if completed_steps % args.logging_frequency == 0:
            averaged_loss_dict = average_loss_dict(accum_loss_dict, usable_micro_batches)
            metrics = gather_scalar_metrics(accelerator, averaged_loss_dict)
        if accelerator.is_main_process and metrics is not None:
            lr = scheduler.get_last_lr()[0]
            log_data = {
                "step": completed_steps,
                "loss": metrics.get("total_loss"),
                "learning_rate": lr,
            }
            for key, value in metrics.items():
                if key.endswith("_loss"):
                    log_data[key] = value
            accelerator.print(json.dumps(log_data))

        if accelerator.is_main_process and args.eval_interval > 0 and completed_steps % args.eval_interval == 0:
            eval_data = run_decoder_eval(
                module=accelerator.unwrap_model(training_module),
                dataloader=dataloader,
                args=args,
                step=completed_steps,
                output_dir=output_dir,
            )
            accelerator.print(json.dumps(eval_data, ensure_ascii=False))
        if args.eval_interval > 0 and completed_steps % args.eval_interval == 0:
            accelerator.wait_for_everyone()

        if args.save_interval > 0 and completed_steps % args.save_interval == 0:
            save_joint_checkpoint(training_module, accelerator, output_dir, completed_steps, args, cfg)

    save_joint_checkpoint(training_module, accelerator, output_dir, completed_steps, args, cfg, final=True)
    progress.close()
    accelerator.wait_for_everyone()


class SimCoTTrainingModule(nn.Module):
    def __init__(
        self,
        vlm_model: nn.Module,
        decoder: nn.Module,
        tokenizer,
        cfg,
        args: argparse.Namespace,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.vlm_model = vlm_model
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.args = args
        self.dtype = dtype

    def forward(self, examples: Sequence[dict]) -> Dict[str, torch.Tensor] | None:
        decoder_batch, outputs = self.build_decoder_batch(examples)
        if decoder_batch is None:
            return None

        autocast_enabled = self.dtype in {torch.float16, torch.bfloat16}
        with torch.autocast("cuda", dtype=self.dtype, enabled=autocast_enabled):
            decoder_out = self.decoder(
                latent_prefix=decoder_batch.latent_prefix.to(dtype=self.dtype),
                latent_prefix_attention_mask=decoder_batch.latent_prefix_attention_mask,
                target_input_ids=decoder_batch.target_input_ids,
                target_labels=decoder_batch.target_labels,
                target_attention_mask=decoder_batch.target_attention_mask,
            )

        decoder_loss = decoder_out["loss"]
        action_token_loss = outputs.get("loss")
        total_loss = self.args.decoder_loss_weight * decoder_loss
        if (
            action_token_loss is not None
            and self.args.action_token_loss_weight
            and self.args.action_token_loss_weight > 0
        ):
            total_loss = total_loss + self.args.action_token_loss_weight * action_token_loss

        result = {
            "total_loss": total_loss,
            "decoder_loss": decoder_loss,
        }
        if action_token_loss is not None:
            result["action_token_loss"] = action_token_loss
        return result

    def build_decoder_batch(self, examples: Sequence[dict]):
        qwen_iface = self.vlm_model.qwen_vl_interface
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        action_tokens = [example.get("action_tokens", "") for example in examples]
        qwen_inputs = qwen_iface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            solutions=None,
            action_tokens=action_tokens,
        )
        outputs = qwen_iface.forward_latent(
            input_ids=qwen_inputs["input_ids"],
            attention_mask=qwen_inputs["attention_mask"],
            pixel_values=qwen_inputs.get("pixel_values"),
            image_grid_thw=qwen_inputs.get("image_grid_thw"),
            labels=qwen_inputs.get("labels"),
            return_latent_embeds=True,
        )

        bridge_cfg = self.cfg.datasets.vla_data.bridge_reasoning
        component_order = normalize_component_order(getattr(bridge_cfg, "component_order", None))
        decoder_batch = build_latent_decoder_batch(
            examples=examples,
            latent_embeds=outputs["latent_embeds"],
            latent_attention_mask=outputs["latent_attention_mask"],
            tokenizer=self.tokenizer,
            component_order=component_order,
            include_bbox=bool(getattr(bridge_cfg, "include_bbox", True)),
            max_target_tokens=self.args.max_target_tokens,
        )
        return decoder_batch, outputs


def load_checkpoint_config(checkpoint: Path):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    run_dir = checkpoint.parents[1]
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.yaml beside checkpoint run dir: {config_path}")
    return OmegaConf.load(config_path)


def apply_runtime_overrides(cfg, args: argparse.Namespace):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.trainer.pretrained_checkpoint = None
    cfg.datasets.vla_data.per_device_batch_size = args.per_device_batch_size
    if args.data_root_dir is not None:
        cfg.datasets.vla_data.data_root_dir = args.data_root_dir
    if args.data_mix is not None:
        cfg.datasets.vla_data.data_mix = args.data_mix
    if args.steps_cache_path is not None:
        cfg.datasets.vla_data.bridge_annotations.steps_cache_path = args.steps_cache_path
    if args.write_steps_cache is not None:
        cfg.datasets.vla_data.bridge_annotations.write_steps_cache = args.write_steps_cache == "true"
    if args.attn_implementation is not None:
        cfg.framework.qwenvl.attn_implementation = args.attn_implementation
    cfg.framework.enable_latent_reasoning = True
    cfg.framework.img_next.enable = False
    cfg.framework.img_next.loss_weight = 0
    cfg.framework.img_next.use_teacher = False
    cfg.datasets.vla_data.bridge_reasoning.include_img_next = False
    return cfg


def build_vlm(cfg, checkpoint: Path, device: torch.device, dtype: torch.dtype, freeze_vlm: bool):
    model = build_framework(cfg)
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    try:
        model.to(device=device, dtype=dtype)
    except ValueError as exc:
        if "device_map" not in str(exc):
            raise
    model.train()
    freeze_unused_action_head(model)
    if freeze_vlm:
        for param in model.qwen_vl_interface.parameters():
            param.requires_grad = False
    else:
        for param in model.qwen_vl_interface.parameters():
            param.requires_grad = True
    return model


def build_decoder(
    args: argparse.Namespace,
    qwen_model: nn.Module,
    hidden_size: int,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    if args.decoder_type == "one_block":
        return OneBlockLatentTextDecoder(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            qwen_input_embeddings=qwen_model.get_input_embeddings(),
            qwen_lm_head=qwen_model.lm_head,
            num_heads=args.decoder_num_heads,
            mlp_ratio=args.decoder_mlp_ratio,
            dropout=args.decoder_dropout,
            max_position_embeddings=args.max_position_embeddings,
            freeze_shared_io=args.freeze_vlm,
        ).to(device=device, dtype=dtype)
    if args.decoder_type == "full_vlm":
        return FullQwenLatentTextDecoder(
            qwen_model=qwen_model,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        ).to(device=device, dtype=dtype)
    raise ValueError(f"unsupported decoder_type: {args.decoder_type}")


def freeze_unused_action_head(model: nn.Module) -> None:
    action_head = getattr(model, "action_model", None)
    if action_head is None:
        return
    for param in action_head.parameters():
        param.requires_grad = False


def build_optimizer_groups(module: SimCoTTrainingModule, args: argparse.Namespace) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    vlm_params = [p for p in module.vlm_model.parameters() if p.requires_grad]
    decoder_params = [p for p in module.decoder.parameters() if p.requires_grad]
    if vlm_params:
        groups.append({"name": "vlm", "params": vlm_params, "lr": args.vlm_learning_rate})
    if decoder_params:
        groups.append({"name": "decoder", "params": decoder_params, "lr": args.decoder_learning_rate})
    if not groups:
        raise ValueError("No trainable parameters found")
    return groups


def merge_loss_dicts(
    current: Dict[str, torch.Tensor] | None,
    update: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if current is None:
        return {
            key: value.detach()
            for key, value in update.items()
            if torch.is_tensor(value)
        }
    for key, value in update.items():
        if torch.is_tensor(value):
            current[key] = current.get(key, value.detach().new_zeros(())) + value.detach()
    return current


def average_loss_dict(loss_dict: Dict[str, torch.Tensor], count: int) -> Dict[str, torch.Tensor]:
    denom = max(1, int(count))
    return {key: value / denom for key, value in loss_dict.items()}


def gather_scalar_metrics(accelerator: Accelerator, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in loss_dict.items():
        if not torch.is_tensor(value):
            continue
        scalar = value.detach().float().reshape(1).to(accelerator.device)
        gathered = accelerator.gather_for_metrics(scalar)
        metrics[key] = float(gathered.mean().cpu())
    return metrics


@torch.no_grad()
def run_decoder_eval(
    module: SimCoTTrainingModule,
    dataloader,
    args: argparse.Namespace,
    step: int,
    output_dir: Path,
) -> Dict[str, Any]:
    was_training = module.training
    module.eval()
    total_losses: List[float] = []
    decoder_losses: List[float] = []
    action_token_losses: List[float] = []
    samples: List[Dict[str, Any]] = []
    data_iter = iter(dataloader)

    for _ in range(max(1, args.eval_batches)):
        try:
            examples = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            examples = next(data_iter)

        decoder_batch, outputs = module.build_decoder_batch(examples)
        if decoder_batch is None:
            continue

        decoder_out = module.decoder(
            latent_prefix=decoder_batch.latent_prefix.to(dtype=module.dtype),
            latent_prefix_attention_mask=decoder_batch.latent_prefix_attention_mask,
            target_input_ids=decoder_batch.target_input_ids,
            target_labels=decoder_batch.target_labels,
            target_attention_mask=decoder_batch.target_attention_mask,
        )
        decoder_loss = decoder_out["loss"]
        total_loss = args.decoder_loss_weight * decoder_loss
        action_token_loss = outputs.get("loss")
        if (
            action_token_loss is not None
            and args.action_token_loss_weight
            and args.action_token_loss_weight > 0
        ):
            total_loss = total_loss + args.action_token_loss_weight * action_token_loss
            action_token_losses.append(float(action_token_loss.detach().cpu()))
        decoder_losses.append(float(decoder_loss.detach().cpu()))
        total_losses.append(float(total_loss.detach().cpu()))

        if len(samples) < args.eval_samples:
            bos_id = module.tokenizer.bos_token_id
            if bos_id is None:
                bos_id = module.tokenizer.eos_token_id
            if bos_id is None:
                bos_id = module.tokenizer.pad_token_id
            eos_id = module.tokenizer.eos_token_id
            pad_id = module.tokenizer.pad_token_id
            decoded_ids = module.decoder.greedy_decode(
                latent_prefix=decoder_batch.latent_prefix.to(dtype=module.dtype),
                latent_prefix_attention_mask=decoder_batch.latent_prefix_attention_mask,
                bos_token_id=int(bos_id),
                eos_token_id=int(eos_id) if eos_id is not None else None,
                pad_token_id=int(pad_id) if pad_id is not None else None,
                max_new_tokens=args.eval_decode_tokens,
            )
            decoded_texts = module.tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)
            for idx, text in enumerate(decoded_texts):
                if len(samples) >= args.eval_samples:
                    break
                samples.append(
                    {
                        "tag": decoder_batch.tags[idx],
                        "decoded": clean_text(text),
                        "target": decoder_batch.target_texts[idx],
                    }
                )

    if was_training:
        module.train()

    eval_total_loss = mean_or_none(total_losses)
    eval_decoder_loss = mean_or_none(decoder_losses)
    eval_action_token_loss = mean_or_none(action_token_losses)
    record = {
        "step": step,
        "eval_total_loss": eval_total_loss,
        "eval_decoder_loss": eval_decoder_loss,
        "eval_action_token_loss": eval_action_token_loss,
        "num_eval_batches": len(total_losses),
        "samples": samples,
    }
    sample_path = output_dir / "eval_samples.jsonl"
    with open(sample_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def clean_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split())


def mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def normalize_component_order(value: Any) -> List[str]:
    if value is None:
        return ["BBOX", "SUBTASK", "REASON"]
    if isinstance(value, str):
        return [x.strip().upper() for x in value.split(",") if x.strip()]
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and "," in item:
            out.extend([x.strip().upper() for x in item.split(",") if x.strip()])
        else:
            out.append(str(item).strip().upper())
    return [x for x in out if x in {"BBOX", "SUBTASK", "REASON"}]


def save_joint_checkpoint(
    module: nn.Module,
    accelerator: Accelerator,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    cfg,
    final: bool = False,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    decoder_name = "final_decoder.pt" if final else f"steps_{step}_decoder.pt"
    vlm_name = "pytorch_model.pt" if final else f"steps_{step}_pytorch_model.pt"
    full_state = accelerator.get_state_dict(module)
    decoder_prefix = "decoder."
    vlm_prefix = "vlm_model."
    decoder_state = {
        key[len(decoder_prefix) :]: value.detach().cpu()
        for key, value in full_state.items()
        if key.startswith(decoder_prefix)
    }
    vlm_state = {
        key[len(vlm_prefix) :]: value.detach().cpu()
        for key, value in full_state.items()
        if key.startswith(vlm_prefix)
    }
    payload = {
        "step": step,
        "decoder_state_dict": decoder_state,
        "args": vars(args),
        "bridge_reasoning": OmegaConf.to_container(cfg.datasets.vla_data.bridge_reasoning, resolve=True),
    }
    if final:
        final_dir = output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, final_dir / decoder_name)
        torch.save(vlm_state, final_dir / vlm_name)
        saved_path = final_dir / decoder_name
    else:
        torch.save(payload, checkpoint_dir / decoder_name)
        torch.save(vlm_state, checkpoint_dir / vlm_name)
        saved_path = checkpoint_dir / decoder_name
    print(f"Saved joint checkpoint at step {step}: {saved_path}")


def save_run_command(output_dir: Path) -> None:
    argv = list(sys.argv)
    launch_args = os.environ.get("ACCELERATE_LAUNCH_ARGS", "").strip()
    if launch_args:
        cmd = " ".join(["accelerate", "launch", launch_args, shlex.quote(argv[0])] + [shlex.quote(x) for x in argv[1:]])
    else:
        cmd = " ".join([shlex.quote(sys.executable)] + [shlex.quote(x) for x in argv])
    path = output_dir / "run_command.sh"
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + cmd + " \"$@\"\n")
    try:
        path.chmod(0o755)
    except OSError:
        pass


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


if __name__ == "__main__":
    main()
