"""Small SIM-CoT-style decoder for LaRA-VLA latent reasoning states.

This module intentionally depends only on standard PyTorch plus tokenizer/model
interfaces already exposed by the Qwen VLM wrapper. It does not patch the main
LaRA-VLA framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100


@dataclass
class LatentDecoderBatch:
    latent_prefix: torch.Tensor
    latent_prefix_attention_mask: torch.Tensor
    target_input_ids: torch.Tensor
    target_labels: torch.Tensor
    target_attention_mask: torch.Tensor
    tags: List[str]
    target_texts: List[str]


class OneBlockLatentTextDecoder(nn.Module):
    """One-layer causal Transformer decoder with shared Qwen embeddings/head.

    The trainable part is intentionally small:
      latent states -> projection -> one Transformer encoder layer run with a
      causal mask. Target token embeddings and logits use Qwen's existing
      input embeddings and lm_head.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        qwen_input_embeddings: nn.Module,
        qwen_lm_head: nn.Module,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_position_embeddings: int = 512,
        freeze_shared_io: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")

        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        # Keep shared Qwen IO as references, not decoder submodules. This avoids
        # saving/broadcasting large frozen weights as part of the small decoder.
        object.__setattr__(self, "qwen_input_embeddings", qwen_input_embeddings)
        object.__setattr__(self, "qwen_lm_head", qwen_lm_head)

        if freeze_shared_io:
            for param in qwen_input_embeddings.parameters():
                param.requires_grad = False
            for param in qwen_lm_head.parameters():
                param.requires_grad = False

        self.latent_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.block = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        latent_prefix: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_labels: torch.Tensor,
        target_attention_mask: Optional[torch.Tensor] = None,
        latent_prefix_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict target labels from latent prefix plus teacher-forced tokens."""
        if latent_prefix.ndim != 3:
            raise ValueError(f"latent_prefix must be [B, P, H], got {tuple(latent_prefix.shape)}")
        if target_input_ids.ndim != 2:
            raise ValueError(f"target_input_ids must be [B, T], got {tuple(target_input_ids.shape)}")
        if target_labels.shape != target_input_ids.shape:
            raise ValueError(
                f"target_labels shape {tuple(target_labels.shape)} must match target_input_ids "
                f"{tuple(target_input_ids.shape)}"
            )

        batch_size, prefix_len, hidden_size = latent_prefix.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"latent hidden size {hidden_size} != decoder hidden size {self.hidden_size}")

        target_input_ids = target_input_ids.to(device=latent_prefix.device)
        target_labels = target_labels.to(device=latent_prefix.device)
        if target_attention_mask is None:
            target_attention_mask = target_input_ids.ne(0)
        target_attention_mask = target_attention_mask.to(device=latent_prefix.device, dtype=torch.bool)
        if latent_prefix_attention_mask is None:
            latent_prefix_attention_mask = torch.ones(
                (batch_size, prefix_len),
                device=latent_prefix.device,
                dtype=torch.bool,
            )
        else:
            latent_prefix_attention_mask = latent_prefix_attention_mask.to(
                device=latent_prefix.device,
                dtype=torch.bool,
            )

        prefix_valid_len = latent_prefix_attention_mask.long().sum(dim=1).clamp_min(1)
        target_len = target_input_ids.shape[1]
        target_logits = None
        per_sample_loss = latent_prefix.new_zeros((batch_size,), dtype=torch.float32)
        active_samples = torch.zeros((batch_size,), device=latent_prefix.device, dtype=torch.float32)

        for valid_len_tensor in torch.unique(prefix_valid_len, sorted=True):
            valid_len = int(valid_len_tensor.item())
            batch_indices = (prefix_valid_len == valid_len).nonzero(as_tuple=False).flatten()
            prefix = self.latent_proj(latent_prefix[batch_indices, :valid_len, :])
            token_embeds = self.qwen_input_embeddings(target_input_ids[batch_indices])
            hidden = torch.cat([prefix, token_embeds], dim=1)

            seq_len = hidden.shape[1]
            if seq_len > self.position_embedding.num_embeddings:
                raise ValueError(
                    f"sequence length {seq_len} exceeds max_position_embeddings="
                    f"{self.position_embedding.num_embeddings}"
                )

            pos = torch.arange(seq_len, device=hidden.device)
            hidden = hidden + self.position_embedding(pos).unsqueeze(0).to(dtype=hidden.dtype)

            prefix_mask = torch.ones(
                (batch_indices.numel(), valid_len),
                device=hidden.device,
                dtype=torch.bool,
            )
            key_padding_mask = ~torch.cat([prefix_mask, target_attention_mask[batch_indices]], dim=1)
            causal_mask = torch.triu(
                torch.ones((seq_len, seq_len), device=hidden.device, dtype=torch.bool),
                diagonal=1,
            )

            hidden = self.block(hidden, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
            hidden = self.final_norm(hidden)
            logits = self.qwen_lm_head(hidden[:, valid_len - 1 : -1, :])

            if target_logits is None:
                target_logits = logits.new_empty((batch_size, target_len, logits.shape[-1]))
            target_logits[batch_indices] = logits

            labels = target_labels[batch_indices]
            valid_mask = labels.ne(IGNORE_INDEX)
            if valid_mask.any():
                token_losses = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size).float(),
                    labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                ).view_as(labels)
                sample_loss = token_losses.sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)
                per_sample_loss[batch_indices] = sample_loss
                active_samples[batch_indices] = valid_mask.any(dim=1).to(dtype=torch.float32)

        if target_logits is None:
            target_logits = latent_prefix.new_empty((batch_size, target_len, self.vocab_size))

        if active_samples.any():
            loss = (per_sample_loss * active_samples).sum() / active_samples.sum().clamp_min(1.0)
        else:
            loss = target_logits.new_zeros(())

        return {"loss": loss, "logits": target_logits}

    @torch.no_grad()
    def greedy_decode(
        self,
        latent_prefix: torch.Tensor,
        bos_token_id: int,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        max_new_tokens: int = 64,
        latent_prefix_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Greedy decode text tokens from latent prefix for inspection."""
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        if pad_token_id is None:
            pad_token_id = eos_token_id if eos_token_id is not None else bos_token_id

        batch_size = latent_prefix.shape[0]
        device = latent_prefix.device
        generated = torch.empty((batch_size, 0), device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, device=device, dtype=torch.bool)
        new_tokens = []

        for _ in range(max_new_tokens):
            next_logits = self._next_token_logits(
                latent_prefix=latent_prefix,
                latent_prefix_attention_mask=latent_prefix_attention_mask,
                generated_ids=generated,
            )
            next_token = next_logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, int(pad_token_id)), next_token)
            new_tokens.append(next_token)
            if eos_token_id is not None:
                finished = finished | next_token.eq(int(eos_token_id))
                if bool(finished.all()):
                    break
            generated = torch.cat([generated, next_token[:, None]], dim=1)

        if not new_tokens:
            return torch.empty((batch_size, 0), device=device, dtype=torch.long)
        return torch.stack(new_tokens, dim=1)

    def _next_token_logits(
        self,
        latent_prefix: torch.Tensor,
        generated_ids: torch.Tensor,
        latent_prefix_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latent_prefix_attention_mask is None:
            latent_prefix_attention_mask = torch.ones(
                latent_prefix.shape[:2],
                device=latent_prefix.device,
                dtype=torch.bool,
            )
        else:
            latent_prefix_attention_mask = latent_prefix_attention_mask.to(
                device=latent_prefix.device,
                dtype=torch.bool,
            )

        prefix_valid_len = latent_prefix_attention_mask.long().sum(dim=1).clamp_min(1)
        next_logits = None
        for valid_len_tensor in torch.unique(prefix_valid_len, sorted=True):
            valid_len = int(valid_len_tensor.item())
            batch_indices = (prefix_valid_len == valid_len).nonzero(as_tuple=False).flatten()
            prefix = self.latent_proj(latent_prefix[batch_indices, :valid_len, :])
            if generated_ids.shape[1] > 0:
                token_embeds = self.qwen_input_embeddings(generated_ids[batch_indices])
                hidden = torch.cat([prefix, token_embeds], dim=1)
            else:
                hidden = prefix

            seq_len = hidden.shape[1]
            if seq_len > self.position_embedding.num_embeddings:
                raise ValueError(
                    f"sequence length {seq_len} exceeds max_position_embeddings="
                    f"{self.position_embedding.num_embeddings}"
                )
            pos = torch.arange(seq_len, device=hidden.device)
            hidden = hidden + self.position_embedding(pos).unsqueeze(0).to(dtype=hidden.dtype)
            causal_mask = torch.triu(
                torch.ones((seq_len, seq_len), device=hidden.device, dtype=torch.bool),
                diagonal=1,
            )
            hidden = self.block(hidden, src_mask=causal_mask)
            hidden = self.final_norm(hidden)
            logits = self.qwen_lm_head(hidden[:, -1, :])
            if next_logits is None:
                next_logits = logits.new_empty((latent_prefix.shape[0], logits.shape[-1]))
            next_logits[batch_indices] = logits

        if next_logits is None:
            next_logits = latent_prefix.new_empty((latent_prefix.shape[0], self.vocab_size))
        return next_logits


class FullQwenLatentTextDecoder(nn.Module):
    """SIM-CoT-style auxiliary decoder backed by a full Qwen LM copy."""

    def __init__(
        self,
        qwen_model: nn.Module,
        hidden_size: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.model = copy.deepcopy(qwen_model)
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.input_embeddings = self.model.get_input_embeddings()

    def forward(
        self,
        latent_prefix: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_labels: torch.Tensor,
        target_attention_mask: Optional[torch.Tensor] = None,
        latent_prefix_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if latent_prefix.ndim != 3:
            raise ValueError(f"latent_prefix must be [B, P, H], got {tuple(latent_prefix.shape)}")
        if target_input_ids.ndim != 2:
            raise ValueError(f"target_input_ids must be [B, T], got {tuple(target_input_ids.shape)}")
        if target_labels.shape != target_input_ids.shape:
            raise ValueError(
                f"target_labels shape {tuple(target_labels.shape)} must match target_input_ids "
                f"{tuple(target_input_ids.shape)}"
            )

        batch_size, prefix_len, hidden_size = latent_prefix.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"latent hidden size {hidden_size} != decoder hidden size {self.hidden_size}")

        device = latent_prefix.device
        target_input_ids = target_input_ids.to(device=device)
        target_labels = target_labels.to(device=device)
        if target_attention_mask is None:
            target_attention_mask = target_input_ids.ne(0)
        target_attention_mask = target_attention_mask.to(device=device, dtype=torch.bool)
        if latent_prefix_attention_mask is None:
            latent_prefix_attention_mask = torch.ones(
                (batch_size, prefix_len),
                device=device,
                dtype=torch.bool,
            )
        else:
            latent_prefix_attention_mask = latent_prefix_attention_mask.to(device=device, dtype=torch.bool)

        target_embeds = self.input_embeddings(target_input_ids)
        inputs_embeds = torch.cat([latent_prefix, target_embeds], dim=1)
        attention_mask = torch.cat([latent_prefix_attention_mask, target_attention_mask], dim=1)
        prefix_labels = target_labels.new_full((batch_size, prefix_len), IGNORE_INDEX)
        labels = torch.cat([prefix_labels, target_labels], dim=1)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits[:, prefix_len - 1 : -1, :]
        loss = outputs.loss if outputs.loss is not None else logits.new_zeros(())
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def greedy_decode(
        self,
        latent_prefix: torch.Tensor,
        bos_token_id: int,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        max_new_tokens: int = 64,
        latent_prefix_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        if pad_token_id is None:
            pad_token_id = eos_token_id if eos_token_id is not None else bos_token_id

        batch_size = latent_prefix.shape[0]
        device = latent_prefix.device
        generated = torch.empty((batch_size, 0), device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, device=device, dtype=torch.bool)
        new_tokens = []

        for _ in range(max_new_tokens):
            next_logits = self._next_token_logits(
                latent_prefix=latent_prefix,
                generated_ids=generated,
                latent_prefix_attention_mask=latent_prefix_attention_mask,
            )
            next_token = next_logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, int(pad_token_id)), next_token)
            new_tokens.append(next_token)
            if eos_token_id is not None:
                finished = finished | next_token.eq(int(eos_token_id))
                if bool(finished.all()):
                    break
            generated = torch.cat([generated, next_token[:, None]], dim=1)

        if not new_tokens:
            return torch.empty((batch_size, 0), device=device, dtype=torch.long)
        return torch.stack(new_tokens, dim=1)

    def _next_token_logits(
        self,
        latent_prefix: torch.Tensor,
        generated_ids: torch.Tensor,
        latent_prefix_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latent_prefix_attention_mask is None:
            latent_prefix_attention_mask = torch.ones(
                latent_prefix.shape[:2],
                device=latent_prefix.device,
                dtype=torch.bool,
            )
        else:
            latent_prefix_attention_mask = latent_prefix_attention_mask.to(
                device=latent_prefix.device,
                dtype=torch.bool,
            )

        if generated_ids.shape[1] > 0:
            token_embeds = self.input_embeddings(generated_ids)
            inputs_embeds = torch.cat([latent_prefix, token_embeds], dim=1)
            token_mask = torch.ones(
                generated_ids.shape,
                device=latent_prefix.device,
                dtype=torch.bool,
            )
            attention_mask = torch.cat([latent_prefix_attention_mask, token_mask], dim=1)
        else:
            inputs_embeds = latent_prefix
            attention_mask = latent_prefix_attention_mask

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits[:, -1, :]


def format_decoder_targets_from_example(
    example: dict,
    component_order: Sequence[str],
    include_bbox: bool = True,
) -> List[Dict[str, str]]:
    """Build explicit component texts in the same order as the latent span."""
    subtask = example.get("cot_subtask", "") or ""
    reasoning = example.get("cot_reasoning", "") or ""
    bbox_valid = bool(example.get("bbox_valid", False))
    bbox = example.get("bbox")
    bbox2_valid = bool(example.get("bbox2_valid", False))
    bbox2 = example.get("bbox2")

    out: List[Dict[str, str]] = []
    for raw_tag in component_order:
        tag = str(raw_tag).strip().upper()
        if tag == "SUBTASK" and subtask:
            out.append({"tag": tag, "text": f"Subtask: {subtask}."})
        elif tag == "REASON" and reasoning:
            out.append({"tag": tag, "text": f"Reasoning: {reasoning}"})
        elif tag == "BBOX" and include_bbox:
            boxes = []
            if bbox_valid and bbox is not None:
                boxes.append(_bbox_to_text(bbox))
            if bbox_valid and bbox2_valid and bbox2 is not None:
                boxes.append(_bbox_to_text(bbox2))
            if not boxes:
                boxes.append(_bbox_to_text([0.0, 0.0, 1.0, 1.0]))
            joined = " ".join(f"[{box}]" for box in boxes)
            out.append({"tag": tag, "text": f"BBox: {joined}."})
    return out


def build_latent_decoder_batch(
    examples: Sequence[dict],
    latent_embeds: torch.Tensor,
    latent_attention_mask: torch.Tensor,
    tokenizer,
    component_order: Sequence[str],
    include_bbox: bool = True,
    max_target_tokens: int = 128,
) -> Optional[LatentDecoderBatch]:
    """Pair each valid latent hidden state with one explicit target text.

    The final latent VLM emits one latent per reasoning component. Decoder
    supervision therefore follows the actual latent sequence returned by the VLM:
    latent[0] -> target[0], latent[1] -> target[1], etc.
    """
    device = latent_embeds.device

    latent_prefixes: List[torch.Tensor] = []
    latent_masks: List[torch.Tensor] = []
    target_texts: List[str] = []
    target_tags: List[str] = []

    for batch_idx, example in enumerate(examples):
        targets = format_decoder_targets_from_example(
            example=example,
            component_order=component_order,
            include_bbox=include_bbox,
        )
        if not targets:
            continue

        valid_positions = torch.nonzero(
            latent_attention_mask[batch_idx].to(dtype=torch.bool),
            as_tuple=False,
        ).flatten()
        for latent_pos, target in zip(valid_positions.tolist(), targets):
            latent_prefixes.append(latent_embeds[batch_idx, latent_pos : latent_pos + 1, :])
            latent_masks.append(latent_attention_mask[batch_idx, latent_pos : latent_pos + 1])
            target_texts.append(target["text"])
            target_tags.append(target["tag"])

    if not latent_prefixes:
        return None

    prefix_len = max(prefix.shape[0] for prefix in latent_prefixes)
    hidden_size = latent_embeds.shape[-1]
    latent_prefix = latent_embeds.new_zeros((len(latent_prefixes), prefix_len, hidden_size))
    latent_prefix_attention_mask = torch.zeros(
        (len(latent_prefixes), prefix_len),
        device=device,
        dtype=torch.bool,
    )
    for idx, (prefix, mask) in enumerate(zip(latent_prefixes, latent_masks)):
        latent_prefix[idx, : prefix.shape[0], :] = prefix
        latent_prefix_attention_mask[idx, : mask.shape[0]] = mask.to(device=device, dtype=torch.bool)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id

    tokenized = tokenizer(
        target_texts,
        padding=True,
        truncation=True,
        max_length=max_target_tokens + 1,
        add_special_tokens=False,
        return_tensors="pt",
    )
    target_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"].bool()

    if target_ids.shape[1] < 1:
        return None

    if eos_id is not None:
        target_ids, attention_mask = _append_eos_to_targets(target_ids, attention_mask, int(eos_id))

    target_input_ids = target_ids.to(device=device)
    target_labels = target_ids.to(device=device)
    target_attention_mask = attention_mask.to(device=device)
    label_mask = attention_mask.to(device=device)
    target_labels = target_labels.masked_fill(~label_mask, IGNORE_INDEX)

    return LatentDecoderBatch(
        latent_prefix=latent_prefix,
        latent_prefix_attention_mask=latent_prefix_attention_mask,
        target_input_ids=target_input_ids,
        target_labels=target_labels,
        target_attention_mask=target_attention_mask,
        tags=target_tags,
        target_texts=target_texts,
    )


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _bbox_to_text(values: Iterable[float]) -> str:
    vals = [float(x) for x in values]
    return " ".join(f"{x:.4f}" for x in vals)


def _append_eos_to_targets(
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    masks = []
    for ids, mask in zip(target_ids, attention_mask):
        valid_len = int(mask.sum().item())
        row = torch.cat([ids[:valid_len], ids.new_tensor([eos_id])], dim=0)
        rows.append(row)
    max_len = max(row.numel() for row in rows)
    pad_value = int(eos_id)
    padded_rows = []
    for row in rows:
        valid_len = row.numel()
        pad_len = max_len - row.numel()
        if pad_len > 0:
            row = torch.cat([row, row.new_full((pad_len,), pad_value)], dim=0)
        padded_rows.append(row)
        masks.append(torch.arange(max_len) < valid_len)
    return torch.stack(padded_rows, dim=0), torch.stack(masks, dim=0)
