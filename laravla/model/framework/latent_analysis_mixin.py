"""
Latent Analysis Mixin — optional diagnostic hook for dumping thinking/img_next
token statistics and embeddings during training.

Mixed into Qwen_GR00T via multiple inheritance so the core forward/predict code
stays compact.  Everything here is gated by ``cfg.framework.latent_analysis.enable``
and has zero overhead when disabled.
"""

import hashlib
import json
import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from laravla.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class LatentAnalysisMixin:
    """Mixin that provides ``_maybe_log_latent_analysis`` and helpers."""

    # ------------------------------------------------------------------
    # Config reading (cached)
    # ------------------------------------------------------------------
    def _get_latent_analysis_cfg(self) -> dict:
        """
        Read config for lightweight latent-token analysis. Defaults to disabled.

        Supported config locations:
          - cfg.framework.latent_analysis
          - cfg.trainer.latent_analysis
        """
        cached = getattr(self, "_latent_analysis_cfg_cache", None)
        if isinstance(cached, dict):
            return cached

        cfg = self.config
        if cfg is None:
            return {}

        def _to_dict(obj):
            if obj is None:
                return {}
            if isinstance(obj, dict):
                return dict(obj)
            try:
                if hasattr(obj, "items"):
                    return dict(obj.items())
            except Exception:
                pass
            try:
                return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}
            except Exception:
                return {}

        fw = getattr(cfg, "framework", None)
        tr = getattr(cfg, "trainer", None)
        fw_cfg = _to_dict(getattr(fw, "latent_analysis", None)) if fw is not None else {}
        tr_cfg = _to_dict(getattr(tr, "latent_analysis", None)) if tr is not None else {}

        merged = {}
        merged.update(fw_cfg)
        merged.update(tr_cfg)
        self._latent_analysis_cfg_cache = merged
        return merged

    # ------------------------------------------------------------------
    # Main hook
    # ------------------------------------------------------------------
    def _maybe_log_latent_analysis(
        self,
        qwen_inputs: dict,
        last_hidden: torch.Tensor,
        vlm_outputs: Optional[dict],
        instructions: List[str],
        use_iterative_forward: bool,
        **kwargs,
    ) -> None:
        """
        Lightweight analysis hook: extract thinking/img_next hidden states and dump summary stats.

        This is intentionally low-overhead and should be gated by config + step interval.
        """
        cfg = self._get_latent_analysis_cfg()
        if not cfg or not bool(cfg.get("enable", False)):
            return

        global_step = kwargs.get("global_step", None)
        sync_gradients = bool(kwargs.get("analysis_sync_gradients", True))

        interval = int(cfg.get("interval_steps", 0) or 0)
        if interval <= 0:
            interval = int(cfg.get("interval_forwards", 0) or 0)
        if interval <= 0:
            return

        is_main_process = kwargs.get("is_main_process", None)
        if is_main_process is None:
            try:
                is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            except Exception:
                is_main_process = True
        is_main_process = bool(is_main_process)
        if not is_main_process:
            return

        if global_step is None:
            if not hasattr(self, "_latent_analysis_forward_calls"):
                self._latent_analysis_forward_calls = 0
            self._latent_analysis_forward_calls += 1
            global_step = int(self._latent_analysis_forward_calls)

        if (not sync_gradients) or (int(global_step) % interval != 0):
            return

        max_samples = int(cfg.get("max_samples", 4) or 4)
        max_latents = int(cfg.get("max_latents", 3) or 3)
        max_img_next = int(cfg.get("max_img_next", 16) or 16)
        dump_embeddings = bool(cfg.get("dump_embeddings", True))
        dump_img_next_embeddings = bool(cfg.get("dump_img_next_embeddings", True))
        embeddings_dtype = str(cfg.get("embeddings_dtype", "float16") or "float16").lower()
        embeddings_subdir = str(cfg.get("embeddings_subdir", "embeddings") or "embeddings")
        unique_in_batch = bool(cfg.get("unique_in_batch", True))
        dedupe_across_run = bool(cfg.get("dedupe_across_run", False))
        max_seen_instructions = int(cfg.get("max_seen_instructions", 50000) or 50000)
        log_saved_instruction_count = bool(cfg.get("log_saved_instruction_count", False))

        dump_dir = getattr(self, "_latent_analysis_dump_dir", None)
        if not dump_dir:
            dump_dir = str(cfg.get("dump_dir") or "")
            if not dump_dir:
                out_dir = getattr(self.config, "output_dir", None)
                dump_dir = os.path.join(str(out_dir), "latent_analysis") if out_dir else "latent_analysis"
            try:
                os.makedirs(dump_dir, exist_ok=True)
            except Exception as exc:
                logger.warning(f"[latent_analysis] cannot create dump_dir={dump_dir}: {exc}")
                return
            self._latent_analysis_dump_dir = dump_dir

        input_ids = qwen_inputs.get("input_ids", None)
        if input_ids is None or not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            return
        attention_mask = qwen_inputs.get("attention_mask", None)
        if attention_mask is None or not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            attention_mask = None

        thinking_token_id = getattr(self.qwen_vl_interface, "thinking_token_id", None)
        img_next_token_id = getattr(self.qwen_vl_interface, "img_next_token_id", None)

        B = int(input_ids.shape[0])

        def _base_instruction(text: str) -> str:
            t = (text or "").strip()
            if " @ " in t:
                t = t.split(" @ ", 1)[0].strip()
            return t

        rows = []
        think_vec_list: List[torch.Tensor] = []
        think_mask_list: List[torch.Tensor] = []
        pre5_vec_list: List[torch.Tensor] = []
        pre5_mask_list: List[torch.Tensor] = []
        img_vec_list: List[torch.Tensor] = []
        img_mask_list: List[torch.Tensor] = []
        input_ids_list: List[torch.Tensor] = []
        attention_mask_list: List[torch.Tensor] = []

        def _cast_dtype(x: torch.Tensor) -> torch.Tensor:
            if embeddings_dtype in ("fp16", "float16", "half"):
                return x.to(dtype=torch.float16)
            if embeddings_dtype in ("bf16", "bfloat16"):
                return x.to(dtype=torch.bfloat16)
            return x.to(dtype=torch.float32)

        batch_seen: set[str] = set()
        if dedupe_across_run:
            if not hasattr(self, "_latent_analysis_seen_instr"):
                self._latent_analysis_seen_instr = set()
            if getattr(self, "_latent_analysis_stop_saving", False):
                return
            if max_seen_instructions > 0 and len(self._latent_analysis_seen_instr) >= max_seen_instructions:
                if not getattr(self, "_latent_analysis_dedupe_overflow_warned", False):
                    logger.warning(
                        "[latent_analysis] max_seen_instructions=%s reached (seen=%s); stop saving further analysis.",
                        max_seen_instructions,
                        len(self._latent_analysis_seen_instr),
                    )
                    self._latent_analysis_dedupe_overflow_warned = True
                self._latent_analysis_stop_saving = True
                return

        for b in range(B):
            if len(rows) >= max_samples:
                break
            row = {
                "global_step": int(global_step),
                "batch_index": int(b),
                "use_iterative_forward": bool(use_iterative_forward),
                "cot_mode": str(getattr(self.config.framework, "cot_mode", "implicit")) if self.config is not None else "unknown",
            }

            instr = instructions[b] if b < len(instructions) else ""
            base = _base_instruction(instr)
            row["instruction"] = base[:200]
            row["instruction_sha1"] = hashlib.sha1(base.encode("utf-8")).hexdigest()

            sha1 = row["instruction_sha1"]
            if unique_in_batch and sha1 in batch_seen:
                continue
            if dedupe_across_run and sha1 in getattr(self, "_latent_analysis_seen_instr", set()):
                continue
            batch_seen.add(sha1)
            if dedupe_across_run:
                self._latent_analysis_seen_instr.add(sha1)

            if vlm_outputs is not None:
                try:
                    row["num_reasoning_passes"] = int(vlm_outputs.get("num_reasoning_passes", 0) or 0)
                except Exception:
                    row["num_reasoning_passes"] = None

            think_positions = []
            think_anchor_pos = None
            if thinking_token_id is not None:
                pos = torch.nonzero(input_ids[b] == int(thinking_token_id), as_tuple=False).squeeze(-1)
                if pos.numel() > 0:
                    think_anchor_pos = int(pos[0].item())
                    think_positions = pos[:max_latents].detach().cpu().tolist()
            row["thinking_token_id"] = int(thinking_token_id) if thinking_token_id is not None else None
            row["thinking_positions"] = think_positions
            row["thinking_count"] = int((input_ids[b] == int(thinking_token_id)).sum().item()) if thinking_token_id is not None else 0
            row["thinking_anchor_pos"] = think_anchor_pos

            if think_positions:
                vecs = last_hidden[b, torch.tensor(think_positions, device=last_hidden.device), :].detach().float()
                norms = torch.linalg.norm(vecs, dim=-1)
                row["thinking_norm_mean"] = float(norms.mean().item())
                row["thinking_norm_std"] = float(norms.std(unbiased=False).item()) if norms.numel() > 1 else 0.0

                v = F.normalize(vecs, dim=-1)
                cos = (v @ v.T).detach().cpu()
                if cos.numel() > 1:
                    off = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]
                    row["thinking_cos_offdiag_mean"] = float(off.mean().item()) if off.numel() else None
                else:
                    row["thinking_cos_offdiag_mean"] = None
                if cos.shape[0] >= 2:
                    row["thinking_cos_01"] = float(cos[0, 1].item())
                if cos.shape[0] >= 3:
                    row["thinking_cos_12"] = float(cos[1, 2].item())
                    row["thinking_cos_02"] = float(cos[0, 2].item())

                if dump_embeddings:
                    H = int(last_hidden.shape[-1])
                    token_count = min(len(think_positions), max_latents)
                    vec_pad = torch.zeros((max_latents, H), dtype=torch.float32, device=vecs.device)
                    mask_pad = torch.zeros((max_latents,), dtype=torch.bool, device=vecs.device)
                    if token_count > 0:
                        vec_pad[:token_count, :] = vecs[:token_count]
                        mask_pad[:token_count] = True
                    think_vec_list.append(_cast_dtype(vec_pad).cpu())
                    think_mask_list.append(mask_pad.cpu())

                    p0 = think_anchor_pos
                    pre5_H = int(last_hidden.shape[-1])
                    pre5_vec_pad = torch.zeros((5, pre5_H), dtype=torch.float32, device=vecs.device)
                    pre5_mask_pad = torch.zeros((5,), dtype=torch.bool, device=vecs.device)
                    pre5_positions: List[int] = []
                    if p0 is not None:
                        raw_positions = [p0 - 8, p0 - 7, p0 - 6, p0 - 5, p0 - 4]
                        valid_positions = [int(p) for p in raw_positions if int(p) >= 0 and int(p) < int(input_ids.shape[1])]
                        pre5_positions = valid_positions
                        if valid_positions:
                            pre_vecs = last_hidden[
                                b, torch.tensor(valid_positions, device=last_hidden.device), :
                            ].detach().float()
                            pre5_vec_pad[: pre_vecs.shape[0], :] = pre_vecs
                            pre5_mask_pad[: pre_vecs.shape[0]] = True
                    row["pre5_positions"] = pre5_positions
                    pre5_vec_list.append(_cast_dtype(pre5_vec_pad).cpu())
                    pre5_mask_list.append(pre5_mask_pad.cpu())

            img_positions = []
            if img_next_token_id is not None:
                pos = torch.nonzero(input_ids[b] == int(img_next_token_id), as_tuple=False).squeeze(-1)
                if pos.numel() > 0:
                    img_positions = pos[:max_img_next].detach().cpu().tolist()
            row["img_next_token_id"] = int(img_next_token_id) if img_next_token_id is not None else None
            row["img_next_count"] = int((input_ids[b] == int(img_next_token_id)).sum().item()) if img_next_token_id is not None else 0
            row["img_next_positions_head"] = img_positions

            if dump_embeddings:
                input_ids_list.append(input_ids[b].detach().cpu())
                if attention_mask is not None:
                    attention_mask_list.append(attention_mask[b].detach().cpu())

            if dump_embeddings and dump_img_next_embeddings and img_next_token_id is not None:
                H = int(last_hidden.shape[-1])
                token_count = min(len(img_positions), max_img_next)
                vec_pad = torch.zeros((max_img_next, H), dtype=torch.float32, device=last_hidden.device)
                mask_pad = torch.zeros((max_img_next,), dtype=torch.bool, device=last_hidden.device)
                if token_count > 0:
                    vecs = last_hidden[b, torch.tensor(img_positions, device=last_hidden.device), :].detach().float()
                    vec_pad[:token_count, :] = vecs[:token_count]
                    mask_pad[:token_count] = True
                img_vec_list.append(_cast_dtype(vec_pad).cpu())
                img_mask_list.append(mask_pad.cpu())

            rows.append(row)

        if not rows:
            return

        if log_saved_instruction_count:
            try:
                total_seen = (
                    len(getattr(self, "_latent_analysis_seen_instr", set()))
                    if dedupe_across_run
                    else None
                )
            except Exception:
                total_seen = None
            logger.info(
                "[latent_analysis] saved_rows=%s unique_in_batch=%s dedupe_across_run=%s total_seen=%s step=%s",
                len(rows),
                unique_in_batch,
                dedupe_across_run,
                total_seen,
                int(global_step),
            )

        out_path = os.path.join(str(dump_dir), "latent_stats.jsonl")
        try:
            with open(out_path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"[latent_analysis] failed to write stats: {exc}")

        if dump_embeddings and think_vec_list and think_mask_list:
            emb_dir = os.path.join(str(dump_dir), embeddings_subdir)
            try:
                os.makedirs(emb_dir, exist_ok=True)
            except Exception as exc:
                logger.warning(f"[latent_analysis] failed to create embeddings dir: {exc}")
                return

            payload = {
                "global_step": int(global_step),
                "use_iterative_forward": bool(use_iterative_forward),
                "cot_mode": str(getattr(self.config.framework, "cot_mode", "implicit")) if self.config is not None else "unknown",
                "rows": rows,
                "thinking_vecs": torch.stack(think_vec_list, dim=0),
                "thinking_mask": torch.stack(think_mask_list, dim=0),
                "pre5_vecs": torch.stack(pre5_vec_list, dim=0) if pre5_vec_list else None,
                "pre5_mask": torch.stack(pre5_mask_list, dim=0) if pre5_mask_list else None,
            }
            if dump_img_next_embeddings and img_vec_list and img_mask_list:
                payload["img_next_vecs"] = torch.stack(img_vec_list, dim=0)
                payload["img_next_mask"] = torch.stack(img_mask_list, dim=0)

            try:
                payload["input_ids"] = torch.stack(input_ids_list, dim=0) if input_ids_list else input_ids[: len(rows)].detach().cpu()
                if attention_mask is not None and attention_mask_list:
                    payload["attention_mask"] = torch.stack(attention_mask_list, dim=0)
            except Exception:
                pass

            emb_path = os.path.join(emb_dir, f"latent_emb_step_{int(global_step):08d}.pt")
            try:
                torch.save(payload, emb_path)
            except Exception as exc:
                logger.warning(f"[latent_analysis] failed to write embeddings: {exc}")
