# ECoT 训练配置与代码精简 — 执行 Schedule

> 目标：默认只支持 **latent reasoning (implicit)** 模式，清除冗余配置和代码。
> 每个 Step 独立可验证，按顺序执行，每完成一步建议 commit 一次。

---

## Phase 1：配置文件精简（3 个 yaml）

> 风险等级：🟢 低（不改代码逻辑，代码中均有 `getattr` / `.get` 默认值兜底）
> 验证方式：`python -c "from omegaconf import OmegaConf; c = OmegaConf.load('laravla/config/training/libero_all_ecot_stage4.yaml'); print(c)"` 确认 yaml 语法正确

### Step 1.1 — 删除 `datasets.vla_data.ecot` 整个节点

**涉及文件（3 个）：**
- `laravla/config/training/libero_all_ecot_stage4.yaml`（L62-63）
- `laravla/config/training/libero_goal_ecot_stage4.yaml`（L74-75）
- `laravla/config/training/bridge_lerobot_stage2.yaml`（L53-54）

**操作：** 删除以下内容（每个文件中）：
```yaml
    ecot:
      scheduled_stage: 4
```

**原因：** `train.py` 的 `main()` 在 implicit 模式下会强制覆写为 4，yaml 中的声明从未被真正读取。

---

### Step 1.2 — 删除 `framework.enable_latent_reasoning` 和 `framework.emit_thinking_tokens`

**涉及文件（2 个）：**
- `laravla/config/training/libero_all_ecot_stage4.yaml`（L69-71，含注释）
- `laravla/config/training/bridge_lerobot_stage2.yaml`（L60-61）

**操作：** 删除以下行：
```yaml
  # Keep parity with bridge configs; ...
  enable_latent_reasoning: true
  emit_thinking_tokens: false
```

**原因：** `train.py` → `main()` 中由 `cot_mode` 自动派生后覆写这两个值。

> 注意：`libero_goal_ecot_stage4.yaml` 中没有这两行（已经是干净的），不需要改。

---

### Step 1.3 — 删除 `framework.latent_reasoning` 中的 3 个 token 定义

**涉及文件（2 个）：**
- `laravla/config/training/libero_all_ecot_stage4.yaml`（L72-77）
- `laravla/config/training/bridge_lerobot_stage2.yaml`（L62-67）

**操作：** 将 `framework.latent_reasoning` 从：
```yaml
  latent_reasoning:
    compute_language_loss: true
    vlm_loss_weight: 1
    thinking_token: "<|thinking|>"
    start_of_thinking_token: "<|start_of_thinking|>"
    end_of_thinking_token: "<|end_of_thinking|>"
```
精简为：
```yaml
  latent_reasoning:
    compute_language_loss: true
    vlm_loss_weight: 1
```

**原因：** token 定义已在 `bridge_reasoning` 中存在，`sync_bridge_reasoning_to_framework()` 会自动同步。删除后 `QWen3.py` 的 `_add_thinking_tokens()` 仍能从 `bridge_reasoning` 读到（Phase 3 中会修改读取逻辑）。在 Phase 3 完成之前，`sync_bridge_reasoning_to_framework()` 仍会把 `bridge_reasoning` 的 token 同步到 `framework.latent_reasoning`，所以不会 break。

---

### Step 1.4 — 删除 `bridge_reasoning.vlm_loss_weight`

**涉及文件（2 个）：**
- `laravla/config/training/libero_all_ecot_stage4.yaml`（L60）
- `laravla/config/training/bridge_lerobot_stage2.yaml`（L52）

**操作：** 删除 `bridge_reasoning` 节点下的：
```yaml
      vlm_loss_weight: 1
```

**原因：** `vlm_loss_weight` 只在 `framework.latent_reasoning` 中保留一份。`sync_bridge_reasoning_to_framework()` 中读取 `bridge_cfg.vlm_loss_weight` 时有 fallback 到 `latent_cfg.vlm_loss_weight`，所以删除 bridge 侧不会 break。

> 注意：`libero_goal_ecot_stage4.yaml` 的 `bridge_reasoning` 中也有 `vlm_loss_weight: 1`（L72），同样删除。

---

### Step 1.5 — 删除 `action_model` 中 9 个未使用的字段

**涉及文件（2 个）：**
- `laravla/config/training/libero_all_ecot_stage4.yaml`（L109-118）
- `laravla/config/training/bridge_lerobot_stage2.yaml`（L97-105）

**操作：** 删除以下 9 行（含注释行）：
```yaml
    # Keep parity with Bridge config knobs ...
    use_reasoning_summary: false
    use_img_next_mlp_compress: false
    reasoning_summary_tokens: 2
    reasoning_summary_heads: 4
    reasoning_summary_dropout: 0.1
    use_reasoning_film: false
    reasoning_film_first_k: 4
    reasoning_film_dropout: 0.1
    reasoning_film_hidden: 1024
```

**原因：** 全部为 false/默认值，功能未使用。代码中 `getattr(..., False)` / `getattr(..., default)` 能正确 fallback。

> 注意：`libero_goal_ecot_stage4.yaml` 中没有这些字段（已经是干净的），不需要改。

---

### Step 1.6 — 处理 `steps_cache_path` 和 `write_steps_cache`

**涉及文件（3 个）：**
- `laravla/config/training/libero_all_ecot_stage4.yaml`（L35-37）
- `laravla/config/training/libero_goal_ecot_stage4.yaml`（L48-49）
- `laravla/config/training/bridge_lerobot_stage2.yaml`（L31）

**操作：** 将硬编码绝对路径改为 `null`：
```yaml
    bridge_annotations:
      ...
      steps_cache_path: null   # 自动推导或由 CLI 指定
      write_steps_cache: true
```

**原因：** 硬编码路径 `/share/project/lvjing/...` 开源后无法使用。设为 null 后需要在 Phase 3 中确认 `gr00t_lerobot/datasets.py` 对 null 的处理（如果没有 null 兜底逻辑，需要加一个）。

---

### Step 1.7 — 删除 `bridge_lerobot_stage2.yaml` 中的 `trainer.latent_analysis` 块

**涉及文件（1 个）：**
- `laravla/config/training/bridge_lerobot_stage2.yaml`（L124-146）

**操作：** 删除整个 `latent_analysis` 子节点（约 23 行）：
```yaml
  latent_analysis:
    enable: false
    interval_steps: 1
    ...
    dump_dir: null
```

**原因：** latent_analysis 是调试/分析功能，开源版不需要在默认配置中暴露。代码中 `_get_latent_analysis_cfg()` 返回空 dict 时自动跳过。

---

### Step 1.8 — 保留 `bridge_lerobot_stage2.yaml` 中经过验证的 `exclude_task_indices` 列表

**涉及文件（1 个）：**
- `laravla/config/training/bridge_lerobot_stage2.yaml`（L38）

**操作：** 保留以下配置：
```yaml
        exclude_task_indices: [4,11,338,1176,3656,3922,4446,4865,6552,9814,11386,11704,12032,12708,13667,15039,16104,16392,16914,17286,18937]
```

**原因：** 这组索引现在被视为有意保留的默认数据过滤配置，而不是临时的私有机器痕迹。既然它对默认训练配方有实际意义，就可以继续保留。

---

#### Phase 1 验证清单

```bash
# 逐个验证 yaml 语法
python -c "from omegaconf import OmegaConf; c = OmegaConf.load('laravla/config/training/libero_all_ecot_stage4.yaml'); print('OK')"
python -c "from omegaconf import OmegaConf; c = OmegaConf.load('laravla/config/training/libero_goal_ecot_stage4.yaml'); print('OK')"
python -c "from omegaconf import OmegaConf; c = OmegaConf.load('laravla/config/training/bridge_lerobot_stage2.yaml'); print('OK')"
```

---

## Phase 2：删除死代码（不改逻辑）

> 风险等级：🟢 低（只删除不可达 / 未调用的代码）
> 验证方式：`python -c "from laravla.training.train import main"` 确认 import 不报错

### Step 2.1 — 删除 `train.py` 中的 `load_fast_tokenizer()`

**涉及文件：** `laravla/training/train.py`（L65-67）

**操作：** 删除：
```python
def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer
```

**原因：** 定义了但从未在此文件中调用。同时删除顶部 `from transformers import AutoProcessor` 中的 `AutoProcessor`（如果只有这里用到）。

> 检查：`AutoProcessor` 在此文件中是否有其他使用？答：没有。可以从 import 中移除。

---

### Step 2.2 — 删除 `train.py` 中的 debugpy 代码

**涉及文件：** `laravla/training/train.py`（L740-745）

**操作：** 删除：
```python
    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()
```

**原因：** 开源版不应包含远程调试入口，且暴露 0.0.0.0 端口有安全风险。

---

### Step 2.3 — 删除 `QwenGR00T.py` 中 `__main__` 块的 debugpy

**涉及文件：** `laravla/model/framework/QwenGR00T.py`（L1033, L1039-1041）

**操作：** 删除 `if __name__ == "__main__"` 块中的：
```python
    import debugpy
    ...
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
```

保留 `__main__` 块的其余部分（smoke test 功能有价值）。

---

### Step 2.4 — 删除 `QwenGR00T.py` 中的 `DEBUG_THINKING_ATTN` 相关代码

**涉及文件（2 个）：**
- `laravla/model/framework/QwenGR00T.py`（L993-1002, L1009）
- `laravla/model/modules/action_model/flow_matching_head/cross_attention_dit.py`（L30, L179, L337, L365）

**操作 A — `QwenGR00T.py`：**

1. 删除 `predict_action()` 中的（L993-1002）：
```python
        if getattr(dit_debug, "DEBUG_THINKING_ATTN", False):
            try:
                cache = self.action_model.model.get_and_clear_thinking_attn_cache()
                if cache:
                    out_dir = "/share/project/lvjing/laravla/results/ANALY"
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, "thinking_attn_cache.pt")
                    torch.save(cache, out_path)
            except Exception as exc:
                logger.warning(f"[thinking_attn] failed to save cache: {exc}")
```

2. 简化 `_extract_reasoning_mask()`（L1007-1010），将：
```python
    def _extract_reasoning_mask(self, qwen_inputs) -> Optional[torch.Tensor]:
        if not (self.use_reasoning_summary or self.use_reasoning_film):
            if not getattr(dit_debug, "DEBUG_THINKING_ATTN", False):
                return None
```
改为：
```python
    def _extract_reasoning_mask(self, qwen_inputs) -> Optional[torch.Tensor]:
        if not (self.use_reasoning_summary or self.use_reasoning_film):
            return None
```

3. 删除顶部的 import（L34）：
```python
from laravla.model.modules.action_model.flow_matching_head import cross_attention_dit as dit_debug
```

**操作 B — `cross_attention_dit.py`：**

删除 `DEBUG_THINKING_ATTN = False`（L30），以及所有 `if DEBUG_THINKING_ATTN:` 分支块（L178-190 区域, L336-338 区域, L365-370 区域）。同时删除相关的 `get_and_clear_thinking_attn_cache` 方法（如果存在）。

> 这一步改动较多，建议先 grep 确认所有 `DEBUG_THINKING_ATTN` 出现位置，逐个删除。

---

### Step 2.5 — 删除 `QwenGR00T.py` 中 `__main__` 块末尾的注释掉的代码

**涉及文件：** `laravla/model/framework/QwenGR00T.py`（L1074-1107）

**操作：** 删除所有被 `# #` 注释掉的 dataloader 测试代码块。

**原因：** 注释掉的代码不应出现在开源仓库中。

---

#### Phase 2 验证清单

```bash
# 确认 import 链不报错
python -c "from laravla.model.framework.QwenGR00T import Qwen_GR00T; print('OK')"
python -c "from laravla.training.train import main; print('OK')"
```

---

## Phase 3：精简训练脚本逻辑

> 风险等级：🟡 中（修改运行时逻辑，需要实际训练验证）
> 验证方式：用 1 个 GPU 跑 10 步训练确认 loss 正常下降

### Step 3.1 — 精简 `cot_mode_utils.py`

**涉及文件：** `laravla/training/trainer_utils/cot_mode_utils.py`

**操作：** 将整个文件重写为：

```python
"""Implicit-only CoT mode utilities."""

IMPLICIT_FLAGS = {
    "enable_latent_reasoning": True,
    "emit_thinking_tokens": False,
    "use_iterative_forward": True,
    "generate_thinking": True,
    "reasoning_stage": 4,
}


def get_implicit_flags() -> dict:
    """Return the fixed flag set for implicit latent reasoning."""
    return dict(IMPLICIT_FLAGS)
```

**原因：** 删除 `CotMode` 枚举、`parse_cot_mode()`、`derive_flags_from_mode()`，只保留 implicit 常量。

---

### Step 3.2 — 精简 `train.py` 的 `main()` 函数

**涉及文件：** `laravla/training/train.py`

**操作 A — 修改 import：**

将：
```python
from laravla.training.trainer_utils.cot_mode_utils import parse_cot_mode, derive_flags_from_mode
```
改为：
```python
from laravla.training.trainer_utils.cot_mode_utils import get_implicit_flags
```

**操作 B — 重写 `main()` 开头（L668-692）：**

将：
```python
def main(cfg) -> None:
    logger.info("ECoT VLA Training :: Warming Up")
    
    cot_mode = parse_cot_mode(cfg)
    mode_flags = derive_flags_from_mode(cot_mode)
    
    cfg.framework.enable_latent_reasoning = mode_flags["enable_latent_reasoning"]
    cfg.framework.emit_thinking_tokens = mode_flags.get("emit_thinking_tokens", False)
    cfg.framework.cot_mode_flags = mode_flags
    training_stage = getattr(cfg.framework, "training_stage", "full")
    if training_stage in ("reasoning_only", "action_only"):
        logger.info(...)
    else:
        cfg.datasets.vla_data.bridge_reasoning.stage = mode_flags["reasoning_stage"]
        cfg.datasets.vla_data.ecot.scheduled_stage = mode_flags["reasoning_stage"]
    
    logger.info(f"[CotMode] mode={cot_mode.value}, flags={mode_flags}")
    sync_bridge_reasoning_to_framework(cfg)
    validate_ecot_config(cfg)
```

改为：
```python
def main(cfg) -> None:
    logger.info("ECoT VLA Training :: Warming Up")

    mode_flags = get_implicit_flags()

    cfg.framework.enable_latent_reasoning = True
    cfg.framework.emit_thinking_tokens = False
    cfg.framework.cot_mode = "implicit"
    cfg.framework.cot_mode_flags = mode_flags

    training_stage = getattr(cfg.framework, "training_stage", "full")
    if training_stage == "full":
        cfg.datasets.vla_data.bridge_reasoning.stage = mode_flags["reasoning_stage"]

    logger.info(f"[Implicit Reasoning] training_stage={training_stage}, flags={mode_flags}")

    sync_bridge_reasoning_to_framework(cfg)
    validate_ecot_config(cfg)
```

**关键变化：**
- 不再解析 `cot_mode`，直接设为 implicit
- 删除 `ecot.scheduled_stage` 的覆写（该节点已在 Phase 1 中删除）
- 简化日志

---

### Step 3.3 — 精简 `sync_bridge_reasoning_to_framework()`

**涉及文件：** `laravla/training/train.py`（L256-314）

**操作：** 重写为：

```python
def sync_bridge_reasoning_to_framework(cfg):
    """
    Sync thinking token definitions from bridge_reasoning to framework.latent_reasoning,
    ensuring VLM and dataloader use the same token strings.
    """
    try:
        bridge_cfg = cfg.datasets.vla_data.bridge_reasoning
    except AttributeError:
        return

    if not getattr(bridge_cfg, "enable", False):
        return

    # Ensure latent reasoning is enabled
    cfg.framework.enable_latent_reasoning = True

    latent_cfg = getattr(cfg.framework, "latent_reasoning", None)
    if latent_cfg is None:
        cfg.framework.latent_reasoning = {}
        latent_cfg = cfg.framework.latent_reasoning

    def _set(key, value):
        if isinstance(latent_cfg, dict):
            latent_cfg[key] = value
        else:
            setattr(latent_cfg, key, value)

    # Sync token definitions (single source of truth: bridge_reasoning)
    _set("thinking_token", getattr(bridge_cfg, "thinking_token", "<|thinking|>"))
    _set("start_of_thinking_token", getattr(bridge_cfg, "start_token", "<|start_of_thinking|>"))
    _set("end_of_thinking_token", getattr(bridge_cfg, "end_token", "<|end_of_thinking|>"))

    tag2think = getattr(bridge_cfg, "tag2think_count", None)
    if tag2think is not None:
        _set("tag2think_count", tag2think)

    _set("compute_language_loss", True)
```

**关键变化：**
- 删除了复杂的 `_get()` 辅助函数
- 删除了 `vlm_loss_weight` 的双向同步（只从 `framework.latent_reasoning` 读取，不再从 bridge 侧读）
- token 定义明确从 `bridge_reasoning` → `framework.latent_reasoning` 单向同步

---

### Step 3.4 — 精简 `validate_ecot_config()`

**涉及文件：** `laravla/training/train.py`（L182-253）

**操作：** 重写为：

```python
def validate_ecot_config(cfg):
    """Validate latent reasoning configuration consistency."""
    latent_cfg = cfg.framework.get("latent_reasoning", {})
    if not latent_cfg:
        logger.warning("latent_reasoning config is missing, using defaults")
        return

    vlm_loss_weight = latent_cfg.get("vlm_loss_weight", 0.1)
    if not (0.0 <= vlm_loss_weight <= 10.0):
        logger.warning(f"vlm_loss_weight={vlm_loss_weight} outside recommended range [0, 10]")

    logger.info(f"Latent reasoning: compute_language_loss={latent_cfg.get('compute_language_loss', False)}, "
                f"vlm_loss_weight={vlm_loss_weight}")

    # Validate img_next config
    img_next_cfg = cfg.framework.get("img_next", {})
    if img_next_cfg and img_next_cfg.get("enable", False):
        loss_w = img_next_cfg.get("loss_weight", 0)
        logger.info(f"img_next: res={img_next_cfg.get('res')}, loss_weight={loss_w}, "
                    f"use_teacher={img_next_cfg.get('use_teacher', True)}")

    logger.info("Config validation passed")
```

**关键变化：**
- 删除了 `enable_latent_reasoning=False` 的分支（默认 implicit 不会走到）
- 删除了 `scheduled_stage` 的检查（该字段已删除）
- 删除了 thinking token 的逐个检查（sync 函数已保证一致性）
- 从 ~70 行精简到 ~15 行

---

### Step 3.5 — 精简 `prepare_data()` 中的 `data_mix` fallback

**涉及文件：** `laravla/training/train.py`（L110-127）

**操作：** 将：
```python
    dataset_py = cfg.datasets.vla_data.dataset_py
    try:
        data_mix = cfg.datasets.vla_data.data_mix
    except (AttributeError, KeyError):
        try:
            data_mix = cfg.datasets.vla_data.ecot.data_mix
        except (AttributeError, KeyError):
            data_mix = "unknown"
```
改为：
```python
    dataset_py = cfg.datasets.vla_data.dataset_py
    data_mix = getattr(cfg.datasets.vla_data, "data_mix", "unknown")
```

**原因：** `ecot.data_mix` 路径已不存在（ecot 节点已删除）。

---

### Step 3.6 — 精简 `_log_training_config()` 中的 `scheduled_stage` 读取

**涉及文件：** `laravla/training/train.py`（L562-591）

**操作：** 将 `scheduled_stage` 的读取从：
```python
                try:
                    scheduled_stage = self.config.datasets.vla_data.ecot.get("scheduled_stage", 0)
                except (AttributeError, KeyError):
                    scheduled_stage = 0
                
                logger.info("***** ECoT Implicit Reasoning Configuration *****")
                logger.info(f"  Enable Latent Reasoning: {enable_latent_reasoning}")
                logger.info(f"  Scheduled Stage: {scheduled_stage}")
```
改为：
```python
                reasoning_stage = getattr(self.config.datasets.vla_data.bridge_reasoning, "stage", 4)

                logger.info("***** Latent Reasoning Configuration *****")
                logger.info(f"  Reasoning Stage: {reasoning_stage}")
```

---

#### Phase 3 验证清单

```bash
# 1. 确认 import 链正常
python -c "from laravla.training.train import main; print('OK')"

# 2. 用 1 GPU 跑 10 步（dry-run）
accelerate launch --num_processes 1 \
  laravla/training/train.py \
  --config_yaml laravla/config/training/libero_all_ecot_stage4.yaml \
  --trainer.max_train_steps 10 \
  --trainer.save_interval 100 \
  --trainer.eval_interval 100
```

---

## Phase 4：精简 QwenGR00T.py 模型代码

> 风险等级：🟡 中（修改 forward/predict 逻辑）
> 验证方式：`python laravla/model/framework/QwenGR00T.py` smoke test + 实际训练 10 步

### Step 4.1 — 删除 `forward()` 中的 explicit 分支

**涉及文件：** `laravla/model/framework/QwenGR00T.py`

**操作：** 在 `forward()` 方法中（约 L129-139），删除：
```python
        cot_mode = getattr(self.config.framework, "cot_mode", "implicit")
        ...
        if cot_mode == "explicit":
            reasoning_mask = None
```

简化为：始终走 implicit 路径。`use_iterative_forward` 的判断简化为：
```python
        enable_latent_reasoning = self.config.framework.get("enable_latent_reasoning", False)
        use_iterative_forward = (
            enable_latent_reasoning
            and hasattr(self.qwen_vl_interface, "forward_latent")
        )
```

---

### Step 4.2 — 删除 `predict_action()` 中的 explicit 分支

**涉及文件：** `laravla/model/framework/QwenGR00T.py`

**操作：** 在 `predict_action()` 方法中（约 L900-951），删除：
```python
        cot_mode = kwargs.get("cot_mode", "implicit")
        emit_thinking_tokens = kwargs.get("emit_thinking_tokens", False)
        think_max_len = kwargs.get("think_max_len", 64)
        think_temp = kwargs.get("think_temp", 0.1)
        think_topp = kwargs.get("think_topp", 0.9)

        if cot_mode == "explicit":
            use_iterative_forward = False
        elif cot_mode == "implicit":
            use_iterative_forward = True
        else:
            use_iterative_forward = False
```

以及整个 `if cot_mode == "explicit":` 分支（L923-951）。

简化为：
```python
        use_iterative_forward = hasattr(self.qwen_vl_interface, 'forward_latent')
```

---

### Step 4.3 — 移出 `_maybe_log_latent_analysis()` 到独立文件

**涉及文件：** `laravla/model/framework/QwenGR00T.py`（L341-721，约 380 行）

**操作：**
1. 创建 `laravla/model/framework/latent_analysis.py`
2. 将 `_get_latent_analysis_cfg()` 和 `_maybe_log_latent_analysis()` 移入
3. 在 `QwenGR00T.py` 中改为：
```python
from laravla.model.framework.latent_analysis import maybe_log_latent_analysis

# 在 forward() 中调用处改为：
maybe_log_latent_analysis(self, qwen_inputs=..., last_hidden=..., ...)
```

**原因：** 380 行分析代码严重干扰了核心 forward 逻辑的可读性。

> 如果你决定开源版完全不需要 latent_analysis 功能，可以直接删除这 380 行，并删除 forward() 中的两处调用。

---

### Step 4.4 — 删除 `QwenGR00T.py` 中 `__main__` 块末尾注释代码

**涉及文件：** `laravla/model/framework/QwenGR00T.py`（L1074-1107）

**操作：** 删除所有 `# #` 注释掉的 dataloader 测试代码。

---

#### Phase 4 验证清单

```bash
# 1. smoke test
python laravla/model/framework/QwenGR00T.py --config_yaml laravla/config/training/libero_all_ecot_stage4.yaml

# 2. 实际训练 10 步
accelerate launch --num_processes 1 \
  laravla/training/train.py \
  --config_yaml laravla/config/training/libero_all_ecot_stage4.yaml \
  --trainer.max_train_steps 10
```

---

## Phase 5：清理 `cross_attention_dit.py` 中的调试代码

> 风险等级：🟡 中
> 前置依赖：Phase 2 Step 2.4 中已删除 QwenGR00T 侧的引用

### Step 5.1 — 清理 `cross_attention_dit.py`

**涉及文件：** `laravla/model/modules/action_model/flow_matching_head/cross_attention_dit.py`

**操作：**
1. 删除 `DEBUG_THINKING_ATTN = False`（L30）
2. 删除所有 `if DEBUG_THINKING_ATTN:` 条件块
3. 删除 `get_and_clear_thinking_attn_cache` 方法（如果存在）
4. 删除 `_thinking_attn_cache` 相关代码

---

## Phase 6：处理 `steps_cache_path` 的代码兜底

> 风险等级：🟡 中
> 前置依赖：Phase 1 Step 1.6

### Step 6.1 — 在 `gr00t_lerobot/datasets.py` 中添加 null 兜底

**涉及文件：** `laravla/dataloader/gr00t_lerobot/datasets.py`

**操作：** 找到读取 `steps_cache_path` 的位置，确保当值为 `null/None` 时：
- 要么自动推导路径（如 `{data_root_dir}/.cache/steps_{hash}.pkl`）
- 要么跳过缓存功能

> 具体改法需要看该文件中 `steps_cache_path` 的使用逻辑，建议先 grep 确认。

---

## 总览表

| Phase | Step | 文件 | 操作类型 | 风险 | 预计耗时 |
|-------|------|------|----------|------|----------|
| 1 | 1.1 | 3 yaml | 删除 ecot 节点 | 🟢 | 2 min |
| 1 | 1.2 | 2 yaml | 删除 2 个 flag | 🟢 | 2 min |
| 1 | 1.3 | 2 yaml | 删除 3 个 token 定义 | 🟢 | 2 min |
| 1 | 1.4 | 3 yaml | 删除 vlm_loss_weight | 🟢 | 2 min |
| 1 | 1.5 | 2 yaml | 删除 9 个 unused 字段 | 🟢 | 3 min |
| 1 | 1.6 | 3 yaml | steps_cache → null | 🟢 | 2 min |
| 1 | 1.7 | 1 yaml | 删除 latent_analysis 块 | 🟢 | 2 min |
| 1 | 1.8 | 1 yaml | exclude_indices → null | 🟢 | 1 min |
| 2 | 2.1 | train.py | 删除 load_fast_tokenizer | 🟢 | 2 min |
| 2 | 2.2 | train.py | 删除 debugpy | 🟢 | 2 min |
| 2 | 2.3 | QwenGR00T.py | 删除 __main__ debugpy | 🟢 | 2 min |
| 2 | 2.4 | QwenGR00T.py + dit | 删除 DEBUG_THINKING_ATTN | 🟢 | 10 min |
| 2 | 2.5 | QwenGR00T.py | 删除注释代码 | 🟢 | 3 min |
| 3 | 3.1 | cot_mode_utils.py | 重写为 implicit-only | 🟡 | 5 min |
| 3 | 3.2 | train.py | 精简 main() | 🟡 | 10 min |
| 3 | 3.3 | train.py | 精简 sync 函数 | 🟡 | 10 min |
| 3 | 3.4 | train.py | 精简 validate 函数 | 🟡 | 5 min |
| 3 | 3.5 | train.py | 精简 prepare_data | 🟡 | 3 min |
| 3 | 3.6 | train.py | 精简 _log_training_config | 🟡 | 3 min |
| 4 | 4.1 | QwenGR00T.py | 删除 forward explicit | 🟡 | 10 min |
| 4 | 4.2 | QwenGR00T.py | 删除 predict explicit | 🟡 | 10 min |
| 4 | 4.3 | QwenGR00T.py | 移出 latent_analysis | 🟡 | 20 min |
| 4 | 4.4 | QwenGR00T.py | 删除注释代码 | 🟢 | 3 min |
| 5 | 5.1 | cross_attention_dit.py | 清理调试代码 | 🟡 | 10 min |
| 6 | 6.1 | datasets.py | steps_cache null 兜底 | 🟡 | 15 min |

**总预计耗时：约 2-3 小时**（含验证）

---

## 建议的 Git Commit 节奏

```
git commit -m "config: remove redundant ecot/latent fields from training yamls"     # Phase 1
git commit -m "cleanup: remove dead code (debugpy, unused functions)"                # Phase 2
git commit -m "refactor: simplify train to implicit-only mode"                  # Phase 3
git commit -m "refactor: remove explicit CoT branches from QwenGR00T"                # Phase 4
git commit -m "cleanup: remove DEBUG_THINKING_ATTN from cross_attention_dit"         # Phase 5
git commit -m "fix: handle null steps_cache_path gracefully"                         # Phase 6
```
