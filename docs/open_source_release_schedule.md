# 仓库开源发布 Schedule

> 目标：将当前仓库从“可在内部/熟悉环境下运行的研究仓库”整理为“可公开发布、可被外部用户理解、安装、复现和贡献”的开源仓库。
> 原则：先清风险，再统一入口，再补文档与验证，最后发布。
> 建议节奏：按 **7 个 Phase** 推进，每完成一个 Phase 建议单独 commit 一次。

---

## 当前状态

### 已完成 / 已开始

- 训练入口已经统一为 `laravla/training/train.py`
- 启动脚本中的训练入口引用已经同步到 `train.py`
- 仓库主线已经基本收敛到 **implicit latent reasoning** 模式
- 已有一份配置/训练清理计划：[config_and_training_cleanup_plan.md](./config_and_training_cleanup_plan.md)

### 当前主要阻塞项

- Some repo-level planning docs still lag behind the current `laravla/` namespace
- Final legal/provenance review is still pending
- Release metadata and community files still need final cleanup
- Minimal validation exists, but release-time consistency checks should still be documented

---

## 总体时间表

### Week 1

- Phase 0：冻结主入口与命名
- Phase 1：安全与私有信息清理
- Phase 2：训练/配置/脚本一致性清理

### Week 2

- Phase 3：文档重写
- Phase 4：许可证与第三方归属审计
- Phase 5：最小验证与 CI

### Week 3

- Phase 6：发布资产准备
- Phase 7：发布彩排与正式开源

> 如果节奏紧，可以把 Phase 3 / 4 / 5 并行推进；但 **Phase 1 必须先完成**。

---

## Phase 0：冻结主入口与命名

> 风险等级：🟢 低
> 目标：让仓库对外只有一套训练入口命名，避免后续文档和脚本继续分叉。

### Tasks

1. 将所有脚本中的旧训练入口调用统一为 `train.py`
2. 将代码注释、yaml 注释、文档中的旧入口描述统一切换为 `train`
3. 检查 `__main__` 默认 `config_yaml` 是否仍指向已删除/不存在的旧配置
4. 确认 `from laravla.training.train import main` 可正常 import

### 涉及重点文件

- `scripts/run_libero_multistage.sh`
- `scripts/run_bridge_multistage.sh`
- `scripts/run_laravla_libero.sh`
- `scripts/run_laravla_bridge.sh`
- `docs/config_and_training_cleanup_plan.md`
- `laravla/config/training/*.yaml`
- `README.md`
- `laravla/model/framework/QwenGR00T.py`
- `laravla/model/modules/vlm/QWen2_5.py`
- `laravla/dataloader/lerobot_datasets.py`

### 验收标准

- 仓库中不再残留旧训练入口名
- 所有训练脚本均调用 `laravla/training/train.py`

---

## Phase 1：安全与私有信息清理

> 风险等级：🔴 高
> 目标：去掉任何不适合公开发布的敏感内容、机器绑定内容和调试入口。

### Tasks

1. 删除仓库中的硬编码密钥
2. 删除或改造内网/私有镜像默认值
3. 删除所有 `debugpy.listen(("0.0.0.0", ...))` 和 `wait_for_client()` 默认入口
4. 将硬编码绝对路径改为：
   - `null`
   - 相对路径
   - 环境变量
   - CLI 参数
5. Review dataset-specific filter configs and remove them only if they are truly private or not meant to be part of the public default setup
6. 检查根目录和 docs 是否包含内部机器名、用户名、私有目录结构

### 当前已发现的重点风险

- `scripts/run_bridge_multistage.sh`
  - `WANDB_API_KEY`
  - `WANDB_BASE_URL`
  - `HF_ENDPOINT`
  - 私有 `steps_cache_path`
- `examples/LIBERO/eval_libero_all.sh`
  - 固定 conda 环境路径
  - 固定 checkpoint 路径
  - 固定 `LIBERO_HOME`
- `examples/SimplerEnv/bridge_eval.sh`
  - 固定 python / 环境 / 路径
- `laravla/config/training/*.yaml`
  - `data_root_dir`
  - `cache_dir`
  - `steps_cache_path`
  - `wandb_entity`
- `deployment/model_server/server_policy.py`
- `examples/LIBERO/eval_libero.py`
- `laravla/dataloader/lerobot_datasets.py`
- `laravla/model/modules/vlm/QWen2_5.py`

### 建议输出

- 一次 “sanitize” commit
- 一份对外安全默认值规范：
  - 默认不启用远程调试
  - 默认不带私有镜像
  - 默认不带任何密钥

### 验收标准

- `rg -n "WANDB_API_KEY|debugpy|/share/project|hf-mirror|bandw|0.0.0.0" .` 不再出现不应公开的默认值
- 所有脚本在没有私有机器路径的环境下也能看懂，并可通过参数补全

---

## Phase 2：训练 / 配置 / 脚本一致性清理

> 风险等级：🟡 中
> 目标：保证代码、yaml、脚本、注释都围绕当前真实主线，即 `train.py + implicit latent reasoning`。

### Tasks

1. 继续执行 [config_and_training_cleanup_plan.md](./config_and_training_cleanup_plan.md) 中与开源强相关的条目
2. 清理 yaml 中未使用或对外无意义的字段
3. 统一脚本和配置中的默认训练入口、训练阶段、stage 说明
4. 清理 README 中过时训练命令
5. 统一说明仓库当前主推荐路径：
   - Bridge 训练
   - LIBERO 训练
   - LIBERO 评测
   - SimplerEnv 评测

### 重点问题

- README 仍引用过时训练入口
- 若干 `__main__` 默认配置仍指向不存在的 legacy config
- yaml 注释仍混用旧入口名 / `cot_mode` 的历史描述

### 建议输出

- 一次 “cleanup: align train/config/scripts” commit
- 更新后的默认训练命令

### 验收标准

- README、脚本、yaml 的训练入口一致
- 新用户只需要看一份主文档就能知道“用哪个脚本训练”

---

## Phase 3：文档重写

> 风险等级：🟡 中
> 目标：把仓库从“内部知道怎么跑”变成“外部第一次打开也知道是什么、怎么装、怎么训练、怎么评测”。

### Tasks

1. 重写顶层 `README.md`
2. 明确写出仓库主卖点：
   - implicit latent reasoning
   - iterative forward
   - hidden state feedback as next reasoning token
3. 给出最小可运行路径：
   - 安装
   - 下载模型
   - 启动训练
   - 启动 policy server
   - 跑 LIBERO / SimplerEnv
4. 补齐空文档：
   - `examples/LIBERO/README.md`
5. Rewrite or remove internal-only documents that do not match the public release scope
6. 增加 FAQ：
   - 数据不公开时如何使用
   - 如何替换 base VLM
   - 如何关闭/开启 latent reasoning 相关能力

### 推荐文档结构

1. 项目简介
2. 亮点与方法概览
3. 安装
4. 快速开始
5. 训练
6. 评测
7. Data and checkpoint guidance
8. Limitations and known issues

### 建议输出

- 一次 “docs: rewrite public-facing documentation” commit

### 验收标准

- 外部用户仅通过 README 就能找到正确入口
- `examples/LIBERO/README.md` is no longer empty
- each example directory has at least one usable README

---

## Phase 4：许可证与第三方代码归属审计

> 风险等级：🔴 高
> 目标：明确仓库中每部分代码的来源、许可证和再分发边界。

### Tasks

1. 盘点所有非纯自研代码来源
2. 确认这些来源的许可证是否兼容当前仓库公开方式
3. 新增 `NOTICE` 或 `THIRD_PARTY_NOTICES.md`
4. 在 README 的 Acknowledgements 基础上补充更正式的归属说明
5. 检查文件头是否需要：
   - SPDX 标识
   - 原始仓库链接
   - 修改说明
6. 特别核查以下目录/文件：
   - `laravla/dataloader/gr00t_lerobot/*`
   - `laravla/model/modules/action_model/GR00T_ActionHeader.py`
   - `laravla/model/modules/action_model/LayerwiseFM_ActionHeader.py`
   - `laravla/model/modules/action_model/DiTActionHeader.py`
   - `laravla/model/modules/action_model/DiT_modules/models.py`
   - `laravla/training/trainer_utils/overwatch.py`

### 重点提醒

- 某些第三方文件头会引用“原仓库根目录的 LICENSE”
- the repository root license and file-level third-party notices must remain consistent
- 公开前必须确认这种再分发方式在法律和文档层面是自洽的

### 建议输出

- `THIRD_PARTY_NOTICES.md`
- 必要时补充 `NOTICE`
- 一次 “legal: add provenance and third-party notices” commit

### 验收标准

- 第三方来源可追溯
- 仓库根目录具备足够的许可证与归属说明

---

## Phase 5：最小验证与 CI

> 风险等级：🟡 中
> 目标：让仓库至少具备基础的可验证性，减少公开后的“装不上 / 一跑就挂 / 根本不知道哪里坏了”问题。

### Tasks

1. 增加最小 smoke tests
2. 增加 config load 检查
3. 增加一个 fake-data forward 检查
4. 增加脚本级 lint / import check
5. 建立最小 GitHub Actions

### 最小建议测试集

- `OmegaConf.load()` 三个训练 yaml
- `from laravla.training.train import main`
- `from laravla.model.framework import build_framework`
- `BridgeReasoningFormatter` 的 stage 格式化行为
- `QwenGR00T` fake sample smoke test

### CI 最小内容

- Python 版本检查
- `make check`
- smoke imports
- yaml syntax check

### 建议输出

- `.github/workflows/ci.yml`
- 一个 `tests/` 目录或 `scripts/smoke/` 目录
- 一次 “ci: add minimal smoke coverage” commit

### 验收标准

- PR 可以自动跑基本检查
- 新用户遇到问题时，仓库里有明确的最小验证命令

---

## Phase 6：发布资产准备

> 风险等级：🟡 中
> 目标：补齐开源项目在 GitHub 上应该具备的元信息和发布素材。

### Tasks

1. 清理和确认 `.gitignore`
2. 删除无意义或不应公开的文件
   - 例如根目录的异常临时文件
3. 完善 `pyproject.toml`
4. 明确 `requirements.txt` / optional dependencies 的角色
5. 增加社区文件：
   - `CONTRIBUTING.md`
   - `CODE_OF_CONDUCT.md`
   - `SECURITY.md`
6. 增加 issue / PR template
7. 准备 checkpoint / dataset 说明
8. 准备 Hugging Face model card / release note

### 推荐发布内容

- GitHub Release 文案
- 支持的 checkpoint 列表
- 最小显存 / 依赖说明
- 复现限制说明
- 已知未开源部分说明

### 验收标准

- 仓库元信息齐全
- 发布页和 README 信息一致

---

## Phase 7：发布彩排与正式开源

> 风险等级：🔴 高
> 目标：在真正公开之前，按外部用户视角完整跑一遍安装与使用流程。

### Tasks

1. 在干净环境中从零 clone 仓库
2. 按 README 完整执行安装流程
3. 执行最小 smoke tests
4. 验证至少一条训练命令可启动
5. 验证至少一条评测命令可启动
6. 检查 README 中所有路径、文件名、命令是否真实存在
7. 整理首发 tag：
   - `v0.1.0-research`
   - 或 `v1.0.0`（如果你希望直接按正式版本公开）

### 发布建议

- 首发版本建议命名为 `v0.1.0-research`
- 先强调：
  - 研究代码
  - implicit latent reasoning 主线
  - 提供训练/评测参考实现
- 暂时不要承诺过多平台和环境支持

### 验收标准

- 一位不了解内部环境的人也能按文档完成最小流程
- GitHub 首屏信息清晰
- 没有明显安全问题和法律风险残留

---

## 建议提交顺序

```bash
git commit -m "refactor: unify training entrypoint name as train"
git commit -m "sanitize: remove secrets, private paths, and debug entrypoints"
git commit -m "cleanup: align training configs, scripts, and docs to train.py"
git commit -m "docs: rewrite public-facing README and example docs"
git commit -m "legal: add third-party notices and provenance records"
git commit -m "ci: add minimal smoke tests and GitHub Actions"
git commit -m "release: prepare open-source metadata and community files"
```

---

## Release Checklist

### P0

- [ ] 所有密钥已移除
- [ ] 所有私有路径已移除
- [ ] 所有 debugpy 默认入口已移除
- [ ] `train.py` 已成为唯一训练入口
- [ ] README 不再引用过时入口

### P1

- [ ] 文档主线已统一
- [ ] `examples/LIBERO/README.md` 已补齐
- [ ] 发布所需许可证/NOTICE 文件已补齐
- [ ] 最小 smoke tests 已可运行

### P2

- [ ] CI 已接入
- [ ] 社区文件已补齐
- [ ] release note / model card 已准备好
- [ ] 首发版本已完成彩排

---

## 建议我们接下来的执行顺序

1. 先完成 Phase 0 剩余收尾，把所有旧入口文档残留统一改掉
2. 立刻进入 Phase 1，优先做 sanitize
3. 然后做 README 和 example 文档重写
4. 再处理第三方许可证与最小 CI
5. 最后做一次从零安装到运行的开源彩排

> 结论：从当前仓库状态看，**最适合的策略不是“一次性大改完再发”，而是按 Phase 连续小步提交**。这样风险最小，也最容易随时停在一个可发布状态。
