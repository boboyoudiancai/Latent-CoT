<h1 align="center">Our Progresses Tracing</h1>

## 1. Lara-VLA 结果复现

### 训练方案

* **VLM 阶段**

  * Loss：`CoT Loss` + `Action Token Loss` + `Next Image Loss`

* **Action Head 阶段**

  * Batch Size：`112`
  * 训练步数：`24K`
  * Loss：`DiT Action Loss`
  * VLM：**不冻结**

### 实验结果

| 测试集         |     成功率 |
| ----------- | ------: |
| LIBERO-Plus | **70%** |

---

## 2. 去除 CoT 的 Lara-VLA

在 Lara-VLA codebase 中移除 CoT 相关内容，直接使用 `Qwen3-VL-Instruct-Action` 训练 Action Head。

### 训练方案

* Loss：`DiT Action Loss`
* Batch Size：`112`

### 实验结果

| 训练步数 | LIBERO-Plus 成功率 |
| ---: | --------------: |
|  24K |         **68%** |
|  40K |         **71%** |

---

## 3. 显式 CoT 上限实验

先训练 VLM 输出显式 CoT，再将生成的 CoT 文本回填到 Hidden States 中，作为 Action Head 的输入。

### 训练方案

* **VLM 阶段**

  * 训练步数：`8K`
  * Loss：`CoT Loss`
  * 不使用 `Action Token Loss`
  * 不使用 `Next Image Loss`

* **Action Head 阶段**

  * Batch Size：`112`
  * Loss：`DiT Action Loss`

### 实验结果

| 训练步数 | LIBERO-Plus 成功率 |
| ---: | --------------: |
|  16K |         **71%** |
|  20K |         **74%** |

> 由于 GPU 资源临时被占用，训练在 20K 时中止。

---

## 显存限制

为避免 VLM 输出过长导致显存溢出，对输入长度进行限制：

```yaml
vlm_max_output_tokens: 128
action_head_max_vlm_hidden_tokens: 320
```
