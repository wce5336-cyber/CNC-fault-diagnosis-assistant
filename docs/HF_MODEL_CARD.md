---
license: mit
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- qwen2.5
- cnc
- fault-diagnosis
- manufacturing
- chinese
language:
- zh
- en
datasets:
- custom
---

# CNC 故障诊断 LoRA · Qwen2.5-7B-Instruct

**CNC Fault Diagnosis Assistant** — LoRA adapter for structured CNC machine fault diagnosis.

输入机床系统、报警代码与故障现象，模型输出结构化诊断报告：

```
【可能原因】
【排查步骤】
【处理建议】
```

> 作者：**WC**  
> 推荐与 **RAG（ChromaDB + 报警手册检索）** 联合使用，效果更佳。

---

## 模型说明

本仓库为 [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) 的 **LoRA 适配器**（非完整模型）。

| 项目 | 说明 |
|------|------|
| 任务 | 数控机床故障诊断 / 报警解读 |
| 微调框架 | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| 数据格式 | Alpaca（instruction / input / output） |
| 覆盖系统 | FANUC、Siemens 840D、中文维修手册等 |

**设计思路：** LoRA 负责学习诊断**格式与表达**，RAG 负责补充**手册与报警原文**。

---

## 训练配置

| 参数 | 值 |
|------|-----|
| 基座模型 | Qwen2.5-7B-Instruct |
| 微调方法 | LoRA |
| LoRA rank / alpha | 8 / 16 |
| LoRA target | all（q/k/v/o_proj, gate/up/down_proj） |
| 训练数据 | `cnc_diagnosis_sft`（~1440 条）+ identity |
| Epochs | 2 |
| Learning rate | 5e-5 |
| Batch size | 2 × 8 grad accum |
| Cutoff len | 2048 |
| 精度 | bf16 |
| Chat template | qwen |

---

## 快速使用

### LLaMA-Factory（推荐）

1. 下载本仓库到本地
2. 打开 WebUI **Chat** 页
3. 基座模型：`Qwen2.5-7B-Instruct`
4. 微调方法：**LoRA**
5. Checkpoint：选择本仓库目录

### transformers + peft

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = "Qwen/Qwen2.5-7B-Instruct"
adapter = "wc8084/cnc-qwen2.5-7b-lora"

tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(model, adapter)
model.eval()

prompt = """请根据以下机床故障信息进行诊断分析，给出可能原因、排查步骤和处理建议。

机床系统：FANUC CNC
报警代码：PS300
故障现象：ILLEGAL"""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=768, temperature=0.3, top_p=0.7)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### 推荐 System Prompt

```
你是 CNC故障诊断智能助手，由 WC 开发，专注于数控机床报警解读、故障诊断与维修建议。
请严格依据参考知识进行分析，输出必须包含【可能原因】【排查步骤】【处理建议】。
不要称自己为通义千问或 Qwen。
```

---

## 推理示例

**输入：**
```
机床系统：FANUC CNC
报警代码：PS300
故障现象：ILLEGAL
```

**输出（节选）：**
```
【可能原因】
1. 报警含义：ILLEGAL
2. COMMAND IN SCALING — scaling 过程中指定了非法 G 代码

【排查步骤】
1. 在 FANUC 诊断界面确认报警号 PS300
2. 记录报警发生时的程序段与操作模式
...

【处理建议】
1. 修改程序中的 scaling 相关 G 代码
2. 修复后执行 RESET 复位
...
```

---

## 效果参考

| 模式 | 三段式格式通过率 | 备注 |
|------|------------------|------|
| 仅 LoRA | ~33% | Siemens 较好，FANUC 偶发重复生成 |
| **RAG + LoRA** | ~67% | 检索手册后输出更稳定 |

> 配合 ChromaDB + `bge-small-zh-v1.5` 检索 FANUC/Siemens 报警手册，FANUC 类问题改善明显。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `adapter_model.safetensors` | LoRA 权重（~77 MB） |
| `adapter_config.json` | LoRA 配置 |
| `tokenizer.json` | 分词器（与基座一致） |
| `chat_template.jinja` | Qwen 对话模板 |

---

## 局限与免责声明

- 本模型**不能替代**机床厂商官方维修手册与持证工程师判断
- FANUC 部分报警文本来源于公开资料解析，可能存在截断或不完整
- 自我认知建议通过 System Prompt 固定，勿仅依赖微调覆盖基座身份
- 适用于辅助诊断与学习参考，生产环境请结合 RAG 与人工复核

---

## 引用

如在项目或论文中使用，请注明：

```
CNC Fault Diagnosis LoRA (Qwen2.5-7B-Instruct)
Author: WC
https://huggingface.co/wc8084/cnc-qwen2.5-7b-lora
Base model: Qwen/Qwen2.5-7B-Instruct
```

---

## License

MIT
