# LoRA 权重（Hugging Face）

**模型地址：** https://huggingface.co/wc8084/cnc-qwen2.5-7b-lora

## 已发布文件

- `adapter_model.safetensors` — LoRA 权重
- `adapter_config.json` — LoRA 配置
- `tokenizer.json` / `tokenizer_config.json` / `chat_template.jinja`

## 下载

```bash
hf download wc8084/cnc-qwen2.5-7b-lora --local-dir ./models/qwen2.5-7b-lora
```

或在 Hugging Face 仓库页面 → **Files and versions** → 手动下载。

## 基座模型

[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

## LLaMA-Factory 加载

1. 下载上述仓库到本地
2. WebUI Chat → 基座：`Qwen2.5-7B-Instruct`
3. 微调方法：LoRA → Checkpoint 选择下载目录

## transformers + peft

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-7B-Instruct"
adapter = "wc8084/cnc-qwen2.5-7b-lora"

tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
```

## 更新 Model Card

可在 Hugging Face 仓库 **README.md** 中补充训练说明；本地完整版见 `qwen2.5-7b-lora/README.md`，可复制到 HF 仓库编辑页。
