#!/usr/bin/env python3
"""
CNC 故障诊断智能助手 — Gradio Demo

展示 RAG 检索片段 + LoRA 微调模型诊断结果。

用法:
    python demo_app.py
    python demo_app.py --share          # 生成公网链接（临时）
    python demo_app.py --skip-load      # WebUI 已加载 LoRA 时
"""

from __future__ import annotations

import argparse

import gradio as gr

from .llm_client import DEFAULT_CHECKPOINT, LlmFactoryClient
from .rag_chat import SYSTEM_PROMPT, diagnose
from .rag_store import DEFAULT_COLLECTION, DEFAULT_EMBED_MODEL, RagRetriever

EXAMPLE = """机床系统：FANUC CNC
报警代码：PS300
故障现象：ILLEGAL"""

_llm: LlmFactoryClient | None = None
_retriever: RagRetriever | None = None


def init_services(checkpoint: str, llm_url: str, skip_load: bool) -> str:
    global _llm, _retriever
    _retriever = RagRetriever(collection_name=DEFAULT_COLLECTION, embed_model=DEFAULT_EMBED_MODEL)
    if _retriever.collection.count() == 0:
        return "错误：ChromaDB 索引为空，请先运行 python build_rag_index.py"

    _llm = LlmFactoryClient(llm_url)
    if not skip_load:
        _llm.load_lora(checkpoint=checkpoint)
    return f"就绪 | 知识库 {_retriever.collection.count()} 条 | LoRA `{checkpoint}`"


def format_hits(hits: list) -> str:
    if not hits:
        return "*未检索到相关片段*"
    parts: list[str] = []
    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata", {})
        code = meta.get("code", "")
        typ = meta.get("type", "")
        dist = hit.get("distance", 0)
        title = f"**片段 {i}** · `{typ}`"
        if code:
            title += f" · 报警 `{code}`"
        title += f" · 距离 `{dist:.4f}`"
        parts.append(f"{title}\n\n{hit['text'][:1200]}")
    return "\n\n---\n\n".join(parts)


def run_diagnosis(fault_text: str, use_rag: bool, top_k: int) -> tuple[str, str]:
    if not fault_text.strip():
        return "*请输入故障信息*", ""
    if _llm is None:
        return "*服务未初始化*", ""

    result = diagnose(
        _llm,
        _retriever if use_rag else None,
        fault_text.strip(),
        top_k=int(top_k),
        use_rag=use_rag,
    )
    hits_md = format_hits(result["retrieved"]) if use_rag else "*RAG 已关闭*"
    mode = "RAG + LoRA" if use_rag else "仅 LoRA"
    answer = f"**模式：** {mode}\n\n{result['answer']}"
    return hits_md, answer


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="CNC 故障诊断智能助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# CNC 故障诊断智能助手\n"
            "微调 LoRA 负责**结构化诊断输出**，ChromaDB RAG 负责**检索报警手册知识**。\n\n"
            f"System Prompt 身份：**CNC故障诊断智能助手**（WC）"
        )
        with gr.Row():
            fault = gr.Textbox(
                label="故障信息",
                placeholder="输入机床系统、报警代码、故障现象…",
                lines=8,
                value=EXAMPLE,
            )
        with gr.Row():
            use_rag = gr.Checkbox(label="启用 RAG 检索", value=True)
            top_k = gr.Slider(1, 10, value=5, step=1, label="检索 Top-K")
            btn = gr.Button("开始诊断", variant="primary")
        with gr.Row():
            with gr.Column():
                hits_out = gr.Markdown(label="检索参考片段")
            with gr.Column():
                diag_out = gr.Markdown(label="诊断结果")
        btn.click(run_diagnosis, [fault, use_rag, top_k], [hits_out, diag_out])
        with gr.Accordion("System Prompt", open=False):
            gr.Markdown(f"```\n{SYSTEM_PROMPT}\n```")
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--llm-url", default="http://127.0.0.1:7860")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    status = init_services(args.checkpoint, args.llm_url, args.skip_load)
    print(status)
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
