#!/usr/bin/env python3
"""
CNC 故障诊断：RAG（ChromaDB）+ 微调 LoRA 联合推理。

用法:
    python build_rag_index.py              # 首次需建索引
    python rag_chat.py --query "..."       # 单次诊断
    python rag_chat.py --interactive       # 交互模式
    python rag_chat.py --no-rag --query "..."  # 对比：仅 LoRA
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .llm_client import DEFAULT_CHECKPOINT, LlmFactoryClient
from .rag_store import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION, DEFAULT_EMBED_MODEL, RagRetriever

SYSTEM_PROMPT = (
    "你是 CNC故障诊断智能助手，由 WC 开发，专注于数控机床报警解读、故障诊断与维修建议。"
    "请严格依据用户提供的检索参考知识进行分析；输出必须包含"
    "【可能原因】【排查步骤】【处理建议】三个部分。"
    "不要称自己为通义千问或 Qwen。"
)

INSTRUCTION_DEFAULT = (
    "请根据以下机床故障信息进行诊断分析，给出可能原因、排查步骤和处理建议。"
)


def build_user_message(
    user_query: str,
    context: str,
    use_rag: bool = True,
) -> str:
    if use_rag:
        return (
            f"{INSTRUCTION_DEFAULT}\n\n"
            f"【检索参考知识】\n{context}\n\n"
            f"【故障信息】\n{user_query}\n\n"
            "请结合检索参考知识，按三段式格式给出诊断。"
        )
    return f"{INSTRUCTION_DEFAULT}\n\n{user_query}"


def diagnose(
    llm: LlmFactoryClient,
    retriever: RagRetriever | None,
    user_query: str,
    top_k: int = 5,
    use_rag: bool = True,
    max_new_tokens: int = 768,
) -> dict:
    hits: list = []
    context = ""
    if use_rag and retriever is not None:
        hits = retriever.search(user_query, top_k=top_k)
        context = RagRetriever.format_context(hits)

    prompt = build_user_message(user_query, context, use_rag=use_rag)
    answer = llm.chat(prompt, system=SYSTEM_PROMPT, max_new_tokens=max_new_tokens)
    return {
        "query": user_query,
        "use_rag": use_rag,
        "retrieved": hits,
        "context": context,
        "answer": answer,
    }


def print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    if result["use_rag"]:
        print("【检索到的参考片段】")
        if result["retrieved"]:
            for i, hit in enumerate(result["retrieved"], 1):
                meta = hit.get("metadata", {})
                print(f"\n--- 片段 {i} ({meta.get('type', '')} distance={hit.get('distance', 0):.4f}) ---")
                preview = hit["text"][:400].replace("\n", " ")
                print(preview + ("..." if len(hit["text"]) > 400 else ""))
        else:
            print("（无）")
    print("\n【模型诊断结果】")
    print(result["answer"])
    print("=" * 60 + "\n")


def interactive_loop(
    llm: LlmFactoryClient,
    retriever: RagRetriever | None,
    top_k: int,
    use_rag: bool,
) -> None:
    mode = "RAG + LoRA" if use_rag else "LoRA only"
    print(f"进入交互模式（{mode}），输入 quit 退出。")
    print("可直接粘贴故障信息，例如：")
    print("  机床系统：FANUC CNC\\n报警代码：PS300\\n故障现象：ILLEGAL")
    while True:
        try:
            text = input("\n故障信息> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit", "q"}:
            print("再见。")
            break
        result = diagnose(llm, retriever, text, top_k=top_k, use_rag=use_rag)
        print_result(result)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="CNC RAG + LoRA diagnosis")
    parser.add_argument("--query", "-q", help="单次诊断输入（可多行，用 \\n）")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rag", action="store_true", help="关闭 RAG，仅使用微调模型")
    parser.add_argument("--skip-load", action="store_true", help="跳过 LoRA 加载（WebUI 已加载时）")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--llm-url", default="http://127.0.0.1:7860")
    args = parser.parse_args()

    use_rag = not args.no_rag
    retriever = None
    if use_rag:
        retriever = RagRetriever(
            chroma_dir=args.chroma_dir,
            collection_name=DEFAULT_COLLECTION,
            embed_model=args.embed_model,
        )
        if retriever.collection.count() == 0:
            print("ChromaDB 索引为空，请先运行: python build_rag_index.py")
            sys.exit(1)
        print(f"RAG 索引已加载，共 {retriever.collection.count()} 条向量。")

    try:
        llm = LlmFactoryClient(args.llm_url)
    except ConnectionError as exc:
        print(exc)
        sys.exit(1)
    if not args.skip_load:
        print(f"加载 LoRA checkpoint: {args.checkpoint}")
        llm.load_lora(checkpoint=args.checkpoint)

    if args.interactive:
        interactive_loop(llm, retriever, args.top_k, use_rag)
        return

    if not args.query:
        parser.error("请提供 --query 或使用 --interactive")

    query = args.query.replace("\\n", "\n")
    result = diagnose(
        llm,
        retriever,
        query,
        top_k=args.top_k,
        use_rag=use_rag,
        max_new_tokens=args.max_new_tokens,
    )
    print_result(result)


if __name__ == "__main__":
    main()
