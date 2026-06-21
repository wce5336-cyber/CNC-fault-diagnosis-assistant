#!/usr/bin/env python3
"""
离线冒烟测试：验证数据、RAG 索引与评测脚本（不依赖 LLaMA-Factory）。

用法:
    python -m src.smoke_test
    python -m src.smoke_test --check-llm
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from .paths import CHROMA_DIR, PROCESSED, RAG_CHUNKS, SFT_JSON


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def check_files() -> bool:
    print("1. 检查产物文件")
    passed = True
    for path, desc in [
        (SFT_JSON, "SFT 微调数据"),
        (RAG_CHUNKS, "RAG 知识库"),
        (PROCESSED / "eval_rag_compare.json", "对比评测结果"),
    ]:
        if path.exists():
            ok(f"{desc}: {path.name}")
        else:
            fail(f"缺少 {desc}: {path}")
            passed = False
    return passed


def check_sft() -> bool:
    print("\n2. 检查 SFT 数据")
    records = json.loads(SFT_JSON.read_text(encoding="utf-8"))
    if len(records) < 500:
        fail(f"样本数不足: {len(records)}")
        return False
    ok(f"样本数 {len(records)}")
    sample = records[0]
    for key in ("instruction", "input", "output"):
        if key not in sample:
            fail(f"缺少字段 {key}")
            return False
    ok("Alpaca 字段完整")
    return True


def check_rag_index() -> bool:
    print("\n3. 检查 RAG 索引")
    if not CHROMA_DIR.exists():
        fail(f"ChromaDB 目录不存在，请运行: python -m src.build_rag_index --reset")
        return False
    try:
        from .rag_store import DEFAULT_COLLECTION, DEFAULT_EMBED_MODEL, RagRetriever

        retriever = RagRetriever(collection_name=DEFAULT_COLLECTION, embed_model=DEFAULT_EMBED_MODEL)
        count = retriever.collection.count()
        if count == 0:
            fail("ChromaDB 为空")
            return False
        ok(f"向量数 {count}")

        hits = retriever.search("FANUC PS300 报警", top_k=3)
        if not hits:
            fail("检索无结果")
            return False
        ok(f"检索返回 {len(hits)} 条")
        if any("PS300" in h.get("text", "") for h in hits):
            ok("Top-K 命中 PS300 相关片段")
        else:
            print("  [WARN] Top-3 未直接命中 PS300，可检查 embedding 或 query 格式")
        return True
    except Exception as exc:
        fail(str(exc))
        return False


def check_eval() -> bool:
    print("\n4. 检查评测脚本")
    eval_json = PROCESSED / "eval_rag_compare.json"
    if not eval_json.exists():
        fail("缺少 eval_rag_compare.json")
        return False
    try:
        from .eval_rag_compare import recompute_from_json

        summary = recompute_from_json(eval_json, update_readme=False)
        n = summary["num_samples"]
        rag = summary["rag_lora"]
        ok(f"重算指标成功 (n={n}, RAG 可用率 {rag['usable_rate']:.0%})")
        return True
    except Exception as exc:
        fail(str(exc))
        return False


def check_imports() -> bool:
    print("\n5. 检查模块导入")
    modules = [
        "src.process_cnc_datasets",
        "src.build_rag_index",
        "src.rag_chat",
        "src.demo_app",
        "src.eval_rag_compare",
    ]
    passed = True
    for name in modules:
        try:
            __import__(name)
            ok(name)
        except Exception as exc:
            fail(f"{name}: {exc}")
            passed = False
    return passed


def check_llm_port(host: str = "127.0.0.1", port: int = 7860) -> bool:
    print(f"\n6. 检查 LLaMA-Factory ({host}:{port})")
    try:
        with socket.create_connection((host, port), timeout=3):
            ok("端口可连接，可运行 demo_app.py / rag_chat.py")
            return True
    except OSError:
        print("  [SKIP] WebUI 未启动 — 推理/Demo 步骤需在 LLaMA-Factory 就绪后手动验证")
        return True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--check-llm", action="store_true")
    args = parser.parse_args()

    print("=== CNC 项目冒烟测试 ===\n")
    results = [
        check_files(),
        check_sft(),
        check_rag_index(),
        check_eval(),
        check_imports(),
    ]
    if args.check_llm:
        results.append(check_llm_port())

    print("\n=== 结果 ===")
    if all(results):
        print("离线流程通过。启动 LLaMA-Factory 后运行: python -m src.demo_app --skip-load")
        sys.exit(0)
    print("存在失败项，请根据上方 [FAIL] 提示修复。")
    sys.exit(1)


if __name__ == "__main__":
    main()
