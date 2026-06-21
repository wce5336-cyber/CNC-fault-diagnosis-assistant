#!/usr/bin/env python3
"""
对比评测：仅 LoRA vs RAG+LoRA。

用法:
    python -m src.eval_rag_compare --num-samples 40 --skip-load
    python -m src.eval_rag_compare --recompute processed/eval_rag_compare.json
    python -m src.eval_rag_compare --recompute processed/eval_rag_compare.json --update-readme
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

from .eval_utils import (
    CATEGORY_LABELS,
    CompareResult,
    aggregate_by_category,
    aggregate_mode,
    build_user_query,
    classify_sample,
    extract_alarm_code,
    sample_records,
)
from .paths import PROCESSED, ROOT, SFT_JSON

DEFAULT_OUTPUT = PROCESSED / "eval_rag_compare.json"
DEFAULT_REPORT = PROCESSED / "eval_rag_compare_report.md"
DEFAULT_SNIPPET = PROCESSED / "eval_readme_snippet.md"
README_PATH = ROOT / "README.md"
README_MARKER_START = "<!-- EVAL:START -->"
README_MARKER_END = "<!-- EVAL:END -->"


def load_results_from_json(data: dict) -> list[CompareResult]:
    items = data.get("items", [])
    results: list[CompareResult] = []
    for row in items:
        results.append(
            CompareResult.from_predictions(
                index=row["index"],
                category=row["category"],
                alarm_code=row.get("alarm_code"),
                reference=row["reference"],
                lora_only=row["lora_only"],
                rag_lora=row["rag_lora"],
            )
        )
    return results


def build_summary(
    results: list[CompareResult],
    adapter: str,
    seed: int | None = None,
) -> dict:
    lora_agg = aggregate_mode(results, "lora")
    rag_agg = aggregate_mode(results, "rag")
    return {
        "adapter": adapter,
        "num_samples": len(results),
        "seed": seed,
        "lora_only": lora_agg,
        "rag_lora": rag_agg,
        "by_category": aggregate_by_category(results),
        "items": [asdict(r) for r in results],
    }


def _pct(v: float) -> str:
    return f"{v:.0%}"


def _delta(a: float, b: float, higher_is_better: bool = True) -> str:
    d = b - a if higher_is_better else a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.0%}"


def _fmt_alarm(rate: float | None) -> str:
    return _pct(rate) if rate is not None else "—"


def write_report(summary: dict, path: Path) -> None:
    lora = summary["lora_only"]
    rag = summary["rag_lora"]
    seed_line = f"- 随机种子: {summary['seed']}" if summary.get("seed") is not None else ""

    lines = [
        "# LoRA vs RAG+LoRA 对比评测",
        "",
        f"- 样本数: {summary['num_samples']}（分层抽样，覆盖 FANUC / Siemens / 维护 / 传感器等类别）",
        f"- Adapter: `{summary.get('adapter', 'cnc-qwen2.5-7b-lora')}`",
        seed_line,
        "",
        "## 总体指标",
        "",
        "| 指标 | 仅 LoRA | RAG + LoRA | 提升 |",
        "|------|---------|------------|------|",
        f"| **综合可用率**¹ | {_pct(lora['usable_rate'])} | **{_pct(rag['usable_rate'])}** | {_delta(lora['usable_rate'], rag['usable_rate'])} |",
        f"| 三段式格式通过率 | {_pct(lora['format_pass_rate'])} | **{_pct(rag['format_pass_rate'])}** | {_delta(lora['format_pass_rate'], rag['format_pass_rate'])} |",
        f"| 关键词召回率² | {_pct(lora['avg_keyword_recall'])} | **{_pct(rag['avg_keyword_recall'])}** | {_delta(lora['avg_keyword_recall'], rag['avg_keyword_recall'])} |",
        f"| 报警码命中率³ | {_fmt_alarm(lora['alarm_hit_rate'])} | **{_fmt_alarm(rag['alarm_hit_rate'])}** | — |",
        f"| 循环重复率⁴ | {_pct(lora['loop_rate'])} | **{_pct(rag['loop_rate'])}** | {_delta(lora['loop_rate'], rag['loop_rate'], higher_is_better=False)} |",
        f"| 平均输出长度 | {lora['avg_len']:.0f} 字 | {rag['avg_len']:.0f} 字 | — |",
        "",
        "## 按类别（综合可用率）",
        "",
        "| 类别 | n | 仅 LoRA | RAG + LoRA |",
        "|------|---|---------|------------|",
    ]
    for _cat, stats in summary.get("by_category", {}).items():
        label = stats.get("label", _cat)
        lines.append(
            f"| {label} | {stats['count']} | {_pct(stats['lora_usable_rate'])} | **{_pct(stats['rag_usable_rate'])}** |"
        )

    lines.extend([
        "",
        "## 指标说明",
        "",
        "1. **综合可用率**：三段式完整 + 无循环重复 + 无超长输出 +（有报警码时）输出含对应报警码。",
        "2. **关键词召回率**：标准答案中关键术语（报警号、参数名、核心故障词）在生成结果中的覆盖率。",
        "3. **报警码命中率**：仅统计含报警代码的样本。",
        "4. **循环重复率**：检测同一句/同一段落重复 ≥3 次，或 n-gram 循环，比单纯长度阈值更准确。",
        "",
        f"> Jaccard 词面重叠（LoRA {_pct(lora['avg_jaccard'])} → RAG {_pct(rag['avg_jaccard'])}）仍记录在 JSON 中，"
        "因开放生成同义改写较多，不作为主展示指标。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def category_insight(summary: dict) -> str:
    improved: list[tuple[str, float]] = []
    weak: list[str] = []
    for stats in summary.get("by_category", {}).values():
        label = stats.get("label", "")
        gain = stats["rag_usable_rate"] - stats["lora_usable_rate"]
        if gain > 0:
            improved.append((label, gain))
        if stats["rag_usable_rate"] < 0.5:
            weak.append(label)
    improved.sort(key=lambda x: x[1], reverse=True)
    parts: list[str] = []
    if improved:
        parts.append(f"RAG 在 {' / '.join(x[0] for x in improved[:2])} 提升最明显")
    if weak:
        parts.append(f"{' / '.join(weak)} 仍需优化")
    return "；".join(parts) + "。" if parts else ""


def write_readme_snippet(summary: dict, path: Path) -> None:
    lora = summary["lora_only"]
    rag = summary["rag_lora"]
    n = summary["num_samples"]
    insight = category_insight(summary)

    lines = [
        f"基于分层抽样 **n={n}** 的对比评测（详见 [eval_rag_compare_report.md](processed/eval_rag_compare_report.md)）：",
        "",
        "| 指标 | 仅 LoRA | RAG + LoRA |",
        "|------|---------|------------|",
        f"| 综合可用率¹ | {_pct(lora['usable_rate'])} | **{_pct(rag['usable_rate'])}** |",
        f"| 三段式格式通过率 | {_pct(lora['format_pass_rate'])} | **{_pct(rag['format_pass_rate'])}** |",
        f"| 关键词召回率² | {_pct(lora['avg_keyword_recall'])} | **{_pct(rag['avg_keyword_recall'])}** |",
        f"| 循环重复率³ | {_pct(lora['loop_rate'])} | **{_pct(rag['loop_rate'])}** |",
        "",
        f"¹ 综合可用率 = 格式完整 + 无循环重复 + 长度正常 + 报警码命中（如有）。",
        "",
    ]
    if insight:
        lines.append(f"**按类别：** {insight}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def patch_readme(summary: dict, readme_path: Path = README_PATH) -> None:
    snippet_path = DEFAULT_SNIPPET
    write_readme_snippet(summary, snippet_path)
    snippet = snippet_path.read_text(encoding="utf-8").strip()

    text = readme_path.read_text(encoding="utf-8")
    if README_MARKER_START in text and README_MARKER_END in text:
        new_block = f"{README_MARKER_START}\n{snippet}\n{README_MARKER_END}"
        text = re.sub(
            rf"{re.escape(README_MARKER_START)}.*?{re.escape(README_MARKER_END)}",
            new_block,
            text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        pattern = (
            r"(## 评测结果\n\n运行对比评测：\n\n```bash\npython eval_rag_compare\.py[^\n]*\n```\n\n)"
            r"(?:.*?\n\n)?(?=## 数据集|\Z)"
        )
        replacement = (
            "## 评测结果\n\n"
            "运行对比评测：\n\n"
            "```bash\n"
            "python -m src.eval_rag_compare --num-samples 40 --skip-load\n"
            "python -m src.eval_rag_compare --recompute processed/eval_rag_compare.json --update-readme\n"
            "```\n\n"
            f"{README_MARKER_START}\n{snippet}\n{README_MARKER_END}\n\n"
        )
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
        if n == 0:
            raise RuntimeError("无法在 README 中定位评测结果段落")

    readme_path.write_text(text, encoding="utf-8")


def recompute_from_json(json_path: Path, update_readme: bool) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = load_results_from_json(data)
    summary = build_summary(
        results,
        adapter=data.get("adapter", "cnc-qwen2.5-7b-lora"),
    )
    write_report(summary, DEFAULT_REPORT)
    write_readme_snippet(summary, DEFAULT_SNIPPET)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if update_readme:
        patch_readme(summary)
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", type=Path, default=SFT_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm-url", default="http://127.0.0.1:7860")
    parser.add_argument("--recompute", type=Path, default=None, help="从已有 JSON 重算指标")
    parser.add_argument("--update-readme", action="store_true", help="将评测摘要写回 README")
    args = parser.parse_args()

    if args.recompute:
        summary = recompute_from_json(args.recompute, args.update_readme)
        lora, rag = summary["lora_only"], summary["rag_lora"]
        print("=== Recomputed ===")
        print(f"usable:  LoRA {_pct(lora['usable_rate'])}  RAG+LoRA {_pct(rag['usable_rate'])}")
        print(f"format:  LoRA {_pct(lora['format_pass_rate'])}  RAG+LoRA {_pct(rag['format_pass_rate'])}")
        print(f"keyword: LoRA {_pct(lora['avg_keyword_recall'])}  RAG+LoRA {_pct(rag['avg_keyword_recall'])}")
        print(f"loop:    LoRA {_pct(lora['loop_rate'])}  RAG+LoRA {_pct(rag['loop_rate'])}")
        print(f"Report:  {DEFAULT_REPORT}")
        if args.update_readme:
            print(f"README:  {README_PATH}")
        return

    from .llm_client import DEFAULT_CHECKPOINT, LlmFactoryClient
    from .rag_chat import diagnose
    from .rag_store import DEFAULT_COLLECTION, DEFAULT_EMBED_MODEL, RagRetriever

    if not args.checkpoint:
        args.checkpoint = DEFAULT_CHECKPOINT

    records = json.loads(args.dataset.read_text(encoding="utf-8"))
    selected = sample_records(records, args.num_samples, args.seed)

    retriever = RagRetriever(collection_name=DEFAULT_COLLECTION, embed_model=DEFAULT_EMBED_MODEL)
    if retriever.collection.count() == 0:
        print("ChromaDB 为空，请先运行: python -m src.build_rag_index --reset")
        sys.exit(1)

    llm = LlmFactoryClient(args.llm_url)
    if not args.skip_load:
        llm.load_lora(checkpoint=args.checkpoint)

    results: list[CompareResult] = []
    for idx, (rec_idx, sample) in enumerate(selected, 1):
        query = build_user_query(sample)
        cat = classify_sample(sample)
        ref = sample["output"]
        alarm = extract_alarm_code(sample.get("input", ""))
        print(f"\n[{idx}/{len(selected)}] #{rec_idx} ({CATEGORY_LABELS.get(cat, cat)}) ...")

        try:
            lora_res = diagnose(llm, None, query, use_rag=False)
            rag_res = diagnose(llm, retriever, query, top_k=args.top_k, use_rag=True)
            item = CompareResult.from_predictions(
                rec_idx, cat, alarm, ref, lora_res["answer"], rag_res["answer"]
            )
        except Exception as exc:
            err = f"[ERROR] {exc}"
            item = CompareResult.from_predictions(rec_idx, cat, alarm, ref, err, err)

        results.append(item)
        print(
            f"  LoRA     usable={item.lora_usable} format={item.lora_format:.0%} "
            f"kw={item.lora_keyword_recall:.0%} loop={item.lora_loop}"
        )
        print(
            f"  RAG+LoRA usable={item.rag_usable} format={item.rag_format:.0%} "
            f"kw={item.rag_keyword_recall:.0%} loop={item.rag_loop}"
        )

    summary = build_summary(results, args.checkpoint, seed=args.seed)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, args.report)
    write_readme_snippet(summary, DEFAULT_SNIPPET)
    if args.update_readme:
        patch_readme(summary)

    lora, rag = summary["lora_only"], summary["rag_lora"]
    print("\n=== Summary ===")
    print(f"usable:  LoRA {_pct(lora['usable_rate'])}  RAG+LoRA {_pct(rag['usable_rate'])}")
    print(f"format:  LoRA {_pct(lora['format_pass_rate'])}  RAG+LoRA {_pct(rag['format_pass_rate'])}")
    print(f"keyword: LoRA {_pct(lora['avg_keyword_recall'])}  RAG+LoRA {_pct(rag['avg_keyword_recall'])}")
    print(f"loop:    LoRA {_pct(lora['loop_rate'])}  RAG+LoRA {_pct(rag['loop_rate'])}")
    print(f"JSON:    {args.output}")
    print(f"Report:  {args.report}")
    if args.update_readme:
        print(f"README:  {README_PATH}")


if __name__ == "__main__":
    main()
