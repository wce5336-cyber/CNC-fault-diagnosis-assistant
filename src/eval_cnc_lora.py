#!/usr/bin/env python3
"""Evaluate CNC LoRA checkpoint against cnc_diagnosis_sft.json via LLaMA-Factory WebUI API."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from gradio_client import Client

from .llm_client import DEFAULT_CHECKPOINT
from .paths import PROCESSED, SFT_JSON

DATASET_PATH = SFT_JSON
DEFAULT_OUTPUT = PROCESSED / "eval_lora_baseline.json"
SECTIONS = ("【可能原因】", "【排查步骤】", "【处理建议】")


def last_assistant_text(history) -> str:
    if not history:
        return ""
    for msg in reversed(history):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content", ""))
        if isinstance(msg, (list, tuple)) and len(msg) >= 2:
            return str(msg[1])
    last = history[-1]
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(last)


def load_model(client: Client, checkpoint: str) -> None:
    try:
        result = client.predict(
            "en",
            "Qwen2.5-7B-Instruct",
            "/root/autodl-tmp/llm/Qwen/Qwen2.5-7B-Instruct",
            "lora",
            [checkpoint],
            "none",
            "bnb",
            "default",
            "none",
            "auto",
            "huggingface",
            "auto",
            '{"vllm_enforce_eager": true}',
            api_name="/load_model",
        )
        if isinstance(result, str) and "unload" in result.lower():
            print("Model already loaded in WebUI, skip reload.")
            return
        print("Model load started, waiting 90s ...")
        time.sleep(90)
    except Exception as exc:
        print(f"Load skipped/failed: {exc}")


def chat_once(client: Client, query: str, max_new_tokens: int = 768) -> str:
    append_result = client.predict([], "user", query, True, api_name="/append")
    chatbot = append_result[0] if isinstance(append_result, tuple) else append_result
    final = None
    for update in client.submit(
        chatbot,
        "en",
        "",
        "",
        None,
        None,
        None,
        max_new_tokens,
        0.7,
        0.3,
        True,
        True,
        False,
        api_name="/stream",
    ):
        final = update
    return last_assistant_text(final)


def format_score(text: str) -> float:
    return sum(1 for s in SECTIONS if s in text) / len(SECTIONS)


def extract_alarm_code(text: str) -> str | None:
    m = re.search(r"报警代码[：:]\s*(\S+)", text)
    return m.group(1) if m else None


def token_set(text: str) -> set[str]:
    parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_./-]{3,}", text.lower())
    return set(parts)


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def classify_sample(sample: dict) -> str:
    inp = sample.get("input", "")
    if "FANUC" in inp:
        return "fanuc"
    if "Siemens" in inp or "840D" in inp:
        return "siemens"
    if "维护" in sample.get("instruction", "") or "维护节点" in inp:
        return "maintenance"
    if "传感器" in inp or "failure" in inp.lower() or "ai4i" in inp.lower():
        return "sensor"
    return "other"


def build_user_query(sample: dict) -> str:
    instruction = sample.get("instruction", "").strip()
    user_input = sample.get("input", "").strip()
    if user_input:
        return f"{instruction}\n\n{user_input}"
    return instruction


@dataclass
class EvalItem:
    index: int
    category: str
    alarm_code: str | None
    instruction: str
    input: str
    reference: str
    prediction: str
    format_score: float
    jaccard: float
    ref_len: int
    pred_len: int


def sample_records(records: list[dict], n: int, seed: int) -> list[tuple[int, dict]]:
    rng = random.Random(seed)
    buckets: dict[str, list[tuple[int, dict]]] = {}
    for i, rec in enumerate(records):
        buckets.setdefault(classify_sample(rec), []).append((i, rec))

    per_bucket = max(1, n // max(len(buckets), 1))
    picked: list[tuple[int, dict]] = []
    for _, items in sorted(buckets.items()):
        rng.shuffle(items)
        picked.extend(items[:per_bucket])

    if len(picked) < n:
        seen = {x for x, _ in picked}
        rest = [(i, r) for i, r in enumerate(records) if i not in seen]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])

    rng.shuffle(picked)
    return picked[:n]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-load", action="store_true")
    args = parser.parse_args()

    records = json.loads(args.dataset.read_text(encoding="utf-8"))
    selected = sample_records(records, args.num_samples, args.seed)

    client = Client("http://127.0.0.1:7860/")
    if not args.skip_load:
        load_model(client, args.checkpoint)

    results: list[EvalItem] = []
    for idx, (rec_idx, sample) in enumerate(selected, 1):
        query = build_user_query(sample)
        cat = classify_sample(sample)
        print(f"\n[{idx}/{len(selected)}] record #{rec_idx} ({cat}) ...")
        try:
            prediction = chat_once(client, query)
        except Exception as exc:
            prediction = f"[ERROR] {exc}"

        ref = sample["output"]
        item = EvalItem(
            index=rec_idx,
            category=cat,
            alarm_code=extract_alarm_code(sample.get("input", "")),
            instruction=sample.get("instruction", ""),
            input=sample.get("input", ""),
            reference=ref,
            prediction=prediction,
            format_score=format_score(prediction),
            jaccard=jaccard(ref, prediction),
            ref_len=len(ref),
            pred_len=len(prediction),
        )
        results.append(item)
        print(
            f"  format={item.format_score:.0%}  jaccard={item.jaccard:.3f}  "
            f"pred_len={item.pred_len}"
        )
        print(f"  preview: {prediction[:120].replace(chr(10), ' ')}")

    summary = {
        "adapter": args.checkpoint,
        "dataset": str(args.dataset),
        "num_samples": len(results),
        "avg_format_score": sum(r.format_score for r in results) / len(results),
        "avg_jaccard": sum(r.jaccard for r in results) / len(results),
        "format_pass_rate": sum(1 for r in results if r.format_score == 1.0) / len(results),
        "by_category": {},
        "items": [asdict(r) for r in results],
    }
    by_cat: dict[str, list[EvalItem]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    for cat, items in by_cat.items():
        summary["by_category"][cat] = {
            "count": len(items),
            "avg_format_score": sum(x.format_score for x in items) / len(items),
            "avg_jaccard": sum(x.jaccard for x in items) / len(items),
        }

    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"samples:          {summary['num_samples']}")
    print(f"format pass rate: {summary['format_pass_rate']:.0%}")
    print(f"avg format score: {summary['avg_format_score']:.0%}")
    print(f"avg jaccard:      {summary['avg_jaccard']:.3f}")
    print(f"report:           {args.output}")


if __name__ == "__main__":
    main()
