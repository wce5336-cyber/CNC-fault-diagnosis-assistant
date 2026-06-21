"""Shared evaluation helpers."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

SECTIONS = ("【可能原因】", "【排查步骤】", "【处理建议】")
RUNAWAY_LEN = 1500
IDEAL_MAX_LEN = 1200
LOOP_MIN_PHRASE = 24
LOOP_MIN_COUNT = 3


def format_score(text: str) -> float:
    return sum(1 for s in SECTIONS if s in text) / len(SECTIONS)


def format_pass(text: str) -> bool:
    return format_score(text) == 1.0


def token_set(text: str) -> set[str]:
    parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_./$-]{3,}", text.lower())
    return set(parts)


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def extract_key_terms(text: str) -> set[str]:
    """Reference-side key terms for recall (skip generic template phrases)."""
    skip = {
        "可能原因", "排查步骤", "处理建议", "报警含义", "系统反应",
        "按手册描述逐项排除故障原因", "修复后执行报警复位", "空运转验证",
        "恢复加工前做首件确认", "记录报警发生时的程序段与操作模式",
        "查阅", "维修手册", "对应章节", "检查与该报警相关的硬件与参数",
    }
    terms: set[str] = set()
    for m in re.finditer(r"报警代码[：:]\s*(\S+)", text):
        terms.add(m.group(1).lower())
    for m in re.finditer(r"(?:PS|SV|SP|OH|OT|SR|DS|IE|BG)\d+", text, re.I):
        terms.add(m.group(0).upper())
    for m in re.finditer(r"\b\d{4,5}\b", text):
        terms.add(m.group(0))
    for m in re.finditer(r"[\u4e00-\u9fff]{3,8}", text):
        t = m.group(0)
        if t not in skip and not t.endswith("步骤") and not t.endswith("建议"):
            terms.add(t)
    for m in re.finditer(r"\$[A-Z_][A-Z0-9_]*", text):
        terms.add(m.group(0))
    return terms


def keyword_recall(reference: str, prediction: str) -> float:
    ref_terms = extract_key_terms(reference)
    if not ref_terms:
        return 1.0
    pred_blob = prediction.lower()
    pred_terms = token_set(prediction)
    hit = sum(
        1 for t in ref_terms
        if t.lower() in pred_blob or t in pred_terms or t.upper() in prediction
    )
    return hit / len(ref_terms)


def has_runaway_length(text: str, threshold: int = RUNAWAY_LEN) -> bool:
    return len(text) >= threshold


def has_loop_repetition(
    text: str,
    min_phrase: int = LOOP_MIN_PHRASE,
    min_count: int = LOOP_MIN_COUNT,
) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip())
    if len(normalized) < min_phrase * min_count:
        return False

    # Long repeated English / mixed phrase
    repeat_tail = "{" + f"{min_count - 1}," + "}"
    pattern = rf"(.{{{min_phrase},200}}?)\1{repeat_tail}"
    for m in re.finditer(pattern, normalized):
        if len(m.group(1).strip()) >= min_phrase:
            return True

    # Same sentence repeated (common in Chinese template + English loop)
    sentences = [s.strip() for s in re.split(r"[。\n；;]", normalized) if len(s.strip()) >= 12]
    if not sentences:
        return False
    from collections import Counter

    counts = Counter(sentences)
    if counts.most_common(1)[0][1] >= min_count:
        return True

    # Word-level loop for short repeated tokens
    words = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{4,}", normalized)
    if len(words) >= 12:
        for size in (3, 4, 5):
            for i in range(len(words) - size * min_count + 1):
                chunk = tuple(words[i : i + size])
                if words[i : i + size * min_count] == list(chunk) * min_count:
                    return True
    return False


def mentions_alarm_code(prediction: str, alarm_code: str | None) -> bool | None:
    if not alarm_code:
        return None
    code = alarm_code.strip()
    variants = {code, code.upper(), code.lower()}
    if re.match(r"PS\d+", code, re.I):
        num = re.search(r"\d+", code)
        if num:
            variants.add(num.group(0))
            variants.add(f"CNC{num.group(0)}")
    blob = prediction.upper()
    return any(v.upper() in blob for v in variants)


def is_usable(text: str, alarm_code: str | None) -> bool:
    """Composite: format ok, no loop, no runaway, alarm mentioned if applicable."""
    if not format_pass(text):
        return False
    if has_loop_repetition(text) or has_runaway_length(text):
        return False
    hit = mentions_alarm_code(text, alarm_code)
    if hit is False:
        return False
    return True


def extract_alarm_code(text: str) -> str | None:
    m = re.search(r"报警代码[：:]\s*(\S+)", text)
    return m.group(1) if m else None


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


CATEGORY_LABELS = {
    "fanuc": "FANUC 报警",
    "siemens": "Siemens 840D",
    "maintenance": "维护手册",
    "sensor": "传感器/产线",
    "other": "通用故障",
}


def build_user_query(sample: dict) -> str:
    instruction = sample.get("instruction", "").strip()
    user_input = sample.get("input", "").strip()
    if user_input:
        return f"{instruction}\n\n{user_input}"
    return instruction


def sample_records(records: list[dict], n: int, seed: int) -> list[tuple[int, dict]]:
    rng = random.Random(seed)
    buckets: dict[str, list[tuple[int, dict]]] = {}
    for i, rec in enumerate(records):
        buckets.setdefault(classify_sample(rec), []).append((i, rec))

    # Stratified: at least 1 per bucket, then round-robin fill
    picked: list[tuple[int, dict]] = []
    bucket_keys = sorted(buckets.keys())
    per_bucket = max(1, n // max(len(buckets), 1))

    for key in bucket_keys:
        items = buckets[key][:]
        rng.shuffle(items)
        picked.extend(items[:per_bucket])

    if len(picked) < n:
        seen = {x for x, _ in picked}
        rest = [(i, r) for i, r in enumerate(records) if i not in seen]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])

    rng.shuffle(picked)
    return picked[:n]


@dataclass
class CompareResult:
    index: int
    category: str
    alarm_code: str | None
    reference: str
    lora_only: str
    rag_lora: str
    lora_format: float = 0.0
    rag_format: float = 0.0
    lora_jaccard: float = 0.0
    rag_jaccard: float = 0.0
    lora_keyword_recall: float = 0.0
    rag_keyword_recall: float = 0.0
    lora_alarm_hit: bool | None = None
    rag_alarm_hit: bool | None = None
    lora_loop: bool = False
    rag_loop: bool = False
    lora_runaway: bool = False
    rag_runaway: bool = False
    lora_usable: bool = False
    rag_usable: bool = False
    lora_len: int = 0
    rag_len: int = 0

    @classmethod
    def from_predictions(
        cls,
        index: int,
        category: str,
        alarm_code: str | None,
        reference: str,
        lora_only: str,
        rag_lora: str,
    ) -> CompareResult:
        return cls(
            index=index,
            category=category,
            alarm_code=alarm_code,
            reference=reference,
            lora_only=lora_only,
            rag_lora=rag_lora,
            lora_format=format_score(lora_only),
            rag_format=format_score(rag_lora),
            lora_jaccard=jaccard(reference, lora_only),
            rag_jaccard=jaccard(reference, rag_lora),
            lora_keyword_recall=keyword_recall(reference, lora_only),
            rag_keyword_recall=keyword_recall(reference, rag_lora),
            lora_alarm_hit=mentions_alarm_code(lora_only, alarm_code),
            rag_alarm_hit=mentions_alarm_code(rag_lora, alarm_code),
            lora_loop=has_loop_repetition(lora_only),
            rag_loop=has_loop_repetition(rag_lora),
            lora_runaway=has_runaway_length(lora_only),
            rag_runaway=has_runaway_length(rag_lora),
            lora_usable=is_usable(lora_only, alarm_code),
            rag_usable=is_usable(rag_lora, alarm_code),
            lora_len=len(lora_only),
            rag_len=len(rag_lora),
        )


def _rate(items: list[CompareResult], field: str) -> float:
    vals = [getattr(x, field) for x in items]
    if not vals:
        return 0.0
    return sum(1 for v in vals if v) / len(vals)


def _alarm_hit_rate(items: list[CompareResult], prefix: str) -> float | None:
    hits = [getattr(x, f"{prefix}_alarm_hit") for x in items if getattr(x, f"{prefix}_alarm_hit") is not None]
    if not hits:
        return None
    return sum(1 for h in hits if h) / len(hits)


def aggregate_mode(items: list[CompareResult], prefix: str) -> dict:
    n = len(items)
    alarm_rate = _alarm_hit_rate(items, prefix)
    return {
        "count": n,
        "avg_format": sum(getattr(x, f"{prefix}_format") for x in items) / n,
        "format_pass_rate": sum(
            1 for x in items if getattr(x, f"{prefix}_format") == 1.0
        ) / n,
        "avg_jaccard": sum(getattr(x, f"{prefix}_jaccard") for x in items) / n,
        "avg_keyword_recall": sum(getattr(x, f"{prefix}_keyword_recall") for x in items) / n,
        "alarm_hit_rate": alarm_rate,
        "loop_rate": _rate(items, f"{prefix}_loop"),
        "runaway_rate": _rate(items, f"{prefix}_runaway"),
        "usable_rate": _rate(items, f"{prefix}_usable"),
        "avg_len": sum(getattr(x, f"{prefix}_len") for x in items) / n,
    }


def aggregate_by_category(items: list[CompareResult]) -> dict[str, dict]:
    buckets: dict[str, list[CompareResult]] = {}
    for item in items:
        buckets.setdefault(item.category, []).append(item)

    out: dict[str, dict] = {}
    for cat, cat_items in sorted(buckets.items()):
        lora = aggregate_mode(cat_items, "lora")
        rag = aggregate_mode(cat_items, "rag")
        out[cat] = {
            "label": CATEGORY_LABELS.get(cat, cat),
            "count": len(cat_items),
            "lora_usable_rate": lora["usable_rate"],
            "rag_usable_rate": rag["usable_rate"],
            "lora_format_pass_rate": lora["format_pass_rate"],
            "rag_format_pass_rate": rag["format_pass_rate"],
            "lora_keyword_recall": lora["avg_keyword_recall"],
            "rag_keyword_recall": rag["avg_keyword_recall"],
            "lora_jaccard": lora["avg_jaccard"],
            "rag_jaccard": rag["avg_jaccard"],
        }
    return out
