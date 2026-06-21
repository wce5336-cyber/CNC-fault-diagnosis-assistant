#!/usr/bin/env python3
"""下载并解析公开 CNC 报警手册，生成结构化知识库素材。"""

from __future__ import annotations

import csv
import re
import urllib.request
from pathlib import Path

from .paths import EXTERNAL_KNOWLEDGE

KNOWLEDGE_DIR = EXTERNAL_KNOWLEDGE
RAW_DIR = KNOWLEDGE_DIR / "raw"
PARSED_DIR = KNOWLEDGE_DIR / "parsed"
MANUAL_DIR = KNOWLEDGE_DIR / "manuals"

FANUC_PDF_URL = "https://cncfixtech.com/wp-content/uploads/2025/02/Fanuc-alarm-list.pdf"
SIEMENS_PDF_URL = "https://support.industry.siemens.com/cs/attachments/109248112/DAsl_0108_en.pdf"

# 优先保留的 FANUC 报警类别（工业现场高频）
FANUC_PRIORITY_PREFIXES = ("SV", "SP", "PS", "OT", "OH", "SR", "DS", "IE", "BG")


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  已存在，跳过下载: {dest.name}")
        return
    print(f"  下载: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())
    print(f"  保存至: {dest}")


def pdf_to_text(pdf_path: Path) -> str:
    try:
        import pypdf
    except ImportError:
        print("  [WARN] 未安装 pypdf，尝试读取同名 .txt")
        txt_path = pdf_path.with_suffix(".txt")
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8", errors="ignore")
        return ""

    reader = pypdf.PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def parse_fanuc_alarms(text: str) -> list[dict[str, str]]:
    """从 FANUC 报警手册文本提取结构化记录。"""
    text = re.sub(r"-\s*\d+\s*-\s*", "\n", text)
    text = re.sub(r"Number Message Description", "\n", text)

    # 4位报警号 + 英文消息 + 描述
    pattern = re.compile(
        r"(?<!\d)(\d{4})\s+([A-Z][A-Z0-9 /\-]{2,60}?)\s+(.+?)(?=\d{4}\s+[A-Z]|\Z)",
        re.DOTALL,
    )
    records: list[dict[str, str]] = []
    seen: set[str] = set()

    for match in pattern.finditer(text):
        code, message, description = match.groups()
        message = re.sub(r"\s+", " ", message).strip()
        description = re.sub(r"\s+", " ", description).strip()
        if len(message) < 3 or len(description) < 10:
            continue
        if code in seen:
            continue
        seen.add(code)

        category = "CNC"
        for prefix in FANUC_PRIORITY_PREFIXES:
            if prefix in message or message.startswith(prefix):
                category = prefix
                break

        records.append(
            {
                "system": "FANUC",
                "code": code,
                "alarm_id": f"PS{code}",
                "message": message,
                "description": description[:800],
                "category": category,
            }
        )

    return records


def parse_siemens_alarms(text: str, max_alarms: int = 400) -> list[dict[str, str]]:
    """从 Siemens 840D 诊断手册提取 NCK 报警。"""
    blocks = re.split(r"(?=\n\d{4,5}\s+[A-Z])", text)
    records: list[dict[str, str]] = []
    seen: set[str] = set()

    for block in blocks:
        header = re.match(r"^\s*(\d{4,5})\s+(.+?)(?:\n|$)", block)
        if not header:
            continue
        code, title = header.groups()
        if code in seen:
            continue
        if not (1000 <= int(code) <= 29999):
            continue

        definition = ""
        remedy = ""
        reaction = ""

        def_match = re.search(r"Definitions:\s*(.+?)(?:Reaction:|Remedy:|Program)", block, re.DOTALL)
        if def_match:
            definition = re.sub(r"\s+", " ", def_match.group(1)).strip()[:500]

        remedy_match = re.search(r"Remedy:\s*(.+?)(?:Program|Overview|\n\d{4})", block, re.DOTALL)
        if remedy_match:
            remedy = re.sub(r"\s+", " ", remedy_match.group(1)).strip()[:500]

        reaction_match = re.search(r"Reaction:\s*(.+?)(?:Remedy:|Program)", block, re.DOTALL)
        if reaction_match:
            reaction = re.sub(r"\s+", " ", reaction_match.group(1)).strip()[:300]

        if not definition and not remedy:
            continue

        seen.add(code)
        records.append(
            {
                "system": "Siemens 840D",
                "code": code,
                "alarm_id": code,
                "message": re.sub(r"\s+", " ", title).strip()[:200],
                "description": definition or title,
                "remedy": remedy,
                "reaction": reaction,
                "category": "NCK",
            }
        )
        if len(records) >= max_alarms:
            break

    return records


def save_csv(records: list[dict[str, str]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in records for k in r})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"  写入 {len(records)} 条 -> {path}")


def write_sources_manifest() -> None:
    manifest = KNOWLEDGE_DIR / "sources.json"
    import json

    data = {
        "fanuc_alarm_list": {
            "url": FANUC_PDF_URL,
            "description": "FANUC CNC Alarm List (Appendix A)",
            "license_note": "公开技术文档，仅供学习研究",
        },
        "siemens_dasl": {
            "url": SIEMENS_PDF_URL,
            "description": "SINUMERIK 840D sl Diagnostics Manual",
            "license_note": "Siemens 官方诊断手册，仅供学习研究",
        },
    }
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 获取外部知识源 ===")

    fanuc_pdf = RAW_DIR / "fanuc_alarm_list.pdf"
    siemens_pdf = RAW_DIR / "siemens_840d_diagnostics.pdf"

    try:
        download_file(FANUC_PDF_URL, fanuc_pdf)
    except Exception as exc:
        print(f"  [WARN] FANUC 下载失败: {exc}")

    try:
        download_file(SIEMENS_PDF_URL, siemens_pdf)
    except Exception as exc:
        print(f"  [WARN] Siemens 下载失败: {exc}")

    # 解析 FANUC
    fanuc_text = ""
    if fanuc_pdf.exists():
        fanuc_text = pdf_to_text(fanuc_pdf)
        if fanuc_text:
            fanuc_raw = RAW_DIR / "fanuc_alarm_list.txt"
            fanuc_raw.write_text(fanuc_text, encoding="utf-8")
            fanuc_records = parse_fanuc_alarms(fanuc_text)
            # 优先保留伺服/主轴/程序类报警
            priority = [r for r in fanuc_records if r["category"] in FANUC_PRIORITY_PREFIXES]
            others = [r for r in fanuc_records if r["category"] == "CNC"]
            fanuc_final = priority + others[: max(0, 350 - len(priority))]
            save_csv(fanuc_final, PARSED_DIR / "fanuc_alarms.csv")

    # 解析 Siemens
    siemens_text = ""
    if siemens_pdf.exists():
        siemens_text = pdf_to_text(siemens_pdf)
        if siemens_text:
            siemens_raw = RAW_DIR / "siemens_840d_diagnostics.txt"
            siemens_raw.write_text(siemens_text, encoding="utf-8")
            siemens_records = parse_siemens_alarms(siemens_text)
            save_csv(siemens_records, PARSED_DIR / "siemens_alarms.csv")

    write_sources_manifest()
    print("=== 外部知识源获取完成 ===")


if __name__ == "__main__":
    main()
