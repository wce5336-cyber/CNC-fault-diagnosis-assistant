#!/usr/bin/env python3
"""
CNC 故障诊断数据流水线：清洗、增强、生成微调数据集 + RAG 知识库。

架构：微调（SFT 标准诊断 QA）+ RAG（报警手册/维修文档分块检索）

用法:
    python process_cnc_datasets.py --fetch          # 先获取外部报警手册
    python process_cnc_datasets.py                  # 构建数据集
    python process_cnc_datasets.py --setup-env      # 创建 conda 环境 qx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
from .paths import DATA, EXTERNAL_KNOWLEDGE, PROCESSED, ROOT

KNOWLEDGE_DIR = EXTERNAL_KNOWLEDGE
PARSED_DIR = KNOWLEDGE_DIR / "parsed"
MANUAL_DIR = KNOWLEDGE_DIR / "manuals"
OUTPUT_DIR = PROCESSED
SFT_JSON = OUTPUT_DIR / "cnc_diagnosis_sft.json"
RAG_DIR = OUTPUT_DIR / "rag_knowledge"
RAG_CHUNKS = RAG_DIR / "chunks.jsonl"
RAG_MANIFEST = RAG_DIR / "manifest.json"
STATS_JSON = OUTPUT_DIR / "processing_stats.json"
DATA_ROOT = DATA

SKIP_DIRS = {".git", "__pycache__", "cankao", "node_modules", ".venv", "external_knowledge"}
SKIP_EXTENSIONS = {".zip", ".h5", ".mat", ".mpf", ".mov", ".ipynb", ".sql"}

INSTRUCTION_VARIANTS = [
    "请根据以下机床故障信息进行诊断分析，给出可能原因、排查步骤和处理建议。",
    "以下为CNC机床报警/故障信息，请分析故障原因并提供详细的排查与处理方案。",
    "作为CNC故障诊断专家，请针对下列故障现象给出专业诊断意见。",
]

# ai4i2020：去重为 5 类标准故障，附带典型传感器特征
AI4I_CANONICAL = {
    "TWF": {
        "name": "刀具磨损故障",
        "symptoms": "刀具磨损时间累计偏高，加工表面粗糙度恶化，切削力波动增大",
        "sensor_hint": "Tool wear [min] 持续上升，扭矩 [Nm] 波动加大",
        "causes": [
            "刀具达到寿命极限未及时更换",
            "切削参数与材料不匹配加速磨损",
            "冷却不足导致刀具过热磨损",
        ],
        "steps": [
            "检查刀具磨损计数与当前刀刃状态",
            "对比正常工况下的扭矩与振动趋势",
            "审查切削速度与每齿进给是否过大",
            "确认冷却液喷射是否对准刀尖",
        ],
        "solutions": [
            "更换磨损刀具并重新对刀",
            "优化切削参数降低刀具负载",
            "恢复冷却液流量与浓度",
            "设置刀具寿命预警阈值",
        ],
    },
    "HDF": {
        "name": "散热故障",
        "symptoms": "工艺温度与空气温度温差异常，机床温升过快",
        "sensor_hint": "Process temperature [K] 偏高，Air/Process 温差缩小",
        "causes": [
            "冷却系统效率下降或堵塞",
            "环境通风不良",
            "长时间连续高负荷运行",
        ],
        "steps": [
            "检查冷却泵、风扇与换热器运行",
            "清理冷却液管路和过滤网",
            "监测主轴与进给轴温升曲线",
            "检查车间环境温度与通风",
        ],
        "solutions": [
            "修复冷却系统并补充冷却液",
            "清理散热通道",
            "合理安排间歇加工降低热负荷",
            "必要时检修换热器",
        ],
    },
    "PWF": {
        "name": "动力故障",
        "symptoms": "主轴转速波动，扭矩输出不稳定，可能伴随驱动器报警",
        "sensor_hint": "Rotational speed [rpm] 波动，Torque [Nm] 异常",
        "causes": [
            "电源电压不稳或缺相",
            "主轴/伺服驱动器故障",
            "电机绕组或接线异常",
        ],
        "steps": [
            "测量进线三相电压与平衡度",
            "检查驱动器报警代码",
            "检查电机接线与接地",
            "查看近期是否有过载工况",
        ],
        "solutions": [
            "排除电网问题后重启驱动器",
            "修复或更换故障驱动模块",
            "检查并紧固电气连接",
            "复核过载保护参数设置",
        ],
    },
    "OSF": {
        "name": "过载故障",
        "symptoms": "切削扭矩超限，机床振动加剧，可能触发负载监控报警",
        "sensor_hint": "Torque [Nm] 超限，Rotational speed [rpm] 下降",
        "causes": [
            "切深或进给过大",
            "刀具钝化导致切削力增大",
            "工件或刀具装夹刚性不足",
        ],
        "steps": [
            "检查当前切削参数",
            "评估刀具磨损与装夹状态",
            "对比历史正常加工电流/扭矩",
            "检查是否存在机械干涉",
        ],
        "solutions": [
            "降低切深与进给速度",
            "更换锋利刀具",
            "改善装夹方案提高刚性",
            "优化刀路减少瞬时负载",
        ],
    },
    "RNF": {
        "name": "随机故障",
        "symptoms": "无明显单一传感器趋势，间歇性异常停机",
        "sensor_hint": "各传感器读数无明显规律，Machine failure 偶发",
        "causes": [
            "电气接触不良或信号干扰",
            "传感器偶发失效",
            "软件/PLC 逻辑边界条件触发",
        ],
        "steps": [
            "调取故障前后报警与操作日志",
            "检查电缆屏蔽与接地",
            "对各传感器做交叉验证",
            "排查近期程序或参数变更",
        ],
        "solutions": [
            "紧固接头并改善屏蔽",
            "更换可疑传感器",
            "回滚异常参数变更",
            "建立故障复现记录便于根因分析",
        ],
    },
}

# 现场故障日志增强模板（英文 -> 结构化中文诊断）
FAILURE_LOG_ENHANCEMENTS = {
    "Motor Failure": {
        "symptoms": "电机无法启动或运行中突然停转，可能伴有异响、过热或电流过载",
        "causes": [
            "电机绕组烧毁或绝缘老化",
            "驱动器功率模块损坏",
            "机械负载过大导致堵转",
            "编码器反馈异常引起保护停机",
        ],
        "steps": [
            "记录驱动器与 NC 报警代码",
            "测量电机三相电阻与绝缘",
            "检查机械传动是否卡滞",
            "查看电机电流与温度历史趋势",
        ],
        "solutions": [
            "更换或维修故障电机",
            "修复驱动器或更换功率单元",
            "排除机械卡滞后重新校准",
            "恢复运行前做空载与负载测试",
        ],
    },
    "Overheating": {
        "symptoms": "机床温升过快，冷却风扇高速运转，可能触发温度保护停机",
        "causes": [
            "冷却液不足或循环泵故障",
            "主轴轴承润滑不良",
            "环境温度过高或通风不良",
            "长时间超负荷切削",
        ],
        "steps": [
            "检查冷却液液位与泵运行",
            "监测主轴与电机温度曲线",
            "检查润滑系统工作状态",
            "回顾近期切削参数与连续运行时间",
        ],
        "solutions": [
            "补充冷却液并修复泵",
            "降低切削参数间歇加工",
            "补充润滑或更换轴承",
            "改善车间通风条件",
        ],
    },
    "Vibration Issue": {
        "symptoms": "加工过程振动明显增大，工件表面出现振纹，可能伴随异常噪声",
        "causes": [
            "刀具磨损或悬伸过长",
            "工件装夹刚性不足",
            "轴承磨损或螺栓松动",
            "切削参数引发共振",
        ],
        "steps": [
            "检查刀具状态与悬伸量",
            "评估工装夹具压紧力",
            "检查主轴与进给传动间隙",
            "试切对比不同转速下的振动",
        ],
        "solutions": [
            "更换刀具并优化悬伸",
            "改善装夹增加支撑",
            "紧固轴承座与导轨螺栓",
            "调整转速避开共振区",
        ],
    },
    "Electrical Failure": {
        "symptoms": "电气柜报警，部分轴或辅助功能失效，可能跳闸或急停",
        "causes": [
            "接线松动或端子氧化",
            "短路或过载保护动作",
            "继电器/接触器故障",
            "外部电网波动",
        ],
        "steps": [
            "检查断路器与熔断器状态",
            "测量三相电压与接地",
            "排查急停与安全回路",
            "分段断电定位短路点",
        ],
        "solutions": [
            "紧固接线并更换损坏端子",
            "排除短路后复位保护",
            "更换故障继电器或接触器",
            "必要时加装稳压设备",
        ],
    },
    "Spindle Breakdown": {
        "symptoms": "主轴无法启动或转速异常，加工精度下降，可能伴随主轴驱动报警",
        "causes": [
            "主轴轴承损坏",
            "主轴电机或驱动器故障",
            "松刀机构异常导致负载",
            "编码器信号丢失",
        ],
        "steps": [
            "记录主轴驱动报警子代码",
            "检查主轴运转声音与温升",
            "测试松刀/夹刀动作",
            "检查编码器接线与信号",
        ],
        "solutions": [
            "更换主轴轴承或送修主轴单元",
            "修复主轴驱动器",
            "调整松刀压力与机构",
            "更换编码器或修复接线",
        ],
    },
}

MAINTENANCE_TYPE_ENHANCEMENTS = {
    "Routine Inspection": "执行例行巡检：导轨润滑、丝杠清洁、冷却液检查、各轴运行声音与精度抽检",
    "Vibration Control": "进行振动治理：检查轴承间隙、联轴器对中、刀具悬伸与切削参数优化",
    "Temperature Check": "开展温升检查：验证冷却系统、主轴润滑、电机散热与车间通风",
    "Electrical System": "电气系统点检：紧固接线端子、检查接地、测试急停与安全回路",
    "Spindle Inspection": "主轴专项检查：测试运转振动、温升、松刀机构与编码器反馈",
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def format_diagnostic_output(
    causes: list[str],
    steps: list[str],
    solutions: list[str],
) -> str:
    parts = ["【可能原因】"]
    parts.extend(f"{i}. {c}" for i, c in enumerate(causes, 1))
    parts.append("\n【排查步骤】")
    parts.extend(f"{i}. {s}" for i, s in enumerate(steps, 1))
    parts.append("\n【处理建议】")
    parts.extend(f"{i}. {s}" for i, s in enumerate(solutions, 1))
    return "\n".join(parts)


def make_diagnostic_record(
    machine_system: str,
    symptoms: str,
    causes: list[str],
    steps: list[str],
    solutions: list[str],
    alarm_code: str = "",
    extra_input: str = "",
    instruction: str | None = None,
    source: str = "",
) -> dict[str, Any]:
    input_lines = [f"机床系统：{machine_system}"]
    if alarm_code:
        input_lines.append(f"报警代码：{alarm_code}")
    input_lines.append(f"故障现象：{symptoms}")
    if extra_input:
        input_lines.append(extra_input)

    record: dict[str, Any] = {
        "instruction": instruction or INSTRUCTION_VARIANTS[0],
        "input": "\n".join(input_lines),
        "output": format_diagnostic_output(causes, steps, solutions),
    }
    if source:
        record["source"] = source
    return record


def is_valid_record(record: dict[str, Any], min_out: int = 40) -> bool:
    return len(record.get("output", "")) >= min_out and len(record.get("input", "")) >= 8


def normalize_col(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    norm_map = {normalize_col(c): c for c in columns}
    for cand in candidates:
        key = normalize_col(cand)
        if key in norm_map:
            return norm_map[key]
    for col in columns:
        ncol = normalize_col(col)
        for cand in candidates:
            if normalize_col(cand) in ncol:
                return col
    return None


def dedupe_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        record["instruction"].lower(),
        record["input"].lower(),
        record["output"].lower(),
    )


def deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for rec in records:
        key = dedupe_key(rec)
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


def augment_instruction_variants(
    records: list[dict[str, Any]],
    variants: list[str] = INSTRUCTION_VARIANTS,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """为每条记录生成 instruction 变体，扩充微调数据。"""
    rng = random.Random(seed)
    augmented: list[dict[str, Any]] = []
    for rec in records:
        augmented.append(rec)
        alt = rng.choice([v for v in variants if v != rec["instruction"]])
        copy_rec = {**rec, "instruction": alt}
        augmented.append(copy_rec)
    return deduplicate_records(augmented)


# ---------------------------------------------------------------------------
# 外部报警代码表
# ---------------------------------------------------------------------------
def fanuc_alarm_to_records(row: dict[str, str]) -> list[dict[str, Any]]:
    code = row.get("code", "")
    message = row.get("message", "")
    desc = row.get("description", "")
    category = row.get("category", "CNC")

    causes = [f"报警含义：{message}"]
    if desc:
        causes.append(desc[:250])

    steps = [
        f"在 FANUC 诊断界面确认报警号 PS{code} 或 {category}{code}",
        "记录报警发生时的程序段与操作模式",
        "查阅 FANUC 维修手册对应章节",
        "检查与该报警相关的硬件与参数",
    ]
    solutions = [
        "按手册描述逐项排除故障原因",
        "修复后执行报警复位（RESET）",
        "空运转验证各轴与主轴动作正常",
        "恢复加工前做首件确认",
    ]

    base = make_diagnostic_record(
        machine_system="FANUC CNC",
        alarm_code=f"PS{code}",
        symptoms=message,
        causes=causes,
        steps=steps,
        solutions=solutions,
        source="fanuc_alarm_list",
    )
    return [base]


def siemens_alarm_to_records(row: dict[str, str]) -> list[dict[str, Any]]:
    code = row.get("code", "")
    message = row.get("message", "")
    desc = row.get("description", "")
    remedy = row.get("remedy", "")
    reaction = row.get("reaction", "")

    causes = [message]
    if desc:
        causes.append(desc[:250])
    if reaction:
        causes.append(f"系统反应：{reaction[:150]}")

    steps = [
        f"在 Siemens 操作面板查看 NCK 报警 {code} 详情",
        "记录报警参数与当前加工状态",
        "查阅 840D 诊断手册对应条目",
        "检查相关轴/驱动/PLC 信号",
    ]
    solutions = []
    if remedy:
        solutions.append(remedy[:300])
    solutions.extend([
        "按手册 Remedy 说明执行修复",
        "报警清除后重新回零验证",
        "确认无后续报警后恢复生产",
    ])

    return [make_diagnostic_record(
        machine_system="Siemens SINUMERIK 840D",
        alarm_code=code,
        symptoms=message,
        causes=causes,
        steps=steps,
        solutions=solutions,
        source="siemens_840d_diagnostics",
    )]


def load_alarm_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8-sig")
    records: list[dict[str, Any]] = []
    system = "fanuc" if "fanuc" in path.stem else "siemens"
    for row in df.to_dict(orient="records"):
        row = {k: clean_text(v) for k, v in row.items()}
        if system == "fanuc":
            records.extend(fanuc_alarm_to_records(row))
        else:
            records.extend(siemens_alarm_to_records(row))
    return records


# ---------------------------------------------------------------------------
# 维修手册解析
# ---------------------------------------------------------------------------
MANUAL_SECTION_RE = re.compile(
    r"##\s*故障现象[：:]\s*(?P<symptoms>.+?)\n\n"
    r"\*\*机床系统\*\*[：:]\s*(?P<system>.+?)\n"
    r"(?:.*?\n)*?"
    r"\*\*可能原因\*\*[：:]\s*\n(?P<causes>.+?)\n\n"
    r"\*\*排查步骤\*\*[：:]\s*\n(?P<steps>.+?)\n\n"
    r"\*\*处理建议\*\*[：:]\s*\n(?P<solutions>.+?)(?=\n---|\n##|\Z)",
    re.DOTALL,
)


def parse_list_block(text: str) -> list[str]:
    items = []
    for line in text.strip().splitlines():
        line = re.sub(r"^\d+\.\s*", "", line.strip())
        if line:
            items.append(line)
    return items


def parse_maintenance_manual(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    for match in MANUAL_SECTION_RE.finditer(content):
        rec = make_diagnostic_record(
            machine_system=match.group("system").strip(),
            symptoms=match.group("symptoms").strip(),
            causes=parse_list_block(match.group("causes")),
            steps=parse_list_block(match.group("steps")),
            solutions=parse_list_block(match.group("solutions")),
            source=f"manual:{path.name}",
        )
        if is_valid_record(rec):
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# ai4i2020 去重 + 传感器特征输入
# ---------------------------------------------------------------------------
def build_ai4i2020_canonical() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for code, info in AI4I_CANONICAL.items():
        sensor_input = f"传感器特征：{info['sensor_hint']}"
        rec = make_diagnostic_record(
            machine_system="预测性维护平台（ai4i2020）",
            symptoms=info["symptoms"],
            causes=info["causes"],
            steps=info["steps"],
            solutions=info["solutions"],
            extra_input=sensor_input,
            source=f"ai4i2020:{code}",
        )
        records.append(rec)
        # 仅传感器数据的变体（模拟纯数据驱动诊断）
        data_only = make_diagnostic_record(
            machine_system="预测性维护平台（ai4i2020）",
            symptoms="根据传感器数据判定异常类型",
            causes=info["causes"],
            steps=info["steps"],
            solutions=info["solutions"],
            extra_input=sensor_input,
            instruction="根据以下传感器监测数据，判断可能的故障类型并给出诊断建议。",
            source=f"ai4i2020:{code}:sensor",
        )
        records.append(data_only)
    return records


# ---------------------------------------------------------------------------
# 故障日志增强
# ---------------------------------------------------------------------------
def detect_csv_header_row(path: Path, max_scan: int = 10) -> int:
    preview = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:max_scan]
    header_keys = (
        "machine_id", "failure_type", "maintenance_type",
        "comments", "maintenance_comments",
    )
    best_row, best_score = 0, -1
    for i, line in enumerate(preview):
        score = sum(1 for k in header_keys if k in line.lower())
        if score > best_score:
            best_score, best_row = score, i
    return best_row


def enhance_failure_log_row(row: dict[str, Any], path_name: str) -> dict[str, Any] | None:
    failure = clean_text(row.get("failure_type") or row.get("Failure_Type", ""))
    machine = clean_text(row.get("machine_id") or row.get("Machine_ID", ""))
    comments = clean_text(row.get("comments") or row.get("Comments", ""))
    downtime = clean_text(row.get("downtime_hours") or row.get("Downtime_Hours", ""))
    repair = clean_text(row.get("repair_time (hours)") or row.get("Repair_Time_hours", ""))

    is_maintenance = "maintenance" in path_name.lower()
    maint_type = clean_text(row.get("maintenance_type") or row.get("Maintenance_Type", ""))
    maint_comments = clean_text(row.get("maintenance_comments") or row.get("Maintenance_Comments", ""))

    if is_maintenance and maint_type:
        template = MAINTENANCE_TYPE_ENHANCEMENTS.get(maint_type, f"执行{maint_type}维护作业")
        return make_diagnostic_record(
            machine_system="汽车产线 CNC（Tata Motors 案例）",
            symptoms=f"机床 {machine} 到达计划维护节点，需执行 {maint_type}",
            causes=[f"按计划维护类型：{maint_type}", "预防性维护以降低突发故障风险"],
            steps=["查阅该机型维护检查表", "确认停机窗口与备件", "执行维护项目并记录", "维护后做空运转测试"],
            solutions=[template, maint_comments or "完成维护项目并更新维护台账"],
            extra_input=f"设备编号：{machine}",
            instruction="请为以下机床维护场景制定维护要点和注意事项。",
            source="maintenance_log",
        )

    if not failure or failure.lower() in {"true", "false", "nan"}:
        return None

    enh = FAILURE_LOG_ENHANCEMENTS.get(failure)
    if enh:
        causes, steps, solutions = enh["causes"], enh["steps"], enh["solutions"]
        symptoms = enh["symptoms"]
    else:
        symptoms = f"机床报告 {failure} 异常"
        causes = [f"现场记录故障类型为 {failure}"]
        steps = ["查阅该机型维修手册", "检查相关子系统", "记录故障现象与报警"]
        solutions = [comments or f"针对 {failure} 进行排查维修"]

    if comments and comments not in str(solutions):
        solutions = [f"现场处理记录：{comments}"] + solutions
    if repair:
        steps.append(f"历史维修耗时约 {repair} 小时，可参考排产")
    if downtime:
        extra = f"设备编号：{machine}；停机时长：{downtime} 小时"
    else:
        extra = f"设备编号：{machine}" if machine else ""

    return make_diagnostic_record(
        machine_system="汽车产线 CNC（Tata Motors 案例）",
        symptoms=symptoms,
        causes=causes,
        steps=steps,
        solutions=solutions,
        extra_input=extra,
        source="failure_log",
    )


def parse_failure_csv(path: Path) -> list[dict[str, Any]]:
    header_row = detect_csv_header_row(path)
    try:
        df = pd.read_csv(path, header=header_row, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, header=header_row, encoding="gbk", on_bad_lines="skip")

    df.columns = [str(c).strip() for c in df.columns]
    col_map = {normalize_col(c): c for c in df.columns}
    records: list[dict[str, Any]] = []

    for row in df.to_dict(orient="records"):
        normalized = {normalize_col(k): v for k, v in row.items()}
        merged = {**row, **{k: v for k, v in normalized.items()}}
        rec = enhance_failure_log_row(merged, path.name)
        if rec and is_valid_record(rec):
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Excel 报警表（用户自行添加）
# ---------------------------------------------------------------------------
INSTRUCTION_COLS = ["故障现象", "故障描述", "报警内容", "现象", "fault", "symptom", "alarm", "description"]
OUTPUT_COLS = ["解决方案", "处理方法", "solution", "remedy", "action", "fix"]
CODE_COLS = ["报警代码", "故障代码", "alarm code", "error code", "code"]


def parse_excel_alarm_table(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
    except Exception as exc:
        print(f"  [WARN] Excel 读取失败 {path}: {exc}")
        return records

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
        if df.empty:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        inst_col = find_column(list(df.columns), INSTRUCTION_COLS)
        out_col = find_column(list(df.columns), OUTPUT_COLS)
        code_col = find_column(list(df.columns), CODE_COLS)
        alarm_hint = any(k in path.stem.lower() + sheet.lower() for k in ("alarm", "报警", "fault", "故障"))
        if not alarm_hint or not ((inst_col and out_col) or (code_col and inst_col)):
            continue

        for _, row in df.iterrows():
            symptom = clean_text(row.get(inst_col, "")) if inst_col else ""
            solution_text = clean_text(row.get(out_col, "")) if out_col else ""
            code = clean_text(row.get(code_col, "")) if code_col else ""
            if not symptom:
                continue
            rec = make_diagnostic_record(
                machine_system="用户报警代码表",
                alarm_code=code,
                symptoms=symptom,
                causes=[f"报警代码 {code} 触发" if code else "报警条件满足"],
                steps=["确认报警代码与当前加工状态", "查阅对应系统手册", "逐项排查可能原因"],
                solutions=[solution_text] if solution_text else ["按手册排除故障后复位"],
                source=f"excel:{path.name}",
            )
            if is_valid_record(rec):
                records.append(rec)
    return records


# ---------------------------------------------------------------------------
# RAG 知识库构建
# ---------------------------------------------------------------------------
def chunk_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def record_to_rag_chunk(record: dict[str, Any]) -> dict[str, Any]:
    text = (
        f"来源：{record.get('source', 'unknown')}\n"
        f"{record['input']}\n\n"
        f"{record['output']}"
    )
    return {
        "id": chunk_id(text),
        "text": text,
        "metadata": {
            "source": record.get("source", ""),
            "type": "diagnostic_qa",
        },
    }


def alarm_row_to_rag_chunk(row: dict[str, str], system: str) -> dict[str, Any]:
    if system == "fanuc":
        text = (
            f"系统：FANUC\n报警号：PS{row.get('code', '')}\n"
            f"消息：{row.get('message', '')}\n"
            f"说明：{row.get('description', '')}"
        )
    else:
        text = (
            f"系统：Siemens 840D\n报警号：{row.get('code', '')}\n"
            f"消息：{row.get('message', '')}\n"
            f"定义：{row.get('description', '')}\n"
            f"处理：{row.get('remedy', '')}"
        )
    return {
        "id": chunk_id(text),
        "text": text,
        "metadata": {"source": system, "type": "alarm_reference", "code": row.get("code", "")},
    }


def build_rag_knowledge_base(
    sft_records: list[dict[str, Any]],
    alarm_csvs: list[Path],
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen: set[str] = set()

    for rec in sft_records:
        chunk = record_to_rag_chunk(rec)
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            chunks.append(chunk)

    for csv_path in alarm_csvs:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        system = "fanuc" if "fanuc" in csv_path.stem else "siemens"
        for row in df.to_dict(orient="records"):
            row = {k: clean_text(v) for k, v in row.items()}
            chunk = alarm_row_to_rag_chunk(row, system)
            if chunk["id"] not in seen:
                seen.add(chunk["id"])
                chunks.append(chunk)

    manual_path = MANUAL_DIR / "cnc_common_faults.md"
    if manual_path.exists():
        content = manual_path.read_text(encoding="utf-8")
        for i, section in enumerate(re.split(r"\n---\n", content)):
            section = section.strip()
            if len(section) < 80:
                continue
            chunk = {
                "id": chunk_id(section),
                "text": section,
                "metadata": {"source": "cnc_common_faults.md", "type": "maintenance_manual", "section": i},
            }
            if chunk["id"] not in seen:
                seen.add(chunk["id"])
                chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------------------
def should_skip_path(path: Path) -> bool:
    if set(path.parts) & SKIP_DIRS:
        return True
    return path.suffix.lower() in SKIP_EXTENSIONS


def collect_dataset_folders(root: Path) -> list[Path]:
    return [
        p for p in sorted(root.iterdir())
        if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith(".")
    ]


def process_all(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "fanuc_alarms": 0,
        "siemens_alarms": 0,
        "manual_sections": 0,
        "ai4i2020_canonical": 0,
        "failure_logs": 0,
        "excel_alarms": 0,
        "dataset_csv": 0,
    }

    # 1. 外部报警代码
    fanuc_path = PARSED_DIR / "fanuc_alarms.csv"
    siemens_path = PARSED_DIR / "siemens_alarms.csv"
    fanuc_recs = load_alarm_csv(fanuc_path)
    siemens_recs = load_alarm_csv(siemens_path)
    stats["fanuc_alarms"] = len(fanuc_recs)
    stats["siemens_alarms"] = len(siemens_recs)
    all_records.extend(fanuc_recs)
    all_records.extend(siemens_recs)
    print(f"FANUC 报警: {len(fanuc_recs)} | Siemens 报警: {len(siemens_recs)}")

    # 2. 维修手册
    for md in MANUAL_DIR.glob("*.md"):
        manual_recs = parse_maintenance_manual(md)
        stats["manual_sections"] += len(manual_recs)
        all_records.extend(manual_recs)
        print(f"维修手册 {md.name}: {len(manual_recs)} 条")

    # 3. ai4i2020 去重
    ai4i_recs = build_ai4i2020_canonical()
    stats["ai4i2020_canonical"] = len(ai4i_recs)
    all_records.extend(ai4i_recs)
    print(f"ai4i2020 标准样本: {len(ai4i_recs)} 条")

    # 4. 遍历本地数据集
    for folder in collect_dataset_folders(root):
        if folder.name == "external_knowledge":
            continue
        print(f"\n处理: {folder.name}")
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or should_skip_path(path):
                continue
            suffix = path.suffix.lower()
            batch: list[dict[str, Any]] = []

            if suffix in {".xlsx", ".xls"}:
                batch = parse_excel_alarm_table(path)
                if batch:
                    stats["excel_alarms"] += len(batch)
                    print(f"  Excel: {path.name} -> {len(batch)}")

            elif suffix == ".csv":
                lower = path.name.lower()
                if "ai4i2020" in lower:
                    continue  # 已用 canonical 替代
                if any(k in lower for k in ("failure", "maintenance")):
                    if any(k in lower for k in ("merged", "xxxxx", "dashboard")):
                        continue
                    batch = parse_failure_csv(path)
                    if batch:
                        stats["failure_logs"] += len(batch)
                        print(f"  CSV: {path.name} -> {len(batch)}")

            all_records.extend(batch)

    stats["records_before_dedup"] = len(all_records)
    return all_records, stats


def strip_for_sft(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """LLaMA-Factory 仅需 instruction/input/output。"""
    return [
        {"instruction": r["instruction"], "input": r["input"], "output": r["output"]}
        for r in records
    ]


def setup_conda_env() -> None:
    env_file = ROOT / "environment.yml"
    subprocess.run(["conda", "env", "create", "-f", str(env_file)], check=False)
    subprocess.run(["conda", "env", "update", "-f", str(env_file), "--prune"], check=False)
    print("完成。执行: conda activate qx")


def run_fetch_sources() -> None:
    subprocess.run([sys.executable, "-m", "src.fetch_sources"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CNC 故障诊断数据流水线（微调+RAG）")
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT, dest="datasets_root")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fetch", action="store_true", help="下载并解析外部报警手册")
    parser.add_argument("--setup-env", action="store_true")
    parser.add_argument("--no-augment", action="store_true", help="不生成 instruction 变体")
    args = parser.parse_args()

    if args.setup_env:
        setup_conda_env()
        return

    if args.fetch:
        run_fetch_sources()

    output_dir = args.output_dir
    rag_dir = output_dir / "rag_knowledge"
    output_dir.mkdir(parents=True, exist_ok=True)
    rag_dir.mkdir(parents=True, exist_ok=True)

    print("=== CNC 故障诊断数据构建 ===")
    print(f"数据根目录: {args.datasets_root}")
    print(f"输出目录:   {output_dir}")

    records, stats = process_all(args.datasets_root)
    records = deduplicate_records(records)
    stats["records_after_dedup"] = len(records)

    if not args.no_augment:
        records = augment_instruction_variants(records, seed=args.seed)
        stats["records_after_augment"] = len(records)

    random.Random(args.seed).shuffle(records)

    sft_data = strip_for_sft(records)
    sft_path = output_dir / "cnc_diagnosis_sft.json"
    with sft_path.open("w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    # RAG 知识库
    alarm_csvs = [PARSED_DIR / "fanuc_alarms.csv", PARSED_DIR / "siemens_alarms.csv"]
    rag_chunks = build_rag_knowledge_base(records, alarm_csvs)
    rag_path = rag_dir / "chunks.jsonl"
    with rag_path.open("w", encoding="utf-8") as f:
        for chunk in rag_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    unique_outputs = len({r["output"] for r in records})
    stats.update({
        "unique_outputs": unique_outputs,
        "sft_file": str(sft_path),
        "sft_records": len(sft_data),
        "rag_chunks": len(rag_chunks),
        "rag_file": str(rag_path),
        "architecture": "fine-tuning (SFT) + RAG",
        "format": "instruction / input / output (LLaMA-Factory Alpaca)",
    })
    with (output_dir / "processing_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # RAG 使用说明
    readme = rag_dir / "README.txt"
    readme.write_text(
        "RAG 知识库使用说明\n"
        "==================\n"
        f"分块文件: {rag_path.name} ({len(rag_chunks)} 块)\n"
        "向量库: ../rag_chroma (ChromaDB，需先运行 build_rag_index.py)\n"
        "推荐检索字段: text\n"
        "metadata.type: alarm_reference | maintenance_manual | diagnostic_qa\n\n"
        "微调数据: ../cnc_diagnosis_sft.json\n"
        "联合推理:\n"
        "  1. python -m src.build_rag_index --reset\n"
        "  2. python -m src.rag_chat --interactive\n",
        encoding="utf-8",
    )

    print("\n========== 构建完成 ==========")
    print(f"SFT 微调样本:     {len(sft_data)}")
    print(f"唯一 output 数:   {unique_outputs}")
    print(f"RAG 知识块:       {len(rag_chunks)}")
    print(f"微调数据:         {sft_path}")
    print(f"RAG 知识库:       {rag_path}")
    if len(sft_data) < 500:
        print(f"\n[提示] 当前 {len(sft_data)} 条，未达最低可用线 500 条。")
        print("  请运行: python process_cnc_datasets.py --fetch")
    else:
        print(f"\n[OK] 已达最低可用线（>=500 条）")


if __name__ == "__main__":
    main()
