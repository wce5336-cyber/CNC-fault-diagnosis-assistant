"""Project path constants (repo root relative)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
PROCESSED = ROOT / "processed"
MODELS_DIR = ROOT / "models"

EXTERNAL_KNOWLEDGE = DATA / "external_knowledge"
SFT_JSON = PROCESSED / "cnc_diagnosis_sft.json"
RAG_CHUNKS = PROCESSED / "rag_knowledge" / "chunks.jsonl"
CHROMA_DIR = PROCESSED / "rag_chroma"
LORA_DIR = MODELS_DIR / "qwen2.5-7b-lora"
