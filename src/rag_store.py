#!/usr/bin/env python3
"""ChromaDB vector store for CNC RAG knowledge base."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .paths import CHROMA_DIR, RAG_CHUNKS

DEFAULT_CHUNKS = RAG_CHUNKS
DEFAULT_CHROMA_DIR = CHROMA_DIR
DEFAULT_COLLECTION = "cnc_knowledge"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def get_embedding_function(model_name: str = DEFAULT_EMBED_MODEL):
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_chroma_client(persist_dir: Path) -> chromadb.PersistentClient:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def build_index(
    chunks_path: Path = DEFAULT_CHUNKS,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embed_model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 64,
    reset: bool = False,
) -> int:
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    client = get_chroma_client(chroma_dir)
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    ef = get_embedding_function(embed_model)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    existing = set(collection.get(include=[]).get("ids", []))
    to_add = [c for c in chunks if c["id"] not in existing]
    if not to_add and collection.count() > 0:
        return collection.count()

    for i in range(0, len(to_add), batch_size):
        batch = to_add[i : i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c.get("metadata", {}) for c in batch],
        )

    return collection.count()


class RagRetriever:
    def __init__(
        self,
        chroma_dir: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        client = get_chroma_client(chroma_dir)
        ef = get_embedding_function(embed_model)
        self.collection = client.get_collection(
            name=collection_name,
            embedding_function=ef,
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        items: list[dict[str, Any]] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]
        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            items.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta or {},
                    "distance": dist,
                }
            )
        return items

    @staticmethod
    def format_context(hits: list[dict[str, Any]]) -> str:
        if not hits:
            return "（未检索到相关参考片段）"
        parts: list[str] = []
        for i, hit in enumerate(hits, 1):
            meta = hit.get("metadata", {})
            label = meta.get("type", "reference")
            code = meta.get("code", "")
            prefix = f"[{label}"
            if code:
                prefix += f" {code}"
            prefix += "]"
            parts.append(f"{i}. {prefix}\n{hit['text']}")
        return "\n\n".join(parts)
