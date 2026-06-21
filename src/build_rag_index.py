#!/usr/bin/env python3
"""Build ChromaDB index from RAG chunks.jsonl."""

from __future__ import annotations

import argparse
from pathlib import Path

from .rag_store import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_CHUNKS,
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    build_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB index for CNC RAG")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild collection")
    args = parser.parse_args()

    count = build_index(
        chunks_path=args.chunks,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        embed_model=args.embed_model,
        reset=args.reset,
    )
    print(f"ChromaDB index ready: {args.chroma_dir}")
    print(f"Collection: {args.collection}")
    print(f"Total vectors: {count}")


if __name__ == "__main__":
    main()
