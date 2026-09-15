"""Índice Chroma opcional."""
from __future__ import annotations

import os

from hospital.db import connect, list_protocols
from hospital.paths import CHROMA_DIR


def build() -> str:
    protocols = list_protocols(connect())
    os.makedirs(CHROMA_DIR, exist_ok=True)
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
        path = os.path.join(CHROMA_DIR, "SKIPPED.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("chromadb/sentence-transformers não instalados. RAG lexical ativo.\n")
        return path

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    col = client.get_or_create_collection("protocols", embedding_function=embed)
    ids = [p["id"] for p in protocols]
    if ids:
        col.upsert(
            ids=ids,
            documents=[p["body"] for p in protocols],
            metadatas=[{"pmid": p["pmid"], "title": p["title"]} for p in protocols],
        )
    return CHROMA_DIR


if __name__ == "__main__":
    print(build())
