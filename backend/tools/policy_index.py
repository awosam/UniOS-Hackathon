"""
policy_index.py — ChromaDB-backed index for scraped policy chunks with section/subsection metadata.

Stores chunks from policy_scraper; search_policies returns chunks with section, subsection, url
so pathfinder and drafter can cite at subsection level (e.g. "Academic Regulations > 90% Rule").
"""

import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

# Persistent store under backend/data so it survives restarts
_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
COLLECTION_NAME = "policy_chunks"


def _get_client():
    """Persistent ChromaDB client; creates data dir if needed."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=_DATA_DIR, settings=Settings(anonymized_telemetry=False))


def _get_collection():
    return _get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "UW policy chunks with section/subsection for RAG"},
    )


def upsert_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Upsert scraped policy chunks into ChromaDB. Each chunk must have:
      id, text, metadata (source_key, section, subsection, url).
    Returns number of chunks upserted.
    """
    if not chunks:
        return 0
    coll = _get_collection()
    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        m = c.get("metadata", {}).copy()
        # ChromaDB metadata values must be str, int, float, or bool
        metadatas.append({
            "source_key": str(m.get("source_key", "")),
            "section": str(m.get("section", "")),
            "subsection": str(m.get("subsection", "")),
            "url": str(m.get("url", "")),
        })
    coll.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def search_policies(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Semantic search over policy chunks. Returns list of dicts with:
      text, url, source_key, section, subsection (for citations).
    """
    coll = _get_collection()
    if coll.count() == 0:
        return []
    result = coll.query(query_texts=[query], n_results=min(k, coll.count()))
    out = []
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    for i, doc_id in enumerate(ids):
        text = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        out.append({
            "text": text,
            "url": meta.get("url", ""),
            "source_key": meta.get("source_key", ""),
            "section": meta.get("section", ""),
            "subsection": meta.get("subsection", ""),
        })
    return out


def clear() -> None:
    """Delete all policy chunks (for full re-scrape)."""
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
