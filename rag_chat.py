from __future__ import annotations

import os
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_DIR / "vectorstore" / "chroma"
COLLECTION_NAME = "nlead_chunks"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
MAX_SOURCE_CHARS = int(os.getenv("RAG_MAX_SOURCE_CHARS", "1200"))

_embed_model: SentenceTransformer | None = None
_collection = None


def get_collection():
    global _embed_model, _collection
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _embed_model, _collection


def retrieve_context(query: str, n_results: int = 4) -> list[dict]:
    embed_model, collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError(
            "Chroma collection is empty. Run notebook section 9 to index chunks first."
        )

    query_embedding = embed_model.encode([query], normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    contexts = []
    for i, (doc, meta, dist) in enumerate(
        zip(result["documents"][0], result["metadatas"][0], result["distances"][0]),
        start=1,
    ):
        contexts.append(
            {
                "ref": i,
                "distance": float(dist),
                "text": doc or "",
                "modality": meta.get("modality", ""),
                "sourceName": meta.get("sourceName", ""),
                "fileName": meta.get("fileName", ""),
                "unit": meta.get("unit", ""),
                "webUrl": meta.get("webUrl", ""),
            }
        )
    return contexts


def build_rag_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []
    for c in contexts:
        label = c["sourceName"]
        if c.get("fileName"):
            label = f"{label} / {c['fileName']}"
        if c.get("unit"):
            label = f"{label} [{c['unit']}]"
        body = c["text"] or ""
        if len(body) > MAX_SOURCE_CHARS:
            body = body[:MAX_SOURCE_CHARS] + "\n...[truncated]..."
        blocks.append(
            f"[Source {c['ref']}] ({c['modality']}) {label}\n"
            f"URL: {c.get('webUrl') or 'n/a'}\n"
            f"{body}"
        )

    context_text = "\n\n---\n\n".join(blocks)
    return (
        "You are a helpful assistant for the NLEAD SharePoint knowledge base.\n"
        "Answer the question using ONLY the sources below.\n"
        "If the sources are not enough, say you do not have enough information.\n"
        "Cite sources inline like [Source 1], [Source 2].\n"
        "Keep the answer clear and concise.\n\n"
        f"SOURCES:\n{context_text}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "Answer only from provided sources and cite them."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "").strip()


def format_sources(contexts: list[dict]) -> str:
    lines = ["", "**Sources**"]
    for c in contexts:
        label = c["sourceName"]
        if c.get("fileName"):
            label += f" / {c['fileName']}"
        if c.get("unit"):
            label += f" [{c['unit']}]"
        lines.append(f"- [Source {c['ref']}] {label}")
        if c.get("webUrl"):
            lines.append(f"  {c['webUrl']}")
    return "\n".join(lines)


def ask_rag(question: str, n_results: int = 4) -> str:
    contexts = retrieve_context(question, n_results=n_results)
    prompt = build_rag_prompt(question, contexts)
    answer = call_ollama(prompt)
    return f"{answer}\n{format_sources(contexts)}"
