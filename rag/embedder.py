import os
import json
import httpx
import numpy as np
from dotenv import load_dotenv
from rag.loader import build_documents

load_dotenv()

OLLAMA_URL    = os.getenv("OLLAMA_URL",         "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
STORE_PATH    = os.getenv("STORE_PATH",         "./chroma_db/store.json")


# ── Embedding via Ollama ──────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    url = f"{OLLAMA_URL}/api/embed"
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json={"model": OLLAMA_MODEL, "input": texts})
        resp.raise_for_status()
        return resp.json()["embeddings"]


# ── Store JSON simple ─────────────────────────────────────────────────────────

def _load_store() -> dict:
    if os.path.exists(STORE_PATH):
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": [], "embeddings": []}


def _save_store(store: dict) -> None:
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)


# ── Indexation ────────────────────────────────────────────────────────────────

def index_documents(force_reindex: bool = False) -> None:
    store = _load_store()

    if store["documents"] and not force_reindex:
        print(f"[Embedder] {len(store['documents'])} documents déjà indexés. Skip.")
        return

    documents = build_documents()
    if not documents:
        print("[Embedder] Aucun document à indexer.")
        return

    total      = len(documents)
    batch_size = 10
    all_embs   = []

    print(f"[Embedder] Indexation de {total} documents...")

    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        print(f"[Embedder] Batch {i+1}-{min(i+batch_size, total)}/{total}...")
        embs = embed_texts(batch)
        all_embs.extend(embs)

    _save_store({"documents": documents, "embeddings": all_embs})
    print(f"[Embedder] ✅ {total} documents indexés dans {STORE_PATH}")


# ── Recherche cosine ──────────────────────────────────────────────────────────

def search_documents(query: str, n_results: int = 5) -> list[str]:
    store = _load_store()
    if not store["documents"]:
        print("[Embedder] Store vide.")
        return []

    q_emb  = np.array(embed_texts([query])[0])
    matrix = np.array(store["embeddings"])

    # similarité cosine
    norms  = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_emb)
    norms  = np.where(norms == 0, 1e-10, norms)
    scores = matrix.dot(q_emb) / norms

    top_idx = np.argsort(scores)[::-1][:n_results]
    docs    = [store["documents"][i] for i in top_idx]
    print(f"[Embedder] {len(docs)} documents trouvés pour: '{query}'")
    return docs