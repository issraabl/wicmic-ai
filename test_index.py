import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from rag.loader import build_documents

print("Chargement documents...")
docs = build_documents()
print(f"{len(docs)} documents")

ef = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

try:
    client.delete_collection("wicmic_energy")
except:
    pass

collection = client.get_or_create_collection(
    name="wicmic_energy",
    embedding_function=ef
)

print("Test batch 1 (5 docs)...")
collection.add(documents=docs[:5], ids=[f"doc_{i}" for i in range(5)])
print("Batch 1 OK !")

print("Test batch 2 (32 docs)...")
collection.add(documents=docs[5:37], ids=[f"doc_{i}" for i in range(5, 37)])
print("Batch 2 OK !")

print(f"Total indexé : {collection.count()}")