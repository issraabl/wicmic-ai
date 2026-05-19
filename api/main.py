from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import uuid
import threading
import time

from rag.embedder import index_documents
from rag.retriever import generate_previsions, generate_benchmark, _chat_with_rag  

app = FastAPI(title="Wicmic Tripower AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)

# ── Store des jobs en mémoire ─────────────────────────────
jobs: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    print("[API] Indexation des documents...")
    index_documents()
    print("[API] ✅ Prêt.")


# ── Schémas ───────────────────────────────────────────────

class MoisData(BaseModel):
    mois:  str
    total: float

class EnergieHistorique(BaseModel):
    nom:   str
    unite: str
    mois:  List[MoisData]

class PrevisionRequest(BaseModel):
    energies: List[EnergieHistorique]

class EnergieBenchmark(BaseModel):
    nom:           str
    unite:         str
    moisActuel:    float
    moisPrecedent: float
    moyenne:       float

class BenchmarkRequest(BaseModel):
    energies: List[EnergieBenchmark]

class ChatRequest(BaseModel):        
    prompt:  str
    context: str = ""


# ── Worker thread ─────────────────────────────────────────

def run_job(job_id: str, fn, data: dict):
    """Exécute fn(data) dans un thread et stocke le résultat."""
    try:
        print(f"[API] Job {job_id} démarré...")
        result = fn(data)
        jobs[job_id] = {"status": "done", "result": result}
        print(f"[API] Job {job_id} terminé ✅")
    except Exception as e:
        print(f"[API] Job {job_id} erreur ❌ : {e}")
        jobs[job_id] = {"status": "error", "result": {"error": str(e)}}

    def cleanup():
        time.sleep(600)
        jobs.pop(job_id, None)
    threading.Thread(target=cleanup, daemon=True).start()


# ── Endpoints base ────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Wicmic Energy AI v2.0 opérationnel"}

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Endpoints prévisions ──────────────────────────────────

@app.post("/previsions/start")
def previsions_start(request: PrevisionRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "result": None}
    data = {"energies": [e.dict() for e in request.energies]}
    t = threading.Thread(target=run_job, args=(job_id, generate_previsions, data), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "pending"}

@app.get("/previsions/result/{job_id}")
def previsions_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable ou expiré.")
    return {"status": job["status"], "result": job.get("result")}


# ── Endpoints benchmark ───────────────────────────────────

@app.post("/benchmark/start")
def benchmark_start(request: BenchmarkRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "result": None}
    data = {"energies": [e.dict() for e in request.energies]}
    t = threading.Thread(target=run_job, args=(job_id, generate_benchmark, data), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "pending"}

@app.get("/benchmark/result/{job_id}")
def benchmark_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable ou expiré.")
    return {"status": job["status"], "result": job.get("result")}


# ── Reindex ───────────────────────────────────────────────

@app.post("/reindex")
def reindex():
    try:
        index_documents(force_reindex=True)
        return {"status": "ok", "message": "Réindexation terminée."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat RAG  ← NOUVEAU ───────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat IA enrichi avec RAG — contexte BD réelle."""
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _chat_with_rag, request.prompt, request.context
        )
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))