"""
main.py — DIA-Legal FastAPI Backend
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

print("Python starting...", flush=True)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
print("FastAPI imported", flush=True)

from dotenv import load_dotenv
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────
BASE_STORAGE = "data"
DB_PATH      = "data/lancedb"
CACHE_FILE   = "data/ingestion_cache.json"
CASES_FILE   = "data/cases_registry.json"

# ── Global system instances ───────────────────────────────────────
_systems = {}

# ── Lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[DIA-Legal] Server starting...", flush=True)
    yield
    print("[DIA-Legal] Shutdown.", flush=True)

app = FastAPI(title="DIA-Legal API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Lazy system loader ────────────────────────────────────────────
def get_systems():
    if not _systems:
        print("[DIA-Legal] Loading ML systems...", flush=True)
        _systems.update(_build_systems())
        print("[DIA-Legal] ML systems ready.", flush=True)
    return _systems

# ── System bootstrap ──────────────────────────────────────────────
def _build_systems():
    print("Loading transformers...", flush=True)
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    print("Loading src modules...", flush=True)
    from src.reader import ReaderRouter
    from src.video_processor import VideoProcessor
    from src.chunker import Chunker, PDFChunker
    from src.embedder import MultiModalEmbedder, TextEmbedder, VisualEmbedder
    from src.vectorstore import LanceDBVectorStore
    from src.retriever import Retriever
    from src.reranker import CrossEncoderReranker
    from src.llmclient import LLMClient
    from src.context_builder import ContextBuilder
    from src.llm_answerer import LLMAnswerer
    from src.pipeline import DIAPipeline
    from src.query_router import QueryRouter
    from src.ingestion import IngestionPipeline
    from src.evidence_classifier import EvidenceClassifier, format_evidence_map
    from src.contradiction_detector import ContradictionDetector, format_contradiction_report
    from src.devils_advocate import DevilsAdvocate
    from src.trial_brief_generator import TrialBriefGenerator
    print("All modules imported", flush=True)

    tokenizer       = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    video_processor = VideoProcessor(base_output_path=BASE_STORAGE, model_size="base")
    chunker         = Chunker(max_duration=20, max_tokens=512, overlap_duration=5, tokenizer=tokenizer)
    pdf_chunker     = PDFChunker(max_tokens=512, overlap_tokens=50)

    text_model      = SentenceTransformer("all-MiniLM-L6-v2")
    text_embedder   = TextEmbedder(model=text_model, batch_size=12, normalize=True)
    visual_embedder = VisualEmbedder(model_name="openai/clip-vit-base-patch32", device="cpu", normalize=True)
    embedder        = MultiModalEmbedder(text_embedder=text_embedder, visual_embedder=visual_embedder)

    vector_store = LanceDBVectorStore(
        table_name="dia_legal_v2", db_path=DB_PATH,
        text_dim=text_embedder.embed_dims, visual_dim=visual_embedder.embed_dim
    )
    reranker  = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=32, normalize=True)
    retriever = Retriever(
        vector_store=vector_store, embedder=embedder,
        max_candidates=30, reranker=reranker,
        enable_mmr=True, mmr_lambda=0.5, min_threshold=0.0, temporary_window=5
    )

    context_builder = ContextBuilder(max_tokens=2000, include_scores=True)
    llm_client      = LLMClient(api_token=os.environ.get("HF_TOKEN"))
    answerer        = LLMAnswerer(llm_client=llm_client, confidence_threshold=0.3, max_history=3)

    pipeline = DIAPipeline(
        retriever=retriever, context_builder=context_builder,
        llm_answerer=answerer, query_router=QueryRouter()
    )
    ingestion_pipeline = IngestionPipeline(
        reader_router=ReaderRouter, video_processor=video_processor,
        chunker=chunker, embedder=embedder,
        vector_store=vector_store, pdf_chunker=pdf_chunker
    )
    evidence_classifier = EvidenceClassifier(
        retriever=retriever, llm_client=llm_client,
        context_builder=context_builder, top_k=50
    )
    contradiction_detector = ContradictionDetector(
        retriever=retriever, llm_client=llm_client,
        embedder=embedder, similarity_threshold=0.4, top_k=100
    )
    devils_advocate = DevilsAdvocate(
        retriever=retriever, llm_client=llm_client,
        storage_path=BASE_STORAGE, top_k=10
    )
    brief_generator = TrialBriefGenerator(
        retriever=retriever, llm_client=llm_client,
        evidence_classifier=evidence_classifier,
        contradiction_detector=contradiction_detector,
        devils_advocate=devils_advocate,
        storage_path=BASE_STORAGE
    )

    # Store format_* functions for use in routes
    _systems["_format_evidence_map"]          = format_evidence_map
    _systems["_format_contradiction_report"]  = format_contradiction_report

    return {
        "pipeline":                pipeline,
        "ingestion":               ingestion_pipeline,
        "evidence_classifier":     evidence_classifier,
        "contradiction_detector":  contradiction_detector,
        "devils_advocate":         devils_advocate,
        "brief_generator":         brief_generator,
        "retriever":               retriever,
        "embedder":                embedder,
    }

# ── Cache helpers ─────────────────────────────────────────────────
def _load_cache():
    Path(CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def _save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def _cache_key(source, case_id):
    return hashlib.md5(f"{source}::{case_id}".encode()).hexdigest()

def _load_cases():
    Path(CASES_FILE).parent.mkdir(parents=True, exist_ok=True)
    if Path(CASES_FILE).exists():
        with open(CASES_FILE) as f:
            return json.load(f)
    return {}

def _save_cases(cases):
    with open(CASES_FILE, "w") as f:
        json.dump(cases, f, indent=2)

def _register_case(case_id, name, source_type, chunks):
    cases = _load_cases()
    if case_id not in cases:
        cases[case_id] = {"sources": [], "total_chunks": 0}
    existing = [s["name"] for s in cases[case_id]["sources"]]
    if name not in existing:
        cases[case_id]["sources"].append({"name": name, "type": source_type, "chunks": chunks})
    cases[case_id]["total_chunks"] = sum(s["chunks"] for s in cases[case_id]["sources"])
    _save_cases(cases)


# ════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    case_id: str
    query: str
    mode: Optional[str] = "evidence"

class EvidenceMapRequest(BaseModel):
    case_id: str
    lawyer_position: str

class ContradictionRequest(BaseModel):
    case_id: str

class DANewSessionRequest(BaseModel):
    case_id: str
    topic: str

class DAArgueRequest(BaseModel):
    session_id: str
    case_id: str
    argument: str

class BriefRequest(BaseModel):
    case_id: str
    lawyer_position: str

class YouTubeIngestRequest(BaseModel):
    case_id: str
    url: str


# ════════════════════════════════════════════════════════════════
# HEALTH — no ML needed, always fast
# ════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


# ════════════════════════════════════════════════════════════════
# ROUTES — CASES
# ════════════════════════════════════════════════════════════════

@app.get("/api/cases")
def list_cases():
    cases = _load_cases()
    result = []
    for cid, data in cases.items():
        result.append({
            "case_id":      cid,
            "source_count": len(data.get("sources", [])),
            "total_chunks": data.get("total_chunks", 0),
            "sources":      data.get("sources", []),
        })
    return {"cases": result}


@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    cases = _load_cases()
    if case_id not in cases:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"case_id": case_id, **cases[case_id]}


@app.delete("/api/cases/{case_id}")
def delete_case(case_id: str):
    cases = _load_cases()
    if case_id not in cases:
        raise HTTPException(status_code=404, detail="Case not found")
    del cases[case_id]
    _save_cases(cases)
    return {"status": "deleted", "case_id": case_id}


# ════════════════════════════════════════════════════════════════
# ROUTES — INGEST
# ════════════════════════════════════════════════════════════════

@app.post("/api/ingest/youtube")
def ingest_youtube(body: YouTubeIngestRequest):
    case_id = body.case_id.strip()
    url     = body.url.strip()
    cache   = _load_cache()
    key     = _cache_key(url, case_id)

    if key in cache:
        return {"status": "cached", "chunks": cache[key].get("chunks", 0)}

    try:
        s = get_systems()
        result = s["ingestion"].ingest(source=url, case_id=case_id, storage_path=BASE_STORAGE)
        n = result.get("chunks_indexed", 0)
        cache[key] = {"source": url, "case_id": case_id, "chunks": n}
        _save_cache(cache)
        _register_case(case_id, url[:60], "video", n)
        return {"status": "success", "chunks": n, "case_id": case_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/video")
async def ingest_video(case_id: str = Form(...), file: UploadFile = File(...)):
    case_id = case_id.strip()
    tmp_dir = Path(BASE_STORAGE) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.filename

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cache = _load_cache()
    key   = _cache_key(str(tmp_path), case_id)

    if key in cache:
        tmp_path.unlink(missing_ok=True)
        return {"status": "cached", "chunks": cache[key].get("chunks", 0)}

    try:
        s = get_systems()
        result = s["ingestion"].ingest(source=str(tmp_path), case_id=case_id, storage_path=BASE_STORAGE)
        n = result.get("chunks_indexed", 0)
        cache[key] = {"source": file.filename, "case_id": case_id, "chunks": n}
        _save_cache(cache)
        _register_case(case_id, file.filename, "video", n)
        return {"status": "success", "chunks": n, "case_id": case_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/ingest/pdf")
async def ingest_pdf(case_id: str = Form(...), files: List[UploadFile] = File(...)):
    case_id = case_id.strip()
    results = []
    tmp_dir = Path(BASE_STORAGE) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()

    for file in files:
        tmp_path = tmp_dir / file.filename
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        key = _cache_key(file.filename, case_id)
        if key in cache:
            tmp_path.unlink(missing_ok=True)
            results.append({"name": file.filename, "status": "cached", "chunks": cache[key].get("chunks", 0)})
            continue

        try:
            s = get_systems()
            result  = s["ingestion"].ingest(source=str(tmp_path), case_id=case_id, storage_path=BASE_STORAGE)
            n       = result.get("chunks_indexed", 0)
            doctype = result.get("doc_type", "document")
            cache[key] = {"source": file.filename, "case_id": case_id, "chunks": n}
            _register_case(case_id, file.filename, doctype, n)
            results.append({"name": file.filename, "status": "success", "chunks": n, "doc_type": doctype})
        except Exception as e:
            results.append({"name": file.filename, "status": "error", "error": str(e)})
        finally:
            tmp_path.unlink(missing_ok=True)

    _save_cache(cache)
    return {"results": results, "case_id": case_id}


# ════════════════════════════════════════════════════════════════
# ROUTES — QUERY
# ════════════════════════════════════════════════════════════════

@app.post("/api/query")
def query(body: QueryRequest):
    try:
        s      = get_systems()
        result = s["pipeline"].run(query=body.query.strip(), case_id=body.case_id.strip())
        return {
            "answer":     result.get("answer", ""),
            "citations":  result.get("citations", []),
            "confidence": result.get("confidence", 0.0),
            "mode":       result.get("mode", body.mode),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════
# ROUTES — EVIDENCE MAP
# ════════════════════════════════════════════════════════════════

@app.post("/api/evidence-map")
def evidence_map(body: EvidenceMapRequest):
    try:
        s   = get_systems()
        em  = s["evidence_classifier"].classify(
            case_id=body.case_id.strip(),
            lawyer_position=body.lawyer_position.strip()
        )
        fmt = s["_format_evidence_map"](em)
        return {
            "supporting": fmt["supporting"]["rows"],
            "opposing":   fmt["opposing"]["rows"],
            "neutral":    fmt["neutral"]["rows"],
            "summary":    em.summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════
# ROUTES — CONTRADICTIONS
# ════════════════════════════════════════════════════════════════

@app.post("/api/contradictions")
def detect_contradictions(body: ContradictionRequest):
    try:
        s      = get_systems()
        report = s["contradiction_detector"].detect(case_id=body.case_id.strip())
        fmt    = s["_format_contradiction_report"](report)
        return {"contradictions": fmt["rows"], "summary": report.summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════
# ROUTES — DEVIL'S ADVOCATE
# ════════════════════════════════════════════════════════════════

@app.post("/api/devils-advocate/session")
def da_new_session(body: DANewSessionRequest):
    try:
        s       = get_systems()
        session = s["devils_advocate"].new_session(
            case_id=body.case_id.strip(), topic=body.topic.strip()
        )
        return {"session_id": session.session_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/devils-advocate/argue")
def da_argue(body: DAArgueRequest):
    try:
        s      = get_systems()
        da     = s["devils_advocate"]
        da.load_session(case_id=body.case_id.strip(), session_id=body.session_id.strip())
        result = da.argue(body.argument.strip())
        return {
            "critique":       result.critique,
            "weaknesses":     result.weaknesses,
            "opposition":     result.opposition_argument,
            "strengthen":     result.how_to_strengthen,
            "round_number":   result.round_number,
            "evidence_count": len(result.supporting_evidence),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/devils-advocate/sessions/{case_id}")
def da_list_sessions(case_id: str):
    try:
        s        = get_systems()
        sessions = s["devils_advocate"].list_sessions(case_id.strip())
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════
# ROUTES — TRIAL BRIEF
# ════════════════════════════════════════════════════════════════

@app.post("/api/brief")
def generate_brief(body: BriefRequest):
    try:
        s     = get_systems()
        brief = s["brief_generator"].generate(
            case_id=body.case_id.strip(),
            lawyer_position=body.lawyer_position.strip()
        )
        return {
            "brief_id":           brief.brief_id,
            "case_strength":      brief.case_strength,
            "case_summary":       brief.case_summary,
            "overall_assessment": brief.overall_assessment,
            "critical_actions":   brief.critical_actions,
            "witness_profiles": [
                {
                    "speaker_id":           w.speaker_id,
                    "inferred_role":        w.inferred_role,
                    "reliability_rating":   w.reliability_rating,
                    "credibility_score":    w.credibility_score,
                    "contradiction_count":  w.contradiction_count,
                    "recommended_approach": w.recommended_approach,
                }
                for w in brief.witness_profiles
            ],
            "contradictions":        brief.contradictions,
            "recommended_questions": brief.recommended_questions,
            "pdf_path":              str(brief.pdf_path) if getattr(brief, "pdf_path", None) else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/brief/download/{brief_id}")
def download_brief(brief_id: str):
    pdf_path = Path(BASE_STORAGE) / "briefs" / f"{brief_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Brief PDF not found")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"trial_brief_{brief_id}.pdf"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)