"""
main.py — DIA-Legal FastAPI Backend
"""

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION"] = "0"
import sys
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

print("Python starting...", flush=True)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import concurrent.futures
import queue as _queue
import threading
import time
import shutil
import uuid as _uuid
print("FastAPI imported", flush=True)

from dotenv import load_dotenv
try:
    dotenv_path = Path(".env")
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)
except Exception as exc:
    print(f"[DIA-Legal] Skipping .env load: {exc}", flush=True)

# ── Constants ─────────────────────────────────────────────────────
BASE_STORAGE = "data"
DB_PATH      = "data/lancedb"
CACHE_FILE   = "data/ingestion_cache.json"
CASES_FILE   = "data/cases_registry.json"

# ── Ingest limits ─────────────────────────────────────────────────
PDF_MAX_FILES       = 5
PDF_MAX_BYTES       = 12  * 1024 * 1024   # 12 MB
VIDEO_MAX_BYTES     = 250 * 1024 * 1024   # 250 MB

# ── Global system instances ───────────────────────────────────────
_systems = {}
_systems_lock = threading.Lock()

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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    # Here you would typically log this to a structured logger or metrics system
    print(f"INFO:     {request.method} {request.url.path} completed in {process_time:.4f}s with status {response.status_code}")
    response.headers["X-Process-Time"] = str(process_time)
    return response
# ── Lazy system loader ────────────────────────────────────────────
def get_systems():
    if not _systems:
        with _systems_lock:
            if not _systems:
                print("[DIA-Legal] Loading ML systems...", flush=True)
                try:
                    _systems.update(_build_systems())
                except Exception as exc:
                    print(f"[DIA-Legal] STARTUP ERROR: {exc}", flush=True)
                    raise HTTPException(status_code=503, detail=str(exc))
                print("[DIA-Legal] ML systems ready.", flush=True)
    return _systems

# ── System bootstrap ──────────────────────────────────────────────
def _build_systems():
    print("Loading transformers (lazy)...", flush=True)
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

    video_processor = VideoProcessor(base_output_path=BASE_STORAGE, model_size="tiny")
    chunker         = Chunker(max_duration=20, max_tokens=512, overlap_duration=5)
    pdf_chunker     = PDFChunker(max_tokens=512, overlap_tokens=50)

    text_embedder   = TextEmbedder(model_name="all-MiniLM-L6-v2", batch_size=32, normalize=True)
    visual_embedder = VisualEmbedder(model_name="openai/clip-vit-base-patch32", device="cpu", normalize=True)
    embedder        = MultiModalEmbedder(text_embedder=text_embedder, visual_embedder=visual_embedder)

    vector_store = LanceDBVectorStore(
        table_name="dia_legal_v2", db_path=DB_PATH,
        text_dim=384, visual_dim=512
    )
    reranker  = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=32, normalize=True)
    retriever = Retriever(
        vector_store=vector_store, embedder=embedder,
        max_candidates=30, reranker=reranker,
        enable_mmr=True, mmr_lambda=0.5, min_threshold=0.0, temporary_window=5
    )

    context_builder = ContextBuilder(max_tokens=200, include_scores=True)
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
        embedder=embedder, similarity_threshold=0.3, top_k=100
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


def _fmt_ts(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _assert_case_has_sources(case_id: str):
    cases = _load_cases()
    data  = cases.get(case_id)
    if not data or data.get("total_chunks", 0) == 0:
        raise HTTPException(
            status_code=422,
            detail="No sources indexed for this case. Please ingest documents or videos first."
        )


# ════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    case_id: str
    query: str
    mode: Optional[str] = "evidence"
    history: List[Dict] = []

class ChatSaveRequest(BaseModel):
    messages: List[Dict]
    session_id: str
    title: Optional[str] = "New Chat"

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

class CreateCaseRequest(BaseModel):
    case_id: str


# ════════════════════════════════════════════════════════════════
# HEALTH — no ML needed, always fast
# ════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


# ════════════════════════════════════════════════════════════════
# ROUTES — CASES
# ════════════════════════════════════════════════════════════════

@app.post("/api/cases")
def create_case(body: CreateCaseRequest):
    case_id = body.case_id.strip()
    if not case_id:
        raise HTTPException(status_code=400, detail="case_id is required")
    cases = _load_cases()
    if case_id not in cases:
        cases[case_id] = {"sources": [], "total_chunks": 0}
        _save_cases(cases)
    return {"case_id": case_id}


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


@app.get("/api/cases/{case_id}/stats")
def case_stats(case_id: str):
    try:
        s     = get_systems()
        stats = s["retriever"].vector_store.get_case_stats(case_id)
        return {"case_id": case_id, **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    content = await file.read()
    if len(content) > VIDEO_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f'"{file.filename}" is {len(content) // (1024*1024)} MB — video files must be 250 MB or smaller'
        )
    tmp_dir = Path(BASE_STORAGE) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.filename
    tmp_path.write_bytes(content)
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
# SSE STREAMING INGEST ENDPOINTS
# ════════════════════════════════════════════════════════════════

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

def _sse(data: dict) -> str:
    import json
    return f"data: {json.dumps(data)}\n\n"


def _make_sse_generator(work_fn, loop, timeout: int = 900):
    """
    Run *work_fn* in a thread-pool executor and stream its progress events
    back as SSE.  work_fn receives a single argument: a thread-safe emit()
    callable that puts dicts onto an asyncio.Queue.
    """
    _KEEPALIVE_INTERVAL = 15  # seconds between keepalive pings

    q: asyncio.Queue = asyncio.Queue()

    def emit(event: dict):
        loop.call_soon_threadsafe(q.put_nowait, event)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, work_fn, emit)

    async def gen():
        elapsed = 0
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=_KEEPALIVE_INTERVAL)
                yield _sse(event)
                if event.get("stage") in ("done", "error", "cached"):
                    break
            except asyncio.TimeoutError:
                elapsed += _KEEPALIVE_INTERVAL
                if elapsed >= timeout:
                    yield _sse({"stage": "error", "error": "Processing timed out"})
                    break
                # SSE comment — proxies see data and keep the connection alive;
                # browsers ignore comment lines so the UI state is unchanged.
                yield ": keepalive\n\n"

    return gen()


@app.post("/api/ingest/youtube/stream")
async def ingest_youtube_stream(body: YouTubeIngestRequest):
    case_id = body.case_id.strip()
    url     = body.url.strip()
    cache   = _load_cache()
    key     = _cache_key(url, case_id)

    if key in cache:
        async def _cached():
            yield _sse({"stage": "cached", "chunks": cache[key].get("chunks", 0)})
        return StreamingResponse(_cached(), media_type="text/event-stream",
                                 headers=_SSE_HEADERS)

    loop = asyncio.get_event_loop()

    def work(emit):
        emit({"stage": "fetching_stream"})
        try:
            s = get_systems()
            result = s["ingestion"].ingest(
                source=url, case_id=case_id,
                storage_path=BASE_STORAGE,
                progress=lambda stage: emit({"stage": stage}),
            )
            n = result.get("chunks_indexed", 0)
            cache[key] = {"source": url, "case_id": case_id, "chunks": n}
            _save_cache(cache)
            _register_case(case_id, url[:60], "video", n)
            emit({"stage": "done", "chunks": n})
        except Exception as e:
            emit({"stage": "error", "error": str(e)})

    return StreamingResponse(
        _make_sse_generator(work, loop),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@app.post("/api/ingest/video/stream")
async def ingest_video_stream(case_id: str = Form(...), file: UploadFile = File(...)):
    """
    Streams the upload directly into a single FFmpeg process that writes
    audio WAV + scene-sampled frames — no temp video file on disk.
    Progress is delivered as SSE.
    """
    case_id    = case_id.strip()
    if file.size and file.size > VIDEO_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f'"{file.filename}" is {file.size // (1024*1024)} MB — video files must be 250 MB or smaller'
        )
    evidence_id = str(_uuid.uuid4())

    base       = Path(BASE_STORAGE)
    audio_dir  = base / case_id / "audio"
    frames_dir = base / case_id / "frames" / evidence_id
    audio_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    audio_path    = audio_dir / f"{evidence_id}.wav"
    frames_pat    = str(frames_dir / "%04d.jpg")

    # Single FFmpeg process: pipe:0 → WAV + JPEG frames (no video saved)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", "pipe:0",
        # audio track
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(audio_path),
        # frame track
        "-an", "-vf", "select='gt(scene,0.3)',scale=640:-1", "-vsync", "vfr",
        frames_pat,
    ]

    proc = await asyncio.create_subprocess_exec(
        *ffmpeg_cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )

    loop = asyncio.get_event_loop()

    async def event_gen():
        yield _sse({"stage": "uploading"})

        # Stream upload into FFmpeg stdin
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            proc.stdin.write(chunk)
        proc.stdin.close()
        await proc.wait()

        if proc.returncode != 0:
            stderr_bytes = await proc.stderr.read()
            yield _sse({"stage": "error", "error": f"FFmpeg: {stderr_bytes.decode()[:300]}"})
            return

        # ML pipeline runs in a thread — progress piped back via asyncio.Queue
        q: asyncio.Queue = asyncio.Queue()

        def ml_work():
            try:
                s = get_systems()
                result = s["ingestion"].ingest_from_files(
                    audio_path=str(audio_path),
                    frames_dir=frames_dir,
                    case_id=case_id,
                    evidence_id=evidence_id,
                    source_type="Local",
                    progress=lambda stage: loop.call_soon_threadsafe(
                        q.put_nowait, {"stage": stage}
                    ),
                )
                n = result.get("chunks_indexed", 0)
                fname = file.filename or "video"
                cache = _load_cache()
                key   = _cache_key(fname, case_id)
                cache[key] = {"source": fname, "case_id": case_id, "chunks": n}
                _save_cache(cache)
                _register_case(case_id, fname, "video", n)
                loop.call_soon_threadsafe(q.put_nowait, {"stage": "done", "chunks": n})
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, {"stage": "error", "error": str(exc)})

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop.run_in_executor(executor, ml_work)

        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=900)
                yield _sse(event)
                if event.get("stage") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                yield _sse({"stage": "error", "error": "ML processing timed out"})
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@app.post("/api/ingest/pdf/stream")
async def ingest_pdf_stream(case_id: str = Form(...), files: List[UploadFile] = File(...)):
    case_id = case_id.strip()

    if len(files) > PDF_MAX_FILES:
        raise HTTPException(
            status_code=413,
            detail=f'You sent {len(files)} PDFs — the limit is {PDF_MAX_FILES} files per ingest'
        )

    tmp_dir = Path(BASE_STORAGE) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Save all uploads to tmp immediately (they're small — PDFs)
    saved: List[dict] = []
    cache = _load_cache()
    for f in files:
        content = await f.read()
        if len(content) > PDF_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f'"{f.filename}" is {len(content) // (1024*1024)} MB — PDF files must be 12 MB or smaller'
            )
        key = _cache_key(f.filename, case_id)
        tmp_path = tmp_dir / f.filename
        tmp_path.write_bytes(content)
        saved.append({"file": f, "tmp_path": tmp_path, "key": key})

    loop = asyncio.get_event_loop()

    def work(emit):
        results = []
        for item in saved:
            tmp_path = item["tmp_path"]
            fname    = item["file"].filename
            key      = item["key"]
            if key in cache:
                emit({"stage": "cached", "file": fname, "chunks": cache[key].get("chunks", 0)})
                results.append({"name": fname, "status": "cached"})
                tmp_path.unlink(missing_ok=True)
                continue
            try:
                emit({"stage": "reading", "file": fname})
                s = get_systems()
                result  = s["ingestion"].ingest(
                    source=str(tmp_path), case_id=case_id,
                    storage_path=BASE_STORAGE,
                    progress=lambda stage, fn=fname: emit({"stage": stage, "file": fn}),
                )
                n       = result.get("chunks_indexed", 0)
                doctype = result.get("doc_type", "document")
                cache[key] = {"source": fname, "case_id": case_id, "chunks": n}
                _register_case(case_id, fname, doctype, n)
                emit({"stage": "file_done", "file": fname, "chunks": n})
                results.append({"name": fname, "status": "success", "chunks": n})
            except Exception as exc:
                emit({"stage": "file_error", "file": fname, "error": str(exc)})
                results.append({"name": fname, "status": "error", "error": str(exc)})
            finally:
                tmp_path.unlink(missing_ok=True)

        _save_cache(cache)
        emit({"stage": "done", "results": results})

    return StreamingResponse(
        _make_sse_generator(work, loop),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ════════════════════════════════════════════════════════════════
# ROUTES — QUERY
# ════════════════════════════════════════════════════════════════

@app.post("/api/query")
def query(body: QueryRequest):
    try:
        s      = get_systems()
        stats  = s["retriever"].vector_store.get_case_stats(body.case_id.strip())
        result = s["pipeline"].run(query=body.query.strip(), case_id=body.case_id.strip(), history=body.history, stats=stats)
        return {
            "answer":     result.get("answer", ""),
            "citations":  result.get("citations", []),
            "confidence": result.get("confidence", 0.0),
            "mode":       result.get("mode", body.mode),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ════════════════════════════════════════════════════════════════
# ROUTES — CHAT MEMORY (SESSIONS)
# ════════════════════════════════════════════════════════════════

@app.get("/api/cases/{case_id}/chat/sessions")
def get_chat_sessions(case_id: str):
    sessions_dir = Path(BASE_STORAGE) / case_id / "chat_sessions"
    if not sessions_dir.exists():
        return {"sessions": []}
    
    sessions = []
    for f in sessions_dir.glob("*.json"):
        try:
            with open(f, "r") as file:
                data = json.load(file)
                sessions.append({
                    "session_id": f.stem,
                    "title": data.get("title", "New Chat"),
                    "updated_at": f.stat().st_mtime
                })
        except:
            continue
    # Sort by updated_at descending
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    return {"sessions": sessions}

@app.get("/api/cases/{case_id}/chat/sessions/{session_id}")
def get_chat_session(case_id: str, session_id: str):
    chat_file = Path(BASE_STORAGE) / case_id / "chat_sessions" / f"{session_id}.json"
    if chat_file.exists():
        try:
            with open(chat_file, "r") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read chat session: {e}")
    return {"messages": [], "title": "New Chat"}

@app.post("/api/cases/{case_id}/chat/sessions/{session_id}")
def save_chat_session(case_id: str, session_id: str, body: ChatSaveRequest):
    sessions_dir = Path(BASE_STORAGE) / case_id / "chat_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    chat_file = sessions_dir / f"{session_id}.json"
    try:
        with open(chat_file, "w") as f:
            json.dump({"messages": body.messages, "title": body.title}, f, indent=2)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save chat session: {e}")

@app.delete("/api/cases/{case_id}/chat/sessions/{session_id}")
def delete_chat_session(case_id: str, session_id: str):
    chat_file = Path(BASE_STORAGE) / case_id / "chat_sessions" / f"{session_id}.json"
    try:
        if chat_file.exists():
            chat_file.unlink()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chat session: {e}")


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
    _assert_case_has_sources(body.case_id.strip())
    try:
        s      = get_systems()
        report = s["contradiction_detector"].detect(case_id=body.case_id.strip())
        contradictions = [
            {
                "severity":          c.severity,
                "speaker_a":         c.statement_a.speaker,
                "citation_a":        c.statement_a.citation_ref,
                "statement_a":       c.statement_a.text[:300],
                "timestamp_a":       _fmt_ts(c.statement_a.start_time) if c.statement_a.start_time > 0 else "",
                "source_name_a":     c.statement_a.original_name,
                "speaker_b":         c.statement_b.speaker,
                "citation_b":        c.statement_b.citation_ref,
                "statement_b":       c.statement_b.text[:300],
                "timestamp_b":       _fmt_ts(c.statement_b.start_time) if c.statement_b.start_time > 0 else "",
                "source_name_b":     c.statement_b.original_name,
                "explanation":       c.explanation,
                "is_cross_source":   c.is_cross_source,
                "conflict_score":    getattr(c, "conflict_score", 0.5),
                "conflict_type":     getattr(c, "conflict_type", "factual"),
                "legal_significance": getattr(c, "legal_significance", ""),
            }
            for c in report.contradictions
        ]
        return {"contradictions": contradictions, "summary": report.summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════
# ROUTES — DEVIL'S ADVOCATE
# ════════════════════════════════════════════════════════════════

@app.post("/api/devils-advocate/session")
def da_new_session(body: DANewSessionRequest):
    _assert_case_has_sources(body.case_id.strip())
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
    _assert_case_has_sources(body.case_id.strip())
    try:
        s      = get_systems()
        da     = s["devils_advocate"]
        da.load_session(case_id=body.case_id.strip(), session_id=body.session_id.strip())
        result = da.argue(body.argument.strip())
        return {
            "critique":                    result.critique,
            "weaknesses":                  result.weaknesses,
            "counter_argument":            result.counter_argument,
            "defense_probability":         result.defense_probability,
            "recommended_defense_strategy": result.recommended_defense_strategy,
            "round_number":                result.round_number,
            "evidence_count":              len(result.evidence_used),
            "lawyer_argument":             body.argument.strip(),
            # backward compat
            "opposition":                  result.counter_argument,
            "strengthen":                  result.how_to_strengthen,
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
    _assert_case_has_sources(body.case_id.strip())
    try:
        s     = get_systems()
        brief = s["brief_generator"].generate(
            case_id=body.case_id.strip(),
            lawyer_position=body.lawyer_position.strip()
        )
        return {
            "brief_id":            brief.brief_id,
            "case_strength":       brief.case_strength,
            "case_strength_score": brief.case_strength_score,
            "case_summary":        brief.case_summary,
            "overall_assessment":  brief.overall_assessment,
            "assessment":          brief.assessment,
            "recommended_actions": brief.recommended_actions,
            "key_risks":           brief.key_risks,
            "opposition_strategy": brief.opposition_strategy,
            "questions_to_ask":    brief.questions_to_ask,
            "strongest_arguments": brief.strongest_arguments,
            "vulnerabilities":     brief.vulnerabilities,
            "key_evidence":        brief.key_evidence,
            "contradictions_summary_detail": brief.contradictions_summary_detail,
            "cross_examination_detail":      brief.cross_examination_detail,
            "brief_weaknesses":              brief.brief_weaknesses,
            "witness_profiles": [
                {
                    "speaker_id":           w.speaker_id,
                    "inferred_role":        w.inferred_role,
                    "reliability_rating":   w.reliability_rating,
                    "credibility_score":    w.credibility_score,
                    "contradiction_count":  w.contradiction_count,
                    "recommended_approach": w.recommended_approach,
                    "key_statements":       w.key_statements,
                    "suggested_questions":  w.suggested_questions,
                }
                for w in brief.witness_profiles
            ],
            "contradictions": brief.contradictions,
            "pdf_path":       str(brief.pdf_path) if getattr(brief, "pdf_path", None) else None,
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


# ════════════════════════════════════════════════════════════════
# ROUTES — STATIC FILES
# Mounting StaticFiles at "/" breaks ASGI lifespan in some Starlette
# versions. Use an explicit assets mount + catch-all GET instead.
# ════════════════════════════════════════════════════════════════
frontend_path = Path("frontend/dist")
if frontend_path.exists():
    assets_path = frontend_path / "assets"
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        from fastapi.responses import FileResponse
        return FileResponse(str(frontend_path / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        from fastapi.responses import FileResponse
        file = frontend_path / full_path
        if file.is_file():
            return FileResponse(str(file))
        return FileResponse(str(frontend_path / "index.html"))
else:
    print(f"Warning: Frontend build directory '{frontend_path}' not found. UI will not be available.", flush=True)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
