<div align="center">

# ⚖️ DIA-Legal

### AI-Powered Legal Intelligence Platform

*Upload case documents and videos → get instant answers, evidence maps, contradiction detection, and a full trial brief — all grounded in your actual files.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![LanceDB](https://img.shields.io/badge/LanceDB-vector_store-6D28D9?style=flat)](https://lancedb.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#features) · [Architecture](#architecture) · [Quick Start](#quick-start) · [API Reference](#api-reference) · [Tech Stack](#tech-stack)

</div>

---

## What is DIA-Legal?

Legal teams spend hours sifting through hundreds of pages of case documents and video depositions before they can prepare arguments or spot witness inconsistencies. **DIA-Legal** eliminates that bottleneck.

It ingests your PDFs (FIRs, court orders, witness statements) and video depositions (including YouTube links), chunks and embeds them into a multi-modal vector store, then exposes five AI-powered analysis modes through a clean web UI.

Every answer is grounded in your documents — no hallucination without a traceable citation.

---

## Features

| Mode | What it does |
|------|-------------|
| **💬 Query / Chat** | Ask natural-language questions about the case. Multi-turn memory, persistent chat sessions, source citations with exact timestamps and page numbers |
| **⚖️ Evidence Map** | Classifies every piece of evidence as Supporting / Opposing / Neutral relative to your stated legal position |
| **🔍 Contradiction Detection** | Cross-references all statements across sources; flags inconsistencies by severity with speaker attribution |
| **🥊 Devil's Advocate** | Multi-round AI opponent that attacks your legal arguments using case evidence — stress-tests before trial |
| **📄 Trial Brief** | Generates a full brief: case strength score, witness credibility profiles, key risks, recommended actions, opposition strategy — downloadable as PDF |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT (React + Vite)                       │
│  Query │ Evidence Map │ Contradictions │ Devil's Advocate │ Brief    │
└────────────────────────────┬────────────────────────────────────────┘
                             │ REST / JSON
┌────────────────────────────▼────────────────────────────────────────┐
│                        FastAPI Backend                               │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │  Ingestion   │  │  RAG Pipeline│  │   Analysis Engines        │ │
│  │  Pipeline    │  │              │  │                           │ │
│  │              │  │  QueryRouter │  │  EvidenceClassifier       │ │
│  │  ReaderRouter│  │      ↓       │  │  ContradictionDetector    │ │
│  │  VideoProc   │  │  Retriever   │  │  DevilsAdvocate           │ │
│  │  PDFChunker  │  │  (MMR+rerank)│  │  TrialBriefGenerator      │ │
│  │  Chunker     │  │      ↓       │  │                           │ │
│  └──────┬───────┘  │  ContextBld  │  └───────────────────────────┘ │
│         │          │      ↓       │                                 │
│  ┌──────▼───────┐  │  LLMAnswerer │                                 │
│  │  Embedder    │  └──────┬───────┘                                 │
│  │              │         │                                         │
│  │  TextEmbed   │  ┌──────▼───────────────────────────────────┐    │
│  │  all-MiniLM  │  │          LanceDB Vector Store             │    │
│  │              │  │  text_dim=384  ·  visual_dim=512 (CLIP)  │    │
│  │  VisualEmbed │  │  MMR diversity · CrossEncoder reranking  │    │
│  │  CLIP ViT-B  │  └──────────────────────────────────────────┘    │
│  └──────────────┘                                                   │
│                                                                     │
│  Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)                   │
│  WhisperX transcription · YT-DLP · OCR (pytesseract)               │
└─────────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline — Step by Step

```
User Query
    │
    ▼
QueryRouter          ← classifies: factual / comparative / timeline / entity
    │
    ▼
Retriever            ← dual embedding: text (MiniLM) + visual (CLIP)
    │ top-30 candidates
    ▼
CrossEncoderReranker ← ms-marco-MiniLM-L-6-v2, batch re-scores
    │ top-k (MMR diversified)
    ▼
ContextBuilder       ← assembles prompt with citations, token budget = 200
    │
    ▼
LLMAnswerer          ← HuggingFace Inference API (Llama-3.1-8B)
    │
    ▼
Response + Citations (source · page/timestamp · confidence)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- A HuggingFace API token with Inference API access (`HF_TOKEN`)

### 1. Clone & set up environment

```bash
git clone https://github.com/asheesh07/DIA-Legal.git
cd dia-legal
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

### 2. Backend

```bash
# Using Make (recommended)
make install
make dev

# Or manually
pip install -r requirements.txt
uvicorn main:app --port 8000 --reload
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev        # dev server at http://localhost:5173
# or: npm run build && cd .. && uvicorn main:app --port 8000
```

### 4. Docker (one command)

```bash
make docker-build
make docker-run
# App available at http://localhost:8000
```

### 5. Run the demo test suite

```bash
# With the server running on :8000
make test
# or:
python demo_full_test.py
```

---

## API Reference

All endpoints are also documented interactively at `http://localhost:8000/docs`.

### Health

```http
GET /api/health
→ { "status": "ok", "version": "2.0.0" }
```

### Cases

```http
GET    /api/cases              → list all cases
GET    /api/cases/{case_id}    → get case metadata
DELETE /api/cases/{case_id}    → delete case
```

### Ingestion

```http
POST /api/ingest/pdf           form: case_id, files[]    → chunks indexed per file
POST /api/ingest/youtube       json: { case_id, url }    → transcript + frames indexed
POST /api/ingest/video         form: case_id, file       → local video file
```

### Query & Chat

```http
POST /api/query
  { "case_id": "...", "query": "...", "history": [...] }
  → { "answer", "citations", "confidence", "mode" }

GET  /api/cases/{id}/chat/sessions
POST /api/cases/{id}/chat/sessions/{session_id}
GET  /api/cases/{id}/chat/sessions/{session_id}
DEL  /api/cases/{id}/chat/sessions/{session_id}
```

### Analysis

```http
POST /api/evidence-map
  { "case_id": "...", "lawyer_position": "prosecution" }
  → { supporting[], opposing[], neutral[], summary }

POST /api/contradictions
  { "case_id": "..." }
  → { contradictions[], summary }

POST /api/devils-advocate/session
  { "case_id": "...", "topic": "..." }
  → { session_id }

POST /api/devils-advocate/argue
  { "session_id": "...", "case_id": "...", "argument": "..." }
  → { critique, weaknesses[], opposition, strengthen, round_number }

POST /api/brief
  { "case_id": "...", "lawyer_position": "..." }
  → { case_strength, witness_profiles[], key_risks[], recommended_actions[], ... }

GET  /api/brief/download/{brief_id}   → PDF download
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | FastAPI + Uvicorn | Async REST API, lifespan management |
| **Embeddings** | `sentence-transformers` all-MiniLM-L6-v2 | 384-dim text embeddings |
| **Visual** | CLIP ViT-B/32 (HuggingFace) | 512-dim frame/image embeddings |
| **Vector Store** | LanceDB | On-disk multi-modal vector DB, no infra needed |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision re-scoring of top candidates |
| **LLM** | Llama 3.1 8B via HF Inference API | Answer generation, analysis |
| **Video** | WhisperX + FFmpeg + OpenCV | Transcription with speaker diarization + frame extraction |
| **PDF** | pdfplumber + ReportLab | Structured extraction + brief PDF generation |
| **Video download** | yt-dlp | YouTube / public video ingestion |
| **OCR** | pytesseract | Text extraction from video frames |
| **Frontend** | React 19 + Vite + Tailwind CSS | Single-page app |
| **UI Components** | shadcn/ui + Radix UI | Accessible, composable UI primitives |
| **Container** | Docker + Gunicorn | Production deployment |
| **Deployment** | Railway / HuggingFace Spaces | Cloud hosting |

---

## Project Structure

```
dia-legal/
├── main.py                    # FastAPI app, all routes
├── requirements.txt
├── Dockerfile
├── Makefile                   # make install / dev / test / build
├── .env.example
│
├── src/                       # Core ML pipeline
│   ├── pipeline.py            # DIAPipeline — orchestrates RAG
│   ├── ingestion.py           # IngestionPipeline (PDF + video)
│   ├── reader.py              # ReaderRouter — detects doc type
│   ├── video_processor.py     # WhisperX transcription + frame extraction
│   ├── chunker.py             # Semantic chunking (video + PDF)
│   ├── embedder.py            # MultiModalEmbedder (text + CLIP visual)
│   ├── vectorstore.py         # LanceDB wrapper
│   ├── retriever.py           # MMR-enabled retriever
│   ├── reranker.py            # CrossEncoder reranker
│   ├── context_builder.py     # Token-budget context assembly
│   ├── llmclient.py           # HuggingFace Inference API client
│   ├── llm_answerer.py        # Answer generation + confidence
│   ├── query_router.py        # Query type classification
│   ├── evidence_classifier.py # EvidenceClassifier → supporting/opposing/neutral
│   ├── contradiction_detector.py # Two-stage contradiction detection
│   ├── devils_advocate.py     # Multi-round adversarial argument
│   └── trial_brief_generator.py  # Full brief + witness profiles + PDF
│
├── frontend/                  # React + Vite SPA
│   └── src/
│       ├── App.jsx
│       ├── api.js             # All API calls (axios)
│       └── components/
│           ├── modes/         # QueryMode, EvidenceMode, ContradictionsMode,
│           │                  # DevilsAdvocateMode, BriefMode
│           ├── layout/        # AppShell, AppHeader, AppSidebar, ModeTabs
│           ├── ingest/        # NewCaseDialog, IngestDialog
│           └── shared/        # CitationBadge, ContradictionCard, EmptyState…
│
├── data/                      # Runtime data (gitignored)
│   ├── lancedb/               # Vector store
│   ├── cases_registry.json    # Case metadata
│   └── ingestion_cache.json   # MD5-based dedup cache
│
├── demo_full_test.py          # Full automated test suite (all endpoints)
└── eval_runner.py             # Offline RAG evaluation suite (RAGAS-style metrics)
```

---

## Evaluation

`eval_runner.py` runs a full offline evaluation suite against a synthetic legal case with ground-truth labels:

| Metric | Measures |
|--------|---------|
| Semantic Similarity | cosine sim between generated and expected answer |
| Context Recall | % of expected answer covered by retrieved chunks |
| Faithfulness | % of generated sentences supported by context |
| Answer Relevance | cosine sim between query and generated answer |
| Token F1 | SQuAD-style precision / recall / F1 |
| Hit Rate @ 1/3/5 | relevant chunk appears in top-K results |
| MRR | Mean Reciprocal Rank of first relevant chunk |
| NDCG @ 3/5 | Normalized Discounted Cumulative Gain |

```bash
python eval_runner.py
```

---

## Key Design Decisions

**Why LanceDB?** Zero-infra multi-modal vector store. Stores text (384d) and visual (512d) embeddings in the same table — no separate services needed.

**Why cross-encoder reranking?** Bi-encoder retrieval (cosine sim) is fast but imprecise. The cross-encoder re-reads the query + chunk together, achieving ~20% better precision at negligible latency cost for legal use-cases where citation accuracy matters.

**Why multi-modal embeddings?** Video depositions contain visual exhibits, document diagrams, and OCR text from displayed exhibits. CLIP embeddings capture visual semantics that pure transcript embeddings miss.

**Why MD5-based ingestion cache?** Re-ingesting the same document (e.g., after server restart) wastes 30-60s. The cache keyed on `(source, case_id)` makes repeated demos instantaneous.

---

## License

MIT — see [LICENSE](LICENSE)

---

<div align="center">
Built by <a href="https://github.com/asheeshdhamacharla">Asheesh Dhamacharla</a> · Fresher AI/ML Developer
</div>
