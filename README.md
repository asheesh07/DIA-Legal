# ⚖️ Dia Legal

### Video Intelligence for Legal Proceedings

Turn any legal proceeding video into a queryable knowledge base.  
Ask questions. Get speaker-attributed answers with timestamp-accurate citations.

<!--**[Live Demo](#your-huggingface-url)--> · [GitHub](#your-github-url)**

---

![Dia Legal Demo](demo.gif)

---

## The Problem

Legal proceedings are recorded as video. Video is unsearchable.

Finding what a specific witness said, when they said it, and how it connects to other testimony requires manually scrubbing through hours of footage. This is slow, expensive, and error-prone.

Existing solutions either:

- Produce raw transcripts with no structure or speaker attribution
- Use keyword search that misses semantic meaning
- Lose the "who said what" context when processing multi-speaker conversations

Dia Legal solves this. Ingest any legal proceeding video. Ask any question in natural language. Get a precise answer — citing which speaker said what, and exactly when.

---

## Features

- **Speaker-attributed answers** — every response identifies which speaker made each statement
- **Timestamp citations** — precise time ranges `[MM:SS → MM:SS]` for every evidence block
- **Multimodal retrieval** — searches across both spoken transcript and visual frame content
- **Adaptive query routing** — classifies each query (speech, visual, temporal, evidence) and adjusts retrieval strategy accordingly
- **MMR diversity** — retrieved evidence blocks are diverse, not redundant chunks of the same content
- **Temporal expansion** — automatically includes surrounding conversational context for each retrieved block
- **Cross-encoder reranking** — retrieved candidates are reranked for precision before answering
- **Confidence scoring** — every answer includes a reliability estimate
- **Out-of-scope detection** — queries requiring visual information not in the transcript are handled cleanly

---

## Architecture

```
Video Input (YouTube URL or local file)
        │
        ▼
┌──────────────────┐
│     FFmpeg       │  Audio extraction
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│    WhisperX      │  Transcription
│                  │  Word-level timestamps
│                  │  Speaker diarization
└──────┬───────────┘
       │  transcript + speaker labels + timestamps
       ▼
┌──────────────────┐
│    Chunker       │  Temporal chunks (20s windows)
│                  │  5s overlap
│                  │  Max 512 tokens
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────────┐
│      MultiModal Embedder         │
│                                  │
│  TextEmbedder                    │
│  (all-MiniLM-L6-v2, 384 dims)   │
│                                  │
│  VisualEmbedder                  │
│  (CLIP ViT-B/32, 512 dims)      │
│  Frame extraction + OCR          │
└──────┬───────────────────────────┘
       │  text_embedding + visual_embedding
       ▼
┌──────────────────┐
│    LanceDB       │  Vector storage
│  (dia_legal      │  Separate text + visual columns
│   .lance)        │  Fast ANN search
└──────┬───────────┘
       │
       ▼  (at query time)
┌──────────────────────────────────┐
│          Retriever               │
│                                  │
│  1. Query classification         │
│     (prototype embeddings →      │
│      speech/visual/temporal/     │
│      evidence/semantic)          │
│                                  │
│  2. Adaptive alpha               │
│     (text/visual balance based   │
│      on query type)              │
│                                  │
│  3. Hybrid retrieval             │
│     (text + visual similarity)   │
│                                  │
│  4. Temporal expansion           │
│     (±5s context window)         │
│                                  │
│  5. MMR diversity                │
│     (remove redundant chunks)    │
│                                  │
│  6. Cross-encoder reranking      │
│     (ms-marco-MiniLM-L-6-v2)    │
└──────┬───────────────────────────┘
       │  ranked evidence blocks
       ▼
┌──────────────────┐
│  Context Builder │  Formats evidence for LLM
│                  │  Max 2000 tokens
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  LLM Answerer    │  Mistral-7B-Instruct
│                  │  via HuggingFace Inference API
└──────┬───────────┘
       │
       ▼
Answer + Citations + Confidence Score
```

---

## Technical Details

### Speaker Diarization

WhisperX provides word-level timestamps and speaker labels (SPEAKER_00, SPEAKER_01, etc.) for every segment. Speaker attribution is preserved through the entire pipeline — chunking, embedding, retrieval, and answer generation. The system never loses "who said what" context.

### Multimodal Embeddings

Two separate embedding spaces stored in LanceDB:

- **Text:** `all-MiniLM-L6-v2` (384 dimensions) — encodes transcript content
- **Visual:** `CLIP ViT-B/32` (512 dimensions) — encodes frame content and OCR text

At retrieval time, both spaces are searched independently and combined via adaptive alpha weighting.

### Query Classification

Each incoming query is classified into one of six types using prototype embeddings:

```
SPEECH   → "What did someone say?"
VISUAL   → "Is something visible in the video?"
TEMPORAL → "What happened at a specific time?"
OCR      → "Find text visible in a frame."
EVIDENCE → "Find supporting or contradicting evidence."
SEMANTIC → "Answer a general question about events."
```

Classification uses cosine similarity between the query embedding and prototype embeddings. The result determines the alpha value (text vs visual weighting) for retrieval.

### Adaptive Alpha

The text/visual retrieval balance adjusts based on query type:

- Speech queries → high text weight (α ≈ 0.95)
- Visual queries → balanced (α ≈ 0.60)
- OCR queries → lower text weight (α ≈ 0.65)

This means a question about what someone said prioritises transcript retrieval, while a question about what appeared on screen shifts toward visual retrieval.

### MMR (Maximal Marginal Relevance)

Retrieved candidates are filtered for diversity. If two chunks contain nearly identical content, only the higher-scoring one is kept. This prevents the LLM from receiving five redundant versions of the same testimony segment.

### Temporal Expansion

Each retrieved chunk is automatically expanded by ±5 seconds. This captures the conversational context surrounding the key moment — the question before an answer, the reaction after a statement.

### Confidence Estimation

Confidence combines three signals:

- Normalised top retrieval score
- Score gap between top-1 and top-2 results (a large gap = higher confidence)
- Coverage (how many strong results were found)

### Vector Store

LanceDB with cosine similarity metric. Two vector columns per record: `text_embedding` and `visual_embedding`. Filtered search by `case_id` and `time_range` for multi-case support.

---

## Get Started

### Prerequisites

- Python 3.9+
- `ffmpeg` installed on your system
- HuggingFace account (free) for LLM API access

### Installation

```bash
git clone https://github.com/[you]/dia-legal
cd dia-legal
pip install -r requirements.txt
```

### Set your HuggingFace token

```bash
export HF_TOKEN=your_huggingface_token_here
```

Or create a `.env` file:

```
HF_TOKEN=your_huggingface_token_here
```

### Run the demo

```bash
python app.py
```

This launches the Gradio interface at `http://localhost:7860`

---

## Usage

### Step 1 — Ingest a Video

Enter a YouTube URL and a Case ID in the **Ingest** tab. The pipeline will:

1. Download the video
2. Extract audio via FFmpeg
3. Transcribe with WhisperX (speaker diarization + word timestamps)
4. Chunk into temporal segments
5. Embed with SentenceTransformers + CLIP
6. Store in LanceDB

Ingestion takes 2-5 minutes depending on video length.

**Case_001 is pre-ingested in the live demo** — you can query immediately.

### Step 2 — Query

Switch to the **Query** tab. Enter your Case ID and ask any question.

**Example queries:**

```
"summary"
→ Full summary of the proceeding

"What did the driver say about the gearbox?"
→ Speaker-attributed answer with timestamps

"Who mentioned the Valencia race?"
→ Identifies the specific speaker and moment

"What was the engineers' reaction?"
→ Retrieves the relevant exchange

"What happened after the second corner?"
→ Temporal query with context expansion
```

### Output Format

```
ANSWER:
Based on the evidence blocks, [speaker-attributed answer]

CITATIONS:
[Block 1] 01:36 → 01:57
[Block 3] 01:19 → 01:35
[Block 6] 00:18 → 00:38

CONFIDENCE: 0.87
```

---

## Stack

| Component                   | Technology                           |
| --------------------------- | ------------------------------------ |
| Audio extraction            | FFmpeg                               |
| Transcription + diarization | WhisperX                             |
| Text embeddings             | all-MiniLM-L6-v2                     |
| Visual embeddings           | CLIP ViT-B/32                        |
| Vector store                | LanceDB                              |
| Reranker                    | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM                         | Mistral-7B-Instruct-v0.2             |
| Interface                   | Gradio                               |
| Deployment                  | HuggingFace Spaces                   |

---

## Limitations and Roadmap

**Current limitations:**

- Visual retrieval depends on OCR text in frames — video with minimal on-screen text relies primarily on transcript retrieval
- Speaker labels are anonymous (SPEAKER_00, SPEAKER_01) — speaker identification by name requires additional input
- Single video per ingestion session — batch ingestion not yet supported
- Ingestion time scales linearly with video length

**Roadmap:**

- [ ] Named speaker identification
- [ ] PDF and document ingestion alongside video
- [ ] Multi-case management dashboard
- [ ] Export citations to legal brief format
- [ ] Real-time streaming ingestion
- [ ] Fine-tuned legal domain LLM
- [ ] Speaker-attributed RAG benchmark dataset

---

## Research Context

Dia Legal addresses an open problem in multimodal RAG: **speaker attribution through the retrieval pipeline.**

Existing RAG systems chunk documents without preserving conversational structure. In multi-speaker proceedings, retrieved chunks lose the "who said what" context that is legally critical. Dia Legal proposes a chunking and retrieval strategy that maintains speaker attribution from ingestion through to the final generated answer.

This is an active research direction. The speaker-attributed retrieval mechanism, evaluation metrics for attribution faithfulness, and benchmark construction are areas of ongoing development.

> ⚠️ **Active Development** — Core pipeline
> is functional and deployed. Visual retrieval,
> named speaker identification, and multi-case
> dashboard are under active development.
> Contributions and feedback welcome.

---
