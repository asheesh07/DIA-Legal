# app.py

import os

from dotenv import load_dotenv
import gradio as gr
from pathlib import Path
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from src.reader import ReaderRouter
from src.video_processor import VideoProcessor
from src.chunker import Chunker
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


load_dotenv()
BASE_STORAGE = "data"
DB_PATH = "data/lancedb"
HF_TOKEN = os.getenv("HF_TOKEN")

def _build_systems():
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2"
    )
    video_processor = VideoProcessor(
        base_output_path=BASE_STORAGE,
        model_size="base"
    )
    chunker = Chunker(
        max_duration=20,
        max_tokens=512,
        overlap_duration=5,
        tokenizer=tokenizer
    )
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    text_embedder = TextEmbedder(
        model=text_model,
        batch_size=12,
        normalize=True
    )
    visual_embedder = VisualEmbedder(
        model_name="openai/clip-vit-base-patch32",
        device="cpu",
        normalize=True
    )
    embedder = MultiModalEmbedder(
        text_embedder=text_embedder,
        visual_embedder=visual_embedder,
        visual_aggregation="mean"
    )
    text_dim = text_embedder.embed_dims
    visual_dim = visual_embedder.embed_dim
    vector_store = LanceDBVectorStore(
        table_name="dia_legal",
        db_path=DB_PATH,
        text_dim=text_dim,
        visual_dim=visual_dim
    )
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=32,
        normalize=True
    )
    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder,
        max_candidates=30,
        reranker=reranker,
        enable_mmr=True,
        mmr_lambda=0.5,
        min_threshold=0.0,
        temporary_window=5
    )
    context_builder = ContextBuilder(
        max_tokens=2000,
        include_scores=True
    )
    llm_client = LLMClient(
        api_token=os.environ.get("HF_TOKEN")
    )
    answerer = LLMAnswerer(
        llm_client=llm_client,
        confidence_threshold=0.3,
        max_history=3
    )
    pipeline = DIAPipeline(
        retriever=retriever,
        context_builder=context_builder,
        llm_answerer=answerer,
        query_router=QueryRouter()
    )
    ingestion_pipeline = IngestionPipeline(
        reader_router=ReaderRouter,
        video_processor=video_processor,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store
    )
    return pipeline, ingestion_pipeline


print("[INFO] Building system...")
pipeline, ingestion_pipeline = _build_systems()
print("[INFO] System ready.")

# ============================================================
# Ingestion Function
# ============================================================

def ingest_video(youtube_url, case_id):
    if not youtube_url.strip():
        return "Please enter a YouTube URL."
    if not case_id.strip():
        return "Please enter a Case ID."
    try:
        result = ingestion_pipeline.ingest(
            source=youtube_url.strip(),
            case_id=case_id.strip(),
            storage_path=BASE_STORAGE
        )
        return f"✅ Ingestion complete. {result}"
    except Exception as e:
        return f"❌ Ingestion failed: {str(e)}"

# ============================================================
# Query Function
# ============================================================

def answer_query(case_id, query):
    if not query.strip():
        return "Please enter a question.", ""
    if not case_id.strip():
        return "Please enter a Case ID.", ""
    try:
        result = pipeline.run(
            query=query.strip(),
            case_id=case_id.strip()
        )

        answer = result.get("answer", "No answer returned.")
        citations = result.get("citations", [])
        confidence = result.get("confidence", 0.0)

        # Format citations
        citation_text = ""
        for c in citations:
            block_id = c.get("block_id", "?")
            time_range = c.get("time_range", [0, 0])
            start = time_range[0]
            end = time_range[1]

            mins_s = int(start // 60)
            secs_s = int(start % 60)
            mins_e = int(end // 60)
            secs_e = int(end % 60)

            citation_text += (
                f"[Block {block_id}] "
                f"{mins_s:02d}:{secs_s:02d} → "
                f"{mins_e:02d}:{secs_e:02d}\n"
            )

        answer_with_confidence = (
            f"{answer}\n\n"
            f"─────────────────\n"
            f"Confidence: {confidence:.2f}"
        )

        return answer_with_confidence, citation_text

    except Exception as e:
        return f"❌ Error: {str(e)}", ""

# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(title="Dia Legal") as demo:

    gr.Markdown("""
    # ⚖️ Dia Legal
    ### Video Intelligence for Legal Proceedings
    Turn any legal video into a queryable knowledge base.
    Speaker-attributed answers with precise timestamp citations.
    ---
    """)

    # ── Tab 1: Ingest ──────────────────────────────────────
    with gr.Tab("📥 Ingest Video"):
        gr.Markdown("### Step 1 — Add a video to the system")

        with gr.Row():
            youtube_input = gr.Textbox(
                label="YouTube URL",
                placeholder="https://www.youtube.com/watch?v=..."
            )
            case_id_ingest = gr.Textbox(
                label="Case ID",
                placeholder="Case_001",
                value="Case_001"
            )

        ingest_btn = gr.Button("Ingest Video", variant="primary")
        ingest_status = gr.Textbox(label="Status", interactive=False)

        ingest_btn.click(
            fn=ingest_video,
            inputs=[youtube_input, case_id_ingest],
            outputs=ingest_status
        )

    # ── Tab 2: Query ───────────────────────────────────────
    with gr.Tab("🔍 Query"):
        gr.Markdown("### Step 2 — Ask questions about the proceeding")

        with gr.Row():
            case_id_query = gr.Textbox(
                label="Case ID",
                placeholder="Case_001",
                value="Case_001"
            )
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="What did the driver say about the gearbox?"
            )

        query_btn = gr.Button("Get Answer", variant="primary")

        with gr.Row():
            answer_output = gr.Textbox(
                label="Answer",
                lines=8,
                interactive=False
            )
            citations_output = gr.Textbox(
                label="Citations with Timestamps",
                lines=8,
                interactive=False
            )

        gr.Examples(
            examples=[
                ["Case_001", "summary"],
                ["Case_001", "What did the driver say about the gearbox?"],
                ["Case_001", "Who mentioned the Valencia race?"],
                ["Case_001", "What was the engineers reaction?"],
                ["Case_001", "How many laps did the driver complete?"],
            ],
            inputs=[case_id_query, query_input]
        )

        query_btn.click(
            fn=answer_query,
            inputs=[case_id_query, query_input],
            outputs=[answer_output, citations_output]
        )

    # ── Footer ─────────────────────────────────────────────
    gr.Markdown("""
    ---
    Built with WhisperX · LanceDB · CLIP · SentenceTransformers · Gradio
    """)

demo.launch()