import os
from pathlib import Path
from transformers import AutoTokenizer

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

from sentence_transformers import SentenceTransformer
# ============================================================
# Config
# ============================================================

BASE_STORAGE = "data"
DB_PATH = "data/lancedb"
HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# Build System Components
# ============================================================

def _build_systems():

    # ------------------------------
    # Tokenizer (for chunking only)
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2"
    )

    # ------------------------------
    # Core Processing
    # ------------------------------
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

    # ------------------------------
    # Embedders
    # ------------------------------
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

    # ------------------------------
    # Vector Store
    # ------------------------------
    text_dim = text_embedder.embed_dims
    visual_dim = visual_embedder.embed_dim
    vector_store = LanceDBVectorStore(
        table_name="dia_legal",
        db_path=DB_PATH,
        text_dim=text_dim,
        visual_dim=visual_dim
    )

    # ------------------------------
    # Reranker
    # ------------------------------
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=32,
        normalize=True
    )

    # ------------------------------
    # Retriever
    # ------------------------------
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

    # ------------------------------
    # Context Builder
    # ------------------------------
    context_builder = ContextBuilder(
        max_tokens=2000,
        include_scores=True
    )

    # ------------------------------
    # LLM
    # ------------------------------
    llm_client = LLMClient(
        api_token=HF_TOKEN
    )

    answerer = LLMAnswerer(
        llm_client=llm_client,
        confidence_threshold=0.3,
        max_history=3
    )

    # ------------------------------
    # Pipeline
    # ------------------------------
    pipeline = DIAPipeline(
        retriever=retriever,
        context_builder=context_builder,
        llm_answerer=answerer,
        query_router=QueryRouter()
    )

    # ------------------------------
    # Ingestion Pipeline
    # ------------------------------
    ingestion_pipeline = IngestionPipeline(
        reader_router=ReaderRouter,
        video_processor=video_processor,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store
    )

    return pipeline, ingestion_pipeline


# ============================================================
# Main Execution
# ============================================================

def main():

    case_id = "Case_001"

    # ----------------------------------------
    # Build System
    # ----------------------------------------
    pipeline, ingestion_pipeline = _build_systems()

    # ----------------------------------------
    # Ingest Evidence (Only Once Per Case)
    # ----------------------------------------
    source = "https://www.youtube.com/shorts/-wkbQhkmGlc"

    print("\n[INFO] Starting ingestion...\n")

    ingestion_result = ingestion_pipeline.ingest(
        source=source,
        case_id=case_id,
        storage_path=BASE_STORAGE
    )

    print("[INFO] Ingestion Result:", ingestion_result)

    # ----------------------------------------
    # Query Loop
    # ----------------------------------------
    while True:

        query = input("\nEnter Query (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        result = pipeline.run(
            query=query,
            case_id=case_id
        )

        print("\n==============================")
        print("ANSWER:\n")
        print(result["answer"])
        print("\nCITATIONS:\n", result["citations"])
        print("\nCONFIDENCE:", result["confidence"])
        print("==============================\n")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
