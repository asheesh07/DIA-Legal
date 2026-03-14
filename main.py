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

from dotenv import load_dotenv
load_dotenv()
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




cat > create_test_docs.py << 'EOF'
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

styles = getSampleStyleSheet()

doc = SimpleDocTemplate("test_fir.pdf", pagesize=letter)
story = []
story.append(Paragraph("FIRST INFORMATION REPORT", styles["Title"]))
story.append(Paragraph("FIR No: 2024/001", styles["Normal"]))
story.append(Spacer(1, 12))
story.append(Paragraph("1. BACKGROUND", styles["Heading1"]))
story.append(Paragraph(
    "On the date of the Valencia Grand Prix the defendant driver "
    "Kevin Magnussen was operating vehicle No. 20. Station House "
    "Officer received complaint at 14:32 local time.", styles["Normal"]))
story.append(Spacer(1, 12))
story.append(Paragraph("2. STATEMENT OF FACTS", styles["Heading1"]))
story.append(Paragraph(
    "The gearbox of vehicle No. 20 showed signs of prior damage "
    "before the race commenced. Engineering telemetry confirmed "
    "anomalous gear shift patterns from lap 3 onwards. The driver "
    "was informed via radio communication at lap 5 that the gearbox "
    "was operating outside normal parameters.", styles["Normal"]))
story.append(Spacer(1, 12))
story.append(Paragraph("3. ALLEGATIONS", styles["Heading1"]))
story.append(Paragraph(
    "It is alleged that the defendant had prior knowledge of the "
    "mechanical fault and continued racing. The defendant denies "
    "all knowledge of the fault prior to lap 12 when the gearbox "
    "failed completely.", styles["Normal"]))
doc.build(story)
print("Created test_fir.pdf")

doc2 = SimpleDocTemplate("test_witness_statement.pdf", pagesize=letter)
story2 = []
story2.append(Paragraph("WITNESS STATEMENT", styles["Title"]))
story2.append(Paragraph("Witness: Guenther Steiner, Team Principal", styles["Normal"]))
story2.append(Spacer(1, 12))
story2.append(Paragraph("EXAMINATION", styles["Heading1"]))
story2.append(Paragraph(
    "I hereby state that on the morning of the race I personally "
    "reviewed the pre-race engineering report. The gearbox was "
    "declared fit for competition by our chief engineer. "
    "I was not aware of any anomaly at the time of race start.",
    styles["Normal"]))
story2.append(Spacer(1, 12))
story2.append(Paragraph("CROSS EXAMINATION", styles["Heading1"]))
story2.append(Paragraph(
    "When pressed on the telemetry data I confirmed that the "
    "engineering team did flag a minor irregularity on lap 3 "
    "but we assessed it as within acceptable tolerance. "
    "The driver was not informed because we did not consider "
    "it a safety risk at that point.", styles["Normal"]))
story2.append(Spacer(1, 12))
story2.append(Paragraph("RE-EXAMINATION", styles["Heading1"]))
story2.append(Paragraph(
    "I stand by my earlier statement. The driver had no knowledge "
    "of the gearbox irregularity until the failure on lap 12. "
    "The decision not to inform the driver was made collectively "
    "by the engineering team.", styles["Normal"]))
doc2.build(story2)
print("Created test_witness_statement.pdf")

doc3 = SimpleDocTemplate("test_court_order.pdf", pagesize=letter)
story3 = []
story3.append(Paragraph("FIA INTERNATIONAL TRIBUNAL", styles["Title"]))
story3.append(Paragraph("COURT ORDER — Case 2024/F1/001", styles["Normal"]))
story3.append(Spacer(1, 12))
story3.append(Paragraph("WHEREAS", styles["Heading1"]))
story3.append(Paragraph(
    "The Tribunal has reviewed all submitted evidence including "
    "telemetry data witness statements and video footage of "
    "the Valencia Grand Prix proceedings.", styles["Normal"]))
story3.append(Spacer(1, 12))
story3.append(Paragraph("THEREFORE", styles["Heading1"]))
story3.append(Paragraph(
    "It is hereby ordered that the defendant team shall produce "
    "all engineering logs and radio communications from lap 1 "
    "through lap 12 of the Valencia Grand Prix.", styles["Normal"]))
story3.append(Spacer(1, 12))
story3.append(Paragraph("ORDER", styles["Heading1"]))
story3.append(Paragraph(
    "The Tribunal finds sufficient evidence to proceed to a full "
    "hearing. The burden of proof rests with the defense to "
    "demonstrate the driver had no prior knowledge of the fault.",
    styles["Normal"]))
doc3.build(story3)
print("Created test_court_order.pdf")
print("\nAll test documents ready.")
EOF
