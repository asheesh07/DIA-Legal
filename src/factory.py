"""
factory.py
──────────
Wiring file — shows exactly how to instantiate the A-RAG stack
and drop it into your existing app.py / Gradio interface.

Before A-RAG (your original wiring):
─────────────────────────────────────
    pipeline = DIAPipeline(
        retriever=retriever,
        context_builder=context_builder,
        llm_answerer=llm_answerer,
        query_router=query_router,
    )
    result = pipeline.run(query, case_id)

After A-RAG (new wiring, everything else unchanged):
─────────────────────────────────────────────────────
    pipeline = build_arag_pipeline(
        retriever=retriever,        # your existing Retriever
        context_builder=context_builder,
        llm_answerer=llm_answerer,
        query_router=query_router,
        llm_client=llm_client,
    )

    # Standard query — same as before
    result = pipeline.run(query, case_id)

    # Full A-RAG with parallel agents + critic loop
    result = pipeline.run_arag(query, case_id)

Install new dependency:
    pip install rank-bm25
"""

from src.keyword_retriever import KeywordRetriever
from src.critic_agent import CriticAgent
from src.arag_retriever import HierarchicalRetriever
from src.pipeline import DIAPipeline


def build_arag_pipeline(
    retriever,
    context_builder,
    llm_answerer,
    query_router,
    llm_client,
    confidence_threshold: float = 0.80,
    max_critic_rounds: int = 3,
    keyword_weight: float = 0.3,
    semantic_weight: float = 0.7,
    top_k: int = 9,
) -> DIAPipeline:
    """
    Factory function — builds the full A-RAG pipeline.

    Args:
        retriever:            Your existing Retriever instance.
        context_builder:      Your existing ContextBuilder instance.
        llm_answerer:         Your existing LLMAnswerer instance.
        query_router:         Your existing QueryRouter instance.
        llm_client:           Your existing LLMClient instance.
        confidence_threshold: Critic loop stops when score >= this.
        max_critic_rounds:    Safety valve for critic loop iterations.
        keyword_weight:       BM25 weight in score fusion (0–1).
        semantic_weight:      Semantic weight in score fusion (0–1).
        top_k:                Final retrieved items count.

    Returns:
        DIAPipeline — with .run() and .run_arag() methods.
    """

    keyword_retriever = KeywordRetriever()

    critic_agent = CriticAgent(
        llm_client=llm_client,
        confidence_threshold=confidence_threshold,
        max_rounds=max_critic_rounds,
    )

    hierarchical_retriever = HierarchicalRetriever(
        retriever=retriever,
        keyword_retriever=keyword_retriever,
        critic_agent=critic_agent,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight,
        top_k=top_k,
    )

    pipeline = DIAPipeline(
        retriever=hierarchical_retriever,
        context_builder=context_builder,
        llm_answerer=llm_answerer,
        query_router=query_router,
    )

    return pipeline


# ─── Usage example ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Your existing setup (unchanged) ───────────────────────────────────
    from sentence_transformers import SentenceTransformer
    from src.embedder import TextEmbedder, VisualEmbedder, MultiModalEmbedder
    from src.vectorstore import LanceDBVectorStore
    from src.retriever import Retriever
    from src.reranker import CrossEncoderReranker
    from src.context_builder import ContextBuilder
    from src.llmclient import LLMClient
    from src.llm_answerer import LLMAnswerer
    from src.query_router import QueryRouter

    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    text_embedder = TextEmbedder(model=text_model, batch_size=32, normalize=True)
    visual_embedder = VisualEmbedder(
        model_name="openai/clip-vit-base-patch32",
        device=None,
        normalize=True,
    )
    embedder = MultiModalEmbedder(text_embedder, visual_embedder)

    vector_store = LanceDBVectorStore(
        table_name="dia_legal",
        db_path="./lancedb",
        text_dim=384,
        visual_dim=512,
    )

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        enable_mmr=True,
    )

    llm_client = LLMClient()
    llm_answerer = LLMAnswerer(llm_client=llm_client)
    context_builder = ContextBuilder()
    query_router = QueryRouter()

    # ── 2. Build A-RAG pipeline (new) ────────────────────────────────────────
    pipeline = build_arag_pipeline(
        retriever=retriever,
        context_builder=context_builder,
        llm_answerer=llm_answerer,
        query_router=query_router,
        llm_client=llm_client,
    )

    # ── 3. Run standard query (unchanged interface) ───────────────────────────
    result = pipeline.run(
        query="Did the witness contradict the FIR statement?",
        case_id="case_001",
    )
    print("Answer:", result["answer"])

    # ── 4. Run full A-RAG with parallel agents + critic ───────────────────────
    result_arag = pipeline.run_arag(
        query="Did the witness contradict the FIR statement?",
        case_id="case_001",
    )
    print("Answer:", result_arag["answer"])
    print("Contradictions found:", len(result_arag["contradictions"]))
    for c in result_arag["contradictions"]:
        print(f"  [{c['confidence']:.2f}] {c['refined_conflict'] or c['proposed_conflict']}")