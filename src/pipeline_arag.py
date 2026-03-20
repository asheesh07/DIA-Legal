"""
pipeline.py  (A-RAG upgrade)
────────────────────────────
Drop-in replacement for the original DIAPipeline.

Two modes:
    1. run()          — standard query answering (same interface as before)
    2. run_arag()     — full A-RAG with parallel agents + critic loop,
                        returns validated contradictions + answer

Nothing in app.py, context_builder.py, llm_answerer.py, or
contradiction_detector.py needs to change.
"""

import asyncio
from typing import Optional, Dict, List


class DIAPipeline:
    def __init__(
        self,
        retriever,                  # HierarchicalRetriever (or legacy Retriever)
        context_builder,
        llm_answerer,
        query_router,
        chunks_store: Optional[Dict[str, List[Dict]]] = None,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm_answerer = llm_answerer
        self.query_router = query_router
        # chunks_store: {case_id: [chunk_dicts]} for BM25 indexing
        self.chunks_store = chunks_store or {}

    # ─── Mode 1: Standard query (same interface as original) ─────────────────

    def run(self, query: str, case_id: str) -> Dict:
        routing = self.query_router.route(query)
        mode = routing["mode"]

        chunks_for_indexing = self.chunks_store.get(case_id)

        # HierarchicalRetriever.retrieve() signature is compatible
        # with legacy Retriever.retrieve() — just adds chunks_for_indexing
        if hasattr(self.retriever, "retrieve_parallel"):
            # A-RAG path
            retrieved_items = self.retriever.retrieve(
                case_id=case_id,
                query=query,
                chunks_for_indexing=chunks_for_indexing,
            )
        else:
            # Legacy fallback
            retrieved_items = self.retriever.retrieve(
                case_id, query, top_k=9
            )

        # Lightweight LLM filter (unchanged)
        retrieved_items_raw = self._items_to_filter_format(retrieved_items)
        filtered = self.llm_answerer.llm_client.filter_chunks(
            query, retrieved_items_raw
        )
        # filter_chunks returns chunk dicts — rebuild RetrievedItems
        if filtered and isinstance(filtered[0], dict):
            retrieved_items = self.retriever._build_retrieved_items(
                filtered,
                self.retriever._estimate_confidence(filtered)
            ) if hasattr(self.retriever, "_build_retrieved_items") else retrieved_items

        retrieval_confidence = self.retriever._estimate_confidence(
            self._items_to_filter_format(retrieved_items)
        ) if hasattr(self.retriever, "_estimate_confidence") else 0.7

        context_result = self.context_builder.build(query, retrieved_items, mode)

        progressive_context = self.llm_answerer.get_progressive_context()
        if progressive_context:
            context_result["context"] += "\n\n" + progressive_context

        result = self.llm_answerer.answer(context_result, retrieval_confidence, mode)
        return result

    # ─── Mode 2: Full A-RAG with parallel agents + critic ────────────────────

    def run_arag(self, query: str, case_id: str) -> Dict:
        """
        Full A-RAG pipeline:
        1. Parallel retrieval — DiarizationAgent (video) + EvidenceReasoner (doc)
        2. Critic loop — cross-source contradiction detection with refinement
        3. Standard answer generation on merged results

        Returns:
            {
                "answer":          str,
                "citations":       list,
                "confidence":      float,
                "contradictions":  list of validated contradiction dicts
            }
        """
        if not hasattr(self.retriever, "retrieve_parallel"):
            # Fallback to standard run if not using HierarchicalRetriever
            return self.run(query, case_id)

        chunks_for_indexing = self.chunks_store.get(case_id)

        # ── Step 1: Parallel agents ───────────────────────────────────────────
        video_items, doc_items = asyncio.run(
            self.retriever.retrieve_parallel(
                case_id=case_id,
                query=query,
                chunks_for_indexing=chunks_for_indexing,
            )
        )

        # ── Step 2: Critic loop contradiction detection ───────────────────────
        contradictions = self.retriever.detect_contradictions_with_critic(
            video_items, doc_items
        )

        # ── Step 3: Merge all items for answer generation ─────────────────────
        all_items = self._deduplicate(video_items + doc_items)

        routing = self.query_router.route(query)
        mode = routing["mode"]

        context_result = self.context_builder.build(query, all_items, mode)

        progressive_context = self.llm_answerer.get_progressive_context()
        if progressive_context:
            context_result["context"] += "\n\n" + progressive_context

        all_items_raw = self._items_to_filter_format(all_items)
        retrieval_confidence = self.retriever._estimate_confidence(all_items_raw)

        result = self.llm_answerer.answer(context_result, retrieval_confidence, mode)
        result["contradictions"] = contradictions

        return result

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _items_to_filter_format(self, items) -> list:
        """
        filter_chunks() expects objects with .structured_transcripts attribute.
        RetrievedItems already have this. If items are dicts, wrap minimally.
        """
        return items  # RetrievedItems work directly with filter_chunks

    def _deduplicate(self, items) -> list:
        seen = set()
        result = []
        for item in items:
            key = item.chunk_ids[0] if item.chunk_ids else id(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result