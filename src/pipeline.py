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

    def run(self, query: str, case_id: str, history: List[Dict] = None, stats: Optional[Dict] = None) -> Dict:
        routing = self.query_router.route(query)
        mode = routing["mode"]

        if mode == "direct":
            return self._answer_direct(query, history or [])

        chunks_for_indexing = self.chunks_store.get(case_id)

        # HierarchicalRetriever.retrieve() signature is compatible
        # with legacy Retriever.retrieve() — just adds chunks_for_indexing
        if hasattr(self.retriever, "retrieve_parallel"):
            # A-RAG path
            retrieved_items = self.retriever.retrieve(
                case_id=case_id,
                query=query,
                chunks_for_indexing=chunks_for_indexing,
                top_k=4
            )
        else:
            # Legacy fallback
            retrieved_items = self.retriever.retrieve(
                case_id, query, top_k=4
            )

        original_retrieved = list(retrieved_items)
        
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

        context_result = self.context_builder.build(query, retrieved_items, mode, history=history)

        if stats:
            context_result["system_prompt"] += (
                f"\n\nCASE METADATA — USE THESE EXACT NUMBERS, DO NOT INVENT COUNTS: "
                f"This case has {stats['chunk_count']} indexed evidence chunks across "
                f"{stats['document_count']} documents ({stats['image_count']} images). "
                f"Never state any other chunk, document, or image counts."
            )

        progressive_context = self.llm_answerer.get_progressive_context()
        if progressive_context:
            context_result["context"] += "\n\n" + progressive_context

        result = self.llm_answerer.answer(context_result, retrieval_confidence, mode)
        result["trace"] = self._build_trace(original_retrieved, retrieved_items)
        
        return result

    # ─── Mode 2: Full A-RAG with parallel agents + critic ────────────────────

    def run_arag(self, query: str, case_id: str, history: List[Dict] = None) -> Dict:
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
        original_pool = video_items + doc_items
        all_items = self._deduplicate(original_pool)

        routing = self.query_router.route(query)
        mode = routing["mode"]

        context_result = self.context_builder.build(query, all_items, mode, history=history)

        progressive_context = self.llm_answerer.get_progressive_context()
        if progressive_context:
            context_result["context"] += "\n\n" + progressive_context

        all_items_raw = self._items_to_filter_format(all_items)
        retrieval_confidence = self.retriever._estimate_confidence(all_items_raw)

        result = self.llm_answerer.answer(context_result, retrieval_confidence, mode)
        result["contradictions"] = contradictions
        result["trace"] = self._build_trace(original_pool, all_items, critic_rounds=len(contradictions) if hasattr(contradictions, "__len__") else 0)

        return result

    # ─── Helpers ─────────────────────────────────────────────────────────────


    def _answer_direct(self, query: str, history: list) -> dict:
        history_ctx = ""
        for m in (history or [])[-4:]:
            role = "User" if m["role"] == "user" else "Assistant"
            history_ctx += f"{role}: {m['content']}
"

        context_package = {
            "system_prompt": (
                "You are an expert legal assistant with comprehensive knowledge of Indian law "
                "(IPC, CrPC, Indian Evidence Act, Constitution, and other statutes). "
                "Answer the following question accurately and concisely from your legal knowledge. "
                "This is a general legal question — do not reference any case files."
            ),
            "context": history_ctx.strip(),
            "user_prompt": (
                f"Question: {query}

"
                "Respond with JSON only:
"
                '{"answer": "your detailed answer", "supporting_evidence": [], "confidence": 0.9}'
            ),
            "citation_map": {},
        }
        result = self.llm_answerer.answer(context_package, retrieval_confidence=1.0, mode="direct")
        result["citations"] = []
        return result

    def _build_trace(self, all_candidates, final_items, critic_rounds=0) -> Dict:
        """
        Build a pipeline execution trace for observability.
        """
        trace = {
            "retrieved_count": len(all_candidates),
            "final_count": len(final_items),
            "critic_rounds": critic_rounds,
            "items_trace": []
        }
        
        final_ids = {item.chunk_ids[0] if getattr(item, 'chunk_ids', None) else id(item) for item in final_items}
        
        for item in all_candidates:
            cid = item.chunk_ids[0] if getattr(item, 'chunk_ids', None) else id(item)
            is_filtered = cid not in final_ids
            
            trace["items_trace"].append({
                "chunk_id": cid,
                "retrieval_layer": getattr(item, "retrieval_stage", "semantic"),
                "retrieval_score": getattr(item, "retrieval_score", 0.0),
                "reranker_score": getattr(item, "rerank_score", 0.0),
                "final_score": getattr(item, "final_score", 0.0),
                "status": "filtered" if is_filtered else "included"
            })
            
        return trace

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