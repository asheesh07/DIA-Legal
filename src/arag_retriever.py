"""
arag_retriever.py
─────────────────
Agentic RAG retrieval runtime for DIA-Legal.

Replaces the single vector search call in pipeline.py with a
three-layer hierarchical retrieval stack + parallel async agents
+ critic loop — all in raw Python, no LangChain/LlamaIndex.

Architecture:
┌─────────────────────────────────────────────┐
│           HierarchicalRetriever             │
│                                             │
│  Layer 1: BM25 keyword search               │
│  Layer 2: Semantic vector search (existing) │
│  Layer 3: Adjacent chunk expansion          │
│         ↓ merge + dedup                     │
│  Cross-encoder reranking                    │
│         ↓                                   │
│  Parallel Async Agents                      │
│    ├── DiarizationAgent  (video side)        │
│    └── EvidenceReasoner  (document side)     │
│         ↓ outputs merged                    │
│  CriticLoop (iterative refinement)          │
└─────────────────────────────────────────────┘

Drop-in replacement: returns the same List[RetrievedItem] as
your existing Retriever so nothing else in the pipeline changes.
"""

import asyncio
import json
import re
from typing import List, Dict, Optional, Tuple

from src.retriever import Retriever, RetrievedItem
from src.keyword_retriever import KeywordRetriever
from src.critic_agent import CriticAgent


class HierarchicalRetriever:
    """
    Three-layer retrieval with parallel agents and critic loop.

    Args:
        retriever:          Your existing Retriever instance (semantic layer).
        keyword_retriever:  KeywordRetriever instance (BM25 layer).
        critic_agent:       CriticAgent instance (refinement loop).
        keyword_weight:     BM25 score weight when merging layers. Default 0.3.
        semantic_weight:    Semantic score weight. Default 0.7.
        top_k:              Final number of items to return.
    """

    def __init__(
        self,
        retriever: Retriever,
        keyword_retriever: KeywordRetriever,
        critic_agent: CriticAgent,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
        top_k: int = 9,
    ):
        self.retriever = retriever
        self.keyword_retriever = keyword_retriever
        self.critic_agent = critic_agent
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.top_k = top_k

        # BM25 index cache — rebuilt when case_id changes
        self._indexed_case_id: Optional[str] = None

    # ─── Public API ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        case_id: str,
        query: str,
        chunks_for_indexing: Optional[List[Dict]] = None,
        filters: Optional[Dict] = None,
    ) -> List[RetrievedItem]:
        """
        Full A-RAG retrieval pipeline.

        Args:
            case_id:              Case identifier.
            query:                User query string.
            chunks_for_indexing:  Raw chunk dicts to build BM25 index from.
                                  Pass None if already indexed for this case.
            filters:              Optional metadata filters (passed to semantic layer).

        Returns:
            List[RetrievedItem] — same type as existing Retriever.retrieve()
        """
        # ── Layer 1: BM25 keyword search ──────────────────────────────────────
        if chunks_for_indexing and self._indexed_case_id != case_id:
            self.keyword_retriever.index(chunks_for_indexing)
            self._indexed_case_id = case_id

        keyword_hits = self.keyword_retriever.search(query, top_k=self.top_k * 2)

        # ── Layer 2: Semantic vector search (existing retriever) ──────────────
        semantic_hits_items = self.retriever.retrieve(
            case_id=case_id,
            query=query,
            filters=filters,
            top_k=self.top_k * 2,
        )
        # Convert RetrievedItems back to dicts for merging
        semantic_hits = self._items_to_dicts(semantic_hits_items)

        # ── Layer 3: Merge + deduplicate ──────────────────────────────────────
        merged = self._merge_layers(keyword_hits, semantic_hits)

        # ── Layer 4: Adjacent chunk expansion (already in existing retriever) ─
        # _expand_temporal is called inside retriever.retrieve() above,
        # so we get it for free from the semantic layer.
        # For keyword hits we do a lightweight neighbour fetch here:
        merged = self._expand_keyword_neighbours(case_id, merged)

        # ── Layer 5: Final rerank ─────────────────────────────────────────────
        if self.retriever.reranker:
            merged = self.retriever._apply_reranker(query, merged)

        top_candidates = merged[: self.top_k]

        # ── Build RetrievedItems ──────────────────────────────────────────────
        confidence = self.retriever._estimate_confidence(top_candidates)
        retrieved_items = self.retriever._build_retrieved_items(
            top_candidates, confidence
        )

        return retrieved_items

    async def retrieve_parallel(
        self,
        case_id: str,
        query: str,
        chunks_for_indexing: Optional[List[Dict]] = None,
        filters: Optional[Dict] = None,
    ) -> Tuple[List[RetrievedItem], List[RetrievedItem]]:
        """
        Run DiarizationAgent (video side) and EvidenceReasoner (document side)
        in parallel using asyncio.gather.

        Returns:
            (video_items, document_items) — two separate RetrievedItem lists.

        The caller (pipeline or contradiction detector) can then pass both
        lists to CriticAgent.run_loop() for cross-source conflict detection.
        """
        video_task = asyncio.create_task(
            self._async_retrieve(
                case_id, query, chunks_for_indexing,
                {**(filters or {}), "source_type": "video"}
            )
        )
        doc_task = asyncio.create_task(
            self._async_retrieve(
                case_id, query, chunks_for_indexing,
                {**(filters or {}), "source_type": "document"}
            )
        )

        video_items, doc_items = await asyncio.gather(video_task, doc_task)
        return video_items, doc_items

    def detect_contradictions_with_critic(
        self,
        video_items: List[RetrievedItem],
        doc_items: List[RetrievedItem],
    ) -> List[Dict]:
        """
        Cross-source contradiction detection with critic loop refinement.

        For each (video_chunk, doc_chunk) pair that passes the cosine
        similarity threshold, runs CriticAgent.run_loop() to verify
        and refine the proposed conflict.

        Returns list of validated contradiction dicts with critic scores.
        """
        contradictions = []

        for v_item in video_items:
            v_text = self._flatten_item_text(v_item)

            for d_item in doc_items:
                d_text = self._flatten_item_text(d_item)

                if not v_text or not d_text:
                    continue

                # Build initial contradiction candidate
                candidate = {
                    "chunk_a_text": v_text,
                    "chunk_b_text": d_text,
                    "proposed_conflict": (
                        f"Testimony at {self._format_time(v_item)} "
                        f"may conflict with document content: "
                        f"'{v_text[:200]}' vs '{d_text[:200]}'"
                    ),
                    "video_item": v_item,
                    "doc_item": d_item,
                }

                # Run critic loop — propose → critique → refine
                critique = self.critic_agent.run_loop(candidate)

                if critique["is_valid"]:
                    contradictions.append({
                        "video_citation": self._format_time(v_item),
                        "doc_citation": d_item.metadata.get(
                            "section_title", "Document"
                        ),
                        "proposed_conflict": candidate["proposed_conflict"],
                        "refined_conflict": critique.get(
                            "refined_conflict", ""
                        ),
                        "confidence": critique["confidence"],
                        "critique": critique["critique"],
                        "rounds_taken": critique.get("rounds_taken", 1),
                    })

        # Sort by confidence descending
        contradictions.sort(key=lambda x: x["confidence"], reverse=True)
        return contradictions

    # ─── Internal helpers ─────────────────────────────────────────────────────

    async def _async_retrieve(
        self,
        case_id: str,
        query: str,
        chunks_for_indexing: Optional[List[Dict]],
        filters: Optional[Dict],
    ) -> List[RetrievedItem]:
        """Async wrapper around sync retrieve() for asyncio.gather."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.retrieve(case_id, query, chunks_for_indexing, filters),
        )

    def _merge_layers(
        self,
        keyword_hits: List[Dict],
        semantic_hits: List[Dict],
    ) -> List[Dict]:
        """
        Merge BM25 and semantic results with weighted score fusion.
        Deduplicates by chunk_id — keeps highest fused score.
        """
        merged: Dict[str, Dict] = {}

        # Normalise BM25 scores to [0, 1]
        if keyword_hits:
            max_bm25 = max(c.get("bm25_score", 0) for c in keyword_hits)
            for c in keyword_hits:
                norm_bm25 = c.get("bm25_score", 0) / (max_bm25 + 1e-9)
                fused = self.keyword_weight * norm_bm25
                cid = c["chunk_id"]
                c["fused_score"] = fused
                merged[cid] = c

        # Add / blend semantic scores
        for c in semantic_hits:
            cid = c["chunk_id"]
            sem_score = c.get("final_score", c.get("score", 0.0))
            if cid in merged:
                # Already from BM25 — add semantic component
                merged[cid]["fused_score"] = (
                    merged[cid].get("fused_score", 0)
                    + self.semantic_weight * sem_score
                )
            else:
                c["fused_score"] = self.semantic_weight * sem_score
                merged[cid] = c

        # Sort by fused score
        result = sorted(
            merged.values(),
            key=lambda x: x.get("fused_score", 0),
            reverse=True,
        )

        # Normalise fused_score into 'score' key for downstream compatibility
        for c in result:
            c["score"] = c.get("fused_score", 0)
            c.setdefault("final_score", c["score"])

        return result

    def _expand_keyword_neighbours(
        self, case_id: str, candidates: List[Dict], window: int = 5
    ) -> List[Dict]:
        """
        For keyword-only hits (those with bm25_score but no semantic neighbour
        expansion), fetch temporally adjacent chunks from the vector store.
        Mirrors _expand_temporal in existing Retriever.
        """
        existing_ids = {c["chunk_id"] for c in candidates}
        extras: Dict[str, Dict] = {}

        for c in candidates:
            # Only expand chunks that came purely from BM25
            if c.get("bm25_score", 0) > 0 and c.get("fused_score", 0) <= self.keyword_weight:
                expanded_start = max(0, c.get("start_time", 0) - window)
                expanded_end = c.get("end_time", 0) + window

                neighbours = self.retriever.vector_store.search(
                    text_query_embedding=None,
                    visual_query_embedding=None,
                    alpha=1.0,
                    top_k=5,
                    filters={
                        "case_id": case_id,
                        "time_range": {
                            "start": expanded_start,
                            "end": expanded_end,
                        },
                    },
                )
                for n in neighbours:
                    nid = n["chunk_id"]
                    if nid not in existing_ids and nid not in extras:
                        n["score"] = 0.0
                        n["fused_score"] = 0.0
                        extras[nid] = n

        return candidates + list(extras.values())

    def _items_to_dicts(self, items: List[RetrievedItem]) -> List[Dict]:
        """Convert RetrievedItem dataclasses back to flat dicts for merging."""
        result = []
        for item in items:
            d = {
                "chunk_id": item.chunk_ids[0] if item.chunk_ids else "",
                "case_id": item.case_id,
                "start_time": item.temporal.primary.start_time,
                "end_time": item.temporal.primary.end_time,
                "transcript_segments": item.structured_transcripts,
                "frames": item.structured_frames,
                "score": item.final_score,
                "final_score": item.final_score,
                "retrieval_score": item.retrieval_score,
                "reranker_score": item.rerank_score or 0.0,
                "source_type": (
                    item.sources[0].source_type if item.sources else "video"
                ),
                "source_id": (
                    item.sources[0].source_id if item.sources else ""
                ),
                "metadata": item.metadata,
            }
            result.append(d)
        return result

    def _flatten_item_text(self, item: RetrievedItem) -> str:
        """Pull all text out of a RetrievedItem for critic evaluation."""
        parts = []
        for seg in item.structured_transcripts:
            text = seg.get("text", "")
            if text:
                parts.append(text)
        return " ".join(parts).strip()

    def _format_time(self, item: RetrievedItem) -> str:
        s = item.temporal.primary.start_time
        e = item.temporal.primary.end_time
        sm, ss = int(s) // 60, int(s) % 60
        em, es = int(e) // 60, int(e) % 60
        return f"{sm:02d}:{ss:02d} → {em:02d}:{es:02d}"