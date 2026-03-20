"""
keyword_retriever.py
────────────────────
BM25-based keyword retrieval layer for DIA-Legal A-RAG pipeline.

Sits BEFORE semantic search in the hierarchical retrieval stack:
    Layer 1 (this file) → exact lexical match for legal terms, names, dates
    Layer 2             → semantic vector search (existing LanceDB retriever)
    Layer 3             → adjacent chunk expansion (existing _expand_temporal)

No new dependencies beyond rank_bm25 (pip install rank-bm25).
"""

import re
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi


class KeywordRetriever:
    """
    Maintains an in-memory BM25 index over all chunks for a case.
    Rebuilt on demand when new chunks are ingested.

    Usage:
        kr = KeywordRetriever()
        kr.index(chunks)                        # build index
        hits = kr.search(query, top_k=10)       # returns list of chunk dicts
    """

    def __init__(self):
        self._corpus: List[Dict] = []       # raw chunk dicts
        self._tokenized: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._case_id: Optional[str] = None

    # ─── Public API ──────────────────────────────────────────────────────────

    def index(self, chunks: List[Dict]) -> None:
        """
        Build BM25 index from a list of chunk dicts.
        Each chunk must have at least one of:
            - transcript_text (str)
            - transcript_segments (list of dicts with 'text')
        """
        self._corpus = chunks
        self._tokenized = [
            self._tokenize(self._extract_text(c))
            for c in chunks
        ]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Return top_k chunks ranked by BM25 score.
        Adds 'bm25_score' key to each returned dict.
        Returns [] if index is empty or query yields no hits.
        """
        if self._bm25 is None or not self._corpus:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Pair with corpus, sort descending
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                break                       # BM25=0 means no keyword overlap
            chunk = dict(self._corpus[idx]) # shallow copy — don't mutate original
            chunk["bm25_score"] = float(score)
            chunk["score"] = float(score)   # uniform key for downstream merging
            results.append(chunk)

        return results

    def is_indexed(self, case_id: str) -> bool:
        return self._bm25 is not None and self._case_id == case_id

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _extract_text(self, chunk: Dict) -> str:
        """
        Pull all text out of a chunk dict in priority order.
        Works for both video chunks (transcript_segments) and
        PDF chunks (transcript_text / text).
        """
        parts = []

        # 1. Flat transcript text (PDF path)
        flat = chunk.get("transcript_text") or chunk.get("text", "")
        if flat:
            parts.append(flat)

        # 2. Structured transcript segments (video path)
        for seg in chunk.get("transcript_segments", []):
            text = seg.get("text", "")
            if text:
                parts.append(text)

        # 3. Frame captions and OCR
        for frame in chunk.get("frames", []):
            if frame.get("caption"):
                parts.append(frame["caption"])
            if frame.get("ocr_text"):
                parts.append(frame["ocr_text"])

        return " ".join(parts).strip()

    def _tokenize(self, text: str) -> List[str]:
        """
        Lowercase, strip punctuation, split on whitespace.
        Legal-aware: preserves section numbers like '302', 'IPC', 'CrPC'.
        """
        if not text:
            return []
        text = text.lower()
        # Keep alphanumeric and hyphens (for terms like 'cross-examination')
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        tokens = text.split()
        # Remove pure stopwords but keep legal abbreviations
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were",
            "in", "on", "at", "to", "of", "and", "or", "but",
            "with", "for", "this", "that", "it", "be", "by"
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]