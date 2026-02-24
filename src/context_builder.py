from typing import List, Dict, Tuple
from enum import Enum
import textwrap
import re


class LegalMode(Enum):
    EVIDENCE = "evidence"
    ASSISTANT = "assistant"
    OPPOSITION = "opposition"


class QueryType(Enum):
    EXISTENCE = "existence"
    TEMPORAL = "temporal"
    ATTRIBUTION = "attribution"
    CONTRADICTION = "contradiction"
    DESCRIPTIVE = "descriptive"
    OTHER = "other"


class ContextBuilder:
    """
    Enterprise-grade, query-aware, sufficiency-detecting
    multimodal legal context orchestrator.

    Responsibilities:
    - Query classification
    - Evidence type tagging
    - Evidence prioritization
    - Sufficiency detection
    - Context pruning
    - Structured assembly
    - Mode-aware policy injection
    """

    def __init__(
        self,
        max_tokens: int = 3500,
        include_scores: bool = False,
        mmr_lambda: float = 0.7
    ):
        self.max_tokens = max_tokens
        self.include_scores = include_scores
        self.mmr_lambda = mmr_lambda

    # ============================================================
    # PUBLIC ENTRY
    # ============================================================
    def build(
        self,
        query: str,
        retrieved_items: List,
        mode: LegalMode
    ) -> Dict:

        query_type = self._classify_query(query)

        enriched_items = [
            self._annotate_evidence(item)
            for item in retrieved_items
        ]

        ranked_items = self._prioritize(enriched_items, query_type)

        decisive_blocks = self._detect_sufficiency(
            ranked_items,
            query,
            query_type
        )

        pruned_items = (
            decisive_blocks
            if decisive_blocks
            else ranked_items
        )

        structured_context,citation_map = self._assemble_context(pruned_items)

        return {
            "system_prompt": self._build_system_prompt(mode, query_type),
            "context": self._truncate(structured_context),
            "user_prompt": self._build_user_prompt(query),
            "citation_map": citation_map,
            "metadata": {
                "query_type": query_type.value,
                "decisive_blocks": [
                    b["block_id"] for b in decisive_blocks
                ] if decisive_blocks else []
            }
        }

    # ============================================================
    # QUERY CLASSIFIER
    # ============================================================
    def _classify_query(self, query: str) -> QueryType:
        q = query.lower()

        if re.search(r"\bis there\b|\bdoes the video show\b", q):
            return QueryType.EXISTENCE

        if re.search(r"\bwhen\b|\bwhat time\b", q):
            return QueryType.TEMPORAL

        if re.search(r"\bwho said\b|\bwho\b", q):
            return QueryType.ATTRIBUTION

        if re.search(r"\bcontradict\b|\binconsisten", q):
            return QueryType.CONTRADICTION

        if re.search(r"\bdescribe\b|\bwhat happens\b", q):
            return QueryType.DESCRIPTIVE

        return QueryType.OTHER

    # ============================================================
    # EVIDENCE ANNOTATION
    # ============================================================
    def _annotate_evidence(self, item):

        visual_direct = False
        transcript_content = False

        for frame in item.structured_frames:
            if frame.get("caption"):
                visual_direct = True

        for seg in item.structured_transcripts:
            if seg.get("text"):
                transcript_content = True

        return {
            "item": item,
            "visual_direct": visual_direct,
            "transcript_content": transcript_content,
            "retrieval_score": getattr(item, "final_score", 0)
        }

    # ============================================================
    # PRIORITIZATION
    # ============================================================
    def _prioritize(self, enriched_items, query_type):

        def score(e):
            base = e["retrieval_score"]

            # Boost visual evidence for existence queries
            if query_type == QueryType.EXISTENCE and e["visual_direct"]:
                base += 0.15

            # Boost transcript for attribution
            if query_type == QueryType.ATTRIBUTION and e["transcript_content"]:
                base += 0.1

            return base

        return sorted(enriched_items, key=score, reverse=True)

    # ============================================================
    # SUFFICIENCY DETECTION
    # ============================================================
    def _detect_sufficiency(self, ranked_items, query, query_type):

        decisive = []

        if query_type == QueryType.EXISTENCE:
            for idx, e in enumerate(ranked_items, start=1):
                if e["visual_direct"]:
                    decisive.append({
                        "block_id": idx,
                        **e
                    })
                    break

        return decisive

    # ============================================================
    # CONTEXT ASSEMBLY
    # ============================================================
    def _assemble_context(self, items):

        blocks = []
        citation_map = {}

        for idx, e in enumerate(items, start=1):
            item = e["item"]
            citation_map[idx] = {
                "chunk_ids": item.chunk_ids,
                "case_id": item.case_id,
                "start_time": item.temporal.primary.start_time,
                "end_time": item.temporal.primary.end_time
            }
            header = (
                f"[Block {idx}]\n"
                f"Case ID: {item.case_id}\n"
                f"Time Range: "
                f"{item.temporal.primary.start_time:.2f}–"
                f"{item.temporal.primary.end_time:.2f}\n"
                f"{'-'*72}\n"
            )

            transcript_section = []
            for seg in item.structured_transcripts:
                text = seg.get("text")
                if text:
                    speaker = seg.get("speaker", "UNKNOWN")
                    start = seg.get("start_time")
                    end = seg.get("end_time")
                    if start and end:
                        transcript_section.append(
                            f"[{start:.2f}-{end:.2f}] {speaker}: {text}"
                        )
                    else:
                        transcript_section.append(
                            f"{speaker}: {text}"
                        )

            visual_section = []
            for frame in item.structured_frames:
                ts = frame.get("timestamp")
                caption = frame.get("caption")
                ocr = frame.get("ocr_text")

                if caption:
                    visual_section.append(
                        f"[Frame @ {ts:.2f}] CAPTION: {caption}"
                    )

                if ocr:
                    visual_section.append(
                        f"[Frame @ {ts:.2f}] OCR: {ocr}"
                    )

            score_block = ""
            if self.include_scores:
                score_block = (
                    f"\n[Scores] Retrieval: "
                    f"{getattr(item, 'retrieval_score', 0):.3f}"
                )

            block_text = (
                header +
                "\n[Transcript]\n" +
                "\n".join(transcript_section) +
                "\n\n[Visual Evidence]\n" +
                "\n".join(visual_section) +
                score_block
            )

            blocks.append(block_text)

        return "\n\n".join(blocks),citation_map

    # ============================================================
    # PROMPT POLICY
    # ============================================================
    def _build_system_prompt(self, mode: LegalMode, query_type: QueryType):

        base_rules = """
You must rely ONLY on explicit statements from the evidence blocks.
Do NOT speculate or infer beyond what is directly stated.
If one block fully answers the question, do not use additional blocks.
Cite evidence using [Block X].
If insufficient evidence exists, explicitly state that it is insufficient.
"""

        if mode == LegalMode.EVIDENCE:
            role = "You are a strict legal evidence verification system."
        elif mode == LegalMode.ASSISTANT:
            role = "You are a legal assistant preparing structured case analysis."
        elif mode == LegalMode.OPPOSITION:
            role = "You are acting as opposing counsel."
        else:
            role = "You are a legal reasoning assistant."

        query_guidance = f"\nQuery Type: {query_type.value}\n"

        return textwrap.dedent(role + "\n" + base_rules + query_guidance)

    def _build_user_prompt(self, query: str):
        return f"""
USER QUERY:
{query}

Use ONLY the evidence blocks above.
Cite every factual claim using [Block X].
"""

    # ============================================================
    # TOKEN GUARD
    # ============================================================
    def _truncate(self, text: str):
        words = text.split()
        if len(words) <= self.max_tokens:
            return text
        return " ".join(words[:self.max_tokens])