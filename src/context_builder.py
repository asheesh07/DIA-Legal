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
    
    def _is_pdf_chunk(self, item) -> bool:
        """PDF chunks have speaker=DOCUMENT and start=end=0.0"""
        if item.structured_transcripts:
            speaker = item.structured_transcripts[0].get("speaker", "")
            if speaker == "DOCUMENT":
                return True
        return (
            item.temporal.primary.start_time == 0.0 and
            item.temporal.primary.end_time == 0.0
        )

    def _build_citation_ref(self, item) -> str:
        """
        Video:  [01:36 → 02:14]
        PDF:    [Witness Statement · § Facts · pp.4-7]
        """
        if self._is_pdf_chunk(item):
            meta          = getattr(item, "metadata", {})
            source_type   = meta.get("source_type", "document")
            section_title = meta.get("section_title", "")
            page_start    = meta.get("page_span_start", 0)
            page_end      = meta.get("page_span_end", 0)

            label = self._source_type_label(source_type)
            parts = [label]
            if section_title:
                parts.append(f"§ {section_title}")
            if page_start and page_end:
                page_str = (
                    f"p.{page_start}" if page_start == page_end
                    else f"pp.{page_start}-{page_end}"
                )
                parts.append(page_str)
            return "[" + " · ".join(parts) + "]"
        else:
            s = item.temporal.primary.start_time
            e = item.temporal.primary.end_time
            sm, ss = int(s) // 60, int(s) % 60
            em, es = int(e) // 60, int(e) % 60
            return f"[{sm:02d}:{ss:02d} → {em:02d}:{es:02d}]"

    def _source_type_label(self, source_type: str) -> str:
        labels = {
            "fir":               "FIR",
            "witness_statement": "Witness Statement",
            "court_order":       "Court Order",
            "evidence":          "Evidence Report",
            "charge_sheet":      "Charge Sheet",
            "document":          "Document",
            "video":             "Video",
            "Youtube":           "Video",
            "Local":             "Video",
        }
        return labels.get(
            source_type,
            source_type.replace("_", " ").title()
        )

    # ============================================================
    # CONTEXT ASSEMBLY
    # ============================================================
    def _assemble_context(self, items):

        blocks: List[str] = []
        citation_map = {}

        for idx, e in enumerate(items, start=1):
            item   = e["item"]
            is_pdf = self._is_pdf_chunk(item)

            # ── Citation map ─────────────────────────────────────
            if is_pdf:
                meta = getattr(item, "metadata", {})
                citation_map[idx] = {
                    "chunk_ids":     item.chunk_ids,
                    "case_id":       item.case_id,
                    "source_type":   meta.get("source_type", "document"),
                    "original_name": meta.get("original_name", ""),
                    "section_title": meta.get("section_title", ""),
                    "page_start":    meta.get("page_span_start", 0),
                    "page_end":      meta.get("page_span_end", 0),
                    "citation_ref":  self._build_citation_ref(item),
                }
            else:
                citation_map[idx] = {
                    "chunk_ids":  item.chunk_ids,
                    "case_id":    item.case_id,
                    "start_time": item.temporal.primary.start_time,
                    "end_time":   item.temporal.primary.end_time,
                    "citation_ref": self._build_citation_ref(item),
                }

            # ── Block header ─────────────────────────────────────
            if is_pdf:
                meta          = getattr(item, "metadata", {})
                source_label  = self._source_type_label(
                    meta.get("source_type", "document")
                )
                section_title = meta.get("section_title", "")
                page_start    = meta.get("page_span_start", 0)
                page_end      = meta.get("page_span_end", 0)
                page_str = (
                    f"p.{page_start}" if page_start == page_end
                    else f"pp.{page_start}-{page_end}"
                )
                header = (
                    f"[Block {idx}]\n"
                    f"Source:  {source_label}\n"
                    f"Section: {section_title}\n"
                    f"Pages:   {page_str}\n"
                    f"Case ID: {item.case_id}\n"
                    f"{'-'*72}\n"
                )
            else:
                header = (
                    f"[Block {idx}]\n"
                    f"Case ID: {item.case_id}\n"
                    f"Time Range: "
                    f"{item.temporal.primary.start_time:.2f}–"
                    f"{item.temporal.primary.end_time:.2f}\n"
                    f"{'-'*72}\n"
                )

            # ── Transcript / document text ────────────────────────
            transcript_section = []
            for seg in item.structured_transcripts:
                text    = seg.get("text")
                speaker = seg.get("speaker", "UNKNOWN")
                if not text:
                    continue
                if is_pdf:
                    # Documents have no timestamps or speakers
                    transcript_section.append(text)
                else:
                    start = seg.get("start_time")
                    end   = seg.get("end_time")
                    if start and end:
                        transcript_section.append(
                            f"[{start:.2f}-{end:.2f}] {speaker}: {text}"
                        )
                    else:
                        transcript_section.append(
                            f"{speaker}: {text}"
                        )

            # ── Visual evidence (video only) ──────────────────────
            visual_section = []
            if not is_pdf:
                for frame in item.structured_frames:
                    ts      = frame.get("timestamp")
                    caption = frame.get("caption")
                    ocr     = frame.get("ocr_text")
                    if caption:
                        visual_section.append(
                            f"[Frame @ {ts:.2f}] CAPTION: {caption}"
                        )
                    if ocr:
                        visual_section.append(
                            f"[Frame @ {ts:.2f}] OCR: {ocr}"
                        )

            # ── Score block ───────────────────────────────────────
            score_block = ""
            if self.include_scores:
                score_block = (
                    f"\n[Scores] Retrieval: "
                    f"{getattr(item, 'retrieval_score', 0):.3f}"
                )

            # ── Assemble ──────────────────────────────────────────
            if is_pdf:
                block_text = (
                    header +
                    "[Document Text]\n" +
                    "\n".join(transcript_section) +
                    score_block
                )
            else:
                block_text = (
                    header +
                    "\n[Transcript]\n" +
                    "\n".join(transcript_section) +
                    "\n\n[Visual Evidence]\n" +
                    "\n".join(visual_section) +
                    score_block
                )

            blocks.append(block_text)

        return "\n\n".join(blocks), citation_map


    # ============================================================
    # PROMPT POLICY
    # ============================================================
    def _build_system_prompt(self, mode: LegalMode, query_type: QueryType):

        base_rules = """
Answer the question strictly using the provided evidence.
- Do not hallucinate
- Cite evidence references using the block IDs (e.g., 1, 2)
- Keep answer concise

Return ONLY a valid JSON object in this format:
{
    "answer": "your concise answer here",
    "supporting_evidence": [1, 2],
    "confidence": 0.9
}
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
"""

    # ============================================================
    # TOKEN GUARD
    # ============================================================
    def _truncate(self, text: str) -> str:
        words = text.split()
        limit = int(self.max_tokens)
        if len(words) <= limit:
            return text
        return " ".join(words[i] for i in range(limit))