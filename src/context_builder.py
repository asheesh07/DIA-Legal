from typing import List, Dict, Tuple
from enum import Enum
import textwrap
import re
from src.citation_utils import is_pdf_chunk, build_citation_ref, source_type_label


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
        max_tokens: int = 1500,
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
        mode: LegalMode,
        history: List[Dict] = None
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
            "user_prompt": self._build_user_prompt(query, history),
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

        blocks: List[str] = []
        citation_map = {}

        for idx, e in enumerate(items, start=1):
            item   = e["item"]
            is_pdf = is_pdf_chunk(item)

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
                    "exhibit_id":    meta.get("exhibit_id", ""),
                    "exhibit_title": meta.get("exhibit_title", ""),
                    "citation_ref":  build_citation_ref(item),
                }
            else:
                speakers = list({
                    seg.get("speaker", "")
                    for seg in item.structured_transcripts
                    if seg.get("speaker") and seg.get("text")
                })
                citation_map[idx] = {
                    "chunk_ids":    item.chunk_ids,
                    "case_id":      item.case_id,
                    "start_time":   item.temporal.primary.start_time,
                    "end_time":     item.temporal.primary.end_time,
                    "speakers":     speakers,
                    "citation_ref": build_citation_ref(item),
                }

            # ── Block header ─────────────────────────────────────
            if is_pdf:
                meta          = getattr(item, "metadata", {})
                source_label  = source_type_label(
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

        # Truncate context to avoid Hugging Face "Bad Request" Token Limit Overflow
        MAX_CHARS = 25000
        current_len = 0
        final_blocks = []
        final_citation_map = {}
        for i, block in enumerate(blocks):
            if current_len + len(block) > MAX_CHARS and i > 0:
                print(f"[ContextBuilder] Truncating context at block {i+1} to fit LLM context limits.", flush=True)
                break
            final_blocks.append(block)
            final_citation_map[i+1] = citation_map[i+1]
            current_len += len(block)

        return "\n\n".join(final_blocks), final_citation_map


    # ============================================================
    # PROMPT POLICY
    # ============================================================
    def _build_system_prompt(self, mode: LegalMode, query_type: QueryType):

        base_rules = """
You are a seasoned, highly persuasive trial lawyer. Your goal is to provide an emotionally compelling, authoritative, and deeply analytical response based on the provided evidence.
- Speak with the conviction, drama, and precision of a top-tier attorney in the courtroom.
- Do not just return document chunks. Synthesize the evidence into a powerful narrative.
- If the user asks a question about the case, do not hallucinate facts outside the provided evidence. Cite evidence references using the block IDs (e.g., 1, 2) naturally in your argument.
- If the user is just making conversation (e.g. "hi", "how are you"), respond conversationally but maintain your sharp, professional lawyer persona. 

Return ONLY a valid JSON object in this format:
{
    "answer": "your persuasive, dramatic, and emotionally compelling lawyer response here",
    "supporting_evidence": [1, 2],
    "confidence": 0.9
}
"""

        if mode == LegalMode.EVIDENCE:
            role = "You are a fierce legal analyst dissecting the evidence to find the absolute truth."
        elif mode == LegalMode.ASSISTANT:
            role = "You are a brilliant legal assistant preparing a compelling narrative for the lead attorney."
        elif mode == LegalMode.OPPOSITION:
            role = "You are a ruthless opposing counsel, looking for any weakness in the evidence to exploit."
        else:
            role = "You are a persuasive legal reasoning assistant."

        query_guidance = f"\nQuery Type: {query_type.value}\n"

        return textwrap.dedent(role + "\n" + base_rules + query_guidance)

    def _build_user_prompt(self, query: str, history: List[Dict] = None):
        history_text = ""
        if history and len(history) > 0:
            history_text = "CONVERSATION HISTORY:\n"
            for msg in history[-5:]:  # Keep only last 5 messages to avoid overflow
                role = "USER" if msg.get("role") == "user" else "ASSISTANT"
                history_text += f"{role}: {msg.get('content', '')}\n"
            history_text += "\n"

        return f"""
{history_text}USER QUERY:
{query}
"""

    # ============================================================
    # TOKEN GUARD
    # ============================================================
    def _truncate(self, text: str) -> str:
        # Use a strict character limit to bypass the 1024 token limit on HF's free tier
        char_limit = 3000
        if len(text) > char_limit:
            return text[:char_limit] + "\n...[Truncated]"
        return text