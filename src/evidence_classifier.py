import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from src.context_builder import ContextBuilder
from src.citation_utils import is_pdf_chunk, build_citation_ref, source_type_label


# ─────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    """Single classified piece of evidence."""

    # Classification
    classification: str          # SUPPORTING | OPPOSING | NEUTRAL
    reason: str                  # One sentence why
    strength: int                # 1 (weak) → 5 (decisive)

    # Content
    text: str                    # The actual evidence text
    speaker: str                 # SPEAKER_XX or "DOCUMENT"

    # Citation
    citation_ref: str            # [01:36 → 02:14] or [FIR · §2 · pp.1-2]
    source_type: str             # video | fir | witness_statement | etc.
    chunk_id: str

    # Video-specific
    start_time: float = 0.0
    end_time: float = 0.0

    # PDF-specific
    section_title: str = ""
    section_type: str = ""
    page_start: int = 0
    page_end: int = 0
    original_name: str = ""


@dataclass
class EvidenceMap:
    """Full classification result for a case."""
    case_id: str
    lawyer_position: str

    supporting: List[EvidenceItem] = field(default_factory=list)
    opposing:   List[EvidenceItem] = field(default_factory=list)
    neutral:    List[EvidenceItem] = field(default_factory=list)

    total_chunks_classified: int = 0
    failed_classifications:  int = 0

    @property
    def summary(self) -> Dict:
        return {
            "supporting_count": len(self.supporting),
            "opposing_count":   len(self.opposing),
            "neutral_count":    len(self.neutral),
            "total_classified": self.total_chunks_classified,
            "failed":           self.failed_classifications,
            "strongest_support": (
                self.supporting[0].reason
                if self.supporting else "None found"
            ),
            "strongest_opposition": (
                self.opposing[0].reason
                if self.opposing else "None found"
            ),
        }


# ─────────────────────────────────────────────────────────────────
# Evidence Classifier
# ─────────────────────────────────────────────────────────────────

class EvidenceClassifier:
    """
    Classifies all retrieved chunks as SUPPORTING, OPPOSING,
    or NEUTRAL relative to the lawyer's position.

    Works across both video chunks and PDF document chunks.
    """

    def __init__(
        self,
        retriever,
        llm_client,
        context_builder: ContextBuilder,
        top_k: int = 50,
    ):
        self.retriever       = retriever
        self.llm_client      = llm_client
        self.context_builder = context_builder
        self.top_k           = top_k

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def classify(
        self,
        case_id: str,
        lawyer_position: str = "",
    ) -> EvidenceMap:
        """
        Main entry point.

        Args:
            case_id:         Case to classify evidence for.
            lawyer_position: e.g. "I am defending the driver
                              against negligence charges."

        Returns:
            EvidenceMap with supporting/opposing/neutral sorted
            by strength descending.
        """

        evidence_map = EvidenceMap(
            case_id=case_id,
            lawyer_position=lawyer_position,
        )

        # 1. Retrieve most relevant chunks for this position
        retrieved_items = self.retriever.retrieve(
            case_id=case_id,
            query=lawyer_position,
            top_k=self.top_k,
        )

        if not retrieved_items:
            return evidence_map

        # 2. Classify each chunk
        for item in retrieved_items:
            evidence_map.total_chunks_classified += 1

            try:
                evidence_item = self._classify_item(
                    item, lawyer_position
                )
                if evidence_item is None:
                    evidence_map.failed_classifications += 1
                    continue

                cls = evidence_item.classification
                if cls == "SUPPORTING":
                    evidence_map.supporting.append(evidence_item)
                elif cls == "OPPOSING":
                    evidence_map.opposing.append(evidence_item)
                else:
                    evidence_map.neutral.append(evidence_item)

            except Exception as e:
                print(f"[EvidenceClassifier] chunk failed: {e}", flush=True)
                evidence_map.failed_classifications += 1
                continue

        print(
            f"[EvidenceClassifier] {evidence_map.total_chunks_classified} chunks → "
            f"S={len(evidence_map.supporting)} O={len(evidence_map.opposing)} "
            f"N={len(evidence_map.neutral)} failed={evidence_map.failed_classifications}",
            flush=True,
        )

        # 3. Sort each list by strength descending
        evidence_map.supporting.sort(
            key=lambda x: x.strength, reverse=True
        )
        evidence_map.opposing.sort(
            key=lambda x: x.strength, reverse=True
        )

        return evidence_map

    # ─────────────────────────────────────────────────────────────
    # Per-chunk classification
    # ─────────────────────────────────────────────────────────────

    def _classify_item(
        self,
        item,
        lawyer_position: str,
    ) -> Optional[EvidenceItem]:

        is_pdf   = is_pdf_chunk(item)
        text     = self._extract_text(item)
        speaker  = self._extract_speaker(item)
        cite_ref = build_citation_ref(item)

        if not text.strip():
            return None

        # Build classification prompt
        prompt = self._build_prompt(
            lawyer_position=lawyer_position,
            text=text,
            speaker=speaker,
            citation_ref=cite_ref,
            is_pdf=is_pdf,
            item=item,
        )

        # Call LLM
        raw_response = self.llm_client.classify(prompt)

        # Parse JSON response
        parsed = self._parse_response(raw_response)
        if parsed is None:
            return None

        # Build EvidenceItem
        meta = getattr(item, "metadata", {})

        return EvidenceItem(
            classification = parsed["classification"],
            reason         = parsed["reason"],
            strength       = parsed["strength"],
            text           = text,
            speaker        = speaker,
            citation_ref   = cite_ref,
            source_type    = meta.get("source_type", "video"),
            chunk_id       = item.chunk_ids[0] if item.chunk_ids else "",

            # Video fields
            start_time = getattr(getattr(getattr(item, "temporal", None), "primary", None), "start_time", 0.0) or 0.0,
            end_time   = getattr(getattr(getattr(item, "temporal", None), "primary", None), "end_time",   0.0) or 0.0,

            # PDF fields
            section_title = meta.get("section_title", ""),
            section_type  = meta.get("section_type", ""),
            page_start    = meta.get("page_span_start", 0),
            page_end      = meta.get("page_span_end", 0),
            original_name = meta.get("original_name", ""),
        )

    # ─────────────────────────────────────────────────────────────
    # Prompt builder
    # ─────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        lawyer_position: str,
        text: str,
        speaker: str,
        citation_ref: str,
        is_pdf: bool,
        item,
    ) -> str:

        meta = getattr(item, "metadata", {})

        if is_pdf:
            source_label = source_type_label(
                meta.get("source_type", "document")
            )
            section = meta.get("section_title", "")
            source_context = (
                f"Source: {source_label}\n"
                f"Section: {section}\n"
                f"Citation: {citation_ref}"
            )
        else:
            source_context = (
                f"Source: Video testimony\n"
                f"Speaker: {speaker}\n"
                f"Citation: {citation_ref}"
            )

        return f"""You are a strict legal analyst.

LAWYER'S POSITION:
{lawyer_position}

EVIDENCE:
{source_context}

{text}

TASK:
Classify this evidence relative to the lawyer's position.

SUPPORTING — directly helps or strengthens the lawyer's case
OPPOSING   — directly hurts or contradicts the lawyer's case
NEUTRAL    — neither helps nor hurts

Respond ONLY in this exact JSON format, nothing else:
{{
    "classification": "SUPPORTING",
    "reason": "one sentence explaining why",
    "strength": 3
}}

Rules:
- classification must be exactly: SUPPORTING, OPPOSING, or NEUTRAL
- reason must be one sentence only
- strength is 1 (barely relevant) to 5 (decisive)
- No explanation outside the JSON"""

    # ─────────────────────────────────────────────────────────────
    # JSON parser
    # ─────────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> Optional[Dict]:
        """
        Safely parse LLM JSON response.
        Handles common LLM formatting issues:
        - Markdown code blocks
        - Extra text before/after JSON
        - Invalid strength values
        """
        # Strip markdown code blocks if present
        raw = re.sub(r"```json|```", "", raw).strip()

        # Extract JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        # Validate classification value
        valid_classes = {"SUPPORTING", "OPPOSING", "NEUTRAL"}
        cls = parsed.get("classification", "").upper().strip()
        if cls not in valid_classes:
            # Try to recover from partial match
            for valid in valid_classes:
                if valid in cls:
                    cls = valid
                    break
            else:
                return None

        # Validate and clamp strength
        try:
            strength = int(parsed.get("strength", 3))
            strength = max(1, min(5, strength))
        except (ValueError, TypeError):
            strength = 3

        return {
            "classification": cls,
            "reason":         str(parsed.get("reason", "")).strip(),
            "strength":       strength,
        }

    # ─────────────────────────────────────────────────────────────
    # Helpers — mirrors context_builder logic
    # ─────────────────────────────────────────────────────────────

    def _extract_text(self, item) -> str:
        texts = []
        for seg in getattr(item, "structured_transcripts", []):
            t = seg.get("text", "").strip()
            if t:
                texts.append(t)
        return " ".join(texts)

    def _extract_speaker(self, item) -> str:
        meta = getattr(item, "metadata", {})
        if meta.get("source_type") in ("fir", "witness_statement", "document", "pdf"):
            return "DOCUMENT"
        for seg in getattr(item, "structured_transcripts", []):
            sp = seg.get("speaker", "")
            if sp and sp not in ("UNKNOWN", ""):
                return sp
        return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────
# Output formatter — for Gradio UI
# ─────────────────────────────────────────────────────────────────

def format_evidence_map(evidence_map: EvidenceMap) -> Dict:
    """
    Formats EvidenceMap into clean dicts for Gradio display.
    Returns three lists ready for gr.Dataframe.
    """

    def format_items(items: List[EvidenceItem]) -> List[List]:
        rows = []
        for item in items:
            rows.append([
                item.strength,
                item.citation_ref,
                item.source_type.replace("_", " ").title(),
                item.reason,
                item.text[:200] + "..." if len(item.text) > 200
                else item.text,
            ])
        return rows

    headers = ["Strength", "Citation", "Source", "Reason", "Text"]

    return {
        "supporting": {
            "headers": headers,
            "rows":    format_items(evidence_map.supporting),
        },
        "opposing": {
            "headers": headers,
            "rows":    format_items(evidence_map.opposing),
        },
        "neutral": {
            "headers": headers,
            "rows":    format_items(evidence_map.neutral),
        },
        "summary": evidence_map.summary,
    }