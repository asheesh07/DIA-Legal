import json
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class Statement:
    """A single statement from any source."""
    text:        str
    speaker:     str        # SPEAKER_01 | DOCUMENT
    source_type: str        # video | fir | witness_statement | etc.
    citation_ref: str       # [01:36 → 02:14] or [Witness Statement · §2]
    embedding:   np.ndarray

    # Video-specific
    start_time:  float = 0.0
    end_time:    float = 0.0
    chunk_id:    str   = ""

    # PDF-specific
    section_title: str = ""
    original_name: str = ""
    page_start:    int = 0
    page_end:      int = 0


@dataclass
class Contradiction:
    """A verified contradiction between two statements."""

    # The two conflicting statements
    statement_a: Statement
    statement_b: Statement

    # LLM verdict
    explanation: str        # Why they contradict
    severity:    str        # high | medium | low
    severity_score: int     # 3=high 2=medium 1=low

    # Cross-source flag
    is_cross_source: bool   # True if video vs PDF


@dataclass
class ContradictionReport:
    """Full contradiction analysis for a case."""
    case_id: str

    contradictions: List[Contradiction] = field(default_factory=list)

    total_statements_checked: int = 0
    total_pairs_flagged:      int = 0
    total_llm_calls:          int = 0
    failed_verifications:     int = 0

    @property
    def summary(self) -> Dict:
        high   = [c for c in self.contradictions if c.severity == "high"]
        medium = [c for c in self.contradictions if c.severity == "medium"]
        low    = [c for c in self.contradictions if c.severity == "low"]
        cross  = [c for c in self.contradictions if c.is_cross_source]

        return {
            "total_contradictions": len(self.contradictions),
            "high_severity":        len(high),
            "medium_severity":      len(medium),
            "low_severity":         len(low),
            "cross_source":         len(cross),
            "statements_checked":   self.total_statements_checked,
            "pairs_flagged":        self.total_pairs_flagged,
            "llm_calls_made":       self.total_llm_calls,
        }


# ─────────────────────────────────────────────────────────────────
# Contradiction Detector
# ─────────────────────────────────────────────────────────────────

class ContradictionDetector:
    """
    Detects contradictions across video testimony and PDF documents.

    Two-stage approach:
    Stage 1 — Embedding filter: fast, free, finds suspicious pairs
    Stage 2 — LLM verification: confirms real contradictions

    Cross-source: compares video speaker statements against
    PDF document chunks (witness statement vs video testimony).
    """

    def __init__(
        self,
        retriever,
        llm_client,
        embedder,
        similarity_threshold: float = 0.4,
        top_k: int = 100,
    ):
        self.retriever            = retriever
        self.llm_client           = llm_client
        self.embedder             = embedder
        self.similarity_threshold = similarity_threshold
        self.top_k                = top_k

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def detect(self, case_id: str) -> ContradictionReport:
        """
        Main entry point.
        Runs full contradiction detection for a case.

        Args:
            case_id: Case to analyze.

        Returns:
            ContradictionReport sorted by severity descending.
        """

        report = ContradictionReport(case_id=case_id)

        # 1. Retrieve all chunks for this case
        all_items = self.retriever.retrieve(
            case_id=case_id,
            query="testimony statement evidence",
            top_k=self.top_k,
        )

        if not all_items:
            return report

        # 2. Build Statement objects from chunks
        statements = self._build_statements(all_items)
        report.total_statements_checked = len(statements)

        # 3. Split into video and PDF statements
        video_statements = [
            s for s in statements
            if s.source_type in ("video", "Youtube", "Local")
        ]
        pdf_statements = [
            s for s in statements
            if s.source_type not in ("video", "Youtube", "Local")
        ]

        # 4. Stage 1 — find suspicious pairs via embeddings
        suspicious_pairs: List[Tuple[Statement, Statement, bool]] = []

        # Within video: same speaker contradictions
        same_speaker_pairs = self._find_same_speaker_pairs(
            video_statements
        )
        suspicious_pairs.extend(
            [(a, b, False) for a, b in same_speaker_pairs]
        )

        # Cross source: video vs PDF
        if video_statements and pdf_statements:
            cross_pairs = self._find_cross_source_pairs(
                video_statements, pdf_statements
            )
            suspicious_pairs.extend(
                [(a, b, True) for a, b in cross_pairs]
            )

        report.total_pairs_flagged = len(suspicious_pairs)

        if not suspicious_pairs:
            return report

        # 5. Stage 2 — LLM verification
        for stmt_a, stmt_b, is_cross in suspicious_pairs:
            report.total_llm_calls += 1

            try:
                contradiction = self._verify_contradiction(
                    stmt_a, stmt_b, is_cross
                )
                if contradiction is not None:
                    report.contradictions.append(contradiction)
            except Exception:
                report.failed_verifications += 1
                continue

        # 6. Sort by severity descending
        severity_order = {"high": 3, "medium": 2, "low": 1}
        report.contradictions.sort(
            key=lambda x: severity_order.get(x.severity, 0),
            reverse=True
        )

        return report

    # ─────────────────────────────────────────────────────────────
    # Stage 1A — Same speaker pairs (within video)
    # ─────────────────────────────────────────────────────────────

    def _find_same_speaker_pairs(
        self, video_statements: List[Statement]
    ) -> List[Tuple[Statement, Statement]]:
        """
        Groups video statements by speaker.
        Flags pairs with low cosine similarity.
        Low similarity = semantically different = potential contradiction.
        """

        # Group by speaker
        by_speaker: Dict[str, List[Statement]] = {}
        for stmt in video_statements:
            spk = stmt.speaker
            if spk not in by_speaker:
                by_speaker[spk] = []
            by_speaker[spk].append(stmt)

        suspicious: List[Tuple[Statement, Statement]] = []

        for speaker, stmts in by_speaker.items():
            if speaker == "DOCUMENT" or len(stmts) < 2:
                continue

            # Pairwise cosine similarity
            for i in range(len(stmts)):
                for j in range(i + 1, len(stmts)):
                    sim = self._cosine_similarity(
                        stmts[i].embedding,
                        stmts[j].embedding
                    )
                    # Low similarity = semantically different
                    if sim < self.similarity_threshold:
                        suspicious.append((stmts[i], stmts[j]))

        return suspicious

    # ─────────────────────────────────────────────────────────────
    # Stage 1B — Cross source pairs (video vs PDF)
    # ─────────────────────────────────────────────────────────────

    def _find_cross_source_pairs(
        self,
        video_statements: List[Statement],
        pdf_statements: List[Statement],
    ) -> List[Tuple[Statement, Statement]]:
        """
        Compares each video statement against all PDF chunks.
        Flags pairs with low cosine similarity.

        Low similarity between a video statement and a PDF chunk
        about the same topic = potential contradiction between
        spoken testimony and written document.
        """

        suspicious: List[Tuple[Statement, Statement]] = []

        for v_stmt in video_statements:
            if v_stmt.speaker == "DOCUMENT":
                continue

            for p_stmt in pdf_statements:
                # Skip very short PDF chunks — not enough context
                if len(p_stmt.text.split()) < 10:
                    continue

                sim = self._cosine_similarity(
                    v_stmt.embedding,
                    p_stmt.embedding
                )

                # For cross-source use slightly higher threshold
                # because different phrasing is expected
                cross_threshold = self.similarity_threshold + 0.1

                if sim < cross_threshold:
                    suspicious.append((v_stmt, p_stmt))

        return suspicious

    # ─────────────────────────────────────────────────────────────
    # Stage 2 — LLM verification
    # ─────────────────────────────────────────────────────────────

    def _verify_contradiction(
        self,
        stmt_a: Statement,
        stmt_b: Statement,
        is_cross_source: bool,
    ) -> Optional[Contradiction]:
        """
        Sends suspicious pair to LLM for verification.
        Returns Contradiction if confirmed, None if not.
        """

        prompt = self._build_verification_prompt(
            stmt_a, stmt_b, is_cross_source
        )

        raw = self.llm_client.classify(prompt)
        parsed = self._parse_verification(raw)

        if parsed is None:
            return None

        if not parsed["is_contradiction"]:
            return None

        return Contradiction(
            statement_a     = stmt_a,
            statement_b     = stmt_b,
            explanation     = parsed["explanation"],
            severity        = parsed["severity"],
            severity_score  = {"high": 3, "medium": 2, "low": 1}.get(
                parsed["severity"], 1
            ),
            is_cross_source = is_cross_source,
        )

    # ─────────────────────────────────────────────────────────────
    # Prompt builder
    # ─────────────────────────────────────────────────────────────

    def _build_verification_prompt(
        self,
        stmt_a: Statement,
        stmt_b: Statement,
        is_cross_source: bool,
    ) -> str:

        if is_cross_source:
            context_label = (
                "These statements come from DIFFERENT sources — "
                "one from video testimony, one from a written document."
            )
        else:
            context_label = (
                "Both statements are from the SAME person "
                "at different points in the proceedings."
            )

        return f"""Analyze the following evidence snippets.
Identify contradictions or inconsistencies.

{context_label}

STATEMENT A:
Source: {stmt_a.source_type}
Speaker/Document: {stmt_a.speaker}
Citation: {stmt_a.citation_ref}
"{stmt_a.text}"

STATEMENT B:
Source: {stmt_b.source_type}
Speaker/Document: {stmt_b.speaker}
Citation: {stmt_b.citation_ref}
"{stmt_b.text}"

Respond ONLY in this exact JSON format:
{{
    "contradiction_pairs": true,
    "explanation": "Identify contradictions or inconsistencies between the pairs",
    "severity": "high/medium/low"
}}

Rules:
- contradiction_pairs: true if they contradict, false otherwise
- explanation: one sentence explaining the exact conflict
- severity: "high" (direct factual conflict), "medium" (likely conflict), "low" (possible conflict)
- No text outside the JSON"""

    # ─────────────────────────────────────────────────────────────
    # Statement builder
    # ─────────────────────────────────────────────────────────────

    def _build_statements(self, items) -> List[Statement]:
        """
        Converts retrieved items into Statement objects.
        Embeds each statement's text for similarity comparison.
        """
        statements = []

        for item in items:
            is_pdf = self._is_pdf_chunk(item)
            meta   = getattr(item, "metadata", {})

            structured_transcripts = getattr(item, "structured_transcripts", [])
            for seg in structured_transcripts:
                text = seg.get("text", "").strip()
                if not text or len(text.split()) < 5:
                    continue

                # Embed the text
                try:
                    embedding = self.embedder.text_embedder.embed(text)
                except Exception:
                    continue

                speaker     = seg.get("speaker", "UNKNOWN")
                source_type = meta.get("source_type", "video")
                cite_ref    = self._build_citation_ref(item)

                stmt = Statement(
                    text         = text,
                    speaker      = speaker,
                    source_type  = source_type,
                    citation_ref = cite_ref,
                    embedding    = embedding,

                    # Video fields
                    start_time = seg.get("start", 0.0),
                    end_time   = seg.get("end", 0.0),
                    chunk_id   = (
                        getattr(item, "chunk_ids", [""])[0] if getattr(item, "chunk_ids", None) else ""
                    ),

                    # PDF fields
                    section_title = meta.get("section_title", ""),
                    original_name = meta.get("original_name", ""),
                    page_start    = meta.get("page_span_start", 0),
                    page_end      = meta.get("page_span_end", 0),
                )

                statements.append(stmt)

        return statements

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _cosine_similarity(
        self, a: np.ndarray, b: np.ndarray
    ) -> float:
        """Normalised cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _is_pdf_chunk(self, item) -> bool:
        structured_transcripts = getattr(item, "structured_transcripts", [])
        if structured_transcripts:
            speaker = structured_transcripts[0].get(
                "speaker", ""
            )
            if speaker == "DOCUMENT":
                return True
        return (
            item.temporal.primary.start_time == 0.0 and
            item.temporal.primary.end_time == 0.0
        )

    def _build_citation_ref(self, item) -> str:
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
                p = (
                    f"p.{page_start}" if page_start == page_end
                    else f"pp.{page_start}-{page_end}"
                )
                parts.append(p)
            return "[" + " · ".join(parts) + "]"
        else:
            s  = item.temporal.primary.start_time
            e  = item.temporal.primary.end_time
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

    def _parse_verification(self, raw: str) -> Optional[Dict]:
        """Parse LLM verification response safely."""
        raw = re.sub(r"```json|```", "", raw).strip()

        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        is_contradiction = parsed.get("contradiction_pairs", False)
        if isinstance(is_contradiction, str):
            is_contradiction = is_contradiction.lower() == "true"

        severity = parsed.get("severity", "low").lower()
        if severity not in ("high", "medium", "low"):
            severity = "low"

        return {
            "is_contradiction": bool(is_contradiction),
            "explanation":      str(parsed.get("explanation", "")).strip(),
            "severity":         severity,
        }


# ─────────────────────────────────────────────────────────────────
# Output formatter — for Gradio UI
# ─────────────────────────────────────────────────────────────────

def format_contradiction_report(
    report: ContradictionReport,
) -> Dict:
    """
    Formats ContradictionReport for Gradio gr.Dataframe display.
    """

    severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}

    rows = []
    for c in report.contradictions:
        emoji = severity_emoji.get(c.severity, "⚪")
        cross = "✅" if c.is_cross_source else "—"

        rows.append([
            f"{emoji} {c.severity.upper()}",
            c.statement_a.citation_ref,
            c.statement_a.text[:120] + "..."
            if len(c.statement_a.text) > 120
            else c.statement_a.text,
            c.statement_b.citation_ref,
            c.statement_b.text[:120] + "..."
            if len(c.statement_b.text) > 120
            else c.statement_b.text,
            c.explanation,
            cross,
        ])

    return {
        "headers": [
            "Severity",
            "Citation A",
            "Statement A",
            "Citation B",
            "Statement B",
            "Explanation",
            "Cross-Source",
        ],
        "rows":    rows,
        "summary": report.summary,
    }