import json
import uuid
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class Round:
    """One back-and-forth exchange."""
    round_number:    int
    lawyer_argument: str
    critique:        str        # Devil's Advocate response
    weaknesses:      List[str]  # Specific weak points found
    opposition_args: List[str]  # What opposition will argue
    evidence_used:   List[Dict] # Citations used in critique
    how_to_strengthen: List[str]# Concrete suggestions
    timestamp:       str


@dataclass
class Session:
    """One devil's advocate session for a case."""
    session_id:  str
    case_id:     str
    topic:       str            # Opening topic/theme
    created_at:  str
    updated_at:  str
    rounds:      List[Round] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "case_id":    self.case_id,
            "topic":      self.topic,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "rounds": [
                {
                    "round_number":     r.round_number,
                    "lawyer_argument":  r.lawyer_argument,
                    "critique":         r.critique,
                    "weaknesses":       r.weaknesses,
                    "opposition_args":  r.opposition_args,
                    "evidence_used":    r.evidence_used,
                    "how_to_strengthen": r.how_to_strengthen,
                    "timestamp":        r.timestamp,
                }
                for r in self.rounds
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Session":
        rounds = [
            Round(
                round_number     = r["round_number"],
                lawyer_argument  = r["lawyer_argument"],
                critique         = r["critique"],
                weaknesses       = r["weaknesses"],
                opposition_args  = r["opposition_args"],
                evidence_used    = r["evidence_used"],
                how_to_strengthen = r["how_to_strengthen"],
                timestamp        = r["timestamp"],
            )
            for r in data.get("rounds", [])
        ]
        return cls(
            session_id = data["session_id"],
            case_id    = data["case_id"],
            topic      = data["topic"],
            created_at = data["created_at"],
            updated_at = data["updated_at"],
            rounds     = rounds,
        )


# ─────────────────────────────────────────────────────────────────
# Devil's Advocate
# ─────────────────────────────────────────────────────────────────

class DevilsAdvocate:
    """
    Attacks lawyer arguments as opposing counsel.
    Maintains persistent multi-session history per case.

    Each session is a back-and-forth dialogue where:
    → Lawyer states an argument
    → System attacks it with evidence
    → Lawyer refines their argument
    → System attacks the refined version
    → ... until argument is bulletproof

    Multiple sessions per case for different argument tracks.
    """

    def __init__(
        self,
        retriever,
        llm_client,
        storage_path: str = "data",
        top_k: int = 10,
    ):
        self.retriever    = retriever
        self.llm_client   = llm_client
        self.storage_path = Path(storage_path)
        self.top_k        = top_k

        # Active session
        self._session: Optional[Session] = None

    # ─────────────────────────────────────────────────────────────
    # Session Management
    # ─────────────────────────────────────────────────────────────

    def new_session(self, case_id: str, topic: str) -> Session:
        """
        Start a new devil's advocate session.
        Saves to disk immediately.
        """
        now = datetime.utcnow().isoformat()

        session = Session(
            session_id = str(uuid.uuid4())[:8],
            case_id    = case_id,
            topic      = topic,
            created_at = now,
            updated_at = now,
        )

        self._session = session
        self._save_session(session)
        return session

    def load_session(
        self, case_id: str, session_id: str
    ) -> Session:
        """
        Load an existing session to resume.
        """
        path = self._session_path(case_id, session_id)

        if not path.exists():
            raise FileNotFoundError(
                f"Session {session_id} not found for case {case_id}"
            )

        with open(path, "r") as f:
            data = json.load(f)

        self._session = Session.from_dict(data)
        return self._session

    def list_sessions(self, case_id: str) -> List[Dict]:
        """
        List all sessions for a case.
        Returns summary of each session.
        """
        folder = self._case_folder(case_id)

        if not folder.exists():
            return []

        sessions = []
        for file in sorted(folder.glob("session_*.json")):
            with open(file, "r") as f:
                data = json.load(f)
            sessions.append({
                "session_id": data["session_id"],
                "topic":      data["topic"],
                "rounds":     len(data.get("rounds", [])),
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
            })

        return sessions

    def get_history(self) -> Optional[Session]:
        """Return current active session."""
        return self._session

    def clear_session(self) -> None:
        """Clear active session from memory (does not delete file)."""
        self._session = None

    # ─────────────────────────────────────────────────────────────
    # Main API — argue()
    # ─────────────────────────────────────────────────────────────

    def argue(self, lawyer_argument: str) -> Round:
        """
        Main entry point.
        Lawyer states argument → system attacks it.

        Args:
            lawyer_argument: The argument to attack.

        Returns:
            Round with critique, weaknesses, opposition args,
            evidence used, and suggestions to strengthen.
        """
        if self._session is None:
            raise RuntimeError(
                "No active session. Call new_session() or "
                "load_session() first."
            )

        # 1. Retrieve opposing evidence
        opposing_items = self.retriever.retrieve(
            case_id=self._session.case_id,
            query=f"evidence against: {lawyer_argument}",
            top_k=self.top_k,
        )

        # 2. Build evidence context
        evidence_context = self._format_evidence(opposing_items)
        evidence_citations = self._extract_citations(opposing_items)

        # 3. Build history context
        history_context = self._format_history()

        # 4. Build prompt
        round_number = len(self._session.rounds) + 1
        prompt = self._build_prompt(
            lawyer_argument  = lawyer_argument,
            evidence_context = evidence_context,
            history_context  = history_context,
            round_number     = round_number,
        )

        # 5. Call LLM
        raw = self.llm_client.classify(prompt)
        parsed = self._parse_response(raw)

        # 6. Build Round
        now = datetime.utcnow().isoformat()
        round_obj = Round(
            round_number     = round_number,
            lawyer_argument  = lawyer_argument,
            critique         = parsed.get("critique", ""),
            weaknesses       = parsed.get("weaknesses", []),
            opposition_args  = parsed.get("opposition_args", []),
            evidence_used    = evidence_citations,
            how_to_strengthen = parsed.get("how_to_strengthen", []),
            timestamp        = now,
        )

        # 7. Save to session
        self._session.rounds.append(round_obj)
        self._session.updated_at = now
        self._save_session(self._session)

        return round_obj

    # ─────────────────────────────────────────────────────────────
    # Prompt builder
    # ─────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        lawyer_argument:  str,
        evidence_context: str,
        history_context:  str,
        round_number:     int,
    ) -> str:

        history_section = ""
        if history_context:
            history_section = f"""
PREVIOUS ROUNDS IN THIS SESSION:
{history_context}

The lawyer has already addressed these points.
Do not repeat weaknesses already resolved.
Focus on what STILL remains weak or newly exposed.
"""

        return f"""You are the opposing counsel in a legal proceeding.
Your job is to brutally and specifically attack the lawyer's argument
using the evidence available.

Round {round_number} of cross-examination.
{history_section}

LAWYER'S CURRENT ARGUMENT:
"{lawyer_argument}"

AVAILABLE EVIDENCE TO USE AGAINST THIS ARGUMENT:
{evidence_context}

YOUR TASK:
Attack this argument as opposing counsel would in court.
Be specific. Use exact citations. Be brutal but accurate.

Respond ONLY in this exact JSON format:
{{
    "critique": "2-3 sentence overall assessment of how weak this argument is",
    "weaknesses": [
        "specific weakness 1 with citation",
        "specific weakness 2 with citation",
        "specific weakness 3 with citation"
    ],
    "opposition_args": [
        "argument opposition will make in court 1",
        "argument opposition will make in court 2",
        "argument opposition will make in court 3"
    ],
    "how_to_strengthen": [
        "concrete suggestion 1 to fix the argument",
        "concrete suggestion 2 to fix the argument"
    ]
}}

Rules:
- weaknesses must reference specific evidence with citations
- opposition_args must be realistic legal arguments
- how_to_strengthen must be actionable and specific
- No text outside the JSON"""

    # ─────────────────────────────────────────────────────────────
    # History formatter
    # ─────────────────────────────────────────────────────────────

    def _format_history(self) -> str:
        """
        Formats previous rounds into context for the LLM.
        Only includes last 3 rounds to stay within token limits.
        """
        if not self._session or not self._session.rounds:
            return ""

        # Last 3 rounds only
        recent_rounds = self._session.rounds[-3:]
        lines = []

        for r in recent_rounds:
            lines.append(
                f"Round {r.round_number}:\n"
                f"  Lawyer argued: \"{r.lawyer_argument}\"\n"
                f"  Weaknesses found: {'; '.join(r.weaknesses[:2])}\n"
                f"  Suggestions given: "
                f"{'; '.join(r.how_to_strengthen[:2])}"
            )

        return "\n\n".join(lines)

    # ─────────────────────────────────────────────────────────────
    # Evidence formatter
    # ─────────────────────────────────────────────────────────────

    def _format_evidence(self, items) -> str:
        """Format retrieved items into readable evidence blocks."""
        if not items:
            return "No direct opposing evidence found."

        blocks = []
        for idx, item in enumerate(items, start=1):
            cite_ref = self._build_citation_ref(item)
            text     = self._extract_text(item)
            if not text:
                continue
            blocks.append(f"[{idx}] {cite_ref}\n{text}")

        return "\n\n".join(blocks) if blocks else "No evidence found."

    def _extract_citations(self, items) -> List[Dict]:
        """Extract citation metadata from retrieved items."""
        citations = []
        for item in items:
            meta = getattr(item, "metadata", {})
            citations.append({
                "citation_ref": self._build_citation_ref(item),
                "source_type":  meta.get("source_type", "video"),
                "chunk_id": (
                    item.chunk_ids[0] if item.chunk_ids else ""
                ),
            })
        return citations

    def _extract_text(self, item) -> str:
        texts = []
        for seg in item.structured_transcripts:
            t = seg.get("text", "").strip()
            if t:
                texts.append(t)
        return " ".join(texts)

    # ─────────────────────────────────────────────────────────────
    # Response parser
    # ─────────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> Dict:
        """Parse LLM JSON response safely."""
        import re

        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if not match:
            return self._fallback_response(raw)

        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return self._fallback_response(raw)

        # Ensure lists are lists
        for key in ("weaknesses", "opposition_args",
                    "how_to_strengthen"):
            if not isinstance(parsed.get(key), list):
                parsed[key] = []

        return parsed

    def _fallback_response(self, raw: str) -> Dict:
        """If JSON parsing fails return raw text as critique."""
        return {
            "critique":          raw[:500] if raw else "Analysis unavailable.",
            "weaknesses":        [],
            "opposition_args":   [],
            "how_to_strengthen": [],
        }

    # ─────────────────────────────────────────────────────────────
    # Storage helpers
    # ─────────────────────────────────────────────────────────────

    def _save_session(self, session: Session) -> None:
        folder = self._case_folder(session.case_id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"session_{session.session_id}.json"
        with open(path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def _session_path(
        self, case_id: str, session_id: str
    ) -> Path:
        return (
            self._case_folder(case_id)
            / f"session_{session_id}.json"
        )

    def _case_folder(self, case_id: str) -> Path:
        return (
            self.storage_path
            / "cases"
            / case_id
            / "devil_advocate"
        )

    # ─────────────────────────────────────────────────────────────
    # Citation helpers — mirrors other modules
    # ─────────────────────────────────────────────────────────────

    def _is_pdf_chunk(self, item) -> bool:
        if item.structured_transcripts:
            speaker = item.structured_transcripts[0].get(
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


# ─────────────────────────────────────────────────────────────────
# Output formatter — for Gradio UI
# ─────────────────────────────────────────────────────────────────

def format_round(round_obj: Round) -> Dict:
    """Format a Round for Gradio display."""
    return {
        "round_number": round_obj.round_number,
        "critique":     round_obj.critique,
        "weaknesses":   "\n".join(
            f"{i+1}. {w}"
            for i, w in enumerate(round_obj.weaknesses)
        ),
        "opposition":   "\n".join(
            f"{i+1}. {a}"
            for i, a in enumerate(round_obj.opposition_args)
        ),
        "strengthen":   "\n".join(
            f"{i+1}. {s}"
            for i, s in enumerate(round_obj.how_to_strengthen)
        ),
        "evidence_count": len(round_obj.evidence_used),
    }


def format_session_list(sessions: List[Dict]) -> List[List]:
    """Format session list for gr.Dataframe."""
    rows = []
    for s in sessions:
        rows.append([
            s["session_id"],
            s["topic"],
            s["rounds"],
            s["created_at"][:10],
            s["updated_at"][:10],
        ])
    return rows