import json
import uuid
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from src.citation_utils import build_citation_ref


# ─────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class Round:
    """One back-and-forth exchange."""
    round_number:                int
    lawyer_argument:             str
    critique:                    str
    weaknesses:                  List          # List[Dict{point,exhibit,severity}]
    counter_argument:            str
    defense_probability:         float
    recommended_defense_strategy: str
    evidence_used:               List[Dict]
    # kept for backward compat with old saved sessions
    opposition_args:             List[str]
    how_to_strengthen:           List[str]
    timestamp:                   str


@dataclass
class Session:
    """One devil's advocate session for a case."""
    session_id:  str
    case_id:     str
    topic:       str
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
                    "round_number":               r.round_number,
                    "lawyer_argument":             r.lawyer_argument,
                    "critique":                    r.critique,
                    "weaknesses":                  r.weaknesses,
                    "counter_argument":            r.counter_argument,
                    "defense_probability":         r.defense_probability,
                    "recommended_defense_strategy": r.recommended_defense_strategy,
                    "evidence_used":               r.evidence_used,
                    "opposition_args":             r.opposition_args,
                    "how_to_strengthen":           r.how_to_strengthen,
                    "timestamp":                   r.timestamp,
                }
                for r in self.rounds
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Session":
        rounds = []
        for r in data.get("rounds", []):
            # Handle old format where weaknesses was List[str]
            weaknesses = r.get("weaknesses", [])
            # Handle old format where counter_argument was in opposition_args
            counter_arg = r.get("counter_argument") or (
                r.get("opposition_args", [""])[0] if r.get("opposition_args") else ""
            )
            rounds.append(Round(
                round_number                 = r["round_number"],
                lawyer_argument              = r["lawyer_argument"],
                critique                     = r["critique"],
                weaknesses                   = weaknesses,
                counter_argument             = counter_arg,
                defense_probability          = r.get("defense_probability", 0.5),
                recommended_defense_strategy  = r.get("recommended_defense_strategy", ""),
                evidence_used                = r.get("evidence_used", []),
                opposition_args              = r.get("opposition_args", []),
                how_to_strengthen            = r.get("how_to_strengthen", []),
                timestamp                    = r.get("timestamp", ""),
            ))
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

_SYSTEM_PROMPT = """You are a hostile but brilliant defense attorney. Your job is to find every weakness in the prosecution's argument and construct the strongest possible counter-argument using ONLY the evidence provided.
Never make up facts. Cite specific exhibits by ID.
You must be adversarial — your job is to win for the defense."""


class DevilsAdvocate:
    """
    Attacks lawyer arguments as opposing counsel.
    Grounds every critique in actual evidence retrieved from LanceDB.
    Builds on previous rounds via full conversation history.
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
        self._session: Optional[Session] = None

    # ─────────────────────────────────────────────────────────────
    # Session Management
    # ─────────────────────────────────────────────────────────────

    def new_session(self, case_id: str, topic: str) -> Session:
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

    def load_session(self, case_id: str, session_id: str) -> Session:
        path = self._session_path(case_id, session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id} not found for case {case_id}")
        with open(path) as f:
            data = json.load(f)
        self._session = Session.from_dict(data)
        return self._session

    def list_sessions(self, case_id: str) -> List[Dict]:
        folder = self._case_folder(case_id)
        if not folder.exists():
            return []
        sessions = []
        for file in sorted(folder.glob("session_*.json")):
            with open(file) as f:
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
        return self._session

    def clear_session(self) -> None:
        self._session = None

    # ─────────────────────────────────────────────────────────────
    # Main API
    # ─────────────────────────────────────────────────────────────

    def argue(self, lawyer_argument: str) -> Round:
        session = self._session
        if session is None:
            raise RuntimeError("No active session. Call new_session() or load_session() first.")

        # Retrieve top 5 most relevant chunks for grounding
        relevant_items = self.retriever.retrieve(
            case_id=session.case_id,
            query=lawyer_argument,
            top_k=5,
        )

        evidence_context    = self._format_evidence(relevant_items)
        evidence_citations  = self._extract_citations(relevant_items)
        history_context     = self._format_history()
        round_number        = len(session.rounds) + 1

        prompt = self._build_prompt(
            lawyer_argument  = lawyer_argument,
            evidence_context = evidence_context,
            history_context  = history_context,
            round_number     = round_number,
        )

        raw    = self.llm_client.classify(prompt)
        parsed = self._parse_response(raw)

        now = datetime.utcnow().isoformat()
        round_obj = Round(
            round_number                  = round_number,
            lawyer_argument               = lawyer_argument,
            critique                      = parsed.get("critique", ""),
            weaknesses                    = parsed.get("weaknesses", []),
            counter_argument              = parsed.get("counter_argument", ""),
            defense_probability           = parsed.get("defense_probability", 0.5),
            recommended_defense_strategy  = parsed.get("recommended_defense_strategy", ""),
            evidence_used                 = evidence_citations,
            # backward compat fields
            opposition_args               = [parsed.get("counter_argument", "")],
            how_to_strengthen             = [],
            timestamp                     = now,
        )

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
PREVIOUS ROUNDS (build on these — do not repeat weaknesses already addressed):
{history_context}
"""

        return f"""{_SYSTEM_PROMPT}

Round {round_number} of cross-examination.
{history_section}
PROSECUTION'S ARGUMENT:
"{lawyer_argument}"

GROUNDING EVIDENCE (cite specific exhibit IDs):
{evidence_context}

Respond ONLY in this exact JSON:
{{
    "critique": "direct attack on the argument in 2-3 sentences",
    "weaknesses": [
        {{"point": "specific weakness 1", "exhibit": "exhibit ID or citation", "severity": "high"}},
        {{"point": "specific weakness 2", "exhibit": "exhibit ID or citation", "severity": "medium"}}
    ],
    "counter_argument": "the strongest defense position in 2-3 sentences",
    "defense_probability": 0.72,
    "recommended_defense_strategy": "one concrete action the defense should take"
}}

Rules:
- critique must directly attack the argument using evidence facts
- weaknesses must reference real exhibit IDs from the evidence above
- severity for each weakness: "high", "medium", or "low"
- defense_probability: 0.0–1.0 float
- No text outside the JSON"""

    # ─────────────────────────────────────────────────────────────
    # History formatter — full conversation context
    # ─────────────────────────────────────────────────────────────

    def _format_history(self) -> str:
        if not self._session or not self._session.rounds:
            return ""

        lines = []
        for r in self._session.rounds:
            weaknesses_text = ""
            if r.weaknesses:
                wlist = []
                for w in r.weaknesses:
                    if isinstance(w, dict):
                        wlist.append(f"  - {w.get('point', '')} [{w.get('exhibit', '')}]")
                    else:
                        wlist.append(f"  - {w}")
                weaknesses_text = "\n".join(wlist)

            lines.append(
                f"Round {r.round_number}:\n"
                f"  Prosecution argued: \"{r.lawyer_argument}\"\n"
                f"  Weaknesses found:\n{weaknesses_text}\n"
                f"  Counter-argument: {r.counter_argument}\n"
                f"  Defense probability: {r.defense_probability}"
            )

        return "\n\n".join(lines)

    # ─────────────────────────────────────────────────────────────
    # Evidence formatter
    # ─────────────────────────────────────────────────────────────

    def _format_evidence(self, items) -> str:
        if not items:
            return "No direct case evidence found."

        blocks = []
        total_chars = 0
        MAX_CHARS = 12000

        for idx, item in enumerate(items, 1):
            cite_ref = build_citation_ref(item)
            text     = self._extract_text(item)
            if not text:
                continue
            block = f"[{cite_ref}]\n{text}"
            if total_chars + len(block) > MAX_CHARS and idx > 1:
                break
            blocks.append(block)
            total_chars += len(block)

        return "\n\n".join(blocks) if blocks else "No evidence found."

    def _extract_citations(self, items) -> List[Dict]:
        citations = []
        for item in items:
            meta = getattr(item, "metadata", {})
            citations.append({
                "citation_ref": build_citation_ref(item),
                "source_type":  meta.get("source_type", "video"),
                "chunk_id":     item.chunk_ids[0] if getattr(item, "chunk_ids", None) else "",
            })
        return citations

    def _extract_text(self, item) -> str:
        texts = []
        for seg in getattr(item, "structured_transcripts", []):
            t = seg.get("text", "").strip()
            if t:
                texts.append(t)
        if not texts:
            raw = (getattr(item, "transcript_text", "") or "").strip()
            if raw:
                return raw[:400]
        return " ".join(texts)

    # ─────────────────────────────────────────────────────────────
    # Response parser
    # ─────────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> Dict:
        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if not match:
            return self._fallback_response(raw)

        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return self._fallback_response(raw)

        # Normalise weaknesses — accept List[str] or List[Dict]
        weaknesses = parsed.get("weaknesses", [])
        if not isinstance(weaknesses, list):
            weaknesses = []
        normalised_weaknesses = []
        for w in weaknesses:
            if isinstance(w, dict):
                normalised_weaknesses.append({
                    "point":    str(w.get("point", "")),
                    "exhibit":  str(w.get("exhibit", "")),
                    "severity": str(w.get("severity", "medium")).lower(),
                })
            elif isinstance(w, str) and w.strip():
                normalised_weaknesses.append({"point": w, "exhibit": "", "severity": "medium"})

        try:
            prob = float(parsed.get("defense_probability", 0.5))
            prob = max(0.0, min(1.0, prob))
        except (TypeError, ValueError):
            prob = 0.5

        return {
            "critique":                   str(parsed.get("critique", "")),
            "weaknesses":                 normalised_weaknesses,
            "counter_argument":           str(parsed.get("counter_argument", "")),
            "defense_probability":        prob,
            "recommended_defense_strategy": str(parsed.get("recommended_defense_strategy", "")),
        }

    def _fallback_response(self, raw: str) -> Dict:
        return {
            "critique":                   raw[:500] if raw else "Analysis unavailable.",
            "weaknesses":                 [],
            "counter_argument":           "",
            "defense_probability":        0.5,
            "recommended_defense_strategy": "",
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

    def _session_path(self, case_id: str, session_id: str) -> Path:
        return self._case_folder(case_id) / f"session_{session_id}.json"

    def _case_folder(self, case_id: str) -> Path:
        return self.storage_path / "cases" / case_id / "devil_advocate"


# ─────────────────────────────────────────────────────────────────
# Output formatters
# ─────────────────────────────────────────────────────────────────

def format_round(round_obj: Round) -> Dict:
    return {
        "round_number":               round_obj.round_number,
        "critique":                   round_obj.critique,
        "counter_argument":           round_obj.counter_argument,
        "defense_probability":        round_obj.defense_probability,
        "recommended_defense_strategy": round_obj.recommended_defense_strategy,
        "weaknesses":                 round_obj.weaknesses,
        "evidence_count":             len(round_obj.evidence_used),
    }


def format_session_list(sessions: List[Dict]) -> List[List]:
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
