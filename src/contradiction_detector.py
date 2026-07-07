import json
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from src.citation_utils import is_pdf_chunk, build_citation_ref


# ─────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class Statement:
    """A single statement from any source."""
    text:        str
    speaker:     str
    source_type: str
    citation_ref: str
    embedding:   np.ndarray

    start_time:    float = 0.0
    end_time:      float = 0.0
    chunk_id:      str   = ""
    section_title: str   = ""
    original_name: str   = ""
    page_start:    int   = 0
    page_end:      int   = 0


@dataclass
class Contradiction:
    """A verified contradiction between two statements."""
    statement_a: Statement
    statement_b: Statement

    explanation:        str
    severity:           str    # high | medium | low
    severity_score:     int    # 3=high 2=medium 1=low
    conflict_score:     float  # 0.0–1.0
    conflict_type:      str    # location/time/identity/action/factual
    legal_significance: str

    is_cross_source: bool


@dataclass
class ContradictionReport:
    """Full contradiction analysis for a case."""
    case_id: str
    contradictions:           List[Contradiction] = field(default_factory=list)
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
# Source-type exclusion
# ─────────────────────────────────────────────────────────────────

_EXCLUDED_SOURCE_TYPES = {"generated", "brief", "summary"}
_GENERATED_PREFIX      = "DIA-Legal processed"

# ─────────────────────────────────────────────────────────────────
# Pre-compiled patterns for Stage 2
# ─────────────────────────────────────────────────────────────────

_NEG_RE = re.compile(
    r"\b(not|never|no|wasn't|didn't|couldn't|hadn't|isn't|aren't|weren't|"
    r"cannot|can't|won't|wouldn't|denied?|refus\w+|absent|false)\b",
    re.IGNORECASE,
)

_CONFLICT_PAIRS: List[Tuple[str, str]] = [
    (r"\byes\b|\bagreed?\b|\bconfirmed?\b|\badmitted?\b",
     r"\bno\b|\bdenied?\b|\brefused?\b|\brejected?\b"),
    (r"\bpresent\b|\battended?\b|\barrived?\b|\bwas there\b",
     r"\babsent\b|\bnever came\b|\bdidn't come\b|\bnot present\b"),
    (r"\bsaw\b|\bwitnessed?\b|\bobserved?\b|\bnoticed?\b",
     r"\bdidn't see\b|\bnever saw\b|\bdid not see\b|\bdid not witness\b"),
    (r"\bguilty\b|\bresponsible\b|\bcommitted\b",
     r"\binnocent\b|\bnot guilty\b|\bdid not commit\b"),
    (r"\bstruck\b|\bhit\b|\battacked?\b|\bassaulted?\b",
     r"\bdid not (?:strike|hit|attack|assault)\b|\bnever (?:struck|hit)\b"),
    (r"\bwas\b|\bwere\b|\bis\b|\bare\b",
     r"\bwas not\b|\bwere not\b|\bis not\b|\bare not\b|\bwasn't\b|\bweren't\b"),
]

_TIME_RE  = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|o'clock)?\b"
    r"|\b(?:morning|afternoon|evening|night|midnight|noon)\b",
    re.IGNORECASE,
)
_LOC_RE   = re.compile(r"\b(?:at|in|near|inside|outside)\s+([A-Za-z][a-z]{2,})\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_STOPWORDS = {
    "the", "this", "that", "his", "her", "its", "their", "our",
    "my", "your", "any", "some", "one", "two", "also", "even",
}


# ─────────────────────────────────────────────────────────────────
# Contradiction Detector
# ─────────────────────────────────────────────────────────────────

class ContradictionDetector:
    """
    Four-stage contradiction detection pipeline:

    Stage 1 — Pair generation: all pairs from indexed chunks, up to 200.
               No similarity filter.
    Stage 2 — Keyword conflict filter: keeps pairs with at least one
               opposing signal (negation asymmetry, conflict pair
               patterns, time/location divergence, shared entity +
               negation).
    Stage 3 — LLM verification: structured JSON verdict per pair.
    Stage 4 — Cross-source boost: +0.1 to conflict_score when the
               contradiction spans two different documents.
    """

    def __init__(
        self,
        retriever,
        llm_client,
        embedder,
        similarity_threshold: float = 0.3,  # kept for API compat, not used
        top_k: int = 100,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.embedder = embedder
        self.top_k = top_k

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def detect(self, case_id: str) -> ContradictionReport:
        report = ContradictionReport(case_id=case_id)

        all_items = self.retriever.retrieve(
            case_id=case_id,
            query="testimony statement evidence fact location time",
            top_k=self.top_k,
        )
        if not all_items:
            return report

        statements = self._build_statements(all_items)
        report.total_statements_checked = len(statements)

        if len(statements) < 2:
            print(f"[ContradictionDetector] Only {len(statements)} statement(s) — need ≥ 2.", flush=True)
            return report

        # Stage 1: two-pool pair generation — cross-source pairs get a guaranteed quota
        # Naive iteration fills the budget with same-source (video↔video) pairs before
        # any video↔PDF pair is ever considered; the two-pool approach prevents that.
        BUDGET      = 200
        CROSS_QUOTA = BUDGET // 2  # at least 100 slots reserved for cross-source

        cross_pairs: List[Tuple[Statement, Statement, bool]] = []
        same_pairs:  List[Tuple[Statement, Statement, bool]] = []

        for i in range(len(statements)):
            if len(cross_pairs) >= CROSS_QUOTA and len(same_pairs) >= BUDGET:
                break
            for j in range(i + 1, len(statements)):
                same_doc = (
                    statements[i].original_name == statements[j].original_name
                    and statements[i].original_name != ""
                )
                if not same_doc:
                    if len(cross_pairs) < CROSS_QUOTA:
                        cross_pairs.append((statements[i], statements[j], True))
                else:
                    if len(same_pairs) < BUDGET:
                        same_pairs.append((statements[i], statements[j], False))
                if len(cross_pairs) >= CROSS_QUOTA and len(same_pairs) >= BUDGET:
                    break

        # Any unused cross-source quota overflows to same-source
        cross_take = min(len(cross_pairs), CROSS_QUOTA)
        same_take  = min(len(same_pairs),  BUDGET - cross_take)
        all_pairs  = cross_pairs[:cross_take] + same_pairs[:same_take]

        # Stage 2: keyword conflict filter
        suspicious_pairs = [
            (a, b, is_cross)
            for a, b, is_cross in all_pairs
            if self._keyword_conflict_filter(a.text, b.text)
        ]

        report.total_pairs_flagged = len(suspicious_pairs)
        print(
            f"[ContradictionDetector] {len(statements)} statements → "
            f"{len(all_pairs)} pairs ({cross_take} cross-source, {same_take} same-source) → "
            f"{len(suspicious_pairs)} passed keyword filter",
            flush=True,
        )

        if not suspicious_pairs:
            print("[ContradictionDetector] ⚠️ No pairs passed keyword filter.", flush=True)
            return report

        # Stage 3: LLM verification
        for stmt_a, stmt_b, is_cross in suspicious_pairs:
            report.total_llm_calls += 1
            try:
                contradiction = self._verify_contradiction(stmt_a, stmt_b, is_cross)
                if contradiction is not None:
                    report.contradictions.append(contradiction)
            except Exception as exc:
                print(f"[ContradictionDetector] LLM error: {exc}", flush=True)
                report.failed_verifications += 1

        # Deduplicate: keep the highest-scoring contradiction per exhibit-ID pair
        seen: Dict[Tuple[str, str], int] = {}
        deduped: List[Contradiction] = []
        for contra in report.contradictions:
            key_a = contra.statement_a.original_name or contra.statement_a.citation_ref
            key_b = contra.statement_b.original_name or contra.statement_b.citation_ref
            key = tuple(sorted([key_a, key_b]))
            existing_idx = seen.get(key)
            if existing_idx is None:
                seen[key] = len(deduped)
                deduped.append(contra)
            elif contra.conflict_score > deduped[existing_idx].conflict_score:
                deduped[existing_idx] = contra
        report.contradictions = deduped

        # Sort by conflict_score descending
        report.contradictions.sort(key=lambda x: x.conflict_score, reverse=True)
        return report

    # ─────────────────────────────────────────────────────────────
    # Stage 2 — Keyword conflict filter
    # ─────────────────────────────────────────────────────────────

    def _keyword_conflict_filter(self, text_a: str, text_b: str) -> bool:
        a_lower = text_a.lower()
        b_lower = text_b.lower()

        # 1. Negation asymmetry
        a_neg = bool(_NEG_RE.search(a_lower))
        b_neg = bool(_NEG_RE.search(b_lower))
        if a_neg != b_neg:
            return True

        # 2. Direct conflict pair patterns
        for pos_pat, neg_pat in _CONFLICT_PAIRS:
            a_pos = bool(re.search(pos_pat, a_lower, re.IGNORECASE))
            b_neg_ = bool(re.search(neg_pat, b_lower, re.IGNORECASE))
            b_pos = bool(re.search(pos_pat, b_lower, re.IGNORECASE))
            a_neg_ = bool(re.search(neg_pat, a_lower, re.IGNORECASE))
            if (a_pos and b_neg_) or (b_pos and a_neg_):
                return True

        # 3. Time conflict: both mention times that differ
        times_a = set(m.group().lower() for m in _TIME_RE.finditer(a_lower))
        times_b = set(m.group().lower() for m in _TIME_RE.finditer(b_lower))
        if times_a and times_b and not times_a.intersection(times_b):
            return True

        # 4. Location conflict: different locations in both texts
        locs_a = {m.group(1).lower() for m in _LOC_RE.finditer(text_a)} - _STOPWORDS
        locs_b = {m.group(1).lower() for m in _LOC_RE.finditer(text_b)} - _STOPWORDS
        if locs_a and locs_b and not locs_a.intersection(locs_b):
            return True

        # 5. Shared proper noun with negation asymmetry
        proper_a = {w.lower() for w in _PROPER_RE.findall(text_a)} - _STOPWORDS
        proper_b = {w.lower() for w in _PROPER_RE.findall(text_b)} - _STOPWORDS
        if (proper_a & proper_b) and (a_neg != b_neg):
            return True

        return False

    # ─────────────────────────────────────────────────────────────
    # Stage 3 — LLM verification
    # ─────────────────────────────────────────────────────────────

    def _verify_contradiction(
        self,
        stmt_a: Statement,
        stmt_b: Statement,
        is_cross_source: bool,
    ) -> Optional[Contradiction]:
        prompt = self._build_verification_prompt(stmt_a, stmt_b)
        raw = self.llm_client.classify(prompt)
        parsed = self._parse_verification(raw)

        if parsed is None or not parsed["is_contradiction"]:
            return None

        # Stage 4: cross-source boost
        score = min(1.0, parsed["conflict_score"] + (0.1 if is_cross_source else 0.0))

        return Contradiction(
            statement_a        = stmt_a,
            statement_b        = stmt_b,
            explanation        = parsed["explanation"],
            severity           = parsed["severity"],
            severity_score     = {"high": 3, "medium": 2, "low": 1}.get(parsed["severity"], 1),
            conflict_score     = score,
            conflict_type      = parsed["conflict_type"],
            legal_significance = parsed["legal_significance"],
            is_cross_source    = is_cross_source,
        )

    def _build_verification_prompt(self, stmt_a: Statement, stmt_b: Statement) -> str:
        exhibit_a = stmt_a.citation_ref or stmt_a.original_name or stmt_a.source_type
        exhibit_b = stmt_b.citation_ref or stmt_b.original_name or stmt_b.source_type
        ts_a = self._fmt_ts(stmt_a.start_time) if stmt_a.start_time else (f"p.{stmt_a.page_start}" if stmt_a.page_start > 0 else "document")
        ts_b = self._fmt_ts(stmt_b.start_time) if stmt_b.start_time else (f"p.{stmt_b.page_start}" if stmt_b.page_start > 0 else "document")

        return f"""You are a precise legal contradiction detector. Determine whether these two case statements assert mutually exclusive facts.

Statement A [{exhibit_a} · {ts_a}]:
{stmt_a.text}

Statement B [{exhibit_b} · {ts_b}]:
{stmt_b.text}

A real contradiction requires ALL three conditions:
1. Both statements refer to the SAME entity, person, or physical object.
2. Both statements refer to the SAME event, time period, or circumstance.
3. The facts they assert are MUTUALLY EXCLUSIVE — they cannot both be true simultaneously.

Do NOT flag any of these as contradictions:
- Statements about different dates, time periods, or separate incidents.
- A summary or brief that references or paraphrases a source document.
- Statements that are unrelated but happen to share a common word or name.
- Where one statement is a subset, elaboration, or more detailed version of the other.
- Statements expressing different opinions or interpretations of the same fact.

Severity guidelines:
- "high": directly undermines a sworn statement, alibi, or chain of custody — damaging in court.
- "medium": likely factual conflict with clear legal relevance, but some ambiguity remains.
- "low": possible inconsistency that could have an innocent explanation.

Respond with JSON only — no text before or after:
{{
  "is_contradiction": true,
  "severity": "high",
  "conflict_score": 0.85,
  "conflict_type": "factual",
  "explanation": "one sentence explaining the specific mutually exclusive fact",
  "legal_significance": "one sentence on why this matters in court"
}}

conflict_type must be one of: location / time / identity / action / factual"""

    # ─────────────────────────────────────────────────────────────
    # Statement builder
    # ─────────────────────────────────────────────────────────────

    def _build_statements(self, items) -> List[Statement]:
        statements = []

        for item in items:
            meta                  = getattr(item, "metadata", {})
            structured_transcripts = getattr(item, "structured_transcripts", [])
            cite_ref              = build_citation_ref(item)
            original_name         = meta.get("original_name", "")
            source_type           = meta.get("source_type", "video")

            # Skip generated/AI-summary chunks — they paraphrase real evidence and
            # create false contradictions against the originals they summarise.
            if source_type in _EXCLUDED_SOURCE_TYPES:
                continue
            raw_text_preview = (getattr(item, "transcript_text", "") or "")
            if raw_text_preview.startswith(_GENERATED_PREFIX):
                continue
            if not original_name:
                continue

            built_any = False
            for seg in structured_transcripts:
                text = seg.get("text", "").strip()
                if not text or len(text.split()) < 5:
                    continue

                try:
                    embedding = self.embedder.text_embedder.embed(text)
                except Exception:
                    embedding = np.zeros(384, dtype=np.float32)

                built_any = True
                statements.append(Statement(
                    text          = text,
                    speaker       = seg.get("speaker", "UNKNOWN"),
                    source_type   = source_type,
                    citation_ref  = cite_ref,
                    embedding     = embedding,
                    start_time    = seg.get("start", 0.0),
                    end_time      = seg.get("end", 0.0),
                    chunk_id      = (getattr(item, "chunk_ids", [""])[0]
                                     if getattr(item, "chunk_ids", None) else ""),
                    section_title = meta.get("section_title", ""),
                    original_name = original_name,
                    page_start    = meta.get("page_span_start", 0),
                    page_end      = meta.get("page_span_end", 0),
                ))

            # Fallback: use raw transcript_text when structured_transcripts yielded nothing
            if not built_any:
                raw_text = (getattr(item, "transcript_text", "") or "").strip()
                if raw_text and len(raw_text.split()) >= 5:
                    text = raw_text[:512]
                    try:
                        embedding = self.embedder.text_embedder.embed(text)
                    except Exception:
                        embedding = np.zeros(384, dtype=np.float32)

                    statements.append(Statement(
                        text          = text,
                        speaker       = "DOCUMENT",
                        source_type   = source_type,
                        citation_ref  = cite_ref,
                        embedding     = embedding,
                        original_name = original_name,
                        section_title = meta.get("section_title", ""),
                        page_start    = meta.get("page_span_start", 0),
                        page_end      = meta.get("page_span_end", 0),
                    ))

        return statements

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_ts(sec: float) -> str:
        sec = max(0, int(sec))
        h, rem = divmod(sec, 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _parse_verification(self, raw: str) -> Optional[Dict]:
        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        is_contradiction = parsed.get("is_contradiction", False)
        if isinstance(is_contradiction, str):
            is_contradiction = is_contradiction.lower() == "true"

        severity = parsed.get("severity", "low").lower()
        if severity not in ("high", "medium", "low"):
            severity = "low"

        conflict_type = parsed.get("conflict_type", "factual").lower()
        if conflict_type not in ("location", "time", "identity", "action", "factual"):
            conflict_type = "factual"

        try:
            conflict_score = float(parsed.get("conflict_score", 0.5))
            conflict_score = max(0.0, min(1.0, conflict_score))
        except (TypeError, ValueError):
            conflict_score = 0.5

        return {
            "is_contradiction":  bool(is_contradiction),
            "explanation":       str(parsed.get("explanation", "")).strip(),
            "severity":          severity,
            "conflict_score":    conflict_score,
            "conflict_type":     conflict_type,
            "legal_significance": str(parsed.get("legal_significance", "")).strip(),
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# ─────────────────────────────────────────────────────────────────
# Output formatter
# ─────────────────────────────────────────────────────────────────

def format_contradiction_report(report: ContradictionReport) -> Dict:
    severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    rows = []
    for c in report.contradictions:
        emoji = severity_emoji.get(c.severity, "⚪")
        cross = "✅" if c.is_cross_source else "—"
        rows.append([
            f"{emoji} {c.severity.upper()}",
            c.statement_a.citation_ref,
            c.statement_a.text[:120] + ("..." if len(c.statement_a.text) > 120 else ""),
            c.statement_b.citation_ref,
            c.statement_b.text[:120] + ("..." if len(c.statement_b.text) > 120 else ""),
            c.explanation,
            cross,
        ])
    return {
        "headers": ["Severity", "Citation A", "Statement A", "Citation B", "Statement B", "Explanation", "Cross-Source"],
        "rows":    rows,
        "summary": report.summary,
    }
