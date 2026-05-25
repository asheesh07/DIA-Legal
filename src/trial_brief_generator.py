import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, PageBreak
)

# ─────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class WitnessProfile:
    speaker_id:        str
    total_appearances: int
    total_statements:  int

    contradiction_count: int
    hedge_count:         int
    refusal_count:       int
    confident_count:     int
    credibility_score:   float

    inferred_role:        str
    reliability_rating:   str
    key_statements:       List[str]
    recommended_approach: str
    suggested_questions:  List[str]

    first_appearance: str
    last_appearance:  str


@dataclass
class TrialBrief:
    brief_id:        str
    case_id:         str
    lawyer_position: str
    generated_at:    str

    case_summary:        Any        # str or dict with paragraph_1/2/3
    strongest_arguments: List[Dict]
    vulnerabilities:     List[Dict]
    witness_profiles:    List[WitnessProfile]
    contradictions:      List[Dict]
    opposition_strategy: List[str]
    questions_to_ask:    List[str]
    key_risks:           List[str]
    case_strength:       str        # STRONG | MODERATE | WEAK
    recommended_actions: List       # List[str] or List[Dict{action,priority,rationale}]
    overall_assessment:  str = ""
    pdf_path:            str = ""

    # New structured fields
    case_strength_score:           int             = 50
    assessment:                    str             = ""
    key_evidence:                  List[Dict]      = field(default_factory=list)
    contradictions_summary_detail: List[Dict]      = field(default_factory=list)
    cross_examination_detail:      List[Dict]      = field(default_factory=list)
    brief_weaknesses:              List[Dict]      = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "brief_id":        self.brief_id,
            "case_id":         self.case_id,
            "lawyer_position": self.lawyer_position,
            "generated_at":    self.generated_at,
            "case_summary":    self.case_summary,
            "strongest_arguments": self.strongest_arguments,
            "vulnerabilities": self.vulnerabilities,
            "witness_profiles": [
                {
                    "speaker_id":           w.speaker_id,
                    "total_appearances":    w.total_appearances,
                    "total_statements":     w.total_statements,
                    "contradiction_count":  w.contradiction_count,
                    "hedge_count":          w.hedge_count,
                    "credibility_score":    w.credibility_score,
                    "inferred_role":        w.inferred_role,
                    "reliability_rating":   w.reliability_rating,
                    "key_statements":       w.key_statements,
                    "recommended_approach": w.recommended_approach,
                    "suggested_questions":  w.suggested_questions,
                }
                for w in self.witness_profiles
            ],
            "contradictions":            self.contradictions,
            "opposition_strategy":       self.opposition_strategy,
            "questions_to_ask":          self.questions_to_ask,
            "key_risks":                 self.key_risks,
            "case_strength":             self.case_strength,
            "recommended_actions":       self.recommended_actions,
            "overall_assessment":        self.overall_assessment,
            "case_strength_score":       self.case_strength_score,
            "assessment":                self.assessment,
            "key_evidence":              self.key_evidence,
            "contradictions_summary_detail": self.contradictions_summary_detail,
            "cross_examination_detail":  self.cross_examination_detail,
            "brief_weaknesses":          self.brief_weaknesses,
            "pdf_path":                  self.pdf_path,
        }


# ─────────────────────────────────────────────────────────────────
# Hedge / Confident word lists
# ─────────────────────────────────────────────────────────────────

_HEDGE_PHRASES = [
    "i think", "i believe", "maybe", "perhaps", "possibly",
    "i'm not sure", "i don't remember", "i can't recall",
    "i'm not certain", "roughly", "approximately", "sort of",
    "kind of", "i suppose", "if i recall", "to the best of",
]
_CONFIDENT_PHRASES = [
    "definitely", "absolutely", "certainly", "i am sure",
    "without doubt", "clearly", "obviously", "i know",
    "i saw", "i witnessed", "i am certain", "positively",
]
_REFUSAL_PHRASES = [
    "i don't know", "i can't answer", "i refuse",
    "i won't", "no comment", "i cannot say",
    "i don't recall", "i have no recollection",
]

_STRENGTH_SCORE = {"STRONG": 82, "MODERATE": 54, "WEAK": 22}


# ─────────────────────────────────────────────────────────────────
# Trial Brief Generator
# ─────────────────────────────────────────────────────────────────

class TrialBriefGenerator:
    """
    Orchestrates all intelligence modules into a complete pre-trial brief.
    All generated sections are grounded in actual indexed evidence.
    """

    def __init__(
        self,
        retriever,
        llm_client,
        evidence_classifier,
        contradiction_detector,
        devils_advocate,
        storage_path: str = "data",
    ):
        self.retriever              = retriever
        self.llm_client             = llm_client
        self.evidence_classifier    = evidence_classifier
        self.contradiction_detector = contradiction_detector
        self.devils_advocate        = devils_advocate
        self.storage_path           = Path(storage_path)

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def generate(
        self,
        case_id:              str,
        lawyer_position:      str,
        evidence_map=None,
        contradiction_report=None,
    ) -> TrialBrief:

        # Step 1: Run or reuse intelligence modules
        if evidence_map is None:
            evidence_map = self.evidence_classifier.classify(
                case_id=case_id, lawyer_position=lawyer_position,
            )
        if contradiction_report is None:
            contradiction_report = self.contradiction_detector.detect(case_id=case_id)

        # Step 2: Retrieve top chunks for context
        all_items = self.retriever.retrieve(
            case_id=case_id, query=lawyer_position, top_k=20,
        )

        # Step 3: Gather case stats and exhibit IDs
        try:
            case_stats = self.retriever.vector_store.get_case_stats(case_id)
        except Exception:
            case_stats = {"chunk_count": 0, "document_count": 0, "image_count": 0}

        exhibit_ids = list({
            self._extract_cite(item)
            for item in all_items
            if self._extract_cite(item)
        })

        # Step 4: Build witness profiles
        witness_profiles = self._build_witness_profiles(all_items, contradiction_report, case_id)

        # Step 5: Pull devil's advocate opposition history
        da_sessions = self.devils_advocate.list_sessions(case_id)
        opposition_strategy = self._extract_opposition_strategy(da_sessions, case_id)

        # Step 6: One comprehensive LLM call for all brief content
        brief_data = self._generate_brief_content(
            all_items          = all_items,
            evidence_map       = evidence_map,
            contradiction_report = contradiction_report,
            lawyer_position    = lawyer_position,
            case_stats         = case_stats,
            exhibit_ids        = exhibit_ids,
        )

        # Step 7: Format evidence sections from evidence_map
        strongest = [
            {
                "exhibit":      e.citation_ref,
                "citation":     e.citation_ref,
                "source":       e.source_type,
                "strength":     e.strength,
                "reason":       e.reason,
                "text":         e.text[:300],
                "significance": e.reason,
                "summary":      e.text[:150],
            }
            for e in evidence_map.supporting[:5]
        ]
        vulnerabilities = [
            {
                "exhibit":   e.citation_ref,
                "citation":  e.citation_ref,
                "source":    e.source_type,
                "strength":  e.strength,
                "reason":    e.reason,
                "text":      e.text[:300],
                "weakness":  e.reason,
                "mitigation": "",
            }
            for e in evidence_map.opposing[:5]
        ]
        contradictions_fmt = [
            {
                "severity":        c.severity,
                "speaker_a":       c.statement_a.speaker,
                "citation_a":      c.statement_a.citation_ref,
                "statement_a":     c.statement_a.text[:200],
                "speaker_b":       c.statement_b.speaker,
                "citation_b":      c.statement_b.citation_ref,
                "statement_b":     c.statement_b.text[:200],
                "explanation":     c.explanation,
                "conflict_score":  getattr(c, "conflict_score", 0.5),
                "is_cross_source": c.is_cross_source,
            }
            for c in contradiction_report.contradictions
        ]
        contradictions_summary_detail = [
            {
                "exhibits":       f"{c.statement_a.citation_ref} vs {c.statement_b.citation_ref}",
                "conflict_score": round(getattr(c, "conflict_score", 0.5), 2),
                "impact":         c.explanation,
            }
            for c in contradiction_report.contradictions[:5]
        ]

        # Step 8: Extract structured fields from LLM response
        case_summary_raw     = brief_data.get("case_summary", {})
        assessment_text      = brief_data.get("assessment", "")
        case_strength_label  = brief_data.get("case_strength_label", "MODERATE").upper()
        if case_strength_label not in ("STRONG", "MODERATE", "WEAK"):
            case_strength_label = "MODERATE"
        case_strength_score  = brief_data.get("case_strength_score", _STRENGTH_SCORE.get(case_strength_label, 54))
        key_evidence         = brief_data.get("key_evidence", [])
        rec_actions          = brief_data.get("recommended_actions", [])
        cross_exam_detail    = brief_data.get("cross_examination_questions", [])
        brief_weaknesses     = brief_data.get("weaknesses", [])
        key_risks            = brief_data.get("key_risks", [])
        questions_to_ask     = [
            q.get("question", q) if isinstance(q, dict) else q
            for q in cross_exam_detail
        ] or brief_data.get("questions_to_ask", [])

        # Step 9: Build TrialBrief
        brief = TrialBrief(
            brief_id             = str(uuid.uuid4())[:8],
            case_id              = case_id,
            lawyer_position      = lawyer_position,
            generated_at         = datetime.utcnow().isoformat(),
            case_summary         = case_summary_raw,
            strongest_arguments  = strongest,
            vulnerabilities      = vulnerabilities,
            witness_profiles     = witness_profiles,
            contradictions       = contradictions_fmt,
            opposition_strategy  = opposition_strategy,
            questions_to_ask     = questions_to_ask,
            key_risks            = key_risks,
            case_strength        = case_strength_label,
            recommended_actions  = rec_actions,
            overall_assessment   = assessment_text,
            case_strength_score  = int(case_strength_score),
            assessment           = assessment_text,
            key_evidence         = key_evidence,
            contradictions_summary_detail = contradictions_summary_detail,
            cross_examination_detail      = cross_exam_detail,
            brief_weaknesses              = brief_weaknesses,
        )

        # Step 10: Save JSON + export PDF
        self._save_brief(brief)
        pdf_path = self._export_pdf(brief)
        brief.pdf_path = str(pdf_path)
        return brief

    def list_briefs(self, case_id: str) -> List[Dict]:
        folder = self._briefs_folder(case_id)
        if not folder.exists():
            return []
        briefs = []
        for f in sorted(folder.glob("brief_*.json"), reverse=True):
            with open(f) as fp:
                data = json.load(fp)
            briefs.append({
                "brief_id":      data["brief_id"],
                "generated_at":  data["generated_at"][:10],
                "case_strength": data.get("case_strength", "—"),
            })
        return briefs

    # ─────────────────────────────────────────────────────────────
    # Comprehensive LLM call
    # ─────────────────────────────────────────────────────────────

    def _generate_brief_content(
        self,
        all_items,
        evidence_map,
        contradiction_report,
        lawyer_position: str,
        case_stats: Dict,
        exhibit_ids: List[str],
    ) -> Dict:

        # Build evidence context from top 10 chunks
        evidence_ctx = "\n\n".join(
            f"[{self._extract_cite(item)}]: {self._extract_text(item)[:400]}"
            for item in all_items[:10]
            if self._extract_text(item)
        )

        # Contradiction context
        contra_ctx = "\n".join(
            f"- {c.explanation} "
            f"(score={getattr(c, 'conflict_score', '?'):.2f}, "
            f"severity={c.severity}): "
            f"{c.statement_a.citation_ref} vs {c.statement_b.citation_ref}"
            for c in contradiction_report.contradictions[:5]
        ) or "No contradictions detected."

        supporting_count = len(getattr(evidence_map, "supporting", []))
        opposing_count   = len(getattr(evidence_map, "opposing", []))

        prompt = f"""You are a senior trial attorney preparing a comprehensive pre-trial brief.

LAWYER'S POSITION: {lawyer_position}

CASE STATS:
- Chunks indexed: {case_stats.get('chunk_count', 0)}
- Documents: {case_stats.get('document_count', 0)}
- Exhibit IDs available: {', '.join(exhibit_ids[:10]) or 'not labeled'}

EVIDENCE SUMMARY:
- Supporting the position: {supporting_count} items
- Opposing the position: {opposing_count} items

KEY CONTRADICTIONS IN EVIDENCE:
{contra_ctx}

EVIDENCE EXCERPTS (top 10 chunks):
{evidence_ctx[:4000]}

Return ONLY this JSON (no other text):
{{
  "case_strength_score": 68,
  "case_strength_label": "MODERATE",
  "assessment": "Two-sentence assessment grounded in the actual evidence above.",
  "case_summary": {{
    "paragraph_1": "What this case is about — factual overview.",
    "paragraph_2": "Key facts established by the indexed evidence.",
    "paragraph_3": "The central legal question and why it matters."
  }},
  "key_evidence": [
    {{"exhibit": "cite ref", "summary": "what it shows", "significance": "why it matters in court"}},
    {{"exhibit": "cite ref", "summary": "what it shows", "significance": "why it matters in court"}}
  ],
  "key_risks": [
    "risk 1",
    "risk 2"
  ],
  "recommended_actions": [
    {{"action": "concrete action", "priority": "high", "rationale": "why"}},
    {{"action": "concrete action", "priority": "medium", "rationale": "why"}}
  ],
  "cross_examination_questions": [
    {{"question": "question text", "target_exhibit": "cite ref", "strategic_goal": "what this exposes"}},
    {{"question": "question text", "target_exhibit": "cite ref", "strategic_goal": "what this exposes"}}
  ],
  "weaknesses": [
    {{"weakness": "specific weakness", "mitigation": "how to address it"}},
    {{"weakness": "specific weakness", "mitigation": "how to address it"}}
  ]
}}"""

        raw = self.llm_client.classify(prompt)
        return self._parse_json_safe(raw)

    # ─────────────────────────────────────────────────────────────
    # Witness Profiling
    # ─────────────────────────────────────────────────────────────

    def _build_witness_profiles(self, items, contradiction_report, case_id: str) -> List[WitnessProfile]:
        by_speaker: Dict[str, List] = {}
        for item in items:
            for seg in item.structured_transcripts:
                speaker = seg.get("speaker", "")
                if not speaker or speaker == "DOCUMENT":
                    continue
                if speaker not in by_speaker:
                    by_speaker[speaker] = []
                by_speaker[speaker].append({
                    "text":     seg.get("text", ""),
                    "start":    seg.get("start", 0.0),
                    "end":      seg.get("end", 0.0),
                    "chunk_id": item.chunk_ids[0] if item.chunk_ids else "",
                })

        if not by_speaker:
            return []

        contradiction_counts: Dict[str, int] = {}
        for c in contradiction_report.contradictions:
            spk = c.statement_a.speaker
            contradiction_counts[spk] = contradiction_counts.get(spk, 0) + 1

        profiles = []
        for speaker, statements in by_speaker.items():
            all_text = " ".join(s["text"].lower() for s in statements)

            hedge_count      = sum(all_text.count(p) for p in _HEDGE_PHRASES)
            confident_count  = sum(all_text.count(p) for p in _CONFIDENT_PHRASES)
            refusal_count    = sum(all_text.count(p) for p in _REFUSAL_PHRASES)
            contradiction_count = contradiction_counts.get(speaker, 0)

            score = self._compute_credibility(
                contradiction_count, hedge_count, refusal_count, confident_count, len(statements),
            )

            sorted_stmts = sorted(statements, key=lambda x: x["start"])
            first = sorted_stmts[0]
            last  = sorted_stmts[-1]

            def fmt_time(t):
                m, s = int(t) // 60, int(t) % 60
                return f"{m:02d}:{s:02d}"

            llm_analysis = self._analyze_witness_llm(
                speaker=speaker, statements=statements,
                contradiction_count=contradiction_count, credibility_score=score,
            )

            profiles.append(WitnessProfile(
                speaker_id           = speaker,
                total_appearances    = len({s["chunk_id"] for s in statements}),
                total_statements     = len(statements),
                contradiction_count  = contradiction_count,
                hedge_count          = hedge_count,
                refusal_count        = refusal_count,
                confident_count      = confident_count,
                credibility_score    = score,
                inferred_role        = llm_analysis.get("role", "Unknown"),
                reliability_rating   = llm_analysis.get("reliability", "NEUTRAL"),
                key_statements       = llm_analysis.get("key_statements", []),
                recommended_approach = llm_analysis.get("recommended_approach", ""),
                suggested_questions  = llm_analysis.get("suggested_questions", []),
                first_appearance     = f"[{fmt_time(first['start'])}]",
                last_appearance      = f"[{fmt_time(last['start'])}]",
            ))

        profiles.sort(key=lambda x: x.credibility_score)
        return profiles

    def _compute_credibility(
        self, contradictions: int, hedges: int, refusals: int, confident: int, total: int,
    ) -> float:
        score = 5.0
        score -= contradictions * 1.0
        score -= min(hedges, 10) * 0.1
        score -= refusals * 0.5
        if total > 0 and hedges / total > 0.3:
            score -= 0.5
        score += min(confident, 5) * 0.1
        return round(max(0.0, min(10.0, score)), 1)

    def _analyze_witness_llm(
        self, speaker: str, statements: List[Dict],
        contradiction_count: int, credibility_score: float,
    ) -> Dict:
        sample = "\n".join(f'- "{s["text"][:150]}"' for s in statements[:8])
        prompt = f"""You are a legal analyst profiling a witness.

WITNESS: {speaker}
CREDIBILITY SCORE: {credibility_score}/10
CONTRADICTIONS FOUND: {contradiction_count}

SAMPLE STATEMENTS:
{sample}

Analyze this witness and respond ONLY in this JSON:
{{
    "role": "Prosecution Witness / Defense Witness / Expert Witness / Judge / Unknown",
    "reliability": "HOSTILE / NEUTRAL / FRIENDLY",
    "key_statements": ["most important thing 1", "most important thing 2"],
    "recommended_approach": "one sentence on how lawyer should handle this witness",
    "suggested_questions": ["question 1", "question 2", "question 3"]
}}

- reliability is from the DEFENSE perspective
- No text outside the JSON"""

        raw = self.llm_client.classify(prompt)
        return self._parse_json_safe(raw)

    def _extract_opposition_strategy(self, sessions: List[Dict], case_id: str) -> List[str]:
        args = []
        for meta in sessions:
            try:
                session = self.devils_advocate.load_session(case_id, meta["session_id"])
                for r in session.rounds:
                    if r.counter_argument:
                        args.append(r.counter_argument)
                    args.extend(r.opposition_args)
            except Exception:
                continue
        seen: set = set()
        unique: List[str] = []
        for a in args:
            key = str(a)[:80].lower()
            if key not in seen:
                seen.add(key)
                unique.append(str(a))
        return unique[:10]

    # ─────────────────────────────────────────────────────────────
    # PDF Export
    # ─────────────────────────────────────────────────────────────

    def _export_pdf(self, brief: TrialBrief) -> Path:
        folder   = self._briefs_folder(brief.case_id)
        pdf_path = folder / f"brief_{brief.brief_id}.pdf"

        doc = SimpleDocTemplate(
            str(pdf_path), pagesize=letter,
            rightMargin=inch, leftMargin=inch,
            topMargin=inch, bottomMargin=inch,
        )
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle("BriefTitle", parent=styles["Title"],
            fontSize=20, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6)
        heading_style = ParagraphStyle("BriefHeading", parent=styles["Heading1"],
            fontSize=13, textColor=colors.HexColor("#16213e"), spaceBefore=16, spaceAfter=6)
        subheading_style = ParagraphStyle("BriefSubHeading", parent=styles["Heading2"],
            fontSize=11, textColor=colors.HexColor("#0f3460"), spaceBefore=10, spaceAfter=4)
        body_style = ParagraphStyle("BriefBody", parent=styles["Normal"],
            fontSize=10, leading=14, spaceAfter=6)
        bullet_style = ParagraphStyle("BriefBullet", parent=styles["Normal"],
            fontSize=10, leading=14, leftIndent=20, spaceAfter=4)
        meta_style = ParagraphStyle("BriefMeta", parent=styles["Normal"],
            fontSize=9, textColor=colors.grey, spaceAfter=4)

        story = []
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph("PRE-TRIAL BRIEF", title_style))
        story.append(Paragraph(f"Case ID: {brief.case_id}", meta_style))
        story.append(Paragraph(f"Generated: {brief.generated_at[:10]}", meta_style))
        story.append(Paragraph(f"Lawyer's Position: {brief.lawyer_position}", meta_style))

        strength_color = {"STRONG": "#28a745", "MODERATE": "#ffc107", "WEAK": "#dc3545"}.get(
            brief.case_strength, "#6c757d"
        )
        story.append(Paragraph(
            f'<font color="{strength_color}"><b>Case Strength: {brief.case_strength} '
            f'({brief.case_strength_score}/100)</b></font>', body_style
        ))
        if brief.assessment:
            story.append(Paragraph(brief.assessment, body_style))

        story.append(HRFlowable(width="100%", thickness=2,
            color=colors.HexColor("#1a1a2e"), spaceAfter=12))

        # Section 1: Case Summary
        story.append(Paragraph("1. CASE SUMMARY", heading_style))
        summary_paras = self._render_summary_paras(brief.case_summary)
        for para in summary_paras:
            story.append(Paragraph(para, body_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=8))

        # Section 2: Strongest Arguments
        story.append(Paragraph("2. STRONGEST ARGUMENTS", heading_style))
        for i, arg in enumerate(brief.strongest_arguments, 1):
            story.append(Paragraph(
                f"<b>[{i}] Strength {arg.get('strength', '?')}/5 — {arg.get('citation', '')}</b>",
                subheading_style))
            story.append(Paragraph(arg.get("reason", ""), bullet_style))
            text = arg.get("text", "")
            if text:
                story.append(Paragraph(f'<i>"{text[:200]}..."</i>', meta_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=8))

        # Section 3: Vulnerabilities
        story.append(Paragraph("3. VULNERABILITIES", heading_style))
        for i, v in enumerate(brief.vulnerabilities, 1):
            story.append(Paragraph(
                f"<b>[{i}] Risk {v.get('strength', '?')}/5 — {v.get('citation', '')}</b>",
                subheading_style))
            story.append(Paragraph(v.get("reason", ""), bullet_style))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=8))
        story.append(PageBreak())

        # Section 4: Witness Profiles
        story.append(Paragraph("4. WITNESS PROFILES", heading_style))
        for w in brief.witness_profiles:
            rating_color = {"HOSTILE": "#dc3545", "NEUTRAL": "#6c757d", "FRIENDLY": "#28a745"}.get(
                w.reliability_rating, "#6c757d"
            )
            story.append(Paragraph(
                f'<b>{w.speaker_id}</b> — <font color="{rating_color}">{w.reliability_rating}</font> — '
                f'Credibility: {w.credibility_score}/10 — {w.inferred_role}', subheading_style))
            story.append(Paragraph(
                f"Statements: {w.total_statements} | Contradictions: {w.contradiction_count} | "
                f"First: {w.first_appearance} | Last: {w.last_appearance}", meta_style))
            if w.recommended_approach:
                story.append(Paragraph(f"<b>Approach:</b> {w.recommended_approach}", bullet_style))
            for q in w.suggested_questions[:3]:
                story.append(Paragraph(f"• {q}", bullet_style))
            story.append(Spacer(1, 0.1 * inch))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=8))

        # Section 5: Contradictions
        story.append(Paragraph("5. WITNESS CONTRADICTIONS", heading_style))
        if not brief.contradictions:
            story.append(Paragraph("No contradictions detected.", body_style))
        else:
            sev_colors = {"high": "#dc3545", "medium": "#ffc107", "low": "#28a745"}
            for i, c in enumerate(brief.contradictions, 1):
                sc = sev_colors.get(c["severity"], "#6c757d")
                story.append(Paragraph(
                    f'<b>[{i}] <font color="{sc}">{c["severity"].upper()}</font></b> — {c["explanation"]}',
                    subheading_style))
                story.append(Paragraph(
                    f"<b>A</b> {c['citation_a']}: <i>\"{c['statement_a'][:150]}\"</i>", bullet_style))
                story.append(Paragraph(
                    f"<b>B</b> {c['citation_b']}: <i>\"{c['statement_b'][:150]}\"</i>", bullet_style))
                if c.get("is_cross_source"):
                    story.append(Paragraph("⚡ Cross-source contradiction", meta_style))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=8))
        story.append(PageBreak())

        # Section 6: Opposition Strategy
        story.append(Paragraph("6. OPPOSITION STRATEGY", heading_style))
        if not brief.opposition_strategy:
            story.append(Paragraph(
                "No devil's advocate sessions found. Run the Devil's Advocate tab to populate this section.",
                body_style))
        else:
            for i, arg in enumerate(brief.opposition_strategy, 1):
                story.append(Paragraph(f"{i}. {arg}", bullet_style))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=8))

        # Section 7: Cross-Examination Questions
        story.append(Paragraph("7. CROSS-EXAMINATION QUESTIONS", heading_style))
        for i, q in enumerate(brief.cross_examination_detail or brief.questions_to_ask, 1):
            if isinstance(q, dict):
                story.append(Paragraph(
                    f"<b>[{i}]</b> {q.get('question', '')} "
                    f"<i>[{q.get('target_exhibit', '')}]</i>", bullet_style))
                if q.get("strategic_goal"):
                    story.append(Paragraph(f"Goal: {q['strategic_goal']}", meta_style))
            else:
                story.append(Paragraph(f"{i}. {q}", bullet_style))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=8))

        # Section 8: Key Risks & Recommended Actions
        story.append(Paragraph("8. KEY RISKS & ACTIONS", heading_style))
        if brief.key_risks:
            story.append(Paragraph("<b>Key Risks:</b>", body_style))
            for risk in brief.key_risks:
                story.append(Paragraph(f"• {risk}", bullet_style))
        if brief.recommended_actions:
            story.append(Paragraph("<b>Recommended Actions:</b>", body_style))
            for action in brief.recommended_actions:
                if isinstance(action, dict):
                    pri = action.get("priority", "")
                    story.append(Paragraph(
                        f"[{pri.upper()}] {action.get('action', '')} — {action.get('rationale', '')}",
                        bullet_style))
                else:
                    story.append(Paragraph(f"⚠ {action}", bullet_style))

        doc.build(story)
        return pdf_path

    # ─────────────────────────────────────────────────────────────
    # Storage
    # ─────────────────────────────────────────────────────────────

    def _save_brief(self, brief: TrialBrief) -> None:
        folder = self._briefs_folder(brief.case_id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"brief_{brief.brief_id}.json"
        with open(path, "w") as f:
            json.dump(brief.to_dict(), f, indent=2)

    def _briefs_folder(self, case_id: str) -> Path:
        return self.storage_path / "cases" / case_id / "briefs"

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _extract_text(self, item) -> str:
        texts = [
            seg.get("text", "").strip()
            for seg in getattr(item, "structured_transcripts", [])
            if seg.get("text", "").strip()
        ]
        if texts:
            return " ".join(texts)
        return (getattr(item, "transcript_text", "") or "").strip()[:400]

    def _extract_cite(self, item) -> str:
        from src.citation_utils import build_citation_ref
        try:
            return build_citation_ref(item)
        except Exception:
            return ""

    def _render_summary_paras(self, summary) -> List[str]:
        """Convert case_summary (str or dict) into a list of paragraphs."""
        if not summary:
            return []
        obj = None
        if isinstance(summary, str):
            try:
                obj = json.loads(summary)
            except Exception:
                pass
        elif isinstance(summary, dict):
            obj = summary
        if obj and isinstance(obj, dict):
            paras = [obj.get("paragraph_1", ""), obj.get("paragraph_2", ""), obj.get("paragraph_3", "")]
            paras = [p.get("text", p) if isinstance(p, dict) else p for p in paras]
            paras = [p.strip() for p in paras if p and str(p).strip()]
            if paras:
                return paras
        return str(summary).split("\n\n")

    def _parse_json_safe(self, raw: str) -> Dict:
        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}
