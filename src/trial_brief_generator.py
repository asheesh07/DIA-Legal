import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

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
    speaker_id:       str
    total_appearances: int
    total_statements:  int

    # Statistical signals (no LLM)
    contradiction_count: int
    hedge_count:         int       # "I think", "maybe", "perhaps"
    refusal_count:       int       # "I don't know", "I can't recall"
    confident_count:     int       # "definitely", "absolutely"
    credibility_score:   float     # 0-10

    # LLM analysis
    inferred_role:       str       # Prosecution/Defense/Expert/Judge
    reliability_rating:  str       # HOSTILE | NEUTRAL | FRIENDLY
    key_statements:      List[str] # Most important things they said
    recommended_approach: str      # How lawyer should handle them
    suggested_questions:  List[str]# Questions to ask in cross

    # Timeline
    first_appearance: str          # Citation ref
    last_appearance:  str

@dataclass
class TrialBrief:
    brief_id:        str
    case_id:         str
    lawyer_position: str
    generated_at:    str

    # Sections
    case_summary:        str
    strongest_arguments: List[Dict]
    vulnerabilities:     List[Dict]
    witness_profiles:    List[WitnessProfile]
    contradictions:      List[Dict]
    opposition_strategy: List[str]
    recommended_questions: List[str]
    overall_assessment:  str
    case_strength:       str       # STRONG | MODERATE | WEAK
    critical_actions:    List[str]

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
                    "speaker_id":         w.speaker_id,
                    "total_appearances":  w.total_appearances,
                    "total_statements":   w.total_statements,
                    "contradiction_count": w.contradiction_count,
                    "hedge_count":        w.hedge_count,
                    "credibility_score":  w.credibility_score,
                    "inferred_role":      w.inferred_role,
                    "reliability_rating": w.reliability_rating,
                    "key_statements":     w.key_statements,
                    "recommended_approach": w.recommended_approach,
                    "suggested_questions": w.suggested_questions,
                }
                for w in self.witness_profiles
            ],
            "contradictions":         self.contradictions,
            "opposition_strategy":    self.opposition_strategy,
            "recommended_questions":  self.recommended_questions,
            "overall_assessment":     self.overall_assessment,
            "case_strength":          self.case_strength,
            "critical_actions":       self.critical_actions,
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
# ─────────────────────────────────────────────────────────────────
# Trial Brief Generator
# ─────────────────────────────────────────────────────────────────

class TrialBriefGenerator:
    """
    Orchestrates all intelligence modules into a complete
    pre-trial brief with witness profiles and PDF export.

    Sections:
    1. Case Summary
    2. Strongest Arguments
    3. Vulnerabilities
    4. Witness Profiles
    5. Contradictions
    6. Opposition Strategy
    7. Recommended Questions
    8. Overall Assessment
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
        self.retriever               = retriever
        self.llm_client              = llm_client
        self.evidence_classifier     = evidence_classifier
        self.contradiction_detector  = contradiction_detector
        self.devils_advocate         = devils_advocate
        self.storage_path            = Path(storage_path)

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────
    def generate(
        self,
        case_id:         str,
        lawyer_position: str,
        evidence_map=None,
        contradiction_report=None,
    ) -> TrialBrief:
        """
        Generate a complete trial brief.

        Args:
            case_id:              Case to brief.
            lawyer_position:      Lawyer's position statement.
            evidence_map:         Pass cached EvidenceMap or None
                                  to run fresh.
            contradiction_report: Pass cached ContradictionReport
                                  or None to run fresh.

        Returns:
            TrialBrief saved to disk + PDF exported.
        """

        # ── Step 1: Run or reuse intelligence modules ─────────────
        if evidence_map is None:
            evidence_map = self.evidence_classifier.classify(
                case_id=case_id,
                lawyer_position=lawyer_position,
            )

        if contradiction_report is None:
            contradiction_report = self.contradiction_detector.detect(
                case_id=case_id
            )

        # ── Step 2: Retrieve chunks for context ───────────────────
        all_items = self.retriever.retrieve(
            case_id=case_id,
            query=lawyer_position,
            top_k=20,
        )

        # ── Step 3: Build witness profiles ────────────────────────
        witness_profiles = self._build_witness_profiles(
            all_items, contradiction_report, case_id
        )

        # ── Step 4: Pull devil's advocate history ─────────────────
        da_sessions = self.devils_advocate.list_sessions(case_id)
        opposition_strategy = self._extract_opposition_strategy(
            da_sessions, case_id
        )

        # ── Step 5: LLM Call 1 — Case Summary ────────────────────
        case_summary = self._generate_case_summary(
            all_items, lawyer_position
        )

        # ── Step 6: LLM Call 2 — Recommended Questions ───────────
        recommended_questions = self._generate_questions(
            contradiction_report, witness_profiles
        )

        # ── Step 7: LLM Call 3 — Overall Assessment ──────────────
        assessment, case_strength, critical_actions = (
            self._generate_assessment(
                evidence_map,
                contradiction_report,
                witness_profiles,
                lawyer_position,
            )
        )
        # ── Step 8: Format evidence sections ─────────────────────
        strongest = [
            {
                "strength":    e.strength,
                "citation":    e.citation_ref,
                "source":      e.source_type,
                "reason":      e.reason,
                "text":        e.text[:300],
            }
            for e in evidence_map.supporting[:5]
        ]

        vulnerabilities = [
            {
                "strength":    e.strength,
                "citation":    e.citation_ref,
                "source":      e.source_type,
                "reason":      e.reason,
                "text":        e.text[:300],
            }
            for e in evidence_map.opposing[:5]
        ]

        contradictions = [
            {
                "severity":       c.severity,
                "speaker_a":      c.statement_a.speaker,
                "citation_a":     c.statement_a.citation_ref,
                "statement_a":    c.statement_a.text[:200],
                "speaker_b":      c.statement_b.speaker,
                "citation_b":     c.statement_b.citation_ref,
                "statement_b":    c.statement_b.text[:200],
                "explanation":    c.explanation,
                "is_cross_source": c.is_cross_source,
            }
            for c in contradiction_report.contradictions
        ]

        # ── Step 9: Build TrialBrief ──────────────────────────────
        brief = TrialBrief(
            brief_id         = str(uuid.uuid4())[:8],
            case_id          = case_id,
            lawyer_position  = lawyer_position,
            generated_at     = datetime.utcnow().isoformat(),
            case_summary     = case_summary,
            strongest_arguments = strongest,
            vulnerabilities  = vulnerabilities,
            witness_profiles = witness_profiles,
            contradictions   = contradictions,
            opposition_strategy = opposition_strategy,
            recommended_questions = recommended_questions,
            overall_assessment = assessment,
            case_strength    = case_strength,
            critical_actions = critical_actions,
        )

        # ── Step 10: Save JSON + export PDF ───────────────────────
        self._save_brief(brief)
        pdf_path = self._export_pdf(brief)
        brief.pdf_path = str(pdf_path)

        return brief

    def list_briefs(self, case_id: str) -> List[Dict]:
        """List all briefs generated for a case."""
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
    # Witness Profiling
    # ─────────────────────────────────────────────────────────────

    def _build_witness_profiles(
        self,
        items,
        contradiction_report,
        case_id: str,
    ) -> List[WitnessProfile]:

        # Group statements by speaker
        by_speaker: Dict[str, List] = {}

        for item in items:
            for seg in item.structured_transcripts:
                speaker = seg.get("speaker", "")
                if not speaker or speaker == "DOCUMENT":
                    continue
                if speaker not in by_speaker:
                    by_speaker[speaker] = []
                by_speaker[speaker].append({
                    "text":      seg.get("text", ""),
                    "start":     seg.get("start", 0.0),
                    "end":       seg.get("end", 0.0),
                    "chunk_id":  item.chunk_ids[0]
                                 if item.chunk_ids else "",
                })

        if not by_speaker:
            return []

        # Count contradictions per speaker
        contradiction_counts: Dict[str, int] = {}
        for c in contradiction_report.contradictions:
            spk = c.statement_a.speaker
            contradiction_counts[spk] = (
                contradiction_counts.get(spk, 0) + 1
            )

        profiles = []
        for speaker, statements in by_speaker.items():

            # Statistical analysis — no LLM
            all_text = " ".join(
                s["text"].lower() for s in statements
            )

            hedge_count = sum(
                all_text.count(p) for p in _HEDGE_PHRASES
            )
            confident_count = sum(
                all_text.count(p) for p in _CONFIDENT_PHRASES
            )
            refusal_count = sum(
                all_text.count(p) for p in _REFUSAL_PHRASES
            )
            contradiction_count = contradiction_counts.get(
                speaker, 0
            )

            # Credibility score
            score = self._compute_credibility(
                contradiction_count,
                hedge_count,
                refusal_count,
                confident_count,
                len(statements),
            )

            # First and last appearance
            sorted_stmts = sorted(
                statements, key=lambda x: x["start"]
            )
            first = sorted_stmts[0]
            last  = sorted_stmts[-1]

            def fmt_time(t):
                m, s = int(t) // 60, int(t) % 60
                return f"{m:02d}:{s:02d}"

            first_ref = f"[{fmt_time(first['start'])}]"
            last_ref  = f"[{fmt_time(last['start'])}]"

            # LLM analysis — one call per witness
            llm_analysis = self._analyze_witness_llm(
                speaker=speaker,
                statements=statements,
                contradiction_count=contradiction_count,
                credibility_score=score,
            )

            profile = WitnessProfile(
                speaker_id        = speaker,
                total_appearances = len(set(
                    s["chunk_id"] for s in statements
                )),
                total_statements  = len(statements),
                contradiction_count = contradiction_count,
                hedge_count       = hedge_count,
                refusal_count     = refusal_count,
                confident_count   = confident_count,
                credibility_score = score,
                inferred_role     = llm_analysis.get(
                    "role", "Unknown"
                ),
                reliability_rating = llm_analysis.get(
                    "reliability", "NEUTRAL"
                ),
                key_statements    = llm_analysis.get(
                    "key_statements", []
                ),
                recommended_approach = llm_analysis.get(
                    "recommended_approach", ""
                ),
                suggested_questions  = llm_analysis.get(
                    "suggested_questions", []
                ),
                first_appearance  = first_ref,
                last_appearance   = last_ref,
            )
            profiles.append(profile)

        # Sort by credibility ascending — least credible first
        profiles.sort(key=lambda x: x.credibility_score)
        return profiles

    def _compute_credibility(
        self,
        contradictions: int,
        hedges:         int,
        refusals:       int,
        confident:      int,
        total:          int,
    ) -> float:

        score = 5.0

        score -= contradictions * 1.0
        score -= min(hedges, 10) * 0.1
        score -= refusals * 0.5

        # Normalize hedge/confident relative to total statements
        if total > 0:
            hedge_ratio = hedges / total
            if hedge_ratio > 0.3:
                score -= 0.5

        score += min(confident, 5) * 0.1
        return round(max(0.0, min(10.0, score)), 1)

    def _analyze_witness_llm(
        self,
        speaker:             str,
        statements:          List[Dict],
        contradiction_count: int,
        credibility_score:   float,
    ) -> Dict:

        sample_texts = "\n".join(
            f"- \"{s['text'][:150]}\""
            for s in statements[:8]
        )

        prompt = f"""You are a legal analyst profiling a witness.

WITNESS: {speaker}
CREDIBILITY SCORE: {credibility_score}/10
CONTRADICTIONS FOUND: {contradiction_count}

SAMPLE STATEMENTS:
{sample_texts}

Analyze this witness and respond ONLY in this JSON:
{{
    "role": "Prosecution Witness / Defense Witness / Expert Witness / Judge / Unknown",
    "reliability": "HOSTILE / NEUTRAL / FRIENDLY",
    "key_statements": [
        "most important thing they said 1",
        "most important thing they said 2",
        "most important thing they said 3"
    ],
    "recommended_approach": "one sentence on how lawyer should handle this witness",
    "suggested_questions": [
        "question to ask in cross examination 1",
        "question to ask in cross examination 2",
        "question to ask in cross examination 3"
    ]
}}

Rules:
- reliability is from the DEFENSE perspective
- key_statements must be direct quotes or close paraphrases
- suggested_questions must expose weaknesses or contradictions
- No text outside the JSON"""

        raw = self.llm_client.classify(prompt)
        return self._parse_json_safe(raw)

    # ─────────────────────────────────────────────────────────────
    # LLM Calls
    # ─────────────────────────────────────────────────────────────

    def _generate_case_summary(
        self, items, lawyer_position: str
    ) -> str:

        context = "\n\n".join(
            self._extract_text(item)
            for item in items[:10]
            if self._extract_text(item)
        )

        prompt = f"""You are a senior legal analyst.

LAWYER'S POSITION:
{lawyer_position}

EVIDENCE AND TESTIMONY (sample):
{context[:3000]}

Write a 3-paragraph case summary:
Paragraph 1: What this case is about
Paragraph 2: Key facts established by evidence
Paragraph 3: The central legal question

Be factual. No speculation. Plain professional language."""

        try:
            raw = self.llm_client.classify(prompt)
            return raw.strip()
        except Exception:
            return "Case summary unavailable."

    def _generate_questions(
        self,
        contradiction_report,
        witness_profiles: List[WitnessProfile],
    ) -> List[str]:

        contradiction_summary = "\n".join(
            f"- {c['explanation']} "
            f"({c['citation_a']} vs {c['citation_b']})"
            for c in [
                {
                    "explanation": c.explanation,
                    "citation_a":  c.statement_a.citation_ref,
                    "citation_b":  c.statement_b.citation_ref,
                }
                for c in contradiction_report.contradictions[:5]
            ]
        ) or "No contradictions found."

        hostile_witnesses = [
            w.speaker_id for w in witness_profiles
            if w.reliability_rating == "HOSTILE"
        ]

        prompt = f"""You are a senior trial lawyer.

CONTRADICTIONS FOUND:
{contradiction_summary}

HOSTILE WITNESSES: {', '.join(hostile_witnesses) or 'None identified'}

Generate 7 cross-examination questions that:
- Expose the contradictions directly
- Target hostile witnesses
- Are specific and legally precise

Respond ONLY in JSON:
{{
    "questions": [
        "question 1",
        "question 2",
        "question 3",
        "question 4",
        "question 5",
        "question 6",
        "question 7"
    ]
}}"""

        raw = self.llm_client.classify(prompt)
        parsed = self._parse_json_safe(raw)
        return parsed.get("questions", [])

    def _generate_assessment(
        self,
        evidence_map,
        contradiction_report,
        witness_profiles,
        lawyer_position: str,
    ) -> tuple:

        supporting = len(evidence_map.supporting)
        opposing   = len(evidence_map.opposing)
        contras    = len(contradiction_report.contradictions)
        hostile    = sum(
            1 for w in witness_profiles
            if w.reliability_rating == "HOSTILE"
        )

        prompt = f"""You are a senior trial lawyer assessing case strength.

LAWYER'S POSITION: {lawyer_position}

EVIDENCE SUMMARY:
- Supporting evidence pieces: {supporting}
- Opposing evidence pieces:   {opposing}
- Contradictions found:       {contras}
- Hostile witnesses:          {hostile}

Provide overall assessment in JSON:
{{
    "assessment": "2-3 sentence overall case assessment",
    "case_strength": "STRONG / MODERATE / WEAK",
    "critical_actions": [
        "most important action before court 1",
        "most important action before court 2",
        "most important action before court 3"
    ]
}}"""

        raw = self.llm_client.classify(prompt)
        parsed = self._parse_json_safe(raw)

        assessment = parsed.get("assessment", "Assessment unavailable.")
        strength   = parsed.get("case_strength", "MODERATE")
        actions    = parsed.get("critical_actions", [])

        if strength not in ("STRONG", "MODERATE", "WEAK"):
            strength = "MODERATE"

        return assessment, strength, actions

    def _extract_opposition_strategy(
        self, sessions: List[Dict], case_id: str
    ) -> List[str]:
        """Pull opposition arguments from devil's advocate history."""
        opposition_args = []

        for session_meta in sessions:
            try:
                session = self.devils_advocate.load_session(
                    case_id, session_meta["session_id"]
                )
                for r in session.rounds:
                    opposition_args.extend(r.opposition_args)
            except Exception:
                continue

        # Deduplicate
        seen = set()
        unique = []
        for arg in opposition_args:
            key = arg[:80].lower()
            if key not in seen:
                seen.add(key)
                unique.append(arg)

        return unique[:10]

    # ─────────────────────────────────────────────────────────────
    # PDF Export
    # ─────────────────────────────────────────────────────────────

    def _export_pdf(self, brief: TrialBrief) -> Path:

        folder   = self._briefs_folder(brief.case_id)
        pdf_path = folder / f"brief_{brief.brief_id}.pdf"

        doc    = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch,
        )

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "BriefTitle",
            parent=styles["Title"],
            fontSize=20,
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=6,
        )
        heading_style = ParagraphStyle(
            "BriefHeading",
            parent=styles["Heading1"],
            fontSize=13,
            textColor=colors.HexColor("#16213e"),
            spaceBefore=16,
            spaceAfter=6,
            borderPad=4,
        )
        subheading_style = ParagraphStyle(
            "BriefSubHeading",
            parent=styles["Heading2"],
            fontSize=11,
            textColor=colors.HexColor("#0f3460"),
            spaceBefore=10,
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "BriefBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
        )
        bullet_style = ParagraphStyle(
            "BriefBullet",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            leftIndent=20,
            spaceAfter=4,
        )
        meta_style = ParagraphStyle(
            "BriefMeta",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.grey,
            spaceAfter=4,
        )
        story = []
        # ── Cover ────────────────────────────────────────────────
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph("PRE-TRIAL BRIEF", title_style))
        story.append(Paragraph(
            f"Case ID: {brief.case_id}", meta_style
        ))
        story.append(Paragraph(
            f"Generated: {brief.generated_at[:10]}", meta_style
        ))
        story.append(Paragraph(
            f"Lawyer's Position: {brief.lawyer_position}",
            meta_style
        ))

        strength_color = {
            "STRONG":   "#28a745",
            "MODERATE": "#ffc107",
            "WEAK":     "#dc3545",
        }.get(brief.case_strength, "#6c757d")

        story.append(Paragraph(
            f'<font color="{strength_color}"><b>'
            f'Case Strength: {brief.case_strength}'
            f'</b></font>',
            body_style
        ))
        story.append(HRFlowable(
            width="100%", thickness=2,
            color=colors.HexColor("#1a1a2e"),
            spaceAfter=12
        ))
        # ── Section 1: Case Summary ───────────────────────────────
        story.append(Paragraph("1. CASE SUMMARY", heading_style))
        story.append(Paragraph(brief.case_summary, body_style))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.lightgrey, spaceAfter=8
        ))
        # ── Section 2: Strongest Arguments ───────────────────────
        story.append(Paragraph(
            "2. STRONGEST ARGUMENTS", heading_style
        ))
        for i, arg in enumerate(brief.strongest_arguments, 1):
            story.append(Paragraph(
                f"<b>[{i}] Strength {arg['strength']}/5 — "
                f"{arg['citation']}</b>",
                subheading_style
            ))
            story.append(Paragraph(arg["reason"], bullet_style))
            story.append(Paragraph(
                f'<i>"{arg["text"][:200]}..."</i>', meta_style
            ))

        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.lightgrey, spaceAfter=8
        ))
        # ── Section 3: Vulnerabilities ────────────────────────────
        story.append(Paragraph(
            "3. VULNERABILITIES", heading_style
        ))
        for i, v in enumerate(brief.vulnerabilities, 1):
            story.append(Paragraph(
                f"<b>[{i}] Risk {v['strength']}/5 — "
                f"{v['citation']}</b>",
                subheading_style
            ))
            story.append(Paragraph(v["reason"], bullet_style))
            story.append(Paragraph(
                f'<i>"{v["text"][:200]}..."</i>', meta_style
            ))

        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.lightgrey, spaceAfter=8
        ))
        # ── Section 4: Witness Profiles ───────────────────────────
        story.append(PageBreak())
        story.append(Paragraph(
            "4. WITNESS PROFILES", heading_style
        ))
        for w in brief.witness_profiles:
            rating_color = {
                "HOSTILE":  "#dc3545",
                "NEUTRAL":  "#6c757d",
                "FRIENDLY": "#28a745",
            }.get(w.reliability_rating, "#6c757d")

            story.append(Paragraph(
                f'<b>{w.speaker_id}</b> — '
                f'<font color="{rating_color}">'
                f'{w.reliability_rating}</font> — '
                f'Credibility: {w.credibility_score}/10 — '
                f'{w.inferred_role}',
                subheading_style
            ))
            story.append(Paragraph(
                f"Statements: {w.total_statements} | "
                f"Contradictions: {w.contradiction_count} | "
                f"Hedge phrases: {w.hedge_count} | "
                f"First: {w.first_appearance} | "
                f"Last: {w.last_appearance}",
                meta_style
            ))
            story.append(Paragraph(
                f"<b>Approach:</b> {w.recommended_approach}",
                bullet_style
            ))
            if w.suggested_questions:
                story.append(Paragraph(
                    "<b>Suggested Questions:</b>", bullet_style
                ))
                for q in w.suggested_questions[:3]:
                    story.append(Paragraph(
                        f"• {q}", bullet_style
                    ))

            story.append(Spacer(1, 0.1 * inch))

        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.lightgrey, spaceAfter=8
        ))
        # ── Section 5: Contradictions ─────────────────────────────
        story.append(Paragraph(
            "5. WITNESS CONTRADICTIONS", heading_style
        ))
        if not brief.contradictions:
            story.append(Paragraph(
                "No contradictions detected.", body_style
            ))
        else:
            sev_colors = {
                "high":   "#dc3545",
                "medium": "#ffc107",
                "low":    "#28a745",
            }
            for i, c in enumerate(brief.contradictions, 1):
                sc = sev_colors.get(c["severity"], "#6c757d")
                story.append(Paragraph(
                    f'<b>[{i}] '
                    f'<font color="{sc}">'
                    f'{c["severity"].upper()}'
                    f'</font></b> — {c["explanation"]}',
                    subheading_style
                ))
                story.append(Paragraph(
                    f"<b>A</b> {c['citation_a']}: "
                    f'<i>"{c["statement_a"][:150]}"</i>',
                    bullet_style
                ))
                story.append(Paragraph(
                    f"<b>B</b> {c['citation_b']}: "
                    f'<i>"{c["statement_b"][:150]}"</i>',
                    bullet_style
                ))
                if c.get("is_cross_source"):
                    story.append(Paragraph(
                        "⚡ Cross-source contradiction "
                        "(video vs document)", meta_style
                    ))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.lightgrey, spaceAfter=8
        ))
        # ── Section 6: Opposition Strategy ───────────────────────
        story.append(PageBreak())
        story.append(Paragraph(
            "6. OPPOSITION STRATEGY", heading_style
        ))
        if not brief.opposition_strategy:
            story.append(Paragraph(
                "No devil's advocate sessions found. "
                "Run the Devil's Advocate tab to populate "
                "this section.",
                body_style
            ))
        else:
            for i, arg in enumerate(
                brief.opposition_strategy, 1
            ):
                story.append(Paragraph(
                    f"{i}. {arg}", bullet_style
                ))

        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.lightgrey, spaceAfter=8
        ))

        # ── Section 7: Recommended Questions ─────────────────────
        story.append(Paragraph(
            "7. RECOMMENDED CROSS-EXAMINATION QUESTIONS",
            heading_style
        ))
        for i, q in enumerate(brief.recommended_questions, 1):
            story.append(Paragraph(f"{i}. {q}", bullet_style))

        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.lightgrey, spaceAfter=8
        ))

        # ── Section 8: Overall Assessment ────────────────────────
        story.append(Paragraph(
            "8. OVERALL ASSESSMENT", heading_style
        ))
        story.append(Paragraph(
            brief.overall_assessment, body_style
        ))
        if brief.critical_actions:
            story.append(Paragraph(
                "<b>Critical Actions Before Court:</b>",
                body_style
            ))
            for action in brief.critical_actions:
                story.append(Paragraph(
                    f"⚠ {action}", bullet_style
                ))

        # ── Build ─────────────────────────────────────────────────
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
        texts = []
        for seg in item.structured_transcripts:
            t = seg.get("text", "").strip()
            if t:
                texts.append(t)
        return " ".join(texts)

    def _parse_json_safe(self, raw: str) -> Dict:
        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}