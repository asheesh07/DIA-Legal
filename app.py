# app.py — Dia Legal AI Lawyer v3
# Chatbot query interface + all modules

import os
import json
import hashlib
import platform
from dotenv import load_dotenv
from pathlib import Path
import gradio as gr
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from src.reader import ReaderRouter
from src.video_processor import VideoProcessor
from src.chunker import Chunker
from src.chunker import PDFChunker
from src.embedder import MultiModalEmbedder, TextEmbedder, VisualEmbedder
from src.vectorstore import LanceDBVectorStore
from src.retriever import Retriever
from src.reranker import CrossEncoderReranker
from src.llmclient import LLMClient
from src.context_builder import ContextBuilder, LegalMode
from src.llm_answerer import LLMAnswerer
from src.pipeline import DIAPipeline
from src.query_router import QueryRouter
from src.ingestion import IngestionPipeline
from src.evidence_classifier import EvidenceClassifier, format_evidence_map
from src.contradiction_detector import (
    ContradictionDetector, format_contradiction_report
)
from src.devils_advocate import DevilsAdvocate, format_round
from src.trial_brief_generator import TrialBriefGenerator

load_dotenv()

BASE_STORAGE = "data"
DB_PATH      = "data/lancedb"
CACHE_FILE   = "data/ingestion_cache.json"
CASES_FILE   = "data/cases_registry.json"

# ============================================================
# Ingestion Cache
# ============================================================

def _load_cache() -> dict:
    Path(CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def _save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def _cache_key(source: str, case_id: str) -> str:
    return hashlib.md5(f"{source}::{case_id}".encode()).hexdigest()

def _is_cached(source: str, case_id: str) -> bool:
    return _cache_key(source, case_id) in _load_cache()

def _mark_cached(source: str, case_id: str, chunks: int):
    cache = _load_cache()
    cache[_cache_key(source, case_id)] = {
        "source":  source,
        "case_id": case_id,
        "chunks":  chunks,
    }
    _save_cache(cache)

# ============================================================
# Case Registry
# ============================================================

def _load_cases() -> dict:
    Path(CASES_FILE).parent.mkdir(parents=True, exist_ok=True)
    if Path(CASES_FILE).exists():
        with open(CASES_FILE) as f:
            return json.load(f)
    return {}

def _save_cases(cases: dict):
    with open(CASES_FILE, "w") as f:
        json.dump(cases, f, indent=2)

def _register_case(case_id: str, name: str,
                   source_type: str, chunks: int):
    cases = _load_cases()
    if case_id not in cases:
        cases[case_id] = {"sources": [], "total_chunks": 0}
    existing_names = [s["name"] for s in cases[case_id]["sources"]]
    if name not in existing_names:
        cases[case_id]["sources"].append({
            "name": name, "type": source_type, "chunks": chunks
        })
    cases[case_id]["total_chunks"] = sum(
        s["chunks"] for s in cases[case_id]["sources"]
    )
    _save_cases(cases)

def _get_case_list() -> list:
    cases = _load_cases()
    rows = []
    for cid, data in cases.items():
        sources = data.get("sources", [])
        rows.append([
            cid,
            len(sources),
            data.get("total_chunks", 0),
            ", ".join(s["name"][:25] for s in sources[:3])
            + ("…" if len(sources) > 3 else "")
        ])
    return rows

def _get_case_summary(case_id: str) -> str:
    if not case_id:
        return ""
    cases = _load_cases()
    if case_id not in cases:
        return f"No data found for {case_id}.\nIngest sources first."
    data    = cases[case_id]
    sources = data.get("sources", [])
    lines   = [
        f"Case ID      : {case_id}",
        f"Total chunks : {data.get('total_chunks', 0)}",
        f"Sources ({len(sources)}):",
    ]
    for s in sources:
        lines.append(
            f"  · {s['name']}  [{s['type']}]  {s['chunks']} chunks"
        )
    return "\n".join(lines)

# ============================================================
# System Bootstrap
# ============================================================

def _build_systems():
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2"
    )
    video_processor = VideoProcessor(
        base_output_path=BASE_STORAGE, model_size="tiny"
    )
    chunker = Chunker(
        max_duration=20, max_tokens=512,
        overlap_duration=5, tokenizer=tokenizer
    )
    pdf_chunker = PDFChunker(max_tokens=512, overlap_tokens=50)

    text_model      = SentenceTransformer("all-MiniLM-L6-v2")
    text_embedder   = TextEmbedder(
        model=text_model, batch_size=12, normalize=True
    )
    visual_embedder = VisualEmbedder(
        model_name="openai/clip-vit-base-patch32",
        device="cpu", normalize=True
    )
    embedder = MultiModalEmbedder(
        text_embedder=text_embedder,
        visual_embedder=visual_embedder,
        visual_aggregation="mean"
    )
    vector_store = LanceDBVectorStore(
        table_name="dia_legal_v2",
        db_path=DB_PATH,
        text_dim=text_embedder.embed_dims,
        visual_dim=visual_embedder.embed_dim
    )
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=32, normalize=True
    )
    retriever = Retriever(
        vector_store=vector_store, embedder=embedder,
        max_candidates=30, reranker=reranker,
        enable_mmr=True, mmr_lambda=0.5,
        min_threshold=0.0, temporary_window=5
    )
    context_builder = ContextBuilder(
        max_tokens=2000, include_scores=True
    )
    llm_client = LLMClient(api_token=os.environ.get("HF_TOKEN"))

    if not hasattr(llm_client, "classify"):
        raise AttributeError(
            "LLMClient is missing classify() method. "
            "Add it to src/llmclient.py."
        )

    answerer = LLMAnswerer(
        llm_client=llm_client,
        confidence_threshold=0.3,
        max_history=3
    )
    pipeline = DIAPipeline(
        retriever=retriever,
        context_builder=context_builder,
        llm_answerer=answerer,
        query_router=QueryRouter()
    )
    ingestion_pipeline = IngestionPipeline(
        reader_router=ReaderRouter,
        video_processor=video_processor,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        pdf_chunker=pdf_chunker
    )
    evidence_classifier = EvidenceClassifier(
        retriever=retriever,
        llm_client=llm_client,
        context_builder=context_builder,
        top_k=50
    )
    contradiction_detector = ContradictionDetector(
        retriever=retriever,
        llm_client=llm_client,
        embedder=embedder,
        similarity_threshold=0.4,
        top_k=100
    )
    devils_advocate = DevilsAdvocate(
        retriever=retriever,
        llm_client=llm_client,
        storage_path=BASE_STORAGE,
        top_k=10
    )
    brief_generator = TrialBriefGenerator(
        retriever=retriever,
        llm_client=llm_client,
        evidence_classifier=evidence_classifier,
        contradiction_detector=contradiction_detector,
        devils_advocate=devils_advocate,
        storage_path=BASE_STORAGE
    )
    return (
        pipeline, ingestion_pipeline,
        evidence_classifier, contradiction_detector,
        devils_advocate, brief_generator,
        retriever, embedder
    )


print("[Dia Legal] Initialising systems...")
(
    pipeline, ingestion_pipeline,
    evidence_classifier, contradiction_detector,
    devils_advocate, brief_generator,
    retriever, embedder
) = _build_systems()
print("[Dia Legal] Ready.")

_da_session = {"id": None, "case_id": None}

# ============================================================
# Handlers
# ============================================================

def ingest_source(source_type, url, local_file,
                  case_id, pdf_files):
    if not case_id.strip():
        return "❌ Case ID is required.", _get_case_list()

    results = []

    video_source = None
    if source_type == "YouTube URL" and url and url.strip():
        video_source = url.strip()
    elif source_type == "Local Video File" and local_file:
        video_source = local_file.name

    if video_source:
        if _is_cached(video_source, case_id.strip()):
            cache  = _load_cache()
            chunks = cache[
                _cache_key(video_source, case_id.strip())
            ].get("chunks", "?")
            results.append(
                f"⚡ Video already ingested — {chunks} chunks (skipped)"
            )
        else:
            try:
                r = ingestion_pipeline.ingest(
                    source=video_source,
                    case_id=case_id.strip(),
                    storage_path=BASE_STORAGE
                )
                n = r.get("chunks_indexed", 0)
                display_name = (
                    video_source[:60]
                    if video_source.startswith("http")
                    else Path(video_source).name
                )
                _mark_cached(video_source, case_id.strip(), n)
                _register_case(
                    case_id.strip(), display_name, "video", n
                )
                results.append(f"✅ Video ingested — {n} chunks")
            except Exception as e:
                results.append(f"❌ Video failed: {e}")

    if pdf_files:
        seen = set()
        for file in pdf_files:
            name = Path(file.name).name
            if name in seen:
                results.append(f"⚠️ Skipped duplicate: {name}")
                continue
            seen.add(name)

            if _is_cached(file.name, case_id.strip()):
                cache  = _load_cache()
                chunks = cache[
                    _cache_key(file.name, case_id.strip())
                ].get("chunks", "?")
                results.append(
                    f"⚡ {name} already ingested — {chunks} chunks (skipped)"
                )
                continue

            try:
                r = ingestion_pipeline.ingest(
                    source=file.name,
                    case_id=case_id.strip(),
                    storage_path=BASE_STORAGE
                )
                n       = r.get("chunks_indexed", 0)
                doctype = r.get("doc_type", "document")
                _mark_cached(file.name, case_id.strip(), n)
                _register_case(
                    case_id.strip(), name, doctype, n
                )
                results.append(f"✅ {name} — {n} chunks ({doctype})")
            except Exception as e:
                results.append(f"❌ {name}: {e}")

    if not results:
        return "⚠️ Provide a URL, local video, or PDF.", _get_case_list()
    return "\n".join(results), _get_case_list()


def answer_query_chat(case_id, message, history):
    if not message.strip():
        return history, "", ""
    if not case_id.strip():
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "❌ Enter a Case ID first."}
        ]
        return history, "", ""
    try:
        result     = pipeline.run(
            query=message.strip(),
            case_id=case_id.strip()
        )
        answer     = result.get("answer", "No answer returned.")
        citations  = result.get("citations", [])
        confidence = result.get("confidence", 0.0)

        cite_parts = []
        for c in citations:
            ref = c.get("citation_ref", "")
            if ref:
                cite_parts.append(ref)
            else:
                tr = c.get("time_range", [0, 0])
                ms = int(tr[0]) // 60
                ss = int(tr[0]) % 60
                me = int(tr[1]) // 60
                se = int(tr[1]) % 60
                cite_parts.append(
                    f"[{ms:02d}:{ss:02d}→{me:02d}:{se:02d}]"
                )

        cite_str = "  ·  ".join(cite_parts) if cite_parts else "No citations"
        bot_response = f"{answer}\n\nConfidence: {confidence:.0%}"

        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": bot_response}
        ]
        return history, "", f"CITATIONS  ·  {cite_str}"

    except Exception as e:
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": f"❌ Error: {e}"}
        ]
        return history, "", ""


def clear_chat():
    return [], "", ""


def run_evidence_map(case_id, position):
    if not case_id.strip() or not position.strip():
        return [], [], [], "⚠️ Fill in Case ID and position."
    try:
        em  = evidence_classifier.classify(
            case_id=case_id.strip(),
            lawyer_position=position.strip()
        )
        fmt = format_evidence_map(em)
        s   = em.summary
        return (
            fmt["supporting"]["rows"],
            fmt["opposing"]["rows"],
            fmt["neutral"]["rows"],
            f"✅ {s['total_classified']} chunks — "
            f"{s['supporting_count']} supporting · "
            f"{s['opposing_count']} opposing · "
            f"{s['neutral_count']} neutral"
        )
    except Exception as e:
        return [], [], [], f"❌ Error: {e}"


def run_contradiction_detection(case_id):
    if not case_id.strip():
        return [], "⚠️ Enter a Case ID."
    try:
        report = contradiction_detector.detect(
            case_id=case_id.strip()
        )
        fmt = format_contradiction_report(report)
        s   = report.summary
        return (
            fmt["rows"],
            f"✅ {s['total_contradictions']} contradictions — "
            f"{s['high_severity']} high · "
            f"{s['medium_severity']} medium · "
            f"{s['low_severity']} low · "
            f"{s['cross_source']} cross-source"
        )
    except Exception as e:
        return [], f"❌ Error: {e}"


def da_new_session(case_id, topic):
    if not case_id.strip() or not topic.strip():
        return "⚠️ Enter Case ID and topic.", gr.update()
    try:
        s = devils_advocate.new_session(
            case_id=case_id.strip(), topic=topic.strip()
        )
        _da_session["id"]      = s.session_id
        _da_session["case_id"] = case_id.strip()
        return (
            f"✅ Session {s.session_id} started",
            gr.update(interactive=True)
        )
    except Exception as e:
        return f"❌ {e}", gr.update()


def da_load_session(case_id, session_id):
    if not case_id.strip() or not session_id.strip():
        return "⚠️ Enter Case ID and Session ID.", gr.update(), []
    try:
        s = devils_advocate.load_session(
            case_id=case_id.strip(),
            session_id=session_id.strip()
        )
        _da_session["id"]      = s.session_id
        _da_session["case_id"] = case_id.strip()
        history = [
            [r.round_number, r.lawyer_argument[:80],
             r.critique[:100], len(r.weaknesses)]
            for r in s.rounds
        ]
        return (
            f"✅ {s.session_id} loaded — {len(s.rounds)} rounds",
            gr.update(interactive=True),
            history
        )
    except Exception as e:
        return f"❌ {e}", gr.update(), []


def da_list_sessions(case_id):
    if not case_id.strip():
        return []
    try:
        sessions = devils_advocate.list_sessions(case_id.strip())
        return [
            [s["session_id"], s["topic"],
             s["rounds"], s["created_at"][:10]]
            for s in sessions
        ]
    except Exception:
        return []


def da_argue(argument):
    if not argument.strip():
        return "", "", "", "", "⚠️ Enter your argument."
    if not _da_session["id"]:
        return "", "", "", "", "⚠️ Start or load a session first."
    try:
        r   = devils_advocate.argue(argument.strip())
        fmt = format_round(r)
        return (
            fmt["critique"],
            fmt["weaknesses"],
            fmt["opposition"],
            fmt["strengthen"],
            f"✅ Round {fmt['round_number']} — "
            f"{fmt['evidence_count']} pieces used"
        )
    except Exception as e:
        return "", "", "", "", f"❌ {e}"


def generate_brief(case_id, position):
    if not case_id.strip() or not position.strip():
        return "⚠️ Fill in Case ID and position.", "", "", "", "", None
    try:
        brief    = brief_generator.generate(
            case_id=case_id.strip(),
            lawyer_position=position.strip()
        )
        pdf_path = getattr(brief, "pdf_path", None)

        profiles_text = ""
        icons = {"HOSTILE": "🔴", "NEUTRAL": "🟡", "FRIENDLY": "🟢"}
        for w in brief.witness_profiles:
            profiles_text += (
                f"{icons.get(w.reliability_rating,'⚪')} "
                f"{w.speaker_id} — {w.inferred_role}\n"
                f"   Credibility {w.credibility_score}/10 · "
                f"Contradictions: {w.contradiction_count} · "
                f"Hedges: {w.hedge_count}\n"
                f"   → {w.recommended_approach}\n\n"
            )

        sev_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        contras_text = ""
        for c in brief.contradictions:
            contras_text += (
                f"{sev_icons.get(c['severity'],'⚪')} "
                f"{c['explanation']}\n"
                f"   A {c['citation_a']}: "
                f"\"{c['statement_a'][:100]}\"\n"
                f"   B {c['citation_b']}: "
                f"\"{c['statement_b'][:100]}\"\n\n"
            )

        si = {"STRONG":"🟢","MODERATE":"🟡","WEAK":"🔴"}.get(
            brief.case_strength, "⚪"
        )
        brief_text = (
            f"{si} CASE STRENGTH: {brief.case_strength}\n\n"
            f"{'━'*48}\nCASE SUMMARY\n{'━'*48}\n"
            f"{brief.case_summary}\n\n"
            f"{'━'*48}\nOVERALL ASSESSMENT\n{'━'*48}\n"
            f"{brief.overall_assessment}\n\n"
            f"{'━'*48}\nCRITICAL ACTIONS\n{'━'*48}\n"
            + "\n".join(f"⚠ {a}" for a in brief.critical_actions)
        )
        questions_text = "\n".join(
            f"{i+1}. {q}"
            for i, q in enumerate(brief.recommended_questions)
        )

        return (
            f"✅ Brief {brief.brief_id} — "
            f"{len(brief.witness_profiles)} witnesses · "
            f"{len(brief.contradictions)} contradictions",
            brief_text,
            profiles_text or "No video witnesses found.",
            contras_text  or "No contradictions detected.",
            questions_text or "No questions generated.",
            pdf_path
        )
    except Exception as e:
        return f"❌ Error: {e}", "", "", "", "", None


# ============================================================
# CSS
# ============================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #07070f;
    --bg2:     #0d0d1a;
    --bg3:     #121220;
    --bg4:     #17172a;
    --border:  #222235;
    --border2: #303050;
    --gold:    #c9a84c;
    --gold2:   #e2c070;
    --text:    #ddddf0;
    --text2:   #7777aa;
    --text3:   #444460;
    --green:   #2ecc71;
    --red:     #e74c3c;
    --amber:   #f39c12;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: var(--text) !important;
}
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* ── Header ── */
.header {
    text-align: center;
    padding: 44px 0 30px;
    border-bottom: 1px solid var(--border);
}
.wordmark {
    font-family: 'Playfair Display', serif;
    font-size: 48px;
    font-weight: 700;
    letter-spacing: -1px;
    color: var(--text);
    line-height: 1;
}
.wordmark span { color: var(--gold); }
.tagline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 5px;
    color: var(--text3);
    text-transform: uppercase;
    margin-top: 8px;
}
.bar {
    width: 36px; height: 2px;
    background: var(--gold);
    margin: 14px auto 0;
}

/* ── Tabs ── */
.tab-nav {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 28px !important;
}
.tab-nav button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: var(--text3) !important;
    padding: 16px 16px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: all 0.15s !important;
    border-radius: 0 !important;
}
.tab-nav button:hover { color: var(--text2) !important; }
.tab-nav button.selected {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
}

/* ── Labels ── */
label, .gr-form > label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text3) !important;
    margin-bottom: 5px !important;
}
.slabel {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--gold);
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}

/* ── Inputs ── */
input, textarea {
    background: var(--bg4) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
    border-radius: 3px !important;
    transition: border-color 0.15s !important;
}
input:focus, textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.07) !important;
    outline: none !important;
}

/* ── Buttons ── */
button.primary {
    background: var(--gold) !important;
    color: #05050e !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 2px !important;
    transition: all 0.15s !important;
}
button.primary:hover {
    background: var(--gold2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.18) !important;
}
button.secondary {
    background: transparent !important;
    border: 1px solid var(--border2) !important;
    color: var(--text2) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    transition: all 0.15s !important;
}
button.secondary:hover {
    border-color: var(--gold) !important;
    color: var(--gold) !important;
}

/* ── Tables ── */
table { background: var(--bg3) !important; border-collapse: collapse !important; }
th {
    background: var(--bg2) !important;
    color: var(--gold) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 8px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 10px 12px !important;
    border: 1px solid var(--border) !important;
}
td {
    color: var(--text) !important;
    font-size: 12px !important;
    padding: 8px 12px !important;
    border: 1px solid var(--border) !important;
}
tr:hover td { background: var(--bg4) !important; }

/* ── Cards ── */
.gr-group, .gr-box {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* ── Divider ── */
.div {
    height: 1px;
    background: var(--border);
    margin: 20px 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 18px;
    margin-top: 36px;
    border-top: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px;
    letter-spacing: 2px;
    color: var(--text3);
    text-transform: uppercase;
}

/* ── Chatbot ── */
.chatbot {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
.chatbot .message {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}
.chatbot .user {
    background: rgba(201,168,76,0.08) !important;
    border: 1px solid rgba(201,168,76,0.2) !important;
    border-radius: 6px 6px 2px 6px !important;
    color: var(--text) !important;
}
.chatbot .bot {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px 6px 6px 2px !important;
    color: var(--text) !important;
}
.chatbot .avatar-container { display: none !important; }

/* Citation bar */
.citation-bar textarea {
    background: var(--bg4) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid rgba(201,168,76,0.15) !important;
    color: var(--text3) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 1px !important;
    border-radius: 3px !important;
    padding: 8px 12px !important;
}

/* Chat input */
.chat-input textarea {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
    border-radius: 4px !important;
}
.chat-input textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.07) !important;
}

/* Pulse dot */
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px rgba(201,168,76,0.5); }
    50%       { opacity: 0.4; box-shadow: none; }
}
.pulse-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--gold);
    animation: pulse 2s infinite;
    margin-right: 8px;
    vertical-align: middle;
}
"""

# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(
    title="Dia Legal",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="amber",
        neutral_hue="slate",
    )
) as demo:

    # Header
    gr.HTML("""
    <div class="header">
        <div class="wordmark">DIA <span>LEGAL</span></div>
        <div class="tagline">AI Intelligence · Legal Proceedings</div>
        <div class="bar"></div>
    </div>
    """)

    # ── 00 · Cases ────────────────────────────────────────────
    with gr.Tab("00 · CASES"):
        gr.HTML('<div class="slabel">Case Management</div>')
        with gr.Row():
            with gr.Column(scale=3):
                cases_table = gr.Dataframe(
                    headers=["Case ID","Sources","Chunks","Files"],
                    value=_get_case_list(),
                    label="All Cases",
                    interactive=False,
                )
            with gr.Column(scale=1):
                selected_case = gr.Textbox(
                    label="Case ID",
                    value="Case_001",
                )
                case_summary_out = gr.Textbox(
                    label="Case Details",
                    interactive=False,
                    lines=10,
                )
                with gr.Row():
                    refresh_btn = gr.Button("REFRESH", variant="secondary")
                    view_btn    = gr.Button("VIEW",    variant="primary")

        refresh_btn.click(fn=_get_case_list, outputs=cases_table)
        view_btn.click(
            fn=_get_case_summary,
            inputs=[selected_case],
            outputs=case_summary_out
        )

    # ── 01 · Ingest ───────────────────────────────────────────
    with gr.Tab("01 · INGEST"):
        gr.HTML('<div class="slabel">Add Sources to Case</div>')
        with gr.Row():
            with gr.Column(scale=3):
                case_id_setup = gr.Textbox(
                    label="Case ID",
                    placeholder="Case_001",
                    value="Case_001"
                )
                gr.HTML('<div class="div"></div>')
                video_source_type = gr.Radio(
                    choices=["YouTube URL","Local Video File"],
                    value="YouTube URL",
                    label="Video Source Type",
                )
                youtube_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    visible=True,
                )
                local_video = gr.File(
                    label="Local Video File",
                    file_types=[".mp4",".mov",".mkv",".avi",".webm"],
                    visible=False,
                )
                gr.HTML('<div class="div"></div>')
                pdf_files = gr.File(
                    label="Legal Documents — PDF (select all at once)",
                    file_types=[".pdf"],
                    file_count="multiple",
                )
            with gr.Column(scale=1):
                gr.HTML('<div class="slabel">Result</div>')
                ingest_btn = gr.Button("INGEST SOURCES", variant="primary")
                ingest_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=10,
                )
                gr.HTML("""
                <div style="font-family:'IBM Plex Mono',monospace;
                     font-size:9px;color:#444460;line-height:2;
                     margin-top:10px;">
                ⚡ Already ingested sources are skipped<br>
                ✅ Select all PDFs at once<br>
                ⚠ Videos are cached — no re-downloads
                </div>
                """)

        def _toggle(choice):
            return (
                gr.update(visible=choice == "YouTube URL"),
                gr.update(visible=choice == "Local Video File"),
            )

        video_source_type.change(
            fn=_toggle,
            inputs=[video_source_type],
            outputs=[youtube_url, local_video]
        )
        ingest_btn.click(
            fn=ingest_source,
            inputs=[video_source_type, youtube_url,
                    local_video, case_id_setup, pdf_files],
            outputs=[ingest_status, cases_table]
        )

    # ── 02 · Query (Chatbot) ──────────────────────────────────
    with gr.Tab("02 · QUERY"):

        # Hidden state to track if chat has started
        chat_started = gr.State(False)

        # Welcome screen (shown when no messages)
        welcome_html = gr.HTML("""
        <div id="dia-welcome" style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 60px 20px 40px;
            text-align: center;
        ">
            <!-- Scales icon -->
            <div style="
                width: 56px; height: 56px;
                background: rgba(201,168,76,0.1);
                border: 1px solid rgba(201,168,76,0.25);
                border-radius: 50%;
                display: flex; align-items: center; justify-content: center;
                font-size: 24px;
                margin-bottom: 24px;
                box-shadow: 0 0 30px rgba(201,168,76,0.08);
            ">⚖</div>

            <!-- Greeting -->
            <h2 style="
                font-family: 'Playfair Display', serif;
                font-size: 26px;
                font-weight: 600;
                color: #ddddf0;
                margin: 0 0 10px;
                letter-spacing: -0.3px;
            ">How can I help with the case?</h2>
            <p style="
                font-family: 'IBM Plex Sans', sans-serif;
                font-size: 13px;
                color: #7777aa;
                margin: 0 0 36px;
                letter-spacing: 0.3px;
            ">Query testimony · Cross-reference documents · Find contradictions · Legal analysis</p>

            <!-- Suggestion pills -->
            <div style="
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                justify-content: center;
                max-width: 680px;
            ">
                <button onclick="dia_fill('Summarise this case in 3 sentences')" class="dia-pill">
                    Summarise this case
                </button>
                <button onclick="dia_fill('What is the strongest evidence against my client?')" class="dia-pill">
                    Strongest evidence against client
                </button>
                <button onclick="dia_fill('What contradictions exist in the witness testimony?')" class="dia-pill">
                    Witness contradictions
                </button>
                <button onclick="dia_fill('List all dates mentioned in the evidence')" class="dia-pill">
                    Timeline of events
                </button>
                <button onclick="dia_fill('Who knew about the fault and when?')" class="dia-pill">
                    Who knew what and when
                </button>
                <button onclick="dia_fill('What IPC sections could apply to this case?')" class="dia-pill">
                    Applicable IPC sections
                </button>
            </div>
        </div>

        <style>
        .dia-pill {
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 12px;
            color: #9999bb;
            background: #0d0d1a;
            border: 1px solid #222235;
            border-radius: 20px;
            padding: 9px 18px;
            cursor: pointer;
            transition: all 0.15s ease;
            letter-spacing: 0.2px;
        }
        .dia-pill:hover {
            border-color: #c9a84c;
            color: #c9a84c;
            background: rgba(201,168,76,0.05);
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(201,168,76,0.1);
        }

        /* Hide welcome when chat has messages */
        .messages-present #dia-welcome { display: none !important; }

        /* Chatbot styling overrides */
        .dia-chatbot {
            background: transparent !important;
            border: none !important;
        }
        .dia-chatbot > div {
            background: transparent !important;
        }

        /* Hide Gradio chatbot chrome */
        .dia-chatbot .copy-btn,
        .dia-chatbot button[aria-label="Share"],
        .dia-chatbot button[aria-label="Delete"],
        .dia-chatbot .chatbot-footer,
        .dia-chatbot .message-buttons {
            display: none !important;
        }

        /* User bubble */
        .dia-chatbot .user {
            background: rgba(201,168,76,0.08) !important;
            border: 1px solid rgba(201,168,76,0.18) !important;
            border-radius: 18px 18px 4px 18px !important;
            color: #ddddf0 !important;
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
            padding: 12px 16px !important;
            max-width: 72% !important;
            margin-left: auto !important;
        }

        /* Bot bubble */
        .dia-chatbot .bot {
            background: #121220 !important;
            border: 1px solid #222235 !important;
            border-radius: 18px 18px 18px 4px !important;
            color: #ddddf0 !important;
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
            padding: 12px 16px !important;
            max-width: 80% !important;
        }

        /* Input bar container */
        .dia-input-wrap {
            position: relative;
            margin: 12px 0 0;
        }
        .dia-input-wrap textarea {
            width: 100% !important;
            background: #0f0f1e !important;
            border: 1.5px solid #2a2a42 !important;
            border-radius: 28px !important;
            color: #ddddf0 !important;
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-size: 14px !important;
            padding: 16px 60px 16px 24px !important;
            resize: none !important;
            transition: border-color 0.2s !important;
            box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
            line-height: 1.5 !important;
        }
        .dia-input-wrap textarea:focus {
            border-color: #c9a84c !important;
            box-shadow: 0 0 0 3px rgba(201,168,76,0.08),
                        0 4px 24px rgba(0,0,0,0.3) !important;
            outline: none !important;
        }
        .dia-input-wrap textarea::placeholder {
            color: #444460 !important;
        }

        /* Send button */
        .dia-send-btn {
            position: absolute !important;
            right: 10px !important;
            bottom: 10px !important;
            width: 38px !important;
            height: 38px !important;
            border-radius: 50% !important;
            background: #c9a84c !important;
            border: none !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            transition: all 0.15s !important;
            padding: 0 !important;
            min-width: unset !important;
            font-size: 16px !important;
            color: #05050e !important;
        }
        .dia-send-btn:hover {
            background: #e2c070 !important;
            transform: scale(1.05) !important;
            box-shadow: 0 4px 16px rgba(201,168,76,0.3) !important;
        }

        /* Case ID + clear row */
        .dia-top-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }

        /* Citation bar */
        .dia-citation textarea {
            background: transparent !important;
            border: none !important;
            border-top: 1px solid #1a1a2e !important;
            border-radius: 0 !important;
            color: #444460 !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 9px !important;
            letter-spacing: 1.5px !important;
            padding: 8px 4px !important;
            text-transform: uppercase !important;
        }
        .dia-citation label { display: none !important; }

        /* Footer hint */
        .dia-footer-hint {
            text-align: center;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 9px;
            color: #333350;
            letter-spacing: 1.5px;
            padding: 6px 0 2px;
            text-transform: uppercase;
        }
        </style>

        <script>
        function dia_fill(text) {
            // Find the textarea in the chat input and fill it
            const inputs = document.querySelectorAll('.dia-chat-input textarea');
            if (inputs.length > 0) {
                const ta = inputs[0];
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(ta, text);
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.focus();
            }
        }
        </script>
        """)

        # Top controls row
        with gr.Row(elem_classes=["dia-top-row"]):
            case_id_query = gr.Textbox(
                label="Case ID",
                value="Case_001",
                scale=1,
                min_width=140,
            )
            clear_btn = gr.Button(
                "✕ Clear",
                variant="secondary",
                scale=0,
                min_width=80,
            )

        # Chatbot
        chatbot = gr.Chatbot(
            label="",
            height=440,
            show_label=False,
            elem_classes=["dia-chatbot"],
        )

        # Citation bar
        citation_bar = gr.Textbox(
            value="",
            label="Citations",
            interactive=False,
            elem_classes=["dia-citation"],
            lines=1,
            max_lines=1,
        )

        # Input + send
        with gr.Row(elem_classes=["dia-input-wrap"]):
            chat_input = gr.Textbox(
                placeholder="Ask anything about the case — testimony, documents, witnesses, legal sections...",
                label="",
                show_label=False,
                scale=9,
                lines=1,
                max_lines=5,
                elem_classes=["dia-chat-input"],
            )
            send_btn = gr.Button(
                "↑",
                variant="primary",
                scale=0,
                min_width=44,
                elem_classes=["dia-send-btn"],
            )

        gr.HTML('<div class="dia-footer-hint">Responses grounded in case evidence · Not legal advice</div>')

        # Wire up
        send_btn.click(
            fn=answer_query_chat,
            inputs=[case_id_query, chat_input, chatbot],
            outputs=[chatbot, chat_input, citation_bar],
        )
        chat_input.submit(
            fn=answer_query_chat,
            inputs=[case_id_query, chat_input, chatbot],
            outputs=[chatbot, chat_input, citation_bar],
        )
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, chat_input, citation_bar],
        )

    # ── 03 · Evidence Map ─────────────────────────────────────
    with gr.Tab("03 · EVIDENCE MAP"):
        gr.HTML('<div class="slabel">Classify All Evidence</div>')
        with gr.Row():
            case_id_evidence = gr.Textbox(
                label="Case ID", value="Case_001", scale=1
            )
            position_evidence = gr.Textbox(
                label="Your Position",
                placeholder="I am defending the driver against negligence charges...",
                scale=5
            )
            evidence_btn = gr.Button(
                "MAP EVIDENCE", variant="primary", scale=1
            )
        evidence_status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="slabel" style="color:#2ecc71">Supporting Your Case</div>')
                supporting_table = gr.Dataframe(
                    headers=["Str","Citation","Source","Reason","Text"],
                    interactive=False, wrap=True,
                )
            with gr.Column():
                gr.HTML('<div class="slabel" style="color:#e74c3c">Against Your Case</div>')
                opposing_table = gr.Dataframe(
                    headers=["Str","Citation","Source","Reason","Text"],
                    interactive=False, wrap=True,
                )
        gr.HTML('<div class="slabel" style="color:#444460">Neutral</div>')
        neutral_table = gr.Dataframe(
            headers=["Str","Citation","Source","Reason","Text"],
            interactive=False, wrap=True,
        )
        evidence_btn.click(
            fn=run_evidence_map,
            inputs=[case_id_evidence, position_evidence],
            outputs=[supporting_table, opposing_table,
                     neutral_table, evidence_status]
        )

    # ── 04 · Contradictions ───────────────────────────────────
    with gr.Tab("04 · CONTRADICTIONS"):
        gr.HTML('<div class="slabel">Detect Witness Contradictions</div>')
        with gr.Row():
            case_id_contra = gr.Textbox(
                label="Case ID", value="Case_001", scale=3
            )
            contra_btn = gr.Button(
                "DETECT CONTRADICTIONS", variant="primary", scale=1
            )
        contra_status = gr.Textbox(label="Status", interactive=False)
        contra_table = gr.Dataframe(
            headers=["Severity","Citation A","Statement A",
                     "Citation B","Statement B",
                     "Explanation","Cross-Source"],
            interactive=False, wrap=True,
        )
        contra_btn.click(
            fn=run_contradiction_detection,
            inputs=[case_id_contra],
            outputs=[contra_table, contra_status]
        )

    # ── 05 · Devil's Advocate ─────────────────────────────────
    with gr.Tab("05 · DEVIL'S ADVOCATE"):
        gr.HTML('<div class="slabel">Challenge Your Arguments</div>')
        gr.HTML("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
             color:#7777aa;margin-bottom:16px;letter-spacing:1px;">
            Submit your legal argument. The system challenges it using
            case evidence, finds weaknesses, and shows what the
            opposition will argue.
        </div>
        """)
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="slabel">Session</div>')
                case_id_da = gr.Textbox(label="Case ID", value="Case_001")
                da_topic = gr.Textbox(
                    label="Topic",
                    placeholder="Gearbox fault defense strategy"
                )
                with gr.Row():
                    da_new_btn  = gr.Button("NEW SESSION", variant="primary")
                    da_load_btn = gr.Button("LOAD",        variant="secondary")
                da_session_id_input = gr.Textbox(
                    label="Session ID to Load",
                    placeholder="a3f2b1c4"
                )
                da_list_btn = gr.Button("LIST SESSIONS", variant="secondary")
                da_sessions_table = gr.Dataframe(
                    headers=["ID","Topic","Rounds","Created"],
                    interactive=False,
                )
                da_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                gr.HTML('<div class="slabel">Argument</div>')
                da_argument = gr.Textbox(
                    label="Your Argument",
                    placeholder="Start a session first...",
                    lines=4,
                    interactive=False,
                )
                da_argue_btn = gr.Button(
                    "SUBMIT ARGUMENT",
                    variant="primary",
                    interactive=False
                )
                da_critique = gr.Textbox(
                    label="Overall Critique",
                    interactive=False, lines=4
                )
                with gr.Row():
                    da_weaknesses = gr.Textbox(
                        label="Weaknesses Found",
                        interactive=False, lines=6
                    )
                    da_opposition = gr.Textbox(
                        label="Opposition Will Argue",
                        interactive=False, lines=6
                    )
                da_strengthen = gr.Textbox(
                    label="How to Strengthen",
                    interactive=False, lines=3
                )
                da_round_status = gr.Textbox(
                    label="Round Status", interactive=False
                )

        da_new_btn.click(
            fn=da_new_session,
            inputs=[case_id_da, da_topic],
            outputs=[da_status, da_argument]
        ).then(
            fn=lambda: (gr.update(interactive=True), gr.update(interactive=True)),
            outputs=[da_argument, da_argue_btn]
        )
        da_load_btn.click(
            fn=da_load_session,
            inputs=[case_id_da, da_session_id_input],
            outputs=[da_status, da_argument, da_sessions_table]
        ).then(
            fn=lambda: (gr.update(interactive=True), gr.update(interactive=True)),
            outputs=[da_argument, da_argue_btn]
        )
        da_list_btn.click(
            fn=da_list_sessions,
            inputs=[case_id_da],
            outputs=[da_sessions_table]
        )
        da_argue_btn.click(
            fn=da_argue,
            inputs=[da_argument],
            outputs=[da_critique, da_weaknesses,
                     da_opposition, da_strengthen,
                     da_round_status]
        )

    # ── 06 · Trial Brief ──────────────────────────────────────
    with gr.Tab("06 · TRIAL BRIEF"):
        gr.HTML('<div class="slabel">Generate Pre-Trial Intelligence Report</div>')
        with gr.Row():
            case_id_brief = gr.Textbox(
                label="Case ID", value="Case_001", scale=1
            )
            position_brief = gr.Textbox(
                label="Your Position",
                placeholder="I am defending the driver against negligence charges...",
                scale=5
            )
            brief_btn = gr.Button(
                "GENERATE BRIEF", variant="primary", scale=1
            )
        brief_status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            brief_main = gr.Textbox(
                label="Case Summary & Assessment",
                lines=16, interactive=False, scale=2
            )
            brief_questions = gr.Textbox(
                label="Cross-Examination Questions",
                lines=16, interactive=False, scale=1
            )
        with gr.Row():
            brief_witnesses = gr.Textbox(
                label="Witness Profiles",
                lines=10, interactive=False, scale=1
            )
            brief_contras = gr.Textbox(
                label="Contradictions",
                lines=10, interactive=False, scale=1
            )
        brief_download = gr.File(
            label="Download PDF Brief",
            interactive=False,
        )
        brief_btn.click(
            fn=generate_brief,
            inputs=[case_id_brief, position_brief],
            outputs=[brief_status, brief_main,
                     brief_witnesses, brief_contras,
                     brief_questions, brief_download]
        )

    # Footer
    gr.HTML("""
    <div class="footer">
        Dia Legal · AI Intelligence for Legal Proceedings ·
        WhisperX · LanceDB · CLIP · SentenceTransformers
    </div>
    """)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)