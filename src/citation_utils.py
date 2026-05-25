def is_pdf_chunk(item) -> bool:
    if getattr(item, "structured_transcripts", None):
        speaker = item.structured_transcripts[0].get("speaker", "")
        if speaker == "DOCUMENT":
            return True
        
    try:
        return (
            item.temporal.primary.start_time == 0.0 and
            item.temporal.primary.end_time == 0.0
        )
    except AttributeError:
        return False

def build_citation_ref(item) -> str:
    if is_pdf_chunk(item):
        meta          = getattr(item, "metadata", {})
        source_type   = meta.get("source_type", "document")
        section_title = meta.get("section_title", "")
        page_start    = meta.get("page_span_start", 0)
        page_end      = meta.get("page_span_end", 0)

        label = source_type_label(source_type)
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
        try:
            s  = item.temporal.primary.start_time
            e  = item.temporal.primary.end_time
            sm, ss = int(s) // 60, int(s) % 60
            em, es = int(e) // 60, int(e) % 60
            return f"[{sm:02d}:{ss:02d} → {em:02d}:{es:02d}]"
        except AttributeError:
            return "[Unknown Time]"

def source_type_label(source_type: str) -> str:
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
