class QueryRouter:
    # Query patterns that indicate general legal knowledge — no case files needed.
    # These are answered directly by the LLM, skipping retrieval entirely.
    _DIRECT_PATTERNS = [
        "what is ", "what are ", "define ", "explain ",
        "how does ", "how do ", "what does the law",
        "ipc section", "section ipc", "under ipc", "under crpc",
        "what is bail", "what is warrant", "what is cognizable",
        "what is non-cognizable", "what is fir", "what is chargesheet",
        "what is acquittal", "what is conviction", "what is appeal",
        "what is jurisdiction", "what is evidence", "what is hearsay",
        "what is burden of proof", "what is reasonable doubt",
        "what is anticipatory bail", "what is remand",
        "punishment for", "penalty for", "sentence for",
        "legal definition", "legally speaking", "in law ",
        "difference between ", "what is the difference",
        "crpc section", "section crpc", "indian evidence act",
        "iea section", "constitution of india", "fundamental right",
    ]

    # Keywords that mean the query IS about the indexed case files.
    # These take priority over _DIRECT_PATTERNS.
    _CASE_MARKERS = [
        "this case", "my case", "the case", "in this case",
        "the fir", "this fir", "the witness", "the accused",
        "the victim", "the evidence", "the document", "the statement",
        "the deposition", "this deposition", "the testimony",
        "according to", "as per the file", "in the file",
        "what happened", "who was present", "where was",
        "exhibit ", "contradict", "inconsisten",
        "the charge sheet", "charge sheet", "the chargesheet",
        "my file", "our file", "uploaded", "indexed",
        "summarise this", "summarize this", "summarise the case",
        "summarize the case", "tell me about this case",
    ]

    def route(self, query: str) -> dict:
        q = query.lower().strip()

        # Case-specific markers take highest priority
        is_case_specific = any(m in q for m in self._CASE_MARKERS)

        # Only consider direct mode when no case markers are present
        is_direct = (not is_case_specific) and any(p in q for p in self._DIRECT_PATTERNS)

        if is_direct:
            mode = "direct"
        elif "contradict" in q or "opposition" in q:
            mode = "opposition"
        elif "summarize" in q or "summary" in q or "summarise" in q:
            mode = "assistant"
        else:
            mode = "evidence"

        query_type = "temporal" if ("when" in q or "time" in q) else "semantic"

        return {"mode": mode, "query_type": query_type}
