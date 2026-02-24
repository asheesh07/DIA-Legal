class QueryRouter:

    def route(self, query: str):

        query_lower = query.lower()

        if "contradict" in query_lower:
            mode = "opposition"
        elif "summarize" in query_lower:
            mode = "assistant"
        else:
            mode = "evidence"

        # Simplified query type detection
        if "when" in query_lower or "time" in query_lower:
            query_type = "temporal"
        else:
            query_type = "semantic"

        return {
            "mode": mode,
            "query_type": query_type
        }
