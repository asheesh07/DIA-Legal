from typing import Dict, Any, List


class LLMAnswerer:
    def __init__(
        self,
        llm_client,
        confidence_threshold: float = 0.45,
        max_history: int = 5
    ):
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        self.max_history = max_history
        self.history: List[Dict] = []
        
    def answer(
        self,
        context_package: Dict,
        retrieval_confidence: float,
        mode: str
    ) -> Dict[str, Any]:
        
        if retrieval_confidence < self.confidence_threshold:
            return self._refuse(
                reason="Insufficient retrieval confidence",
                retrieval_confidence=retrieval_confidence
            )
            
        result = self.llm_client.generate(
            context_package=context_package,
            retrieval_confidence=retrieval_confidence,
            mode=mode
        )

        if not result["citations"]:
            return self._refuse(
                reason="No valid evidence citations produced",
                retrieval_confidence=retrieval_confidence
            )

        self._update_history(result)

        return result

    def _refuse(
        self,
        reason: str,
        retrieval_confidence: float
    ) -> Dict[str, Any]:

        message = (
            "The available evidence is insufficient to provide a reliable answer. "
            "Reason: " + reason + ". "
            "Please provide additional evidence or refine the query."
        )

        return {
            "answer": message,
            "citations": [],
            "confidence": retrieval_confidence,
            "mode": "refusal"
        }

    def _update_history(self, result: Dict[str, Any]):

        self.history.append({
            "answer": result["answer"],
            "citations": result["citations"]
        })

        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def get_progressive_context(self) -> str:
        
        if not self.history:
            return ""

        progressive = "\n\nPREVIOUS ANALYSIS:\n"

        for idx, entry in enumerate(self.history, start=1):
            progressive += (
                f"\n[Previous Step {idx}]\n"
                f"{entry['answer']}\n"
            )

        return progressive
