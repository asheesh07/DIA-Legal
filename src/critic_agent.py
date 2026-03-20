"""
critic_agent.py
───────────────
Critic loop for DIA-Legal A-RAG pipeline.

Pattern:
    Evidence Reasoner proposes a contradiction candidate
    → Critic evaluates it and assigns a confidence score + critique
    → Evidence Reasoner refines until confidence >= threshold
    → Loop exits (or hits max_rounds as safety valve)

No LangChain. No LlamaIndex. Raw Python + your existing LLMClient.
"""

import json
import re
from typing import Dict, Optional


class CriticAgent:
    """
    Wraps your existing LLMClient.classify() to evaluate
    contradiction candidates and return structured critiques.

    Args:
        llm_client:        Your existing LLMClient instance.
        confidence_threshold: Stop refining when score >= this. Default 0.80.
        max_rounds:        Safety valve — never loop more than this. Default 3.
    """

    def __init__(
        self,
        llm_client,
        confidence_threshold: float = 0.80,
        max_rounds: int = 3,
    ):
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        self.max_rounds = max_rounds

    # ─── Public API ──────────────────────────────────────────────────────────

    def evaluate(self, contradiction: Dict) -> Dict:
        """
        Evaluate a single contradiction candidate.

        Args:
            contradiction: dict with keys:
                - chunk_a_text  (str)  testimony/video chunk
                - chunk_b_text  (str)  document chunk
                - proposed_conflict (str)  what the reasoner claims conflicts

        Returns dict with keys:
            - confidence    (float 0–1)
            - critique      (str)    what's weak or missing
            - is_valid      (bool)   True if confidence >= threshold
            - refined_conflict (str) suggested rephrasing (can be empty)
        """
        prompt = self._build_eval_prompt(contradiction)

        try:
            raw = self.llm_client.classify(prompt)
            return self._parse_response(raw)
        except Exception as e:
            # Fail safe — don't crash the pipeline
            return {
                "confidence": 0.0,
                "critique": f"Critic evaluation failed: {e}",
                "is_valid": False,
                "refined_conflict": "",
            }

    def run_loop(self, initial_contradiction: Dict) -> Dict:
        """
        Propose → Critique → Refine loop.

        Starts from initial_contradiction and iteratively
        refines the proposed_conflict string until:
            (a) confidence >= self.confidence_threshold, OR
            (b) max_rounds reached

        Returns the final critique dict with an added
        'rounds_taken' key for observability.
        """
        current = dict(initial_contradiction)
        last_critique = {}

        for round_num in range(1, self.max_rounds + 1):
            critique = self.evaluate(current)
            critique["rounds_taken"] = round_num

            if critique["confidence"] >= self.confidence_threshold:
                return critique

            # If critic suggested a refinement, use it next round
            if critique.get("refined_conflict"):
                current["proposed_conflict"] = critique["refined_conflict"]

            last_critique = critique

        # Max rounds hit — return best we got
        last_critique["rounds_taken"] = self.max_rounds
        return last_critique

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _build_eval_prompt(self, contradiction: Dict) -> str:
        return f"""You are a strict legal evidence critic.

Evaluate the proposed contradiction between these two evidence chunks.

TESTIMONY / VIDEO CHUNK:
{contradiction.get('chunk_a_text', '')}

DOCUMENT CHUNK:
{contradiction.get('chunk_b_text', '')}

PROPOSED CONFLICT:
{contradiction.get('proposed_conflict', '')}

Your job:
1. Assess whether the proposed conflict is real, specific, and legally meaningful.
2. Assign a confidence score (0.0 = completely wrong, 1.0 = definitive contradiction).
3. Write a short critique identifying what is weak or missing.
4. If the conflict can be stated more precisely, provide a refined version.

Respond ONLY in valid JSON — no preamble, no markdown:
{{
    "confidence": 0.0,
    "critique": "your critique here",
    "refined_conflict": "a more precise statement of the conflict, or empty string if none"
}}"""

    def _parse_response(self, raw: str) -> Dict:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if not match:
            return {
                "confidence": 0.0,
                "critique": "Critic returned unparseable response.",
                "is_valid": False,
                "refined_conflict": "",
            }

        parsed = json.loads(match.group())
        confidence = float(parsed.get("confidence", 0.0))

        return {
            "confidence": confidence,
            "critique": parsed.get("critique", ""),
            "is_valid": confidence >= self.confidence_threshold,
            "refined_conflict": parsed.get("refined_conflict", ""),
        }