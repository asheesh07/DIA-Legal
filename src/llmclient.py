import os
import re
import time
from typing import Dict, List, Any
from huggingface_hub import InferenceClient


class LLMClient:
    """
    Enterprise-grade Hugging Face LLM client for DIA-Legal.
    Supports:
    - Structured citation extraction
    - Confidence blending
    - Usage + latency tracking
    - Backend abstraction
    """

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature: float = 0.2,
        max_tokens: int = 800,
        api_token: str | None = None
    ):

        self.api_token = api_token or os.getenv("HF_TOKEN")

        if not self.api_token:
            raise ValueError(
                "HF_TOKEN not found. Set it as an environment variable."
            )

        self.client = InferenceClient(
            model=model,
            token=self.api_token
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ============================================================
    # Main Generation Method
    # ============================================================
    def generate(
        self,
        context_package: Dict,
        retrieval_confidence: float,
        mode: str
    ) -> Dict[str, Any]:

        system_prompt = context_package["system_prompt"]
        context_text = context_package["context"]
        user_prompt = context_package["user_prompt"]
        citation_map = context_package["citation_map"]

        full_prompt = f"""
{system_prompt}

EVIDENCE BLOCKS:
{context_text}

{user_prompt}
"""

        start_time = time.time()

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer_text = response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}")

        latency = round(time.time() - start_time, 3)

        cited_blocks = self._extract_block_references(answer_text)

        validated_citations = self._validate_citations(
            cited_blocks,
            citation_map
        )

        final_confidence = self._compute_final_confidence(
            retrieval_confidence,
            len(validated_citations)
        )

        return {
            "answer": answer_text,
            "citations": validated_citations,
            "confidence": final_confidence,
            "mode": mode,
            "llm_metadata": {
                "model": self.model,
                "latency_seconds": latency,
                "cited_blocks_count": len(validated_citations)
            }
        }

    # ============================================================
    # Extract [Block X] references
    # ============================================================
    def _extract_block_references(self, text: str) -> List[int]:
        pattern = r"\[block (\d+)\]"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        return list(set(int(m) for m in matches))

    # ============================================================
    # Validate citations against citation_map
    # ============================================================
    def _validate_citations(
        self,
        cited_blocks: List[int],
        citation_map: Dict
    ) -> List[Dict]:

        valid = []

        for block_id in cited_blocks:
            if block_id in citation_map:
                metadata = citation_map[block_id]

                valid.append({
                    "block_id": block_id,
                    "chunk_ids": metadata["chunk_ids"],
                    "case_id": metadata["case_id"],
                    "time_range": [
                        metadata["start_time"],
                        metadata["end_time"]
                    ]
                })

        return valid

    # ============================================================
    # Confidence Blending
    # ============================================================
    def _compute_final_confidence(
        self,
        retrieval_confidence: float,
        citation_count: int
    ) -> float:

        citation_bonus = min(0.1, citation_count * 0.03)

        confidence = retrieval_confidence + citation_bonus

        return round(min(confidence, 1.0), 3)
