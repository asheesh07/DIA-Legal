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
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.2,
        max_tokens: int = 300,
        api_token: str | None = None
    ):

        self.api_token = api_token or os.getenv("HF_TOKEN")
        self.client = None
        if self.api_token:
            # timeout keeps a stalled provider call from hanging past the
            # HF Space proxy limit — fail fast and surface a retryable error.
            self.client = InferenceClient(
                model=model,
                token=self.api_token,
                timeout=45,
            )
        else:
            print("[LLMClient] HF_TOKEN missing; running in fallback mode.", flush=True)

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
EVIDENCE BLOCKS:
{context_text}

{user_prompt}
"""

        start_time = time.time()
        print("--- FULL PROMPT LENGTH ---", len(full_prompt), flush=True)
        if len(full_prompt) < 1000:
            print("--- FULL PROMPT ---", full_prompt, flush=True)

        if self.client is None:
            answer_text = (
                "The language model is currently unavailable, so I can only provide a "
                "limited evidence-based response from the retrieved context."
            )
            cited_blocks = []
            llm_confidence = 0.0
            latency = round(float(time.time() - start_time), 3)
            validated_citations = self._validate_citations(cited_blocks, citation_map)
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

        try:
            response = None
            for attempt in (1, 2):
                try:
                    response = self.client.chat_completion(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": full_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    break
                except Exception as retry_err:
                    if attempt == 2:
                        raise
                    print("HF RETRY after error:", str(retry_err), flush=True)

            raw_text = response.choices[0].message.content.strip()

            import json
            import re
            
            # Extract JSON
            cleaned = re.sub(r"```json|```", "", raw_text).strip()
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            
            if match:
                parsed = json.loads(match.group())
                answer_text = parsed.get("answer", "Answer unavailable.")
                cited_blocks = parsed.get("supporting_evidence", [])
                llm_confidence = parsed.get("confidence", 0.0)
            else:
                answer_text = raw_text
                cited_blocks = self._extract_block_references(raw_text)
                llm_confidence = 0.5
                
        except Exception as e:
            error_str = str(e)
            print("HF ERROR:", error_str, flush=True)
            if "validation error" in error_str.lower() or "bad request" in error_str.lower():
                answer_text = "The context was too large for the LLM to process. Try asking a more specific query."
            else:
                answer_text = f"LLM generation failed: {error_str}"
            cited_blocks = []
            llm_confidence = 0.0

        latency = round(float(time.time() - start_time), 3)

        validated_citations = self._validate_citations(
            cited_blocks,
            citation_map
        )

        final_confidence = self._compute_final_confidence(
            retrieval_confidence,
            len(validated_citations)
        )
        
        # blend llm_confidence and retrieval_confidence
        combined_confidence = (final_confidence + llm_confidence) / 2.0

        return {
            "answer": answer_text,
            "citations": validated_citations,
            "confidence": combined_confidence,
            "mode": mode,
            "llm_metadata": {
                "model": self.model,
                "latency_seconds": latency,
                "cited_blocks_count": len(validated_citations)
            }
        }

    # ============================================================
    # Extract [Block X] references (fallback)
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
                    "block_id":   block_id,
                    "chunk_ids":  metadata["chunk_ids"],
                    "case_id":    metadata["case_id"],
                    "citation_ref": metadata.get("citation_ref", ""),
                    # Video
                    "time_range": [
                        metadata.get("start_time", 0),
                        metadata.get("end_time", 0)
                    ],
                    "speakers":  metadata.get("speakers", []),
                    # PDF
                    "page_start":    metadata.get("page_start", 0),
                    "page_end":      metadata.get("page_end", 0),
                    "section_title": metadata.get("section_title", ""),
                    "source_type":   metadata.get("source_type", ""),
                    "original_name": metadata.get("original_name", ""),
                    "exhibit_id":    metadata.get("exhibit_id", ""),
                    "exhibit_title": metadata.get("exhibit_title", ""),
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
    
    def classify(self, prompt: str) -> str:
        """
        Lightweight call for classification tasks.
        Returns raw text response.
        Used by EvidenceClassifier, ContradictionDetector etc.
        """
        if self.client is None:
            return "{}"
        try:
            response = self.client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a legal analyst. "
                                "Respond only in valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,   # Low temp for classification
                max_tokens=200,    # Classification needs few tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("HF FILTER ERROR:", str(e), flush=True)
            raise RuntimeError(f"Classification failed: {e}")

    # ============================================================
    # 1. RETRIEVAL FILTER PROMPT (LIGHTWEIGHT)
    # ============================================================
    def filter_chunks(self, query: str, retrieved_items: list) -> list:
        """
        Purpose: decide relevance / filter noise
        Given the query and retrieved chunks, select the most relevant evidence.
        Return only the top relevant pieces with short justification.
        No reasoning, no verbosity.
        """
        if not retrieved_items:
            return []

        # Prepare evidence texts for the prompt
        evidence_blocks = []
        for idx, item in enumerate(retrieved_items):
            # Extract basic text across all transcript segments
            texts = [seg.get("text", "") for seg in getattr(item, "structured_transcripts", [])]
            content_str = str(" ".join(texts).strip())
            if not content_str:
                # for documents
                 if getattr(item, "structured_transcripts", []):
                     content_str = str(" ".join([str(seg.get("text", "")) for seg in getattr(item, "structured_transcripts", [])]))
            
            # Limit length for lightweight call
            content_str = content_str[:300]
            evidence_blocks.append(f"[Chunk {idx}]\n{content_str}")

        evidence_text = "\n\n".join(evidence_blocks)

        prompt = f"""Given the query and retrieved chunks, select the most relevant evidence.
Return only the top relevant pieces with short justification.
No reasoning, no verbosity.

QUERY: {query}

EVIDENCE CHUNKS:
{evidence_text}

Respond ONLY in valid JSON format exactly like this:
{{
    "relevant_chunks": [
        {{"chunk_index": 0, "justification": "short justification here"}}
    ]
}}"""

        try:
            raw = self.classify(prompt)
            # Extact JSON
            import json
            raw = re.sub(r"```json|```", "", raw).strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return retrieved_items[:3] # fallback

            parsed = json.loads(match.group())
            relevant_indices = [
                c.get("chunk_index") for c in parsed.get("relevant_chunks", [])
                if isinstance(c.get("chunk_index"), int)
            ]

            filtered_items = []
            for idx in relevant_indices:
                if 0 <= idx < len(retrieved_items):
                    filtered_items.append(retrieved_items[idx])

            # fallback if filtered everything but items existed
            if not filtered_items and retrieved_items:
                return retrieved_items[:3]

            return filtered_items
        except Exception:
            return retrieved_items[:3]
