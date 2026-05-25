"""
evaluator.py — Local RAG Evaluation (zero API calls)

Metrics:
  1. ROUGE-L        — longest common subsequence overlap (answer vs expected)
  2. Semantic Similarity — cosine sim via sentence-transformers (same model as pipeline)
  3. Context Recall — fraction of expected answer sentences semantically covered
                      by the retrieved context chunks
  4. Answer Relevance — semantic sim between query and generated answer
  5. Average         — mean of all four scores
"""

import numpy as np
from typing import List, Dict
from rouge_score import rouge_scorer


class RAGEvaluator:
    """
    Fully local evaluator — no LLM API required.

    Parameters
    ----------
    text_embedder : TextEmbedder
        The same sentence-transformer embedder used by the pipeline.
        Reused here so no second model is loaded.
    """

    def __init__(self, text_embedder):
        self.embedder = text_embedder
        self._scorer  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # ── Public entry point ────────────────────────────────────────────────

    def evaluate(
        self,
        query:           str,
        expected_answer: str,
        pipeline_result: Dict,
        context_items:   List,
    ) -> Dict:
        answer = pipeline_result.get("answer", "")

        # Build flat context string from retrieved items
        context_texts: List[str] = []
        for item in context_items:
            segs = getattr(item, "structured_transcripts", [])
            texts = [s.get("text", "") for s in segs if s.get("text")]
            if texts:
                context_texts.append(" ".join(texts))
        full_context = " ".join(context_texts)

        rouge_l          = self._rouge_l(answer, expected_answer)
        sem_sim          = self._semantic_similarity(answer, expected_answer)
        context_recall   = self._context_recall(expected_answer, context_texts)
        answer_relevance = self._answer_relevance(query, answer)

        # Average over the three semantic metrics only — ROUGE-L is reported
        # separately as a lexical sanity check but excluded from the main score
        # because it penalizes paraphrasing even when meaning is identical.
        avg = (sem_sim + context_recall + answer_relevance) / 3.0

        return {
            "rouge_l":            {"score": rouge_l,          "reason": "LCS F1 lexical overlap (reported, not in avg)"},
            "semantic_similarity":{"score": sem_sim,          "reason": "Cosine similarity via all-MiniLM-L6-v2"},
            "context_recall":     {"score": context_recall,   "reason": "Fraction of expected sentences covered by context"},
            "answer_relevance":   {"score": answer_relevance, "reason": "Cosine similarity between query and generated answer"},
            "average_score":      avg,
        }

    # ── Metric implementations ────────────────────────────────────────────

    def _rouge_l(self, hypothesis: str, reference: str) -> float:
        if not hypothesis.strip() or not reference.strip():
            return 0.0
        scores = self._scorer.score(reference, hypothesis)
        return round(float(scores["rougeL"].fmeasure), 4)

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        if not text_a.strip() or not text_b.strip():
            return 0.0
        vec_a = self.embedder.embed(text_a)
        vec_b = self.embedder.embed(text_b)
        sim = float(np.dot(vec_a, vec_b) /
                    (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9))
        return round(max(sim, 0.0), 4)

    def _context_recall(self, expected: str, context_texts: List[str]) -> float:
        """
        Split expected answer into sentences.
        For each sentence, find the MAX cosine sim across all context chunks
        (not just the blob) and consider it covered if sim >= 0.35.
        Score = fraction of sentences covered.
        """
        if not expected.strip() or not context_texts:
            return 0.0

        sentences = [s.strip() for s in expected.replace("\n", " ").split(".") if len(s.strip()) > 10]
        if not sentences:
            return 0.0

        # Embed each chunk separately for per-chunk max scoring
        ctx_vecs = [self.embedder.embed(c) for c in context_texts if c.strip()]
        if not ctx_vecs:
            return 0.0

        covered = 0
        for sent in sentences:
            sent_vec  = self.embedder.embed(sent)
            sent_norm = np.linalg.norm(sent_vec) + 1e-9
            best_sim  = 0.0
            for cv in ctx_vecs:
                cv_norm = np.linalg.norm(cv) + 1e-9
                sim     = float(np.dot(sent_vec, cv) / (sent_norm * cv_norm))
                best_sim = max(best_sim, sim)
            if best_sim >= 0.35:
                covered += 1

        return round(covered / len(sentences), 4)

    def _answer_relevance(self, query: str, answer: str) -> float:
        if not answer.strip() or not query.strip():
            return 0.0
        return self._semantic_similarity(query, answer)
