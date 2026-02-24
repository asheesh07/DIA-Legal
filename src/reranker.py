import torch
import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 16, normalize: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)
        self.batch_size = batch_size
        self.normalize = normalize

    def _build_input_text(self, chunk: dict) -> str:
        content = []

        # Transcript segments
        for seg in chunk.get("transcript_segments", []):
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg.get("text", "")
            if text:
                content.append(f"{speaker}: {text}")

        # Visual frames
        for frame in chunk.get("frames", []):
            caption = frame.get("caption")
            ocr_text = frame.get("ocr_text")

            if caption:
                content.append(f"[caption]: {caption}")
            if ocr_text:
                content.append(f"[ocr]: {ocr_text}")

        return " ".join(content).strip()

    def score_batch(self, query: str, candidates: list[dict]) -> list[float]:
        if not candidates:
            return []

        try:
            pairs = [
                (query, self._build_input_text(chunk))
                for chunk in candidates
            ]

            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )

            scores = np.array(scores, dtype=float)

            if self.normalize and len(scores) > 1:
                min_s = scores.min()
                max_s = scores.max()

                if max_s - min_s > 1e-6:
                    scores = (scores - min_s) / (max_s - min_s)
                else:
                    scores = np.clip(scores, 0.0, 1.0)

            return scores.tolist()

        except Exception:
            return [0.0] * len(candidates)