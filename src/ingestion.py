from typing import Dict

class IngestionPipeline:
    """
    Full ingestion pipeline for DIA-Legal.

    Steps:
    - Read source
    - Process video
    - Chunk
    - Embed
    - Normalize record
    - Store in vector DB
    """

    def __init__(
        self,
        reader_router,
        video_processor,
        chunker,
        embedder,
        vector_store
    ):
        self.reader_router = reader_router
        self.video_processor = video_processor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    # ============================================================
    # Main Ingestion Entry
    # ============================================================
    def ingest(self, source: str, case_id: str, storage_path: str) -> Dict:

        # --------------------------------------------------------
        # 1️⃣ Read & Store Raw Video
        # --------------------------------------------------------
        reader = self.reader_router.from_path(source)
        reader.validate(source)

        asset = reader.load(
            source=source,
            case_id=case_id,
            output_path=storage_path
        )

        # --------------------------------------------------------
        # 2️⃣ Process Video (ASR + Frames + OCR)
        # --------------------------------------------------------
        processed_segments = self.video_processor.process(asset)

        # --------------------------------------------------------
        # 3️⃣ Chunk
        # --------------------------------------------------------
        chunks = self.chunker.chunk(processed_segments)

        # --------------------------------------------------------
        # 4️⃣ Embed + Normalize + Store
        # --------------------------------------------------------
        indexed_count = 0

        self.vector_store.clear()
        for chunk in chunks:

            embeddings = self.embedder.embed_chunk(chunk)
            transcript_text = self._flatten_transcript(chunk)
            speakers = self._extract_speakers(chunk)

            has_ocr = any(
                frame.get("ocr_text")
                for frame in chunk.get("frames", [])
            )

            has_frames = len(chunk.get("frames", [])) > 0

            record = {
                "chunk_id": chunk["chunk_id"],
                "case_id": case_id,
                "start_time": float(chunk["time_range"]["start"]),
                "end_time": float(chunk["time_range"]["end"]),
                "text_embedding": embeddings["text_embedding"],
                "visual_embedding": (
                    embeddings["visual_embedding"]
                    if embeddings["visual_embedding"] is not None
                    else None
                ),
                "transcript_segments": chunk.get("transcript_segments", []),
                "frames": chunk.get("frames", []),
                "transcript_text": transcript_text,
                "speakers": speakers,
                "has_ocr": has_ocr,
                "has_frames": has_frames,
                "source_type": asset.source_type
            }

            self.vector_store.upsert([record])

            indexed_count += 1
        return {
            "status": "success",
            "case_id": case_id,
            "chunks_indexed": indexed_count
        }

    # ============================================================
    # Utilities
    # ============================================================

    def _flatten_transcript(self, chunk):

        texts = []

        for seg in chunk.get("transcript_segments", []):
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg.get("text", "")
            texts.append(f"[{speaker}] {text}")

        return " ".join(texts)

    def _extract_speakers(self, chunk):

        speakers = set()

        for seg in chunk.get("transcript_segments", []):
            if seg.get("speaker"):
                speakers.add(seg["speaker"])

        return list(speakers)
