from typing import Dict
from src.reader import PDFAsset
from src.chunker import PDFChunker


class IngestionPipeline:
    """
    Full ingestion pipeline for DIA-Legal.

    Supports:
    - Video (YouTube + Local): Read → Process → Chunk → Embed → Store
    - PDF documents:           Read → Chunk → Embed → Store
                               (no video processor needed)
    """

    def __init__(
        self,
        reader_router,
        video_processor,
        chunker,
        embedder,
        vector_store,
        pdf_chunker: PDFChunker = None,
    ):
        self.reader_router   = reader_router
        self.video_processor = video_processor
        self.chunker         = chunker
        self.embedder        = embedder
        self.vector_store    = vector_store
        self.pdf_chunker     = pdf_chunker or PDFChunker()

    # ============================================================
    # Main Entry Point
    # ============================================================

    def ingest(self, source: str, case_id: str,
               storage_path: str) -> Dict:

        # 1. Route to correct reader
        reader = self.reader_router.from_path(source)
        reader.validate(source)

        asset = reader.load(
            source=source,
            case_id=case_id,
            output_path=storage_path
        )

        # 2. Branch: PDF vs Video
        if isinstance(asset, PDFAsset):
            return self._ingest_pdf(asset, case_id)
        else:
            return self._ingest_video(asset, case_id)

    # ============================================================
    # PDF Path
    # ============================================================

    def _ingest_pdf(self, asset: PDFAsset, case_id: str) -> Dict:

        # Chunk sections into LanceDB-ready dicts
        chunks = self.pdf_chunker.chunk(asset)

        indexed_count = 0

        for chunk in chunks:

            embeddings = self.embedder.embed_chunk(chunk)

            record = {
                # ── Core fields ────────────────────────────────
                "chunk_id":    chunk["chunk_id"],
                "case_id":     case_id,
                "start_time":  0.0,
                "end_time":    0.0,

                # ── Embeddings ─────────────────────────────────
                "text_embedding":   embeddings["text_embedding"],
                "visual_embedding": embeddings.get(
                    "visual_embedding"
                ),

                # ── Transcript (mirrors video structure) ───────
                "transcript_segments": chunk.get(
                    "transcript_segments", []
                ),
                "frames":          [],
                "transcript_text": chunk.get("text", ""),
                "speakers":        ["DOCUMENT"],
                "has_ocr":         False,
                "has_frames":      False,

                # ── PDF provenance ─────────────────────────────
                "source_id":     chunk.get("source_id", ""),
                "source_type":   chunk.get("source_type", "document"),
                "original_name": chunk.get("original_name", ""),
                "section_title": chunk.get("section_title", ""),
                "section_type":  chunk.get("section_type", ""),
                "page_span":     chunk.get("page_span", {}),
            }

            self.vector_store.upsert([record])
            indexed_count += 1

        return {
            "status":         "success",
            "case_id":        case_id,
            "source":         "pdf",
            "doc_type":       asset.source_type,
            "original_name":  asset.original_name,
            "sections":       len(asset.sections),
            "chunks_indexed": indexed_count,
        }

    # ============================================================
    # Video Path — unchanged from your original
    # ============================================================

    def _ingest_video(self, asset, case_id: str) -> Dict:

        # Process video (ASR + Frames + OCR)
        processed_segments = self.video_processor.process(asset)

        # Chunk
        chunks = self.chunker.chunk(processed_segments)

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
                "chunk_id":   chunk["chunk_id"],
                "case_id":    case_id,
                "start_time": float(chunk["time_range"]["start"]),
                "end_time":   float(chunk["time_range"]["end"]),

                "text_embedding":   embeddings["text_embedding"],
                "visual_embedding": (
                    embeddings["visual_embedding"]
                    if embeddings["visual_embedding"] is not None
                    else None
                ),

                "transcript_segments": chunk.get(
                    "transcript_segments", []
                ),
                "frames":          chunk.get("frames", []),
                "transcript_text": transcript_text,
                "speakers":        speakers,
                "has_ocr":         has_ocr,
                "has_frames":      has_frames,
                "source_type":     asset.source_type,

                # PDF fields empty for video chunks
                "source_id":     "",
                "original_name": "",
                "section_title": "",
                "section_type":  "",
                "page_span":     {},
            }

            self.vector_store.upsert([record])
            indexed_count += 1

        return {
            "status":         "success",
            "case_id":        case_id,
            "source":         "video",
            "chunks_indexed": indexed_count,
        }

    # ============================================================
    # Utilities
    # ============================================================

    def _flatten_transcript(self, chunk) -> str:
        texts = []
        for seg in chunk.get("transcript_segments", []):
            speaker = seg.get("speaker", "UNKNOWN")
            text    = seg.get("text", "")
            texts.append(f"[{speaker}] {text}")
        return " ".join(texts)

    def _extract_speakers(self, chunk):
        speakers = set()
        for seg in chunk.get("transcript_segments", []):
            if seg.get("speaker"):
                speakers.add(seg["speaker"])
        return list(speakers)