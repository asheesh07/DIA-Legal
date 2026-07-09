from typing import Dict, Callable, Optional
from src.reader import PDFAsset
from src.chunker import PDFChunker
import numpy as np
import os


class IngestionPipeline:
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
               storage_path: str,
               progress: Optional[Callable[[str], None]] = None) -> Dict:

        def emit(stage: str):
            if progress:
                progress(stage)

        reader = self.reader_router.from_path(source)
        reader.validate(source)

        asset = reader.load(
            source=source,
            case_id=case_id,
            output_path=storage_path
        )

        if isinstance(asset, PDFAsset):
            return self._ingest_pdf(asset, case_id, emit)
        else:
            return self._ingest_video(asset, case_id, emit)

    # ============================================================
    # PDF Path
    # ============================================================

    def _ingest_pdf(self, asset: PDFAsset, case_id: str, emit) -> Dict:
        import time as _time
        _t0 = _time.time()

        emit("reading")
        chunks = self.pdf_chunker.chunk(asset)
        _t_read = _time.time()
        print(f"[TIMING][PDF] reading+chunking: {_t_read - _t0:.2f}s  chunks={len(chunks)}", flush=True)

        if not chunks:
            return {
                "status": "success", "case_id": case_id, "source": "pdf",
                "doc_type": asset.source_type, "original_name": asset.original_name,
                "sections": len(asset.sections), "chunks_indexed": 0,
            }

        emit("chunking")
        texts = [chunk.get("text", "") for chunk in chunks]

        emit("embedding")
        _t_emb_start = _time.time()
        text_embeddings = self.embedder.text_embedder.embed_batch(texts)
        _t_emb = _time.time()
        print(f"[TIMING][PDF] embedding {len(texts)} chunks: {_t_emb - _t_emb_start:.2f}s", flush=True)

        visual_dim = (
            self.embedder.visual_embedder.embed_dim
            if self.embedder.visual_embedder else 512
        )
        zero_visual = [0.0] * visual_dim

        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                "chunk_id":    chunk["chunk_id"],
                "case_id":     case_id,
                "start_time":  0.0,
                "end_time":    0.0,

                "text_embedding":   text_embeddings[i].tolist(),
                "visual_embedding": zero_visual,

                "transcript_segments": chunk.get("transcript_segments", []),
                "frames":          [],
                "transcript_text": chunk.get("text", ""),
                "speakers":        ["DOCUMENT"],
                "has_ocr":         False,
                "has_frames":      False,

                "source_id":     chunk.get("source_id", ""),
                "source_type":   chunk.get("source_type", "document"),
                "original_name": chunk.get("original_name", ""),
                "section_title": chunk.get("section_title", ""),
                "section_type":  chunk.get("section_type", ""),
                "page_span":     chunk.get("page_span", {}),
            })

        emit("indexing")
        _t_idx_start = _time.time()
        self.vector_store.upsert(records)
        _t_idx = _time.time()
        print(f"[TIMING][PDF] indexing {len(records)} records: {_t_idx - _t_idx_start:.2f}s", flush=True)
        print(f"[TIMING][PDF] TOTAL for {asset.original_name}: {_t_idx - _t0:.2f}s", flush=True)

        return {
            "status":         "success",
            "case_id":        case_id,
            "source":         "pdf",
            "doc_type":       asset.source_type,
            "original_name":  asset.original_name,
            "sections":       len(asset.sections),
            "chunks_indexed": len(records),
        }

    # ============================================================
    # Video Path
    # ============================================================

    def _ingest_video(self, asset, case_id: str, emit) -> Dict:
        processed_segments = self.video_processor.process(asset, progress=emit)

        emit("chunking")
        chunks = self.chunker.chunk(processed_segments)

        if not chunks:
            return {"status": "success", "case_id": case_id, "source": "video", "chunks_indexed": 0}

        emit("embedding")
        records = self._build_video_records(chunks, case_id, asset.source_type)

        emit("indexing")
        self.vector_store.upsert(records)

        return {
            "status":         "success",
            "case_id":        case_id,
            "source":         "video",
            "chunks_indexed": len(records),
        }

    # ============================================================
    # Video path from pre-extracted files (streaming upload)
    # ============================================================

    def ingest_from_files(self, audio_path: str, frames_dir,
                          case_id: str, evidence_id: str,
                          source_type: str = "Local",
                          progress: Optional[Callable[[str], None]] = None) -> Dict:
        """
        Used when the route has already run FFmpeg (streaming upload).
        Skips the reader/FFmpeg step and goes straight to ML.
        """
        def emit(stage: str):
            if progress:
                progress(stage)

        processed_segments = self.video_processor.process_from_files(
            audio_path=audio_path,
            frames_dir=frames_dir,
            case_id=case_id,
            evidence_id=evidence_id,
            progress=emit,
        )

        from src.reader import VideoAsset
        from datetime import datetime
        from pathlib import Path

        fake_asset = VideoAsset(
            evidence_id=evidence_id,
            case_id=case_id,
            source_type=source_type,
            original_name="uploaded_video",
            stored_path=audio_path,
            created_at=datetime.utcnow(),
            status="streamed",
        )

        emit("chunking")
        chunks = self.chunker.chunk(processed_segments)

        if not chunks:
            return {"status": "success", "case_id": case_id, "source": "video", "chunks_indexed": 0}

        emit("embedding")
        records = self._build_video_records(chunks, case_id, source_type)

        emit("indexing")
        self.vector_store.upsert(records)

        return {
            "status":         "success",
            "case_id":        case_id,
            "source":         "video",
            "chunks_indexed": len(records),
        }

    def _build_video_records(self, chunks, case_id: str, source_type: str):
        texts = [self.embedder._build_text_blob(chunk) for chunk in chunks]
        text_embeddings = self.embedder.text_embedder.embed_batch(texts)

        visual_dim = (
            self.embedder.visual_embedder.embed_dim
            if self.embedder.visual_embedder else 512
        )
        zero_visual = [0.0] * visual_dim
        frame_vectors = {}
        index_visual = (
            self.embedder.visual_embedder is not None
            and os.getenv("INDEX_VISUAL_EMBEDDINGS", "0").lower() in {"1", "true", "yes", "on"}
        )

        if index_visual:
            frame_paths = []
            for chunk in chunks:
                for frame in chunk.get("frames", []):
                    path = frame.get("image_path")
                    key = str(path) if path else ""
                    if key and key not in frame_vectors:
                        frame_vectors[key] = None
                        frame_paths.append(path)
            if frame_paths:
                vectors = self.embedder.visual_embedder.embed_batch(frame_paths)
                for path, vec in zip(frame_paths, vectors):
                    frame_vectors[str(path)] = vec

        records = []
        for i, chunk in enumerate(chunks):
            chunk_frame_vectors = [
                frame_vectors[str(f["image_path"])]
                for f in chunk.get("frames", [])
                if f.get("image_path") and frame_vectors.get(str(f["image_path"])) is not None
            ]
            if chunk_frame_vectors:
                visual_emb = self.embedder.visual_embedder.aggregate(
                    np.vstack(chunk_frame_vectors)
                ).tolist()
            else:
                visual_emb = zero_visual

            transcript_text = " ".join(
                f"[{seg.get('speaker','?')}] {seg.get('text','')}"
                for seg in chunk.get("transcript_segments", [])
            )
            speakers = list({
                seg.get("speaker") for seg in chunk.get("transcript_segments", [])
                if seg.get("speaker")
            })

            records.append({
                "chunk_id":   chunk["chunk_id"],
                "case_id":    case_id,
                "start_time": float(chunk["time_range"]["start"]),
                "end_time":   float(chunk["time_range"]["end"]),

                "text_embedding":   text_embeddings[i].tolist(),
                "visual_embedding": visual_emb,

                "transcript_segments": chunk.get("transcript_segments", []),
                "frames":          chunk.get("frames", []),
                "transcript_text": transcript_text,
                "speakers":        speakers,
                "has_ocr":         any(f.get("ocr_text") for f in chunk.get("frames", [])),
                "has_frames":      bool(chunk.get("frames")),
                "source_type":     source_type,

                "source_id":     "",
                "original_name": "",
                "section_title": "",
                "section_type":  "",
                "page_span":     {},
            })
        return records
