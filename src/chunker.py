import uuid
from typing import List,Dict
from src.reader import PDFAsset, LegalSection
import re
class Chunker:
    def __init__(self,max_duration:int=20,max_tokens:int=512,overlap_duration:int=5):
        self.max_duration= max_duration
        self.max_tokens=max_tokens
        self.overlap_duration = overlap_duration
        
    def _estimate_tokens(self,text):
        return len(text.split())
    
    def _build_chunk_text(self, segments: List[Dict]) -> str:
        texts = []
        for seg in segments:
            texts.append(seg["transcript"])
            for frame in seg.get("frames", []):
                if frame.get("caption"):
                    texts.append(frame["caption"])
                if frame.get("ocr_text"):
                    texts.append(frame["ocr_text"])
        return " ".join(texts)
    
    def _build_transcripts(self, segments):
        transcripts = []
        for seg in segments:
            transcripts.append({
                "start":           seg['start'],
                "end":             seg['end'],
                "text":            seg['transcript'],
                "speaker":         seg['speaker'],
                "timestamp_start": seg.get('timestamp_start', ''),
                "timestamp_end":   seg.get('timestamp_end', ''),
            })
        return transcripts
    def _build_frames(self,segments):
        frames =[]
        for seg in segments:
            for frame in seg['frames']:
                frames.append({
                    "timestamp":frame['timestamp'],
                    "caption":frame['caption'],
                    "ocr_text":frame['ocr_text'],
                    "image_path":str(frame['image_path'])
                })
            
        return frames
    def chunk(self,aligned_segments:List[Dict]):
        chunks =[]
        current_segments = []
        current_start = None
        transcripts = []
        frames=[]
        for i,segment in enumerate(aligned_segments):
            if not current_segments:
                current_segments.append(segment)
                current_start = segment['start']
                continue
            candidate_segments = current_segments + [segment]
            candidate_text = self._build_chunk_text(candidate_segments)
                
            duration = segment['end'] - current_start
            token_count = self._estimate_tokens(candidate_text)
            if (duration <= self.max_duration and token_count <= self.max_tokens):
                current_segments.append(segment)
            else:
                transcripts = self._build_transcripts(current_segments)
                frames = self._build_frames(current_segments)
                chunk_id = f"{segment['case_id']}_{i}"
                chunks.append({
                    "chunk_id":chunk_id,
                    "case_id":segment["case_id"],
                    "time_range":{
                        "start": current_segments[0]['start'],
                        "end":current_segments[-1]['end']
                    },
                    "transcript_segments":transcripts,
                    "frames":frames
                    
                })

                new_start_time = segment['start'] - self.overlap_duration

                overlap_segments = [
                    seg for seg in current_segments
                    if seg['end'] >= new_start_time
                ]
                current_segments = overlap_segments + [segment]
                current_start = current_segments[0]['start']
        if current_segments:
            transcripts = self._build_transcripts(current_segments)
            frames = self._build_frames(current_segments)
            final_text = self._build_chunk_text(current_segments)
            chunks.append({
                    "chunk_id":str(uuid.uuid4()),
                    "case_id":current_segments[0]["case_id"],
                    "text":final_text,
                    "time_range":{
                        "start": current_segments[0]['start'],
                        "end":current_segments[-1]['end']
                    },
                    "transcript_segments":transcripts,
                    "frames":frames
                    
                })
        return chunks
                    
class PDFChunker:
    """
    Converts a PDFAsset into chunks that are structurally
    identical to your video Chunker output.

    Every chunk dict has the same keys as your video chunks:
        chunk_id, case_id, text, start_time, end_time,
        transcript_segments, frames

    Plus PDF-specific provenance fields:
        source_id, source_type, doc_type, original_name,
        section_id, section_title, section_type,
        chunk_index, total_chunks_in_section, page_span
    """

    def __init__(self, max_tokens: int = 512,
                 overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, asset: PDFAsset) -> List[Dict]:
        all_chunks: List[Dict] = []
        for section in asset.sections:
            section_chunks = self._chunk_section(section, asset)
            all_chunks.extend(section_chunks)
        return all_chunks

    # ── Internal helpers ─────────────────────────────────────────

    def _chunk_section(
        self,
        section: LegalSection,
        asset: PDFAsset,
    ) -> List[Dict]:

        sentences = self._split_sentences(section.text)
        groups = self._group_sentences(sentences)
        total = len(groups)
        chunks = []

        for idx, group_sentences in enumerate(groups):
            text = " ".join(group_sentences).strip()
            if not text:
                continue

            chunk_id = (
                f"{asset.case_id}_"
                f"{section.section_id[:8]}_"
                f"{idx}"
            )

            chunk = {
                # ── Core fields — identical to video chunker ──
                "chunk_id":           chunk_id,
                "case_id":            asset.case_id,
                "text":               text,

                # PDFs have no timestamps — 0.0 as sentinel
                # Retriever already handles 0.0 gracefully
                "start_time":         0.0,
                "end_time":           0.0,

                # Mirrors video chunker transcript_segments
                # Speaker = "DOCUMENT" so retriever won't
                # confuse PDF text with spoken testimony
                "transcript_segments": [{
                    "start":   0.0,
                    "end":     0.0,
                    "text":    text,
                    "speaker": "DOCUMENT",
                }],

                # PDFs have no frames
                "frames": [],

                # ── Source / provenance ───────────────────────
                "source_id":          asset.evidence_id,
                "source_type":        asset.source_type,
                "doc_type":           asset.source_type,
                "original_name":      asset.original_name,

                # ── Section metadata ──────────────────────────
                "section_id":         section.section_id,
                "section_title":      section.title,
                "section_type":       section.section_type,
                "chunk_index":        idx,
                "total_chunks_in_section": total,

                # ── Page span — for citations only ────────────
                "page_span": {
                    "start_page": section.page_span.start_page,
                    "end_page":   section.page_span.end_page,
                },
            }

            chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Sentence splitter that protects common legal
        abbreviations from being treated as sentence ends.
        """
        # Protect abbreviations
        text = re.sub(
            r'\b(No|Vs|vs|Sec|Art|Para|Cl|Vol|'
            r'Pvt|Ltd|Co|Corp|Inc|Dr|Mr|Mrs|Ms|'
            r'Hon|Smt|Shri|u/s|S/o|D/o|W/o)\.',
            r'\1<DOT>', text
        )

        # Split at real sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Restore protected abbreviation dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def _group_sentences(
        self, sentences: List[str]
    ) -> List[List[str]]:
        """
        Group sentences into token-limited windows with overlap.
        Never splits mid-sentence.
        """
        if not sentences:
            return []

        groups: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            s_tokens = len(sentence.split())

            fits = (current_tokens + s_tokens <= self.max_tokens)

            if fits or not current:
                current.append(sentence)
                current_tokens += s_tokens
            else:
                groups.append(current)

                # Carry last N tokens as overlap into next chunk
                overlap: List[str] = []
                overlap_tokens = 0
                for s in reversed(current):
                    t = len(s.split())
                    if overlap_tokens + t <= self.overlap_tokens:
                        overlap.insert(0, s)
                        overlap_tokens += t
                    else:
                        break

                current = overlap + [sentence]
                current_tokens = sum(
                    len(s.split()) for s in current
                )

        if current:
            groups.append(current)

        return groups

        
            
