import uuid
from typing import List,Dict
class Chunker:
    def __init__(self,max_duration:int=20,max_tokens:int=512,overlap_duration:int=5,tokenizer=None):
        self.max_duration= max_duration
        self.max_tokens=max_tokens
        self.overlap_duration = overlap_duration
        self.tokenizer = tokenizer
        
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
    
    def _build_transcripts(self,segments):
        transcripts = []
        for seg in segments:
            transcripts.append({
                "start":seg['start'],
                'end':seg['end'],
                'text':seg['transcript'],
                'speaker':seg['speaker']
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
                    
        
            
        