from dataclasses import dataclass,field
from typing import List,Dict,Optional,Any
from enum import Enum
import re
import numpy as np
from torch import cosine_similarity

@dataclass
class TimeRange:
    start_time:float
    end_time : float
    
@dataclass
class TemporalData:
    primary : TimeRange
    spans : List[TimeRange]
    duration : float
    segment_count : int
    
@dataclass
class Source:
    source_id: str
    source_type: str  
    case_id: str
    file_path: Optional[str] = None

@dataclass
class RetrievedItem:
    chunk_ids: List[str]
    case_id: str
    
    temporal : TemporalData
    
    structured_transcripts: List[Dict]
    structured_frames: List[Dict]
    
    sources: List[Source]    
    
    retrieval_score: float
    rerank_score : Optional[float]
    final_score : float
    
    confidence : float
    retrieval_stage :str
    
    metadata : Dict[str,Any] =field(default_factory=dict)
    

class QueryType(Enum):
        SEMANTIC = 'semantic'
        VISUAL = 'visual'
        SUMMARY = 'summary'
        SPEECH = 'speech'
        OCR = 'ocr'
        EVIDENCE = 'evidence'
        TEMPORAL = 'temporal'
        
class Retriever:
    def __init__(
        self,
        vector_store,
        embedder,
        max_candidates=50,
        reranker=None,
        enable_mmr=False,
        mmr_lambda = 0.5,
        min_threshold = 0.0,
        temporary_window=5
        ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.max_candidates = max_candidates
        self.reranker = reranker
        self.enable_mmr =enable_mmr
        self.mmr_lambda = mmr_lambda
        self.min_score_threshold = min_threshold
        self.temporary_window= temporary_window
        
    def retrieve(self,case_id,query,filters=None,top_k=5,**kwargs):
        if not query or not query.strip():
            return []
        
        query= self._process_query(query)
        
        query_embed = self.embedder.embed_query(query)
        text_embedding = query_embed['text_query_embedding']
        visual_embedding = query_embed['visual_query_embedding']
        
        query_type = self._classify_query(query)
        
        temporal_constraints = self._extract_temporal(query)
        
        effective_filters = filters.copy() if filters else {}
        
        effective_filters['case_id'] = case_id
        
        if temporal_constraints:
            effective_filters['time_range'] = {
                "start_time":temporal_constraints.start_time ,
                "end_time":temporal_constraints.end_time
                
                
            }
            
        alpha = self._adaptive_alpha(query_type)
        
        max_candidates = kwargs.get('max_candidates',self.max_candidates)
        candidates = self._retrieve_candidates(text_embedding,visual_embedding,alpha,effective_filters,max_candidates)
        candidates = self._expand_temporal(case_id,candidates)    

        if self.enable_mmr:
            candidates = self._apply_mmr(candidates,text_embedding,visual_embedding,alpha,top_k*2)
            
           
        if self.reranker:
            candidates = self._apply_reranker(query,candidates)
        
        
        confidence = self._estimate_confidence(candidates)
        
        retrieved_items = self._build_retrieved_items(candidates[:top_k],confidence)
        return retrieved_items
        
    def _classify_query(self,query):
        prototypes = {
            QueryType.SPEECH: self.embedder.embed_query("What did someone say?"),
            QueryType.VISUAL: self.embedder.embed_query("Is something visible in the video?"),
            QueryType.TEMPORAL: self.embedder.embed_query("What happened at a specific time?"),
            QueryType.OCR: self.embedder.embed_query("Find text visible in a frame."),
            QueryType.EVIDENCE: self.embedder.embed_query("Find supporting or contradicting evidence."),
            QueryType.SEMANTIC: self.embedder.embed_query("Answer a general question about events.")
       
        }
        
        query_vec = self.embedder.embed_query(query)['text_query_embedding']
        
        raw_scores = {}
        
        for qtype , proto_vec in prototypes.items():
            raw_scores[qtype] = np.dot(proto_vec['text_query_embedding'],query_vec)
            
        scores = np.array(list(raw_scores.values()))
        
        min_s,max_s = scores.min() , scores.max()
        
        if max_s - min_s >1e-6:
            norm_scores = (scores - min_s) / (max_s - min_s)
        else :
            norm_scores = scores
        exp_scores = np.exp(norm_scores)
        
        prob_scores = exp_scores / exp_scores.sum()
        
        prob_map = {
            list(raw_scores.keys())[i]:float(prob_scores[i])
            for i in range(len(prob_scores))
        }
        
        selected = [qtype for qtype, prob in prob_map.items() if prob > 0.25]
        
        if not selected:
            selected = [QueryType.SEMANTIC]
        
        entropy = -sum(p * np.log(p + 1e-9) for p in prob_scores)
        confidence = 1 - entropy / np.log(len(prob_scores))

        return {
            "primary": max(prob_map, key=prob_map.get),
            "multi_label": selected,
            "probabilities": prob_map,
            "confidence": confidence
        }
        
        
    
    def _extract_temporal(self,query):
        time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?)'
        
        def to_seconds(time_str: str) -> float:
            parts = list(map(int, time_str.split(":")))
            if len(parts) == 2:
                minutes, seconds = parts
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                return hours * 3600 + minutes * 60 + seconds
            else:
                raise ValueError("Invalid time format")
            
        range_match = re.search(
            rf'(?:between|from)\s+{time_pattern}\s+(?:and|to)\s+{time_pattern}',
            query
        )

        if range_match:
            start = to_seconds(range_match.group(1))
            end = to_seconds(range_match.group(2))
            return TimeRange(start_time=start, end_time=end)

        # 2️⃣ Before X
        before_match = re.search(rf'before\s+{time_pattern}', query)
        if before_match:
            end = to_seconds(before_match.group(1))
            return TimeRange(start_time=0, end_time=end)

        # 3️⃣ After X
        after_match = re.search(rf'after\s+{time_pattern}', query)
        if after_match:
            start = to_seconds(after_match.group(1))
            return TimeRange(start_time=start, end_time=float("inf"))

        # 4️⃣ Single timestamp
        single_match = re.search(time_pattern, query)
        if single_match:
            t = to_seconds(single_match.group(1))
            window = 5  # ±5 seconds buffer
            return TimeRange(
                start_time=max(0, t - window),
                end_time=t + window
            )

        return None
    
    def _process_query(self,query):
        if not query :
            raise ValueError("Query cannot be empty")

        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        query = re.sub(r'([!?.]){2,}', r'\1', query)
        
        return query
    
    
    def _retrieve_candidates(self,text_embedding,visual_embedding,alpha,filters,limit):
        candidates = self.vector_store.search(text_embedding,visual_embedding,alpha,limit*3,filters)
        return candidates[:limit]
    
    def _apply_mmr(self, candidates, query_text_emb, query_vis_emb, alpha, top_k):

        if not candidates:
            return []

        candidate_text_embs = []
        fused_relevance_scores = []

        # -----------------------------
        # Step 1: Compute fused relevance scores
        # -----------------------------
        for c in candidates:
            text_emb = np.array(c.get("text_embedding")) if c.get("text_embedding") is not None else None
            vis_emb = np.array(c.get("visual_embedding")) if c.get("visual_embedding") is not None else None

            # Store text embedding for diversity later
            candidate_text_embs.append(text_emb)

            # Compute text similarity
            if text_emb is not None:
                text_sim = float(np.dot(query_text_emb, text_emb))
            else:
                text_sim = 0.0

            # Compute visual similarity (if available)
            if query_vis_emb is not None and vis_emb is not None:
                vis_sim = float(np.dot(query_vis_emb, vis_emb))
            else:
                vis_sim = 0.0

            fused_score = alpha * text_sim + (1 - alpha) * vis_sim
            fused_relevance_scores.append(fused_score)

        selected = []
        selected_indices = []

        # -----------------------------
        # Step 2: MMR Selection Loop
        # -----------------------------
        while len(selected) < min(top_k, len(candidates)):

            if not selected_indices:
                idx = int(np.argmax(fused_relevance_scores))
                selected.append(candidates[idx])
                selected_indices.append(idx)
                continue

            mmr_scores = []

            for i in range(len(candidates)):

                if i in selected_indices:
                    mmr_scores.append(-np.inf)
                    continue

                relevance = fused_relevance_scores[i]

                # Compute redundancy in TEXT space only
                redundancy = 0.0
                for j in selected_indices:
                    if candidate_text_embs[i] is not None and candidate_text_embs[j] is not None:
                        sim = float(np.dot(candidate_text_embs[i], candidate_text_embs[j]))
                        redundancy = max(redundancy, sim)

                mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * redundancy
                mmr_scores.append(mmr_score)

            idx = int(np.argmax(mmr_scores))
            selected.append(candidates[idx])
            selected_indices.append(idx)

        return selected
    
    def _apply_reranker(self, query, candidates, weight=0.7):

        if not candidates:
            return []

        candidates = candidates[:20]

        reranker_scores = self.reranker.score_batch(query, candidates)

        for i, c in enumerate(candidates):
            reranker_score = float(reranker_scores[i])
            retrieval_score = float(c.get("score", 0.0))
            

            c["retrieval_score"] = retrieval_score
            c["reranker_score"] = reranker_score
            c["final_score"] = (
                weight * retrieval_score +
                (1 - weight) * reranker_score
            )

        candidates.sort(
            key=lambda x: x.get("final_score", 0.0),
            reverse=True
        )

        return candidates
            
    
    def _adaptive_alpha(self,querytype_dict):
        probs = querytype_dict.get('probabilities')
        
        visual_prob = probs.get(QueryType.VISUAL,0.0)
        speech_prob = probs.get(QueryType.SPEECH,0.0)
        ocr_prob = probs.get(QueryType.OCR,0.0)
        summary_prob = probs.get(QueryType.SUMMARY,0.0)
        
        base_alpha = 0.85 
        
        alpha = base_alpha - 0.5* visual_prob
        alpha += 0.1 * speech_prob
        alpha -= 0.2 * ocr_prob 
        alpha -= 0.1 * summary_prob
        
        alpha = max(0.25,min(0.95,alpha))
        
        return alpha
    
    def _expand_temporal(self,case_id,candidates,window=5):
        if not candidates:
            return candidates
        results = {c['chunk_id']:c for c in candidates}
        
        expanded = []
        for item in candidates:
            expanded_start = max(0,item['start_time']-window)
            expanded_end = item['end_time'] + window
            
            neighbors = self.vector_store.search(
                text_query_embedding = None,
                visual_query_embedding = None,
                alpha = 1.0,
                top_k = 10,
                filters ={
                    "case_id": case_id,
                    "time_range": {
        "start": expanded_start,
        "end": expanded_end
    }
                }
                )
            for neighbor in neighbors:
                nid = neighbor['chunk_id']
                if nid not in results:
                    neighbor['score'] = 0.0
                    results[nid] = neighbor
        return sorted(results.values(),
                      key = lambda x: x.get('score',0.0),
                        reverse=True)
    
    def _estimate_confidence(self, candidates):

        if not candidates:
            return 0.0

        def get_score(x):
            if isinstance(x, dict):
                return x.get("final_score", x.get("score", 0.0))
            else:
                return getattr(x, "final_score", 0.0)

        sorted_candidates = sorted(
            candidates,
            key=get_score,
            reverse=True
        )

        top_score = get_score(sorted_candidates[0])

        if len(sorted_candidates) > 1:
            second_score = get_score(sorted_candidates[1])
            gap = top_score - second_score
        else:
            gap = top_score

        coverage = min(1.0, len(sorted_candidates) / 5.0)
        normalized_score = max(0.0, min(1.0, top_score))

        confidence = (
            0.5 * normalized_score +
            0.3 * min(1.0, gap * 2) +
            0.2 * coverage
        )

        return max(0.0, min(1.0, confidence))
    
    
    def _apply_metadata_filters(self,candidates,filters):
        
        if not filters:
            return candidates
        filtered = []
        
        for c in candidates:
            if 'case_id' in filters:
                if c['case_id'] != filters['case_id']:
                    continue
                
            if 'speaker' in filters:
                speakers = [seg.get("speaker") for seg in c.get("transcript_segments", [])]
                if filters['speaker'] not in speakers:
                    continue
                
            if 'source_type' in filters:
                if c['source_type'] != filters['source_type']:
                    continue
            
            if "time_range" in filters:
                start = filters["time_range"].get("start")
                end = filters["time_range"].get("end")  

                chunk_start = c.get("start_time")
                chunk_end = c.get("end_time")


                if not (
                    chunk_start <= end and
                    chunk_end >= start
                ):
                    continue
            filtered.append(c)
            
        return filtered
                
    
    def _build_retrieved_items(self,selected,confidence=0.0,retrieval_stage='success'):
        
        retrieved_items = []
        
        for c in selected:
            primary_range = TimeRange(
            start_time=c["start_time"],
            end_time=c["end_time"]
            )
            spans = [
                TimeRange(
                    start_time=seg["start"],
                    end_time=seg["end"]
                )
                for seg in c.get("transcript_segments", [])
            ]
            temporal = TemporalData(
                primary=primary_range,
                spans=spans,
                duration=primary_range.end_time - primary_range.start_time,
                segment_count=len(spans)
            )
            retrieved_score = c.get('retrieval_score',c.get('score',0.0))
            reranker_score = c.get('reranker_score',0.0)
            final_score = c.get('final_score',c.get('score',0.0))
            item = RetrievedItem(
                chunk_ids=[c['chunk_id']],
                case_id=c['case_id'],
                temporal=temporal,
                structured_transcripts=c.get('transcript_segments',[]),
                structured_frames=c.get('frames',[]),
                sources=[
                    Source(
                        source_id=c.get("source_id"),
                        source_type=c.get("source_type"),
                        case_id=c.get("case_id")
                    )
                ],  
                retrieval_score=retrieved_score,
                rerank_score=reranker_score,
                final_score=final_score,
                confidence=confidence,
                retrieval_stage=retrieval_stage
                
                
            )
            retrieved_items.append(item)
        return retrieved_items
        

