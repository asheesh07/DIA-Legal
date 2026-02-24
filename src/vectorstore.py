import lancedb
import pyarrow as pa
import numpy as np
import json
class LanceDBVectorStore:
    def __init__(self,table_name,db_path,text_dim,visual_dim):
        self.db = lancedb.connect(db_path)
        self.table_name = table_name
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        
        if self.table_name not in self.db.table_names():
            self._create_table()
        else:
            self.table = self.db.open_table(table_name)
    def _create_table(self):
        schema = pa.schema([
            ("chunk_id",pa.string()),
            ("case_id",pa.string()),
            ("start_time",pa.float32()),
            ("end_time",pa.float32()),
            ("text_embedding", pa.list_(pa.float32(), self.text_dim)),
            ("visual_embedding", pa.list_(pa.float32(), self.visual_dim)),
            ("transcript_text", pa.string()),
            ("structured_data", pa.string())
            
        ])
        self.table =self.db.create_table(self.table_name,schema=schema)
        
    def upsert(self,records):
        formatted = []

        for r in records:

            formatted.append({
                "chunk_id": r["chunk_id"],
                "case_id": r["case_id"],
                "start_time": float(r["start_time"]),
                "end_time": float(r["end_time"]),
                "text_embedding": np.array(r["text_embedding"], dtype=np.float32).tolist(),
                "visual_embedding": np.array(
                    r["visual_embedding"] if r["visual_embedding"] is not None else [0.0] * self.visual_dim,
                    dtype=np.float32
                ).tolist(),
                "transcript_text": r.get("transcript_text", ""),
                "structured_data": json.dumps({
                    "transcript_segments": r.get("transcript_segments", []),
                    "frames": r.get("frames", []),
                    "speakers": r.get("speakers", []),
                    "has_ocr": r.get("has_ocr", False),
                    "has_frames": r.get("has_frames", False),
                    "source_type": r.get("source_type", "video")
                })
                    })

        self.table.add(formatted)
        
    def search(self, text_query_embedding, visual_query_embedding, alpha, top_k, filters=None):

        if text_query_embedding is not None:
            candidates = (
                self.table.search(
                    text_query_embedding,
                    vector_column_name="text_embedding",
                )
                .metric("cosine")
                .limit(top_k * 5)
                .to_list()
            )
        else:
            candidates = self.table.to_pandas().to_dict(orient="records")

        for c in candidates:
            structured_raw = c.get("structured_data")
            if structured_raw:
                try:
                    data = json.loads(structured_raw)
                    c["transcript_segments"] = data.get("transcript_segments", [])
                    c["frames"] = data.get("frames", [])
                    c["speakers"] = data.get("speakers", [])
                    c["has_ocr"] = data.get("has_ocr", False)
                    c["has_frames"] = data.get("has_frames", False)
                    c["source_type"] = data.get("source_type", "video")
                except Exception:
                    raise ValueError('unable to grab segments')
        if filters:
            candidates = [c for c in candidates if self._match_filters(c, filters)]

        scored = []

        for c in candidates:
            distance = c.get("_distance")
            if distance is not None:
                text_sim = 1.0 - float(distance)
            else:
                text_sim = 0.0

            visual_sim = 0.0
            has_visual = False
            if visual_query_embedding is not None and c.get("visual_embedding") is not None:
                visual_vec = np.array(c["visual_embedding"])
                visual_norm = np.linalg.norm(visual_vec)
                query_norm = np.linalg.norm(visual_query_embedding)
                if visual_norm > 0 and query_norm > 0:
                    visual_vec_norm = visual_vec / visual_norm
                    query_normalized = visual_query_embedding / query_norm
                    visual_sim = float(np.dot(query_normalized, visual_vec_norm))
                    visual_sim = max(visual_sim, 0.0)
                    has_visual = True
            if has_visual:
                final_score = alpha * text_sim + (1 - alpha) * visual_sim
            else:
                final_score = text_sim

            c["score"] = float(final_score)

            scored.append(c)

        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return scored[:top_k]
    
    def _match_filters(self,record,filters):
        for key,value in filters.items():
            if key == "case_id":
                if record['case_id'] != value:
                    return False
            elif key == "time_range":
                start = value.get('start')
                end = value.get('end')
                
                if start is not None and record['end_time']<start:
                    return False
                if end is not None and record['start_time']>end:
                    return False
            else:
                if record.get(key)!=value:
                    return False
        return True
    
    def count_table_rows(self):
        return self.table.count_rows()
    
    def clear(self):
        self.db.drop_table(self.table_name)
        self._create_table()
    
    def get(self,chunk_ids):
        self.table.search().where(
            f"chunk_id in {chunk_ids}"
        ).to_list()
            
        
    