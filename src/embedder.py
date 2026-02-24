import numpy as np
import torch
from torch import no_grad
from transformers import CLIPProcessor,CLIPModel
from sentence_transformers import SentenceTransformer
from PIL import Image
class TextEmbedder:
    def __init__(self,model,batch_size,normalize):
        self.model = model
        self.embed_dims=self.model.get_sentence_embedding_dimension() 
        self.batch_size = batch_size
        self.normalize =normalize
    def embed(self,text):
        if not text.strip():
            return np.zeros(self.embed_dims)
        try:
            embedding = self.model.encode(text)
            embedding = np.array(embedding)
            if embedding.ndim == 2:
                embedding = embedding[0]
            if self.normalize:
                embedding = self._normalize(embedding)
            return embedding
        
        except Exception as e:
            raise RuntimeError(f"Text embedding failed: {e}")
            
        
    def embed_batch(self,texts):
        if not texts:
            return np.zeros((0,self.embed_dims))
        valid_texts =[]
        valid_indices =[]
        for i,text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        embedding = np.zeros((len(texts),self.embed_dims))
        if not valid_texts:
            return embedding
        try:
            valid_embedding = self.model.encode(valid_texts)
            valid_embedding = np.asarray(valid_embedding)
            if valid_embedding.ndim ==1:
                valid_embedding = valid_embedding.reshape(-1,self.embed_dims)
            if valid_embedding.shape[0] == 1 and len(valid_texts)>1:
                valid_embedding = np.repeat(valid_embedding,len(valid_texts),axis=0)
            if valid_embedding.shape[0]!=len(valid_texts):
                raise ValueError(
                    f"Embedding count mismatch: got {valid_embedding.shape[0]}, "
                f"expected {len(valid_texts)}"
                )
            if self.normalize:
                valid_embedding =self._normalize(valid_embedding)
            for idx,valid_index in enumerate(valid_indices):
                embedding[valid_index]=valid_embedding[idx]
            return embedding
        except Exception as e:
            raise RuntimeError(f"Error embedding batch: {e}")
        
    def query(self,query):
        return self.embed(query)
    
    def _normalize(self,embeddings):
        if embeddings.ndim ==1:
            norms = np.linalg.norm(embeddings)
            return embeddings/norms if norms>0 else embeddings
        else:
            norms = np.linalg.norm(embeddings,axis=1,keepdims=True)
            norms = np.where(norms>0,norms,1)
            return embeddings/norms
    
class VisualEmbedder:
    def __init__(self,model_name,device,normalize):
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize= normalize
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model= CLIPModel.from_pretrained(model_name).to(self.device)
        self.embed_dim = self.model.config.projection_dim
        self.model.eval()
        
    def embed_image(self,image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return np.zeros(self.embed_dim)
        input = self.processor(images= image,return_tensors="pt").to(self.device)
        with no_grad():
            image_features =self.model.get_image_features(**input)
        
        vector = image_features[0].cpu().numpy()
        if self.normalize:
            vector =self._normalize(vector)
        return vector
        
    def embed_batch(self,image_paths,batch_size=32):
        if not image_paths:
            return np.zeros((0,self.embed_dim))
        all_vectors =[]
        for i in range(0,len(image_paths),batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images =[]
            valid_indices =[]
            for idx,path in enumerate(batch_paths):
                try:
                    images.append(Image.open(path).convert("RGB"))
                    valid_indices.append(idx)
                except Exception:
                    images.append(None)
            valid_images =[img for img in images if img is not None]
            if not valid_images:
                continue
            inputs = self.processor(
                images=valid_images,
                return_tensors='pt',
                padding=True
                
            ).to(self.device)
            
            with no_grad():
                image_features = self.model.get_image_features(**inputs)
            vectors = image_features.cpu().numpy()
            if self.normalize:
                vectors = self._normalize(vectors)
            all_vectors.append(vectors)
        if not all_vectors:
            return np.zeros((0,self.embed_dim))
            
        return np.vstack(all_vectors)
    def embed_query_text(self, text) :
        
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        with no_grad():
            text_features = self.model.get_text_features(**inputs)

        vector = text_features[0].cpu().numpy()

        if self.normalize:
            vector = self._normalize(vector)

        return vector
    def aggregate(self, vectors: np.ndarray, method: str = "mean") -> np.ndarray:
        if vectors is None or len(vectors) == 0:
            return np.zeros(self.embed_dim)

        if method == "mean":
            return np.mean(vectors, axis=0)

        elif method == "max":
            return np.max(vectors, axis=0)

        else:
            raise ValueError("Unsupported aggregation method")
        
    def _normalize(self,vectors):
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return vectors / norm if norm>0 else vectors
        else:
            norm = np.linalg.norm(vectors,axis=1,keepdims=True)
            norm = np.where(norm>0,norm,1)
            return vectors / norm
class MultiModalEmbedder:
    def __init__(self,text_embedder,visual_embedder = None,visual_aggregation="mean"):
        self.text_embedder =text_embedder
        self.visual_embedder = visual_embedder
        self.visual_aggregation = visual_aggregation
        pass
    def embed_chunk(self,chunk):
        text_blob = self._build_text_blob(chunk)
        text_vector = self.text_embedder.embed(text_blob)

        if len(text_vector) != self.text_embedder.embed_dims:
            raise ValueError(
                f"Text embedding dimension mismatch: "
                f"got {len(text_vector)}, expected {self.text_embedder.embed_dims}"
            )

        visual_vector = None

        if self.visual_embedder and chunk.get('frames'):
            frame_paths = [
                frame['image_path'] for frame in chunk['frames'] 
                if frame.get('image_path')
            ]

            if frame_paths:
                frame_vectors = self.visual_embedder.embed_batch(frame_paths)
                visual_vector = self.visual_embedder.aggregate(
                    frame_vectors,
                    method=self.visual_aggregation
                )

        if self.visual_embedder:
            if visual_vector is None:
                visual_vector = np.zeros(self.visual_embedder.embed_dim)

            if len(visual_vector) != self.visual_embedder.embed_dim:
                raise ValueError(
                    f"Visual embedding dimension mismatch: "
                    f"got {len(visual_vector)}, expected {self.visual_embedder.embed_dim}"
                )
        else:
            visual_vector = None

        return {
            "text_embedding": text_vector.tolist(),
            "visual_embedding": visual_vector.tolist() if visual_vector is not None else None
        }
    def embed_query(self,query):
        text_query_vector = self.text_embedder.embed(query)
        visual_query_vector = None
        if self.visual_embedder:
            visual_query_vector = self.visual_embedder.embed_query_text(query)
        return{"text_query_embedding": text_query_vector,
            "visual_query_embedding": visual_query_vector}
        
    def _build_text_blob(self,chunks):
        texts =[]
        
        for seg in chunks.get('transcript_segments',[]):
            if seg.get('text'):
                texts.append(seg['text'])
        for frame in chunks.get('frames',[]):
            if frame.get('ocr_text'):
                texts.append(frame['ocr_text'])
            if frame.get('caption'):
                texts.append(frame['caption'])
        return ' '.join(texts).strip()
            
    