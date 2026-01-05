from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List


class LocalEmbeddings(Embeddings):
    """Local embedding model adapter, compatible with LangChain Embeddings interface"""
    
    def __init__(self, model_name: str = "/share/home/ecnuzwx/UnifiedRAG/cache/models--BAAI--bge-m3", device: str = "cuda", batch_size: int = 8, normalize: bool = True):
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        self.batch_size = batch_size
        self.normalize = normalize
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed document list"""
        embeddings = self.model.encode(texts, batch_size=self.batch_size, convert_to_tensor=False, normalize_embeddings=self.normalize, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        embedding = self.model.encode([text], batch_size=1, convert_to_tensor=False, normalize_embeddings=self.normalize, show_progress_bar=False)
        return embedding[0].tolist()

class SemanticChunking:
    """Semantic Chunking class, wrapping SemanticChunker"""
    
    def __init__(self, embed_model_name: str = "/share/home/ecnuzwx/UnifiedRAG/cache/models--BAAI--bge-m3", 
                 device: str = "cuda",
                 batch_size: int = 16,
                 normalize: bool = True):
        self.embed_model_name = embed_model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        
        self.embeddings = LocalEmbeddings(model_name=embed_model_name, device=device, batch_size=batch_size, normalize=normalize)
        
        self.text_splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=50.0)
    
    def chunk(self, text: str) -> List[str]:
        """Chunk text"""
        try:
            docs = self.text_splitter.create_documents([text])
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"⚠️ SemanticChunking.chunk failed: {e}")
            return []
