import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Encoder:
    """Encode text into a vector."""
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, text):
        return self.model.encode(text)

class Searcher:
    """Search for the most relevant documents."""
    def __init__(self, documents):
        """
        documents: dict of {doc_id: text}
        """
        if not documents:
            raise ValueError("Documents dictionary cannot be empty")
            
        self.documents = documents
        self.encoder = Encoder()
        self.doc_ids = list(documents.keys())
        self.doc_embeddings = self._encode_documents()
        self.index = self._build_index()
    
    def _encode_documents(self):
        """Encode documents into a vector."""
        texts = [self.documents[doc_id] for doc_id in self.doc_ids]
        embeddings = self.encoder.encode(texts)
        return embeddings.astype('float32')  
    
    def _build_index(self):
        """Build an index for the documents."""
        dimension = self.doc_embeddings.shape[1]
        # inner product similarity
        index = faiss.IndexFlatIP(dimension)  
        index.add(self.doc_embeddings)
        return index
    
    def search(self, query_vector, k=10):
        """Search for the most relevant documents."""
        query_vector = query_vector.astype('float32').reshape(1, -1)
        similarities, indices = self.index.search(query_vector, k)
        
        class Hit:
            def __init__(self, docid, score):
                self.docid = docid
                self.score = score
        
        hits = [Hit(self.doc_ids[idx], float(score)) 
                for idx, score in zip(indices[0], similarities[0])]
        return hits 