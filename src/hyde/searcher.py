import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import numpy as np
from sentence_transformers import SentenceTransformer

class Encoder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, text):
        return self.model.encode(text)

class Searcher:
    def __init__(self, documents):
        """
        documents: dict of {doc_id: text}
        """
        self.documents = documents
        self.encoder = Encoder()
        self.doc_ids = list(documents.keys())
        self.doc_embeddings = self._encode_documents()
    
    def _encode_documents(self):
        texts = [self.documents[doc_id] for doc_id in self.doc_ids]
        return self.encoder.encode(texts)
    
    def search(self, query_vector, k=10):
        similarities = np.dot(self.doc_embeddings, query_vector.T).squeeze()
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        
        class Hit:
            def __init__(self, docid):
                self.docid = docid
        
        hits = [Hit(self.doc_ids[idx]) for idx in top_k_idx]
        return hits 