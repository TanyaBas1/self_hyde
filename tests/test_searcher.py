import pytest
import numpy as np
from hyde.searcher import Searcher, Encoder

@pytest.fixture
def sample_documents():
    return {
    "doc1": "The 1735 trial and acquittal in Manhattan of John Peter Zenger, who had been accused of seditious libel after criticizing colonial governor William Cosby, helped to establish freedom of the press in North America.",
    "doc2":  "Designed by Frederick Law Olmsted and Calvert Vaux, Central Park opened in 1876 and is the most visited urban park in the United States.", 
    "doc3": "New York City's population exceeded 8 million for the first time in the 2000 census; further records were set in the 2010 and 2020 censuses. Important new economic sectors, such as Silicon Alley, emerged."
    }

@pytest.fixture
def searcher(sample_documents):
    return Searcher(sample_documents)

def test_encoder_init():
    encoder = Encoder()
    assert encoder.model is not None
    assert encoder.model.get_sentence_embedding_dimension() > 0

def test_encoder_encode():
    encoder = Encoder()
    text = "New York City"
    embedding = encoder.encode(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape[0] == encoder.model.get_sentence_embedding_dimension()

def test_searcher_init(searcher, sample_documents):
    assert searcher.documents == sample_documents
    assert len(searcher.doc_ids) == len(sample_documents)
    assert isinstance(searcher.encoder, Encoder)
    assert searcher.index is not None

def test_searcher_encode_documents(searcher, sample_documents):
    embeddings = searcher._encode_documents()
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(sample_documents)
    assert embeddings.dtype == np.float32

def test_searcher_search(searcher):
    # Create a query vector using the same encoder
    query = "New York City landmarks"
    query_vector = searcher.encoder.encode(query)
    
    # Search
    hits = searcher.search(query_vector, k=2)
    
    # Test results
    assert len(hits) == 2
    assert all(hasattr(hit, 'docid') for hit in hits)
    assert all(hasattr(hit, 'score') for hit in hits)
    assert all(isinstance(hit.score, float) for hit in hits)
    assert all(hit.docid in searcher.doc_ids for hit in hits)
    
    # scores should be between -1 and 1 (cosine similarity)
    assert all(-1 <= hit.score <= 1 for hit in hits)
    
    # 1st hit should have higher score than second
    assert hits[0].score >= hits[1].score


def test_searcher_with_empty_documents():
    empty_docs = {}
    with pytest.raises(ValueError):
        Searcher(empty_docs) 