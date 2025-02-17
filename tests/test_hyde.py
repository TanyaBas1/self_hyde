import pytest
import numpy as np
from hyde.hyde import HyDE

class MockPromptor:
    def build_prompt(self, query):
        return f"Generate relevant documents for: {query}"

class MockGenerator:
    def generate(self, prompt):
        # return predictable "hypothetical documents" for testing
        return [
            "The Empire State Building, completed in 1931, stands 1,454 feet tall in Manhattan.",
            "New York City consists of five boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island."
        ]

class MockEncoder:
    def encode(self, text):
        # return a consistent mock embedding
        # using text length as a simple way to generate different but deterministic vectors
        if isinstance(text, str):
            base = len(text) % 5
        else:
            base = len(text[0]) % 5
        return np.array([base, base + 1, base + 2])

class MockSearcher:
    def __init__(self):
        self.last_vector = None
        
    def search(self, vector, k=10):
        self.last_vector = vector
        
        class MockHit:
            def __init__(self, docid, score):
                self.docid = docid
                self.score = score
        
        return [
            MockHit("doc1", 0.9),
            MockHit("doc2", 0.8),
            MockHit("doc3", 0.7)
        ][:k]

@pytest.fixture
def hyde_instance():
    return HyDE(
        promptor=MockPromptor(),
        generator=MockGenerator(),
        encoder=MockEncoder(),
        searcher=MockSearcher()
    )

def test_hyde_initialization(hyde_instance):
    assert isinstance(hyde_instance.promptor, MockPromptor)
    assert isinstance(hyde_instance.generator, MockGenerator)
    assert isinstance(hyde_instance.encoder, MockEncoder)
    assert isinstance(hyde_instance.searcher, MockSearcher)

def test_prompt_generation(hyde_instance):
    query = "Tell me about New York City's landmarks"
    prompt = hyde_instance.prompt(query)
    assert isinstance(prompt, str)
    assert "New York City's landmarks" in prompt

def test_document_generation(hyde_instance):
    query = "What are the main features of NYC?"
    docs = hyde_instance.generate(query)
    
    assert isinstance(docs, list)
    assert len(docs) == 2
    assert any("empire state" in doc.lower() for doc in docs)
    assert any("boroughs" in doc.lower() for doc in docs)

def test_encoding(hyde_instance):
    query = "Test query"
    hypothesis_docs = [
        "First test document",
        "Second test document"
    ]
    
    hyde_vector = hyde_instance.encode(query, hypothesis_docs)
    
    assert isinstance(hyde_vector, np.ndarray)
    assert hyde_vector.ndim == 2  # 2D array
    assert hyde_vector.shape[1] == 3  # based on mock encoder output

def test_search(hyde_instance):
    mock_vector = np.array([[1.0, 2.0, 3.0]])
    hits = hyde_instance.search(mock_vector, k=2)
    
    assert len(hits) == 2
    assert hasattr(hits[0], 'docid')
    assert hasattr(hits[0], 'score')
    assert hits[0].score > hits[1].score  # check order

def test_e2e_search(hyde_instance):
    query = "What are the major landmarks in New York City?"
    hits = hyde_instance.e2e_search(query, k=3)

    assert len(hits) == 3
    assert all(hasattr(hit, 'docid') for hit in hits)
    assert all(hasattr(hit, 'score') for hit in hits)
    assert all(hit.score <= 1.0 for hit in hits) 
    assert hits[0].score >= hits[-1].score  

def test_edge_cases(hyde_instance):
    # empty query
    hits = hyde_instance.e2e_search("", k=1)
    assert len(hits) == 1
    
    # very short query
    hits = hyde_instance.e2e_search("a", k=1)
    assert len(hits) == 1
    
    # k=1
    hits = hyde_instance.e2e_search("New York City", k=1)
    assert len(hits) == 1
    
    # large k
    hits = hyde_instance.e2e_search("New York City", k=100)
    assert len(hits) == 3  # expected to return all available mock hits

def test_vector_consistency(hyde_instance):
    # check if the same query produces consistent vectors
    query = "New York City landmarks"
    
    vector1 = hyde_instance.encode(
        query, 
        hyde_instance.generate(hyde_instance.prompt(query))
    )
    
    vector2 = hyde_instance.encode(
        query, 
        hyde_instance.generate(hyde_instance.prompt(query))
    )
    
    assert np.array_equal(vector1, vector2) 