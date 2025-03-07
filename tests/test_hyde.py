import pytest
import numpy as np
from hyde.hyde import HyDE, SelfRAGHyDE
from hyde.segment_scorer import ReflectionTokens


class MockPromptor:
    def build_prompt(self, query):
        return f"Generate relevant documents for: {query}"

class MockGenerator:
    def generate(self, prompt):
        # return predictable hypothetical documents for testing
        return [
            "The Empire State Building, completed in 1931, stands 1,454 feet tall in Manhattan.",
            "New York City consists of five boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island."
        ]
    
    def generate_retrieval_decision(self, prompt):
        # Return RETRIEVE for factual questions about NYC, NO_RETRIEVE for opinion questions
        if any(word in prompt.lower() for word in ['nyc', 'new york', 'manhattan', 'empire']):
            return ReflectionTokens.RETRIEVE
        return ReflectionTokens.NO_RETRIEVE
    
    def generate_with_reflection(self, prompt, retrieved_docs=None):
        # Mock reflection generation
        if retrieved_docs:
            response = "Here's information about NYC based on retrieved documents."
            critique = {
                'relevance': ReflectionTokens.RELEVANT,
                'support': ReflectionTokens.SUPPORTED
            }
        else:
            response = "This is a subjective response without retrieval."
            critique = {
                'relevance': ReflectionTokens.NO_RETRIEVE,
                'support': ReflectionTokens.NO_RETRIEVE
            }
        return response, critique

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

        self.documents = {
            "doc1": "The 1735 trial and acquittal in Manhattan of John Peter Zenger, who had been accused of seditious libel after criticizing colonial governor William Cosby, helped to establish freedom of the press in North America.",
            "doc2":  "Designed by Frederick Law Olmsted and Calvert Vaux, Central Park opened in 1876 and is the most visited urban park in the United States.",
            "doc3": "New York City's population exceeded 8 million for the first time in the 2000 census; further records were set in the 2010 and 2020 censuses. Important new economic sectors, such as Silicon Alley, emerged."
        }
        
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

@pytest.fixture
def selfrag_hyde_instance():
    return SelfRAGHyDE(
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

def test_selfrag_hyde_no_retrieval_path(selfrag_hyde_instance):
    """Test the no-retrieval decision path"""
    query = "What's your opinion on AI?"
    response, critique, hits = selfrag_hyde_instance.e2e_search(query)
    
    assert isinstance(response, str)
    assert isinstance(critique, dict)
    assert hits == []  # No hits when retrieval is skipped

def test_selfrag_hyde_retrieval_path(selfrag_hyde_instance):
    """Test the retrieval decision path"""
    query = "Tell me about the Empire State Building"
    response, critique, hits = selfrag_hyde_instance.e2e_search(query)
    
    assert isinstance(response, str)
    assert isinstance(critique, dict)
    assert len(hits) > 0  # Should have hits when retrieval is used
    assert all(hasattr(hit, 'docid') for hit in hits)
    assert all(hasattr(hit, 'score') for hit in hits)

def test_selfrag_hyde_intermediate_results(selfrag_hyde_instance):
    """Test the intermediate results functionality"""
    query = "What are NYC's landmarks?"
    results = selfrag_hyde_instance.get_intermediate_results(query)
    
    # Check all intermediate steps are present
    assert 'prompt' in results
    assert 'hypothesis_documents' in results
    assert 'hyde_vector' in results
    assert 'hits' in results
    assert 'final_response' in results
    assert 'critique' in results
    
    # Verify types
    assert isinstance(results['prompt'], str)
    assert isinstance(results['hypothesis_documents'], list)
    if results['hyde_vector'] is not None:
        assert isinstance(results['hyde_vector'], np.ndarray)
    assert isinstance(results['hits'], list)
    assert isinstance(results['final_response'], str)
    assert isinstance(results['critique'], dict) 