import pytest
from hyde.generator import Generator, OpenAIGenerator, SelfRAGGenerator
from hyde.config import get_openai_api_key
from hyde.segment_scorer import SegmentScorer, ReflectionTokens

# base class tests
def test_generator_base():
    generator = Generator("test-model", "test-key")
    assert generator.model_name == "test-model"
    assert generator.api_key == "test-key"
    assert generator.generate() == ""

@pytest.fixture
def openai_generator():
    try:
        api_key = get_openai_api_key()
    except ValueError:
        api_key = "dummy-key"
    return OpenAIGenerator(
        model_name="gpt-3.5-turbo",
        api_key=api_key,
        n=1,
        max_tokens=100
    )

def test_openai_generator_init(openai_generator):
    assert openai_generator.model_name == "gpt-3.5-turbo"
    assert openai_generator.n == 1
    assert openai_generator.max_tokens == 100
    assert openai_generator.temperature == 0.7  
    assert openai_generator.stop == ['\n\n\n']  

@pytest.mark.skipif(not get_openai_api_key(), reason="OpenAI API key not found")
def test_openai_generator_generate(openai_generator):
    prompt = "Write a one-sentence description of what Python is:"
    results = openai_generator.generate(prompt)
    
    assert isinstance(results, list)
    assert len(results) == 1  # since n=1
    assert isinstance(results[0], str)
    assert len(results[0]) > 0

def test_parse_response(openai_generator):
    # Create a mock response object
    class MockChoice:
        def __init__(self, content):
            self.message = type('Message', (), {'content': content})
    
    class MockResponse:
        def __init__(self, choices):
            self.choices = choices
    
    mock_response = MockResponse([
        MockChoice("First response"),
        MockChoice("Second response")
    ])
    
    results = openai_generator.parse_response(mock_response)
    assert results == ["First response", "Second response"]

def test_openai_generator_error_handling():
    # invalid API key
    generator = OpenAIGenerator(
        model_name="gpt-3.5-turbo",
        api_key="invalid-key",
        wait_till_success=False
    )
    
    with pytest.raises(Exception):
        generator.generate("This should fail")

@pytest.fixture
def selfrag_generator():
    try:
        api_key = get_openai_api_key()
    except ValueError:
        api_key = "dummy-key"
    return SelfRAGGenerator(
        model_name="gpt-3.5-turbo",
        api_key=api_key,
        n_segments=3
    )

def test_selfrag_generator_init(selfrag_generator):
    assert selfrag_generator.model_name == "gpt-3.5-turbo"
    assert selfrag_generator.n_segments == 3
    assert isinstance(selfrag_generator.scorer, SegmentScorer)

def test_selfrag_generator_retrieval_decision(selfrag_generator):
    # factual question
    factual_prompt = "Where NYC is located?"
    decision = selfrag_generator.generate_retrieval_decision(factual_prompt)
    assert decision in [ReflectionTokens.RETRIEVE, ReflectionTokens.NO_RETRIEVE]
    
    # opinion question
    opinion_prompt = "What's your favorite color?"
    decision = selfrag_generator.generate_retrieval_decision(opinion_prompt)
    assert decision in [ReflectionTokens.RETRIEVE, ReflectionTokens.NO_RETRIEVE]

def test_selfrag_generator_critique(selfrag_generator):
    # Test critique without retrieval
    critique = selfrag_generator.generate_critique(
        segment="This is a test segment",
        retrieval_used=False
    )
    assert critique['relevance'] == ReflectionTokens.NO_RETRIEVE
    assert critique['support'] == ReflectionTokens.NO_RETRIEVE
    
    # Test critique with retrieval
    critique = selfrag_generator.generate_critique(
        segment="Times Square is in New York City",
        reference_doc="Times Square is located in Midtown Manhattan. It attracts over 50 million visitors annually.",
        retrieval_used=True
    )
    assert critique['relevance'] == ReflectionTokens.RELEVANT
    assert critique['support'] == ReflectionTokens.SUPPORTED


@pytest.mark.skipif(not get_openai_api_key(), reason="OpenAI API key not found")
def test_selfrag_generator_generate_with_reflection(selfrag_generator):
    prompt = "What is the significance of Central Park in NYC?"

    # from wiki scraped docs
    retrieved_docs = [
        "The 1735 trial and acquittal in Manhattan of John Peter Zenger, who had been accused of seditious libel after criticizing colonial governor William Cosby, helped to establish freedom of the press in North America.",
        "Designed by Frederick Law Olmsted and Calvert Vaux, Central Park opened in 1876 and is the most visited urban park in the United States.",
        "New York City's population exceeded 8 million for the first time in the 2000 census; further records were set in the 2010 and 2020 censuses. Important new economic sectors, such as Silicon Alley, emerged."
    ]
    
    response, critique = selfrag_generator.generate_with_reflection(
        prompt,
        retrieved_docs=retrieved_docs
    )
    
    assert isinstance(response, str)
    assert isinstance(critique, dict)
    assert 'relevance' in critique
    assert 'support' in critique 