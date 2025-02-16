import pytest
from hyde.generator import Generator, OpenAIGenerator
from hyde.config import get_openai_api_key

# Base class tests
def test_generator_base():
    generator = Generator("test-model", "test-key")
    assert generator.model_name == "test-model"
    assert generator.api_key == "test-key"
    assert generator.generate() == ""

# OpenAI Generator tests
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
    assert openai_generator.temperature == 0.7  # default value
    assert openai_generator.stop == ['\n\n\n']  # default value

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
    # Test with invalid API key
    generator = OpenAIGenerator(
        model_name="gpt-3.5-turbo",
        api_key="invalid-key",
        wait_till_success=False
    )
    
    with pytest.raises(Exception):
        generator.generate("This should fail") 