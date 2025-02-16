import pytest
from hyde.config import get_openai_api_key

def test_api_key_loaded():
    """Test that we can load the API key from .env"""
    api_key = get_openai_api_key()
    assert api_key is not None
    assert len(api_key) > 0
    assert api_key.startswith('sk-') 