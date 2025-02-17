import pytest
from hyde.segment_scorer import SegmentScorer, ReflectionTokens

@pytest.fixture
def scorer():
    return SegmentScorer()

def test_segment_scorer_init(scorer):
    assert scorer.weights['relevance'] == 0.3
    assert scorer.weights['support'] == 0.7
    assert len(scorer.weights) == 2

def test_score_segment_relevant_supported(scorer):
    critique_tokens = {
        'relevance': ReflectionTokens.RELEVANT,
        'support': ReflectionTokens.SUPPORTED
    }
    score = scorer.score_segment(segment="test", critique_tokens=critique_tokens)
    # 0.3 * 1.0 + 0.7 * 1.0 = 1.0
    assert score == 1.0

def test_score_segment_relevant_partially(scorer):
    critique_tokens = {
        'relevance': ReflectionTokens.RELEVANT,
        'support': ReflectionTokens.PARTIALLY
    }
    score = scorer.score_segment(segment="test", critique_tokens=critique_tokens)
    # 0.3 * 1.0 + 0.7 * 0.7 = 0.79
    assert score == pytest.approx(0.79, rel=1e-2)

def test_score_segment_relevant_contradictory(scorer):
    critique_tokens = {
        'relevance': ReflectionTokens.RELEVANT,
        'support': ReflectionTokens.CONTRADICTORY
    }
    score = scorer.score_segment(segment="test", critique_tokens=critique_tokens)
    # 0.3 * 1.0 + 0.7 * 0.3 = 0.51
    assert score == pytest.approx(0.51, rel=1e-2)

def test_score_segment_irrelevant(scorer):
    critique_tokens = {
        'relevance': ReflectionTokens.IRRELEVANT,
        'support': ReflectionTokens.SUPPORTED
    }
    score = scorer.score_segment(segment="test", critique_tokens=critique_tokens)
    # 0.3 * 0.0 + 0.7 * 1.0 = 0.7
    assert score == 0.7

def test_score_segment_no_retrieve(scorer):
    critique_tokens = {
        'relevance': ReflectionTokens.NO_RETRIEVE,
        'support': ReflectionTokens.NO_RETRIEVE
    }
    score = scorer.score_segment(segment="test", critique_tokens=critique_tokens)
    # Should return 0 when no retrieval
    assert score == 0.0

def test_score_segment_missing_tokens(scorer):
    # Test with missing tokens
    critique_tokens = {}
    score = scorer.score_segment(segment="test", critique_tokens=critique_tokens)
    assert score == 0.0

    # Test with partial tokens
    critique_tokens = {'relevance': ReflectionTokens.RELEVANT}
    score = scorer.score_segment(segment="test", critique_tokens=critique_tokens)
    assert score == 0.3  # Only relevance score 