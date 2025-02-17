class SegmentScorer:
    def __init__(self):
        self.weights = {
            "relevance": 0.3,
            "support": 0.7,
        }
    
    def score_segment(self, segment, critique_tokens):
        relevance_score = 1.0 if critique_tokens.get('relevance') == ReflectionTokens.RELEVANT else 0.0
        
        support_scores = {
            ReflectionTokens.SUPPORTED: 1.0,
            ReflectionTokens.PARTIALLY: 0.7,
            ReflectionTokens.CONTRADICTORY: 0.3
        }
        support_score = support_scores.get(critique_tokens.get('support'), 0.0)
        
        return (self.weights['relevance'] * relevance_score + 
                self.weights['support'] * support_score)



class ReflectionTokens:
    RETRIEVE = "<retrieve>"
    NO_RETRIEVE = "<no_retrieve>"
    RELEVANT = "<relevant>"
    IRRELEVANT = "<irrelevant>"
    SUPPORTED = "<supported>"
    PARTIALLY = "<partially>"
    CONTRADICTORY = "<contradictory>"