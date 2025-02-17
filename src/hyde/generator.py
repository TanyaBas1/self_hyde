import time
import openai
from src.hyde.segment_scorer import SegmentScorer, ReflectionTokens

class Generator:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key

    
    def generate(self):
        return ""


class OpenAIGenerator(Generator):
    def __init__(self, model_name, api_key, base_url=None, n=1, max_tokens=512, 
                 temperature=0.7, top_p=1, frequency_penalty=0.0, 
                 presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
        super().__init__(model_name, api_key)
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.wait_till_success = wait_till_success
        self.base_url = base_url
        self._client_init()
    
    def _client_init(self):
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def parse_response(self, response):
        """Parse the response from the OpenAI API."""
        texts = []
        for choice in response.choices:
            texts.append(choice.message.content)
        return texts
    
    def generate(self, prompt):
        get_results = False
        while not get_results:
            try:
                result = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    top_p=self.top_p,
                    n=self.n,
                    stop=self.stop
                )
                get_results = True
            except Exception as e:
                if self.wait_till_success:
                    time.sleep(1)
                else:
                    raise e
        return self.parse_response(result)


class SelfRAGGenerator:
    def __init__(self, model_name, api_key, n_segments=3, base_url=None):
        self.model_name = model_name
        self.n_segments = n_segments
        self.scorer = SegmentScorer()
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, prompt):
        result = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=512,
                temperature=0.3,
                n=1)
        return [choice.message.content for choice in result.choices]

    def generate_with_reflection(self, prompt, retrieved_docs=None):
        # check if we need rag to asnwer this quesiton
        retrieval_decision = self.generate_retrieval_decision(prompt)
        
        if retrieval_decision == ReflectionTokens.NO_RETRIEVE:
            response = self.generate(prompt)[0]
            critique = self.generate_critique(response, retrieval_used = False)
            return response, critique

        segments = []
        critiques = []
        
        for i in range(self.n_segments):
            segment = self.generate(prompt)[0]
            critique = self.generate_critique(segment, retrieved_docs[i] if retrieved_docs else None)
            segments.append(segment)
            critiques.append(critique)
        
        scores = [self.scorer.score_segment(seg, crit) for seg, crit in zip(segments, critiques)]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        
        return segments[best_idx], critiques[best_idx]

    def generate_retrieval_decision(self, prompt):
        decision_prompt = f"""
        Should I retrieve external information for the following prompt: {prompt}\n.
        Respond {ReflectionTokens.RETRIEVE} if the prompt involves general knowledge or requests factual information.
        Respond {ReflectionTokens.NO_RETRIEVE} if the prompt seeks subjective opinions or creative input
        """
        decision = self.generate(decision_prompt)[0].strip()
        return decision

    def generate_critique(self, segment, reference_doc=None, retrieval_used=True):

        if not retrieval_used:
            return {
                'relevance': ReflectionTokens.NO_RETRIEVE,
                'support': ReflectionTokens.NO_RETRIEVE
            }
        else:
            critique_prompt = f"""
            Evaluate the following text and provide your assessment for relevance and support:\n{segment}
            if you are 70% confident that the document is relevat say 'relevant', othervise say 'irrelevant'
            if you are 70% confident that docuemnt supports the question say 'supported' otherwise say 'contradicts'
            """
            if reference_doc:
                critique_prompt += f"\n here are the reference documents:\n{reference_doc}"
                
            critique_response = self.generate(critique_prompt)[0].lower()
            
            return {
                'relevance': ReflectionTokens.RELEVANT if "relevant" in critique_response else ReflectionTokens.IRRELEVANT,
                'support': ReflectionTokens.SUPPORTED if "supported" in critique_response else ReflectionTokens.PARTIALLY
            }