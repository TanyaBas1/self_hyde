import numpy as np
from hyde.segment_scorer import ReflectionTokens


class HyDE:
    def __init__(self, promptor, generator, encoder, searcher):
        """
        encoder: a class with encode() method that returns embeddings
        searcher: a class with search() method that takes vector and returns hits
        promptor: a class with build_prompt() method that takes query and returns prompt
        generator: a class with generate() method that takes prompt and returns an answer 
        """
        self.promptor = promptor
        self.generator = generator
        self.encoder = encoder
        self.searcher = searcher
    
    def prompt(self, query):
        """Generate a prompt for the query."""
        return self.promptor.build_prompt(query)

    def generate(self, query):
        """Generate hypothesis documents."""
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        return hypothesis_documents
    
    def encode(self, query, hypothesis_documents):
        """Encode query and hypothesis documents into a vector."""
        all_emb_c = []
        for c in [query] + hypothesis_documents:
            c_emb = self.encoder.encode(c)
            all_emb_c.append(np.array(c_emb))
        all_emb_c = np.array(all_emb_c)
        avg_emb_c = np.mean(all_emb_c, axis=0)
        hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
        return hyde_vector
    
    def search(self, hyde_vector, k=10):
        """Search for the most relevant documents."""
        hits = self.searcher.search(hyde_vector, k=k)
        return hits
    

    def e2e_search(self, query, k=10):
        """End-to-end search process."""
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        hyde_vector = self.encode(query, hypothesis_documents)
        hits = self.searcher.search(hyde_vector, k=k)
        return hits


class SelfRAGHyDE(HyDE):
    """Self-HyDE: Self RAG framework on top of HyDE."""
    def __init__(self, promptor, generator, encoder, searcher):
        super().__init__(promptor, generator, encoder, searcher)

    def prompt(self, query):
        """Generate the initial prompt"""
        return self.promptor.build_prompt(query)

    def generate(self, query):
        """Generate hypothesis documents with retrieval decision"""
        retrieval_decision = self.generator.generate_retrieval_decision(query)
        
        if retrieval_decision == ReflectionTokens.NO_RETRIEVE:
            return []
            
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        return hypothesis_documents
    
    def encode(self, query, hypothesis_documents):
        """Encode query and hypothesis documents into a vector"""
        if not hypothesis_documents:  # If no retrieval needed
            return None
            
        all_emb_c = []
        for c in [query] + hypothesis_documents:
            c_emb = self.encoder.encode(c)
            all_emb_c.append(np.array(c_emb))
        all_emb_c = np.array(all_emb_c)
        avg_emb_c = np.mean(all_emb_c, axis=0)
        hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
        return hyde_vector
    
    def search(self, hyde_vector, k=10):
        """Perform search if hyde_vector is available"""
        if hyde_vector is None:
            return []
        hits = self.searcher.search(hyde_vector, k=k)
        return hits
    
    def generate_response(self, query, retrieved_docs):
        """Generate final response with reflection"""
        response, critique = self.generator.generate_with_reflection(query, retrieved_docs)
        return response, critique

    def e2e_search(self, query, k=10):
        """End-to-end search process with self-reflection"""
        hypothesis_documents = self.generate(query)
        
        # if no retrieval needed
        if not hypothesis_documents:
            final_response, critique = self.generator.generate_with_reflection(query)
            return final_response, critique, []
        
        # encode and search
        hyde_vector = self.encode(query, hypothesis_documents)
        hits = self.search(hyde_vector, k=k)
        
        # get retrieved documents
        retrieved_docs = [self.searcher.documents[hit.docid] for hit in hits]
        
        # generate final response 
        final_response, critique = self.generate_response(query, retrieved_docs)
        
        return final_response, critique, hits

    def get_intermediate_results(self, query, k=10):
        """Get results from each step for debugging and analysis"""
        results = {}
        
        # prompt
        results['prompt'] = self.prompt(query)
        
        # generate hypothesis
        results['hypothesis_documents'] = self.generate(query)
        
        # encode
        hyde_vector = self.encode(query, results['hypothesis_documents'])
        results['hyde_vector'] = hyde_vector
        
        # search
        hits = self.search(hyde_vector, k=k) if hyde_vector is not None else []
        results['hits'] = hits
        
        # generate final response
        retrieved_docs = [self.searcher.documents[hit.docid] for hit in hits] if hits else []
        response, critique = self.generate_response(query, retrieved_docs)
        results['final_response'] = response
        results['critique'] = critique
        
        return results