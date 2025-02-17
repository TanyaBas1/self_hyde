import json
import logging
from datetime import datetime
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rouge_score import rouge_scorer
from typing import Dict, List
from abc import ABC, abstractmethod

from src.hyde import Promptor, OpenAIGenerator, HyDE, SelfRAGHyDE
from src.hyde.searcher import Encoder, Searcher
from src.hyde.segment_scorer import SegmentScorer, ReflectionTokens
from src.hyde.generator import SelfRAGGenerator
from hyde.config import get_openai_api_key

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        pass

class VanillaRetriever(BaseRetriever):
    def __init__(self, encoder, searcher, documents):
        self.encoder = encoder
        self.searcher = searcher
        self.documents = documents
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        # Direct semantic search without hypothesis generation
        query_vector = self.encoder.encode([query])[0]
        hits = self.searcher.search(query_vector, k=k)
        return [self.documents[hit.docid] for hit in hits]

class HyDERetriever(BaseRetriever):
    def __init__(self, hyde_system):
        self.hyde = hyde_system
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        # HyDE retrieval process
        hypothesis_docs = self.hyde.generate(query)
        hyde_vector = self.hyde.encode(query, hypothesis_docs)
        hits = self.hyde.search(hyde_vector, k=k)
        return [self.hyde.documents[hit.docid] for hit in hits]

class SelfRAGHyDERetriever(BaseRetriever):
    def __init__(self, selfrag_hyde_system):
        self.hyde = selfrag_hyde_system
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        # Get intermediate results which includes retrieval decision
        results = self.hyde.get_intermediate_results(query)
        
        # If no retrieval needed, return empty list
        if not results['hypothesis_documents']:
            return []
            
        # Return retrieved documents if available
        return [self.hyde.documents[hit.docid] for hit in results['hits'][:k]]

class RAGEvaluator:
    def __init__(self, model_name: str = "cross-encoder/qnli-electra-base"):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Add logging
        self.logger = logging.getLogger(__name__)
        
    def evaluate_retrieval(self, retrieved_docs: List[str], context: str) -> Dict[str, float]:
        """Evaluate retrieval performance by comparing retrieved docs with ground truth context"""
        combined_retrieved = " ".join(retrieved_docs)
        scores = self.rouge_scorer.score(context, combined_retrieved)
        
        return {
            'retrieval_rouge1': scores['rouge1'].fmeasure,
            'retrieval_rouge2': scores['rouge2'].fmeasure,
            'retrieval_rougeL': scores['rougeL'].fmeasure
        }

    def evaluate_answer(self, predicted_answer: str, ground_truth: List[str]) -> Dict[str, float]:
        """Evaluate answer quality using ROUGE and semantic similarity"""
        # Calculate ROUGE scores
        rouge_scores = {f"answer_rouge{k}": v.fmeasure 
                      for k, v in self.rouge_scorer.score(ground_truth[0], predicted_answer).items()}
        
        # Calculate relevance score
        inputs = self.tokenizer(
            ground_truth[0],
            predicted_answer,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            relevance_score = torch.sigmoid(outputs.logits[0]).item()
        
        rouge_scores['answer_relevance'] = relevance_score
        return rouge_scores

def print_comparative_results(results: Dict[str, Dict[str, float]]):
    """Pretty print comparative evaluation results"""
    print("\nComparative RAG Evaluation Results:")
    print("-" * 80)
    
    metrics_groups = {
        'Retrieval Metrics': 'retrieval',
        'Answer Metrics': 'answer',
        'Performance Metrics': 'time'
    }
    
    # Calculate improvement percentages relative to vanilla
    improvements = {
        approach: {
            k: ((results[approach][k] - results['vanilla'][k]) / results['vanilla'][k] * 100)
            for k in results[approach].keys()
        }
        for approach in ['hyde', 'selfrag_hyde']
    }
    
    for group_name, prefix in metrics_groups.items():
        print(f"\n{group_name}:")
        print("-" * 60)
        print(f"{'Metric':25s} {'Vanilla':10s} {'HyDE':10s} {'Self-RAG':10s} {'HyDE %':10s} {'Self-RAG %':10s}")
        print("-" * 80)
        
        for k in results['vanilla'].keys():
            if k.startswith(prefix):
                print(f"{k:25s} {results['vanilla'][k]:10.4f} {results['hyde'][k]:10.4f} "
                      f"{results['selfrag_hyde'][k]:10.4f} {improvements['hyde'][k]:10.2f}% "
                      f"{improvements['selfrag_hyde'][k]:10.2f}%")

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # NYC subset was chosen because of the highest count of the NYC related questions
    # the size of NYC subset of the squad dataset is 817, for speed and cost reasons testing on 100
    NUM_EXAMPLES = 100
    logger.info(f"Starting NYC-specific RAG evaluation with {NUM_EXAMPLES} examples...")
    
    # Load dataset and filter for NYC
    squad_dataset = load_dataset("squad")
    nyc_dataset = squad_dataset['train'].filter(lambda example: example['title'] == 'New_York_City')
    nyc_dataset = nyc_dataset.select(range(min(NUM_EXAMPLES, len(nyc_dataset))))
    
    logger.info(f"Evaluating {len(nyc_dataset)} NYC-related questions")
    
    # Initialize components
    promptor = Promptor('web search')
    generator = OpenAIGenerator('gpt-3.5-turbo', get_openai_api_key())
    selfrag_generator = SelfRAGGenerator('gpt-3.5-turbo', get_openai_api_key())
    encoder = Encoder()
    
    # Create corpus from the NYC dataset
    corpus = {item['id']: item['context'] for item in nyc_dataset}
    searcher = Searcher(corpus)
    
    # Initialize retrievers with corpus
    vanilla_retriever = VanillaRetriever(encoder, searcher, corpus)
    hyde = HyDE(promptor, generator, encoder, searcher)
    hyde.documents = corpus  # Add documents to HyDE
    hyde_retriever = HyDERetriever(hyde)
    selfrag_hyde = SelfRAGHyDE(promptor, selfrag_generator, encoder, searcher)
    selfrag_hyde.documents = corpus  # Add documents to SelfRAGHyDE
    selfrag_hyde_retriever = SelfRAGHyDERetriever(selfrag_hyde)
    
    evaluator = RAGEvaluator()
    
    # Run evaluation comparing all approaches
    metrics = {
        'vanilla': defaultdict(list),
        'hyde': defaultdict(list),
        'selfrag_hyde': defaultdict(list)
    }
    
    for item in tqdm(nyc_dataset, desc="Evaluating NYC questions"):
        question = item['question']
        context = item['context']
        answers = item['answers']['text']
        
        # Evaluate all approaches
        for approach, retriever in [
            ('vanilla', vanilla_retriever),
            ('hyde', hyde_retriever),
            ('selfrag_hyde', selfrag_hyde_retriever)
        ]:
            try:
                # Measure retrieval time
                start_time = time.time()
                retrieved_docs = retriever.retrieve(question)
                retrieval_time = time.time() - start_time
                
                # Skip evaluation if no retrieval needed (SelfRAG decision)
                if not retrieved_docs and approach == 'selfrag_hyde':
                    # Get direct response from SelfRAG
                    intermediate_results = selfrag_hyde.get_intermediate_results(question)
                    predicted_answer = intermediate_results['final_response']
                else:
                    # Generate answer using retrieved documents
                    prompt = f"""Based on the retrieved documents, answer the question: {question}
                    Retrieved information: {' '.join(retrieved_docs)}
                    Please provide a clear and concise answer:"""
                    predicted_answer = generator.generate(prompt)[0]
                
                # Evaluate retrieval and answer
                if retrieved_docs:
                    retrieval_metrics = evaluator.evaluate_retrieval(retrieved_docs, context)
                else:
                    # Zero scores for retrieval metrics when no retrieval
                    retrieval_metrics = {
                        'retrieval_rouge1': 0.0,
                        'retrieval_rouge2': 0.0,
                        'retrieval_rougeL': 0.0
                    }
                
                answer_metrics = evaluator.evaluate_answer(predicted_answer, answers)
                
                # Store metrics
                metrics[approach]['retrieval_time'].append(retrieval_time)
                for k, v in {**retrieval_metrics, **answer_metrics}.items():
                    metrics[approach][k].append(v)
                    
            except Exception as e:
                logger.error(f"Error evaluating {approach} approach: {str(e)}")
                # Add zero values for failed evaluations
                metrics[approach]['retrieval_time'].append(0.0)
                for metric in ['retrieval_rouge1', 'retrieval_rouge2', 'retrieval_rougeL',
                             'answer_rouge1', 'answer_rouge2', 'answer_rougeL', 'answer_relevance']:
                    metrics[approach][metric].append(0.0)
    
    # Calculate average metrics
    final_results = {
        approach: {k: np.mean(v) for k, v in approach_metrics.items()}
        for approach, approach_metrics in metrics.items()
    }
    
    # Print results
    print(f"\nResults for {len(nyc_dataset)} NYC-related questions:")
    for approach in ['vanilla', 'hyde', 'selfrag_hyde']:
        print(f"\n{approach.upper()} Results:")
        for metric, value in final_results[approach].items():
            print(f"{metric}: {value:.3f}")
    
    # Save results
    output = {
        'results': final_results,
        'metadata': {
            'dataset': 'squad-nyc',
            'num_examples': len(nyc_dataset),
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'generator': 'gpt-3.5-turbo',
                'relevance_model': 'cross-encoder/qnli-electra-base'
            }
        }
    }
    
    output_file = 'nyc_rag_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()