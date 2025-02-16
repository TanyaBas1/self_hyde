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

from src.hyde import Promptor, OpenAIGenerator, HyDE
from src.hyde.searcher import Encoder, Searcher
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

    def compare_approaches(
        self,
        vanilla_retriever: VanillaRetriever,
        hyde_retriever: HyDERetriever,
        generator,
        eval_dataset,
        num_samples: int = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare HyDE and vanilla approaches"""
        self.logger.info(f"Starting comparison with {num_samples or 'all'} samples")
        
        metrics = {
            'hyde': defaultdict(list),
            'vanilla': defaultdict(list)
        }
        
        eval_data = eval_dataset['validation']
        if num_samples:
            indices = np.random.choice(len(eval_data), num_samples, replace=False)
            eval_data = [eval_data[i] for i in indices]
        
        for sample in tqdm(eval_data, desc="Evaluating samples"):
            query = sample['question']
            
            for approach, retriever in [
                ('vanilla', vanilla_retriever),
                ('hyde', hyde_retriever)
            ]:
                try:
                    # Measure retrieval time
                    start_time = time.time()
                    retrieved_docs = retriever.retrieve(query)
                    retrieval_time = time.time() - start_time
                    
                    # Generate final answer
                    final_prompt = f"""Based on the retrieved documents, answer the question: {query}
                    Retrieved information: {' '.join(retrieved_docs)}
                    Please provide a clear and concise answer:"""
                    
                    predicted_answer = generator.generate(final_prompt)[0]
                    
                    # Evaluate retrieval and answer
                    retrieval_metrics = self.evaluate_retrieval(retrieved_docs, sample['context'])
                    answer_metrics = self.evaluate_answer(predicted_answer, sample['answers']['text'])
                    
                    # Store metrics
                    metrics[approach]['retrieval_time'].append(retrieval_time)
                    for k, v in {**retrieval_metrics, **answer_metrics}.items():
                        metrics[approach][k].append(v)
                        
                except Exception as e:
                    self.logger.error(f"Error evaluating {approach} approach: {str(e)}")
                    # Add zero values for failed evaluations
                    metrics[approach]['retrieval_time'].append(0.0)
                    for metric in ['retrieval_rouge1', 'retrieval_rouge2', 'retrieval_rougeL',
                                 'answer_rouge1', 'answer_rouge2', 'answer_rougeL', 'answer_relevance']:
                        metrics[approach][metric].append(0.0)
        
        # Calculate average metrics
        results = {
            approach: {k: np.mean(v) for k, v in approach_metrics.items()}
            for approach, approach_metrics in metrics.items()
        }
        
        self.logger.info("Evaluation completed successfully")
        return results

def print_comparative_results(results: Dict[str, Dict[str, float]]):
    """Pretty print comparative evaluation results"""
    print("\nComparative RAG Evaluation Results:")
    print("-" * 60)
    
    metrics_groups = {
        'Retrieval Metrics': 'retrieval',
        'Answer Metrics': 'answer',
        'Performance Metrics': 'time'
    }
    
    # Calculate improvement percentages
    improvements = {
        k: ((results['hyde'][k] - results['vanilla'][k]) / results['vanilla'][k] * 100)
        for k in results['hyde'].keys()
    }
    
    for group_name, prefix in metrics_groups.items():
        print(f"\n{group_name}:")
        print("-" * 40)
        print(f"{'Metric':25s} {'Vanilla':10s} {'HyDE':10s} {'% Improvement':15s}")
        print("-" * 60)
        
        for k in results['hyde'].keys():
            if k.startswith(prefix):
                print(f"{k:25s} {results['vanilla'][k]:10.4f} {results['hyde'][k]:10.4f} {improvements[k]:15.2f}%")

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # the size of NYC subset of the eval dataset is 817, testing on 100 to speed up iteration 
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
    encoder = Encoder()
    
    # Create corpus from the NYC dataset
    corpus = {item['id']: item['context'] for item in nyc_dataset}
    searcher = Searcher(corpus)
    
    # Initialize HyDE
    hyde = HyDE(promptor, generator, encoder, searcher)
    evaluator = RAGEvaluator()
    
    # Run evaluation comparing vanilla search vs HyDE
    results = {
        'vanilla': defaultdict(list),
        'hyde': defaultdict(list)
    }
    
    for item in tqdm(nyc_dataset, desc="Evaluating NYC questions"):
        question = item['question']
        context = item['context']
        answers = item['answers']['text']
        
        # Vanilla search
        question_vector = encoder.encode(question)
        vanilla_hits = searcher.search(question_vector, k=5)
        vanilla_docs = [corpus[hit.docid] for hit in vanilla_hits]
        
        # HyDE search
        hyde_hits = hyde.e2e_search(question, k=5)
        hyde_docs = [corpus[hit.docid] for hit in hyde_hits]
        
        # Evaluate both approaches
        for approach, docs in [('vanilla', vanilla_docs), ('hyde', hyde_docs)]:
            # Generate answer
            prompt = f"""Based on the retrieved documents, answer the question: {question}
            Retrieved information: {' '.join(docs)}
            Please provide a clear and concise answer:"""
            
            predicted_answer = generator.generate(prompt)[0]
            
            # Evaluate retrieval and answer
            retrieval_metrics = evaluator.evaluate_retrieval(docs, context)
            answer_metrics = evaluator.evaluate_answer(predicted_answer, answers)
            
            # Store metrics
            for k, v in {**retrieval_metrics, **answer_metrics}.items():
                results[approach][k].append(v)
    
    # Calculate average metrics
    final_results = {
        approach: {k: np.mean(v) for k, v in approach_metrics.items()}
        for approach, approach_metrics in results.items()
    }
    
    # Print results
    print(f"\nResults for {len(nyc_dataset)} NYC-related questions:")
    print("\nVanilla Search Results:")
    for metric, value in final_results['vanilla'].items():
        print(f"{metric}: {value:.3f}")
    
    print("\nHyDE Results:")
    for metric, value in final_results['hyde'].items():
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