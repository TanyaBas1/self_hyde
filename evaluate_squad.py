import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from typing import Dict, List, Tuple, NamedTuple
from collections import defaultdict
import logging
from datetime import datetime

from src.hyde import Promptor, OpenAIGenerator, HyDE
from src.hyde.searcher import Encoder, Searcher
from hyde.config import get_openai_api_key

"""
structure of the dataset:
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 87599
    })
    validation: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 10570
    })
})
sample output 
first_example = squad_dataset['train'][0]

{'id': '5733be284776f41900661182',
 'title': 'University_of_Notre_Dame',
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'answers': {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}}

"""

class EvaluationResult(NamedTuple):
    """Store evaluation metrics"""
    mrr: float  # Mean Reciprocal Rank
    hits_at_1: float  # Accuracy of top result
    hits_at_5: float  # Accuracy in top 5 results
    hits_at_10: float  # Accuracy in top 10 results
    mean_position: float  # Average position of correct answer

def create_corpus_from_squad(dataset):
    """Create corpus from SQuAD dataset with unique identifiers"""
    corpus = {}
    for item in dataset:
        # Using id as docid since it's unique in SQuAD
        corpus[item['id']] = item['context']
    return corpus

def evaluate_squad(num_examples=100):
    """Main evaluation function"""
    # Load dataset
    squad_dataset = load_dataset("squad")
    
    # We'll use validation set for evaluation
    eval_set = squad_dataset['validation'].select(range(num_examples))
    
    # Create corpus from all contexts in the validation set
    print("Building corpus...")
    corpus = create_corpus_from_squad(eval_set)
    
    # Initialize components
    promptor = Promptor('web search')
    generator = OpenAIGenerator('gpt-3.5-turbo', get_openai_api_key())
    encoder = Encoder()
    searcher = Searcher(corpus)
    
    hyde = HyDE(promptor, generator, encoder, searcher)
    
    def get_correct_positions(hits: List, item_id: str) -> List[int]:
        """Find positions where correct context appears in results"""
        positions = []
        correct_context = corpus[item_id]
        for i, hit in enumerate(hits):
            hit_text = corpus[hit.docid]
            # Use string similarity instead of exact match
            if string_similarity(hit_text, correct_context) > 0.95:
                positions.append(i + 1)
        return positions
    
    # Results storage
    baseline_results = []
    hyde_results = []
    
    print(f"Evaluating {len(eval_set)} questions...")
    for item in tqdm(eval_set):
        question = item['question']
        item_id = item['id']
        
        # Baseline search
        try:
            question_vector = encoder.encode(question)
            baseline_hits = searcher.search(question_vector, k=10)
            baseline_positions = get_correct_positions(baseline_hits, item_id)
            baseline_results.append(baseline_positions)
        except Exception as e:
            print(f"Error in baseline search for question: {question}")
            print(f"Error: {e}")
            baseline_results.append(None)
        
        # HyDE search
        try:
            hyde_hits = hyde.e2e_search(question, k=10)
            hyde_positions = get_correct_positions(hyde_hits, item_id)
            hyde_results.append(hyde_positions)
        except Exception as e:
            print(f"Error in HyDE search for question: {question}")
            print(f"Error: {e}")
            hyde_results.append(None)
    
    # Calculate metrics
    baseline_metrics = calculate_metrics(baseline_results)
    hyde_metrics = calculate_metrics(hyde_results)
    
    return baseline_metrics, hyde_metrics, eval_set

def string_similarity(s1: str, s2: str) -> float:
    """Calculate string similarity using character-level comparison"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, s1, s2).ratio()

def calculate_metrics(results: List[List[int]]) -> EvaluationResult:
    """Calculate evaluation metrics from result positions"""
    valid_results = [r for r in results if r is not None and r]  # Only include non-empty valid results
    
    if not valid_results:
        return EvaluationResult(
            mrr=0.0,
            hits_at_1=0.0,
            hits_at_5=0.0,
            hits_at_10=0.0,
            mean_position=float('inf')
        )
    
    # Calculate MRR
    mrr = np.mean([1/min(positions) if positions else 0 
                   for positions in valid_results])
    
    # Calculate Hits@K
    hits_1 = np.mean([1 if positions and min(positions) <= 1 else 0 
                      for positions in valid_results])
    hits_5 = np.mean([1 if positions and min(positions) <= 5 else 0 
                      for positions in valid_results])
    hits_10 = np.mean([1 if positions and min(positions) <= 10 else 0 
                       for positions in valid_results])
    
    # Calculate mean position
    mean_pos = np.mean([min(positions) if positions else float('inf') 
                        for positions in valid_results])
    
    return EvaluationResult(
        mrr=mrr,
        hits_at_1=hits_1,
        hits_at_5=hits_5,
        hits_at_10=hits_10,
        mean_position=mean_pos
    )

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    NUM_EXAMPLES = 100  # Start with 100 examples for testing
    
    logger.info(f"Starting SQuAD evaluation with {NUM_EXAMPLES} examples...")
    
    try:
        baseline_metrics, hyde_metrics, eval_set = evaluate_squad(NUM_EXAMPLES)
        
        # Print results
        print("\nBaseline Results:")
        print(f"MRR: {baseline_metrics.mrr:.3f}")
        print(f"Hits@1: {baseline_metrics.hits_at_1:.3f}")
        print(f"Hits@5: {baseline_metrics.hits_at_5:.3f}")
        print(f"Hits@10: {baseline_metrics.hits_at_10:.3f}")
        print(f"Mean Position: {baseline_metrics.mean_position:.2f}")
        
        print("\nHyDE Results:")
        print(f"MRR: {hyde_metrics.mrr:.3f}")
        print(f"Hits@1: {hyde_metrics.hits_at_1:.3f}")
        print(f"Hits@5: {hyde_metrics.hits_at_5:.3f}")
        print(f"Hits@10: {hyde_metrics.hits_at_10:.3f}")
        print(f"Mean Position: {hyde_metrics.mean_position:.2f}")
        
        # Save results with detailed metadata
        results = {
            'baseline': baseline_metrics._asdict(),
            'hyde': hyde_metrics._asdict(),
            'metadata': {
                'dataset': 'squad',
                'split': 'validation',
                'num_examples': NUM_EXAMPLES,
                'timestamp': datetime.now().isoformat(),
                'sample_questions': [
                    {
                        'id': eval_set[i]['id'],
                        'question': eval_set[i]['question'],
                        'answer': eval_set[i]['answers']['text'][0]
                    }
                    for i in range(min(5, len(eval_set)))  # Save first 5 questions as samples
                ]
            }
        }
        
        output_file = f'squad_evaluation_results_{NUM_EXAMPLES}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()