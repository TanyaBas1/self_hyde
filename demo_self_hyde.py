import os
import json
import logging
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Import the components
from hyde import Promptor, SelfRAGHyDE
from src.hyde.searcher import Encoder, Searcher
from src.hyde.segment_scorer import SegmentScorer, ReflectionTokens
from src.hyde.generator import SelfRAGGenerator
from src.hyde.config import get_openai_api_key

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("self-rag-demo")
console = Console()

def print_section(title, content):
    rprint(Panel(str(content), title=title, border_style="cyan"))

def print_results_table(results):
    table = Table(title="Segment Results")
    table.add_column("Segment #", style="cyan")
    table.add_column("Content", style="white")
    table.add_column("Relevance", style="green")
    table.add_column("Support", style="yellow")
    table.add_column("Score", style="red")
    
    for i, (segment, critique, score) in enumerate(results, 1):
        table.add_row(
            str(i),
            segment[:100] + "...",  # Truncate long segments
            critique['relevance'],
            critique['support'],
            f"{score:.2f}"
        )
    
    console.print(table)

def demo_step_by_step():
    """Demonstrate step-by-step usage of Self-RAG HyDE"""
    logger.info("Loading document corpus...")
    with open('new_york_corpus.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Initialize components
    promptor = Promptor('web search')
    generator = SelfRAGGenerator('gpt-3.5-turbo', os.getenv('OPENAI_API_KEY'))
    encoder = Encoder()
    searcher = Searcher(documents)
    hyde = SelfRAGHyDE(promptor, generator, encoder, searcher)

    # Example query
    query = "When New York was the capital of the US?"
    print_section("Query", query)

    logger.info("Step 1: Generating initial prompt...")
    prompt = hyde.prompt(query)
    print_section("Generated Prompt", prompt)

    logger.info("Step 2: Making retrieval decision and generating hypotheses...")
    hypothesis_docs = hyde.generate(query)
    
    if not hypothesis_docs:
        print_section("Retrieval Decision", "No retrieval needed for this query")
        response, critique = generator.generate_with_reflection(query)
        print_section("Direct Response", response)
        print_section("Critique", critique)
        return

    print_section("Hypothesis Documents", "\n\n".join(hypothesis_docs))

    logger.info("Step 3: Encoding query and hypotheses...")
    hyde_vector = hyde.encode(query, hypothesis_docs)

    logger.info("Step 4: Performing search...")
    hits = hyde.search(hyde_vector, k=3)
    
    console.print("\n[bold cyan]Retrieved Documents:[/bold cyan]")
    for i, hit in enumerate(hits, 1):
        print_section(f"Retrieved Document #{i}", 
                     f"Score: {hit.score:.4f}\n\n{documents[hit.docid]}")

    # Step 5: Generate final response
    logger.info("Step 5: Generating final response with reflection...")
    retrieved_docs = [documents[hit.docid] for hit in hits]
    response, critique = hyde.generate_response(query, retrieved_docs)
    
    print_section("Final Response", response)
    print_section("Final Critique", 
                 f"Relevance: {critique['relevance']}\nSupport: {critique['support']}")

def demo_e2e():
    """Demonstrate end-to-end usage of Self-RAG HyDE"""
    logger.info("Loading document corpus...")
    with open('new_york_corpus.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Initialize components
    promptor = Promptor('web search')
    generator = SelfRAGGenerator('gpt-3.5-turbo', os.getenv('OPENAI_API_KEY'))
    encoder = Encoder()
    searcher = Searcher(documents)
    hyde = SelfRAGHyDE(promptor, generator, encoder, searcher)

    queries = [
        "What is your favourite color?", # an example of non relevant query 
        "When New York was the capital of the US?",
    ]

    for query in queries:
        console.print(f"\n[bold magenta]Processing Query: {query}[/bold magenta]")
        results = hyde.get_intermediate_results(query)
        
        # Print retrieval decision
        if not results['hypothesis_documents']:
            print_section("Retrieval Decision", "No retrieval needed")
        else:
            print_section("Retrieval Decision", "Retrieval performed")
            print_section("Retrieved Documents", 
                         f"Found {len(results['hits'])} relevant documents")
        
        # Print final response and critique
        print_section("Final Response", results['final_response'])
        print_section("Critique", 
                     f"Relevance: {results['critique']['relevance']}\n"
                     f"Support: {results['critique']['support']}")
        
        console.print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    console.print("\n[bold magenta]Self-RAG HyDE Demo[/bold magenta]\n")
    
    console.print("\n[bold]Step-by-Step Demo:[/bold]")
    demo_step_by_step()
    
    console.print("\n[bold]End-to-End Demo:[/bold]")
    demo_e2e()