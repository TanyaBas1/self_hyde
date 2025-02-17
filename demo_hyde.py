import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from rich.logging import RichHandler
from src.hyde import Promptor, OpenAIGenerator, HyDE
from src.hyde.searcher import Encoder, Searcher
from hyde.config import get_openai_api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("hyde-demo")
console = Console()

def print_section(title, content):
    rprint(Panel(str(content), title=title, border_style="cyan"))

def main():
    logger.info("Loading document corpus...")
    with open('new_york_corpus.json', 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # initialize components 
    promptor = Promptor('web search')
    generator = OpenAIGenerator('gpt-3.5-turbo', get_openai_api_key())
    encoder = Encoder()
    searcher = Searcher(documents)
    hyde = HyDE(promptor, generator, encoder, searcher)

    query = "When New York was the capital of the US?"
    print_section("Query", query)
    
    prompt = hyde.prompt(query)
    print_section("Generated Prompt", prompt)
    
    # generate hypothesis documents
    logger.info("Generating hypothesis documents...")
    hypothesis_documents = hyde.generate(query)
    for i, doc in enumerate(hypothesis_documents, 1):
        print_section(f"Hypothesis Document #{i}", doc.strip())
    
    # encode and search
    logger.info("Performing semantic search...")
    hyde_vector = hyde.encode(query, hypothesis_documents)
    hits = hyde.search(hyde_vector, k=5)
    
    # show retrieved documents (based on hypothesis documents )
    console.print("\n[bold cyan]Top Retrieved Documents:[/bold cyan]")
    for i, hit in enumerate(hits, 1):
        panel_content = f"Document ID: {hit.docid}\n\n{documents[hit.docid]}"
        print_section(f"Retrieved Document #{i}", panel_content)
    
    # generate final response
    logger.info("Generating final response...")
    final_prompt = f"""Based on the retrieved documents, please provide a comprehensive answer to the question: {query}
                    Retrieved information:
                    {' '.join(documents[hit.docid] for hit in hits)}

                    Please provide a clear and concise answer:"""
    
    final_response = generator.generate(final_prompt)[0]
    print_section("Final Answer", final_response)

if __name__ == "__main__":
    console.print("\n[bold magenta] HyDE Demo - New York Query[/bold magenta]\n")
    main()
    