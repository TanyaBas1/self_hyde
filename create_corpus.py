import wikipedia
import json
from bs4 import BeautifulSoup
import re

def clean_text(text):
    # Remove citations 
    text = re.sub(r'\[\d+\]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_wiki_content(title):
    page = wikipedia.page(title, auto_suggest=False)
    paragraphs = page.content.split('\n')
    cleaned_paragraphs = [clean_text(p) for p in paragraphs if len(p.strip()) > 50]
    return cleaned_paragraphs
   

def create_marie_curie_corpus():
    topics = [
        "Marie Curie",
        "Pierre Curie",
        "Radioactivity",
        "Radium",
        "Polonium",
        "Nobel Prize in Physics",
        "Nobel Prize in Chemistry",
        "Institut Curie",
        "X-ray",
        "Radiation"
    ]
    
    documents = {}
    doc_id = 1
    
    for topic in topics:
        print(f"Fetching content for: {topic}")
        paragraphs = get_wiki_content(topic)
        
        # Create a document for each substantial paragraph
        for para in paragraphs:
            if len(para) > 100:  
                doc_key = f"doc{doc_id}"
                documents[doc_key] = para
                doc_id += 1
    
    with open('marie_curie_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"Created corpus with {len(documents)} documents")
    return documents

if __name__ == "__main__":
    documents = create_marie_curie_corpus() 
