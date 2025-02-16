import wikipedia
import json
import re

def clean_text(text):
    # Remove citations
    text = re.sub(r'\[\d+\]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_wiki_content(title):
    try:
        page = wikipedia.page(title, auto_suggest=False)
    except Exception as e:
        print(f"Error fetching page for {title}: {e}")
        return []
    paragraphs = page.content.split('\n')
    cleaned_paragraphs = [clean_text(p) for p in paragraphs if len(p.strip()) > 50]
    return cleaned_paragraphs

def create_new_york_corpus():
    # A list of topics covering historical, cultural, and infrastructural aspects of New York
    topics = [
        "New York City",
        "History of New York City",
        "Boroughs of New York City",
        "Manhattan",
        "Brooklyn",
        "Queens",
        "The Bronx",
        "Staten Island",
        "New York City Subway",
        "Times Square",
        "Central Park",
        "Wall Street",
        "Financial District, New York City",
        "United Nations Headquarters",
        "Statue of Liberty",
        "Ellis Island",
        "Empire State Building",
        "One World Trade Center",
        "9/11 Memorial",
        "Brooklyn Bridge",
        "New York Public Library",
        "Metropolitan Museum of Art",
        "Solomon R. Guggenheim Museum",
        "Broadway Bridge (Manhattan)",
        "New York Fashion Week",
        "Harlem Renaissance",
        "Immigration to New York City",
        "New York City Draft Riots",
        "Triangle Shirtwaist Factory fire",
        "Peter Stuyvesant",
        "Lenape people",
        "Gentrification in New York City",
        "New York City Parks",
        "Coney Island",
        "New York Yankees",
        "New York Mets",
        "Madison Square Garden",
        "New York City cuisine",
        "New York City skyline",
        "New York City media",
        "New York City Council",
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
    
    with open('new_york_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"Created corpus with {len(documents)} documents")
    return documents

if __name__ == "__main__":
    documents = create_new_york_corpus()
