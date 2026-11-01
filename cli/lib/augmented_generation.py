from lib.hybrid_search import HybridSearch
from cli.keyword_search_cli import load_movies
from cli.llm import fetch_rag_prompte, fetch_llm_summarize_prmpt

movies = load_movies()
idx = HybridSearch(movies)

def summarize_command(query, limit):
    rrf_search_res = idx.rrf_search(query=query, k=60, limit=5)
    
    retrieved_documents = []
    for doc_id, doc in rrf_search_res:
        retrieved_documents.append({**doc})
    
    res = fetch_llm_summarize_prmpt("/home/mzougari/Desktop/Rag-Search-Engine/cli/prompts/llm_summarization.md",
                            query, 
                            retrieved_documents
                            )

    print("Search Results:")
    for doc_id, doc in rrf_search_res:
        print(f"- {doc['title']}")
        
    print("\nLLM Summary:")
    print(res)

def rag_command(query):
    rrf_search_res = idx.rrf_search(query=query, k=60, limit=5)
    retrieved_documents = []
    for doc_id, doc in rrf_search_res:
        retrieved_documents.append({**doc})

    res = fetch_rag_prompte("/home/mzougari/Desktop/Rag-Search-Engine/cli/prompts/Augmented_Generation_prompte.md", query, retrieved_documents)

    print("Search Results:")
    for doc_id, doc in rrf_search_res:
        print(f"- {doc['title']}")
    
    print("\nRAG Response:")
    print(res)