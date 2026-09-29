from sentence_transformers import SentenceTransformer
from keyword_search_cli import load_movies
import numpy as np
from pathlib import Path
import numpy as np

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = None
        self.documents = None
        self.document_map = {}


    def verify_model(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")


    def generate_embedding(self, text):
        if len(text) == 0 or text.isspace():
            raise ValueError("Text most not be empty and shouldn't contain just waithspaces.")

        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        movie_strings = []

        for doc in self.documents:
            self.document_map[doc['id']] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_strings)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings (self, documents):
        self.documents = documents
        self.document_map = {}

        for doc in self.documents:
            self.document_map[doc['id']] = doc

        if Path("cache/movie_embeddings.npy").exists():
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
            
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if not Path("cache/movie_embeddings.npy").exists():
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embeddings = self.generate_embedding(query)
        similarities = []
        for (doc_emb, doc) in zip(self.embeddings, self.documents):
            similarity_score = cosine_similarity(doc_emb, query_embeddings)
            similarities.append((similarity_score, doc))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [{"score": sim, 
               'title': doc['title'],
               'description': doc['description']}
              for sim, doc in similarities[:limit]]
        
            
def embed_text(text):
    sim_search = SemanticSearch()
    embedding = sim_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    sim_search = SemanticSearch()
    documents = list(load_movies())
    embeddings = sim_search.load_or_create_embeddings(documents)
    print(f"Number of docs:  {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    sim_search = SemanticSearch()
    embedding = sim_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape[0]}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_command(query, limit):
    sim_search = SemanticSearch()
    documents = list(load_movies())
    sim_search.load_or_create_embeddings(documents)
    result = sim_search.search(query, limit)

    for i, res in enumerate(result):
        print(f"{i+1}. {res['title']} (score: {res['score']:.4f})\n"
              f"{res['description'][:100]}"
            )

def chunking_command(text, n):
    splited_text = text.split()
    chunked_list = [" ".join(splited_text[i:i+n]) for i in range(0, len(splited_text), n)]

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunked_list):
        print(f"{i+1}. {chunk}")