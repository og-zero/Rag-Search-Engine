from sentence_transformers import SentenceTransformer
from keyword_search_cli import load_movies
import numpy as np
from pathlib import Path

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
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    sim_search = SemanticSearch()
    embedding = sim_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape[0]}")