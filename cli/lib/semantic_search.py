from sentence_transformers import SentenceTransformer
<<<<<<< HEAD
=======
from keyword_search_cli import load_movies
import numpy as np
from pathlib import Path
>>>>>>> 0f7d832 (feat(cli): add text embedding generation to semantic search)

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
<<<<<<< HEAD
=======
        self.embeddings = None
        self.documents = None
        self.document_map = {}
>>>>>>> 0f7d832 (feat(cli): add text embedding generation to semantic search)

    def verify_model(self):
        print(f"Model loaded: {self.model}")
        print(f"Max sequence length: {self.model.max_seq_length}")

<<<<<<< HEAD
=======

    def generate_embedding(self, text):
        if len(text) == 0 or text.isspace():
            raise ValueError("Text most not be empty and shouldn't contain just waithspaces.")


        return self.model.encode([text])[0]


def embed_text(text):
    sim_search = SemanticSearch()
    embedding = sim_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
>>>>>>> 0f7d832 (feat(cli): add text embedding generation to semantic search)
