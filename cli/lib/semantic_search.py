from sentence_transformers import SentenceTransformer
from keyword_search_cli import load_movies
import numpy as np
from pathlib import Path
import numpy as np
import re
import os
import json
from tqdm import tqdm

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
        
        
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
     
    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        
        self.documents = documents
        self.document_map = {}

        for doc in self.documents:
            self.document_map[doc['id']] = doc
            
        all_chunks = []
        metadata = []
    
        for midx, doc in enumerate(self.documents):
            if not doc.get('description'):
                continue
            
            chunks = simantic_chunking(doc['description'], 4, 1)
            
            for cidx, chunk_text in enumerate(chunks):
                all_chunks.append(chunk_text) 
                
                metadata.append({
                    'movie_idx': midx,
                    'movie_id': doc['id'],
                    'chunk_idx': cidx,
                })

        print(f"Total chunks generated: {len(all_chunks)}. Starting embedding calculation...")
        

        self.chunk_embeddings = self.model.encode(
            all_chunks,
        )
        
        self.chunk_metadata = metadata
        

        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        return self.chunk_embeddings

        
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}

        for doc in self.documents:
            self.document_map[doc['id']] = doc
            
        embeddings_path = Path("cache/chunk_embeddings.npy")
        metadata_path = Path("cache/chunk_metadata.json")

        if embeddings_path.exists() and metadata_path.exists():
            self.chunk_embeddings = np.load(embeddings_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.chunk_metadata = json.load(f)

            return self.chunk_embeddings
            
        return self.build_chunk_embeddings(self.documents)
            
    
    def search_chunks(self, query: str, limit: int = 10):
        query_embedings = self.generate_embedding(query)
        chunks_score = []
        movie_score = {}
        res = []
        
        for cidx, chunk in enumerate(self.chunk_embeddings):
            cosin_sim = cosine_similarity(query_embedings, chunk)
            metadata = self.chunk_metadata[cidx]
            actual_movie_id = metadata['movie_idx']
            chunks_score.append(
                {
                    'chunk_idx': metadata['chunk_idx'],
                    'movie_idx': actual_movie_id,
                    'chunk_metadata': metadata,
                    'score': cosin_sim
                }
            )
            if actual_movie_id in movie_score:
                movie_score[actual_movie_id] = max(movie_score[actual_movie_id], cosin_sim)
            else:
                movie_score[actual_movie_id] = cosin_sim

        sorted_movie_score = sorted(movie_score.items(), key=lambda x: x[1], reverse=True)
        for midx, score in sorted_movie_score[:limit]:
            doc = self.documents[midx]
            res.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": round(score, 3),
                }
            )
        return res
        
          
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
        
        
def simantic_chunking(text, n, ov):
    text = text.strip()
    if not text:
        return []
    
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunked_list = []

    if len(sentences) == 1 and sentences[0].endswith(('.', '?', '!')):
        pass

    for i in range(0, len(sentences), n-ov):
        sentence = " ".join(sentences[i:i+n])
        chunked_sentence = sentence.strip()
        if not chunked_sentence:
            continue
        
        chunked_list.append(chunked_sentence)

        if i + n >= len(sentences):
            break
    
    return chunked_list

def embed_chunks_command():
    documents = load_movies()
    sim_search = ChunkedSemanticSearch()
    embeddings = sim_search.load_or_create_chunk_embeddings(documents)
    
    print(f"Generated {len(embeddings)} chunked embeddings")
    
    
def seach_chunks_command(query, limit):
    documents = load_movies()
    sim_search = ChunkedSemanticSearch()
    embeddings = sim_search.load_or_create_chunk_embeddings(documents)
    res = sim_search.search_chunks(query, limit)
    
    for i, r in enumerate(res):
        print(f"\n{i}. {r['title']} (score: {r['score']:.4f})")
        print(f"   {r['document']}...")