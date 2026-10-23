import os

from cli.inverted_index import InvertedIndex #keyword search
from .semantic_search import ChunkedSemanticSearch
from cli.keyword_search_cli import load_movies
from cli.llm import fetch_prompte

class HybridSearch():
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.mapping_documents_ids = {}

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        self.idx.build()
        BM25_results = self.idx.bm25_search(query, k1=1.5, b=0.75, limit=limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)
        Semantic_result = [{key: d[key] for key in ("id", "score") if key in d} for d in semantic_results]

        bm25_scores = {id: score for id, score in BM25_results}
        semantic_scores = {d["id"]: float(d["score"]) for d in Semantic_result if "id" in d and "score" in d}

        norm_bm25 = normalize(bm25_scores)
        norm_semantic = normalize(semantic_scores)

        all_doc_ids = set(bm25_scores.keys()).union(semantic_scores.keys())

        for id in all_doc_ids:
            k_score = norm_bm25.get(id, 0.0)
            s_score = norm_semantic.get(id, 0.0)
        
            h_score = hybrid_score(k_score, s_score, alpha)
           
            self.mapping_documents_ids[id] = {
                    "title": self.idx.docmap[id]['title'],
                    "description": self.idx.docmap[id]['description'],
                    "keyword_score": bm25_scores.get(id, 0.0),
                    "semantic_score": semantic_scores.get(id, 0.0),
                    "normalized_keyword": k_score,
                    "normalized_semantic": s_score,
                    "hybrid_score": h_score
                }

        sorted_results = sorted(self.mapping_documents_ids.items(),
               key=lambda x: x[1]['hybrid_score'],
               reverse=True
            )
        return sorted_results[:limit]
        

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        self.idx.load()
        BM25_results = self.idx.bm25_search(query, k1=1.5, b=0.75, limit=limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)
        Semantic_result = [{key: d[key] for key in ("id", "score") if key in d} for d in semantic_results]

        bm25_ranks = {d[0]: rank+1 for rank, d in enumerate(BM25_results)}
        semantic_ranks = {d["id"]: rank+1 for rank, d in enumerate(Semantic_result) if "id" in d}

        mapping_doc_ids = {}
        all_doc_ids = set(bm25_ranks.keys()).union(semantic_ranks.keys())
        for id in all_doc_ids:
            rrf_rank_BM25 = rrf_score(bm25_ranks.get(id), k)
            rrf_rank_semantic = rrf_score(semantic_ranks.get(id), k)
            
            rrf_rank = rrf_rank_BM25 + rrf_rank_semantic

            mapping_doc_ids[id] = {
                'title': self.idx.docmap[id]['title'],
                'description': self.idx.docmap[id]['description'],
                'bm25_rank': bm25_ranks.get(id),
                'semantic_rank': semantic_ranks.get(id),
                'rrf_rank': rrf_rank
            }

        sorted_results = sorted(mapping_doc_ids.items(), 
                               key=lambda x: x[1]['rrf_rank'],
                               reverse=True)
        return sorted_results[:limit]


def rrf_score(rank: int, k: int = 60) -> float:
    if rank is None:
        return 0.0
    return 1 / (k + rank)

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize(scores: dict) -> dict:
    """Normalizes a dictionary of {doc_id: raw_score} using Min-Max scaling to a 0-1 range."""
    if not scores:
        return {}
    
    min_score = min(scores.values())
    max_score = max(scores.values())
    
    if max_score == min_score:
        return {id: 1.0 for id in scores}
        
    return {
        id: (score - min_score) / (max_score - min_score) 
        for id, score in scores.items()
    }

def normalize_command(scores: list):
    if not scores:
        return
    
    min_score = float(min(scores))
    max_score = float(max(scores))

    if min_score == max_score:
        for s in scores:
            print(f"* 1.0")


    denominator = max_score - min_score
    if denominator == 0:
        for score in scores:
            print(f"* 0.0000")
    
    for score in scores:
        scor = (float(score) - min_score) / denominator
        print(f"* {scor:.4f}")


def weighted_search_command(query, alpha, limit):
    movies = load_movies()
    hybred_search = HybridSearch(movies)
    result = hybred_search.weighted_search(query, alpha, limit)

    for i, r in enumerate(result):
        print(f"{i+1}. {r[1]['title']}")
        print(f"Hybrid Score: {r[1]['hybrid_score']:.3f}")
        print(f"BM25: {r[1]['normalized_keyword']:.3f}, Semantic: {r[1]['normalized_semantic']:.3f}")
        print(r[1]['description'][:100])


def rrf_search_command(query: str, k: int, limit: int, enhance = None):
    movies = load_movies()
    hybred_search = HybridSearch(movies)

    match enhance:
        case "spell":
            enhance_query = fetch_prompte(query, "/home/mzougari/Desktop/Rag-Search-Engine/cli/prompts/enhance.md")
            print(f"Enhanced query (spell): '{query}' -> '{enhance_query}'\n")
            query = enhance_query
    result = hybred_search.rrf_search(query, k, limit)

    for i, r in enumerate(result):
        print(f"{i+1}. {r[1]['title']}")
        print(f"RRF Score: {r[1]['rrf_rank']:.3f}")
        print(f"BM25 Rank: {r[1]['bm25_rank']}, Semantic Rank: {r[1]['semantic_rank']}")
        print(r[1]['description'][:100])