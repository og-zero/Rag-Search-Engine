import os

from cli.inverted_index import InvertedIndex #keyword search
from .semantic_search import ChunkedSemanticSearch
from cli.keyword_search_cli import load_movies
from cli.llm import (fetch_prompte, fetch_rr_prompte, 
                     fetch_batch_rr_prompte, fetch_llm_evl_prompte)
from sentence_transformers import CrossEncoder
import json

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
        # sorted_res = sorted(result.items(), key=lambda x: x['rerank_response'], reverse=True)
        # print(sorted_res)

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
                'document': self.idx.docmap[id]['description'],
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
        print(r[1]['document'][:100])


def rrf_search_command(query: str, k: int, limit: int, enhance = None, rerank_method = None, evaluate = None):
    movies = load_movies()
    hybred_search = HybridSearch(movies)

    match enhance:
        case "spell":
            enhance_query = fetch_prompte(query, "/home/mzougari/Desktop/Rag-Search-Engine/cli/prompts/enhance.md")
            print(f"Enhanced query (spell): '{query}' -> '{enhance_query}'\n")
            query = enhance_query
        case "rewrite":
            enhance_query = fetch_prompte(query, "/home/mzougari/Desktop/Rag-Search-Engine/cli/prompts/rewriting.md")
            print(f"Enhanced query (rewrite): '{query}' -> '{enhance_query}'\n")
            query = enhance_query
        case "expand":
            enhance_query = fetch_prompte(query, "/home/mzougari/Desktop/Rag-Search-Engine/cli/prompts/expand.md")
            print(f"Enhanced query (expand): '{query}' -> '{enhance_query}'\n")
            query = enhance_query

    candidate_limit = limit * 5 if rerank_method in ("individual", "batch") else limit

    rrf_results = hybred_search.rrf_search(query, k, candidate_limit)

    if rerank_method in ["individual", "batch", "cross_encoder"]:
        results = []

        if rerank_method == "batch":
            prompt = "/home/mzougari/Desktop/Rag-Search-Engine/cli/prompts/batch-re-ranking.md"

            candidates = [
                {
                    "id": doc_id,
                    **doc
                }
                for doc_id, doc in rrf_results
            ]

            doc_list_str = ""
            doc_data = "{doc_id}\n{title}\n{desc}\n\n"


            for idx, doc in enumerate(candidates):
                description = doc.get("description", doc.get("document", ""))

                doc_list_str += doc_data.format(
                    doc_id=idx,
                    title=doc.get("title", ""),
                    desc=description[:300]
                )

            response = fetch_batch_rr_prompte(
                prompt,
                query,
                doc_list_str
            )

            try:
                response_parsed = json.loads(response)

                if not isinstance(response_parsed, list):
                    raise ValueError("Expected a JSON array")
                if (
                    not all(type(idx) is int for idx in response_parsed)
                    or len(set(response_parsed)) != len(response_parsed)
                    or not set(response_parsed).issubset(set(range(len(movies))))
                ):
                    raise ValueError("Invalid or duplicate movie IDs")

                results = []

                for rank, candidate_idx in enumerate(response_parsed, start=1):
                    doc = candidates[candidate_idx]

                    results.append({
                        **doc,
                        "rerank_rank": rank
                    })

                results.sort(
                    key=lambda x: x["rerank_rank"],
                )
                results = results[:limit]
            
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                print(f"Invalid batch reranking response: {response!r}")
                print(f"Error: {e}")
        elif rerank_method == "individual":
            prompt = "/home/mzougari/Desktop/Rag-Search-Engine/cli/prompts/re-ranking.md"

            for doc_id, doc in rrf_results:
                response = fetch_rr_prompte(
                    prompt,
                    query,
                    doc["title"],
                    doc["document"]
                )
                try:
                    score = float(response)

                except (ValueError, TypeError):
                    print(f"Invalid reranking score: {response!r}")
                    continue

                results.append({
                    **doc,
                    "id": doc_id,
                    "rerank_response": score
                })

            results.sort(
                key=lambda x: x["rerank_response"],
            )

            results = results[:limit]
        elif rerank_method == "cross_encoder":
            candidate_limit = limit * 5
            rrf_results = hybred_search.rrf_search(query, k, candidate_limit)
            pairs = []

            candidates = [
                {
                    "id": doc_id,
                    **doc
                }
                for doc_id, doc in rrf_results
            ]

            for doc in candidates:
                pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            scores = cross_encoder.predict(pairs)
            results = []
            for idx, doc in enumerate(candidates):
               
                results.append(
                    {
                        **doc,
                        "cross_encoder_rr": scores[idx]
                    }
                )
            results.sort(key=lambda x: x["cross_encoder_rr"], reverse=True)
            results = results[:limit]

    else:
        results = [
            {"id": doc_id, **doc}
            for doc_id, doc in rrf_results
        ]
    for i, doc in enumerate(results, 1):

        if rerank_method == "individual":
            print(f"Rerank Score: {doc['rerank_response']:.3f}")

        elif rerank_method == "batch":
            print(f"Re-ranking top {limit} results using batch method...")
            print(f"Reciprocal Rank Fusion Results for '{query}' (k=60):\n")
            print(f"Re-rank Rank: {doc['rerank_rank']}")
            print(f"\n{i}. {doc['title']}")
            print(f"RRF Score: {doc['rrf_rank']:.3f}")
            print(f"BM25 Rank: {doc['bm25_rank']}, Semantic Rank: {doc['semantic_rank']}")
            print(doc["document"][:100])

        elif rerank_method == "cross_encoder":
            print(f"Re-ranking top {limit*5} results using cross_encoder method...")
            print(f"Reciprocal Rank Fusion Results for {query} (k=60):\n")
            print(f"Cross Encoder Score: {doc['cross_encoder_rr']:.3f}")
            print(f"{i}. {doc['title']}")
            print(f"RRF Score: {doc['rrf_rank']:.3f}")
            print(f"BM25 Rank: {doc['bm25_rank']}, Semantic Rank: {doc['semantic_rank']}")
            print(doc["document"][:100])

    if evaluate:
        json_res = fetch_llm_evl_prompte("/home/mzougari/Desktop/Rag-Search-Engine/cli/prompts/llm_evaluation.md", query, results)
        res = json.loads(json_res)
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc['title']}: {res[i]}/3")