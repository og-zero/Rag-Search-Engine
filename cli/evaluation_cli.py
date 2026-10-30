from lib.hybrid_search import HybridSearch
from keyword_search_cli import load_movies
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json", "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    movies = load_movies()
    rrf_results = []
    idx = HybridSearch(movies)
    for data in golden_dataset["test_cases"]:
        rrf_res = idx.rrf_search(query=data['query'], k=60, limit=limit)
        retrieved_titles = [r[1]['title'] for r in rrf_res]
        
        relevant_docs = data["relevant_docs"]

        relevant_retrieved = sum(
            1 for title in retrieved_titles
            if title in relevant_docs
        )

        precision = relevant_retrieved / len(retrieved_titles) if retrieved_titles else 0
        recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
        f1 = 2 * (precision * recall) / (precision + recall)
        rrf_results.append(
            {
                'query': data['query'],
                'precision': precision,
                'retrieved': retrieved_titles,
                'relevant': data['relevant_docs'],
                'recall': recall,
                'f1_score': f1
            }
        )

    print(f"k={limit}\n")
    for r in rrf_results:
        print(f"- Query: {r['query']}")
        print(f"- Precision@{limit}: {r['precision']:.4f}")
        print(f"- Recall@{limit}: {r['recall']:.4f}")
        print(f"- F1 Score: {r['f1_score']:.4f}")
        print(f"- Retrieved: {r['retrieved'] }")
        print(f"- Relevant: {r['relevant']}\n")


if __name__ == "__main__":
    main()
